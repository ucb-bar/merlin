builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "int8_weight_only"} {
  func.func @forward(%0: tensor<256x128xf32>, %1: tensor<128xf32>, %2: tensor<128x128xi8>, %3: tensor<128xf32>, %4: tensor<128xi64>, %5: tensor<128x128xi8>, %6: tensor<128xf32>, %7: tensor<128xi64>, %8: tensor<128x128xi8>, %9: tensor<128xf32>, %10: tensor<128xi64>, %11: tensor<128x128xi8>, %12: tensor<128xf32>, %13: tensor<128xi64>, %14: tensor<128xf32>, %15: tensor<344x128xi8>, %16: tensor<344xf32>, %17: tensor<344xi64>, %18: tensor<344x128xi8>, %19: tensor<344xf32>, %20: tensor<344xi64>, %21: tensor<128x344xi8>, %22: tensor<128xf32>, %23: tensor<128xi64>, %24: tensor<128xf32>, %25: tensor<128x128xi8>, %26: tensor<128xf32>, %27: tensor<128xi64>, %28: tensor<128x128xi8>, %29: tensor<128xf32>, %30: tensor<128xi64>, %31: tensor<128x128xi8>, %32: tensor<128xf32>, %33: tensor<128xi64>, %34: tensor<128x128xi8>, %35: tensor<128xf32>, %36: tensor<128xi64>, %37: tensor<128xf32>, %38: tensor<344x128xi8>, %39: tensor<344xf32>, %40: tensor<344xi64>, %41: tensor<344x128xi8>, %42: tensor<344xf32>, %43: tensor<344xi64>, %44: tensor<128x344xi8>, %45: tensor<128xf32>, %46: tensor<128xi64>, %47: tensor<128xf32>, %48: tensor<256x128xi8>, %49: tensor<256xf32>, %50: tensor<256xi64>, %51: tensor<1x8xi64>) -> tensor<1x8x256xf32> {
    %52 = tensor.empty() : tensor<8xi64>
    %53 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%52 : tensor<8xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb0(%54: i64):
      %55 = linalg.index 0 : index
      %56 = arith.index_cast %55 : index to i64
      %57 = arith.constant 1 : i64
      %58 = arith.muli %56, %57 : i64
      %59 = arith.constant 0 : i64
      %60 = arith.addi %59, %58 : i64
      linalg.yield %60 : i64
    } -> tensor<8xi64>
    %61 = tensor.empty() : tensor<1x8x128xf32>
    %62 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%51 : tensor<1x8xi64>) outs(%61 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32", prov.module = "emb", prov.fqn = "emb"} {
    ^bb1(%63: i64, %64: f32):
      %65 = arith.index_cast %63 : i64 to index
      %66 = linalg.index 2 : index
      %67 = tensor.extract %0[%65, %66] : tensor<256x128xf32>
      linalg.yield %67 : f32
    } -> tensor<1x8x128xf32>
    %68 = tensor.empty() : tensor<1x8x128xf32>
    %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62 : tensor<1x8x128xf32>) outs(%68 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb2(%70: f32, %71: f32):
      %72 = arith.constant 2.000000e+00 : f32
      %73 = math.powf %70, %72 : f32
      linalg.yield %73 : f32
    } -> tensor<1x8x128xf32>
    %74 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} 0.000000e+00 : f32
    %75 = tensor.splat %74 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<1x8xf32>
    %76 = linalg.reduce ins(%69:tensor<1x8x128xf32>) outs(%75:tensor<1x8xf32>) dimensions = [2]
    (%77: f32, %78: f32) {
      %79 = arith.addf %77, %78 : f32
      linalg.yield %79 : f32
    }
    %80 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} 1.280000e+02 : f32
    %81 = tensor.splat %80 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<1x8xf32>
    %82 = tensor.empty() : tensor<1x8xf32>
    %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%76, %81 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%82 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb3(%84: f32, %85: f32, %86: f32):
      %87 = arith.divf %84, %85 : f32
      linalg.yield %87 : f32
    } -> tensor<1x8xf32>
    %88 = tensor.collapse_shape %83 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<1x8xf32> into tensor<8xf32>
    %89 = tensor.expand_shape %88 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<8xf32> into tensor<1x8x1xf32>
    %90 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} 1.000000e-05 : f32
    %91 = tensor.splat %90 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<1x8x1xf32>
    %92 = tensor.empty() : tensor<1x8x1xf32>
    %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %91 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%92 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb4(%94: f32, %95: f32, %96: f32):
      %97 = arith.addf %94, %95 : f32
      linalg.yield %97 : f32
    } -> tensor<1x8x1xf32>
    %98 = tensor.empty() : tensor<1x8x1xf32>
    %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%93 : tensor<1x8x1xf32>) outs(%98 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb5(%100: f32, %101: f32):
      %102 = math.rsqrt %100 : f32
      linalg.yield %102 : f32
    } -> tensor<1x8x1xf32>
    %103 = tensor.empty() : tensor<1x8x128xf32>
    %104 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62, %99 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%103 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb6(%105: f32, %106: f32, %107: f32):
      %108 = arith.mulf %105, %106 : f32
      linalg.yield %108 : f32
    } -> tensor<1x8x128xf32>
    %109 = tensor.empty() : tensor<1x8x128xf32>
    %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%104, %1 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%109 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb7(%111: f32, %112: f32, %113: f32):
      %114 = arith.mulf %111, %112 : f32
      linalg.yield %114 : f32
    } -> tensor<1x8x128xf32>
    %115 = tensor.empty() : tensor<128x128xi8>
    %116 = linalg.transpose ins(%2:tensor<128x128xi8>) outs(%115:tensor<128x128xi8>) permutation = [1, 0]
    %117 = tensor.collapse_shape %110 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %118 = tensor.expand_shape %117 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} : tensor<1024xf32> into tensor<8x128xf32>
    %119 = tensor.empty() : tensor<128x128xf32>
    %120 = arith.constant 0 : i32
    %121 = tensor.splat %120 : tensor<128xi32>
    %122 = "quant_ext.dequantize_per_channel"(%116, %3, %121) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<128x128xi8>, tensor<128xf32>, tensor<128xi32>) -> tensor<128x128xf32>
    %123 = tensor.empty() : tensor<8x128xf32>
    %124 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %125 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%124 : f32) outs(%123 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %126 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} ins(%118, %122 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%125 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %127 = tensor.empty() : tensor<8x128xf32>
    %128 = tensor.collapse_shape %126 [[0 : i64, 1 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} : tensor<8x128xf32> into tensor<1024xf32>
    %129 = tensor.expand_shape %128 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %130 = tensor.collapse_shape %129 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %131 = tensor.expand_shape %130 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %132 = tensor.empty() : tensor<1x4x8x32xf32>
    %133 = linalg.transpose ins(%131:tensor<1x8x4x32xf32>) outs(%132:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %134 = tensor.empty() : tensor<128x128xi8>
    %135 = linalg.transpose ins(%5:tensor<128x128xi8>) outs(%134:tensor<128x128xi8>) permutation = [1, 0]
    %136 = tensor.collapse_shape %110 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %137 = tensor.expand_shape %136 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} : tensor<1024xf32> into tensor<8x128xf32>
    %138 = tensor.empty() : tensor<128x128xf32>
    %139 = arith.constant 0 : i32
    %140 = tensor.splat %139 : tensor<128xi32>
    %141 = "quant_ext.dequantize_per_channel"(%135, %6, %140) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<128x128xi8>, tensor<128xf32>, tensor<128xi32>) -> tensor<128x128xf32>
    %142 = tensor.empty() : tensor<8x128xf32>
    %143 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %144 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%143 : f32) outs(%142 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %145 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} ins(%137, %141 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%144 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %146 = tensor.empty() : tensor<8x128xf32>
    %147 = tensor.collapse_shape %145 [[0 : i64, 1 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} : tensor<8x128xf32> into tensor<1024xf32>
    %148 = tensor.expand_shape %147 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %149 = tensor.collapse_shape %148 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %150 = tensor.expand_shape %149 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %151 = tensor.empty() : tensor<1x4x8x32xf32>
    %152 = linalg.transpose ins(%150:tensor<1x8x4x32xf32>) outs(%151:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %153 = tensor.empty() : tensor<128x128xi8>
    %154 = linalg.transpose ins(%8:tensor<128x128xi8>) outs(%153:tensor<128x128xi8>) permutation = [1, 0]
    %155 = tensor.collapse_shape %110 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %156 = tensor.expand_shape %155 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} : tensor<1024xf32> into tensor<8x128xf32>
    %157 = tensor.empty() : tensor<128x128xf32>
    %158 = arith.constant 0 : i32
    %159 = tensor.splat %158 : tensor<128xi32>
    %160 = "quant_ext.dequantize_per_channel"(%154, %9, %159) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<128x128xi8>, tensor<128xf32>, tensor<128xi32>) -> tensor<128x128xf32>
    %161 = tensor.empty() : tensor<8x128xf32>
    %162 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %163 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%162 : f32) outs(%161 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %164 = linalg.matmul {prov.region_id = "matmul_2", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} ins(%156, %160 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%163 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %165 = tensor.empty() : tensor<8x128xf32>
    %166 = tensor.collapse_shape %164 [[0 : i64, 1 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} : tensor<8x128xf32> into tensor<1024xf32>
    %167 = tensor.expand_shape %166 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %168 = tensor.collapse_shape %167 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %169 = tensor.expand_shape %168 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %170 = tensor.empty() : tensor<1x4x8x32xf32>
    %171 = linalg.transpose ins(%169:tensor<1x8x4x32xf32>) outs(%170:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %172 = tensor.empty() : tensor<16xf32>
    %173 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%172 : tensor<16xf32>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb8(%174: f32):
      %175 = linalg.index 0 : index
      %176 = arith.index_cast %175 : index to i64
      %177 = arith.sitofp %176 : i64 to f32
      %178 = arith.constant 1.000000e+00 : f32
      %179 = arith.mulf %177, %178 : f32
      %180 = arith.constant 0.000000e+00 : f32
      %181 = arith.addf %180, %179 : f32
      linalg.yield %181 : f32
    } -> tensor<16xf32>
    %182 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1.600000e+01 : f32
    %183 = tensor.splat %182 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32>
    %184 = tensor.empty() : tensor<16xf32>
    %185 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%173, %183 : tensor<16xf32>, tensor<16xf32>) outs(%184 : tensor<16xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb9(%186: f32, %187: f32, %188: f32):
      %189 = arith.divf %186, %187 : f32
      linalg.yield %189 : f32
    } -> tensor<16xf32>
    %190 = tensor.empty() : tensor<16xf32>
    %191 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%185 : tensor<16xf32>) outs(%190 : tensor<16xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb10(%192: f32, %193: f32):
      %194 = arith.constant 1.000000e+04 : f32
      %195 = math.powf %194, %192 : f32
      linalg.yield %195 : f32
    } -> tensor<16xf32>
    %196 = tensor.empty() : tensor<16xf32>
    %197 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%191 : tensor<16xf32>) outs(%196 : tensor<16xf32>) attrs =  {prov.region_id = "elementwise_0", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb11(%198: f32, %199: f32):
      %200 = arith.constant 1.000000e+00 : f32
      %201 = arith.divf %200, %198 : f32
      linalg.yield %201 : f32
    } -> tensor<16xf32>
    %202 = arith.constant {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1.000000e+00 : f32
    %203 = tensor.splat %202 {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32>
    %204 = tensor.empty() : tensor<16xf32>
    %205 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%197, %203 : tensor<16xf32>, tensor<16xf32>) outs(%204 : tensor<16xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb12(%206: f32, %207: f32, %208: f32):
      %209 = arith.mulf %206, %207 : f32
      linalg.yield %209 : f32
    } -> tensor<16xf32>
    %210 = tensor.expand_shape %53 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %211 = tensor.empty() : tensor<8x1xf32>
    %212 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%210 : tensor<8x1xi64>) outs(%211 : tensor<8x1xf32>) attrs =  {prov.region_id = "dtype_cast_3", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb13(%213: i64, %214: f32):
      %215 = arith.sitofp %213 : i64 to f32
      linalg.yield %215 : f32
    } -> tensor<8x1xf32>
    %216 = tensor.expand_shape %205 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32> into tensor<1x16xf32>
    %217 = tensor.empty() : tensor<8x16xf32>
    %218 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%212, %216 : tensor<8x1xf32>, tensor<1x16xf32>) outs(%217 : tensor<8x16xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb14(%219: f32, %220: f32, %221: f32):
      %222 = arith.mulf %219, %220 : f32
      linalg.yield %222 : f32
    } -> tensor<8x16xf32>
    %223 = tensor.empty() : tensor<8x16xf32>
    %224 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%218 : tensor<8x16xf32>) outs(%223 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb15(%225: f32, %226: f32):
      %227 = math.cos %225 : f32
      linalg.yield %227 : f32
    } -> tensor<8x16xf32>
    %228 = tensor.empty() : tensor<8x16xf32>
    %229 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%218 : tensor<8x16xf32>) outs(%228 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_1", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb16(%230: f32, %231: f32):
      %232 = math.cos %230 : f32
      linalg.yield %232 : f32
    } -> tensor<8x16xf32>
    %233 = tensor.concat dim(1) %224, %229 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %234 = tensor.collapse_shape %233 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %235 = tensor.expand_shape %234 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %236 = tensor.collapse_shape %235 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %237 = tensor.expand_shape %236 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %238 = tensor.empty() : tensor<8x16xf32>
    %239 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%218 : tensor<8x16xf32>) outs(%238 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb17(%240: f32, %241: f32):
      %242 = math.sin %240 : f32
      linalg.yield %242 : f32
    } -> tensor<8x16xf32>
    %243 = tensor.empty() : tensor<8x16xf32>
    %244 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%218 : tensor<8x16xf32>) outs(%243 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_1", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb18(%245: f32, %246: f32):
      %247 = math.sin %245 : f32
      linalg.yield %247 : f32
    } -> tensor<8x16xf32>
    %248 = tensor.concat dim(1) %239, %244 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %249 = tensor.collapse_shape %248 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %250 = tensor.expand_shape %249 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %251 = tensor.collapse_shape %250 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %252 = tensor.expand_shape %251 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %253 = "tensor.extract_slice"(%133) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %254 = "tensor.extract_slice"(%133) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %255 = tensor.empty() : tensor<1x4x8x16xf32>
    %256 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%254 : tensor<1x4x8x16xf32>) outs(%255 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb19(%257: f32, %258: f32):
      %259 = arith.negf %257 : f32
      linalg.yield %259 : f32
    } -> tensor<1x4x8x16xf32>
    %260 = tensor.concat dim(3) %256, %253 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %261 = tensor.empty() : tensor<1x4x8x32xf32>
    %262 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%133, %237 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%261 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb20(%263: f32, %264: f32, %265: f32):
      %266 = arith.mulf %263, %264 : f32
      linalg.yield %266 : f32
    } -> tensor<1x4x8x32xf32>
    %267 = tensor.empty() : tensor<1x4x8x32xf32>
    %268 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%260, %252 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%267 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb21(%269: f32, %270: f32, %271: f32):
      %272 = arith.mulf %269, %270 : f32
      linalg.yield %272 : f32
    } -> tensor<1x4x8x32xf32>
    %273 = tensor.empty() : tensor<1x4x8x32xf32>
    %274 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%262, %268 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%273 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb22(%275: f32, %276: f32, %277: f32):
      %278 = arith.addf %275, %276 : f32
      linalg.yield %278 : f32
    } -> tensor<1x4x8x32xf32>
    %279 = tensor.empty() : tensor<16xf32>
    %280 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%279 : tensor<16xf32>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb23(%281: f32):
      %282 = linalg.index 0 : index
      %283 = arith.index_cast %282 : index to i64
      %284 = arith.sitofp %283 : i64 to f32
      %285 = arith.constant 1.000000e+00 : f32
      %286 = arith.mulf %284, %285 : f32
      %287 = arith.constant 0.000000e+00 : f32
      %288 = arith.addf %287, %286 : f32
      linalg.yield %288 : f32
    } -> tensor<16xf32>
    %289 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1.600000e+01 : f32
    %290 = tensor.splat %289 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32>
    %291 = tensor.empty() : tensor<16xf32>
    %292 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%280, %290 : tensor<16xf32>, tensor<16xf32>) outs(%291 : tensor<16xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb24(%293: f32, %294: f32, %295: f32):
      %296 = arith.divf %293, %294 : f32
      linalg.yield %296 : f32
    } -> tensor<16xf32>
    %297 = tensor.empty() : tensor<16xf32>
    %298 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%292 : tensor<16xf32>) outs(%297 : tensor<16xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb25(%299: f32, %300: f32):
      %301 = arith.constant 1.000000e+04 : f32
      %302 = math.powf %301, %299 : f32
      linalg.yield %302 : f32
    } -> tensor<16xf32>
    %303 = tensor.empty() : tensor<16xf32>
    %304 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%298 : tensor<16xf32>) outs(%303 : tensor<16xf32>) attrs =  {prov.region_id = "elementwise_1", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb26(%305: f32, %306: f32):
      %307 = arith.constant 1.000000e+00 : f32
      %308 = arith.divf %307, %305 : f32
      linalg.yield %308 : f32
    } -> tensor<16xf32>
    %309 = arith.constant {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1.000000e+00 : f32
    %310 = tensor.splat %309 {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32>
    %311 = tensor.empty() : tensor<16xf32>
    %312 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%304, %310 : tensor<16xf32>, tensor<16xf32>) outs(%311 : tensor<16xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb27(%313: f32, %314: f32, %315: f32):
      %316 = arith.mulf %313, %314 : f32
      linalg.yield %316 : f32
    } -> tensor<16xf32>
    %317 = tensor.expand_shape %53 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %318 = tensor.empty() : tensor<8x1xf32>
    %319 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%317 : tensor<8x1xi64>) outs(%318 : tensor<8x1xf32>) attrs =  {prov.region_id = "dtype_cast_4", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb28(%320: i64, %321: f32):
      %322 = arith.sitofp %320 : i64 to f32
      linalg.yield %322 : f32
    } -> tensor<8x1xf32>
    %323 = tensor.expand_shape %312 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32> into tensor<1x16xf32>
    %324 = tensor.empty() : tensor<8x16xf32>
    %325 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%319, %323 : tensor<8x1xf32>, tensor<1x16xf32>) outs(%324 : tensor<8x16xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb29(%326: f32, %327: f32, %328: f32):
      %329 = arith.mulf %326, %327 : f32
      linalg.yield %329 : f32
    } -> tensor<8x16xf32>
    %330 = tensor.empty() : tensor<8x16xf32>
    %331 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%325 : tensor<8x16xf32>) outs(%330 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_2", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb30(%332: f32, %333: f32):
      %334 = math.cos %332 : f32
      linalg.yield %334 : f32
    } -> tensor<8x16xf32>
    %335 = tensor.empty() : tensor<8x16xf32>
    %336 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%325 : tensor<8x16xf32>) outs(%335 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_3", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb31(%337: f32, %338: f32):
      %339 = math.cos %337 : f32
      linalg.yield %339 : f32
    } -> tensor<8x16xf32>
    %340 = tensor.concat dim(1) %331, %336 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %341 = tensor.collapse_shape %340 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %342 = tensor.expand_shape %341 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %343 = tensor.collapse_shape %342 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %344 = tensor.expand_shape %343 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %345 = tensor.empty() : tensor<8x16xf32>
    %346 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%325 : tensor<8x16xf32>) outs(%345 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_2", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb32(%347: f32, %348: f32):
      %349 = math.sin %347 : f32
      linalg.yield %349 : f32
    } -> tensor<8x16xf32>
    %350 = tensor.empty() : tensor<8x16xf32>
    %351 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%325 : tensor<8x16xf32>) outs(%350 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_3", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb33(%352: f32, %353: f32):
      %354 = math.sin %352 : f32
      linalg.yield %354 : f32
    } -> tensor<8x16xf32>
    %355 = tensor.concat dim(1) %346, %351 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %356 = tensor.collapse_shape %355 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %357 = tensor.expand_shape %356 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %358 = tensor.collapse_shape %357 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %359 = tensor.expand_shape %358 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %360 = "tensor.extract_slice"(%152) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %361 = "tensor.extract_slice"(%152) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %362 = tensor.empty() : tensor<1x4x8x16xf32>
    %363 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%361 : tensor<1x4x8x16xf32>) outs(%362 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb34(%364: f32, %365: f32):
      %366 = arith.negf %364 : f32
      linalg.yield %366 : f32
    } -> tensor<1x4x8x16xf32>
    %367 = tensor.concat dim(3) %363, %360 {prov.region_id = "cat_5", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %368 = tensor.empty() : tensor<1x4x8x32xf32>
    %369 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%152, %344 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%368 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb35(%370: f32, %371: f32, %372: f32):
      %373 = arith.mulf %370, %371 : f32
      linalg.yield %373 : f32
    } -> tensor<1x4x8x32xf32>
    %374 = tensor.empty() : tensor<1x4x8x32xf32>
    %375 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%367, %359 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%374 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb36(%376: f32, %377: f32, %378: f32):
      %379 = arith.mulf %376, %377 : f32
      linalg.yield %379 : f32
    } -> tensor<1x4x8x32xf32>
    %380 = tensor.empty() : tensor<1x4x8x32xf32>
    %381 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%369, %375 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%380 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb37(%382: f32, %383: f32, %384: f32):
      %385 = arith.addf %382, %383 : f32
      linalg.yield %385 : f32
    } -> tensor<1x4x8x32xf32>
    %386 = tensor.empty() : tensor<1x4x32x8xf32>
    %387 = linalg.transpose ins(%381:tensor<1x4x8x32xf32>) outs(%386:tensor<1x4x32x8xf32>) permutation = [0, 1, 3, 2]
    %388 = tensor.empty() : tensor<1x4x8x32xf32>
    %389 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%274 : tensor<1x4x8x32xf32>) outs(%388 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb38(%390: f32, %391: f32):
      linalg.yield %390 : f32
    } -> tensor<1x4x8x32xf32>
    %392 = tensor.collapse_shape %389 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8x32xf32> into tensor<1024xf32>
    %393 = tensor.expand_shape %392 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 32] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<4x8x32xf32>
    %394 = tensor.empty() : tensor<1x4x32x8xf32>
    %395 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%387 : tensor<1x4x32x8xf32>) outs(%394 : tensor<1x4x32x8xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb39(%396: f32, %397: f32):
      linalg.yield %396 : f32
    } -> tensor<1x4x32x8xf32>
    %398 = tensor.collapse_shape %395 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x32x8xf32> into tensor<1024xf32>
    %399 = tensor.expand_shape %398 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 32, 8] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<4x32x8xf32>
    %400 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0.000000e+00 : f32
    %401 = tensor.splat %400 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<4x8x8xf32>
    %402 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%393, %399 : tensor<4x8x32xf32>, tensor<4x32x8xf32>) outs(%401 : tensor<4x8x8xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb40(%403: f32, %404: f32, %405: f32):
      %406 = arith.mulf %403, %404 : f32
      %407 = arith.addf %405, %406 : f32
      linalg.yield %407 : f32
    } -> tensor<4x8x8xf32>
    %408 = tensor.collapse_shape %402 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<4x8x8xf32> into tensor<256xf32>
    %409 = tensor.expand_shape %408 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 8] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x4x8x8xf32>
    %410 = arith.constant {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 5.65685415 : f32
    %411 = tensor.splat %410 {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8x8xf32>
    %412 = tensor.empty() : tensor<1x4x8x8xf32>
    %413 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%409, %411 : tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>) outs(%412 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb41(%414: f32, %415: f32, %416: f32):
      %417 = arith.divf %414, %415 : f32
      linalg.yield %417 : f32
    } -> tensor<1x4x8x8xf32>
    %418 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0xff800000 : f32
    %419 = tensor.splat %418 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x8xf32>
    %420 = tensor.empty() : tensor<8xi64>
    %421 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%420 : tensor<8xi64>) attrs =  {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb42(%422: i64):
      %423 = linalg.index 0 : index
      %424 = arith.index_cast %423 : index to i64
      %425 = arith.constant 1 : i64
      %426 = arith.muli %424, %425 : i64
      %427 = arith.constant 0 : i64
      %428 = arith.addi %427, %426 : i64
      linalg.yield %428 : i64
    } -> tensor<8xi64>
    %429 = tensor.expand_shape %421 [[0 : i64, 1 : i64]] output_shape [1, 8] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8xi64> into tensor<1x8xi64>
    %430 = tensor.empty() : tensor<8xi64>
    %431 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%430 : tensor<8xi64>) attrs =  {prov.region_id = "iota_4", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb43(%432: i64):
      %433 = linalg.index 0 : index
      %434 = arith.index_cast %433 : index to i64
      %435 = arith.constant 1 : i64
      %436 = arith.muli %434, %435 : i64
      %437 = arith.constant 0 : i64
      %438 = arith.addi %437, %436 : i64
      linalg.yield %438 : i64
    } -> tensor<8xi64>
    %439 = tensor.expand_shape %431 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %440 = tensor.empty() : tensor<8x8xi64>
    %441 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%429, %439 : tensor<1x8xi64>, tensor<8x1xi64>) outs(%440 : tensor<8x8xi64>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb44(%442: i64, %443: i64, %444: i64):
      %445 = arith.subi %442, %443 : i64
      linalg.yield %445 : i64
    } -> tensor<8x8xi64>
    %446 = arith.constant {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1 : i64
    %447 = tensor.splat %446 {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x8xi64>
    %448 = tensor.empty() : tensor<8x8xi1>
    %449 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%441, %447 : tensor<8x8xi64>, tensor<8x8xi64>) outs(%448 : tensor<8x8xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb45(%450: i64, %451: i64, %452: i1):
      %453 = arith.cmpi sge, %450, %451 : i64
      linalg.yield %453 : i1
    } -> tensor<8x8xi1>
    %454 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0.000000e+00 : f32
    %455 = tensor.splat %454 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<f32>
    %456 = tensor.empty() : tensor<8x8xf32>
    %457 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%449, %419, %455 : tensor<8x8xi1>, tensor<8x8xf32>, tensor<f32>) outs(%456 : tensor<8x8xf32>) attrs =  {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb46(%458: i1, %459: f32, %460: f32, %461: f32):
      %462 = arith.select %458, %459, %460 : f32
      linalg.yield %462 : f32
    } -> tensor<8x8xf32>
    %463 = tensor.empty() : tensor<1x4x8x8xf32>
    %464 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%413, %457 : tensor<1x4x8x8xf32>, tensor<8x8xf32>) outs(%463 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb47(%465: f32, %466: f32, %467: f32):
      %468 = arith.addf %465, %466 : f32
      linalg.yield %468 : f32
    } -> tensor<1x4x8x8xf32>
    %469 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0xff800000 : f32
    %470 = tensor.splat %469 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8xf32>
    %471 = linalg.reduce ins(%464:tensor<1x4x8x8xf32>) outs(%470:tensor<1x4x8xf32>) dimensions = [3]
    (%472: f32, %473: f32) {
      %474 = arith.maximumf %472, %473 : f32
      linalg.yield %474 : f32
    }
    %475 = tensor.collapse_shape %471 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8xf32> into tensor<32xf32>
    %476 = tensor.expand_shape %475 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %477 = tensor.empty() : tensor<1x4x8x8xf32>
    %478 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%464, %476 : tensor<1x4x8x8xf32>, tensor<1x4x8x1xf32>) outs(%477 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb48(%479: f32, %480: f32, %481: f32):
      %482 = arith.subf %479, %480 : f32
      linalg.yield %482 : f32
    } -> tensor<1x4x8x8xf32>
    %483 = tensor.empty() : tensor<1x4x8x8xf32>
    %484 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%478 : tensor<1x4x8x8xf32>) outs(%483 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb49(%485: f32, %486: f32):
      %487 = math.exp %485 : f32
      linalg.yield %487 : f32
    } -> tensor<1x4x8x8xf32>
    %488 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0.000000e+00 : f32
    %489 = tensor.splat %488 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8xf32>
    %490 = linalg.reduce ins(%484:tensor<1x4x8x8xf32>) outs(%489:tensor<1x4x8xf32>) dimensions = [3]
    (%491: f32, %492: f32) {
      %493 = arith.addf %491, %492 : f32
      linalg.yield %493 : f32
    }
    %494 = tensor.collapse_shape %490 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8xf32> into tensor<32xf32>
    %495 = tensor.expand_shape %494 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %496 = tensor.empty() : tensor<1x4x8x8xf32>
    %497 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%484, %495 : tensor<1x4x8x8xf32>, tensor<1x4x8x1xf32>) outs(%496 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb50(%498: f32, %499: f32, %500: f32):
      %501 = arith.divf %498, %499 : f32
      linalg.yield %501 : f32
    } -> tensor<1x4x8x8xf32>
    %502 = tensor.empty() : tensor<1x4x8x8xf32>
    %503 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%497 : tensor<1x4x8x8xf32>) outs(%502 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb51(%504: f32, %505: f32):
      linalg.yield %504 : f32
    } -> tensor<1x4x8x8xf32>
    %506 = tensor.collapse_shape %503 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8x8xf32> into tensor<256xf32>
    %507 = tensor.expand_shape %506 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 8] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<4x8x8xf32>
    %508 = tensor.empty() : tensor<1x4x8x32xf32>
    %509 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%171 : tensor<1x4x8x32xf32>) outs(%508 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb52(%510: f32, %511: f32):
      linalg.yield %510 : f32
    } -> tensor<1x4x8x32xf32>
    %512 = tensor.collapse_shape %509 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8x32xf32> into tensor<1024xf32>
    %513 = tensor.expand_shape %512 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 32] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<4x8x32xf32>
    %514 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0.000000e+00 : f32
    %515 = tensor.splat %514 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<4x8x32xf32>
    %516 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%507, %513 : tensor<4x8x8xf32>, tensor<4x8x32xf32>) outs(%515 : tensor<4x8x32xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb53(%517: f32, %518: f32, %519: f32):
      %520 = arith.mulf %517, %518 : f32
      %521 = arith.addf %519, %520 : f32
      linalg.yield %521 : f32
    } -> tensor<4x8x32xf32>
    %522 = tensor.collapse_shape %516 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<4x8x32xf32> into tensor<1024xf32>
    %523 = tensor.expand_shape %522 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 32] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x4x8x32xf32>
    %524 = tensor.empty() : tensor<1x8x4x32xf32>
    %525 = linalg.transpose ins(%523:tensor<1x4x8x32xf32>) outs(%524:tensor<1x8x4x32xf32>) permutation = [0, 2, 1, 3]
    %526 = tensor.collapse_shape %525 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x4x32xf32> into tensor<1024xf32>
    %527 = tensor.expand_shape %526 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %528 = tensor.empty() : tensor<128x128xi8>
    %529 = linalg.transpose ins(%11:tensor<128x128xi8>) outs(%528:tensor<128x128xi8>) permutation = [1, 0]
    %530 = tensor.collapse_shape %527 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %531 = tensor.expand_shape %530 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} : tensor<1024xf32> into tensor<8x128xf32>
    %532 = tensor.empty() : tensor<128x128xf32>
    %533 = arith.constant 0 : i32
    %534 = tensor.splat %533 : tensor<128xi32>
    %535 = "quant_ext.dequantize_per_channel"(%529, %12, %534) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<128x128xi8>, tensor<128xf32>, tensor<128xi32>) -> tensor<128x128xf32>
    %536 = tensor.empty() : tensor<8x128xf32>
    %537 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %538 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%537 : f32) outs(%536 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %539 = linalg.matmul {prov.region_id = "matmul_5", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} ins(%531, %535 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%538 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %540 = tensor.empty() : tensor<8x128xf32>
    %541 = tensor.collapse_shape %539 [[0 : i64, 1 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} : tensor<8x128xf32> into tensor<1024xf32>
    %542 = tensor.expand_shape %541 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %543 = tensor.empty() : tensor<1x8x128xf32>
    %544 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62, %542 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%543 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0"} {
    ^bb54(%545: f32, %546: f32, %547: f32):
      %548 = arith.addf %545, %546 : f32
      linalg.yield %548 : f32
    } -> tensor<1x8x128xf32>
    %549 = tensor.empty() : tensor<1x8x128xf32>
    %550 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%544 : tensor<1x8x128xf32>) outs(%549 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb55(%551: f32, %552: f32):
      %553 = arith.constant 2.000000e+00 : f32
      %554 = math.powf %551, %553 : f32
      linalg.yield %554 : f32
    } -> tensor<1x8x128xf32>
    %555 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} 0.000000e+00 : f32
    %556 = tensor.splat %555 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<1x8xf32>
    %557 = linalg.reduce ins(%550:tensor<1x8x128xf32>) outs(%556:tensor<1x8xf32>) dimensions = [2]
    (%558: f32, %559: f32) {
      %560 = arith.addf %558, %559 : f32
      linalg.yield %560 : f32
    }
    %561 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} 1.280000e+02 : f32
    %562 = tensor.splat %561 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<1x8xf32>
    %563 = tensor.empty() : tensor<1x8xf32>
    %564 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%557, %562 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%563 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb56(%565: f32, %566: f32, %567: f32):
      %568 = arith.divf %565, %566 : f32
      linalg.yield %568 : f32
    } -> tensor<1x8xf32>
    %569 = tensor.collapse_shape %564 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<1x8xf32> into tensor<8xf32>
    %570 = tensor.expand_shape %569 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<8xf32> into tensor<1x8x1xf32>
    %571 = arith.constant {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} 1.000000e-05 : f32
    %572 = tensor.splat %571 {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<1x8x1xf32>
    %573 = tensor.empty() : tensor<1x8x1xf32>
    %574 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%570, %572 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%573 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb57(%575: f32, %576: f32, %577: f32):
      %578 = arith.addf %575, %576 : f32
      linalg.yield %578 : f32
    } -> tensor<1x8x1xf32>
    %579 = tensor.empty() : tensor<1x8x1xf32>
    %580 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%574 : tensor<1x8x1xf32>) outs(%579 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb58(%581: f32, %582: f32):
      %583 = math.rsqrt %581 : f32
      linalg.yield %583 : f32
    } -> tensor<1x8x1xf32>
    %584 = tensor.empty() : tensor<1x8x128xf32>
    %585 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%544, %580 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%584 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb59(%586: f32, %587: f32, %588: f32):
      %589 = arith.mulf %586, %587 : f32
      linalg.yield %589 : f32
    } -> tensor<1x8x128xf32>
    %590 = tensor.empty() : tensor<1x8x128xf32>
    %591 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%585, %14 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%590 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb60(%592: f32, %593: f32, %594: f32):
      %595 = arith.mulf %592, %593 : f32
      linalg.yield %595 : f32
    } -> tensor<1x8x128xf32>
    %596 = tensor.empty() : tensor<128x344xi8>
    %597 = linalg.transpose ins(%15:tensor<344x128xi8>) outs(%596:tensor<128x344xi8>) permutation = [1, 0]
    %598 = tensor.collapse_shape %591 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %599 = tensor.expand_shape %598 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} : tensor<1024xf32> into tensor<8x128xf32>
    %600 = tensor.empty() : tensor<128x344xf32>
    %601 = arith.constant 0 : i32
    %602 = tensor.splat %601 : tensor<344xi32>
    %603 = "quant_ext.dequantize_per_channel"(%597, %16, %602) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<128x344xi8>, tensor<344xf32>, tensor<344xi32>) -> tensor<128x344xf32>
    %604 = tensor.empty() : tensor<8x344xf32>
    %605 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %606 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%605 : f32) outs(%604 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %607 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} ins(%599, %603 : tensor<8x128xf32>, tensor<128x344xf32>) outs(%606 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %608 = tensor.empty() : tensor<8x344xf32>
    %609 = tensor.collapse_shape %607 [[0 : i64, 1 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} : tensor<8x344xf32> into tensor<2752xf32>
    %610 = tensor.expand_shape %609 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 344] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} : tensor<2752xf32> into tensor<1x8x344xf32>
    %611 = tensor.empty() : tensor<1x8x344xf32>
    %612 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%610 : tensor<1x8x344xf32>) outs(%611 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp"} {
    ^bb61(%613: f32, %614: f32):
      %615 = arith.constant 1.000000e+00 : f32
      %616 = arith.negf %613 : f32
      %617 = math.exp %616 : f32
      %618 = arith.addf %615, %617 : f32
      %619 = arith.divf %615, %618 : f32
      linalg.yield %619 : f32
    } -> tensor<1x8x344xf32>
    %620 = tensor.empty() : tensor<1x8x344xf32>
    %621 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%610, %612 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%620 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp"} {
    ^bb62(%622: f32, %623: f32, %624: f32):
      %625 = arith.mulf %622, %623 : f32
      linalg.yield %625 : f32
    } -> tensor<1x8x344xf32>
    %626 = tensor.empty() : tensor<128x344xi8>
    %627 = linalg.transpose ins(%18:tensor<344x128xi8>) outs(%626:tensor<128x344xi8>) permutation = [1, 0]
    %628 = tensor.collapse_shape %591 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %629 = tensor.expand_shape %628 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} : tensor<1024xf32> into tensor<8x128xf32>
    %630 = tensor.empty() : tensor<128x344xf32>
    %631 = arith.constant 0 : i32
    %632 = tensor.splat %631 : tensor<344xi32>
    %633 = "quant_ext.dequantize_per_channel"(%627, %19, %632) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<128x344xi8>, tensor<344xf32>, tensor<344xi32>) -> tensor<128x344xf32>
    %634 = tensor.empty() : tensor<8x344xf32>
    %635 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %636 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%635 : f32) outs(%634 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %637 = linalg.matmul {prov.region_id = "matmul_7", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} ins(%629, %633 : tensor<8x128xf32>, tensor<128x344xf32>) outs(%636 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %638 = tensor.empty() : tensor<8x344xf32>
    %639 = tensor.collapse_shape %637 [[0 : i64, 1 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} : tensor<8x344xf32> into tensor<2752xf32>
    %640 = tensor.expand_shape %639 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 344] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} : tensor<2752xf32> into tensor<1x8x344xf32>
    %641 = tensor.empty() : tensor<1x8x344xf32>
    %642 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%621, %640 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%641 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp"} {
    ^bb63(%643: f32, %644: f32, %645: f32):
      %646 = arith.mulf %643, %644 : f32
      linalg.yield %646 : f32
    } -> tensor<1x8x344xf32>
    %647 = tensor.empty() : tensor<344x128xi8>
    %648 = linalg.transpose ins(%21:tensor<128x344xi8>) outs(%647:tensor<344x128xi8>) permutation = [1, 0]
    %649 = tensor.collapse_shape %642 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} : tensor<1x8x344xf32> into tensor<2752xf32>
    %650 = tensor.expand_shape %649 [[0 : i64, 1 : i64]] output_shape [8, 344] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} : tensor<2752xf32> into tensor<8x344xf32>
    %651 = tensor.empty() : tensor<344x128xf32>
    %652 = arith.constant 0 : i32
    %653 = tensor.splat %652 : tensor<128xi32>
    %654 = "quant_ext.dequantize_per_channel"(%648, %22, %653) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<344x128xi8>, tensor<128xf32>, tensor<128xi32>) -> tensor<344x128xf32>
    %655 = tensor.empty() : tensor<8x128xf32>
    %656 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %657 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%656 : f32) outs(%655 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %658 = linalg.matmul {prov.region_id = "matmul_8", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} ins(%650, %654 : tensor<8x344xf32>, tensor<344x128xf32>) outs(%657 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %659 = tensor.empty() : tensor<8x128xf32>
    %660 = tensor.collapse_shape %658 [[0 : i64, 1 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} : tensor<8x128xf32> into tensor<1024xf32>
    %661 = tensor.expand_shape %660 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %662 = tensor.empty() : tensor<1x8x128xf32>
    %663 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%544, %661 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%662 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0"} {
    ^bb64(%664: f32, %665: f32, %666: f32):
      %667 = arith.addf %664, %665 : f32
      linalg.yield %667 : f32
    } -> tensor<1x8x128xf32>
    %668 = tensor.empty() : tensor<1x8x128xf32>
    %669 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%663 : tensor<1x8x128xf32>) outs(%668 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb65(%670: f32, %671: f32):
      %672 = arith.constant 2.000000e+00 : f32
      %673 = math.powf %670, %672 : f32
      linalg.yield %673 : f32
    } -> tensor<1x8x128xf32>
    %674 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} 0.000000e+00 : f32
    %675 = tensor.splat %674 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<1x8xf32>
    %676 = linalg.reduce ins(%669:tensor<1x8x128xf32>) outs(%675:tensor<1x8xf32>) dimensions = [2]
    (%677: f32, %678: f32) {
      %679 = arith.addf %677, %678 : f32
      linalg.yield %679 : f32
    }
    %680 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} 1.280000e+02 : f32
    %681 = tensor.splat %680 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<1x8xf32>
    %682 = tensor.empty() : tensor<1x8xf32>
    %683 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%676, %681 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%682 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb66(%684: f32, %685: f32, %686: f32):
      %687 = arith.divf %684, %685 : f32
      linalg.yield %687 : f32
    } -> tensor<1x8xf32>
    %688 = tensor.collapse_shape %683 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<1x8xf32> into tensor<8xf32>
    %689 = tensor.expand_shape %688 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<8xf32> into tensor<1x8x1xf32>
    %690 = arith.constant {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} 1.000000e-05 : f32
    %691 = tensor.splat %690 {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<1x8x1xf32>
    %692 = tensor.empty() : tensor<1x8x1xf32>
    %693 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%689, %691 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%692 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb67(%694: f32, %695: f32, %696: f32):
      %697 = arith.addf %694, %695 : f32
      linalg.yield %697 : f32
    } -> tensor<1x8x1xf32>
    %698 = tensor.empty() : tensor<1x8x1xf32>
    %699 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%693 : tensor<1x8x1xf32>) outs(%698 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb68(%700: f32, %701: f32):
      %702 = math.rsqrt %700 : f32
      linalg.yield %702 : f32
    } -> tensor<1x8x1xf32>
    %703 = tensor.empty() : tensor<1x8x128xf32>
    %704 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%663, %699 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%703 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb69(%705: f32, %706: f32, %707: f32):
      %708 = arith.mulf %705, %706 : f32
      linalg.yield %708 : f32
    } -> tensor<1x8x128xf32>
    %709 = tensor.empty() : tensor<1x8x128xf32>
    %710 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%704, %24 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%709 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb70(%711: f32, %712: f32, %713: f32):
      %714 = arith.mulf %711, %712 : f32
      linalg.yield %714 : f32
    } -> tensor<1x8x128xf32>
    %715 = tensor.empty() : tensor<128x128xi8>
    %716 = linalg.transpose ins(%25:tensor<128x128xi8>) outs(%715:tensor<128x128xi8>) permutation = [1, 0]
    %717 = tensor.collapse_shape %710 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %718 = tensor.expand_shape %717 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} : tensor<1024xf32> into tensor<8x128xf32>
    %719 = tensor.empty() : tensor<128x128xf32>
    %720 = arith.constant 0 : i32
    %721 = tensor.splat %720 : tensor<128xi32>
    %722 = "quant_ext.dequantize_per_channel"(%716, %26, %721) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<128x128xi8>, tensor<128xf32>, tensor<128xi32>) -> tensor<128x128xf32>
    %723 = tensor.empty() : tensor<8x128xf32>
    %724 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %725 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%724 : f32) outs(%723 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %726 = linalg.matmul {prov.region_id = "matmul_9", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} ins(%718, %722 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%725 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %727 = tensor.empty() : tensor<8x128xf32>
    %728 = tensor.collapse_shape %726 [[0 : i64, 1 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} : tensor<8x128xf32> into tensor<1024xf32>
    %729 = tensor.expand_shape %728 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %730 = tensor.collapse_shape %729 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %731 = tensor.expand_shape %730 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %732 = tensor.empty() : tensor<1x4x8x32xf32>
    %733 = linalg.transpose ins(%731:tensor<1x8x4x32xf32>) outs(%732:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %734 = tensor.empty() : tensor<128x128xi8>
    %735 = linalg.transpose ins(%28:tensor<128x128xi8>) outs(%734:tensor<128x128xi8>) permutation = [1, 0]
    %736 = tensor.collapse_shape %710 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %737 = tensor.expand_shape %736 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} : tensor<1024xf32> into tensor<8x128xf32>
    %738 = tensor.empty() : tensor<128x128xf32>
    %739 = arith.constant 0 : i32
    %740 = tensor.splat %739 : tensor<128xi32>
    %741 = "quant_ext.dequantize_per_channel"(%735, %29, %740) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<128x128xi8>, tensor<128xf32>, tensor<128xi32>) -> tensor<128x128xf32>
    %742 = tensor.empty() : tensor<8x128xf32>
    %743 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %744 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%743 : f32) outs(%742 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %745 = linalg.matmul {prov.region_id = "matmul_10", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} ins(%737, %741 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%744 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %746 = tensor.empty() : tensor<8x128xf32>
    %747 = tensor.collapse_shape %745 [[0 : i64, 1 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} : tensor<8x128xf32> into tensor<1024xf32>
    %748 = tensor.expand_shape %747 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %749 = tensor.collapse_shape %748 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %750 = tensor.expand_shape %749 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %751 = tensor.empty() : tensor<1x4x8x32xf32>
    %752 = linalg.transpose ins(%750:tensor<1x8x4x32xf32>) outs(%751:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %753 = tensor.empty() : tensor<128x128xi8>
    %754 = linalg.transpose ins(%31:tensor<128x128xi8>) outs(%753:tensor<128x128xi8>) permutation = [1, 0]
    %755 = tensor.collapse_shape %710 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %756 = tensor.expand_shape %755 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} : tensor<1024xf32> into tensor<8x128xf32>
    %757 = tensor.empty() : tensor<128x128xf32>
    %758 = arith.constant 0 : i32
    %759 = tensor.splat %758 : tensor<128xi32>
    %760 = "quant_ext.dequantize_per_channel"(%754, %32, %759) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<128x128xi8>, tensor<128xf32>, tensor<128xi32>) -> tensor<128x128xf32>
    %761 = tensor.empty() : tensor<8x128xf32>
    %762 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %763 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%762 : f32) outs(%761 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %764 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} ins(%756, %760 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%763 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %765 = tensor.empty() : tensor<8x128xf32>
    %766 = tensor.collapse_shape %764 [[0 : i64, 1 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} : tensor<8x128xf32> into tensor<1024xf32>
    %767 = tensor.expand_shape %766 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %768 = tensor.collapse_shape %767 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %769 = tensor.expand_shape %768 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %770 = tensor.empty() : tensor<1x4x8x32xf32>
    %771 = linalg.transpose ins(%769:tensor<1x8x4x32xf32>) outs(%770:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %772 = tensor.empty() : tensor<16xf32>
    %773 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%772 : tensor<16xf32>) attrs =  {prov.region_id = "iota_5", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb71(%774: f32):
      %775 = linalg.index 0 : index
      %776 = arith.index_cast %775 : index to i64
      %777 = arith.sitofp %776 : i64 to f32
      %778 = arith.constant 1.000000e+00 : f32
      %779 = arith.mulf %777, %778 : f32
      %780 = arith.constant 0.000000e+00 : f32
      %781 = arith.addf %780, %779 : f32
      linalg.yield %781 : f32
    } -> tensor<16xf32>
    %782 = arith.constant {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1.600000e+01 : f32
    %783 = tensor.splat %782 {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32>
    %784 = tensor.empty() : tensor<16xf32>
    %785 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%773, %783 : tensor<16xf32>, tensor<16xf32>) outs(%784 : tensor<16xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb72(%786: f32, %787: f32, %788: f32):
      %789 = arith.divf %786, %787 : f32
      linalg.yield %789 : f32
    } -> tensor<16xf32>
    %790 = tensor.empty() : tensor<16xf32>
    %791 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%785 : tensor<16xf32>) outs(%790 : tensor<16xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb73(%792: f32, %793: f32):
      %794 = arith.constant 1.000000e+04 : f32
      %795 = math.powf %794, %792 : f32
      linalg.yield %795 : f32
    } -> tensor<16xf32>
    %796 = tensor.empty() : tensor<16xf32>
    %797 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%791 : tensor<16xf32>) outs(%796 : tensor<16xf32>) attrs =  {prov.region_id = "elementwise_2", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb74(%798: f32, %799: f32):
      %800 = arith.constant 1.000000e+00 : f32
      %801 = arith.divf %800, %798 : f32
      linalg.yield %801 : f32
    } -> tensor<16xf32>
    %802 = arith.constant {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1.000000e+00 : f32
    %803 = tensor.splat %802 {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32>
    %804 = tensor.empty() : tensor<16xf32>
    %805 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%797, %803 : tensor<16xf32>, tensor<16xf32>) outs(%804 : tensor<16xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb75(%806: f32, %807: f32, %808: f32):
      %809 = arith.mulf %806, %807 : f32
      linalg.yield %809 : f32
    } -> tensor<16xf32>
    %810 = tensor.expand_shape %53 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %811 = tensor.empty() : tensor<8x1xf32>
    %812 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%810 : tensor<8x1xi64>) outs(%811 : tensor<8x1xf32>) attrs =  {prov.region_id = "dtype_cast_12", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb76(%813: i64, %814: f32):
      %815 = arith.sitofp %813 : i64 to f32
      linalg.yield %815 : f32
    } -> tensor<8x1xf32>
    %816 = tensor.expand_shape %805 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32> into tensor<1x16xf32>
    %817 = tensor.empty() : tensor<8x16xf32>
    %818 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%812, %816 : tensor<8x1xf32>, tensor<1x16xf32>) outs(%817 : tensor<8x16xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb77(%819: f32, %820: f32, %821: f32):
      %822 = arith.mulf %819, %820 : f32
      linalg.yield %822 : f32
    } -> tensor<8x16xf32>
    %823 = tensor.empty() : tensor<8x16xf32>
    %824 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%818 : tensor<8x16xf32>) outs(%823 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_4", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb78(%825: f32, %826: f32):
      %827 = math.cos %825 : f32
      linalg.yield %827 : f32
    } -> tensor<8x16xf32>
    %828 = tensor.empty() : tensor<8x16xf32>
    %829 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%818 : tensor<8x16xf32>) outs(%828 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_5", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb79(%830: f32, %831: f32):
      %832 = math.cos %830 : f32
      linalg.yield %832 : f32
    } -> tensor<8x16xf32>
    %833 = tensor.concat dim(1) %824, %829 {prov.region_id = "cat_6", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %834 = tensor.collapse_shape %833 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %835 = tensor.expand_shape %834 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %836 = tensor.collapse_shape %835 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %837 = tensor.expand_shape %836 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %838 = tensor.empty() : tensor<8x16xf32>
    %839 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%818 : tensor<8x16xf32>) outs(%838 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_4", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb80(%840: f32, %841: f32):
      %842 = math.sin %840 : f32
      linalg.yield %842 : f32
    } -> tensor<8x16xf32>
    %843 = tensor.empty() : tensor<8x16xf32>
    %844 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%818 : tensor<8x16xf32>) outs(%843 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_5", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb81(%845: f32, %846: f32):
      %847 = math.sin %845 : f32
      linalg.yield %847 : f32
    } -> tensor<8x16xf32>
    %848 = tensor.concat dim(1) %839, %844 {prov.region_id = "cat_7", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %849 = tensor.collapse_shape %848 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %850 = tensor.expand_shape %849 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %851 = tensor.collapse_shape %850 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %852 = tensor.expand_shape %851 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %853 = "tensor.extract_slice"(%733) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %854 = "tensor.extract_slice"(%733) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %855 = tensor.empty() : tensor<1x4x8x16xf32>
    %856 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%854 : tensor<1x4x8x16xf32>) outs(%855 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb82(%857: f32, %858: f32):
      %859 = arith.negf %857 : f32
      linalg.yield %859 : f32
    } -> tensor<1x4x8x16xf32>
    %860 = tensor.concat dim(3) %856, %853 {prov.region_id = "cat_8", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %861 = tensor.empty() : tensor<1x4x8x32xf32>
    %862 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%733, %837 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%861 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb83(%863: f32, %864: f32, %865: f32):
      %866 = arith.mulf %863, %864 : f32
      linalg.yield %866 : f32
    } -> tensor<1x4x8x32xf32>
    %867 = tensor.empty() : tensor<1x4x8x32xf32>
    %868 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%860, %852 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%867 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb84(%869: f32, %870: f32, %871: f32):
      %872 = arith.mulf %869, %870 : f32
      linalg.yield %872 : f32
    } -> tensor<1x4x8x32xf32>
    %873 = tensor.empty() : tensor<1x4x8x32xf32>
    %874 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%862, %868 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%873 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb85(%875: f32, %876: f32, %877: f32):
      %878 = arith.addf %875, %876 : f32
      linalg.yield %878 : f32
    } -> tensor<1x4x8x32xf32>
    %879 = tensor.empty() : tensor<16xf32>
    %880 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%879 : tensor<16xf32>) attrs =  {prov.region_id = "iota_6", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb86(%881: f32):
      %882 = linalg.index 0 : index
      %883 = arith.index_cast %882 : index to i64
      %884 = arith.sitofp %883 : i64 to f32
      %885 = arith.constant 1.000000e+00 : f32
      %886 = arith.mulf %884, %885 : f32
      %887 = arith.constant 0.000000e+00 : f32
      %888 = arith.addf %887, %886 : f32
      linalg.yield %888 : f32
    } -> tensor<16xf32>
    %889 = arith.constant {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1.600000e+01 : f32
    %890 = tensor.splat %889 {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32>
    %891 = tensor.empty() : tensor<16xf32>
    %892 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%880, %890 : tensor<16xf32>, tensor<16xf32>) outs(%891 : tensor<16xf32>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb87(%893: f32, %894: f32, %895: f32):
      %896 = arith.divf %893, %894 : f32
      linalg.yield %896 : f32
    } -> tensor<16xf32>
    %897 = tensor.empty() : tensor<16xf32>
    %898 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%892 : tensor<16xf32>) outs(%897 : tensor<16xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb88(%899: f32, %900: f32):
      %901 = arith.constant 1.000000e+04 : f32
      %902 = math.powf %901, %899 : f32
      linalg.yield %902 : f32
    } -> tensor<16xf32>
    %903 = tensor.empty() : tensor<16xf32>
    %904 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%898 : tensor<16xf32>) outs(%903 : tensor<16xf32>) attrs =  {prov.region_id = "elementwise_3", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb89(%905: f32, %906: f32):
      %907 = arith.constant 1.000000e+00 : f32
      %908 = arith.divf %907, %905 : f32
      linalg.yield %908 : f32
    } -> tensor<16xf32>
    %909 = arith.constant {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1.000000e+00 : f32
    %910 = tensor.splat %909 {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32>
    %911 = tensor.empty() : tensor<16xf32>
    %912 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%904, %910 : tensor<16xf32>, tensor<16xf32>) outs(%911 : tensor<16xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb90(%913: f32, %914: f32, %915: f32):
      %916 = arith.mulf %913, %914 : f32
      linalg.yield %916 : f32
    } -> tensor<16xf32>
    %917 = tensor.expand_shape %53 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %918 = tensor.empty() : tensor<8x1xf32>
    %919 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%917 : tensor<8x1xi64>) outs(%918 : tensor<8x1xf32>) attrs =  {prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb91(%920: i64, %921: f32):
      %922 = arith.sitofp %920 : i64 to f32
      linalg.yield %922 : f32
    } -> tensor<8x1xf32>
    %923 = tensor.expand_shape %912 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32> into tensor<1x16xf32>
    %924 = tensor.empty() : tensor<8x16xf32>
    %925 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%919, %923 : tensor<8x1xf32>, tensor<1x16xf32>) outs(%924 : tensor<8x16xf32>) attrs =  {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb92(%926: f32, %927: f32, %928: f32):
      %929 = arith.mulf %926, %927 : f32
      linalg.yield %929 : f32
    } -> tensor<8x16xf32>
    %930 = tensor.empty() : tensor<8x16xf32>
    %931 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%925 : tensor<8x16xf32>) outs(%930 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_6", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb93(%932: f32, %933: f32):
      %934 = math.cos %932 : f32
      linalg.yield %934 : f32
    } -> tensor<8x16xf32>
    %935 = tensor.empty() : tensor<8x16xf32>
    %936 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%925 : tensor<8x16xf32>) outs(%935 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_7", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb94(%937: f32, %938: f32):
      %939 = math.cos %937 : f32
      linalg.yield %939 : f32
    } -> tensor<8x16xf32>
    %940 = tensor.concat dim(1) %931, %936 {prov.region_id = "cat_9", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %941 = tensor.collapse_shape %940 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %942 = tensor.expand_shape %941 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %943 = tensor.collapse_shape %942 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %944 = tensor.expand_shape %943 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %945 = tensor.empty() : tensor<8x16xf32>
    %946 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%925 : tensor<8x16xf32>) outs(%945 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_6", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb95(%947: f32, %948: f32):
      %949 = math.sin %947 : f32
      linalg.yield %949 : f32
    } -> tensor<8x16xf32>
    %950 = tensor.empty() : tensor<8x16xf32>
    %951 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%925 : tensor<8x16xf32>) outs(%950 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_7", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb96(%952: f32, %953: f32):
      %954 = math.sin %952 : f32
      linalg.yield %954 : f32
    } -> tensor<8x16xf32>
    %955 = tensor.concat dim(1) %946, %951 {prov.region_id = "cat_10", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %956 = tensor.collapse_shape %955 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %957 = tensor.expand_shape %956 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %958 = tensor.collapse_shape %957 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %959 = tensor.expand_shape %958 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %960 = "tensor.extract_slice"(%752) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %961 = "tensor.extract_slice"(%752) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %962 = tensor.empty() : tensor<1x4x8x16xf32>
    %963 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%961 : tensor<1x4x8x16xf32>) outs(%962 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb97(%964: f32, %965: f32):
      %966 = arith.negf %964 : f32
      linalg.yield %966 : f32
    } -> tensor<1x4x8x16xf32>
    %967 = tensor.concat dim(3) %963, %960 {prov.region_id = "cat_11", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %968 = tensor.empty() : tensor<1x4x8x32xf32>
    %969 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%752, %944 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%968 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb98(%970: f32, %971: f32, %972: f32):
      %973 = arith.mulf %970, %971 : f32
      linalg.yield %973 : f32
    } -> tensor<1x4x8x32xf32>
    %974 = tensor.empty() : tensor<1x4x8x32xf32>
    %975 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%967, %959 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%974 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb99(%976: f32, %977: f32, %978: f32):
      %979 = arith.mulf %976, %977 : f32
      linalg.yield %979 : f32
    } -> tensor<1x4x8x32xf32>
    %980 = tensor.empty() : tensor<1x4x8x32xf32>
    %981 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%969, %975 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%980 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb100(%982: f32, %983: f32, %984: f32):
      %985 = arith.addf %982, %983 : f32
      linalg.yield %985 : f32
    } -> tensor<1x4x8x32xf32>
    %986 = tensor.empty() : tensor<1x4x32x8xf32>
    %987 = linalg.transpose ins(%981:tensor<1x4x8x32xf32>) outs(%986:tensor<1x4x32x8xf32>) permutation = [0, 1, 3, 2]
    %988 = tensor.empty() : tensor<1x4x8x32xf32>
    %989 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%874 : tensor<1x4x8x32xf32>) outs(%988 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb101(%990: f32, %991: f32):
      linalg.yield %990 : f32
    } -> tensor<1x4x8x32xf32>
    %992 = tensor.collapse_shape %989 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8x32xf32> into tensor<1024xf32>
    %993 = tensor.expand_shape %992 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 32] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<4x8x32xf32>
    %994 = tensor.empty() : tensor<1x4x32x8xf32>
    %995 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%987 : tensor<1x4x32x8xf32>) outs(%994 : tensor<1x4x32x8xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb102(%996: f32, %997: f32):
      linalg.yield %996 : f32
    } -> tensor<1x4x32x8xf32>
    %998 = tensor.collapse_shape %995 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x32x8xf32> into tensor<1024xf32>
    %999 = tensor.expand_shape %998 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 32, 8] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<4x32x8xf32>
    %1000 = arith.constant {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0.000000e+00 : f32
    %1001 = tensor.splat %1000 {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<4x8x8xf32>
    %1002 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%993, %999 : tensor<4x8x32xf32>, tensor<4x32x8xf32>) outs(%1001 : tensor<4x8x8xf32>) attrs =  {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb103(%1003: f32, %1004: f32, %1005: f32):
      %1006 = arith.mulf %1003, %1004 : f32
      %1007 = arith.addf %1005, %1006 : f32
      linalg.yield %1007 : f32
    } -> tensor<4x8x8xf32>
    %1008 = tensor.collapse_shape %1002 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<4x8x8xf32> into tensor<256xf32>
    %1009 = tensor.expand_shape %1008 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 8] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x4x8x8xf32>
    %1010 = arith.constant {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 5.65685415 : f32
    %1011 = tensor.splat %1010 {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8x8xf32>
    %1012 = tensor.empty() : tensor<1x4x8x8xf32>
    %1013 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1009, %1011 : tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>) outs(%1012 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb104(%1014: f32, %1015: f32, %1016: f32):
      %1017 = arith.divf %1014, %1015 : f32
      linalg.yield %1017 : f32
    } -> tensor<1x4x8x8xf32>
    %1018 = arith.constant {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0xff800000 : f32
    %1019 = tensor.splat %1018 {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x8xf32>
    %1020 = tensor.empty() : tensor<8xi64>
    %1021 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1020 : tensor<8xi64>) attrs =  {prov.region_id = "iota_7", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb105(%1022: i64):
      %1023 = linalg.index 0 : index
      %1024 = arith.index_cast %1023 : index to i64
      %1025 = arith.constant 1 : i64
      %1026 = arith.muli %1024, %1025 : i64
      %1027 = arith.constant 0 : i64
      %1028 = arith.addi %1027, %1026 : i64
      linalg.yield %1028 : i64
    } -> tensor<8xi64>
    %1029 = tensor.expand_shape %1021 [[0 : i64, 1 : i64]] output_shape [1, 8] {prov.region_id = "unsqueeze_26", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8xi64> into tensor<1x8xi64>
    %1030 = tensor.empty() : tensor<8xi64>
    %1031 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1030 : tensor<8xi64>) attrs =  {prov.region_id = "iota_8", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb106(%1032: i64):
      %1033 = linalg.index 0 : index
      %1034 = arith.index_cast %1033 : index to i64
      %1035 = arith.constant 1 : i64
      %1036 = arith.muli %1034, %1035 : i64
      %1037 = arith.constant 0 : i64
      %1038 = arith.addi %1037, %1036 : i64
      linalg.yield %1038 : i64
    } -> tensor<8xi64>
    %1039 = tensor.expand_shape %1031 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_27", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %1040 = tensor.empty() : tensor<8x8xi64>
    %1041 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1029, %1039 : tensor<1x8xi64>, tensor<8x1xi64>) outs(%1040 : tensor<8x8xi64>) attrs =  {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb107(%1042: i64, %1043: i64, %1044: i64):
      %1045 = arith.subi %1042, %1043 : i64
      linalg.yield %1045 : i64
    } -> tensor<8x8xi64>
    %1046 = arith.constant {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1 : i64
    %1047 = tensor.splat %1046 {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x8xi64>
    %1048 = tensor.empty() : tensor<8x8xi1>
    %1049 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1041, %1047 : tensor<8x8xi64>, tensor<8x8xi64>) outs(%1048 : tensor<8x8xi1>) attrs =  {prov.region_id = "compare_1", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb108(%1050: i64, %1051: i64, %1052: i1):
      %1053 = arith.cmpi sge, %1050, %1051 : i64
      linalg.yield %1053 : i1
    } -> tensor<8x8xi1>
    %1054 = arith.constant {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0.000000e+00 : f32
    %1055 = tensor.splat %1054 {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<f32>
    %1056 = tensor.empty() : tensor<8x8xf32>
    %1057 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1049, %1019, %1055 : tensor<8x8xi1>, tensor<8x8xf32>, tensor<f32>) outs(%1056 : tensor<8x8xf32>) attrs =  {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb109(%1058: i1, %1059: f32, %1060: f32, %1061: f32):
      %1062 = arith.select %1058, %1059, %1060 : f32
      linalg.yield %1062 : f32
    } -> tensor<8x8xf32>
    %1063 = tensor.empty() : tensor<1x4x8x8xf32>
    %1064 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1013, %1057 : tensor<1x4x8x8xf32>, tensor<8x8xf32>) outs(%1063 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb110(%1065: f32, %1066: f32, %1067: f32):
      %1068 = arith.addf %1065, %1066 : f32
      linalg.yield %1068 : f32
    } -> tensor<1x4x8x8xf32>
    %1069 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0xff800000 : f32
    %1070 = tensor.splat %1069 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8xf32>
    %1071 = linalg.reduce ins(%1064:tensor<1x4x8x8xf32>) outs(%1070:tensor<1x4x8xf32>) dimensions = [3]
    (%1072: f32, %1073: f32) {
      %1074 = arith.maximumf %1072, %1073 : f32
      linalg.yield %1074 : f32
    }
    %1075 = tensor.collapse_shape %1071 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8xf32> into tensor<32xf32>
    %1076 = tensor.expand_shape %1075 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %1077 = tensor.empty() : tensor<1x4x8x8xf32>
    %1078 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1064, %1076 : tensor<1x4x8x8xf32>, tensor<1x4x8x1xf32>) outs(%1077 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb111(%1079: f32, %1080: f32, %1081: f32):
      %1082 = arith.subf %1079, %1080 : f32
      linalg.yield %1082 : f32
    } -> tensor<1x4x8x8xf32>
    %1083 = tensor.empty() : tensor<1x4x8x8xf32>
    %1084 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1078 : tensor<1x4x8x8xf32>) outs(%1083 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb112(%1085: f32, %1086: f32):
      %1087 = math.exp %1085 : f32
      linalg.yield %1087 : f32
    } -> tensor<1x4x8x8xf32>
    %1088 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0.000000e+00 : f32
    %1089 = tensor.splat %1088 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8xf32>
    %1090 = linalg.reduce ins(%1084:tensor<1x4x8x8xf32>) outs(%1089:tensor<1x4x8xf32>) dimensions = [3]
    (%1091: f32, %1092: f32) {
      %1093 = arith.addf %1091, %1092 : f32
      linalg.yield %1093 : f32
    }
    %1094 = tensor.collapse_shape %1090 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8xf32> into tensor<32xf32>
    %1095 = tensor.expand_shape %1094 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %1096 = tensor.empty() : tensor<1x4x8x8xf32>
    %1097 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1084, %1095 : tensor<1x4x8x8xf32>, tensor<1x4x8x1xf32>) outs(%1096 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb113(%1098: f32, %1099: f32, %1100: f32):
      %1101 = arith.divf %1098, %1099 : f32
      linalg.yield %1101 : f32
    } -> tensor<1x4x8x8xf32>
    %1102 = tensor.empty() : tensor<1x4x8x8xf32>
    %1103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1097 : tensor<1x4x8x8xf32>) outs(%1102 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb114(%1104: f32, %1105: f32):
      linalg.yield %1104 : f32
    } -> tensor<1x4x8x8xf32>
    %1106 = tensor.collapse_shape %1103 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8x8xf32> into tensor<256xf32>
    %1107 = tensor.expand_shape %1106 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 8] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<4x8x8xf32>
    %1108 = tensor.empty() : tensor<1x4x8x32xf32>
    %1109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%771 : tensor<1x4x8x32xf32>) outs(%1108 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb115(%1110: f32, %1111: f32):
      linalg.yield %1110 : f32
    } -> tensor<1x4x8x32xf32>
    %1112 = tensor.collapse_shape %1109 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8x32xf32> into tensor<1024xf32>
    %1113 = tensor.expand_shape %1112 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 32] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<4x8x32xf32>
    %1114 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0.000000e+00 : f32
    %1115 = tensor.splat %1114 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<4x8x32xf32>
    %1116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1107, %1113 : tensor<4x8x8xf32>, tensor<4x8x32xf32>) outs(%1115 : tensor<4x8x32xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb116(%1117: f32, %1118: f32, %1119: f32):
      %1120 = arith.mulf %1117, %1118 : f32
      %1121 = arith.addf %1119, %1120 : f32
      linalg.yield %1121 : f32
    } -> tensor<4x8x32xf32>
    %1122 = tensor.collapse_shape %1116 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<4x8x32xf32> into tensor<1024xf32>
    %1123 = tensor.expand_shape %1122 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 32] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x4x8x32xf32>
    %1124 = tensor.empty() : tensor<1x8x4x32xf32>
    %1125 = linalg.transpose ins(%1123:tensor<1x4x8x32xf32>) outs(%1124:tensor<1x8x4x32xf32>) permutation = [0, 2, 1, 3]
    %1126 = tensor.collapse_shape %1125 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x4x32xf32> into tensor<1024xf32>
    %1127 = tensor.expand_shape %1126 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %1128 = tensor.empty() : tensor<128x128xi8>
    %1129 = linalg.transpose ins(%34:tensor<128x128xi8>) outs(%1128:tensor<128x128xi8>) permutation = [1, 0]
    %1130 = tensor.collapse_shape %1127 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1131 = tensor.expand_shape %1130 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} : tensor<1024xf32> into tensor<8x128xf32>
    %1132 = tensor.empty() : tensor<128x128xf32>
    %1133 = arith.constant 0 : i32
    %1134 = tensor.splat %1133 : tensor<128xi32>
    %1135 = "quant_ext.dequantize_per_channel"(%1129, %35, %1134) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<128x128xi8>, tensor<128xf32>, tensor<128xi32>) -> tensor<128x128xf32>
    %1136 = tensor.empty() : tensor<8x128xf32>
    %1137 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1138 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1137 : f32) outs(%1136 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %1139 = linalg.matmul {prov.region_id = "matmul_14", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} ins(%1131, %1135 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%1138 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %1140 = tensor.empty() : tensor<8x128xf32>
    %1141 = tensor.collapse_shape %1139 [[0 : i64, 1 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} : tensor<8x128xf32> into tensor<1024xf32>
    %1142 = tensor.expand_shape %1141 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %1143 = tensor.empty() : tensor<1x8x128xf32>
    %1144 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%663, %1142 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%1143 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1"} {
    ^bb117(%1145: f32, %1146: f32, %1147: f32):
      %1148 = arith.addf %1145, %1146 : f32
      linalg.yield %1148 : f32
    } -> tensor<1x8x128xf32>
    %1149 = tensor.empty() : tensor<1x8x128xf32>
    %1150 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1144 : tensor<1x8x128xf32>) outs(%1149 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb118(%1151: f32, %1152: f32):
      %1153 = arith.constant 2.000000e+00 : f32
      %1154 = math.powf %1151, %1153 : f32
      linalg.yield %1154 : f32
    } -> tensor<1x8x128xf32>
    %1155 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} 0.000000e+00 : f32
    %1156 = tensor.splat %1155 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<1x8xf32>
    %1157 = linalg.reduce ins(%1150:tensor<1x8x128xf32>) outs(%1156:tensor<1x8xf32>) dimensions = [2]
    (%1158: f32, %1159: f32) {
      %1160 = arith.addf %1158, %1159 : f32
      linalg.yield %1160 : f32
    }
    %1161 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} 1.280000e+02 : f32
    %1162 = tensor.splat %1161 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<1x8xf32>
    %1163 = tensor.empty() : tensor<1x8xf32>
    %1164 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1157, %1162 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%1163 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb119(%1165: f32, %1166: f32, %1167: f32):
      %1168 = arith.divf %1165, %1166 : f32
      linalg.yield %1168 : f32
    } -> tensor<1x8xf32>
    %1169 = tensor.collapse_shape %1164 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<1x8xf32> into tensor<8xf32>
    %1170 = tensor.expand_shape %1169 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<8xf32> into tensor<1x8x1xf32>
    %1171 = arith.constant {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} 1.000000e-05 : f32
    %1172 = tensor.splat %1171 {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<1x8x1xf32>
    %1173 = tensor.empty() : tensor<1x8x1xf32>
    %1174 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1170, %1172 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%1173 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb120(%1175: f32, %1176: f32, %1177: f32):
      %1178 = arith.addf %1175, %1176 : f32
      linalg.yield %1178 : f32
    } -> tensor<1x8x1xf32>
    %1179 = tensor.empty() : tensor<1x8x1xf32>
    %1180 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1174 : tensor<1x8x1xf32>) outs(%1179 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb121(%1181: f32, %1182: f32):
      %1183 = math.rsqrt %1181 : f32
      linalg.yield %1183 : f32
    } -> tensor<1x8x1xf32>
    %1184 = tensor.empty() : tensor<1x8x128xf32>
    %1185 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1144, %1180 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%1184 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_35", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb122(%1186: f32, %1187: f32, %1188: f32):
      %1189 = arith.mulf %1186, %1187 : f32
      linalg.yield %1189 : f32
    } -> tensor<1x8x128xf32>
    %1190 = tensor.empty() : tensor<1x8x128xf32>
    %1191 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1185, %37 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%1190 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_36", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb123(%1192: f32, %1193: f32, %1194: f32):
      %1195 = arith.mulf %1192, %1193 : f32
      linalg.yield %1195 : f32
    } -> tensor<1x8x128xf32>
    %1196 = tensor.empty() : tensor<128x344xi8>
    %1197 = linalg.transpose ins(%38:tensor<344x128xi8>) outs(%1196:tensor<128x344xi8>) permutation = [1, 0]
    %1198 = tensor.collapse_shape %1191 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1199 = tensor.expand_shape %1198 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} : tensor<1024xf32> into tensor<8x128xf32>
    %1200 = tensor.empty() : tensor<128x344xf32>
    %1201 = arith.constant 0 : i32
    %1202 = tensor.splat %1201 : tensor<344xi32>
    %1203 = "quant_ext.dequantize_per_channel"(%1197, %39, %1202) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<128x344xi8>, tensor<344xf32>, tensor<344xi32>) -> tensor<128x344xf32>
    %1204 = tensor.empty() : tensor<8x344xf32>
    %1205 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1206 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1205 : f32) outs(%1204 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %1207 = linalg.matmul {prov.region_id = "matmul_15", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} ins(%1199, %1203 : tensor<8x128xf32>, tensor<128x344xf32>) outs(%1206 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %1208 = tensor.empty() : tensor<8x344xf32>
    %1209 = tensor.collapse_shape %1207 [[0 : i64, 1 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} : tensor<8x344xf32> into tensor<2752xf32>
    %1210 = tensor.expand_shape %1209 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 344] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} : tensor<2752xf32> into tensor<1x8x344xf32>
    %1211 = tensor.empty() : tensor<1x8x344xf32>
    %1212 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1210 : tensor<1x8x344xf32>) outs(%1211 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp"} {
    ^bb124(%1213: f32, %1214: f32):
      %1215 = arith.constant 1.000000e+00 : f32
      %1216 = arith.negf %1213 : f32
      %1217 = math.exp %1216 : f32
      %1218 = arith.addf %1215, %1217 : f32
      %1219 = arith.divf %1215, %1218 : f32
      linalg.yield %1219 : f32
    } -> tensor<1x8x344xf32>
    %1220 = tensor.empty() : tensor<1x8x344xf32>
    %1221 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1210, %1212 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%1220 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_38", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp"} {
    ^bb125(%1222: f32, %1223: f32, %1224: f32):
      %1225 = arith.mulf %1222, %1223 : f32
      linalg.yield %1225 : f32
    } -> tensor<1x8x344xf32>
    %1226 = tensor.empty() : tensor<128x344xi8>
    %1227 = linalg.transpose ins(%41:tensor<344x128xi8>) outs(%1226:tensor<128x344xi8>) permutation = [1, 0]
    %1228 = tensor.collapse_shape %1191 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1229 = tensor.expand_shape %1228 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} : tensor<1024xf32> into tensor<8x128xf32>
    %1230 = tensor.empty() : tensor<128x344xf32>
    %1231 = arith.constant 0 : i32
    %1232 = tensor.splat %1231 : tensor<344xi32>
    %1233 = "quant_ext.dequantize_per_channel"(%1227, %42, %1232) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<128x344xi8>, tensor<344xf32>, tensor<344xi32>) -> tensor<128x344xf32>
    %1234 = tensor.empty() : tensor<8x344xf32>
    %1235 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1236 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1235 : f32) outs(%1234 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %1237 = linalg.matmul {prov.region_id = "matmul_16", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} ins(%1229, %1233 : tensor<8x128xf32>, tensor<128x344xf32>) outs(%1236 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %1238 = tensor.empty() : tensor<8x344xf32>
    %1239 = tensor.collapse_shape %1237 [[0 : i64, 1 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} : tensor<8x344xf32> into tensor<2752xf32>
    %1240 = tensor.expand_shape %1239 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 344] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} : tensor<2752xf32> into tensor<1x8x344xf32>
    %1241 = tensor.empty() : tensor<1x8x344xf32>
    %1242 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1221, %1240 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%1241 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_40", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp"} {
    ^bb126(%1243: f32, %1244: f32, %1245: f32):
      %1246 = arith.mulf %1243, %1244 : f32
      linalg.yield %1246 : f32
    } -> tensor<1x8x344xf32>
    %1247 = tensor.empty() : tensor<344x128xi8>
    %1248 = linalg.transpose ins(%44:tensor<128x344xi8>) outs(%1247:tensor<344x128xi8>) permutation = [1, 0]
    %1249 = tensor.collapse_shape %1242 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} : tensor<1x8x344xf32> into tensor<2752xf32>
    %1250 = tensor.expand_shape %1249 [[0 : i64, 1 : i64]] output_shape [8, 344] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} : tensor<2752xf32> into tensor<8x344xf32>
    %1251 = tensor.empty() : tensor<344x128xf32>
    %1252 = arith.constant 0 : i32
    %1253 = tensor.splat %1252 : tensor<128xi32>
    %1254 = "quant_ext.dequantize_per_channel"(%1248, %45, %1253) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<344x128xi8>, tensor<128xf32>, tensor<128xi32>) -> tensor<344x128xf32>
    %1255 = tensor.empty() : tensor<8x128xf32>
    %1256 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1257 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1256 : f32) outs(%1255 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %1258 = linalg.matmul {prov.region_id = "matmul_17", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} ins(%1250, %1254 : tensor<8x344xf32>, tensor<344x128xf32>) outs(%1257 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %1259 = tensor.empty() : tensor<8x128xf32>
    %1260 = tensor.collapse_shape %1258 [[0 : i64, 1 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} : tensor<8x128xf32> into tensor<1024xf32>
    %1261 = tensor.expand_shape %1260 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %1262 = tensor.empty() : tensor<1x8x128xf32>
    %1263 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1144, %1261 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%1262 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1"} {
    ^bb127(%1264: f32, %1265: f32, %1266: f32):
      %1267 = arith.addf %1264, %1265 : f32
      linalg.yield %1267 : f32
    } -> tensor<1x8x128xf32>
    %1268 = tensor.empty() : tensor<1x8x128xf32>
    %1269 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1263 : tensor<1x8x128xf32>) outs(%1268 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb128(%1270: f32, %1271: f32):
      %1272 = arith.constant 2.000000e+00 : f32
      %1273 = math.powf %1270, %1272 : f32
      linalg.yield %1273 : f32
    } -> tensor<1x8x128xf32>
    %1274 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} 0.000000e+00 : f32
    %1275 = tensor.splat %1274 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x8xf32>
    %1276 = linalg.reduce ins(%1269:tensor<1x8x128xf32>) outs(%1275:tensor<1x8xf32>) dimensions = [2]
    (%1277: f32, %1278: f32) {
      %1279 = arith.addf %1277, %1278 : f32
      linalg.yield %1279 : f32
    }
    %1280 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} 1.280000e+02 : f32
    %1281 = tensor.splat %1280 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x8xf32>
    %1282 = tensor.empty() : tensor<1x8xf32>
    %1283 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1276, %1281 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%1282 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb129(%1284: f32, %1285: f32, %1286: f32):
      %1287 = arith.divf %1284, %1285 : f32
      linalg.yield %1287 : f32
    } -> tensor<1x8xf32>
    %1288 = tensor.collapse_shape %1283 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x8xf32> into tensor<8xf32>
    %1289 = tensor.expand_shape %1288 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<8xf32> into tensor<1x8x1xf32>
    %1290 = arith.constant {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} 1.000000e-05 : f32
    %1291 = tensor.splat %1290 {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x8x1xf32>
    %1292 = tensor.empty() : tensor<1x8x1xf32>
    %1293 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1289, %1291 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%1292 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb130(%1294: f32, %1295: f32, %1296: f32):
      %1297 = arith.addf %1294, %1295 : f32
      linalg.yield %1297 : f32
    } -> tensor<1x8x1xf32>
    %1298 = tensor.empty() : tensor<1x8x1xf32>
    %1299 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1293 : tensor<1x8x1xf32>) outs(%1298 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb131(%1300: f32, %1301: f32):
      %1302 = math.rsqrt %1300 : f32
      linalg.yield %1302 : f32
    } -> tensor<1x8x1xf32>
    %1303 = tensor.empty() : tensor<1x8x128xf32>
    %1304 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1263, %1299 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%1303 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_42", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb132(%1305: f32, %1306: f32, %1307: f32):
      %1308 = arith.mulf %1305, %1306 : f32
      linalg.yield %1308 : f32
    } -> tensor<1x8x128xf32>
    %1309 = tensor.empty() : tensor<1x8x128xf32>
    %1310 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1304, %47 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%1309 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb133(%1311: f32, %1312: f32, %1313: f32):
      %1314 = arith.mulf %1311, %1312 : f32
      linalg.yield %1314 : f32
    } -> tensor<1x8x128xf32>
    %1315 = tensor.empty() : tensor<128x256xi8>
    %1316 = linalg.transpose ins(%48:tensor<256x128xi8>) outs(%1315:tensor<128x256xi8>) permutation = [1, 0]
    %1317 = tensor.collapse_shape %1310 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1318 = tensor.expand_shape %1317 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : tensor<1024xf32> into tensor<8x128xf32>
    %1319 = tensor.empty() : tensor<128x256xf32>
    %1320 = arith.constant 0 : i32
    %1321 = tensor.splat %1320 : tensor<256xi32>
    %1322 = "quant_ext.dequantize_per_channel"(%1316, %49, %1321) <{axis = 1 : i64, input_dtype = "i8"}> {prov.op = "dequantize", prov.family = "quantize"} : (tensor<128x256xi8>, tensor<256xf32>, tensor<256xi32>) -> tensor<128x256xf32>
    %1323 = tensor.empty() : tensor<8x256xf32>
    %1324 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %1325 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%1324 : f32) outs(%1323 : tensor<8x256xf32>) -> tensor<8x256xf32>
    %1326 = linalg.matmul {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} ins(%1318, %1322 : tensor<8x128xf32>, tensor<128x256xf32>) outs(%1325 : tensor<8x256xf32>) -> tensor<8x256xf32>
    %1327 = tensor.empty() : tensor<8x256xf32>
    %1328 = tensor.collapse_shape %1326 [[0 : i64, 1 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : tensor<8x256xf32> into tensor<2048xf32>
    %1329 = tensor.expand_shape %1328 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 256] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : tensor<2048xf32> into tensor<1x8x256xf32>
    func.return %1329 : tensor<1x8x256xf32>
  }
}
