builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<256x128xf32>, %1: tensor<128xf32>, %2: tensor<128x128xf32>, %3: tensor<128x128xf32>, %4: tensor<128x128xf32>, %5: tensor<128x128xf32>, %6: tensor<128xf32>, %7: tensor<344x128xf32>, %8: tensor<344x128xf32>, %9: tensor<128x344xf32>, %10: tensor<128xf32>, %11: tensor<128x128xf32>, %12: tensor<128x128xf32>, %13: tensor<128x128xf32>, %14: tensor<128x128xf32>, %15: tensor<128xf32>, %16: tensor<344x128xf32>, %17: tensor<344x128xf32>, %18: tensor<128x344xf32>, %19: tensor<128xf32>, %20: tensor<256x128xf32>, %21: tensor<1x8xi64>) -> tensor<1x8x256xf32> {
    %22 = tensor.empty() : tensor<8xi64>
    %23 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%22 : tensor<8xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb0(%24: i64):
      %25 = linalg.index 0 : index
      %26 = arith.index_cast %25 : index to i64
      %27 = arith.constant 1 : i64
      %28 = arith.muli %26, %27 : i64
      %29 = arith.constant 0 : i64
      %30 = arith.addi %29, %28 : i64
      linalg.yield %30 : i64
    } -> tensor<8xi64>
    %31 = tensor.empty() : tensor<1x8x128xf32>
    %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%21 : tensor<1x8xi64>) outs(%31 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32", prov.module = "emb", prov.fqn = "emb"} {
    ^bb1(%33: i64, %34: f32):
      %35 = arith.index_cast %33 : i64 to index
      %36 = linalg.index 2 : index
      %37 = tensor.extract %0[%35, %36] : tensor<256x128xf32>
      linalg.yield %37 : f32
    } -> tensor<1x8x128xf32>
    %38 = tensor.empty() : tensor<1x8x128xf32>
    %39 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32 : tensor<1x8x128xf32>) outs(%38 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb2(%40: f32, %41: f32):
      %42 = arith.constant 2.000000e+00 : f32
      %43 = math.powf %40, %42 : f32
      linalg.yield %43 : f32
    } -> tensor<1x8x128xf32>
    %44 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} 0.000000e+00 : f32
    %45 = tensor.splat %44 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<1x8xf32>
    %46 = linalg.reduce ins(%39:tensor<1x8x128xf32>) outs(%45:tensor<1x8xf32>) dimensions = [2]
    (%47: f32, %48: f32) {
      %49 = arith.addf %47, %48 : f32
      linalg.yield %49 : f32
    }
    %50 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} 1.280000e+02 : f32
    %51 = tensor.splat %50 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<1x8xf32>
    %52 = tensor.empty() : tensor<1x8xf32>
    %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%46, %51 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%52 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb3(%54: f32, %55: f32, %56: f32):
      %57 = arith.divf %54, %55 : f32
      linalg.yield %57 : f32
    } -> tensor<1x8xf32>
    %58 = tensor.collapse_shape %53 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<1x8xf32> into tensor<8xf32>
    %59 = tensor.expand_shape %58 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<8xf32> into tensor<1x8x1xf32>
    %60 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} 1.000000e-05 : f32
    %61 = tensor.splat %60 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} : tensor<1x8x1xf32>
    %62 = tensor.empty() : tensor<1x8x1xf32>
    %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %61 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%62 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb4(%64: f32, %65: f32, %66: f32):
      %67 = arith.addf %64, %65 : f32
      linalg.yield %67 : f32
    } -> tensor<1x8x1xf32>
    %68 = tensor.empty() : tensor<1x8x1xf32>
    %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63 : tensor<1x8x1xf32>) outs(%68 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb5(%70: f32, %71: f32):
      %72 = math.rsqrt %70 : f32
      linalg.yield %72 : f32
    } -> tensor<1x8x1xf32>
    %73 = tensor.empty() : tensor<1x8x128xf32>
    %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %69 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%73 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb6(%75: f32, %76: f32, %77: f32):
      %78 = arith.mulf %75, %76 : f32
      linalg.yield %78 : f32
    } -> tensor<1x8x128xf32>
    %79 = tensor.empty() : tensor<1x8x128xf32>
    %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %1 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%79 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n1"} {
    ^bb7(%81: f32, %82: f32, %83: f32):
      %84 = arith.mulf %81, %82 : f32
      linalg.yield %84 : f32
    } -> tensor<1x8x128xf32>
    %85 = tensor.empty() : tensor<128x128xf32>
    %86 = linalg.transpose ins(%2:tensor<128x128xf32>) outs(%85:tensor<128x128xf32>) permutation = [1, 0]
    %87 = tensor.collapse_shape %80 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %88 = tensor.expand_shape %87 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} : tensor<1024xf32> into tensor<8x128xf32>
    %89 = tensor.empty() : tensor<8x128xf32>
    %90 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %91 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%90 : f32) outs(%89 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %92 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q", prov.transposed_b = "true"} ins(%88, %86 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%91 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %93 = tensor.collapse_shape %92 [[0 : i64, 1 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} : tensor<8x128xf32> into tensor<1024xf32>
    %94 = tensor.expand_shape %93 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.q"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %95 = tensor.collapse_shape %94 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %96 = tensor.expand_shape %95 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %97 = tensor.empty() : tensor<1x4x8x32xf32>
    %98 = linalg.transpose ins(%96:tensor<1x8x4x32xf32>) outs(%97:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %99 = tensor.empty() : tensor<128x128xf32>
    %100 = linalg.transpose ins(%3:tensor<128x128xf32>) outs(%99:tensor<128x128xf32>) permutation = [1, 0]
    %101 = tensor.collapse_shape %80 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %102 = tensor.expand_shape %101 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} : tensor<1024xf32> into tensor<8x128xf32>
    %103 = tensor.empty() : tensor<8x128xf32>
    %104 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %105 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%104 : f32) outs(%103 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %106 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k", prov.transposed_b = "true"} ins(%102, %100 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%105 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %107 = tensor.collapse_shape %106 [[0 : i64, 1 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} : tensor<8x128xf32> into tensor<1024xf32>
    %108 = tensor.expand_shape %107 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.k"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %109 = tensor.collapse_shape %108 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %110 = tensor.expand_shape %109 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %111 = tensor.empty() : tensor<1x4x8x32xf32>
    %112 = linalg.transpose ins(%110:tensor<1x8x4x32xf32>) outs(%111:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %113 = tensor.empty() : tensor<128x128xf32>
    %114 = linalg.transpose ins(%4:tensor<128x128xf32>) outs(%113:tensor<128x128xf32>) permutation = [1, 0]
    %115 = tensor.collapse_shape %80 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %116 = tensor.expand_shape %115 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} : tensor<1024xf32> into tensor<8x128xf32>
    %117 = tensor.empty() : tensor<8x128xf32>
    %118 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %119 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%118 : f32) outs(%117 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %120 = linalg.matmul {prov.region_id = "matmul_2", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v", prov.transposed_b = "true"} ins(%116, %114 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%119 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %121 = tensor.collapse_shape %120 [[0 : i64, 1 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} : tensor<8x128xf32> into tensor<1024xf32>
    %122 = tensor.expand_shape %121 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.v"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %123 = tensor.collapse_shape %122 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %124 = tensor.expand_shape %123 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %125 = tensor.empty() : tensor<1x4x8x32xf32>
    %126 = linalg.transpose ins(%124:tensor<1x8x4x32xf32>) outs(%125:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %127 = tensor.empty() : tensor<16xf32>
    %128 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%127 : tensor<16xf32>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb8(%129: f32):
      %130 = linalg.index 0 : index
      %131 = arith.index_cast %130 : index to i64
      %132 = arith.sitofp %131 : i64 to f32
      %133 = arith.constant 1.000000e+00 : f32
      %134 = arith.mulf %132, %133 : f32
      %135 = arith.constant 0.000000e+00 : f32
      %136 = arith.addf %135, %134 : f32
      linalg.yield %136 : f32
    } -> tensor<16xf32>
    %137 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1.600000e+01 : f32
    %138 = tensor.splat %137 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32>
    %139 = tensor.empty() : tensor<16xf32>
    %140 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%128, %138 : tensor<16xf32>, tensor<16xf32>) outs(%139 : tensor<16xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb9(%141: f32, %142: f32, %143: f32):
      %144 = arith.divf %141, %142 : f32
      linalg.yield %144 : f32
    } -> tensor<16xf32>
    %145 = tensor.empty() : tensor<16xf32>
    %146 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%140 : tensor<16xf32>) outs(%145 : tensor<16xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb10(%147: f32, %148: f32):
      %149 = arith.constant 1.000000e+04 : f32
      %150 = math.powf %149, %147 : f32
      linalg.yield %150 : f32
    } -> tensor<16xf32>
    %151 = tensor.empty() : tensor<16xf32>
    %152 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%146 : tensor<16xf32>) outs(%151 : tensor<16xf32>) attrs =  {prov.region_id = "elementwise_0", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb11(%153: f32, %154: f32):
      %155 = arith.constant 1.000000e+00 : f32
      %156 = arith.divf %155, %153 : f32
      linalg.yield %156 : f32
    } -> tensor<16xf32>
    %157 = arith.constant {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1.000000e+00 : f32
    %158 = tensor.splat %157 {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32>
    %159 = tensor.empty() : tensor<16xf32>
    %160 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%152, %158 : tensor<16xf32>, tensor<16xf32>) outs(%159 : tensor<16xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb12(%161: f32, %162: f32, %163: f32):
      %164 = arith.mulf %161, %162 : f32
      linalg.yield %164 : f32
    } -> tensor<16xf32>
    %165 = tensor.expand_shape %23 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %166 = tensor.empty() : tensor<8x1xf32>
    %167 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%165 : tensor<8x1xi64>) outs(%166 : tensor<8x1xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb13(%168: i64, %169: f32):
      %170 = arith.sitofp %168 : i64 to f32
      linalg.yield %170 : f32
    } -> tensor<8x1xf32>
    %171 = tensor.expand_shape %160 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32> into tensor<1x16xf32>
    %172 = tensor.empty() : tensor<8x16xf32>
    %173 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%167, %171 : tensor<8x1xf32>, tensor<1x16xf32>) outs(%172 : tensor<8x16xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb14(%174: f32, %175: f32, %176: f32):
      %177 = arith.mulf %174, %175 : f32
      linalg.yield %177 : f32
    } -> tensor<8x16xf32>
    %178 = tensor.empty() : tensor<8x16xf32>
    %179 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%173 : tensor<8x16xf32>) outs(%178 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb15(%180: f32, %181: f32):
      %182 = math.cos %180 : f32
      linalg.yield %182 : f32
    } -> tensor<8x16xf32>
    %183 = tensor.empty() : tensor<8x16xf32>
    %184 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%173 : tensor<8x16xf32>) outs(%183 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_1", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb16(%185: f32, %186: f32):
      %187 = math.cos %185 : f32
      linalg.yield %187 : f32
    } -> tensor<8x16xf32>
    %188 = tensor.concat dim(1) %179, %184 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %189 = tensor.collapse_shape %188 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %190 = tensor.expand_shape %189 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %191 = tensor.collapse_shape %190 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %192 = tensor.expand_shape %191 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %193 = tensor.empty() : tensor<8x16xf32>
    %194 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%173 : tensor<8x16xf32>) outs(%193 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb17(%195: f32, %196: f32):
      %197 = math.sin %195 : f32
      linalg.yield %197 : f32
    } -> tensor<8x16xf32>
    %198 = tensor.empty() : tensor<8x16xf32>
    %199 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%173 : tensor<8x16xf32>) outs(%198 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_1", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb18(%200: f32, %201: f32):
      %202 = math.sin %200 : f32
      linalg.yield %202 : f32
    } -> tensor<8x16xf32>
    %203 = tensor.concat dim(1) %194, %199 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %204 = tensor.collapse_shape %203 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %205 = tensor.expand_shape %204 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %206 = tensor.collapse_shape %205 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %207 = tensor.expand_shape %206 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %208 = "tensor.extract_slice"(%98) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %209 = "tensor.extract_slice"(%98) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %210 = tensor.empty() : tensor<1x4x8x16xf32>
    %211 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%209 : tensor<1x4x8x16xf32>) outs(%210 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb19(%212: f32, %213: f32):
      %214 = arith.negf %212 : f32
      linalg.yield %214 : f32
    } -> tensor<1x4x8x16xf32>
    %215 = tensor.concat dim(3) %211, %208 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %216 = tensor.empty() : tensor<1x4x8x32xf32>
    %217 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%98, %192 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%216 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb20(%218: f32, %219: f32, %220: f32):
      %221 = arith.mulf %218, %219 : f32
      linalg.yield %221 : f32
    } -> tensor<1x4x8x32xf32>
    %222 = tensor.empty() : tensor<1x4x8x32xf32>
    %223 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%215, %207 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%222 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb21(%224: f32, %225: f32, %226: f32):
      %227 = arith.mulf %224, %225 : f32
      linalg.yield %227 : f32
    } -> tensor<1x4x8x32xf32>
    %228 = tensor.empty() : tensor<1x4x8x32xf32>
    %229 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%217, %223 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%228 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb22(%230: f32, %231: f32, %232: f32):
      %233 = arith.addf %230, %231 : f32
      linalg.yield %233 : f32
    } -> tensor<1x4x8x32xf32>
    %234 = tensor.empty() : tensor<16xf32>
    %235 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%234 : tensor<16xf32>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb23(%236: f32):
      %237 = linalg.index 0 : index
      %238 = arith.index_cast %237 : index to i64
      %239 = arith.sitofp %238 : i64 to f32
      %240 = arith.constant 1.000000e+00 : f32
      %241 = arith.mulf %239, %240 : f32
      %242 = arith.constant 0.000000e+00 : f32
      %243 = arith.addf %242, %241 : f32
      linalg.yield %243 : f32
    } -> tensor<16xf32>
    %244 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1.600000e+01 : f32
    %245 = tensor.splat %244 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32>
    %246 = tensor.empty() : tensor<16xf32>
    %247 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%235, %245 : tensor<16xf32>, tensor<16xf32>) outs(%246 : tensor<16xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb24(%248: f32, %249: f32, %250: f32):
      %251 = arith.divf %248, %249 : f32
      linalg.yield %251 : f32
    } -> tensor<16xf32>
    %252 = tensor.empty() : tensor<16xf32>
    %253 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%247 : tensor<16xf32>) outs(%252 : tensor<16xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb25(%254: f32, %255: f32):
      %256 = arith.constant 1.000000e+04 : f32
      %257 = math.powf %256, %254 : f32
      linalg.yield %257 : f32
    } -> tensor<16xf32>
    %258 = tensor.empty() : tensor<16xf32>
    %259 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%253 : tensor<16xf32>) outs(%258 : tensor<16xf32>) attrs =  {prov.region_id = "elementwise_1", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb26(%260: f32, %261: f32):
      %262 = arith.constant 1.000000e+00 : f32
      %263 = arith.divf %262, %260 : f32
      linalg.yield %263 : f32
    } -> tensor<16xf32>
    %264 = arith.constant {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1.000000e+00 : f32
    %265 = tensor.splat %264 {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32>
    %266 = tensor.empty() : tensor<16xf32>
    %267 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%259, %265 : tensor<16xf32>, tensor<16xf32>) outs(%266 : tensor<16xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb27(%268: f32, %269: f32, %270: f32):
      %271 = arith.mulf %268, %269 : f32
      linalg.yield %271 : f32
    } -> tensor<16xf32>
    %272 = tensor.expand_shape %23 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %273 = tensor.empty() : tensor<8x1xf32>
    %274 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%272 : tensor<8x1xi64>) outs(%273 : tensor<8x1xf32>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb28(%275: i64, %276: f32):
      %277 = arith.sitofp %275 : i64 to f32
      linalg.yield %277 : f32
    } -> tensor<8x1xf32>
    %278 = tensor.expand_shape %267 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<16xf32> into tensor<1x16xf32>
    %279 = tensor.empty() : tensor<8x16xf32>
    %280 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%274, %278 : tensor<8x1xf32>, tensor<1x16xf32>) outs(%279 : tensor<8x16xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb29(%281: f32, %282: f32, %283: f32):
      %284 = arith.mulf %281, %282 : f32
      linalg.yield %284 : f32
    } -> tensor<8x16xf32>
    %285 = tensor.empty() : tensor<8x16xf32>
    %286 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%280 : tensor<8x16xf32>) outs(%285 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_2", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb30(%287: f32, %288: f32):
      %289 = math.cos %287 : f32
      linalg.yield %289 : f32
    } -> tensor<8x16xf32>
    %290 = tensor.empty() : tensor<8x16xf32>
    %291 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%280 : tensor<8x16xf32>) outs(%290 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_3", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb31(%292: f32, %293: f32):
      %294 = math.cos %292 : f32
      linalg.yield %294 : f32
    } -> tensor<8x16xf32>
    %295 = tensor.concat dim(1) %286, %291 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %296 = tensor.collapse_shape %295 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %297 = tensor.expand_shape %296 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %298 = tensor.collapse_shape %297 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %299 = tensor.expand_shape %298 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %300 = tensor.empty() : tensor<8x16xf32>
    %301 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%280 : tensor<8x16xf32>) outs(%300 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_2", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb32(%302: f32, %303: f32):
      %304 = math.sin %302 : f32
      linalg.yield %304 : f32
    } -> tensor<8x16xf32>
    %305 = tensor.empty() : tensor<8x16xf32>
    %306 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%280 : tensor<8x16xf32>) outs(%305 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_3", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb33(%307: f32, %308: f32):
      %309 = math.sin %307 : f32
      linalg.yield %309 : f32
    } -> tensor<8x16xf32>
    %310 = tensor.concat dim(1) %301, %306 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %311 = tensor.collapse_shape %310 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %312 = tensor.expand_shape %311 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %313 = tensor.collapse_shape %312 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %314 = tensor.expand_shape %313 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %315 = "tensor.extract_slice"(%112) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %316 = "tensor.extract_slice"(%112) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %317 = tensor.empty() : tensor<1x4x8x16xf32>
    %318 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%316 : tensor<1x4x8x16xf32>) outs(%317 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb34(%319: f32, %320: f32):
      %321 = arith.negf %319 : f32
      linalg.yield %321 : f32
    } -> tensor<1x4x8x16xf32>
    %322 = tensor.concat dim(3) %318, %315 {prov.region_id = "cat_5", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %323 = tensor.empty() : tensor<1x4x8x32xf32>
    %324 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%112, %299 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%323 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb35(%325: f32, %326: f32, %327: f32):
      %328 = arith.mulf %325, %326 : f32
      linalg.yield %328 : f32
    } -> tensor<1x4x8x32xf32>
    %329 = tensor.empty() : tensor<1x4x8x32xf32>
    %330 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%322, %314 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%329 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb36(%331: f32, %332: f32, %333: f32):
      %334 = arith.mulf %331, %332 : f32
      linalg.yield %334 : f32
    } -> tensor<1x4x8x32xf32>
    %335 = tensor.empty() : tensor<1x4x8x32xf32>
    %336 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%324, %330 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%335 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb37(%337: f32, %338: f32, %339: f32):
      %340 = arith.addf %337, %338 : f32
      linalg.yield %340 : f32
    } -> tensor<1x4x8x32xf32>
    %341 = tensor.empty() : tensor<1x4x32x8xf32>
    %342 = linalg.transpose ins(%336:tensor<1x4x8x32xf32>) outs(%341:tensor<1x4x32x8xf32>) permutation = [0, 1, 3, 2]
    %343 = tensor.empty() : tensor<1x4x8x32xf32>
    %344 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%229 : tensor<1x4x8x32xf32>) outs(%343 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb38(%345: f32, %346: f32):
      linalg.yield %345 : f32
    } -> tensor<1x4x8x32xf32>
    %347 = tensor.collapse_shape %344 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8x32xf32> into tensor<1024xf32>
    %348 = tensor.expand_shape %347 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 32] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<4x8x32xf32>
    %349 = tensor.empty() : tensor<1x4x32x8xf32>
    %350 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%342 : tensor<1x4x32x8xf32>) outs(%349 : tensor<1x4x32x8xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb39(%351: f32, %352: f32):
      linalg.yield %351 : f32
    } -> tensor<1x4x32x8xf32>
    %353 = tensor.collapse_shape %350 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x32x8xf32> into tensor<1024xf32>
    %354 = tensor.expand_shape %353 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 32, 8] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<4x32x8xf32>
    %355 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0.000000e+00 : f32
    %356 = tensor.splat %355 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<4x8x8xf32>
    %357 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%348, %354 : tensor<4x8x32xf32>, tensor<4x32x8xf32>) outs(%356 : tensor<4x8x8xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb40(%358: f32, %359: f32, %360: f32):
      %361 = arith.mulf %358, %359 : f32
      %362 = arith.addf %360, %361 : f32
      linalg.yield %362 : f32
    } -> tensor<4x8x8xf32>
    %363 = tensor.collapse_shape %357 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<4x8x8xf32> into tensor<256xf32>
    %364 = tensor.expand_shape %363 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 8] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<1x4x8x8xf32>
    %365 = arith.constant {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 5.65685415 : f32
    %366 = tensor.splat %365 {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8x8xf32>
    %367 = tensor.empty() : tensor<1x4x8x8xf32>
    %368 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%364, %366 : tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>) outs(%367 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb41(%369: f32, %370: f32, %371: f32):
      %372 = arith.divf %369, %370 : f32
      linalg.yield %372 : f32
    } -> tensor<1x4x8x8xf32>
    %373 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0xff800000 : f32
    %374 = tensor.splat %373 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x8xf32>
    %375 = tensor.empty() : tensor<8xi64>
    %376 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%375 : tensor<8xi64>) attrs =  {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb42(%377: i64):
      %378 = linalg.index 0 : index
      %379 = arith.index_cast %378 : index to i64
      %380 = arith.constant 1 : i64
      %381 = arith.muli %379, %380 : i64
      %382 = arith.constant 0 : i64
      %383 = arith.addi %382, %381 : i64
      linalg.yield %383 : i64
    } -> tensor<8xi64>
    %384 = tensor.expand_shape %376 [[0 : i64, 1 : i64]] output_shape [1, 8] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8xi64> into tensor<1x8xi64>
    %385 = tensor.empty() : tensor<8xi64>
    %386 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%385 : tensor<8xi64>) attrs =  {prov.region_id = "iota_4", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb43(%387: i64):
      %388 = linalg.index 0 : index
      %389 = arith.index_cast %388 : index to i64
      %390 = arith.constant 1 : i64
      %391 = arith.muli %389, %390 : i64
      %392 = arith.constant 0 : i64
      %393 = arith.addi %392, %391 : i64
      linalg.yield %393 : i64
    } -> tensor<8xi64>
    %394 = tensor.expand_shape %386 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %395 = tensor.empty() : tensor<8x8xi64>
    %396 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%384, %394 : tensor<1x8xi64>, tensor<8x1xi64>) outs(%395 : tensor<8x8xi64>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb44(%397: i64, %398: i64, %399: i64):
      %400 = arith.subi %397, %398 : i64
      linalg.yield %400 : i64
    } -> tensor<8x8xi64>
    %401 = arith.constant {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 1 : i64
    %402 = tensor.splat %401 {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<8x8xi64>
    %403 = tensor.empty() : tensor<8x8xi1>
    %404 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%396, %402 : tensor<8x8xi64>, tensor<8x8xi64>) outs(%403 : tensor<8x8xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb45(%405: i64, %406: i64, %407: i1):
      %408 = arith.cmpi sge, %405, %406 : i64
      linalg.yield %408 : i1
    } -> tensor<8x8xi1>
    %409 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0.000000e+00 : f32
    %410 = tensor.splat %409 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<f32>
    %411 = tensor.empty() : tensor<8x8xf32>
    %412 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%404, %374, %410 : tensor<8x8xi1>, tensor<8x8xf32>, tensor<f32>) outs(%411 : tensor<8x8xf32>) attrs =  {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb46(%413: i1, %414: f32, %415: f32, %416: f32):
      %417 = arith.select %413, %414, %415 : f32
      linalg.yield %417 : f32
    } -> tensor<8x8xf32>
    %418 = tensor.empty() : tensor<1x4x8x8xf32>
    %419 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%368, %412 : tensor<1x4x8x8xf32>, tensor<8x8xf32>) outs(%418 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb47(%420: f32, %421: f32, %422: f32):
      %423 = arith.addf %420, %421 : f32
      linalg.yield %423 : f32
    } -> tensor<1x4x8x8xf32>
    %424 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0xff800000 : f32
    %425 = tensor.splat %424 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8xf32>
    %426 = linalg.reduce ins(%419:tensor<1x4x8x8xf32>) outs(%425:tensor<1x4x8xf32>) dimensions = [3]
    (%427: f32, %428: f32) {
      %429 = arith.maximumf %427, %428 : f32
      linalg.yield %429 : f32
    }
    %430 = tensor.collapse_shape %426 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8xf32> into tensor<32xf32>
    %431 = tensor.expand_shape %430 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %432 = tensor.empty() : tensor<1x4x8x8xf32>
    %433 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%419, %431 : tensor<1x4x8x8xf32>, tensor<1x4x8x1xf32>) outs(%432 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb48(%434: f32, %435: f32, %436: f32):
      %437 = arith.subf %434, %435 : f32
      linalg.yield %437 : f32
    } -> tensor<1x4x8x8xf32>
    %438 = tensor.empty() : tensor<1x4x8x8xf32>
    %439 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%433 : tensor<1x4x8x8xf32>) outs(%438 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb49(%440: f32, %441: f32):
      %442 = math.exp %440 : f32
      linalg.yield %442 : f32
    } -> tensor<1x4x8x8xf32>
    %443 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0.000000e+00 : f32
    %444 = tensor.splat %443 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8xf32>
    %445 = linalg.reduce ins(%439:tensor<1x4x8x8xf32>) outs(%444:tensor<1x4x8xf32>) dimensions = [3]
    (%446: f32, %447: f32) {
      %448 = arith.addf %446, %447 : f32
      linalg.yield %448 : f32
    }
    %449 = tensor.collapse_shape %445 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8xf32> into tensor<32xf32>
    %450 = tensor.expand_shape %449 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %451 = tensor.empty() : tensor<1x4x8x8xf32>
    %452 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%439, %450 : tensor<1x4x8x8xf32>, tensor<1x4x8x1xf32>) outs(%451 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb50(%453: f32, %454: f32, %455: f32):
      %456 = arith.divf %453, %454 : f32
      linalg.yield %456 : f32
    } -> tensor<1x4x8x8xf32>
    %457 = tensor.empty() : tensor<1x4x8x8xf32>
    %458 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%452 : tensor<1x4x8x8xf32>) outs(%457 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb51(%459: f32, %460: f32):
      linalg.yield %459 : f32
    } -> tensor<1x4x8x8xf32>
    %461 = tensor.collapse_shape %458 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8x8xf32> into tensor<256xf32>
    %462 = tensor.expand_shape %461 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 8] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<256xf32> into tensor<4x8x8xf32>
    %463 = tensor.empty() : tensor<1x4x8x32xf32>
    %464 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%126 : tensor<1x4x8x32xf32>) outs(%463 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb52(%465: f32, %466: f32):
      linalg.yield %465 : f32
    } -> tensor<1x4x8x32xf32>
    %467 = tensor.collapse_shape %464 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x4x8x32xf32> into tensor<1024xf32>
    %468 = tensor.expand_shape %467 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 32] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<4x8x32xf32>
    %469 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} 0.000000e+00 : f32
    %470 = tensor.splat %469 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<4x8x32xf32>
    %471 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%462, %468 : tensor<4x8x8xf32>, tensor<4x8x32xf32>) outs(%470 : tensor<4x8x32xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} {
    ^bb53(%472: f32, %473: f32, %474: f32):
      %475 = arith.mulf %472, %473 : f32
      %476 = arith.addf %474, %475 : f32
      linalg.yield %476 : f32
    } -> tensor<4x8x32xf32>
    %477 = tensor.collapse_shape %471 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<4x8x32xf32> into tensor<1024xf32>
    %478 = tensor.expand_shape %477 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 32] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x4x8x32xf32>
    %479 = tensor.empty() : tensor<1x8x4x32xf32>
    %480 = linalg.transpose ins(%478:tensor<1x4x8x32xf32>) outs(%479:tensor<1x8x4x32xf32>) permutation = [0, 2, 1, 3]
    %481 = tensor.collapse_shape %480 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1x8x4x32xf32> into tensor<1024xf32>
    %482 = tensor.expand_shape %481 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %483 = tensor.empty() : tensor<128x128xf32>
    %484 = linalg.transpose ins(%5:tensor<128x128xf32>) outs(%483:tensor<128x128xf32>) permutation = [1, 0]
    %485 = tensor.collapse_shape %482 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %486 = tensor.expand_shape %485 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} : tensor<1024xf32> into tensor<8x128xf32>
    %487 = tensor.empty() : tensor<8x128xf32>
    %488 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %489 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%488 : f32) outs(%487 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %490 = linalg.matmul {prov.region_id = "matmul_5", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o", prov.transposed_b = "true"} ins(%486, %484 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%489 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %491 = tensor.collapse_shape %490 [[0 : i64, 1 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} : tensor<8x128xf32> into tensor<1024xf32>
    %492 = tensor.expand_shape %491 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.attn.o"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %493 = tensor.empty() : tensor<1x8x128xf32>
    %494 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %492 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%493 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0"} {
    ^bb54(%495: f32, %496: f32, %497: f32):
      %498 = arith.addf %495, %496 : f32
      linalg.yield %498 : f32
    } -> tensor<1x8x128xf32>
    %499 = tensor.empty() : tensor<1x8x128xf32>
    %500 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%494 : tensor<1x8x128xf32>) outs(%499 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb55(%501: f32, %502: f32):
      %503 = arith.constant 2.000000e+00 : f32
      %504 = math.powf %501, %503 : f32
      linalg.yield %504 : f32
    } -> tensor<1x8x128xf32>
    %505 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} 0.000000e+00 : f32
    %506 = tensor.splat %505 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<1x8xf32>
    %507 = linalg.reduce ins(%500:tensor<1x8x128xf32>) outs(%506:tensor<1x8xf32>) dimensions = [2]
    (%508: f32, %509: f32) {
      %510 = arith.addf %508, %509 : f32
      linalg.yield %510 : f32
    }
    %511 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} 1.280000e+02 : f32
    %512 = tensor.splat %511 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<1x8xf32>
    %513 = tensor.empty() : tensor<1x8xf32>
    %514 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%507, %512 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%513 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb56(%515: f32, %516: f32, %517: f32):
      %518 = arith.divf %515, %516 : f32
      linalg.yield %518 : f32
    } -> tensor<1x8xf32>
    %519 = tensor.collapse_shape %514 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<1x8xf32> into tensor<8xf32>
    %520 = tensor.expand_shape %519 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<8xf32> into tensor<1x8x1xf32>
    %521 = arith.constant {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} 1.000000e-05 : f32
    %522 = tensor.splat %521 {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} : tensor<1x8x1xf32>
    %523 = tensor.empty() : tensor<1x8x1xf32>
    %524 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%520, %522 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%523 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb57(%525: f32, %526: f32, %527: f32):
      %528 = arith.addf %525, %526 : f32
      linalg.yield %528 : f32
    } -> tensor<1x8x1xf32>
    %529 = tensor.empty() : tensor<1x8x1xf32>
    %530 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%524 : tensor<1x8x1xf32>) outs(%529 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb58(%531: f32, %532: f32):
      %533 = math.rsqrt %531 : f32
      linalg.yield %533 : f32
    } -> tensor<1x8x1xf32>
    %534 = tensor.empty() : tensor<1x8x128xf32>
    %535 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%494, %530 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%534 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb59(%536: f32, %537: f32, %538: f32):
      %539 = arith.mulf %536, %537 : f32
      linalg.yield %539 : f32
    } -> tensor<1x8x128xf32>
    %540 = tensor.empty() : tensor<1x8x128xf32>
    %541 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%535, %6 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%540 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.n2"} {
    ^bb60(%542: f32, %543: f32, %544: f32):
      %545 = arith.mulf %542, %543 : f32
      linalg.yield %545 : f32
    } -> tensor<1x8x128xf32>
    %546 = tensor.empty() : tensor<128x344xf32>
    %547 = linalg.transpose ins(%7:tensor<344x128xf32>) outs(%546:tensor<128x344xf32>) permutation = [1, 0]
    %548 = tensor.collapse_shape %541 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %549 = tensor.expand_shape %548 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} : tensor<1024xf32> into tensor<8x128xf32>
    %550 = tensor.empty() : tensor<8x344xf32>
    %551 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %552 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%551 : f32) outs(%550 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %553 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g", prov.transposed_b = "true"} ins(%549, %547 : tensor<8x128xf32>, tensor<128x344xf32>) outs(%552 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %554 = tensor.collapse_shape %553 [[0 : i64, 1 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} : tensor<8x344xf32> into tensor<2752xf32>
    %555 = tensor.expand_shape %554 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 344] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.g"} : tensor<2752xf32> into tensor<1x8x344xf32>
    %556 = tensor.empty() : tensor<1x8x344xf32>
    %557 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%555 : tensor<1x8x344xf32>) outs(%556 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp"} {
    ^bb61(%558: f32, %559: f32):
      %560 = arith.constant 1.000000e+00 : f32
      %561 = arith.negf %558 : f32
      %562 = math.exp %561 : f32
      %563 = arith.addf %560, %562 : f32
      %564 = arith.divf %560, %563 : f32
      linalg.yield %564 : f32
    } -> tensor<1x8x344xf32>
    %565 = tensor.empty() : tensor<1x8x344xf32>
    %566 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%555, %557 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%565 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp"} {
    ^bb62(%567: f32, %568: f32, %569: f32):
      %570 = arith.mulf %567, %568 : f32
      linalg.yield %570 : f32
    } -> tensor<1x8x344xf32>
    %571 = tensor.empty() : tensor<128x344xf32>
    %572 = linalg.transpose ins(%8:tensor<344x128xf32>) outs(%571:tensor<128x344xf32>) permutation = [1, 0]
    %573 = tensor.collapse_shape %541 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %574 = tensor.expand_shape %573 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} : tensor<1024xf32> into tensor<8x128xf32>
    %575 = tensor.empty() : tensor<8x344xf32>
    %576 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %577 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%576 : f32) outs(%575 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %578 = linalg.matmul {prov.region_id = "matmul_7", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u", prov.transposed_b = "true"} ins(%574, %572 : tensor<8x128xf32>, tensor<128x344xf32>) outs(%577 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %579 = tensor.collapse_shape %578 [[0 : i64, 1 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} : tensor<8x344xf32> into tensor<2752xf32>
    %580 = tensor.expand_shape %579 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 344] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.u"} : tensor<2752xf32> into tensor<1x8x344xf32>
    %581 = tensor.empty() : tensor<1x8x344xf32>
    %582 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%566, %580 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%581 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp"} {
    ^bb63(%583: f32, %584: f32, %585: f32):
      %586 = arith.mulf %583, %584 : f32
      linalg.yield %586 : f32
    } -> tensor<1x8x344xf32>
    %587 = tensor.empty() : tensor<344x128xf32>
    %588 = linalg.transpose ins(%9:tensor<128x344xf32>) outs(%587:tensor<344x128xf32>) permutation = [1, 0]
    %589 = tensor.collapse_shape %582 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} : tensor<1x8x344xf32> into tensor<2752xf32>
    %590 = tensor.expand_shape %589 [[0 : i64, 1 : i64]] output_shape [8, 344] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} : tensor<2752xf32> into tensor<8x344xf32>
    %591 = tensor.empty() : tensor<8x128xf32>
    %592 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %593 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%592 : f32) outs(%591 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %594 = linalg.matmul {prov.region_id = "matmul_8", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn", prov.transposed_b = "true"} ins(%590, %588 : tensor<8x344xf32>, tensor<344x128xf32>) outs(%593 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %595 = tensor.collapse_shape %594 [[0 : i64, 1 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} : tensor<8x128xf32> into tensor<1024xf32>
    %596 = tensor.expand_shape %595 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.dn"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %597 = tensor.empty() : tensor<1x8x128xf32>
    %598 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%494, %596 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%597 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0"} {
    ^bb64(%599: f32, %600: f32, %601: f32):
      %602 = arith.addf %599, %600 : f32
      linalg.yield %602 : f32
    } -> tensor<1x8x128xf32>
    %603 = tensor.empty() : tensor<1x8x128xf32>
    %604 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%598 : tensor<1x8x128xf32>) outs(%603 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb65(%605: f32, %606: f32):
      %607 = arith.constant 2.000000e+00 : f32
      %608 = math.powf %605, %607 : f32
      linalg.yield %608 : f32
    } -> tensor<1x8x128xf32>
    %609 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} 0.000000e+00 : f32
    %610 = tensor.splat %609 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<1x8xf32>
    %611 = linalg.reduce ins(%604:tensor<1x8x128xf32>) outs(%610:tensor<1x8xf32>) dimensions = [2]
    (%612: f32, %613: f32) {
      %614 = arith.addf %612, %613 : f32
      linalg.yield %614 : f32
    }
    %615 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} 1.280000e+02 : f32
    %616 = tensor.splat %615 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<1x8xf32>
    %617 = tensor.empty() : tensor<1x8xf32>
    %618 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%611, %616 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%617 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb66(%619: f32, %620: f32, %621: f32):
      %622 = arith.divf %619, %620 : f32
      linalg.yield %622 : f32
    } -> tensor<1x8xf32>
    %623 = tensor.collapse_shape %618 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<1x8xf32> into tensor<8xf32>
    %624 = tensor.expand_shape %623 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<8xf32> into tensor<1x8x1xf32>
    %625 = arith.constant {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} 1.000000e-05 : f32
    %626 = tensor.splat %625 {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} : tensor<1x8x1xf32>
    %627 = tensor.empty() : tensor<1x8x1xf32>
    %628 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%624, %626 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%627 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb67(%629: f32, %630: f32, %631: f32):
      %632 = arith.addf %629, %630 : f32
      linalg.yield %632 : f32
    } -> tensor<1x8x1xf32>
    %633 = tensor.empty() : tensor<1x8x1xf32>
    %634 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%628 : tensor<1x8x1xf32>) outs(%633 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb68(%635: f32, %636: f32):
      %637 = math.rsqrt %635 : f32
      linalg.yield %637 : f32
    } -> tensor<1x8x1xf32>
    %638 = tensor.empty() : tensor<1x8x128xf32>
    %639 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%598, %634 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%638 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb69(%640: f32, %641: f32, %642: f32):
      %643 = arith.mulf %640, %641 : f32
      linalg.yield %643 : f32
    } -> tensor<1x8x128xf32>
    %644 = tensor.empty() : tensor<1x8x128xf32>
    %645 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%639, %10 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%644 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n1"} {
    ^bb70(%646: f32, %647: f32, %648: f32):
      %649 = arith.mulf %646, %647 : f32
      linalg.yield %649 : f32
    } -> tensor<1x8x128xf32>
    %650 = tensor.empty() : tensor<128x128xf32>
    %651 = linalg.transpose ins(%11:tensor<128x128xf32>) outs(%650:tensor<128x128xf32>) permutation = [1, 0]
    %652 = tensor.collapse_shape %645 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %653 = tensor.expand_shape %652 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} : tensor<1024xf32> into tensor<8x128xf32>
    %654 = tensor.empty() : tensor<8x128xf32>
    %655 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %656 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%655 : f32) outs(%654 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %657 = linalg.matmul {prov.region_id = "matmul_9", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q", prov.transposed_b = "true"} ins(%653, %651 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%656 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %658 = tensor.collapse_shape %657 [[0 : i64, 1 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} : tensor<8x128xf32> into tensor<1024xf32>
    %659 = tensor.expand_shape %658 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.q"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %660 = tensor.collapse_shape %659 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %661 = tensor.expand_shape %660 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %662 = tensor.empty() : tensor<1x4x8x32xf32>
    %663 = linalg.transpose ins(%661:tensor<1x8x4x32xf32>) outs(%662:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %664 = tensor.empty() : tensor<128x128xf32>
    %665 = linalg.transpose ins(%12:tensor<128x128xf32>) outs(%664:tensor<128x128xf32>) permutation = [1, 0]
    %666 = tensor.collapse_shape %645 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %667 = tensor.expand_shape %666 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} : tensor<1024xf32> into tensor<8x128xf32>
    %668 = tensor.empty() : tensor<8x128xf32>
    %669 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %670 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%669 : f32) outs(%668 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %671 = linalg.matmul {prov.region_id = "matmul_10", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k", prov.transposed_b = "true"} ins(%667, %665 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%670 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %672 = tensor.collapse_shape %671 [[0 : i64, 1 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} : tensor<8x128xf32> into tensor<1024xf32>
    %673 = tensor.expand_shape %672 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.k"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %674 = tensor.collapse_shape %673 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %675 = tensor.expand_shape %674 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %676 = tensor.empty() : tensor<1x4x8x32xf32>
    %677 = linalg.transpose ins(%675:tensor<1x8x4x32xf32>) outs(%676:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %678 = tensor.empty() : tensor<128x128xf32>
    %679 = linalg.transpose ins(%13:tensor<128x128xf32>) outs(%678:tensor<128x128xf32>) permutation = [1, 0]
    %680 = tensor.collapse_shape %645 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %681 = tensor.expand_shape %680 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} : tensor<1024xf32> into tensor<8x128xf32>
    %682 = tensor.empty() : tensor<8x128xf32>
    %683 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %684 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%683 : f32) outs(%682 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %685 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v", prov.transposed_b = "true"} ins(%681, %679 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%684 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %686 = tensor.collapse_shape %685 [[0 : i64, 1 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} : tensor<8x128xf32> into tensor<1024xf32>
    %687 = tensor.expand_shape %686 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.v"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %688 = tensor.collapse_shape %687 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %689 = tensor.expand_shape %688 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %690 = tensor.empty() : tensor<1x4x8x32xf32>
    %691 = linalg.transpose ins(%689:tensor<1x8x4x32xf32>) outs(%690:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %692 = tensor.empty() : tensor<16xf32>
    %693 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%692 : tensor<16xf32>) attrs =  {prov.region_id = "iota_5", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb71(%694: f32):
      %695 = linalg.index 0 : index
      %696 = arith.index_cast %695 : index to i64
      %697 = arith.sitofp %696 : i64 to f32
      %698 = arith.constant 1.000000e+00 : f32
      %699 = arith.mulf %697, %698 : f32
      %700 = arith.constant 0.000000e+00 : f32
      %701 = arith.addf %700, %699 : f32
      linalg.yield %701 : f32
    } -> tensor<16xf32>
    %702 = arith.constant {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1.600000e+01 : f32
    %703 = tensor.splat %702 {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32>
    %704 = tensor.empty() : tensor<16xf32>
    %705 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%693, %703 : tensor<16xf32>, tensor<16xf32>) outs(%704 : tensor<16xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb72(%706: f32, %707: f32, %708: f32):
      %709 = arith.divf %706, %707 : f32
      linalg.yield %709 : f32
    } -> tensor<16xf32>
    %710 = tensor.empty() : tensor<16xf32>
    %711 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%705 : tensor<16xf32>) outs(%710 : tensor<16xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb73(%712: f32, %713: f32):
      %714 = arith.constant 1.000000e+04 : f32
      %715 = math.powf %714, %712 : f32
      linalg.yield %715 : f32
    } -> tensor<16xf32>
    %716 = tensor.empty() : tensor<16xf32>
    %717 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%711 : tensor<16xf32>) outs(%716 : tensor<16xf32>) attrs =  {prov.region_id = "elementwise_2", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb74(%718: f32, %719: f32):
      %720 = arith.constant 1.000000e+00 : f32
      %721 = arith.divf %720, %718 : f32
      linalg.yield %721 : f32
    } -> tensor<16xf32>
    %722 = arith.constant {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1.000000e+00 : f32
    %723 = tensor.splat %722 {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32>
    %724 = tensor.empty() : tensor<16xf32>
    %725 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%717, %723 : tensor<16xf32>, tensor<16xf32>) outs(%724 : tensor<16xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb75(%726: f32, %727: f32, %728: f32):
      %729 = arith.mulf %726, %727 : f32
      linalg.yield %729 : f32
    } -> tensor<16xf32>
    %730 = tensor.expand_shape %23 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %731 = tensor.empty() : tensor<8x1xf32>
    %732 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%730 : tensor<8x1xi64>) outs(%731 : tensor<8x1xf32>) attrs =  {prov.region_id = "dtype_cast_2", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb76(%733: i64, %734: f32):
      %735 = arith.sitofp %733 : i64 to f32
      linalg.yield %735 : f32
    } -> tensor<8x1xf32>
    %736 = tensor.expand_shape %725 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32> into tensor<1x16xf32>
    %737 = tensor.empty() : tensor<8x16xf32>
    %738 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%732, %736 : tensor<8x1xf32>, tensor<1x16xf32>) outs(%737 : tensor<8x16xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb77(%739: f32, %740: f32, %741: f32):
      %742 = arith.mulf %739, %740 : f32
      linalg.yield %742 : f32
    } -> tensor<8x16xf32>
    %743 = tensor.empty() : tensor<8x16xf32>
    %744 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%738 : tensor<8x16xf32>) outs(%743 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_4", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb78(%745: f32, %746: f32):
      %747 = math.cos %745 : f32
      linalg.yield %747 : f32
    } -> tensor<8x16xf32>
    %748 = tensor.empty() : tensor<8x16xf32>
    %749 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%738 : tensor<8x16xf32>) outs(%748 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_5", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb79(%750: f32, %751: f32):
      %752 = math.cos %750 : f32
      linalg.yield %752 : f32
    } -> tensor<8x16xf32>
    %753 = tensor.concat dim(1) %744, %749 {prov.region_id = "cat_6", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %754 = tensor.collapse_shape %753 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %755 = tensor.expand_shape %754 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %756 = tensor.collapse_shape %755 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %757 = tensor.expand_shape %756 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %758 = tensor.empty() : tensor<8x16xf32>
    %759 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%738 : tensor<8x16xf32>) outs(%758 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_4", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb80(%760: f32, %761: f32):
      %762 = math.sin %760 : f32
      linalg.yield %762 : f32
    } -> tensor<8x16xf32>
    %763 = tensor.empty() : tensor<8x16xf32>
    %764 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%738 : tensor<8x16xf32>) outs(%763 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_5", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb81(%765: f32, %766: f32):
      %767 = math.sin %765 : f32
      linalg.yield %767 : f32
    } -> tensor<8x16xf32>
    %768 = tensor.concat dim(1) %759, %764 {prov.region_id = "cat_7", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %769 = tensor.collapse_shape %768 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %770 = tensor.expand_shape %769 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %771 = tensor.collapse_shape %770 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %772 = tensor.expand_shape %771 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %773 = "tensor.extract_slice"(%663) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %774 = "tensor.extract_slice"(%663) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %775 = tensor.empty() : tensor<1x4x8x16xf32>
    %776 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%774 : tensor<1x4x8x16xf32>) outs(%775 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb82(%777: f32, %778: f32):
      %779 = arith.negf %777 : f32
      linalg.yield %779 : f32
    } -> tensor<1x4x8x16xf32>
    %780 = tensor.concat dim(3) %776, %773 {prov.region_id = "cat_8", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %781 = tensor.empty() : tensor<1x4x8x32xf32>
    %782 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%663, %757 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%781 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb83(%783: f32, %784: f32, %785: f32):
      %786 = arith.mulf %783, %784 : f32
      linalg.yield %786 : f32
    } -> tensor<1x4x8x32xf32>
    %787 = tensor.empty() : tensor<1x4x8x32xf32>
    %788 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%780, %772 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%787 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb84(%789: f32, %790: f32, %791: f32):
      %792 = arith.mulf %789, %790 : f32
      linalg.yield %792 : f32
    } -> tensor<1x4x8x32xf32>
    %793 = tensor.empty() : tensor<1x4x8x32xf32>
    %794 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%782, %788 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%793 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb85(%795: f32, %796: f32, %797: f32):
      %798 = arith.addf %795, %796 : f32
      linalg.yield %798 : f32
    } -> tensor<1x4x8x32xf32>
    %799 = tensor.empty() : tensor<16xf32>
    %800 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%799 : tensor<16xf32>) attrs =  {prov.region_id = "iota_6", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb86(%801: f32):
      %802 = linalg.index 0 : index
      %803 = arith.index_cast %802 : index to i64
      %804 = arith.sitofp %803 : i64 to f32
      %805 = arith.constant 1.000000e+00 : f32
      %806 = arith.mulf %804, %805 : f32
      %807 = arith.constant 0.000000e+00 : f32
      %808 = arith.addf %807, %806 : f32
      linalg.yield %808 : f32
    } -> tensor<16xf32>
    %809 = arith.constant {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1.600000e+01 : f32
    %810 = tensor.splat %809 {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32>
    %811 = tensor.empty() : tensor<16xf32>
    %812 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%800, %810 : tensor<16xf32>, tensor<16xf32>) outs(%811 : tensor<16xf32>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb87(%813: f32, %814: f32, %815: f32):
      %816 = arith.divf %813, %814 : f32
      linalg.yield %816 : f32
    } -> tensor<16xf32>
    %817 = tensor.empty() : tensor<16xf32>
    %818 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%812 : tensor<16xf32>) outs(%817 : tensor<16xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb88(%819: f32, %820: f32):
      %821 = arith.constant 1.000000e+04 : f32
      %822 = math.powf %821, %819 : f32
      linalg.yield %822 : f32
    } -> tensor<16xf32>
    %823 = tensor.empty() : tensor<16xf32>
    %824 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%818 : tensor<16xf32>) outs(%823 : tensor<16xf32>) attrs =  {prov.region_id = "elementwise_3", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb89(%825: f32, %826: f32):
      %827 = arith.constant 1.000000e+00 : f32
      %828 = arith.divf %827, %825 : f32
      linalg.yield %828 : f32
    } -> tensor<16xf32>
    %829 = arith.constant {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1.000000e+00 : f32
    %830 = tensor.splat %829 {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32>
    %831 = tensor.empty() : tensor<16xf32>
    %832 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%824, %830 : tensor<16xf32>, tensor<16xf32>) outs(%831 : tensor<16xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb90(%833: f32, %834: f32, %835: f32):
      %836 = arith.mulf %833, %834 : f32
      linalg.yield %836 : f32
    } -> tensor<16xf32>
    %837 = tensor.expand_shape %23 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %838 = tensor.empty() : tensor<8x1xf32>
    %839 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%837 : tensor<8x1xi64>) outs(%838 : tensor<8x1xf32>) attrs =  {prov.region_id = "dtype_cast_3", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb91(%840: i64, %841: f32):
      %842 = arith.sitofp %840 : i64 to f32
      linalg.yield %842 : f32
    } -> tensor<8x1xf32>
    %843 = tensor.expand_shape %832 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<16xf32> into tensor<1x16xf32>
    %844 = tensor.empty() : tensor<8x16xf32>
    %845 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%839, %843 : tensor<8x1xf32>, tensor<1x16xf32>) outs(%844 : tensor<8x16xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb92(%846: f32, %847: f32, %848: f32):
      %849 = arith.mulf %846, %847 : f32
      linalg.yield %849 : f32
    } -> tensor<8x16xf32>
    %850 = tensor.empty() : tensor<8x16xf32>
    %851 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%845 : tensor<8x16xf32>) outs(%850 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_6", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb93(%852: f32, %853: f32):
      %854 = math.cos %852 : f32
      linalg.yield %854 : f32
    } -> tensor<8x16xf32>
    %855 = tensor.empty() : tensor<8x16xf32>
    %856 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%845 : tensor<8x16xf32>) outs(%855 : tensor<8x16xf32>) attrs =  {prov.region_id = "cos_7", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb94(%857: f32, %858: f32):
      %859 = math.cos %857 : f32
      linalg.yield %859 : f32
    } -> tensor<8x16xf32>
    %860 = tensor.concat dim(1) %851, %856 {prov.region_id = "cat_9", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %861 = tensor.collapse_shape %860 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %862 = tensor.expand_shape %861 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %863 = tensor.collapse_shape %862 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %864 = tensor.expand_shape %863 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %865 = tensor.empty() : tensor<8x16xf32>
    %866 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%845 : tensor<8x16xf32>) outs(%865 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_6", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb95(%867: f32, %868: f32):
      %869 = math.sin %867 : f32
      linalg.yield %869 : f32
    } -> tensor<8x16xf32>
    %870 = tensor.empty() : tensor<8x16xf32>
    %871 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%845 : tensor<8x16xf32>) outs(%870 : tensor<8x16xf32>) attrs =  {prov.region_id = "sin_7", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb96(%872: f32, %873: f32):
      %874 = math.sin %872 : f32
      linalg.yield %874 : f32
    } -> tensor<8x16xf32>
    %875 = tensor.concat dim(1) %866, %871 {prov.region_id = "cat_10", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x32xf32>
    %876 = tensor.collapse_shape %875 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x32xf32> into tensor<256xf32>
    %877 = tensor.expand_shape %876 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x8x32xf32>
    %878 = tensor.collapse_shape %877 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %879 = tensor.expand_shape %878 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %880 = "tensor.extract_slice"(%677) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %881 = "tensor.extract_slice"(%677) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %882 = tensor.empty() : tensor<1x4x8x16xf32>
    %883 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%881 : tensor<1x4x8x16xf32>) outs(%882 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb97(%884: f32, %885: f32):
      %886 = arith.negf %884 : f32
      linalg.yield %886 : f32
    } -> tensor<1x4x8x16xf32>
    %887 = tensor.concat dim(3) %883, %880 {prov.region_id = "cat_11", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %888 = tensor.empty() : tensor<1x4x8x32xf32>
    %889 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%677, %864 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%888 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb98(%890: f32, %891: f32, %892: f32):
      %893 = arith.mulf %890, %891 : f32
      linalg.yield %893 : f32
    } -> tensor<1x4x8x32xf32>
    %894 = tensor.empty() : tensor<1x4x8x32xf32>
    %895 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%887, %879 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%894 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb99(%896: f32, %897: f32, %898: f32):
      %899 = arith.mulf %896, %897 : f32
      linalg.yield %899 : f32
    } -> tensor<1x4x8x32xf32>
    %900 = tensor.empty() : tensor<1x4x8x32xf32>
    %901 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%889, %895 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%900 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb100(%902: f32, %903: f32, %904: f32):
      %905 = arith.addf %902, %903 : f32
      linalg.yield %905 : f32
    } -> tensor<1x4x8x32xf32>
    %906 = tensor.empty() : tensor<1x4x32x8xf32>
    %907 = linalg.transpose ins(%901:tensor<1x4x8x32xf32>) outs(%906:tensor<1x4x32x8xf32>) permutation = [0, 1, 3, 2]
    %908 = tensor.empty() : tensor<1x4x8x32xf32>
    %909 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%794 : tensor<1x4x8x32xf32>) outs(%908 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb101(%910: f32, %911: f32):
      linalg.yield %910 : f32
    } -> tensor<1x4x8x32xf32>
    %912 = tensor.collapse_shape %909 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8x32xf32> into tensor<1024xf32>
    %913 = tensor.expand_shape %912 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 32] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<4x8x32xf32>
    %914 = tensor.empty() : tensor<1x4x32x8xf32>
    %915 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%907 : tensor<1x4x32x8xf32>) outs(%914 : tensor<1x4x32x8xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb102(%916: f32, %917: f32):
      linalg.yield %916 : f32
    } -> tensor<1x4x32x8xf32>
    %918 = tensor.collapse_shape %915 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x32x8xf32> into tensor<1024xf32>
    %919 = tensor.expand_shape %918 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 32, 8] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<4x32x8xf32>
    %920 = arith.constant {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0.000000e+00 : f32
    %921 = tensor.splat %920 {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<4x8x8xf32>
    %922 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%913, %919 : tensor<4x8x32xf32>, tensor<4x32x8xf32>) outs(%921 : tensor<4x8x8xf32>) attrs =  {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb103(%923: f32, %924: f32, %925: f32):
      %926 = arith.mulf %923, %924 : f32
      %927 = arith.addf %925, %926 : f32
      linalg.yield %927 : f32
    } -> tensor<4x8x8xf32>
    %928 = tensor.collapse_shape %922 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<4x8x8xf32> into tensor<256xf32>
    %929 = tensor.expand_shape %928 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 8] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<1x4x8x8xf32>
    %930 = arith.constant {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 5.65685415 : f32
    %931 = tensor.splat %930 {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8x8xf32>
    %932 = tensor.empty() : tensor<1x4x8x8xf32>
    %933 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%929, %931 : tensor<1x4x8x8xf32>, tensor<1x4x8x8xf32>) outs(%932 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb104(%934: f32, %935: f32, %936: f32):
      %937 = arith.divf %934, %935 : f32
      linalg.yield %937 : f32
    } -> tensor<1x4x8x8xf32>
    %938 = arith.constant {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0xff800000 : f32
    %939 = tensor.splat %938 {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x8xf32>
    %940 = tensor.empty() : tensor<8xi64>
    %941 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%940 : tensor<8xi64>) attrs =  {prov.region_id = "iota_7", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb105(%942: i64):
      %943 = linalg.index 0 : index
      %944 = arith.index_cast %943 : index to i64
      %945 = arith.constant 1 : i64
      %946 = arith.muli %944, %945 : i64
      %947 = arith.constant 0 : i64
      %948 = arith.addi %947, %946 : i64
      linalg.yield %948 : i64
    } -> tensor<8xi64>
    %949 = tensor.expand_shape %941 [[0 : i64, 1 : i64]] output_shape [1, 8] {prov.region_id = "unsqueeze_26", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8xi64> into tensor<1x8xi64>
    %950 = tensor.empty() : tensor<8xi64>
    %951 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%950 : tensor<8xi64>) attrs =  {prov.region_id = "iota_8", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb106(%952: i64):
      %953 = linalg.index 0 : index
      %954 = arith.index_cast %953 : index to i64
      %955 = arith.constant 1 : i64
      %956 = arith.muli %954, %955 : i64
      %957 = arith.constant 0 : i64
      %958 = arith.addi %957, %956 : i64
      linalg.yield %958 : i64
    } -> tensor<8xi64>
    %959 = tensor.expand_shape %951 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_27", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8xi64> into tensor<8x1xi64>
    %960 = tensor.empty() : tensor<8x8xi64>
    %961 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%949, %959 : tensor<1x8xi64>, tensor<8x1xi64>) outs(%960 : tensor<8x8xi64>) attrs =  {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb107(%962: i64, %963: i64, %964: i64):
      %965 = arith.subi %962, %963 : i64
      linalg.yield %965 : i64
    } -> tensor<8x8xi64>
    %966 = arith.constant {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 1 : i64
    %967 = tensor.splat %966 {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<8x8xi64>
    %968 = tensor.empty() : tensor<8x8xi1>
    %969 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%961, %967 : tensor<8x8xi64>, tensor<8x8xi64>) outs(%968 : tensor<8x8xi1>) attrs =  {prov.region_id = "compare_1", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb108(%970: i64, %971: i64, %972: i1):
      %973 = arith.cmpi sge, %970, %971 : i64
      linalg.yield %973 : i1
    } -> tensor<8x8xi1>
    %974 = arith.constant {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0.000000e+00 : f32
    %975 = tensor.splat %974 {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<f32>
    %976 = tensor.empty() : tensor<8x8xf32>
    %977 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%969, %939, %975 : tensor<8x8xi1>, tensor<8x8xf32>, tensor<f32>) outs(%976 : tensor<8x8xf32>) attrs =  {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb109(%978: i1, %979: f32, %980: f32, %981: f32):
      %982 = arith.select %978, %979, %980 : f32
      linalg.yield %982 : f32
    } -> tensor<8x8xf32>
    %983 = tensor.empty() : tensor<1x4x8x8xf32>
    %984 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%933, %977 : tensor<1x4x8x8xf32>, tensor<8x8xf32>) outs(%983 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb110(%985: f32, %986: f32, %987: f32):
      %988 = arith.addf %985, %986 : f32
      linalg.yield %988 : f32
    } -> tensor<1x4x8x8xf32>
    %989 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0xff800000 : f32
    %990 = tensor.splat %989 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8xf32>
    %991 = linalg.reduce ins(%984:tensor<1x4x8x8xf32>) outs(%990:tensor<1x4x8xf32>) dimensions = [3]
    (%992: f32, %993: f32) {
      %994 = arith.maximumf %992, %993 : f32
      linalg.yield %994 : f32
    }
    %995 = tensor.collapse_shape %991 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8xf32> into tensor<32xf32>
    %996 = tensor.expand_shape %995 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %997 = tensor.empty() : tensor<1x4x8x8xf32>
    %998 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%984, %996 : tensor<1x4x8x8xf32>, tensor<1x4x8x1xf32>) outs(%997 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb111(%999: f32, %1000: f32, %1001: f32):
      %1002 = arith.subf %999, %1000 : f32
      linalg.yield %1002 : f32
    } -> tensor<1x4x8x8xf32>
    %1003 = tensor.empty() : tensor<1x4x8x8xf32>
    %1004 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%998 : tensor<1x4x8x8xf32>) outs(%1003 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb112(%1005: f32, %1006: f32):
      %1007 = math.exp %1005 : f32
      linalg.yield %1007 : f32
    } -> tensor<1x4x8x8xf32>
    %1008 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0.000000e+00 : f32
    %1009 = tensor.splat %1008 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8xf32>
    %1010 = linalg.reduce ins(%1004:tensor<1x4x8x8xf32>) outs(%1009:tensor<1x4x8xf32>) dimensions = [3]
    (%1011: f32, %1012: f32) {
      %1013 = arith.addf %1011, %1012 : f32
      linalg.yield %1013 : f32
    }
    %1014 = tensor.collapse_shape %1010 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8xf32> into tensor<32xf32>
    %1015 = tensor.expand_shape %1014 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %1016 = tensor.empty() : tensor<1x4x8x8xf32>
    %1017 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1004, %1015 : tensor<1x4x8x8xf32>, tensor<1x4x8x1xf32>) outs(%1016 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb113(%1018: f32, %1019: f32, %1020: f32):
      %1021 = arith.divf %1018, %1019 : f32
      linalg.yield %1021 : f32
    } -> tensor<1x4x8x8xf32>
    %1022 = tensor.empty() : tensor<1x4x8x8xf32>
    %1023 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1017 : tensor<1x4x8x8xf32>) outs(%1022 : tensor<1x4x8x8xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb114(%1024: f32, %1025: f32):
      linalg.yield %1024 : f32
    } -> tensor<1x4x8x8xf32>
    %1026 = tensor.collapse_shape %1023 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8x8xf32> into tensor<256xf32>
    %1027 = tensor.expand_shape %1026 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 8] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<256xf32> into tensor<4x8x8xf32>
    %1028 = tensor.empty() : tensor<1x4x8x32xf32>
    %1029 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%691 : tensor<1x4x8x32xf32>) outs(%1028 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb115(%1030: f32, %1031: f32):
      linalg.yield %1030 : f32
    } -> tensor<1x4x8x32xf32>
    %1032 = tensor.collapse_shape %1029 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x4x8x32xf32> into tensor<1024xf32>
    %1033 = tensor.expand_shape %1032 [[0 : i64, 1 : i64, 2 : i64]] output_shape [4, 8, 32] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<4x8x32xf32>
    %1034 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} 0.000000e+00 : f32
    %1035 = tensor.splat %1034 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<4x8x32xf32>
    %1036 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1027, %1033 : tensor<4x8x8xf32>, tensor<4x8x32xf32>) outs(%1035 : tensor<4x8x32xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} {
    ^bb116(%1037: f32, %1038: f32, %1039: f32):
      %1040 = arith.mulf %1037, %1038 : f32
      %1041 = arith.addf %1039, %1040 : f32
      linalg.yield %1041 : f32
    } -> tensor<4x8x32xf32>
    %1042 = tensor.collapse_shape %1036 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<4x8x32xf32> into tensor<1024xf32>
    %1043 = tensor.expand_shape %1042 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 32] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x4x8x32xf32>
    %1044 = tensor.empty() : tensor<1x8x4x32xf32>
    %1045 = linalg.transpose ins(%1043:tensor<1x4x8x32xf32>) outs(%1044:tensor<1x8x4x32xf32>) permutation = [0, 2, 1, 3]
    %1046 = tensor.collapse_shape %1045 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1x8x4x32xf32> into tensor<1024xf32>
    %1047 = tensor.expand_shape %1046 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %1048 = tensor.empty() : tensor<128x128xf32>
    %1049 = linalg.transpose ins(%14:tensor<128x128xf32>) outs(%1048:tensor<128x128xf32>) permutation = [1, 0]
    %1050 = tensor.collapse_shape %1047 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1051 = tensor.expand_shape %1050 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} : tensor<1024xf32> into tensor<8x128xf32>
    %1052 = tensor.empty() : tensor<8x128xf32>
    %1053 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1054 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1053 : f32) outs(%1052 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %1055 = linalg.matmul {prov.region_id = "matmul_14", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o", prov.transposed_b = "true"} ins(%1051, %1049 : tensor<8x128xf32>, tensor<128x128xf32>) outs(%1054 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %1056 = tensor.collapse_shape %1055 [[0 : i64, 1 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} : tensor<8x128xf32> into tensor<1024xf32>
    %1057 = tensor.expand_shape %1056 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.attn.o"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %1058 = tensor.empty() : tensor<1x8x128xf32>
    %1059 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%598, %1057 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%1058 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1"} {
    ^bb117(%1060: f32, %1061: f32, %1062: f32):
      %1063 = arith.addf %1060, %1061 : f32
      linalg.yield %1063 : f32
    } -> tensor<1x8x128xf32>
    %1064 = tensor.empty() : tensor<1x8x128xf32>
    %1065 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1059 : tensor<1x8x128xf32>) outs(%1064 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb118(%1066: f32, %1067: f32):
      %1068 = arith.constant 2.000000e+00 : f32
      %1069 = math.powf %1066, %1068 : f32
      linalg.yield %1069 : f32
    } -> tensor<1x8x128xf32>
    %1070 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} 0.000000e+00 : f32
    %1071 = tensor.splat %1070 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<1x8xf32>
    %1072 = linalg.reduce ins(%1065:tensor<1x8x128xf32>) outs(%1071:tensor<1x8xf32>) dimensions = [2]
    (%1073: f32, %1074: f32) {
      %1075 = arith.addf %1073, %1074 : f32
      linalg.yield %1075 : f32
    }
    %1076 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} 1.280000e+02 : f32
    %1077 = tensor.splat %1076 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<1x8xf32>
    %1078 = tensor.empty() : tensor<1x8xf32>
    %1079 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1072, %1077 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%1078 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb119(%1080: f32, %1081: f32, %1082: f32):
      %1083 = arith.divf %1080, %1081 : f32
      linalg.yield %1083 : f32
    } -> tensor<1x8xf32>
    %1084 = tensor.collapse_shape %1079 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<1x8xf32> into tensor<8xf32>
    %1085 = tensor.expand_shape %1084 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<8xf32> into tensor<1x8x1xf32>
    %1086 = arith.constant {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} 1.000000e-05 : f32
    %1087 = tensor.splat %1086 {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} : tensor<1x8x1xf32>
    %1088 = tensor.empty() : tensor<1x8x1xf32>
    %1089 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1085, %1087 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%1088 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb120(%1090: f32, %1091: f32, %1092: f32):
      %1093 = arith.addf %1090, %1091 : f32
      linalg.yield %1093 : f32
    } -> tensor<1x8x1xf32>
    %1094 = tensor.empty() : tensor<1x8x1xf32>
    %1095 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1089 : tensor<1x8x1xf32>) outs(%1094 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb121(%1096: f32, %1097: f32):
      %1098 = math.rsqrt %1096 : f32
      linalg.yield %1098 : f32
    } -> tensor<1x8x1xf32>
    %1099 = tensor.empty() : tensor<1x8x128xf32>
    %1100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1059, %1095 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%1099 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb122(%1101: f32, %1102: f32, %1103: f32):
      %1104 = arith.mulf %1101, %1102 : f32
      linalg.yield %1104 : f32
    } -> tensor<1x8x128xf32>
    %1105 = tensor.empty() : tensor<1x8x128xf32>
    %1106 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1100, %15 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%1105 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.n2"} {
    ^bb123(%1107: f32, %1108: f32, %1109: f32):
      %1110 = arith.mulf %1107, %1108 : f32
      linalg.yield %1110 : f32
    } -> tensor<1x8x128xf32>
    %1111 = tensor.empty() : tensor<128x344xf32>
    %1112 = linalg.transpose ins(%16:tensor<344x128xf32>) outs(%1111:tensor<128x344xf32>) permutation = [1, 0]
    %1113 = tensor.collapse_shape %1106 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1114 = tensor.expand_shape %1113 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} : tensor<1024xf32> into tensor<8x128xf32>
    %1115 = tensor.empty() : tensor<8x344xf32>
    %1116 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1117 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1116 : f32) outs(%1115 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %1118 = linalg.matmul {prov.region_id = "matmul_15", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g", prov.transposed_b = "true"} ins(%1114, %1112 : tensor<8x128xf32>, tensor<128x344xf32>) outs(%1117 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %1119 = tensor.collapse_shape %1118 [[0 : i64, 1 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} : tensor<8x344xf32> into tensor<2752xf32>
    %1120 = tensor.expand_shape %1119 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 344] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.g"} : tensor<2752xf32> into tensor<1x8x344xf32>
    %1121 = tensor.empty() : tensor<1x8x344xf32>
    %1122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1120 : tensor<1x8x344xf32>) outs(%1121 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp"} {
    ^bb124(%1123: f32, %1124: f32):
      %1125 = arith.constant 1.000000e+00 : f32
      %1126 = arith.negf %1123 : f32
      %1127 = math.exp %1126 : f32
      %1128 = arith.addf %1125, %1127 : f32
      %1129 = arith.divf %1125, %1128 : f32
      linalg.yield %1129 : f32
    } -> tensor<1x8x344xf32>
    %1130 = tensor.empty() : tensor<1x8x344xf32>
    %1131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1120, %1122 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%1130 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp"} {
    ^bb125(%1132: f32, %1133: f32, %1134: f32):
      %1135 = arith.mulf %1132, %1133 : f32
      linalg.yield %1135 : f32
    } -> tensor<1x8x344xf32>
    %1136 = tensor.empty() : tensor<128x344xf32>
    %1137 = linalg.transpose ins(%17:tensor<344x128xf32>) outs(%1136:tensor<128x344xf32>) permutation = [1, 0]
    %1138 = tensor.collapse_shape %1106 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1139 = tensor.expand_shape %1138 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} : tensor<1024xf32> into tensor<8x128xf32>
    %1140 = tensor.empty() : tensor<8x344xf32>
    %1141 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1142 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1141 : f32) outs(%1140 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %1143 = linalg.matmul {prov.region_id = "matmul_16", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u", prov.transposed_b = "true"} ins(%1139, %1137 : tensor<8x128xf32>, tensor<128x344xf32>) outs(%1142 : tensor<8x344xf32>) -> tensor<8x344xf32>
    %1144 = tensor.collapse_shape %1143 [[0 : i64, 1 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} : tensor<8x344xf32> into tensor<2752xf32>
    %1145 = tensor.expand_shape %1144 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 344] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.u"} : tensor<2752xf32> into tensor<1x8x344xf32>
    %1146 = tensor.empty() : tensor<1x8x344xf32>
    %1147 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1131, %1145 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%1146 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp"} {
    ^bb126(%1148: f32, %1149: f32, %1150: f32):
      %1151 = arith.mulf %1148, %1149 : f32
      linalg.yield %1151 : f32
    } -> tensor<1x8x344xf32>
    %1152 = tensor.empty() : tensor<344x128xf32>
    %1153 = linalg.transpose ins(%18:tensor<128x344xf32>) outs(%1152:tensor<344x128xf32>) permutation = [1, 0]
    %1154 = tensor.collapse_shape %1147 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} : tensor<1x8x344xf32> into tensor<2752xf32>
    %1155 = tensor.expand_shape %1154 [[0 : i64, 1 : i64]] output_shape [8, 344] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} : tensor<2752xf32> into tensor<8x344xf32>
    %1156 = tensor.empty() : tensor<8x128xf32>
    %1157 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1158 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1157 : f32) outs(%1156 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %1159 = linalg.matmul {prov.region_id = "matmul_17", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn", prov.transposed_b = "true"} ins(%1155, %1153 : tensor<8x344xf32>, tensor<344x128xf32>) outs(%1158 : tensor<8x128xf32>) -> tensor<8x128xf32>
    %1160 = tensor.collapse_shape %1159 [[0 : i64, 1 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} : tensor<8x128xf32> into tensor<1024xf32>
    %1161 = tensor.expand_shape %1160 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.dn"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %1162 = tensor.empty() : tensor<1x8x128xf32>
    %1163 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1059, %1161 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%1162 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1"} {
    ^bb127(%1164: f32, %1165: f32, %1166: f32):
      %1167 = arith.addf %1164, %1165 : f32
      linalg.yield %1167 : f32
    } -> tensor<1x8x128xf32>
    %1168 = tensor.empty() : tensor<1x8x128xf32>
    %1169 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1163 : tensor<1x8x128xf32>) outs(%1168 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb128(%1170: f32, %1171: f32):
      %1172 = arith.constant 2.000000e+00 : f32
      %1173 = math.powf %1170, %1172 : f32
      linalg.yield %1173 : f32
    } -> tensor<1x8x128xf32>
    %1174 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} 0.000000e+00 : f32
    %1175 = tensor.splat %1174 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x8xf32>
    %1176 = linalg.reduce ins(%1169:tensor<1x8x128xf32>) outs(%1175:tensor<1x8xf32>) dimensions = [2]
    (%1177: f32, %1178: f32) {
      %1179 = arith.addf %1177, %1178 : f32
      linalg.yield %1179 : f32
    }
    %1180 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} 1.280000e+02 : f32
    %1181 = tensor.splat %1180 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x8xf32>
    %1182 = tensor.empty() : tensor<1x8xf32>
    %1183 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1176, %1181 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%1182 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb129(%1184: f32, %1185: f32, %1186: f32):
      %1187 = arith.divf %1184, %1185 : f32
      linalg.yield %1187 : f32
    } -> tensor<1x8xf32>
    %1188 = tensor.collapse_shape %1183 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x8xf32> into tensor<8xf32>
    %1189 = tensor.expand_shape %1188 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<8xf32> into tensor<1x8x1xf32>
    %1190 = arith.constant {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} 1.000000e-05 : f32
    %1191 = tensor.splat %1190 {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} : tensor<1x8x1xf32>
    %1192 = tensor.empty() : tensor<1x8x1xf32>
    %1193 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1189, %1191 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%1192 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb130(%1194: f32, %1195: f32, %1196: f32):
      %1197 = arith.addf %1194, %1195 : f32
      linalg.yield %1197 : f32
    } -> tensor<1x8x1xf32>
    %1198 = tensor.empty() : tensor<1x8x1xf32>
    %1199 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1193 : tensor<1x8x1xf32>) outs(%1198 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb131(%1200: f32, %1201: f32):
      %1202 = math.rsqrt %1200 : f32
      linalg.yield %1202 : f32
    } -> tensor<1x8x1xf32>
    %1203 = tensor.empty() : tensor<1x8x128xf32>
    %1204 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1163, %1199 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%1203 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb132(%1205: f32, %1206: f32, %1207: f32):
      %1208 = arith.mulf %1205, %1206 : f32
      linalg.yield %1208 : f32
    } -> tensor<1x8x128xf32>
    %1209 = tensor.empty() : tensor<1x8x128xf32>
    %1210 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1204, %19 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%1209 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "norm", prov.fqn = "norm"} {
    ^bb133(%1211: f32, %1212: f32, %1213: f32):
      %1214 = arith.mulf %1211, %1212 : f32
      linalg.yield %1214 : f32
    } -> tensor<1x8x128xf32>
    %1215 = tensor.empty() : tensor<128x256xf32>
    %1216 = linalg.transpose ins(%20:tensor<256x128xf32>) outs(%1215:tensor<128x256xf32>) permutation = [1, 0]
    %1217 = tensor.collapse_shape %1210 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1218 = tensor.expand_shape %1217 [[0 : i64, 1 : i64]] output_shape [8, 128] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : tensor<1024xf32> into tensor<8x128xf32>
    %1219 = tensor.empty() : tensor<8x256xf32>
    %1220 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %1221 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%1220 : f32) outs(%1219 : tensor<8x256xf32>) -> tensor<8x256xf32>
    %1222 = linalg.matmul {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm", prov.transposed_b = "true"} ins(%1218, %1216 : tensor<8x128xf32>, tensor<128x256xf32>) outs(%1221 : tensor<8x256xf32>) -> tensor<8x256xf32>
    %1223 = tensor.collapse_shape %1222 [[0 : i64, 1 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : tensor<8x256xf32> into tensor<2048xf32>
    %1224 = tensor.expand_shape %1223 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 256] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : tensor<2048xf32> into tensor<1x8x256xf32>
    func.return %1224 : tensor<1x8x256xf32>
  }
}
