builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func private @aten_index_put_default(tensor<4x15x64xf32>, tensor<8xi64>, tensor<1x4x8x64xf32>) -> tensor<1x4x15x64xf32>
  func.func private @aten_index_put_default_wl0(tensor<1x7xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x7xi64>
  func.func private @aten_index_put_default_1_wl1(tensor<4x15x64xf32>, tensor<1xi64>, tensor<1x4x1x64xf32>) -> tensor<1x4x15x64xf32>
  func.func @forward(%0: tensor<32xf32>, %1: tensor<32000x2048xf32>, %2: tensor<2048x2048xf32>, %3: tensor<256x2048xf32>, %4: tensor<256x2048xf32>, %5: tensor<2048x2048xf32>, %6: tensor<5632x2048xf32>, %7: tensor<5632x2048xf32>, %8: tensor<2048x5632xf32>, %9: tensor<2048xf32>, %10: tensor<2048xf32>, %11: tensor<2048x2048xf32>, %12: tensor<256x2048xf32>, %13: tensor<256x2048xf32>, %14: tensor<2048x2048xf32>, %15: tensor<5632x2048xf32>, %16: tensor<5632x2048xf32>, %17: tensor<2048x5632xf32>, %18: tensor<2048xf32>, %19: tensor<2048xf32>, %20: tensor<2048xf32>, %21: tensor<32000x2048xf32>, %22: tensor<32000x2048xf32>, %23: tensor<2048xf32>, %24: tensor<2048x5632xf32>, %25: tensor<5632x2048xf32>, %26: tensor<2048xf32>, %27: tensor<256x2048xf32>, %28: tensor<2048x2048xf32>, %29: tensor<2048xf32>, %30: tensor<2048x2048xf32>, %31: tensor<5632x2048xf32>, %32: tensor<256x2048xf32>, %33: tensor<2048x5632xf32>, %34: tensor<5632x2048xf32>, %35: tensor<2048xf32>, %36: tensor<256x2048xf32>, %37: tensor<2048x2048xf32>, %38: tensor<2048xf32>, %39: tensor<2048x2048xf32>, %40: tensor<5632x2048xf32>, %41: tensor<256x2048xf32>, %42: tensor<32000x2048xf32>, %43: tensor<i64>, %44: tensor<32xf32>, %45: tensor<32xf32>, %46: tensor<1x8xi64>) -> tensor<1x7xi64> {
    %47 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %48 = tensor.splat %47 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32"} : tensor<2x1x4x15x64xf32>
    %49 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %50 = tensor.splat %49 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32"} : tensor<2x1x4x15x64xf32>
    %51 = tensor.empty() : tensor<1x8x2048xf32>
    %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46 : tensor<1x8xi64>) outs(%51 : tensor<1x8x2048xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32"} {
    ^bb0(%53: i64, %54: f32):
      %55 = arith.index_cast %53 : i64 to index
      %56 = linalg.index 2 : index
      %57 = tensor.extract %22[%55, %56] : tensor<32000x2048xf32>
      linalg.yield %57 : f32
    } -> tensor<1x8x2048xf32>
    %58 = arith.constant {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "int64"} 0 : i64
    %59 = tensor.splat %58 {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "int64"} : tensor<i64>
    %60 = tensor.empty() : tensor<8xi64>
    %61 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%60 : tensor<8xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb1(%62: i64):
      %63 = linalg.index 0 : index
      %64 = arith.index_cast %63 : index to i64
      %65 = arith.constant 1 : i64
      %66 = arith.muli %64, %65 : i64
      %67 = arith.constant 0 : i64
      %68 = arith.addi %67, %66 : i64
      linalg.yield %68 : i64
    } -> tensor<8xi64>
    %69 = tensor.expand_shape %61 [[0 : i64, 1 : i64]] output_shape [1, 8] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<8xi64> into tensor<1x8xi64>
    %70 = tensor.expand_shape %0 [[0 : i64, 1 : i64]] output_shape [1, 32] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32xf32>
    %71 = tensor.collapse_shape %70 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
    %72 = tensor.expand_shape %71 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32x1xf32>
    %73 = tensor.empty() : tensor<1x32x1xf32>
    %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%72 : tensor<1x32x1xf32>) outs(%73 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb2(%75: f32, %76: f32):
      linalg.yield %75 : f32
    } -> tensor<1x32x1xf32>
    %77 = tensor.collapse_shape %69 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1x8xi64> into tensor<8xi64>
    %78 = tensor.expand_shape %77 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 8] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<8xi64> into tensor<1x1x8xi64>
    %79 = tensor.empty() : tensor<1x1x8xf32>
    %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78 : tensor<1x1x8xi64>) outs(%79 : tensor<1x1x8xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb3(%81: i64, %82: f32):
      %83 = arith.sitofp %81 : i64 to f32
      linalg.yield %83 : f32
    } -> tensor<1x1x8xf32>
    %84 = tensor.empty() : tensor<1x32x1xf32>
    %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74 : tensor<1x32x1xf32>) outs(%84 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb4(%86: f32, %87: f32):
      linalg.yield %86 : f32
    } -> tensor<1x32x1xf32>
    %88 = tensor.empty() : tensor<1x1x8xf32>
    %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%80 : tensor<1x1x8xf32>) outs(%88 : tensor<1x1x8xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb5(%90: f32, %91: f32):
      linalg.yield %90 : f32
    } -> tensor<1x1x8xf32>
    %92 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %93 = tensor.splat %92 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<1x32x8xf32>
    %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%85, %89 : tensor<1x32x1xf32>, tensor<1x1x8xf32>) outs(%93 : tensor<1x32x8xf32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb6(%95: f32, %96: f32, %97: f32):
      %98 = arith.mulf %95, %96 : f32
      %99 = arith.addf %97, %98 : f32
      linalg.yield %99 : f32
    } -> tensor<1x32x8xf32>
    %100 = tensor.empty() : tensor<1x8x32xf32>
    %101 = linalg.transpose ins(%94:tensor<1x32x8xf32>) outs(%100:tensor<1x8x32xf32>) permutation = [0, 2, 1]
    %102 = tensor.concat dim(2) %101, %101 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x8x32xf32>, tensor<1x8x32xf32>) -> tensor<1x8x64xf32>
    %103 = tensor.empty() : tensor<1x8x64xf32>
    %104 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%102 : tensor<1x8x64xf32>) outs(%103 : tensor<1x8x64xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32"} {
    ^bb7(%105: f32, %106: f32):
      %107 = math.cos %105 : f32
      linalg.yield %107 : f32
    } -> tensor<1x8x64xf32>
    %108 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
    %109 = tensor.splat %108 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x64xf32>
    %110 = tensor.empty() : tensor<1x8x64xf32>
    %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%104, %109 : tensor<1x8x64xf32>, tensor<1x8x64xf32>) outs(%110 : tensor<1x8x64xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb8(%112: f32, %113: f32, %114: f32):
      %115 = arith.mulf %112, %113 : f32
      linalg.yield %115 : f32
    } -> tensor<1x8x64xf32>
    %116 = tensor.empty() : tensor<1x8x64xf32>
    %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%102 : tensor<1x8x64xf32>) outs(%116 : tensor<1x8x64xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32"} {
    ^bb9(%118: f32, %119: f32):
      %120 = math.sin %118 : f32
      linalg.yield %120 : f32
    } -> tensor<1x8x64xf32>
    %121 = arith.constant {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
    %122 = tensor.splat %121 {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x64xf32>
    %123 = tensor.empty() : tensor<1x8x64xf32>
    %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%117, %122 : tensor<1x8x64xf32>, tensor<1x8x64xf32>) outs(%123 : tensor<1x8x64xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb10(%125: f32, %126: f32, %127: f32):
      %128 = arith.mulf %125, %126 : f32
      linalg.yield %128 : f32
    } -> tensor<1x8x64xf32>
    %129 = tensor.empty() : tensor<1x8x2048xf32>
    %130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%52 : tensor<1x8x2048xf32>) outs(%129 : tensor<1x8x2048xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb11(%131: f32, %132: f32):
      %133 = arith.constant 2.000000e+00 : f32
      %134 = math.powf %131, %133 : f32
      linalg.yield %134 : f32
    } -> tensor<1x8x2048xf32>
    %135 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %136 = tensor.splat %135 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %137 = linalg.reduce ins(%130:tensor<1x8x2048xf32>) outs(%136:tensor<1x8xf32>) dimensions = [2]
    (%138: f32, %139: f32) {
      %140 = arith.addf %138, %139 : f32
      linalg.yield %140 : f32
    }
    %141 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 2.048000e+03 : f32
    %142 = tensor.splat %141 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %143 = tensor.empty() : tensor<1x8xf32>
    %144 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%137, %142 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%143 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb12(%145: f32, %146: f32, %147: f32):
      %148 = arith.divf %145, %146 : f32
      linalg.yield %148 : f32
    } -> tensor<1x8xf32>
    %149 = tensor.collapse_shape %144 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %150 = tensor.expand_shape %149 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %151 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
    %152 = tensor.splat %151 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %153 = tensor.empty() : tensor<1x8x1xf32>
    %154 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%150, %152 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%153 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb13(%155: f32, %156: f32, %157: f32):
      %158 = arith.addf %155, %156 : f32
      linalg.yield %158 : f32
    } -> tensor<1x8x1xf32>
    %159 = tensor.empty() : tensor<1x8x1xf32>
    %160 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%154 : tensor<1x8x1xf32>) outs(%159 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb14(%161: f32, %162: f32):
      %163 = math.rsqrt %161 : f32
      linalg.yield %163 : f32
    } -> tensor<1x8x1xf32>
    %164 = tensor.empty() : tensor<1x8x2048xf32>
    %165 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%52, %160 : tensor<1x8x2048xf32>, tensor<1x8x1xf32>) outs(%164 : tensor<1x8x2048xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb15(%166: f32, %167: f32, %168: f32):
      %169 = arith.mulf %166, %167 : f32
      linalg.yield %169 : f32
    } -> tensor<1x8x2048xf32>
    %170 = tensor.empty() : tensor<1x8x2048xf32>
    %171 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %165 : tensor<2048xf32>, tensor<1x8x2048xf32>) outs(%170 : tensor<1x8x2048xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb16(%172: f32, %173: f32, %174: f32):
      %175 = arith.mulf %172, %173 : f32
      linalg.yield %175 : f32
    } -> tensor<1x8x2048xf32>
    %176 = tensor.empty() : tensor<2048x2048xf32>
    %177 = linalg.transpose ins(%30:tensor<2048x2048xf32>) outs(%176:tensor<2048x2048xf32>) permutation = [1, 0]
    %178 = tensor.collapse_shape %171 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x2048xf32> into tensor<16384xf32>
    %179 = tensor.expand_shape %178 [[0 : i64, 1 : i64]] output_shape [8, 2048] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<8x2048xf32>
    %180 = tensor.empty() : tensor<8x2048xf32>
    %181 = arith.constant 0.000000e+00 : f32
    %182 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%181 : f32) outs(%180 : tensor<8x2048xf32>) -> tensor<8x2048xf32>
    %183 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%179, %177 : tensor<8x2048xf32>, tensor<2048x2048xf32>) outs(%182 : tensor<8x2048xf32>) -> tensor<8x2048xf32>
    %184 = tensor.collapse_shape %183 [[0 : i64, 1 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<8x2048xf32> into tensor<16384xf32>
    %185 = tensor.expand_shape %184 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 2048] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<1x8x2048xf32>
    %186 = tensor.collapse_shape %185 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x2048xf32> into tensor<16384xf32>
    %187 = tensor.expand_shape %186 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 64] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<1x8x32x64xf32>
    %188 = tensor.empty() : tensor<1x32x8x64xf32>
    %189 = linalg.transpose ins(%187:tensor<1x8x32x64xf32>) outs(%188:tensor<1x32x8x64xf32>) permutation = [0, 2, 1, 3]
    %190 = tensor.empty() : tensor<2048x256xf32>
    %191 = linalg.transpose ins(%27:tensor<256x2048xf32>) outs(%190:tensor<2048x256xf32>) permutation = [1, 0]
    %192 = tensor.collapse_shape %171 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x2048xf32> into tensor<16384xf32>
    %193 = tensor.expand_shape %192 [[0 : i64, 1 : i64]] output_shape [8, 2048] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<8x2048xf32>
    %194 = tensor.empty() : tensor<8x256xf32>
    %195 = arith.constant 0.000000e+00 : f32
    %196 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%195 : f32) outs(%194 : tensor<8x256xf32>) -> tensor<8x256xf32>
    %197 = linalg.matmul {prov.region_id = "matmul_2", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%193, %191 : tensor<8x2048xf32>, tensor<2048x256xf32>) outs(%196 : tensor<8x256xf32>) -> tensor<8x256xf32>
    %198 = tensor.collapse_shape %197 [[0 : i64, 1 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<8x256xf32> into tensor<2048xf32>
    %199 = tensor.expand_shape %198 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 256] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x8x256xf32>
    %200 = tensor.collapse_shape %199 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x256xf32> into tensor<2048xf32>
    %201 = tensor.expand_shape %200 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 64] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x8x4x64xf32>
    %202 = tensor.empty() : tensor<1x4x8x64xf32>
    %203 = linalg.transpose ins(%201:tensor<1x8x4x64xf32>) outs(%202:tensor<1x4x8x64xf32>) permutation = [0, 2, 1, 3]
    %204 = tensor.empty() : tensor<2048x256xf32>
    %205 = linalg.transpose ins(%32:tensor<256x2048xf32>) outs(%204:tensor<2048x256xf32>) permutation = [1, 0]
    %206 = tensor.collapse_shape %171 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x2048xf32> into tensor<16384xf32>
    %207 = tensor.expand_shape %206 [[0 : i64, 1 : i64]] output_shape [8, 2048] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<8x2048xf32>
    %208 = tensor.empty() : tensor<8x256xf32>
    %209 = arith.constant 0.000000e+00 : f32
    %210 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%209 : f32) outs(%208 : tensor<8x256xf32>) -> tensor<8x256xf32>
    %211 = linalg.matmul {prov.region_id = "matmul_3", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%207, %205 : tensor<8x2048xf32>, tensor<2048x256xf32>) outs(%210 : tensor<8x256xf32>) -> tensor<8x256xf32>
    %212 = tensor.collapse_shape %211 [[0 : i64, 1 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<8x256xf32> into tensor<2048xf32>
    %213 = tensor.expand_shape %212 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 256] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x8x256xf32>
    %214 = tensor.collapse_shape %213 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x256xf32> into tensor<2048xf32>
    %215 = tensor.expand_shape %214 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 64] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x8x4x64xf32>
    %216 = tensor.empty() : tensor<1x4x8x64xf32>
    %217 = linalg.transpose ins(%215:tensor<1x8x4x64xf32>) outs(%216:tensor<1x4x8x64xf32>) permutation = [0, 2, 1, 3]
    %218 = tensor.collapse_shape %111 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x64xf32> into tensor<512xf32>
    %219 = tensor.expand_shape %218 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 64] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x8x64xf32>
    %220 = tensor.collapse_shape %124 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x64xf32> into tensor<512xf32>
    %221 = tensor.expand_shape %220 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 64] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x8x64xf32>
    %222 = tensor.empty() : tensor<1x32x8x64xf32>
    %223 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%189, %219 : tensor<1x32x8x64xf32>, tensor<1x1x8x64xf32>) outs(%222 : tensor<1x32x8x64xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb17(%224: f32, %225: f32, %226: f32):
      %227 = arith.mulf %224, %225 : f32
      linalg.yield %227 : f32
    } -> tensor<1x32x8x64xf32>
    %228 = "tensor.extract_slice"(%189) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 32, 8, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x32xf32>
    %229 = "tensor.extract_slice"(%189) <{static_offsets = array<i64: 0, 0, 0, 32>, static_sizes = array<i64: 1, 32, 8, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x32xf32>
    %230 = tensor.empty() : tensor<1x32x8x32xf32>
    %231 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%229 : tensor<1x32x8x32xf32>) outs(%230 : tensor<1x32x8x32xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb18(%232: f32, %233: f32):
      %234 = arith.negf %232 : f32
      linalg.yield %234 : f32
    } -> tensor<1x32x8x32xf32>
    %235 = tensor.concat dim(3) %231, %228 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x32x8x32xf32>, tensor<1x32x8x32xf32>) -> tensor<1x32x8x64xf32>
    %236 = tensor.empty() : tensor<1x32x8x64xf32>
    %237 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%235, %221 : tensor<1x32x8x64xf32>, tensor<1x1x8x64xf32>) outs(%236 : tensor<1x32x8x64xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb19(%238: f32, %239: f32, %240: f32):
      %241 = arith.mulf %238, %239 : f32
      linalg.yield %241 : f32
    } -> tensor<1x32x8x64xf32>
    %242 = tensor.empty() : tensor<1x32x8x64xf32>
    %243 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%223, %237 : tensor<1x32x8x64xf32>, tensor<1x32x8x64xf32>) outs(%242 : tensor<1x32x8x64xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb20(%244: f32, %245: f32, %246: f32):
      %247 = arith.addf %244, %245 : f32
      linalg.yield %247 : f32
    } -> tensor<1x32x8x64xf32>
    %248 = tensor.empty() : tensor<1x4x8x64xf32>
    %249 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%203, %219 : tensor<1x4x8x64xf32>, tensor<1x1x8x64xf32>) outs(%248 : tensor<1x4x8x64xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb21(%250: f32, %251: f32, %252: f32):
      %253 = arith.mulf %250, %251 : f32
      linalg.yield %253 : f32
    } -> tensor<1x4x8x64xf32>
    %254 = "tensor.extract_slice"(%203) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x64xf32>) -> tensor<1x4x8x32xf32>
    %255 = "tensor.extract_slice"(%203) <{static_offsets = array<i64: 0, 0, 0, 32>, static_sizes = array<i64: 1, 4, 8, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x64xf32>) -> tensor<1x4x8x32xf32>
    %256 = tensor.empty() : tensor<1x4x8x32xf32>
    %257 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%255 : tensor<1x4x8x32xf32>) outs(%256 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb22(%258: f32, %259: f32):
      %260 = arith.negf %258 : f32
      linalg.yield %260 : f32
    } -> tensor<1x4x8x32xf32>
    %261 = tensor.concat dim(3) %257, %254 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) -> tensor<1x4x8x64xf32>
    %262 = tensor.empty() : tensor<1x4x8x64xf32>
    %263 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%261, %221 : tensor<1x4x8x64xf32>, tensor<1x1x8x64xf32>) outs(%262 : tensor<1x4x8x64xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb23(%264: f32, %265: f32, %266: f32):
      %267 = arith.mulf %264, %265 : f32
      linalg.yield %267 : f32
    } -> tensor<1x4x8x64xf32>
    %268 = tensor.empty() : tensor<1x4x8x64xf32>
    %269 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%249, %263 : tensor<1x4x8x64xf32>, tensor<1x4x8x64xf32>) outs(%268 : tensor<1x4x8x64xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb24(%270: f32, %271: f32, %272: f32):
      %273 = arith.addf %270, %271 : f32
      linalg.yield %273 : f32
    } -> tensor<1x4x8x64xf32>
    %274 = tensor.empty() : tensor<8xi64>
    %275 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%274 : tensor<8xi64>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb25(%276: i64):
      %277 = linalg.index 0 : index
      %278 = arith.index_cast %277 : index to i64
      %279 = arith.constant 1 : i64
      %280 = arith.muli %278, %279 : i64
      %281 = arith.constant 0 : i64
      %282 = arith.addi %281, %280 : i64
      linalg.yield %282 : i64
    } -> tensor<8xi64>
    %283 = tensor.empty() : tensor<8xi64>
    %284 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%59, %275 : tensor<i64>, tensor<8xi64>) outs(%283 : tensor<8xi64>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb26(%285: i64, %286: i64, %287: i64):
      %288 = arith.addi %285, %286 : i64
      linalg.yield %288 : i64
    } -> tensor<8xi64>
    %289 = "tensor.extract_slice"(%48) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 15, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x15x64xf32>) -> tensor<4x15x64xf32>
    %290 = func.call @aten_index_put_default(%289, %284, %269) {prov.region_id = "aten_index_put_default_0", prov.dispatch_id = "aten_index_put_default_0"} : (tensor<4x15x64xf32>, tensor<8xi64>, tensor<1x4x8x64xf32>) -> tensor<1x4x15x64xf32>
    %291 = "tensor.extract_slice"(%50) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 15, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x15x64xf32>) -> tensor<4x15x64xf32>
    %292 = func.call @aten_index_put_default(%291, %284, %217) {prov.region_id = "aten_index_put_default_1", prov.dispatch_id = "aten_index_put_default_1"} : (tensor<4x15x64xf32>, tensor<8xi64>, tensor<1x4x8x64xf32>) -> tensor<1x4x15x64xf32>
    %293 = tensor.collapse_shape %290 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x15x64xf32> into tensor<3840xf32>
    %294 = tensor.expand_shape %293 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 15, 64] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<3840xf32> into tensor<1x4x1x15x64xf32>
    %295 = tensor.empty() : tensor<1x4x8x15x64xf32>
    %296 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%294 : tensor<1x4x1x15x64xf32>) outs(%295 : tensor<1x4x8x15x64xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb27(%297: f32, %298: f32):
      linalg.yield %297 : f32
    } -> tensor<1x4x8x15x64xf32>
    %299 = tensor.collapse_shape %296 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x8x15x64xf32> into tensor<30720xf32>
    %300 = tensor.expand_shape %299 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 64] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<30720xf32> into tensor<1x32x15x64xf32>
    %301 = tensor.collapse_shape %292 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x15x64xf32> into tensor<3840xf32>
    %302 = tensor.expand_shape %301 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 15, 64] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<3840xf32> into tensor<1x4x1x15x64xf32>
    %303 = tensor.empty() : tensor<1x4x8x15x64xf32>
    %304 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%302 : tensor<1x4x1x15x64xf32>) outs(%303 : tensor<1x4x8x15x64xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb28(%305: f32, %306: f32):
      linalg.yield %305 : f32
    } -> tensor<1x4x8x15x64xf32>
    %307 = tensor.collapse_shape %304 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x8x15x64xf32> into tensor<30720xf32>
    %308 = tensor.expand_shape %307 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 64] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<30720xf32> into tensor<1x32x15x64xf32>
    %309 = tensor.empty() : tensor<1x32x64x15xf32>
    %310 = linalg.transpose ins(%300:tensor<1x32x15x64xf32>) outs(%309:tensor<1x32x64x15xf32>) permutation = [0, 1, 3, 2]
    %311 = tensor.empty() : tensor<1x32x8x64xf32>
    %312 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%243 : tensor<1x32x8x64xf32>) outs(%311 : tensor<1x32x8x64xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb29(%313: f32, %314: f32):
      linalg.yield %313 : f32
    } -> tensor<1x32x8x64xf32>
    %315 = tensor.collapse_shape %312 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x8x64xf32> into tensor<16384xf32>
    %316 = tensor.expand_shape %315 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 64] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<32x8x64xf32>
    %317 = tensor.empty() : tensor<1x32x64x15xf32>
    %318 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%310 : tensor<1x32x64x15xf32>) outs(%317 : tensor<1x32x64x15xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb30(%319: f32, %320: f32):
      linalg.yield %319 : f32
    } -> tensor<1x32x64x15xf32>
    %321 = tensor.collapse_shape %318 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x64x15xf32> into tensor<30720xf32>
    %322 = tensor.expand_shape %321 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 64, 15] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<30720xf32> into tensor<32x64x15xf32>
    %323 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %324 = tensor.splat %323 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<32x8x15xf32>
    %325 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%316, %322 : tensor<32x8x64xf32>, tensor<32x64x15xf32>) outs(%324 : tensor<32x8x15xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb31(%326: f32, %327: f32, %328: f32):
      %329 = arith.mulf %326, %327 : f32
      %330 = arith.addf %328, %329 : f32
      linalg.yield %330 : f32
    } -> tensor<32x8x15xf32>
    %331 = tensor.collapse_shape %325 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32x8x15xf32> into tensor<3840xf32>
    %332 = tensor.expand_shape %331 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 15] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3840xf32> into tensor<1x32x8x15xf32>
    %333 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 8.000000e+00 : f32
    %334 = tensor.splat %333 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x8x15xf32>
    %335 = tensor.empty() : tensor<1x32x8x15xf32>
    %336 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%332, %334 : tensor<1x32x8x15xf32>, tensor<1x32x8x15xf32>) outs(%335 : tensor<1x32x8x15xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb32(%337: f32, %338: f32, %339: f32):
      %340 = arith.divf %337, %338 : f32
      linalg.yield %340 : f32
    } -> tensor<1x32x8x15xf32>
    %341 = tensor.empty() : tensor<15xi64>
    %342 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%341 : tensor<15xi64>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb33(%343: i64):
      %344 = linalg.index 0 : index
      %345 = arith.index_cast %344 : index to i64
      %346 = arith.constant 1 : i64
      %347 = arith.muli %345, %346 : i64
      %348 = arith.constant 0 : i64
      %349 = arith.addi %348, %347 : i64
      linalg.yield %349 : i64
    } -> tensor<15xi64>
    %350 = tensor.empty() : tensor<8xi64>
    %351 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%350 : tensor<8xi64>) attrs =  {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb34(%352: i64):
      %353 = linalg.index 0 : index
      %354 = arith.index_cast %353 : index to i64
      %355 = arith.constant 1 : i64
      %356 = arith.muli %354, %355 : i64
      %357 = arith.constant 0 : i64
      %358 = arith.addi %357, %356 : i64
      linalg.yield %358 : i64
    } -> tensor<8xi64>
    %359 = tensor.empty() : tensor<8xi64>
    %360 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%59, %351 : tensor<i64>, tensor<8xi64>) outs(%359 : tensor<8xi64>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb35(%361: i64, %362: i64, %363: i64):
      %364 = arith.addi %361, %362 : i64
      linalg.yield %364 : i64
    } -> tensor<8xi64>
    %365 = tensor.expand_shape %360 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<8xi64> into tensor<8x1xi64>
    %366 = tensor.expand_shape %342 [[0 : i64, 1 : i64]] output_shape [1, 15] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<15xi64> into tensor<1x15xi64>
    %367 = tensor.empty() : tensor<8x15xi1>
    %368 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%366, %365 : tensor<1x15xi64>, tensor<8x1xi64>) outs(%367 : tensor<8x15xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
    ^bb36(%369: i64, %370: i64, %371: i1):
      %372 = arith.cmpi sle, %369, %370 : i64
      linalg.yield %372 : i1
    } -> tensor<8x15xi1>
    %373 = tensor.collapse_shape %368 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<8x15xi1> into tensor<120xi1>
    %374 = tensor.expand_shape %373 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 15] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<120xi1> into tensor<1x8x15xi1>
    %375 = tensor.collapse_shape %374 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x8x15xi1> into tensor<120xi1>
    %376 = tensor.expand_shape %375 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 15] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<120xi1> into tensor<1x1x8x15xi1>
    %377 = tensor.empty() : tensor<1x1x8x15xi1>
    %378 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%376 : tensor<1x1x8x15xi1>) outs(%377 : tensor<1x1x8x15xi1>) attrs =  {prov.region_id = "bitwise_0", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
    ^bb37(%379: i1, %380: i1):
      %381 = arith.constant true
      %382 = arith.xori %379, %381 : i1
      linalg.yield %382 : i1
    } -> tensor<1x1x8x15xi1>
    %383 = arith.constant {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %384 = tensor.splat %383 {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %385 = tensor.empty() : tensor<1x32x8x15xf32>
    %386 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%378, %384, %336 : tensor<1x1x8x15xi1>, tensor<f32>, tensor<1x32x8x15xf32>) outs(%385 : tensor<1x32x8x15xf32>) attrs =  {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32"} {
    ^bb38(%387: i1, %388: f32, %389: f32, %390: f32):
      %391 = arith.select %387, %388, %389 : f32
      linalg.yield %391 : f32
    } -> tensor<1x32x8x15xf32>
    %392 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %393 = tensor.splat %392 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32x8xf32>
    %394 = linalg.reduce ins(%386:tensor<1x32x8x15xf32>) outs(%393:tensor<1x32x8xf32>) dimensions = [3]
    (%395: f32, %396: f32) {
      %397 = arith.maximumf %395, %396 : f32
      linalg.yield %397 : f32
    }
    %398 = tensor.collapse_shape %394 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32x8xf32> into tensor<256xf32>
    %399 = tensor.expand_shape %398 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x32x8x1xf32>
    %400 = tensor.empty() : tensor<1x32x8x15xf32>
    %401 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%386, %399 : tensor<1x32x8x15xf32>, tensor<1x32x8x1xf32>) outs(%400 : tensor<1x32x8x15xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb39(%402: f32, %403: f32, %404: f32):
      %405 = arith.subf %402, %403 : f32
      linalg.yield %405 : f32
    } -> tensor<1x32x8x15xf32>
    %406 = tensor.empty() : tensor<1x32x8x15xf32>
    %407 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%401 : tensor<1x32x8x15xf32>) outs(%406 : tensor<1x32x8x15xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb40(%408: f32, %409: f32):
      %410 = math.exp %408 : f32
      linalg.yield %410 : f32
    } -> tensor<1x32x8x15xf32>
    %411 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %412 = tensor.splat %411 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32x8xf32>
    %413 = linalg.reduce ins(%407:tensor<1x32x8x15xf32>) outs(%412:tensor<1x32x8xf32>) dimensions = [3]
    (%414: f32, %415: f32) {
      %416 = arith.addf %414, %415 : f32
      linalg.yield %416 : f32
    }
    %417 = tensor.collapse_shape %413 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32x8xf32> into tensor<256xf32>
    %418 = tensor.expand_shape %417 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x32x8x1xf32>
    %419 = tensor.empty() : tensor<1x32x8x15xf32>
    %420 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%407, %418 : tensor<1x32x8x15xf32>, tensor<1x32x8x1xf32>) outs(%419 : tensor<1x32x8x15xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb41(%421: f32, %422: f32, %423: f32):
      %424 = arith.divf %421, %422 : f32
      linalg.yield %424 : f32
    } -> tensor<1x32x8x15xf32>
    %425 = tensor.empty() : tensor<1x32x8x15xf32>
    %426 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%420 : tensor<1x32x8x15xf32>) outs(%425 : tensor<1x32x8x15xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb42(%427: f32, %428: f32):
      linalg.yield %427 : f32
    } -> tensor<1x32x8x15xf32>
    %429 = tensor.collapse_shape %426 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x8x15xf32> into tensor<3840xf32>
    %430 = tensor.expand_shape %429 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 15] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3840xf32> into tensor<32x8x15xf32>
    %431 = tensor.empty() : tensor<1x32x15x64xf32>
    %432 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%308 : tensor<1x32x15x64xf32>) outs(%431 : tensor<1x32x15x64xf32>) attrs =  {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb43(%433: f32, %434: f32):
      linalg.yield %433 : f32
    } -> tensor<1x32x15x64xf32>
    %435 = tensor.collapse_shape %432 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x15x64xf32> into tensor<30720xf32>
    %436 = tensor.expand_shape %435 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 15, 64] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<30720xf32> into tensor<32x15x64xf32>
    %437 = arith.constant {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %438 = tensor.splat %437 {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<32x8x64xf32>
    %439 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%430, %436 : tensor<32x8x15xf32>, tensor<32x15x64xf32>) outs(%438 : tensor<32x8x64xf32>) attrs =  {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb44(%440: f32, %441: f32, %442: f32):
      %443 = arith.mulf %440, %441 : f32
      %444 = arith.addf %442, %443 : f32
      linalg.yield %444 : f32
    } -> tensor<32x8x64xf32>
    %445 = tensor.collapse_shape %439 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32x8x64xf32> into tensor<16384xf32>
    %446 = tensor.expand_shape %445 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 64] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<1x32x8x64xf32>
    %447 = tensor.empty() : tensor<1x8x32x64xf32>
    %448 = linalg.transpose ins(%446:tensor<1x32x8x64xf32>) outs(%447:tensor<1x8x32x64xf32>) permutation = [0, 2, 1, 3]
    %449 = tensor.collapse_shape %448 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x32x64xf32> into tensor<16384xf32>
    %450 = tensor.expand_shape %449 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 2048] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<1x8x2048xf32>
    %451 = tensor.empty() : tensor<2048x2048xf32>
    %452 = linalg.transpose ins(%28:tensor<2048x2048xf32>) outs(%451:tensor<2048x2048xf32>) permutation = [1, 0]
    %453 = tensor.collapse_shape %450 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x2048xf32> into tensor<16384xf32>
    %454 = tensor.expand_shape %453 [[0 : i64, 1 : i64]] output_shape [8, 2048] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<8x2048xf32>
    %455 = tensor.empty() : tensor<8x2048xf32>
    %456 = arith.constant 0.000000e+00 : f32
    %457 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%456 : f32) outs(%455 : tensor<8x2048xf32>) -> tensor<8x2048xf32>
    %458 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%454, %452 : tensor<8x2048xf32>, tensor<2048x2048xf32>) outs(%457 : tensor<8x2048xf32>) -> tensor<8x2048xf32>
    %459 = tensor.collapse_shape %458 [[0 : i64, 1 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<8x2048xf32> into tensor<16384xf32>
    %460 = tensor.expand_shape %459 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 2048] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<1x8x2048xf32>
    %461 = tensor.empty() : tensor<1x8x2048xf32>
    %462 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%52, %460 : tensor<1x8x2048xf32>, tensor<1x8x2048xf32>) outs(%461 : tensor<1x8x2048xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb45(%463: f32, %464: f32, %465: f32):
      %466 = arith.addf %463, %464 : f32
      linalg.yield %466 : f32
    } -> tensor<1x8x2048xf32>
    %467 = tensor.empty() : tensor<1x8x2048xf32>
    %468 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%462 : tensor<1x8x2048xf32>) outs(%467 : tensor<1x8x2048xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb46(%469: f32, %470: f32):
      %471 = arith.constant 2.000000e+00 : f32
      %472 = math.powf %469, %471 : f32
      linalg.yield %472 : f32
    } -> tensor<1x8x2048xf32>
    %473 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %474 = tensor.splat %473 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %475 = linalg.reduce ins(%468:tensor<1x8x2048xf32>) outs(%474:tensor<1x8xf32>) dimensions = [2]
    (%476: f32, %477: f32) {
      %478 = arith.addf %476, %477 : f32
      linalg.yield %478 : f32
    }
    %479 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 2.048000e+03 : f32
    %480 = tensor.splat %479 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %481 = tensor.empty() : tensor<1x8xf32>
    %482 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%475, %480 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%481 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb47(%483: f32, %484: f32, %485: f32):
      %486 = arith.divf %483, %484 : f32
      linalg.yield %486 : f32
    } -> tensor<1x8xf32>
    %487 = tensor.collapse_shape %482 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %488 = tensor.expand_shape %487 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %489 = arith.constant {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
    %490 = tensor.splat %489 {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %491 = tensor.empty() : tensor<1x8x1xf32>
    %492 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%488, %490 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%491 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb48(%493: f32, %494: f32, %495: f32):
      %496 = arith.addf %493, %494 : f32
      linalg.yield %496 : f32
    } -> tensor<1x8x1xf32>
    %497 = tensor.empty() : tensor<1x8x1xf32>
    %498 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%492 : tensor<1x8x1xf32>) outs(%497 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb49(%499: f32, %500: f32):
      %501 = math.rsqrt %499 : f32
      linalg.yield %501 : f32
    } -> tensor<1x8x1xf32>
    %502 = tensor.empty() : tensor<1x8x2048xf32>
    %503 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%462, %498 : tensor<1x8x2048xf32>, tensor<1x8x1xf32>) outs(%502 : tensor<1x8x2048xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb50(%504: f32, %505: f32, %506: f32):
      %507 = arith.mulf %504, %505 : f32
      linalg.yield %507 : f32
    } -> tensor<1x8x2048xf32>
    %508 = tensor.empty() : tensor<1x8x2048xf32>
    %509 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %503 : tensor<2048xf32>, tensor<1x8x2048xf32>) outs(%508 : tensor<1x8x2048xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb51(%510: f32, %511: f32, %512: f32):
      %513 = arith.mulf %510, %511 : f32
      linalg.yield %513 : f32
    } -> tensor<1x8x2048xf32>
    %514 = tensor.empty() : tensor<2048x5632xf32>
    %515 = linalg.transpose ins(%25:tensor<5632x2048xf32>) outs(%514:tensor<2048x5632xf32>) permutation = [1, 0]
    %516 = tensor.collapse_shape %509 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x2048xf32> into tensor<16384xf32>
    %517 = tensor.expand_shape %516 [[0 : i64, 1 : i64]] output_shape [8, 2048] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<8x2048xf32>
    %518 = tensor.empty() : tensor<8x5632xf32>
    %519 = arith.constant 0.000000e+00 : f32
    %520 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%519 : f32) outs(%518 : tensor<8x5632xf32>) -> tensor<8x5632xf32>
    %521 = linalg.matmul {prov.region_id = "matmul_7", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%517, %515 : tensor<8x2048xf32>, tensor<2048x5632xf32>) outs(%520 : tensor<8x5632xf32>) -> tensor<8x5632xf32>
    %522 = tensor.collapse_shape %521 [[0 : i64, 1 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<8x5632xf32> into tensor<45056xf32>
    %523 = tensor.expand_shape %522 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 5632] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<45056xf32> into tensor<1x8x5632xf32>
    %524 = tensor.empty() : tensor<1x8x5632xf32>
    %525 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%523 : tensor<1x8x5632xf32>) outs(%524 : tensor<1x8x5632xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32"} {
    ^bb52(%526: f32, %527: f32):
      %528 = arith.constant 1.000000e+00 : f32
      %529 = arith.negf %526 : f32
      %530 = math.exp %529 : f32
      %531 = arith.addf %528, %530 : f32
      %532 = arith.divf %528, %531 : f32
      linalg.yield %532 : f32
    } -> tensor<1x8x5632xf32>
    %533 = tensor.empty() : tensor<1x8x5632xf32>
    %534 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%523, %525 : tensor<1x8x5632xf32>, tensor<1x8x5632xf32>) outs(%533 : tensor<1x8x5632xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb53(%535: f32, %536: f32, %537: f32):
      %538 = arith.mulf %535, %536 : f32
      linalg.yield %538 : f32
    } -> tensor<1x8x5632xf32>
    %539 = tensor.empty() : tensor<2048x5632xf32>
    %540 = linalg.transpose ins(%31:tensor<5632x2048xf32>) outs(%539:tensor<2048x5632xf32>) permutation = [1, 0]
    %541 = tensor.collapse_shape %509 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x2048xf32> into tensor<16384xf32>
    %542 = tensor.expand_shape %541 [[0 : i64, 1 : i64]] output_shape [8, 2048] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<8x2048xf32>
    %543 = tensor.empty() : tensor<8x5632xf32>
    %544 = arith.constant 0.000000e+00 : f32
    %545 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%544 : f32) outs(%543 : tensor<8x5632xf32>) -> tensor<8x5632xf32>
    %546 = linalg.matmul {prov.region_id = "matmul_8", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%542, %540 : tensor<8x2048xf32>, tensor<2048x5632xf32>) outs(%545 : tensor<8x5632xf32>) -> tensor<8x5632xf32>
    %547 = tensor.collapse_shape %546 [[0 : i64, 1 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<8x5632xf32> into tensor<45056xf32>
    %548 = tensor.expand_shape %547 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 5632] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<45056xf32> into tensor<1x8x5632xf32>
    %549 = tensor.empty() : tensor<1x8x5632xf32>
    %550 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%534, %548 : tensor<1x8x5632xf32>, tensor<1x8x5632xf32>) outs(%549 : tensor<1x8x5632xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb54(%551: f32, %552: f32, %553: f32):
      %554 = arith.mulf %551, %552 : f32
      linalg.yield %554 : f32
    } -> tensor<1x8x5632xf32>
    %555 = tensor.empty() : tensor<5632x2048xf32>
    %556 = linalg.transpose ins(%24:tensor<2048x5632xf32>) outs(%555:tensor<5632x2048xf32>) permutation = [1, 0]
    %557 = tensor.collapse_shape %550 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x5632xf32> into tensor<45056xf32>
    %558 = tensor.expand_shape %557 [[0 : i64, 1 : i64]] output_shape [8, 5632] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<45056xf32> into tensor<8x5632xf32>
    %559 = tensor.empty() : tensor<8x2048xf32>
    %560 = arith.constant 0.000000e+00 : f32
    %561 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%560 : f32) outs(%559 : tensor<8x2048xf32>) -> tensor<8x2048xf32>
    %562 = linalg.matmul {prov.region_id = "matmul_9", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%558, %556 : tensor<8x5632xf32>, tensor<5632x2048xf32>) outs(%561 : tensor<8x2048xf32>) -> tensor<8x2048xf32>
    %563 = tensor.collapse_shape %562 [[0 : i64, 1 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<8x2048xf32> into tensor<16384xf32>
    %564 = tensor.expand_shape %563 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 2048] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<1x8x2048xf32>
    %565 = tensor.empty() : tensor<1x8x2048xf32>
    %566 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%462, %564 : tensor<1x8x2048xf32>, tensor<1x8x2048xf32>) outs(%565 : tensor<1x8x2048xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb55(%567: f32, %568: f32, %569: f32):
      %570 = arith.addf %567, %568 : f32
      linalg.yield %570 : f32
    } -> tensor<1x8x2048xf32>
    %571 = tensor.empty() : tensor<1x8x2048xf32>
    %572 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%566 : tensor<1x8x2048xf32>) outs(%571 : tensor<1x8x2048xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb56(%573: f32, %574: f32):
      %575 = arith.constant 2.000000e+00 : f32
      %576 = math.powf %573, %575 : f32
      linalg.yield %576 : f32
    } -> tensor<1x8x2048xf32>
    %577 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %578 = tensor.splat %577 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %579 = linalg.reduce ins(%572:tensor<1x8x2048xf32>) outs(%578:tensor<1x8xf32>) dimensions = [2]
    (%580: f32, %581: f32) {
      %582 = arith.addf %580, %581 : f32
      linalg.yield %582 : f32
    }
    %583 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 2.048000e+03 : f32
    %584 = tensor.splat %583 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %585 = tensor.empty() : tensor<1x8xf32>
    %586 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%579, %584 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%585 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb57(%587: f32, %588: f32, %589: f32):
      %590 = arith.divf %587, %588 : f32
      linalg.yield %590 : f32
    } -> tensor<1x8xf32>
    %591 = tensor.collapse_shape %586 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %592 = tensor.expand_shape %591 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %593 = arith.constant {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
    %594 = tensor.splat %593 {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %595 = tensor.empty() : tensor<1x8x1xf32>
    %596 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%592, %594 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%595 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb58(%597: f32, %598: f32, %599: f32):
      %600 = arith.addf %597, %598 : f32
      linalg.yield %600 : f32
    } -> tensor<1x8x1xf32>
    %601 = tensor.empty() : tensor<1x8x1xf32>
    %602 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%596 : tensor<1x8x1xf32>) outs(%601 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb59(%603: f32, %604: f32):
      %605 = math.rsqrt %603 : f32
      linalg.yield %605 : f32
    } -> tensor<1x8x1xf32>
    %606 = tensor.empty() : tensor<1x8x2048xf32>
    %607 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%566, %602 : tensor<1x8x2048xf32>, tensor<1x8x1xf32>) outs(%606 : tensor<1x8x2048xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb60(%608: f32, %609: f32, %610: f32):
      %611 = arith.mulf %608, %609 : f32
      linalg.yield %611 : f32
    } -> tensor<1x8x2048xf32>
    %612 = tensor.empty() : tensor<1x8x2048xf32>
    %613 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %607 : tensor<2048xf32>, tensor<1x8x2048xf32>) outs(%612 : tensor<1x8x2048xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb61(%614: f32, %615: f32, %616: f32):
      %617 = arith.mulf %614, %615 : f32
      linalg.yield %617 : f32
    } -> tensor<1x8x2048xf32>
    %618 = tensor.empty() : tensor<2048x2048xf32>
    %619 = linalg.transpose ins(%39:tensor<2048x2048xf32>) outs(%618:tensor<2048x2048xf32>) permutation = [1, 0]
    %620 = tensor.collapse_shape %613 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x2048xf32> into tensor<16384xf32>
    %621 = tensor.expand_shape %620 [[0 : i64, 1 : i64]] output_shape [8, 2048] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<8x2048xf32>
    %622 = tensor.empty() : tensor<8x2048xf32>
    %623 = arith.constant 0.000000e+00 : f32
    %624 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%623 : f32) outs(%622 : tensor<8x2048xf32>) -> tensor<8x2048xf32>
    %625 = linalg.matmul {prov.region_id = "matmul_10", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%621, %619 : tensor<8x2048xf32>, tensor<2048x2048xf32>) outs(%624 : tensor<8x2048xf32>) -> tensor<8x2048xf32>
    %626 = tensor.collapse_shape %625 [[0 : i64, 1 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<8x2048xf32> into tensor<16384xf32>
    %627 = tensor.expand_shape %626 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 2048] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<1x8x2048xf32>
    %628 = tensor.collapse_shape %627 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x2048xf32> into tensor<16384xf32>
    %629 = tensor.expand_shape %628 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 64] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<1x8x32x64xf32>
    %630 = tensor.empty() : tensor<1x32x8x64xf32>
    %631 = linalg.transpose ins(%629:tensor<1x8x32x64xf32>) outs(%630:tensor<1x32x8x64xf32>) permutation = [0, 2, 1, 3]
    %632 = tensor.empty() : tensor<2048x256xf32>
    %633 = linalg.transpose ins(%36:tensor<256x2048xf32>) outs(%632:tensor<2048x256xf32>) permutation = [1, 0]
    %634 = tensor.collapse_shape %613 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x2048xf32> into tensor<16384xf32>
    %635 = tensor.expand_shape %634 [[0 : i64, 1 : i64]] output_shape [8, 2048] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<8x2048xf32>
    %636 = tensor.empty() : tensor<8x256xf32>
    %637 = arith.constant 0.000000e+00 : f32
    %638 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%637 : f32) outs(%636 : tensor<8x256xf32>) -> tensor<8x256xf32>
    %639 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%635, %633 : tensor<8x2048xf32>, tensor<2048x256xf32>) outs(%638 : tensor<8x256xf32>) -> tensor<8x256xf32>
    %640 = tensor.collapse_shape %639 [[0 : i64, 1 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<8x256xf32> into tensor<2048xf32>
    %641 = tensor.expand_shape %640 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 256] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x8x256xf32>
    %642 = tensor.collapse_shape %641 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x256xf32> into tensor<2048xf32>
    %643 = tensor.expand_shape %642 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 64] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x8x4x64xf32>
    %644 = tensor.empty() : tensor<1x4x8x64xf32>
    %645 = linalg.transpose ins(%643:tensor<1x8x4x64xf32>) outs(%644:tensor<1x4x8x64xf32>) permutation = [0, 2, 1, 3]
    %646 = tensor.empty() : tensor<2048x256xf32>
    %647 = linalg.transpose ins(%41:tensor<256x2048xf32>) outs(%646:tensor<2048x256xf32>) permutation = [1, 0]
    %648 = tensor.collapse_shape %613 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x2048xf32> into tensor<16384xf32>
    %649 = tensor.expand_shape %648 [[0 : i64, 1 : i64]] output_shape [8, 2048] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<8x2048xf32>
    %650 = tensor.empty() : tensor<8x256xf32>
    %651 = arith.constant 0.000000e+00 : f32
    %652 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%651 : f32) outs(%650 : tensor<8x256xf32>) -> tensor<8x256xf32>
    %653 = linalg.matmul {prov.region_id = "matmul_12", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%649, %647 : tensor<8x2048xf32>, tensor<2048x256xf32>) outs(%652 : tensor<8x256xf32>) -> tensor<8x256xf32>
    %654 = tensor.collapse_shape %653 [[0 : i64, 1 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<8x256xf32> into tensor<2048xf32>
    %655 = tensor.expand_shape %654 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 256] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x8x256xf32>
    %656 = tensor.collapse_shape %655 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x256xf32> into tensor<2048xf32>
    %657 = tensor.expand_shape %656 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 64] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x8x4x64xf32>
    %658 = tensor.empty() : tensor<1x4x8x64xf32>
    %659 = linalg.transpose ins(%657:tensor<1x8x4x64xf32>) outs(%658:tensor<1x4x8x64xf32>) permutation = [0, 2, 1, 3]
    %660 = tensor.collapse_shape %111 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x64xf32> into tensor<512xf32>
    %661 = tensor.expand_shape %660 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 64] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x8x64xf32>
    %662 = tensor.collapse_shape %124 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x64xf32> into tensor<512xf32>
    %663 = tensor.expand_shape %662 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 64] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x8x64xf32>
    %664 = tensor.empty() : tensor<1x32x8x64xf32>
    %665 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%631, %661 : tensor<1x32x8x64xf32>, tensor<1x1x8x64xf32>) outs(%664 : tensor<1x32x8x64xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb62(%666: f32, %667: f32, %668: f32):
      %669 = arith.mulf %666, %667 : f32
      linalg.yield %669 : f32
    } -> tensor<1x32x8x64xf32>
    %670 = "tensor.extract_slice"(%631) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 32, 8, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x32xf32>
    %671 = "tensor.extract_slice"(%631) <{static_offsets = array<i64: 0, 0, 0, 32>, static_sizes = array<i64: 1, 32, 8, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x32x8x64xf32>) -> tensor<1x32x8x32xf32>
    %672 = tensor.empty() : tensor<1x32x8x32xf32>
    %673 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%671 : tensor<1x32x8x32xf32>) outs(%672 : tensor<1x32x8x32xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb63(%674: f32, %675: f32):
      %676 = arith.negf %674 : f32
      linalg.yield %676 : f32
    } -> tensor<1x32x8x32xf32>
    %677 = tensor.concat dim(3) %673, %670 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x32x8x32xf32>, tensor<1x32x8x32xf32>) -> tensor<1x32x8x64xf32>
    %678 = tensor.empty() : tensor<1x32x8x64xf32>
    %679 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%677, %663 : tensor<1x32x8x64xf32>, tensor<1x1x8x64xf32>) outs(%678 : tensor<1x32x8x64xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb64(%680: f32, %681: f32, %682: f32):
      %683 = arith.mulf %680, %681 : f32
      linalg.yield %683 : f32
    } -> tensor<1x32x8x64xf32>
    %684 = tensor.empty() : tensor<1x32x8x64xf32>
    %685 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%665, %679 : tensor<1x32x8x64xf32>, tensor<1x32x8x64xf32>) outs(%684 : tensor<1x32x8x64xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb65(%686: f32, %687: f32, %688: f32):
      %689 = arith.addf %686, %687 : f32
      linalg.yield %689 : f32
    } -> tensor<1x32x8x64xf32>
    %690 = tensor.empty() : tensor<1x4x8x64xf32>
    %691 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%645, %661 : tensor<1x4x8x64xf32>, tensor<1x1x8x64xf32>) outs(%690 : tensor<1x4x8x64xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb66(%692: f32, %693: f32, %694: f32):
      %695 = arith.mulf %692, %693 : f32
      linalg.yield %695 : f32
    } -> tensor<1x4x8x64xf32>
    %696 = "tensor.extract_slice"(%645) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x64xf32>) -> tensor<1x4x8x32xf32>
    %697 = "tensor.extract_slice"(%645) <{static_offsets = array<i64: 0, 0, 0, 32>, static_sizes = array<i64: 1, 4, 8, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x64xf32>) -> tensor<1x4x8x32xf32>
    %698 = tensor.empty() : tensor<1x4x8x32xf32>
    %699 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%697 : tensor<1x4x8x32xf32>) outs(%698 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb67(%700: f32, %701: f32):
      %702 = arith.negf %700 : f32
      linalg.yield %702 : f32
    } -> tensor<1x4x8x32xf32>
    %703 = tensor.concat dim(3) %699, %696 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) -> tensor<1x4x8x64xf32>
    %704 = tensor.empty() : tensor<1x4x8x64xf32>
    %705 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%703, %663 : tensor<1x4x8x64xf32>, tensor<1x1x8x64xf32>) outs(%704 : tensor<1x4x8x64xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb68(%706: f32, %707: f32, %708: f32):
      %709 = arith.mulf %706, %707 : f32
      linalg.yield %709 : f32
    } -> tensor<1x4x8x64xf32>
    %710 = tensor.empty() : tensor<1x4x8x64xf32>
    %711 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%691, %705 : tensor<1x4x8x64xf32>, tensor<1x4x8x64xf32>) outs(%710 : tensor<1x4x8x64xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb69(%712: f32, %713: f32, %714: f32):
      %715 = arith.addf %712, %713 : f32
      linalg.yield %715 : f32
    } -> tensor<1x4x8x64xf32>
    %716 = tensor.empty() : tensor<8xi64>
    %717 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%716 : tensor<8xi64>) attrs =  {prov.region_id = "iota_4", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb70(%718: i64):
      %719 = linalg.index 0 : index
      %720 = arith.index_cast %719 : index to i64
      %721 = arith.constant 1 : i64
      %722 = arith.muli %720, %721 : i64
      %723 = arith.constant 0 : i64
      %724 = arith.addi %723, %722 : i64
      linalg.yield %724 : i64
    } -> tensor<8xi64>
    %725 = tensor.empty() : tensor<8xi64>
    %726 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%59, %717 : tensor<i64>, tensor<8xi64>) outs(%725 : tensor<8xi64>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb71(%727: i64, %728: i64, %729: i64):
      %730 = arith.addi %727, %728 : i64
      linalg.yield %730 : i64
    } -> tensor<8xi64>
    %731 = "tensor.extract_slice"(%48) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 15, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x15x64xf32>) -> tensor<4x15x64xf32>
    %732 = func.call @aten_index_put_default(%731, %726, %711) {prov.region_id = "aten_index_put_default_2", prov.dispatch_id = "aten_index_put_default_2"} : (tensor<4x15x64xf32>, tensor<8xi64>, tensor<1x4x8x64xf32>) -> tensor<1x4x15x64xf32>
    %733 = "tensor.extract_slice"(%50) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 15, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x15x64xf32>) -> tensor<4x15x64xf32>
    %734 = func.call @aten_index_put_default(%733, %726, %659) {prov.region_id = "aten_index_put_default_3", prov.dispatch_id = "aten_index_put_default_3"} : (tensor<4x15x64xf32>, tensor<8xi64>, tensor<1x4x8x64xf32>) -> tensor<1x4x15x64xf32>
    %735 = tensor.collapse_shape %732 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x15x64xf32> into tensor<3840xf32>
    %736 = tensor.expand_shape %735 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 15, 64] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<3840xf32> into tensor<1x4x1x15x64xf32>
    %737 = tensor.empty() : tensor<1x4x8x15x64xf32>
    %738 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%736 : tensor<1x4x1x15x64xf32>) outs(%737 : tensor<1x4x8x15x64xf32>) attrs =  {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb72(%739: f32, %740: f32):
      linalg.yield %739 : f32
    } -> tensor<1x4x8x15x64xf32>
    %741 = tensor.collapse_shape %738 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x8x15x64xf32> into tensor<30720xf32>
    %742 = tensor.expand_shape %741 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 64] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<30720xf32> into tensor<1x32x15x64xf32>
    %743 = tensor.collapse_shape %734 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x15x64xf32> into tensor<3840xf32>
    %744 = tensor.expand_shape %743 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 15, 64] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<3840xf32> into tensor<1x4x1x15x64xf32>
    %745 = tensor.empty() : tensor<1x4x8x15x64xf32>
    %746 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%744 : tensor<1x4x1x15x64xf32>) outs(%745 : tensor<1x4x8x15x64xf32>) attrs =  {prov.region_id = "expand_10", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb73(%747: f32, %748: f32):
      linalg.yield %747 : f32
    } -> tensor<1x4x8x15x64xf32>
    %749 = tensor.collapse_shape %746 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x8x15x64xf32> into tensor<30720xf32>
    %750 = tensor.expand_shape %749 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 64] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<30720xf32> into tensor<1x32x15x64xf32>
    %751 = tensor.empty() : tensor<1x32x64x15xf32>
    %752 = linalg.transpose ins(%742:tensor<1x32x15x64xf32>) outs(%751:tensor<1x32x64x15xf32>) permutation = [0, 1, 3, 2]
    %753 = tensor.empty() : tensor<1x32x8x64xf32>
    %754 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%685 : tensor<1x32x8x64xf32>) outs(%753 : tensor<1x32x8x64xf32>) attrs =  {prov.region_id = "expand_11", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb74(%755: f32, %756: f32):
      linalg.yield %755 : f32
    } -> tensor<1x32x8x64xf32>
    %757 = tensor.collapse_shape %754 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x8x64xf32> into tensor<16384xf32>
    %758 = tensor.expand_shape %757 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 64] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<32x8x64xf32>
    %759 = tensor.empty() : tensor<1x32x64x15xf32>
    %760 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%752 : tensor<1x32x64x15xf32>) outs(%759 : tensor<1x32x64x15xf32>) attrs =  {prov.region_id = "expand_12", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb75(%761: f32, %762: f32):
      linalg.yield %761 : f32
    } -> tensor<1x32x64x15xf32>
    %763 = tensor.collapse_shape %760 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x64x15xf32> into tensor<30720xf32>
    %764 = tensor.expand_shape %763 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 64, 15] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<30720xf32> into tensor<32x64x15xf32>
    %765 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %766 = tensor.splat %765 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<32x8x15xf32>
    %767 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%758, %764 : tensor<32x8x64xf32>, tensor<32x64x15xf32>) outs(%766 : tensor<32x8x15xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb76(%768: f32, %769: f32, %770: f32):
      %771 = arith.mulf %768, %769 : f32
      %772 = arith.addf %770, %771 : f32
      linalg.yield %772 : f32
    } -> tensor<32x8x15xf32>
    %773 = tensor.collapse_shape %767 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32x8x15xf32> into tensor<3840xf32>
    %774 = tensor.expand_shape %773 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 15] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3840xf32> into tensor<1x32x8x15xf32>
    %775 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 8.000000e+00 : f32
    %776 = tensor.splat %775 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x8x15xf32>
    %777 = tensor.empty() : tensor<1x32x8x15xf32>
    %778 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%774, %776 : tensor<1x32x8x15xf32>, tensor<1x32x8x15xf32>) outs(%777 : tensor<1x32x8x15xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb77(%779: f32, %780: f32, %781: f32):
      %782 = arith.divf %779, %780 : f32
      linalg.yield %782 : f32
    } -> tensor<1x32x8x15xf32>
    %783 = tensor.empty() : tensor<15xi64>
    %784 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%783 : tensor<15xi64>) attrs =  {prov.region_id = "iota_5", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb78(%785: i64):
      %786 = linalg.index 0 : index
      %787 = arith.index_cast %786 : index to i64
      %788 = arith.constant 1 : i64
      %789 = arith.muli %787, %788 : i64
      %790 = arith.constant 0 : i64
      %791 = arith.addi %790, %789 : i64
      linalg.yield %791 : i64
    } -> tensor<15xi64>
    %792 = tensor.empty() : tensor<8xi64>
    %793 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%792 : tensor<8xi64>) attrs =  {prov.region_id = "iota_6", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb79(%794: i64):
      %795 = linalg.index 0 : index
      %796 = arith.index_cast %795 : index to i64
      %797 = arith.constant 1 : i64
      %798 = arith.muli %796, %797 : i64
      %799 = arith.constant 0 : i64
      %800 = arith.addi %799, %798 : i64
      linalg.yield %800 : i64
    } -> tensor<8xi64>
    %801 = tensor.empty() : tensor<8xi64>
    %802 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%59, %793 : tensor<i64>, tensor<8xi64>) outs(%801 : tensor<8xi64>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb80(%803: i64, %804: i64, %805: i64):
      %806 = arith.addi %803, %804 : i64
      linalg.yield %806 : i64
    } -> tensor<8xi64>
    %807 = tensor.expand_shape %802 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<8xi64> into tensor<8x1xi64>
    %808 = tensor.expand_shape %784 [[0 : i64, 1 : i64]] output_shape [1, 15] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<15xi64> into tensor<1x15xi64>
    %809 = tensor.empty() : tensor<8x15xi1>
    %810 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%808, %807 : tensor<1x15xi64>, tensor<8x1xi64>) outs(%809 : tensor<8x15xi1>) attrs =  {prov.region_id = "compare_1", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
    ^bb81(%811: i64, %812: i64, %813: i1):
      %814 = arith.cmpi sle, %811, %812 : i64
      linalg.yield %814 : i1
    } -> tensor<8x15xi1>
    %815 = tensor.collapse_shape %810 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<8x15xi1> into tensor<120xi1>
    %816 = tensor.expand_shape %815 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 15] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<120xi1> into tensor<1x8x15xi1>
    %817 = tensor.collapse_shape %816 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x8x15xi1> into tensor<120xi1>
    %818 = tensor.expand_shape %817 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 15] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<120xi1> into tensor<1x1x8x15xi1>
    %819 = tensor.empty() : tensor<1x1x8x15xi1>
    %820 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%818 : tensor<1x1x8x15xi1>) outs(%819 : tensor<1x1x8x15xi1>) attrs =  {prov.region_id = "bitwise_1", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
    ^bb82(%821: i1, %822: i1):
      %823 = arith.constant true
      %824 = arith.xori %821, %823 : i1
      linalg.yield %824 : i1
    } -> tensor<1x1x8x15xi1>
    %825 = arith.constant {prov.region_id = "fill_4", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %826 = tensor.splat %825 {prov.region_id = "fill_4", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %827 = tensor.empty() : tensor<1x32x8x15xf32>
    %828 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%820, %826, %778 : tensor<1x1x8x15xi1>, tensor<f32>, tensor<1x32x8x15xf32>) outs(%827 : tensor<1x32x8x15xf32>) attrs =  {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32"} {
    ^bb83(%829: i1, %830: f32, %831: f32, %832: f32):
      %833 = arith.select %829, %830, %831 : f32
      linalg.yield %833 : f32
    } -> tensor<1x32x8x15xf32>
    %834 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %835 = tensor.splat %834 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32x8xf32>
    %836 = linalg.reduce ins(%828:tensor<1x32x8x15xf32>) outs(%835:tensor<1x32x8xf32>) dimensions = [3]
    (%837: f32, %838: f32) {
      %839 = arith.maximumf %837, %838 : f32
      linalg.yield %839 : f32
    }
    %840 = tensor.collapse_shape %836 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32x8xf32> into tensor<256xf32>
    %841 = tensor.expand_shape %840 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x32x8x1xf32>
    %842 = tensor.empty() : tensor<1x32x8x15xf32>
    %843 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%828, %841 : tensor<1x32x8x15xf32>, tensor<1x32x8x1xf32>) outs(%842 : tensor<1x32x8x15xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb84(%844: f32, %845: f32, %846: f32):
      %847 = arith.subf %844, %845 : f32
      linalg.yield %847 : f32
    } -> tensor<1x32x8x15xf32>
    %848 = tensor.empty() : tensor<1x32x8x15xf32>
    %849 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%843 : tensor<1x32x8x15xf32>) outs(%848 : tensor<1x32x8x15xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb85(%850: f32, %851: f32):
      %852 = math.exp %850 : f32
      linalg.yield %852 : f32
    } -> tensor<1x32x8x15xf32>
    %853 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %854 = tensor.splat %853 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32x8xf32>
    %855 = linalg.reduce ins(%849:tensor<1x32x8x15xf32>) outs(%854:tensor<1x32x8xf32>) dimensions = [3]
    (%856: f32, %857: f32) {
      %858 = arith.addf %856, %857 : f32
      linalg.yield %858 : f32
    }
    %859 = tensor.collapse_shape %855 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32x8xf32> into tensor<256xf32>
    %860 = tensor.expand_shape %859 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x32x8x1xf32>
    %861 = tensor.empty() : tensor<1x32x8x15xf32>
    %862 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%849, %860 : tensor<1x32x8x15xf32>, tensor<1x32x8x1xf32>) outs(%861 : tensor<1x32x8x15xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb86(%863: f32, %864: f32, %865: f32):
      %866 = arith.divf %863, %864 : f32
      linalg.yield %866 : f32
    } -> tensor<1x32x8x15xf32>
    %867 = tensor.empty() : tensor<1x32x8x15xf32>
    %868 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%862 : tensor<1x32x8x15xf32>) outs(%867 : tensor<1x32x8x15xf32>) attrs =  {prov.region_id = "expand_13", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb87(%869: f32, %870: f32):
      linalg.yield %869 : f32
    } -> tensor<1x32x8x15xf32>
    %871 = tensor.collapse_shape %868 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x8x15xf32> into tensor<3840xf32>
    %872 = tensor.expand_shape %871 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 8, 15] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3840xf32> into tensor<32x8x15xf32>
    %873 = tensor.empty() : tensor<1x32x15x64xf32>
    %874 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%750 : tensor<1x32x15x64xf32>) outs(%873 : tensor<1x32x15x64xf32>) attrs =  {prov.region_id = "expand_14", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb88(%875: f32, %876: f32):
      linalg.yield %875 : f32
    } -> tensor<1x32x15x64xf32>
    %877 = tensor.collapse_shape %874 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x15x64xf32> into tensor<30720xf32>
    %878 = tensor.expand_shape %877 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 15, 64] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<30720xf32> into tensor<32x15x64xf32>
    %879 = arith.constant {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %880 = tensor.splat %879 {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<32x8x64xf32>
    %881 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%872, %878 : tensor<32x8x15xf32>, tensor<32x15x64xf32>) outs(%880 : tensor<32x8x64xf32>) attrs =  {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb89(%882: f32, %883: f32, %884: f32):
      %885 = arith.mulf %882, %883 : f32
      %886 = arith.addf %884, %885 : f32
      linalg.yield %886 : f32
    } -> tensor<32x8x64xf32>
    %887 = tensor.collapse_shape %881 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32x8x64xf32> into tensor<16384xf32>
    %888 = tensor.expand_shape %887 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 64] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<1x32x8x64xf32>
    %889 = tensor.empty() : tensor<1x8x32x64xf32>
    %890 = linalg.transpose ins(%888:tensor<1x32x8x64xf32>) outs(%889:tensor<1x8x32x64xf32>) permutation = [0, 2, 1, 3]
    %891 = tensor.collapse_shape %890 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x32x64xf32> into tensor<16384xf32>
    %892 = tensor.expand_shape %891 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 2048] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<1x8x2048xf32>
    %893 = tensor.empty() : tensor<2048x2048xf32>
    %894 = linalg.transpose ins(%37:tensor<2048x2048xf32>) outs(%893:tensor<2048x2048xf32>) permutation = [1, 0]
    %895 = tensor.collapse_shape %892 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x2048xf32> into tensor<16384xf32>
    %896 = tensor.expand_shape %895 [[0 : i64, 1 : i64]] output_shape [8, 2048] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<8x2048xf32>
    %897 = tensor.empty() : tensor<8x2048xf32>
    %898 = arith.constant 0.000000e+00 : f32
    %899 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%898 : f32) outs(%897 : tensor<8x2048xf32>) -> tensor<8x2048xf32>
    %900 = linalg.matmul {prov.region_id = "matmul_15", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%896, %894 : tensor<8x2048xf32>, tensor<2048x2048xf32>) outs(%899 : tensor<8x2048xf32>) -> tensor<8x2048xf32>
    %901 = tensor.collapse_shape %900 [[0 : i64, 1 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<8x2048xf32> into tensor<16384xf32>
    %902 = tensor.expand_shape %901 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 2048] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<1x8x2048xf32>
    %903 = tensor.empty() : tensor<1x8x2048xf32>
    %904 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%566, %902 : tensor<1x8x2048xf32>, tensor<1x8x2048xf32>) outs(%903 : tensor<1x8x2048xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb90(%905: f32, %906: f32, %907: f32):
      %908 = arith.addf %905, %906 : f32
      linalg.yield %908 : f32
    } -> tensor<1x8x2048xf32>
    %909 = tensor.empty() : tensor<1x8x2048xf32>
    %910 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%904 : tensor<1x8x2048xf32>) outs(%909 : tensor<1x8x2048xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb91(%911: f32, %912: f32):
      %913 = arith.constant 2.000000e+00 : f32
      %914 = math.powf %911, %913 : f32
      linalg.yield %914 : f32
    } -> tensor<1x8x2048xf32>
    %915 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %916 = tensor.splat %915 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %917 = linalg.reduce ins(%910:tensor<1x8x2048xf32>) outs(%916:tensor<1x8xf32>) dimensions = [2]
    (%918: f32, %919: f32) {
      %920 = arith.addf %918, %919 : f32
      linalg.yield %920 : f32
    }
    %921 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 2.048000e+03 : f32
    %922 = tensor.splat %921 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %923 = tensor.empty() : tensor<1x8xf32>
    %924 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%917, %922 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%923 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb92(%925: f32, %926: f32, %927: f32):
      %928 = arith.divf %925, %926 : f32
      linalg.yield %928 : f32
    } -> tensor<1x8xf32>
    %929 = tensor.collapse_shape %924 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %930 = tensor.expand_shape %929 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %931 = arith.constant {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
    %932 = tensor.splat %931 {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %933 = tensor.empty() : tensor<1x8x1xf32>
    %934 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%930, %932 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%933 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb93(%935: f32, %936: f32, %937: f32):
      %938 = arith.addf %935, %936 : f32
      linalg.yield %938 : f32
    } -> tensor<1x8x1xf32>
    %939 = tensor.empty() : tensor<1x8x1xf32>
    %940 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%934 : tensor<1x8x1xf32>) outs(%939 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb94(%941: f32, %942: f32):
      %943 = math.rsqrt %941 : f32
      linalg.yield %943 : f32
    } -> tensor<1x8x1xf32>
    %944 = tensor.empty() : tensor<1x8x2048xf32>
    %945 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%904, %940 : tensor<1x8x2048xf32>, tensor<1x8x1xf32>) outs(%944 : tensor<1x8x2048xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb95(%946: f32, %947: f32, %948: f32):
      %949 = arith.mulf %946, %947 : f32
      linalg.yield %949 : f32
    } -> tensor<1x8x2048xf32>
    %950 = tensor.empty() : tensor<1x8x2048xf32>
    %951 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %945 : tensor<2048xf32>, tensor<1x8x2048xf32>) outs(%950 : tensor<1x8x2048xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb96(%952: f32, %953: f32, %954: f32):
      %955 = arith.mulf %952, %953 : f32
      linalg.yield %955 : f32
    } -> tensor<1x8x2048xf32>
    %956 = tensor.empty() : tensor<2048x5632xf32>
    %957 = linalg.transpose ins(%34:tensor<5632x2048xf32>) outs(%956:tensor<2048x5632xf32>) permutation = [1, 0]
    %958 = tensor.collapse_shape %951 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x2048xf32> into tensor<16384xf32>
    %959 = tensor.expand_shape %958 [[0 : i64, 1 : i64]] output_shape [8, 2048] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<8x2048xf32>
    %960 = tensor.empty() : tensor<8x5632xf32>
    %961 = arith.constant 0.000000e+00 : f32
    %962 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%961 : f32) outs(%960 : tensor<8x5632xf32>) -> tensor<8x5632xf32>
    %963 = linalg.matmul {prov.region_id = "matmul_16", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%959, %957 : tensor<8x2048xf32>, tensor<2048x5632xf32>) outs(%962 : tensor<8x5632xf32>) -> tensor<8x5632xf32>
    %964 = tensor.collapse_shape %963 [[0 : i64, 1 : i64]] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<8x5632xf32> into tensor<45056xf32>
    %965 = tensor.expand_shape %964 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 5632] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<45056xf32> into tensor<1x8x5632xf32>
    %966 = tensor.empty() : tensor<1x8x5632xf32>
    %967 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%965 : tensor<1x8x5632xf32>) outs(%966 : tensor<1x8x5632xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32"} {
    ^bb97(%968: f32, %969: f32):
      %970 = arith.constant 1.000000e+00 : f32
      %971 = arith.negf %968 : f32
      %972 = math.exp %971 : f32
      %973 = arith.addf %970, %972 : f32
      %974 = arith.divf %970, %973 : f32
      linalg.yield %974 : f32
    } -> tensor<1x8x5632xf32>
    %975 = tensor.empty() : tensor<1x8x5632xf32>
    %976 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%965, %967 : tensor<1x8x5632xf32>, tensor<1x8x5632xf32>) outs(%975 : tensor<1x8x5632xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb98(%977: f32, %978: f32, %979: f32):
      %980 = arith.mulf %977, %978 : f32
      linalg.yield %980 : f32
    } -> tensor<1x8x5632xf32>
    %981 = tensor.empty() : tensor<2048x5632xf32>
    %982 = linalg.transpose ins(%40:tensor<5632x2048xf32>) outs(%981:tensor<2048x5632xf32>) permutation = [1, 0]
    %983 = tensor.collapse_shape %951 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x2048xf32> into tensor<16384xf32>
    %984 = tensor.expand_shape %983 [[0 : i64, 1 : i64]] output_shape [8, 2048] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<8x2048xf32>
    %985 = tensor.empty() : tensor<8x5632xf32>
    %986 = arith.constant 0.000000e+00 : f32
    %987 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%986 : f32) outs(%985 : tensor<8x5632xf32>) -> tensor<8x5632xf32>
    %988 = linalg.matmul {prov.region_id = "matmul_17", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%984, %982 : tensor<8x2048xf32>, tensor<2048x5632xf32>) outs(%987 : tensor<8x5632xf32>) -> tensor<8x5632xf32>
    %989 = tensor.collapse_shape %988 [[0 : i64, 1 : i64]] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<8x5632xf32> into tensor<45056xf32>
    %990 = tensor.expand_shape %989 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 5632] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<45056xf32> into tensor<1x8x5632xf32>
    %991 = tensor.empty() : tensor<1x8x5632xf32>
    %992 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%976, %990 : tensor<1x8x5632xf32>, tensor<1x8x5632xf32>) outs(%991 : tensor<1x8x5632xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb99(%993: f32, %994: f32, %995: f32):
      %996 = arith.mulf %993, %994 : f32
      linalg.yield %996 : f32
    } -> tensor<1x8x5632xf32>
    %997 = tensor.empty() : tensor<5632x2048xf32>
    %998 = linalg.transpose ins(%33:tensor<2048x5632xf32>) outs(%997:tensor<5632x2048xf32>) permutation = [1, 0]
    %999 = tensor.collapse_shape %992 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x5632xf32> into tensor<45056xf32>
    %1000 = tensor.expand_shape %999 [[0 : i64, 1 : i64]] output_shape [8, 5632] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<45056xf32> into tensor<8x5632xf32>
    %1001 = tensor.empty() : tensor<8x2048xf32>
    %1002 = arith.constant 0.000000e+00 : f32
    %1003 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1002 : f32) outs(%1001 : tensor<8x2048xf32>) -> tensor<8x2048xf32>
    %1004 = linalg.matmul {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%1000, %998 : tensor<8x5632xf32>, tensor<5632x2048xf32>) outs(%1003 : tensor<8x2048xf32>) -> tensor<8x2048xf32>
    %1005 = tensor.collapse_shape %1004 [[0 : i64, 1 : i64]] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<8x2048xf32> into tensor<16384xf32>
    %1006 = tensor.expand_shape %1005 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 2048] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16384xf32> into tensor<1x8x2048xf32>
    %1007 = tensor.empty() : tensor<1x8x2048xf32>
    %1008 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%904, %1006 : tensor<1x8x2048xf32>, tensor<1x8x2048xf32>) outs(%1007 : tensor<1x8x2048xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb100(%1009: f32, %1010: f32, %1011: f32):
      %1012 = arith.addf %1009, %1010 : f32
      linalg.yield %1012 : f32
    } -> tensor<1x8x2048xf32>
    %1013 = tensor.concat dim(0) %290, %732 {prov.region_id = "cat_5", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x15x64xf32>, tensor<1x4x15x64xf32>) -> tensor<2x4x15x64xf32>
    %1014 = tensor.collapse_shape %1013 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2x4x15x64xf32> into tensor<7680xf32>
    %1015 = tensor.expand_shape %1014 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [2, 1, 4, 15, 64] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<7680xf32> into tensor<2x1x4x15x64xf32>
    %1016 = tensor.concat dim(0) %292, %734 {prov.region_id = "cat_6", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x15x64xf32>, tensor<1x4x15x64xf32>) -> tensor<2x4x15x64xf32>
    %1017 = tensor.collapse_shape %1016 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2x4x15x64xf32> into tensor<7680xf32>
    %1018 = tensor.expand_shape %1017 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [2, 1, 4, 15, 64] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<7680xf32> into tensor<2x1x4x15x64xf32>
    %1019 = "tensor.extract_slice"(%1008) <{static_offsets = array<i64: 0, 7, 0>, static_sizes = array<i64: 1, 1, 2048>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x2048xf32>) -> tensor<1x1x2048xf32>
    %1020 = tensor.empty() : tensor<1x1x2048xf32>
    %1021 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1019 : tensor<1x1x2048xf32>) outs(%1020 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb101(%1022: f32, %1023: f32):
      %1024 = arith.constant 2.000000e+00 : f32
      %1025 = math.powf %1022, %1024 : f32
      linalg.yield %1025 : f32
    } -> tensor<1x1x2048xf32>
    %1026 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %1027 = tensor.splat %1026 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
    %1028 = linalg.reduce ins(%1021:tensor<1x1x2048xf32>) outs(%1027:tensor<1x1xf32>) dimensions = [2]
    (%1029: f32, %1030: f32) {
      %1031 = arith.addf %1029, %1030 : f32
      linalg.yield %1031 : f32
    }
    %1032 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 2.048000e+03 : f32
    %1033 = tensor.splat %1032 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
    %1034 = tensor.empty() : tensor<1x1xf32>
    %1035 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1028, %1033 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%1034 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb102(%1036: f32, %1037: f32, %1038: f32):
      %1039 = arith.divf %1036, %1037 : f32
      linalg.yield %1039 : f32
    } -> tensor<1x1xf32>
    %1040 = tensor.collapse_shape %1035 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
    %1041 = tensor.expand_shape %1040 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
    %1042 = arith.constant {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
    %1043 = tensor.splat %1042 {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
    %1044 = tensor.empty() : tensor<1x1x1xf32>
    %1045 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1041, %1043 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%1044 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb103(%1046: f32, %1047: f32, %1048: f32):
      %1049 = arith.addf %1046, %1047 : f32
      linalg.yield %1049 : f32
    } -> tensor<1x1x1xf32>
    %1050 = tensor.empty() : tensor<1x1x1xf32>
    %1051 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1045 : tensor<1x1x1xf32>) outs(%1050 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb104(%1052: f32, %1053: f32):
      %1054 = math.rsqrt %1052 : f32
      linalg.yield %1054 : f32
    } -> tensor<1x1x1xf32>
    %1055 = tensor.empty() : tensor<1x1x2048xf32>
    %1056 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1019, %1051 : tensor<1x1x2048xf32>, tensor<1x1x1xf32>) outs(%1055 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb105(%1057: f32, %1058: f32, %1059: f32):
      %1060 = arith.mulf %1057, %1058 : f32
      linalg.yield %1060 : f32
    } -> tensor<1x1x2048xf32>
    %1061 = tensor.empty() : tensor<1x1x2048xf32>
    %1062 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %1056 : tensor<2048xf32>, tensor<1x1x2048xf32>) outs(%1061 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb106(%1063: f32, %1064: f32, %1065: f32):
      %1066 = arith.mulf %1063, %1064 : f32
      linalg.yield %1066 : f32
    } -> tensor<1x1x2048xf32>
    %1067 = tensor.empty() : tensor<2048x32000xf32>
    %1068 = linalg.transpose ins(%42:tensor<32000x2048xf32>) outs(%1067:tensor<2048x32000xf32>) permutation = [1, 0]
    %1069 = tensor.collapse_shape %1062 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x2048xf32> into tensor<2048xf32>
    %1070 = tensor.expand_shape %1069 [[0 : i64, 1 : i64]] output_shape [1, 2048] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x2048xf32>
    %1071 = tensor.empty() : tensor<1x32000xf32>
    %1072 = arith.constant 0.000000e+00 : f32
    %1073 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1072 : f32) outs(%1071 : tensor<1x32000xf32>) -> tensor<1x32000xf32>
    %1074 = linalg.matmul {prov.region_id = "matmul_19", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%1070, %1068 : tensor<1x2048xf32>, tensor<2048x32000xf32>) outs(%1073 : tensor<1x32000xf32>) -> tensor<1x32000xf32>
    %1075 = tensor.collapse_shape %1074 [[0 : i64, 1 : i64]] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32000xf32> into tensor<32000xf32>
    %1076 = tensor.expand_shape %1075 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32000] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32000xf32> into tensor<1x1x32000xf32>
    %1077 = "tensor.extract_slice"(%1076) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 32000>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<1x1x32000xf32>) -> tensor<32000xf32>
    %1078 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0xff800000 : f32
    %1079 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0 : i64
    %1080 = tensor.splat %1078 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<f32>
    %1081 = tensor.splat %1079 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<i64>
    %1082, %1083 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%1077 : tensor<32000xf32>) outs(%1080, %1081 : tensor<f32>, tensor<i64>) attrs =  {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} {
    ^bb107(%1084: f32, %1085: f32, %1086: i64):
      %1087 = linalg.index 0 : index
      %1088 = arith.index_cast %1087 : index to i64
      %1089 = arith.cmpf ogt, %1084, %1085 : f32
      %1090 = arith.select %1089, %1084, %1085 : f32
      %1091 = arith.select %1089, %1088, %1086 : i64
      linalg.yield %1090, %1091 : f32, i64
    } -> (tensor<f32>, tensor<i64>)
    %1092 = tensor.extract %1082[] : tensor<f32>
    %1093 = tensor.from_elements %1092 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xf32>
    %1094 = tensor.extract %1083[] : tensor<i64>
    %1095 = tensor.from_elements %1094 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xi64>
    %1096 = arith.constant {prov.region_id = "fill_5", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "int64"} 0 : i64
    %1097 = tensor.splat %1096 {prov.region_id = "fill_5", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "int64"} : tensor<i64>
    %1098 = arith.constant {prov.region_id = "fill_6", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "int64"} 0 : i64
    %1099 = tensor.splat %1098 {prov.region_id = "fill_6", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "int64"} : tensor<1x7xi64>
    %1100 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 0 : index
    %1101 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 7 : index
    %1102 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 1 : index
    %1103, %1104, %1105, %1106, %1107 = scf.for %1108 = %1100 to %1101 step %1102 iter_args(%1109 = %1097, %1110 = %1095, %1111 = %1099, %1112 = %1015, %1113 = %1018) -> (tensor<i64>, tensor<1xi64>, tensor<1x7xi64>, tensor<2x1x4x15x64xf32>, tensor<2x1x4x15x64xf32>) {
      %1114 = tensor.extract %1109[] : tensor<i64>
      %1115 = tensor.from_elements %1114 {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1xi64>
      %1116 = func.call @aten_index_put_default_wl0(%1111, %1115, %1110) {prov.region_id = "aten_index_put_default_0", prov.dispatch_id = "aten_index_put_default_0"} : (tensor<1x7xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x7xi64>
      %1117 = tensor.empty() : tensor<i64>
      %1118 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%43, %1109 : tensor<i64>, tensor<i64>) outs(%1117 : tensor<i64>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb108(%1119: i64, %1120: i64, %1121: i64):
        %1122 = arith.addi %1119, %1120 : i64
        linalg.yield %1122 : i64
      } -> tensor<i64>
      %1123 = tensor.empty() : tensor<1x1x2048xf32>
      %1124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1110 : tensor<1xi64>) outs(%1123 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32"} {
      ^bb109(%1125: i64, %1126: f32):
        %1127 = arith.index_cast %1125 : i64 to index
        %1128 = linalg.index 2 : index
        %1129 = tensor.extract %22[%1127, %1128] : tensor<32000x2048xf32>
        linalg.yield %1129 : f32
      } -> tensor<1x1x2048xf32>
      %1130 = tensor.extract %1118[] : tensor<i64>
      %1131 = tensor.from_elements %1130 {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int64"} : tensor<1x1xi64>
      %1132 = tensor.expand_shape %0 [[0 : i64, 1 : i64]] output_shape [1, 32] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32xf32>
      %1133 = tensor.collapse_shape %1132 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
      %1134 = tensor.expand_shape %1133 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32x1xf32>
      %1135 = tensor.empty() : tensor<1x32x1xf32>
      %1136 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1134 : tensor<1x32x1xf32>) outs(%1135 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb110(%1137: f32, %1138: f32):
        linalg.yield %1137 : f32
      } -> tensor<1x32x1xf32>
      %1139 = tensor.collapse_shape %1131 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1x1xi64> into tensor<1xi64>
      %1140 = tensor.expand_shape %1139 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1xi64> into tensor<1x1x1xi64>
      %1141 = tensor.empty() : tensor<1x1x1xf32>
      %1142 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1140 : tensor<1x1x1xi64>) outs(%1141 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
      ^bb111(%1143: i64, %1144: f32):
        %1145 = arith.sitofp %1143 : i64 to f32
        linalg.yield %1145 : f32
      } -> tensor<1x1x1xf32>
      %1146 = tensor.empty() : tensor<1x32x1xf32>
      %1147 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1136 : tensor<1x32x1xf32>) outs(%1146 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb112(%1148: f32, %1149: f32):
        linalg.yield %1148 : f32
      } -> tensor<1x32x1xf32>
      %1150 = tensor.empty() : tensor<1x1x1xf32>
      %1151 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1142 : tensor<1x1x1xf32>) outs(%1150 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb113(%1152: f32, %1153: f32):
        linalg.yield %1152 : f32
      } -> tensor<1x1x1xf32>
      %1154 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1155 = tensor.splat %1154 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<1x32x1xf32>
      %1156 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1147, %1151 : tensor<1x32x1xf32>, tensor<1x1x1xf32>) outs(%1155 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
      ^bb114(%1157: f32, %1158: f32, %1159: f32):
        %1160 = arith.mulf %1157, %1158 : f32
        %1161 = arith.addf %1159, %1160 : f32
        linalg.yield %1161 : f32
      } -> tensor<1x32x1xf32>
      %1162 = tensor.empty() : tensor<1x1x32xf32>
      %1163 = linalg.transpose ins(%1156:tensor<1x32x1xf32>) outs(%1162:tensor<1x1x32xf32>) permutation = [0, 2, 1]
      %1164 = tensor.concat dim(2) %1163, %1163 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x1x32xf32>, tensor<1x1x32xf32>) -> tensor<1x1x64xf32>
      %1165 = tensor.empty() : tensor<1x1x64xf32>
      %1166 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1164 : tensor<1x1x64xf32>) outs(%1165 : tensor<1x1x64xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32"} {
      ^bb115(%1167: f32, %1168: f32):
        %1169 = math.cos %1167 : f32
        linalg.yield %1169 : f32
      } -> tensor<1x1x64xf32>
      %1170 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
      %1171 = tensor.splat %1170 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x64xf32>
      %1172 = tensor.empty() : tensor<1x1x64xf32>
      %1173 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1166, %1171 : tensor<1x1x64xf32>, tensor<1x1x64xf32>) outs(%1172 : tensor<1x1x64xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb116(%1174: f32, %1175: f32, %1176: f32):
        %1177 = arith.mulf %1174, %1175 : f32
        linalg.yield %1177 : f32
      } -> tensor<1x1x64xf32>
      %1178 = tensor.empty() : tensor<1x1x64xf32>
      %1179 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1164 : tensor<1x1x64xf32>) outs(%1178 : tensor<1x1x64xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32"} {
      ^bb117(%1180: f32, %1181: f32):
        %1182 = math.sin %1180 : f32
        linalg.yield %1182 : f32
      } -> tensor<1x1x64xf32>
      %1183 = arith.constant {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
      %1184 = tensor.splat %1183 {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x64xf32>
      %1185 = tensor.empty() : tensor<1x1x64xf32>
      %1186 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1179, %1184 : tensor<1x1x64xf32>, tensor<1x1x64xf32>) outs(%1185 : tensor<1x1x64xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb118(%1187: f32, %1188: f32, %1189: f32):
        %1190 = arith.mulf %1187, %1188 : f32
        linalg.yield %1190 : f32
      } -> tensor<1x1x64xf32>
      %1191 = tensor.empty() : tensor<1x1x2048xf32>
      %1192 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1124 : tensor<1x1x2048xf32>) outs(%1191 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb119(%1193: f32, %1194: f32):
        %1195 = arith.constant 2.000000e+00 : f32
        %1196 = math.powf %1193, %1195 : f32
        linalg.yield %1196 : f32
      } -> tensor<1x1x2048xf32>
      %1197 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1198 = tensor.splat %1197 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1199 = linalg.reduce ins(%1192:tensor<1x1x2048xf32>) outs(%1198:tensor<1x1xf32>) dimensions = [2]
      (%1200: f32, %1201: f32) {
        %1202 = arith.addf %1200, %1201 : f32
        linalg.yield %1202 : f32
      }
      %1203 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 2.048000e+03 : f32
      %1204 = tensor.splat %1203 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1205 = tensor.empty() : tensor<1x1xf32>
      %1206 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1199, %1204 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%1205 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb120(%1207: f32, %1208: f32, %1209: f32):
        %1210 = arith.divf %1207, %1208 : f32
        linalg.yield %1210 : f32
      } -> tensor<1x1xf32>
      %1211 = tensor.collapse_shape %1206 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %1212 = tensor.expand_shape %1211 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %1213 = arith.constant {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %1214 = tensor.splat %1213 {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %1215 = tensor.empty() : tensor<1x1x1xf32>
      %1216 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1212, %1214 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%1215 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb121(%1217: f32, %1218: f32, %1219: f32):
        %1220 = arith.addf %1217, %1218 : f32
        linalg.yield %1220 : f32
      } -> tensor<1x1x1xf32>
      %1221 = tensor.empty() : tensor<1x1x1xf32>
      %1222 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1216 : tensor<1x1x1xf32>) outs(%1221 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb122(%1223: f32, %1224: f32):
        %1225 = math.rsqrt %1223 : f32
        linalg.yield %1225 : f32
      } -> tensor<1x1x1xf32>
      %1226 = tensor.empty() : tensor<1x1x2048xf32>
      %1227 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1124, %1222 : tensor<1x1x2048xf32>, tensor<1x1x1xf32>) outs(%1226 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb123(%1228: f32, %1229: f32, %1230: f32):
        %1231 = arith.mulf %1228, %1229 : f32
        linalg.yield %1231 : f32
      } -> tensor<1x1x2048xf32>
      %1232 = tensor.empty() : tensor<1x1x2048xf32>
      %1233 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %1227 : tensor<2048xf32>, tensor<1x1x2048xf32>) outs(%1232 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb124(%1234: f32, %1235: f32, %1236: f32):
        %1237 = arith.mulf %1234, %1235 : f32
        linalg.yield %1237 : f32
      } -> tensor<1x1x2048xf32>
      %1238 = tensor.empty() : tensor<2048x2048xf32>
      %1239 = linalg.transpose ins(%30:tensor<2048x2048xf32>) outs(%1238:tensor<2048x2048xf32>) permutation = [1, 0]
      %1240 = tensor.collapse_shape %1233 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x2048xf32> into tensor<2048xf32>
      %1241 = tensor.expand_shape %1240 [[0 : i64, 1 : i64]] output_shape [1, 2048] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x2048xf32>
      %1242 = tensor.empty() : tensor<1x2048xf32>
      %1243 = arith.constant 0.000000e+00 : f32
      %1244 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1243 : f32) outs(%1242 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %1245 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%1241, %1239 : tensor<1x2048xf32>, tensor<2048x2048xf32>) outs(%1244 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %1246 = tensor.collapse_shape %1245 [[0 : i64, 1 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x2048xf32> into tensor<2048xf32>
      %1247 = tensor.expand_shape %1246 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 2048] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x1x2048xf32>
      %1248 = tensor.collapse_shape %1247 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x2048xf32> into tensor<2048xf32>
      %1249 = tensor.expand_shape %1248 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 64] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x1x32x64xf32>
      %1250 = tensor.empty() : tensor<1x32x1x64xf32>
      %1251 = linalg.transpose ins(%1249:tensor<1x1x32x64xf32>) outs(%1250:tensor<1x32x1x64xf32>) permutation = [0, 2, 1, 3]
      %1252 = tensor.empty() : tensor<2048x256xf32>
      %1253 = linalg.transpose ins(%27:tensor<256x2048xf32>) outs(%1252:tensor<2048x256xf32>) permutation = [1, 0]
      %1254 = tensor.collapse_shape %1233 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x2048xf32> into tensor<2048xf32>
      %1255 = tensor.expand_shape %1254 [[0 : i64, 1 : i64]] output_shape [1, 2048] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x2048xf32>
      %1256 = tensor.empty() : tensor<1x256xf32>
      %1257 = arith.constant 0.000000e+00 : f32
      %1258 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1257 : f32) outs(%1256 : tensor<1x256xf32>) -> tensor<1x256xf32>
      %1259 = linalg.matmul {prov.region_id = "matmul_2", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%1255, %1253 : tensor<1x2048xf32>, tensor<2048x256xf32>) outs(%1258 : tensor<1x256xf32>) -> tensor<1x256xf32>
      %1260 = tensor.collapse_shape %1259 [[0 : i64, 1 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x256xf32> into tensor<256xf32>
      %1261 = tensor.expand_shape %1260 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 256] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x256xf32>
      %1262 = tensor.collapse_shape %1261 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x256xf32> into tensor<256xf32>
      %1263 = tensor.expand_shape %1262 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 64] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x4x64xf32>
      %1264 = tensor.empty() : tensor<1x4x1x64xf32>
      %1265 = linalg.transpose ins(%1263:tensor<1x1x4x64xf32>) outs(%1264:tensor<1x4x1x64xf32>) permutation = [0, 2, 1, 3]
      %1266 = tensor.empty() : tensor<2048x256xf32>
      %1267 = linalg.transpose ins(%32:tensor<256x2048xf32>) outs(%1266:tensor<2048x256xf32>) permutation = [1, 0]
      %1268 = tensor.collapse_shape %1233 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x2048xf32> into tensor<2048xf32>
      %1269 = tensor.expand_shape %1268 [[0 : i64, 1 : i64]] output_shape [1, 2048] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x2048xf32>
      %1270 = tensor.empty() : tensor<1x256xf32>
      %1271 = arith.constant 0.000000e+00 : f32
      %1272 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1271 : f32) outs(%1270 : tensor<1x256xf32>) -> tensor<1x256xf32>
      %1273 = linalg.matmul {prov.region_id = "matmul_3", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%1269, %1267 : tensor<1x2048xf32>, tensor<2048x256xf32>) outs(%1272 : tensor<1x256xf32>) -> tensor<1x256xf32>
      %1274 = tensor.collapse_shape %1273 [[0 : i64, 1 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x256xf32> into tensor<256xf32>
      %1275 = tensor.expand_shape %1274 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 256] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x256xf32>
      %1276 = tensor.collapse_shape %1275 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x256xf32> into tensor<256xf32>
      %1277 = tensor.expand_shape %1276 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 64] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x4x64xf32>
      %1278 = tensor.empty() : tensor<1x4x1x64xf32>
      %1279 = linalg.transpose ins(%1277:tensor<1x1x4x64xf32>) outs(%1278:tensor<1x4x1x64xf32>) permutation = [0, 2, 1, 3]
      %1280 = tensor.collapse_shape %1173 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x64xf32> into tensor<64xf32>
      %1281 = tensor.expand_shape %1280 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 64] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<64xf32> into tensor<1x1x1x64xf32>
      %1282 = tensor.collapse_shape %1186 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x64xf32> into tensor<64xf32>
      %1283 = tensor.expand_shape %1282 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 64] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<64xf32> into tensor<1x1x1x64xf32>
      %1284 = tensor.empty() : tensor<1x32x1x64xf32>
      %1285 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1251, %1281 : tensor<1x32x1x64xf32>, tensor<1x1x1x64xf32>) outs(%1284 : tensor<1x32x1x64xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb125(%1286: f32, %1287: f32, %1288: f32):
        %1289 = arith.mulf %1286, %1287 : f32
        linalg.yield %1289 : f32
      } -> tensor<1x32x1x64xf32>
      %1290 = "tensor.extract_slice"(%1251) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 32, 1, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x32xf32>
      %1291 = "tensor.extract_slice"(%1251) <{static_offsets = array<i64: 0, 0, 0, 32>, static_sizes = array<i64: 1, 32, 1, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x32xf32>
      %1292 = tensor.empty() : tensor<1x32x1x32xf32>
      %1293 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1291 : tensor<1x32x1x32xf32>) outs(%1292 : tensor<1x32x1x32xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb126(%1294: f32, %1295: f32):
        %1296 = arith.negf %1294 : f32
        linalg.yield %1296 : f32
      } -> tensor<1x32x1x32xf32>
      %1297 = tensor.concat dim(3) %1293, %1290 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x32x1x32xf32>, tensor<1x32x1x32xf32>) -> tensor<1x32x1x64xf32>
      %1298 = tensor.empty() : tensor<1x32x1x64xf32>
      %1299 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1297, %1283 : tensor<1x32x1x64xf32>, tensor<1x1x1x64xf32>) outs(%1298 : tensor<1x32x1x64xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb127(%1300: f32, %1301: f32, %1302: f32):
        %1303 = arith.mulf %1300, %1301 : f32
        linalg.yield %1303 : f32
      } -> tensor<1x32x1x64xf32>
      %1304 = tensor.empty() : tensor<1x32x1x64xf32>
      %1305 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1285, %1299 : tensor<1x32x1x64xf32>, tensor<1x32x1x64xf32>) outs(%1304 : tensor<1x32x1x64xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb128(%1306: f32, %1307: f32, %1308: f32):
        %1309 = arith.addf %1306, %1307 : f32
        linalg.yield %1309 : f32
      } -> tensor<1x32x1x64xf32>
      %1310 = tensor.empty() : tensor<1x4x1x64xf32>
      %1311 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1265, %1281 : tensor<1x4x1x64xf32>, tensor<1x1x1x64xf32>) outs(%1310 : tensor<1x4x1x64xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb129(%1312: f32, %1313: f32, %1314: f32):
        %1315 = arith.mulf %1312, %1313 : f32
        linalg.yield %1315 : f32
      } -> tensor<1x4x1x64xf32>
      %1316 = "tensor.extract_slice"(%1265) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x64xf32>) -> tensor<1x4x1x32xf32>
      %1317 = "tensor.extract_slice"(%1265) <{static_offsets = array<i64: 0, 0, 0, 32>, static_sizes = array<i64: 1, 4, 1, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x64xf32>) -> tensor<1x4x1x32xf32>
      %1318 = tensor.empty() : tensor<1x4x1x32xf32>
      %1319 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1317 : tensor<1x4x1x32xf32>) outs(%1318 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb130(%1320: f32, %1321: f32):
        %1322 = arith.negf %1320 : f32
        linalg.yield %1322 : f32
      } -> tensor<1x4x1x32xf32>
      %1323 = tensor.concat dim(3) %1319, %1316 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x1x32xf32>, tensor<1x4x1x32xf32>) -> tensor<1x4x1x64xf32>
      %1324 = tensor.empty() : tensor<1x4x1x64xf32>
      %1325 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1323, %1283 : tensor<1x4x1x64xf32>, tensor<1x1x1x64xf32>) outs(%1324 : tensor<1x4x1x64xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb131(%1326: f32, %1327: f32, %1328: f32):
        %1329 = arith.mulf %1326, %1327 : f32
        linalg.yield %1329 : f32
      } -> tensor<1x4x1x64xf32>
      %1330 = tensor.empty() : tensor<1x4x1x64xf32>
      %1331 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1311, %1325 : tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>) outs(%1330 : tensor<1x4x1x64xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb132(%1332: f32, %1333: f32, %1334: f32):
        %1335 = arith.addf %1332, %1333 : f32
        linalg.yield %1335 : f32
      } -> tensor<1x4x1x64xf32>
      %1336 = tensor.empty() : tensor<1xi64>
      %1337 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1336 : tensor<1xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
      ^bb133(%1338: i64):
        %1339 = linalg.index 0 : index
        %1340 = arith.index_cast %1339 : index to i64
        %1341 = arith.constant 1 : i64
        %1342 = arith.muli %1340, %1341 : i64
        %1343 = arith.constant 0 : i64
        %1344 = arith.addi %1343, %1342 : i64
        linalg.yield %1344 : i64
      } -> tensor<1xi64>
      %1345 = tensor.empty() : tensor<1xi64>
      %1346 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1118, %1337 : tensor<i64>, tensor<1xi64>) outs(%1345 : tensor<1xi64>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb134(%1347: i64, %1348: i64, %1349: i64):
        %1350 = arith.addi %1347, %1348 : i64
        linalg.yield %1350 : i64
      } -> tensor<1xi64>
      %1351 = "tensor.extract_slice"(%1112) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 15, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x15x64xf32>) -> tensor<4x15x64xf32>
      %1352 = func.call @aten_index_put_default_1_wl1(%1351, %1346, %1331) {prov.region_id = "aten_index_put_default_1_0", prov.dispatch_id = "aten_index_put_default_1_0"} : (tensor<4x15x64xf32>, tensor<1xi64>, tensor<1x4x1x64xf32>) -> tensor<1x4x15x64xf32>
      %1353 = "tensor.extract_slice"(%1113) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 15, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x15x64xf32>) -> tensor<4x15x64xf32>
      %1354 = func.call @aten_index_put_default_1_wl1(%1353, %1346, %1279) {prov.region_id = "aten_index_put_default_1_1", prov.dispatch_id = "aten_index_put_default_1_1"} : (tensor<4x15x64xf32>, tensor<1xi64>, tensor<1x4x1x64xf32>) -> tensor<1x4x15x64xf32>
      %1355 = tensor.collapse_shape %1352 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x15x64xf32> into tensor<3840xf32>
      %1356 = tensor.expand_shape %1355 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 15, 64] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<3840xf32> into tensor<1x4x1x15x64xf32>
      %1357 = tensor.empty() : tensor<1x4x8x15x64xf32>
      %1358 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1356 : tensor<1x4x1x15x64xf32>) outs(%1357 : tensor<1x4x8x15x64xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb135(%1359: f32, %1360: f32):
        linalg.yield %1359 : f32
      } -> tensor<1x4x8x15x64xf32>
      %1361 = tensor.collapse_shape %1358 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x8x15x64xf32> into tensor<30720xf32>
      %1362 = tensor.expand_shape %1361 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 64] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<30720xf32> into tensor<1x32x15x64xf32>
      %1363 = tensor.collapse_shape %1354 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x15x64xf32> into tensor<3840xf32>
      %1364 = tensor.expand_shape %1363 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 15, 64] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<3840xf32> into tensor<1x4x1x15x64xf32>
      %1365 = tensor.empty() : tensor<1x4x8x15x64xf32>
      %1366 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1364 : tensor<1x4x1x15x64xf32>) outs(%1365 : tensor<1x4x8x15x64xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb136(%1367: f32, %1368: f32):
        linalg.yield %1367 : f32
      } -> tensor<1x4x8x15x64xf32>
      %1369 = tensor.collapse_shape %1366 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x8x15x64xf32> into tensor<30720xf32>
      %1370 = tensor.expand_shape %1369 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 64] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<30720xf32> into tensor<1x32x15x64xf32>
      %1371 = tensor.empty() : tensor<1x32x64x15xf32>
      %1372 = linalg.transpose ins(%1362:tensor<1x32x15x64xf32>) outs(%1371:tensor<1x32x64x15xf32>) permutation = [0, 1, 3, 2]
      %1373 = tensor.empty() : tensor<1x32x1x64xf32>
      %1374 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1305 : tensor<1x32x1x64xf32>) outs(%1373 : tensor<1x32x1x64xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb137(%1375: f32, %1376: f32):
        linalg.yield %1375 : f32
      } -> tensor<1x32x1x64xf32>
      %1377 = tensor.collapse_shape %1374 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x1x64xf32> into tensor<2048xf32>
      %1378 = tensor.expand_shape %1377 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 1, 64] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<32x1x64xf32>
      %1379 = tensor.empty() : tensor<1x32x64x15xf32>
      %1380 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1372 : tensor<1x32x64x15xf32>) outs(%1379 : tensor<1x32x64x15xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb138(%1381: f32, %1382: f32):
        linalg.yield %1381 : f32
      } -> tensor<1x32x64x15xf32>
      %1383 = tensor.collapse_shape %1380 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x64x15xf32> into tensor<30720xf32>
      %1384 = tensor.expand_shape %1383 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 64, 15] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<30720xf32> into tensor<32x64x15xf32>
      %1385 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1386 = tensor.splat %1385 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<32x1x15xf32>
      %1387 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1378, %1384 : tensor<32x1x64xf32>, tensor<32x64x15xf32>) outs(%1386 : tensor<32x1x15xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
      ^bb139(%1388: f32, %1389: f32, %1390: f32):
        %1391 = arith.mulf %1388, %1389 : f32
        %1392 = arith.addf %1390, %1391 : f32
        linalg.yield %1392 : f32
      } -> tensor<32x1x15xf32>
      %1393 = tensor.collapse_shape %1387 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32x1x15xf32> into tensor<480xf32>
      %1394 = tensor.expand_shape %1393 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 15] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<480xf32> into tensor<1x32x1x15xf32>
      %1395 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 8.000000e+00 : f32
      %1396 = tensor.splat %1395 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x1x15xf32>
      %1397 = tensor.empty() : tensor<1x32x1x15xf32>
      %1398 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1394, %1396 : tensor<1x32x1x15xf32>, tensor<1x32x1x15xf32>) outs(%1397 : tensor<1x32x1x15xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
      ^bb140(%1399: f32, %1400: f32, %1401: f32):
        %1402 = arith.divf %1399, %1400 : f32
        linalg.yield %1402 : f32
      } -> tensor<1x32x1x15xf32>
      %1403 = tensor.empty() : tensor<15xi64>
      %1404 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1403 : tensor<15xi64>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
      ^bb141(%1405: i64):
        %1406 = linalg.index 0 : index
        %1407 = arith.index_cast %1406 : index to i64
        %1408 = arith.constant 1 : i64
        %1409 = arith.muli %1407, %1408 : i64
        %1410 = arith.constant 0 : i64
        %1411 = arith.addi %1410, %1409 : i64
        linalg.yield %1411 : i64
      } -> tensor<15xi64>
      %1412 = tensor.empty() : tensor<1xi64>
      %1413 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1412 : tensor<1xi64>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
      ^bb142(%1414: i64):
        %1415 = linalg.index 0 : index
        %1416 = arith.index_cast %1415 : index to i64
        %1417 = arith.constant 1 : i64
        %1418 = arith.muli %1416, %1417 : i64
        %1419 = arith.constant 0 : i64
        %1420 = arith.addi %1419, %1418 : i64
        linalg.yield %1420 : i64
      } -> tensor<1xi64>
      %1421 = tensor.empty() : tensor<1xi64>
      %1422 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1118, %1413 : tensor<i64>, tensor<1xi64>) outs(%1421 : tensor<1xi64>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb143(%1423: i64, %1424: i64, %1425: i64):
        %1426 = arith.addi %1423, %1424 : i64
        linalg.yield %1426 : i64
      } -> tensor<1xi64>
      %1427 = tensor.expand_shape %1422 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1xi64> into tensor<1x1xi64>
      %1428 = tensor.expand_shape %1404 [[0 : i64, 1 : i64]] output_shape [1, 15] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<15xi64> into tensor<1x15xi64>
      %1429 = tensor.empty() : tensor<1x15xi1>
      %1430 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1428, %1427 : tensor<1x15xi64>, tensor<1x1xi64>) outs(%1429 : tensor<1x15xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
      ^bb144(%1431: i64, %1432: i64, %1433: i1):
        %1434 = arith.cmpi sle, %1431, %1432 : i64
        linalg.yield %1434 : i1
      } -> tensor<1x15xi1>
      %1435 = tensor.collapse_shape %1430 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x15xi1> into tensor<15xi1>
      %1436 = tensor.expand_shape %1435 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 15] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<15xi1> into tensor<1x1x15xi1>
      %1437 = tensor.collapse_shape %1436 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x1x15xi1> into tensor<15xi1>
      %1438 = tensor.expand_shape %1437 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 15] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<15xi1> into tensor<1x1x1x15xi1>
      %1439 = tensor.empty() : tensor<1x1x1x15xi1>
      %1440 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1438 : tensor<1x1x1x15xi1>) outs(%1439 : tensor<1x1x1x15xi1>) attrs =  {prov.region_id = "bitwise_0", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
      ^bb145(%1441: i1, %1442: i1):
        %1443 = arith.constant true
        %1444 = arith.xori %1441, %1443 : i1
        linalg.yield %1444 : i1
      } -> tensor<1x1x1x15xi1>
      %1445 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %1446 = tensor.splat %1445 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
      %1447 = tensor.empty() : tensor<1x32x1x15xf32>
      %1448 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1440, %1446, %1398 : tensor<1x1x1x15xi1>, tensor<f32>, tensor<1x32x1x15xf32>) outs(%1447 : tensor<1x32x1x15xf32>) attrs =  {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32"} {
      ^bb146(%1449: i1, %1450: f32, %1451: f32, %1452: f32):
        %1453 = arith.select %1449, %1450, %1451 : f32
        linalg.yield %1453 : f32
      } -> tensor<1x32x1x15xf32>
      %1454 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %1455 = tensor.splat %1454 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32x1xf32>
      %1456 = linalg.reduce ins(%1448:tensor<1x32x1x15xf32>) outs(%1455:tensor<1x32x1xf32>) dimensions = [3]
      (%1457: f32, %1458: f32) {
        %1459 = arith.maximumf %1457, %1458 : f32
        linalg.yield %1459 : f32
      }
      %1460 = tensor.collapse_shape %1456 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32x1xf32> into tensor<32xf32>
      %1461 = tensor.expand_shape %1460 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32x1x1xf32>
      %1462 = tensor.empty() : tensor<1x32x1x15xf32>
      %1463 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1448, %1461 : tensor<1x32x1x15xf32>, tensor<1x32x1x1xf32>) outs(%1462 : tensor<1x32x1x15xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb147(%1464: f32, %1465: f32, %1466: f32):
        %1467 = arith.subf %1464, %1465 : f32
        linalg.yield %1467 : f32
      } -> tensor<1x32x1x15xf32>
      %1468 = tensor.empty() : tensor<1x32x1x15xf32>
      %1469 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1463 : tensor<1x32x1x15xf32>) outs(%1468 : tensor<1x32x1x15xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb148(%1470: f32, %1471: f32):
        %1472 = math.exp %1470 : f32
        linalg.yield %1472 : f32
      } -> tensor<1x32x1x15xf32>
      %1473 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1474 = tensor.splat %1473 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32x1xf32>
      %1475 = linalg.reduce ins(%1469:tensor<1x32x1x15xf32>) outs(%1474:tensor<1x32x1xf32>) dimensions = [3]
      (%1476: f32, %1477: f32) {
        %1478 = arith.addf %1476, %1477 : f32
        linalg.yield %1478 : f32
      }
      %1479 = tensor.collapse_shape %1475 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32x1xf32> into tensor<32xf32>
      %1480 = tensor.expand_shape %1479 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32x1x1xf32>
      %1481 = tensor.empty() : tensor<1x32x1x15xf32>
      %1482 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1469, %1480 : tensor<1x32x1x15xf32>, tensor<1x32x1x1xf32>) outs(%1481 : tensor<1x32x1x15xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb149(%1483: f32, %1484: f32, %1485: f32):
        %1486 = arith.divf %1483, %1484 : f32
        linalg.yield %1486 : f32
      } -> tensor<1x32x1x15xf32>
      %1487 = tensor.empty() : tensor<1x32x1x15xf32>
      %1488 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1482 : tensor<1x32x1x15xf32>) outs(%1487 : tensor<1x32x1x15xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb150(%1489: f32, %1490: f32):
        linalg.yield %1489 : f32
      } -> tensor<1x32x1x15xf32>
      %1491 = tensor.collapse_shape %1488 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x1x15xf32> into tensor<480xf32>
      %1492 = tensor.expand_shape %1491 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 1, 15] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<480xf32> into tensor<32x1x15xf32>
      %1493 = tensor.empty() : tensor<1x32x15x64xf32>
      %1494 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1370 : tensor<1x32x15x64xf32>) outs(%1493 : tensor<1x32x15x64xf32>) attrs =  {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb151(%1495: f32, %1496: f32):
        linalg.yield %1495 : f32
      } -> tensor<1x32x15x64xf32>
      %1497 = tensor.collapse_shape %1494 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x15x64xf32> into tensor<30720xf32>
      %1498 = tensor.expand_shape %1497 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 15, 64] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<30720xf32> into tensor<32x15x64xf32>
      %1499 = arith.constant {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1500 = tensor.splat %1499 {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<32x1x64xf32>
      %1501 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1492, %1498 : tensor<32x1x15xf32>, tensor<32x15x64xf32>) outs(%1500 : tensor<32x1x64xf32>) attrs =  {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
      ^bb152(%1502: f32, %1503: f32, %1504: f32):
        %1505 = arith.mulf %1502, %1503 : f32
        %1506 = arith.addf %1504, %1505 : f32
        linalg.yield %1506 : f32
      } -> tensor<32x1x64xf32>
      %1507 = tensor.collapse_shape %1501 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32x1x64xf32> into tensor<2048xf32>
      %1508 = tensor.expand_shape %1507 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 64] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x32x1x64xf32>
      %1509 = tensor.empty() : tensor<1x1x32x64xf32>
      %1510 = linalg.transpose ins(%1508:tensor<1x32x1x64xf32>) outs(%1509:tensor<1x1x32x64xf32>) permutation = [0, 2, 1, 3]
      %1511 = tensor.collapse_shape %1510 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x32x64xf32> into tensor<2048xf32>
      %1512 = tensor.expand_shape %1511 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 2048] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x1x2048xf32>
      %1513 = tensor.empty() : tensor<2048x2048xf32>
      %1514 = linalg.transpose ins(%28:tensor<2048x2048xf32>) outs(%1513:tensor<2048x2048xf32>) permutation = [1, 0]
      %1515 = tensor.collapse_shape %1512 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x2048xf32> into tensor<2048xf32>
      %1516 = tensor.expand_shape %1515 [[0 : i64, 1 : i64]] output_shape [1, 2048] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x2048xf32>
      %1517 = tensor.empty() : tensor<1x2048xf32>
      %1518 = arith.constant 0.000000e+00 : f32
      %1519 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1518 : f32) outs(%1517 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %1520 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%1516, %1514 : tensor<1x2048xf32>, tensor<2048x2048xf32>) outs(%1519 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %1521 = tensor.collapse_shape %1520 [[0 : i64, 1 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x2048xf32> into tensor<2048xf32>
      %1522 = tensor.expand_shape %1521 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 2048] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x1x2048xf32>
      %1523 = tensor.empty() : tensor<1x1x2048xf32>
      %1524 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1124, %1522 : tensor<1x1x2048xf32>, tensor<1x1x2048xf32>) outs(%1523 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb153(%1525: f32, %1526: f32, %1527: f32):
        %1528 = arith.addf %1525, %1526 : f32
        linalg.yield %1528 : f32
      } -> tensor<1x1x2048xf32>
      %1529 = tensor.empty() : tensor<1x1x2048xf32>
      %1530 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1524 : tensor<1x1x2048xf32>) outs(%1529 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb154(%1531: f32, %1532: f32):
        %1533 = arith.constant 2.000000e+00 : f32
        %1534 = math.powf %1531, %1533 : f32
        linalg.yield %1534 : f32
      } -> tensor<1x1x2048xf32>
      %1535 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1536 = tensor.splat %1535 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1537 = linalg.reduce ins(%1530:tensor<1x1x2048xf32>) outs(%1536:tensor<1x1xf32>) dimensions = [2]
      (%1538: f32, %1539: f32) {
        %1540 = arith.addf %1538, %1539 : f32
        linalg.yield %1540 : f32
      }
      %1541 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 2.048000e+03 : f32
      %1542 = tensor.splat %1541 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1543 = tensor.empty() : tensor<1x1xf32>
      %1544 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1537, %1542 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%1543 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb155(%1545: f32, %1546: f32, %1547: f32):
        %1548 = arith.divf %1545, %1546 : f32
        linalg.yield %1548 : f32
      } -> tensor<1x1xf32>
      %1549 = tensor.collapse_shape %1544 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %1550 = tensor.expand_shape %1549 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %1551 = arith.constant {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %1552 = tensor.splat %1551 {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %1553 = tensor.empty() : tensor<1x1x1xf32>
      %1554 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1550, %1552 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%1553 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb156(%1555: f32, %1556: f32, %1557: f32):
        %1558 = arith.addf %1555, %1556 : f32
        linalg.yield %1558 : f32
      } -> tensor<1x1x1xf32>
      %1559 = tensor.empty() : tensor<1x1x1xf32>
      %1560 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1554 : tensor<1x1x1xf32>) outs(%1559 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb157(%1561: f32, %1562: f32):
        %1563 = math.rsqrt %1561 : f32
        linalg.yield %1563 : f32
      } -> tensor<1x1x1xf32>
      %1564 = tensor.empty() : tensor<1x1x2048xf32>
      %1565 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1524, %1560 : tensor<1x1x2048xf32>, tensor<1x1x1xf32>) outs(%1564 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb158(%1566: f32, %1567: f32, %1568: f32):
        %1569 = arith.mulf %1566, %1567 : f32
        linalg.yield %1569 : f32
      } -> tensor<1x1x2048xf32>
      %1570 = tensor.empty() : tensor<1x1x2048xf32>
      %1571 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %1565 : tensor<2048xf32>, tensor<1x1x2048xf32>) outs(%1570 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb159(%1572: f32, %1573: f32, %1574: f32):
        %1575 = arith.mulf %1572, %1573 : f32
        linalg.yield %1575 : f32
      } -> tensor<1x1x2048xf32>
      %1576 = tensor.empty() : tensor<2048x5632xf32>
      %1577 = linalg.transpose ins(%25:tensor<5632x2048xf32>) outs(%1576:tensor<2048x5632xf32>) permutation = [1, 0]
      %1578 = tensor.collapse_shape %1571 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x2048xf32> into tensor<2048xf32>
      %1579 = tensor.expand_shape %1578 [[0 : i64, 1 : i64]] output_shape [1, 2048] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x2048xf32>
      %1580 = tensor.empty() : tensor<1x5632xf32>
      %1581 = arith.constant 0.000000e+00 : f32
      %1582 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1581 : f32) outs(%1580 : tensor<1x5632xf32>) -> tensor<1x5632xf32>
      %1583 = linalg.matmul {prov.region_id = "matmul_7", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%1579, %1577 : tensor<1x2048xf32>, tensor<2048x5632xf32>) outs(%1582 : tensor<1x5632xf32>) -> tensor<1x5632xf32>
      %1584 = tensor.collapse_shape %1583 [[0 : i64, 1 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x5632xf32> into tensor<5632xf32>
      %1585 = tensor.expand_shape %1584 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 5632] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<5632xf32> into tensor<1x1x5632xf32>
      %1586 = tensor.empty() : tensor<1x1x5632xf32>
      %1587 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1585 : tensor<1x1x5632xf32>) outs(%1586 : tensor<1x1x5632xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32"} {
      ^bb160(%1588: f32, %1589: f32):
        %1590 = arith.constant 1.000000e+00 : f32
        %1591 = arith.negf %1588 : f32
        %1592 = math.exp %1591 : f32
        %1593 = arith.addf %1590, %1592 : f32
        %1594 = arith.divf %1590, %1593 : f32
        linalg.yield %1594 : f32
      } -> tensor<1x1x5632xf32>
      %1595 = tensor.empty() : tensor<1x1x5632xf32>
      %1596 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1585, %1587 : tensor<1x1x5632xf32>, tensor<1x1x5632xf32>) outs(%1595 : tensor<1x1x5632xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb161(%1597: f32, %1598: f32, %1599: f32):
        %1600 = arith.mulf %1597, %1598 : f32
        linalg.yield %1600 : f32
      } -> tensor<1x1x5632xf32>
      %1601 = tensor.empty() : tensor<2048x5632xf32>
      %1602 = linalg.transpose ins(%31:tensor<5632x2048xf32>) outs(%1601:tensor<2048x5632xf32>) permutation = [1, 0]
      %1603 = tensor.collapse_shape %1571 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x2048xf32> into tensor<2048xf32>
      %1604 = tensor.expand_shape %1603 [[0 : i64, 1 : i64]] output_shape [1, 2048] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x2048xf32>
      %1605 = tensor.empty() : tensor<1x5632xf32>
      %1606 = arith.constant 0.000000e+00 : f32
      %1607 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1606 : f32) outs(%1605 : tensor<1x5632xf32>) -> tensor<1x5632xf32>
      %1608 = linalg.matmul {prov.region_id = "matmul_8", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%1604, %1602 : tensor<1x2048xf32>, tensor<2048x5632xf32>) outs(%1607 : tensor<1x5632xf32>) -> tensor<1x5632xf32>
      %1609 = tensor.collapse_shape %1608 [[0 : i64, 1 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x5632xf32> into tensor<5632xf32>
      %1610 = tensor.expand_shape %1609 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 5632] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<5632xf32> into tensor<1x1x5632xf32>
      %1611 = tensor.empty() : tensor<1x1x5632xf32>
      %1612 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1596, %1610 : tensor<1x1x5632xf32>, tensor<1x1x5632xf32>) outs(%1611 : tensor<1x1x5632xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb162(%1613: f32, %1614: f32, %1615: f32):
        %1616 = arith.mulf %1613, %1614 : f32
        linalg.yield %1616 : f32
      } -> tensor<1x1x5632xf32>
      %1617 = tensor.empty() : tensor<5632x2048xf32>
      %1618 = linalg.transpose ins(%24:tensor<2048x5632xf32>) outs(%1617:tensor<5632x2048xf32>) permutation = [1, 0]
      %1619 = tensor.collapse_shape %1612 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x5632xf32> into tensor<5632xf32>
      %1620 = tensor.expand_shape %1619 [[0 : i64, 1 : i64]] output_shape [1, 5632] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<5632xf32> into tensor<1x5632xf32>
      %1621 = tensor.empty() : tensor<1x2048xf32>
      %1622 = arith.constant 0.000000e+00 : f32
      %1623 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1622 : f32) outs(%1621 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %1624 = linalg.matmul {prov.region_id = "matmul_9", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%1620, %1618 : tensor<1x5632xf32>, tensor<5632x2048xf32>) outs(%1623 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %1625 = tensor.collapse_shape %1624 [[0 : i64, 1 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x2048xf32> into tensor<2048xf32>
      %1626 = tensor.expand_shape %1625 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 2048] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x1x2048xf32>
      %1627 = tensor.empty() : tensor<1x1x2048xf32>
      %1628 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1524, %1626 : tensor<1x1x2048xf32>, tensor<1x1x2048xf32>) outs(%1627 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb163(%1629: f32, %1630: f32, %1631: f32):
        %1632 = arith.addf %1629, %1630 : f32
        linalg.yield %1632 : f32
      } -> tensor<1x1x2048xf32>
      %1633 = tensor.empty() : tensor<1x1x2048xf32>
      %1634 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1628 : tensor<1x1x2048xf32>) outs(%1633 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb164(%1635: f32, %1636: f32):
        %1637 = arith.constant 2.000000e+00 : f32
        %1638 = math.powf %1635, %1637 : f32
        linalg.yield %1638 : f32
      } -> tensor<1x1x2048xf32>
      %1639 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1640 = tensor.splat %1639 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1641 = linalg.reduce ins(%1634:tensor<1x1x2048xf32>) outs(%1640:tensor<1x1xf32>) dimensions = [2]
      (%1642: f32, %1643: f32) {
        %1644 = arith.addf %1642, %1643 : f32
        linalg.yield %1644 : f32
      }
      %1645 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 2.048000e+03 : f32
      %1646 = tensor.splat %1645 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1647 = tensor.empty() : tensor<1x1xf32>
      %1648 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1641, %1646 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%1647 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb165(%1649: f32, %1650: f32, %1651: f32):
        %1652 = arith.divf %1649, %1650 : f32
        linalg.yield %1652 : f32
      } -> tensor<1x1xf32>
      %1653 = tensor.collapse_shape %1648 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %1654 = tensor.expand_shape %1653 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %1655 = arith.constant {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %1656 = tensor.splat %1655 {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %1657 = tensor.empty() : tensor<1x1x1xf32>
      %1658 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1654, %1656 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%1657 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb166(%1659: f32, %1660: f32, %1661: f32):
        %1662 = arith.addf %1659, %1660 : f32
        linalg.yield %1662 : f32
      } -> tensor<1x1x1xf32>
      %1663 = tensor.empty() : tensor<1x1x1xf32>
      %1664 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1658 : tensor<1x1x1xf32>) outs(%1663 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb167(%1665: f32, %1666: f32):
        %1667 = math.rsqrt %1665 : f32
        linalg.yield %1667 : f32
      } -> tensor<1x1x1xf32>
      %1668 = tensor.empty() : tensor<1x1x2048xf32>
      %1669 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1628, %1664 : tensor<1x1x2048xf32>, tensor<1x1x1xf32>) outs(%1668 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb168(%1670: f32, %1671: f32, %1672: f32):
        %1673 = arith.mulf %1670, %1671 : f32
        linalg.yield %1673 : f32
      } -> tensor<1x1x2048xf32>
      %1674 = tensor.empty() : tensor<1x1x2048xf32>
      %1675 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %1669 : tensor<2048xf32>, tensor<1x1x2048xf32>) outs(%1674 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb169(%1676: f32, %1677: f32, %1678: f32):
        %1679 = arith.mulf %1676, %1677 : f32
        linalg.yield %1679 : f32
      } -> tensor<1x1x2048xf32>
      %1680 = tensor.empty() : tensor<2048x2048xf32>
      %1681 = linalg.transpose ins(%39:tensor<2048x2048xf32>) outs(%1680:tensor<2048x2048xf32>) permutation = [1, 0]
      %1682 = tensor.collapse_shape %1675 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x2048xf32> into tensor<2048xf32>
      %1683 = tensor.expand_shape %1682 [[0 : i64, 1 : i64]] output_shape [1, 2048] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x2048xf32>
      %1684 = tensor.empty() : tensor<1x2048xf32>
      %1685 = arith.constant 0.000000e+00 : f32
      %1686 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1685 : f32) outs(%1684 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %1687 = linalg.matmul {prov.region_id = "matmul_10", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%1683, %1681 : tensor<1x2048xf32>, tensor<2048x2048xf32>) outs(%1686 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %1688 = tensor.collapse_shape %1687 [[0 : i64, 1 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x2048xf32> into tensor<2048xf32>
      %1689 = tensor.expand_shape %1688 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 2048] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x1x2048xf32>
      %1690 = tensor.collapse_shape %1689 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x2048xf32> into tensor<2048xf32>
      %1691 = tensor.expand_shape %1690 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 64] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x1x32x64xf32>
      %1692 = tensor.empty() : tensor<1x32x1x64xf32>
      %1693 = linalg.transpose ins(%1691:tensor<1x1x32x64xf32>) outs(%1692:tensor<1x32x1x64xf32>) permutation = [0, 2, 1, 3]
      %1694 = tensor.empty() : tensor<2048x256xf32>
      %1695 = linalg.transpose ins(%36:tensor<256x2048xf32>) outs(%1694:tensor<2048x256xf32>) permutation = [1, 0]
      %1696 = tensor.collapse_shape %1675 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x2048xf32> into tensor<2048xf32>
      %1697 = tensor.expand_shape %1696 [[0 : i64, 1 : i64]] output_shape [1, 2048] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x2048xf32>
      %1698 = tensor.empty() : tensor<1x256xf32>
      %1699 = arith.constant 0.000000e+00 : f32
      %1700 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1699 : f32) outs(%1698 : tensor<1x256xf32>) -> tensor<1x256xf32>
      %1701 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%1697, %1695 : tensor<1x2048xf32>, tensor<2048x256xf32>) outs(%1700 : tensor<1x256xf32>) -> tensor<1x256xf32>
      %1702 = tensor.collapse_shape %1701 [[0 : i64, 1 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x256xf32> into tensor<256xf32>
      %1703 = tensor.expand_shape %1702 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 256] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x256xf32>
      %1704 = tensor.collapse_shape %1703 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x256xf32> into tensor<256xf32>
      %1705 = tensor.expand_shape %1704 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 64] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x4x64xf32>
      %1706 = tensor.empty() : tensor<1x4x1x64xf32>
      %1707 = linalg.transpose ins(%1705:tensor<1x1x4x64xf32>) outs(%1706:tensor<1x4x1x64xf32>) permutation = [0, 2, 1, 3]
      %1708 = tensor.empty() : tensor<2048x256xf32>
      %1709 = linalg.transpose ins(%41:tensor<256x2048xf32>) outs(%1708:tensor<2048x256xf32>) permutation = [1, 0]
      %1710 = tensor.collapse_shape %1675 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x2048xf32> into tensor<2048xf32>
      %1711 = tensor.expand_shape %1710 [[0 : i64, 1 : i64]] output_shape [1, 2048] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x2048xf32>
      %1712 = tensor.empty() : tensor<1x256xf32>
      %1713 = arith.constant 0.000000e+00 : f32
      %1714 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1713 : f32) outs(%1712 : tensor<1x256xf32>) -> tensor<1x256xf32>
      %1715 = linalg.matmul {prov.region_id = "matmul_12", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%1711, %1709 : tensor<1x2048xf32>, tensor<2048x256xf32>) outs(%1714 : tensor<1x256xf32>) -> tensor<1x256xf32>
      %1716 = tensor.collapse_shape %1715 [[0 : i64, 1 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x256xf32> into tensor<256xf32>
      %1717 = tensor.expand_shape %1716 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 256] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x256xf32>
      %1718 = tensor.collapse_shape %1717 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x256xf32> into tensor<256xf32>
      %1719 = tensor.expand_shape %1718 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 64] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x4x64xf32>
      %1720 = tensor.empty() : tensor<1x4x1x64xf32>
      %1721 = linalg.transpose ins(%1719:tensor<1x1x4x64xf32>) outs(%1720:tensor<1x4x1x64xf32>) permutation = [0, 2, 1, 3]
      %1722 = tensor.collapse_shape %1173 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x64xf32> into tensor<64xf32>
      %1723 = tensor.expand_shape %1722 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 64] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<64xf32> into tensor<1x1x1x64xf32>
      %1724 = tensor.collapse_shape %1186 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x64xf32> into tensor<64xf32>
      %1725 = tensor.expand_shape %1724 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 64] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<64xf32> into tensor<1x1x1x64xf32>
      %1726 = tensor.empty() : tensor<1x32x1x64xf32>
      %1727 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1693, %1723 : tensor<1x32x1x64xf32>, tensor<1x1x1x64xf32>) outs(%1726 : tensor<1x32x1x64xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb170(%1728: f32, %1729: f32, %1730: f32):
        %1731 = arith.mulf %1728, %1729 : f32
        linalg.yield %1731 : f32
      } -> tensor<1x32x1x64xf32>
      %1732 = "tensor.extract_slice"(%1693) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 32, 1, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x32xf32>
      %1733 = "tensor.extract_slice"(%1693) <{static_offsets = array<i64: 0, 0, 0, 32>, static_sizes = array<i64: 1, 32, 1, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x32x1x64xf32>) -> tensor<1x32x1x32xf32>
      %1734 = tensor.empty() : tensor<1x32x1x32xf32>
      %1735 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1733 : tensor<1x32x1x32xf32>) outs(%1734 : tensor<1x32x1x32xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb171(%1736: f32, %1737: f32):
        %1738 = arith.negf %1736 : f32
        linalg.yield %1738 : f32
      } -> tensor<1x32x1x32xf32>
      %1739 = tensor.concat dim(3) %1735, %1732 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x32x1x32xf32>, tensor<1x32x1x32xf32>) -> tensor<1x32x1x64xf32>
      %1740 = tensor.empty() : tensor<1x32x1x64xf32>
      %1741 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1739, %1725 : tensor<1x32x1x64xf32>, tensor<1x1x1x64xf32>) outs(%1740 : tensor<1x32x1x64xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb172(%1742: f32, %1743: f32, %1744: f32):
        %1745 = arith.mulf %1742, %1743 : f32
        linalg.yield %1745 : f32
      } -> tensor<1x32x1x64xf32>
      %1746 = tensor.empty() : tensor<1x32x1x64xf32>
      %1747 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1727, %1741 : tensor<1x32x1x64xf32>, tensor<1x32x1x64xf32>) outs(%1746 : tensor<1x32x1x64xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb173(%1748: f32, %1749: f32, %1750: f32):
        %1751 = arith.addf %1748, %1749 : f32
        linalg.yield %1751 : f32
      } -> tensor<1x32x1x64xf32>
      %1752 = tensor.empty() : tensor<1x4x1x64xf32>
      %1753 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1707, %1723 : tensor<1x4x1x64xf32>, tensor<1x1x1x64xf32>) outs(%1752 : tensor<1x4x1x64xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb174(%1754: f32, %1755: f32, %1756: f32):
        %1757 = arith.mulf %1754, %1755 : f32
        linalg.yield %1757 : f32
      } -> tensor<1x4x1x64xf32>
      %1758 = "tensor.extract_slice"(%1707) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x64xf32>) -> tensor<1x4x1x32xf32>
      %1759 = "tensor.extract_slice"(%1707) <{static_offsets = array<i64: 0, 0, 0, 32>, static_sizes = array<i64: 1, 4, 1, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x64xf32>) -> tensor<1x4x1x32xf32>
      %1760 = tensor.empty() : tensor<1x4x1x32xf32>
      %1761 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1759 : tensor<1x4x1x32xf32>) outs(%1760 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb175(%1762: f32, %1763: f32):
        %1764 = arith.negf %1762 : f32
        linalg.yield %1764 : f32
      } -> tensor<1x4x1x32xf32>
      %1765 = tensor.concat dim(3) %1761, %1758 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x1x32xf32>, tensor<1x4x1x32xf32>) -> tensor<1x4x1x64xf32>
      %1766 = tensor.empty() : tensor<1x4x1x64xf32>
      %1767 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1765, %1725 : tensor<1x4x1x64xf32>, tensor<1x1x1x64xf32>) outs(%1766 : tensor<1x4x1x64xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb176(%1768: f32, %1769: f32, %1770: f32):
        %1771 = arith.mulf %1768, %1769 : f32
        linalg.yield %1771 : f32
      } -> tensor<1x4x1x64xf32>
      %1772 = tensor.empty() : tensor<1x4x1x64xf32>
      %1773 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1753, %1767 : tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>) outs(%1772 : tensor<1x4x1x64xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb177(%1774: f32, %1775: f32, %1776: f32):
        %1777 = arith.addf %1774, %1775 : f32
        linalg.yield %1777 : f32
      } -> tensor<1x4x1x64xf32>
      %1778 = tensor.empty() : tensor<1xi64>
      %1779 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1778 : tensor<1xi64>) attrs =  {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
      ^bb178(%1780: i64):
        %1781 = linalg.index 0 : index
        %1782 = arith.index_cast %1781 : index to i64
        %1783 = arith.constant 1 : i64
        %1784 = arith.muli %1782, %1783 : i64
        %1785 = arith.constant 0 : i64
        %1786 = arith.addi %1785, %1784 : i64
        linalg.yield %1786 : i64
      } -> tensor<1xi64>
      %1787 = tensor.empty() : tensor<1xi64>
      %1788 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1118, %1779 : tensor<i64>, tensor<1xi64>) outs(%1787 : tensor<1xi64>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb179(%1789: i64, %1790: i64, %1791: i64):
        %1792 = arith.addi %1789, %1790 : i64
        linalg.yield %1792 : i64
      } -> tensor<1xi64>
      %1793 = "tensor.extract_slice"(%1112) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 15, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x15x64xf32>) -> tensor<4x15x64xf32>
      %1794 = func.call @aten_index_put_default_1_wl1(%1793, %1788, %1773) {prov.region_id = "aten_index_put_default_1_2", prov.dispatch_id = "aten_index_put_default_1_2"} : (tensor<4x15x64xf32>, tensor<1xi64>, tensor<1x4x1x64xf32>) -> tensor<1x4x15x64xf32>
      %1795 = "tensor.extract_slice"(%1113) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 15, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x15x64xf32>) -> tensor<4x15x64xf32>
      %1796 = func.call @aten_index_put_default_1_wl1(%1795, %1788, %1721) {prov.region_id = "aten_index_put_default_1_3", prov.dispatch_id = "aten_index_put_default_1_3"} : (tensor<4x15x64xf32>, tensor<1xi64>, tensor<1x4x1x64xf32>) -> tensor<1x4x15x64xf32>
      %1797 = tensor.collapse_shape %1794 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x15x64xf32> into tensor<3840xf32>
      %1798 = tensor.expand_shape %1797 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 15, 64] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<3840xf32> into tensor<1x4x1x15x64xf32>
      %1799 = tensor.empty() : tensor<1x4x8x15x64xf32>
      %1800 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1798 : tensor<1x4x1x15x64xf32>) outs(%1799 : tensor<1x4x8x15x64xf32>) attrs =  {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb180(%1801: f32, %1802: f32):
        linalg.yield %1801 : f32
      } -> tensor<1x4x8x15x64xf32>
      %1803 = tensor.collapse_shape %1800 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x8x15x64xf32> into tensor<30720xf32>
      %1804 = tensor.expand_shape %1803 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 64] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<30720xf32> into tensor<1x32x15x64xf32>
      %1805 = tensor.collapse_shape %1796 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x15x64xf32> into tensor<3840xf32>
      %1806 = tensor.expand_shape %1805 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 15, 64] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<3840xf32> into tensor<1x4x1x15x64xf32>
      %1807 = tensor.empty() : tensor<1x4x8x15x64xf32>
      %1808 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1806 : tensor<1x4x1x15x64xf32>) outs(%1807 : tensor<1x4x8x15x64xf32>) attrs =  {prov.region_id = "expand_10", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb181(%1809: f32, %1810: f32):
        linalg.yield %1809 : f32
      } -> tensor<1x4x8x15x64xf32>
      %1811 = tensor.collapse_shape %1808 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x8x15x64xf32> into tensor<30720xf32>
      %1812 = tensor.expand_shape %1811 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 15, 64] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<30720xf32> into tensor<1x32x15x64xf32>
      %1813 = tensor.empty() : tensor<1x32x64x15xf32>
      %1814 = linalg.transpose ins(%1804:tensor<1x32x15x64xf32>) outs(%1813:tensor<1x32x64x15xf32>) permutation = [0, 1, 3, 2]
      %1815 = tensor.empty() : tensor<1x32x1x64xf32>
      %1816 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1747 : tensor<1x32x1x64xf32>) outs(%1815 : tensor<1x32x1x64xf32>) attrs =  {prov.region_id = "expand_11", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb182(%1817: f32, %1818: f32):
        linalg.yield %1817 : f32
      } -> tensor<1x32x1x64xf32>
      %1819 = tensor.collapse_shape %1816 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x1x64xf32> into tensor<2048xf32>
      %1820 = tensor.expand_shape %1819 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 1, 64] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<32x1x64xf32>
      %1821 = tensor.empty() : tensor<1x32x64x15xf32>
      %1822 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1814 : tensor<1x32x64x15xf32>) outs(%1821 : tensor<1x32x64x15xf32>) attrs =  {prov.region_id = "expand_12", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb183(%1823: f32, %1824: f32):
        linalg.yield %1823 : f32
      } -> tensor<1x32x64x15xf32>
      %1825 = tensor.collapse_shape %1822 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x64x15xf32> into tensor<30720xf32>
      %1826 = tensor.expand_shape %1825 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 64, 15] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<30720xf32> into tensor<32x64x15xf32>
      %1827 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1828 = tensor.splat %1827 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<32x1x15xf32>
      %1829 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1820, %1826 : tensor<32x1x64xf32>, tensor<32x64x15xf32>) outs(%1828 : tensor<32x1x15xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
      ^bb184(%1830: f32, %1831: f32, %1832: f32):
        %1833 = arith.mulf %1830, %1831 : f32
        %1834 = arith.addf %1832, %1833 : f32
        linalg.yield %1834 : f32
      } -> tensor<32x1x15xf32>
      %1835 = tensor.collapse_shape %1829 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32x1x15xf32> into tensor<480xf32>
      %1836 = tensor.expand_shape %1835 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 15] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<480xf32> into tensor<1x32x1x15xf32>
      %1837 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 8.000000e+00 : f32
      %1838 = tensor.splat %1837 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x1x15xf32>
      %1839 = tensor.empty() : tensor<1x32x1x15xf32>
      %1840 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1836, %1838 : tensor<1x32x1x15xf32>, tensor<1x32x1x15xf32>) outs(%1839 : tensor<1x32x1x15xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
      ^bb185(%1841: f32, %1842: f32, %1843: f32):
        %1844 = arith.divf %1841, %1842 : f32
        linalg.yield %1844 : f32
      } -> tensor<1x32x1x15xf32>
      %1845 = tensor.empty() : tensor<15xi64>
      %1846 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1845 : tensor<15xi64>) attrs =  {prov.region_id = "iota_4", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
      ^bb186(%1847: i64):
        %1848 = linalg.index 0 : index
        %1849 = arith.index_cast %1848 : index to i64
        %1850 = arith.constant 1 : i64
        %1851 = arith.muli %1849, %1850 : i64
        %1852 = arith.constant 0 : i64
        %1853 = arith.addi %1852, %1851 : i64
        linalg.yield %1853 : i64
      } -> tensor<15xi64>
      %1854 = tensor.empty() : tensor<1xi64>
      %1855 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1854 : tensor<1xi64>) attrs =  {prov.region_id = "iota_5", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
      ^bb187(%1856: i64):
        %1857 = linalg.index 0 : index
        %1858 = arith.index_cast %1857 : index to i64
        %1859 = arith.constant 1 : i64
        %1860 = arith.muli %1858, %1859 : i64
        %1861 = arith.constant 0 : i64
        %1862 = arith.addi %1861, %1860 : i64
        linalg.yield %1862 : i64
      } -> tensor<1xi64>
      %1863 = tensor.empty() : tensor<1xi64>
      %1864 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1118, %1855 : tensor<i64>, tensor<1xi64>) outs(%1863 : tensor<1xi64>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb188(%1865: i64, %1866: i64, %1867: i64):
        %1868 = arith.addi %1865, %1866 : i64
        linalg.yield %1868 : i64
      } -> tensor<1xi64>
      %1869 = tensor.expand_shape %1864 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1xi64> into tensor<1x1xi64>
      %1870 = tensor.expand_shape %1846 [[0 : i64, 1 : i64]] output_shape [1, 15] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<15xi64> into tensor<1x15xi64>
      %1871 = tensor.empty() : tensor<1x15xi1>
      %1872 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1870, %1869 : tensor<1x15xi64>, tensor<1x1xi64>) outs(%1871 : tensor<1x15xi1>) attrs =  {prov.region_id = "compare_1", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
      ^bb189(%1873: i64, %1874: i64, %1875: i1):
        %1876 = arith.cmpi sle, %1873, %1874 : i64
        linalg.yield %1876 : i1
      } -> tensor<1x15xi1>
      %1877 = tensor.collapse_shape %1872 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x15xi1> into tensor<15xi1>
      %1878 = tensor.expand_shape %1877 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 15] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<15xi1> into tensor<1x1x15xi1>
      %1879 = tensor.collapse_shape %1878 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x1x15xi1> into tensor<15xi1>
      %1880 = tensor.expand_shape %1879 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 15] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<15xi1> into tensor<1x1x1x15xi1>
      %1881 = tensor.empty() : tensor<1x1x1x15xi1>
      %1882 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1880 : tensor<1x1x1x15xi1>) outs(%1881 : tensor<1x1x1x15xi1>) attrs =  {prov.region_id = "bitwise_1", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
      ^bb190(%1883: i1, %1884: i1):
        %1885 = arith.constant true
        %1886 = arith.xori %1883, %1885 : i1
        linalg.yield %1886 : i1
      } -> tensor<1x1x1x15xi1>
      %1887 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %1888 = tensor.splat %1887 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
      %1889 = tensor.empty() : tensor<1x32x1x15xf32>
      %1890 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1882, %1888, %1840 : tensor<1x1x1x15xi1>, tensor<f32>, tensor<1x32x1x15xf32>) outs(%1889 : tensor<1x32x1x15xf32>) attrs =  {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32"} {
      ^bb191(%1891: i1, %1892: f32, %1893: f32, %1894: f32):
        %1895 = arith.select %1891, %1892, %1893 : f32
        linalg.yield %1895 : f32
      } -> tensor<1x32x1x15xf32>
      %1896 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %1897 = tensor.splat %1896 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32x1xf32>
      %1898 = linalg.reduce ins(%1890:tensor<1x32x1x15xf32>) outs(%1897:tensor<1x32x1xf32>) dimensions = [3]
      (%1899: f32, %1900: f32) {
        %1901 = arith.maximumf %1899, %1900 : f32
        linalg.yield %1901 : f32
      }
      %1902 = tensor.collapse_shape %1898 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32x1xf32> into tensor<32xf32>
      %1903 = tensor.expand_shape %1902 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32x1x1xf32>
      %1904 = tensor.empty() : tensor<1x32x1x15xf32>
      %1905 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1890, %1903 : tensor<1x32x1x15xf32>, tensor<1x32x1x1xf32>) outs(%1904 : tensor<1x32x1x15xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb192(%1906: f32, %1907: f32, %1908: f32):
        %1909 = arith.subf %1906, %1907 : f32
        linalg.yield %1909 : f32
      } -> tensor<1x32x1x15xf32>
      %1910 = tensor.empty() : tensor<1x32x1x15xf32>
      %1911 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1905 : tensor<1x32x1x15xf32>) outs(%1910 : tensor<1x32x1x15xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb193(%1912: f32, %1913: f32):
        %1914 = math.exp %1912 : f32
        linalg.yield %1914 : f32
      } -> tensor<1x32x1x15xf32>
      %1915 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1916 = tensor.splat %1915 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32x1xf32>
      %1917 = linalg.reduce ins(%1911:tensor<1x32x1x15xf32>) outs(%1916:tensor<1x32x1xf32>) dimensions = [3]
      (%1918: f32, %1919: f32) {
        %1920 = arith.addf %1918, %1919 : f32
        linalg.yield %1920 : f32
      }
      %1921 = tensor.collapse_shape %1917 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x32x1xf32> into tensor<32xf32>
      %1922 = tensor.expand_shape %1921 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32x1x1xf32>
      %1923 = tensor.empty() : tensor<1x32x1x15xf32>
      %1924 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1911, %1922 : tensor<1x32x1x15xf32>, tensor<1x32x1x1xf32>) outs(%1923 : tensor<1x32x1x15xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb194(%1925: f32, %1926: f32, %1927: f32):
        %1928 = arith.divf %1925, %1926 : f32
        linalg.yield %1928 : f32
      } -> tensor<1x32x1x15xf32>
      %1929 = tensor.empty() : tensor<1x32x1x15xf32>
      %1930 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1924 : tensor<1x32x1x15xf32>) outs(%1929 : tensor<1x32x1x15xf32>) attrs =  {prov.region_id = "expand_13", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb195(%1931: f32, %1932: f32):
        linalg.yield %1931 : f32
      } -> tensor<1x32x1x15xf32>
      %1933 = tensor.collapse_shape %1930 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x1x15xf32> into tensor<480xf32>
      %1934 = tensor.expand_shape %1933 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 1, 15] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<480xf32> into tensor<32x1x15xf32>
      %1935 = tensor.empty() : tensor<1x32x15x64xf32>
      %1936 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1812 : tensor<1x32x15x64xf32>) outs(%1935 : tensor<1x32x15x64xf32>) attrs =  {prov.region_id = "expand_14", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb196(%1937: f32, %1938: f32):
        linalg.yield %1937 : f32
      } -> tensor<1x32x15x64xf32>
      %1939 = tensor.collapse_shape %1936 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x15x64xf32> into tensor<30720xf32>
      %1940 = tensor.expand_shape %1939 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 15, 64] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<30720xf32> into tensor<32x15x64xf32>
      %1941 = arith.constant {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1942 = tensor.splat %1941 {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<32x1x64xf32>
      %1943 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1934, %1940 : tensor<32x1x15xf32>, tensor<32x15x64xf32>) outs(%1942 : tensor<32x1x64xf32>) attrs =  {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
      ^bb197(%1944: f32, %1945: f32, %1946: f32):
        %1947 = arith.mulf %1944, %1945 : f32
        %1948 = arith.addf %1946, %1947 : f32
        linalg.yield %1948 : f32
      } -> tensor<32x1x64xf32>
      %1949 = tensor.collapse_shape %1943 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32x1x64xf32> into tensor<2048xf32>
      %1950 = tensor.expand_shape %1949 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 1, 64] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x32x1x64xf32>
      %1951 = tensor.empty() : tensor<1x1x32x64xf32>
      %1952 = linalg.transpose ins(%1950:tensor<1x32x1x64xf32>) outs(%1951:tensor<1x1x32x64xf32>) permutation = [0, 2, 1, 3]
      %1953 = tensor.collapse_shape %1952 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x32x64xf32> into tensor<2048xf32>
      %1954 = tensor.expand_shape %1953 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 2048] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x1x2048xf32>
      %1955 = tensor.empty() : tensor<2048x2048xf32>
      %1956 = linalg.transpose ins(%37:tensor<2048x2048xf32>) outs(%1955:tensor<2048x2048xf32>) permutation = [1, 0]
      %1957 = tensor.collapse_shape %1954 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x2048xf32> into tensor<2048xf32>
      %1958 = tensor.expand_shape %1957 [[0 : i64, 1 : i64]] output_shape [1, 2048] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x2048xf32>
      %1959 = tensor.empty() : tensor<1x2048xf32>
      %1960 = arith.constant 0.000000e+00 : f32
      %1961 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1960 : f32) outs(%1959 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %1962 = linalg.matmul {prov.region_id = "matmul_15", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%1958, %1956 : tensor<1x2048xf32>, tensor<2048x2048xf32>) outs(%1961 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %1963 = tensor.collapse_shape %1962 [[0 : i64, 1 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x2048xf32> into tensor<2048xf32>
      %1964 = tensor.expand_shape %1963 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 2048] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x1x2048xf32>
      %1965 = tensor.empty() : tensor<1x1x2048xf32>
      %1966 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1628, %1964 : tensor<1x1x2048xf32>, tensor<1x1x2048xf32>) outs(%1965 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb198(%1967: f32, %1968: f32, %1969: f32):
        %1970 = arith.addf %1967, %1968 : f32
        linalg.yield %1970 : f32
      } -> tensor<1x1x2048xf32>
      %1971 = tensor.empty() : tensor<1x1x2048xf32>
      %1972 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1966 : tensor<1x1x2048xf32>) outs(%1971 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb199(%1973: f32, %1974: f32):
        %1975 = arith.constant 2.000000e+00 : f32
        %1976 = math.powf %1973, %1975 : f32
        linalg.yield %1976 : f32
      } -> tensor<1x1x2048xf32>
      %1977 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1978 = tensor.splat %1977 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1979 = linalg.reduce ins(%1972:tensor<1x1x2048xf32>) outs(%1978:tensor<1x1xf32>) dimensions = [2]
      (%1980: f32, %1981: f32) {
        %1982 = arith.addf %1980, %1981 : f32
        linalg.yield %1982 : f32
      }
      %1983 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 2.048000e+03 : f32
      %1984 = tensor.splat %1983 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1985 = tensor.empty() : tensor<1x1xf32>
      %1986 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1979, %1984 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%1985 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb200(%1987: f32, %1988: f32, %1989: f32):
        %1990 = arith.divf %1987, %1988 : f32
        linalg.yield %1990 : f32
      } -> tensor<1x1xf32>
      %1991 = tensor.collapse_shape %1986 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %1992 = tensor.expand_shape %1991 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %1993 = arith.constant {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %1994 = tensor.splat %1993 {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %1995 = tensor.empty() : tensor<1x1x1xf32>
      %1996 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1992, %1994 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%1995 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb201(%1997: f32, %1998: f32, %1999: f32):
        %2000 = arith.addf %1997, %1998 : f32
        linalg.yield %2000 : f32
      } -> tensor<1x1x1xf32>
      %2001 = tensor.empty() : tensor<1x1x1xf32>
      %2002 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1996 : tensor<1x1x1xf32>) outs(%2001 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb202(%2003: f32, %2004: f32):
        %2005 = math.rsqrt %2003 : f32
        linalg.yield %2005 : f32
      } -> tensor<1x1x1xf32>
      %2006 = tensor.empty() : tensor<1x1x2048xf32>
      %2007 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1966, %2002 : tensor<1x1x2048xf32>, tensor<1x1x1xf32>) outs(%2006 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb203(%2008: f32, %2009: f32, %2010: f32):
        %2011 = arith.mulf %2008, %2009 : f32
        linalg.yield %2011 : f32
      } -> tensor<1x1x2048xf32>
      %2012 = tensor.empty() : tensor<1x1x2048xf32>
      %2013 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %2007 : tensor<2048xf32>, tensor<1x1x2048xf32>) outs(%2012 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb204(%2014: f32, %2015: f32, %2016: f32):
        %2017 = arith.mulf %2014, %2015 : f32
        linalg.yield %2017 : f32
      } -> tensor<1x1x2048xf32>
      %2018 = tensor.empty() : tensor<2048x5632xf32>
      %2019 = linalg.transpose ins(%34:tensor<5632x2048xf32>) outs(%2018:tensor<2048x5632xf32>) permutation = [1, 0]
      %2020 = tensor.collapse_shape %2013 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x2048xf32> into tensor<2048xf32>
      %2021 = tensor.expand_shape %2020 [[0 : i64, 1 : i64]] output_shape [1, 2048] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x2048xf32>
      %2022 = tensor.empty() : tensor<1x5632xf32>
      %2023 = arith.constant 0.000000e+00 : f32
      %2024 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%2023 : f32) outs(%2022 : tensor<1x5632xf32>) -> tensor<1x5632xf32>
      %2025 = linalg.matmul {prov.region_id = "matmul_16", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%2021, %2019 : tensor<1x2048xf32>, tensor<2048x5632xf32>) outs(%2024 : tensor<1x5632xf32>) -> tensor<1x5632xf32>
      %2026 = tensor.collapse_shape %2025 [[0 : i64, 1 : i64]] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x5632xf32> into tensor<5632xf32>
      %2027 = tensor.expand_shape %2026 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 5632] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<5632xf32> into tensor<1x1x5632xf32>
      %2028 = tensor.empty() : tensor<1x1x5632xf32>
      %2029 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2027 : tensor<1x1x5632xf32>) outs(%2028 : tensor<1x1x5632xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32"} {
      ^bb205(%2030: f32, %2031: f32):
        %2032 = arith.constant 1.000000e+00 : f32
        %2033 = arith.negf %2030 : f32
        %2034 = math.exp %2033 : f32
        %2035 = arith.addf %2032, %2034 : f32
        %2036 = arith.divf %2032, %2035 : f32
        linalg.yield %2036 : f32
      } -> tensor<1x1x5632xf32>
      %2037 = tensor.empty() : tensor<1x1x5632xf32>
      %2038 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2027, %2029 : tensor<1x1x5632xf32>, tensor<1x1x5632xf32>) outs(%2037 : tensor<1x1x5632xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb206(%2039: f32, %2040: f32, %2041: f32):
        %2042 = arith.mulf %2039, %2040 : f32
        linalg.yield %2042 : f32
      } -> tensor<1x1x5632xf32>
      %2043 = tensor.empty() : tensor<2048x5632xf32>
      %2044 = linalg.transpose ins(%40:tensor<5632x2048xf32>) outs(%2043:tensor<2048x5632xf32>) permutation = [1, 0]
      %2045 = tensor.collapse_shape %2013 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x2048xf32> into tensor<2048xf32>
      %2046 = tensor.expand_shape %2045 [[0 : i64, 1 : i64]] output_shape [1, 2048] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x2048xf32>
      %2047 = tensor.empty() : tensor<1x5632xf32>
      %2048 = arith.constant 0.000000e+00 : f32
      %2049 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%2048 : f32) outs(%2047 : tensor<1x5632xf32>) -> tensor<1x5632xf32>
      %2050 = linalg.matmul {prov.region_id = "matmul_17", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%2046, %2044 : tensor<1x2048xf32>, tensor<2048x5632xf32>) outs(%2049 : tensor<1x5632xf32>) -> tensor<1x5632xf32>
      %2051 = tensor.collapse_shape %2050 [[0 : i64, 1 : i64]] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x5632xf32> into tensor<5632xf32>
      %2052 = tensor.expand_shape %2051 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 5632] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<5632xf32> into tensor<1x1x5632xf32>
      %2053 = tensor.empty() : tensor<1x1x5632xf32>
      %2054 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2038, %2052 : tensor<1x1x5632xf32>, tensor<1x1x5632xf32>) outs(%2053 : tensor<1x1x5632xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb207(%2055: f32, %2056: f32, %2057: f32):
        %2058 = arith.mulf %2055, %2056 : f32
        linalg.yield %2058 : f32
      } -> tensor<1x1x5632xf32>
      %2059 = tensor.empty() : tensor<5632x2048xf32>
      %2060 = linalg.transpose ins(%33:tensor<2048x5632xf32>) outs(%2059:tensor<5632x2048xf32>) permutation = [1, 0]
      %2061 = tensor.collapse_shape %2054 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x5632xf32> into tensor<5632xf32>
      %2062 = tensor.expand_shape %2061 [[0 : i64, 1 : i64]] output_shape [1, 5632] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<5632xf32> into tensor<1x5632xf32>
      %2063 = tensor.empty() : tensor<1x2048xf32>
      %2064 = arith.constant 0.000000e+00 : f32
      %2065 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%2064 : f32) outs(%2063 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %2066 = linalg.matmul {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%2062, %2060 : tensor<1x5632xf32>, tensor<5632x2048xf32>) outs(%2065 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %2067 = tensor.collapse_shape %2066 [[0 : i64, 1 : i64]] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x2048xf32> into tensor<2048xf32>
      %2068 = tensor.expand_shape %2067 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 2048] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x1x2048xf32>
      %2069 = tensor.empty() : tensor<1x1x2048xf32>
      %2070 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1966, %2068 : tensor<1x1x2048xf32>, tensor<1x1x2048xf32>) outs(%2069 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb208(%2071: f32, %2072: f32, %2073: f32):
        %2074 = arith.addf %2071, %2072 : f32
        linalg.yield %2074 : f32
      } -> tensor<1x1x2048xf32>
      %2075 = tensor.concat dim(0) %1352, %1794 {prov.region_id = "cat_5", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x15x64xf32>, tensor<1x4x15x64xf32>) -> tensor<2x4x15x64xf32>
      %2076 = tensor.collapse_shape %2075 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2x4x15x64xf32> into tensor<7680xf32>
      %2077 = tensor.expand_shape %2076 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [2, 1, 4, 15, 64] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<7680xf32> into tensor<2x1x4x15x64xf32>
      %2078 = tensor.concat dim(0) %1354, %1796 {prov.region_id = "cat_6", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x15x64xf32>, tensor<1x4x15x64xf32>) -> tensor<2x4x15x64xf32>
      %2079 = tensor.collapse_shape %2078 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2x4x15x64xf32> into tensor<7680xf32>
      %2080 = tensor.expand_shape %2079 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [2, 1, 4, 15, 64] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<7680xf32> into tensor<2x1x4x15x64xf32>
      %2081 = tensor.empty() : tensor<1x1x2048xf32>
      %2082 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2070 : tensor<1x1x2048xf32>) outs(%2081 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb209(%2083: f32, %2084: f32):
        %2085 = arith.constant 2.000000e+00 : f32
        %2086 = math.powf %2083, %2085 : f32
        linalg.yield %2086 : f32
      } -> tensor<1x1x2048xf32>
      %2087 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %2088 = tensor.splat %2087 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %2089 = linalg.reduce ins(%2082:tensor<1x1x2048xf32>) outs(%2088:tensor<1x1xf32>) dimensions = [2]
      (%2090: f32, %2091: f32) {
        %2092 = arith.addf %2090, %2091 : f32
        linalg.yield %2092 : f32
      }
      %2093 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 2.048000e+03 : f32
      %2094 = tensor.splat %2093 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %2095 = tensor.empty() : tensor<1x1xf32>
      %2096 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2089, %2094 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%2095 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb210(%2097: f32, %2098: f32, %2099: f32):
        %2100 = arith.divf %2097, %2098 : f32
        linalg.yield %2100 : f32
      } -> tensor<1x1xf32>
      %2101 = tensor.collapse_shape %2096 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %2102 = tensor.expand_shape %2101 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %2103 = arith.constant {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %2104 = tensor.splat %2103 {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %2105 = tensor.empty() : tensor<1x1x1xf32>
      %2106 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2102, %2104 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%2105 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb211(%2107: f32, %2108: f32, %2109: f32):
        %2110 = arith.addf %2107, %2108 : f32
        linalg.yield %2110 : f32
      } -> tensor<1x1x1xf32>
      %2111 = tensor.empty() : tensor<1x1x1xf32>
      %2112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2106 : tensor<1x1x1xf32>) outs(%2111 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb212(%2113: f32, %2114: f32):
        %2115 = math.rsqrt %2113 : f32
        linalg.yield %2115 : f32
      } -> tensor<1x1x1xf32>
      %2116 = tensor.empty() : tensor<1x1x2048xf32>
      %2117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2070, %2112 : tensor<1x1x2048xf32>, tensor<1x1x1xf32>) outs(%2116 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb213(%2118: f32, %2119: f32, %2120: f32):
        %2121 = arith.mulf %2118, %2119 : f32
        linalg.yield %2121 : f32
      } -> tensor<1x1x2048xf32>
      %2122 = tensor.empty() : tensor<1x1x2048xf32>
      %2123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %2117 : tensor<2048xf32>, tensor<1x1x2048xf32>) outs(%2122 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb214(%2124: f32, %2125: f32, %2126: f32):
        %2127 = arith.mulf %2124, %2125 : f32
        linalg.yield %2127 : f32
      } -> tensor<1x1x2048xf32>
      %2128 = tensor.empty() : tensor<2048x32000xf32>
      %2129 = linalg.transpose ins(%42:tensor<32000x2048xf32>) outs(%2128:tensor<2048x32000xf32>) permutation = [1, 0]
      %2130 = tensor.collapse_shape %2123 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x2048xf32> into tensor<2048xf32>
      %2131 = tensor.expand_shape %2130 [[0 : i64, 1 : i64]] output_shape [1, 2048] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x2048xf32>
      %2132 = tensor.empty() : tensor<1x32000xf32>
      %2133 = arith.constant 0.000000e+00 : f32
      %2134 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%2133 : f32) outs(%2132 : tensor<1x32000xf32>) -> tensor<1x32000xf32>
      %2135 = linalg.matmul {prov.region_id = "matmul_19", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%2131, %2129 : tensor<1x2048xf32>, tensor<2048x32000xf32>) outs(%2134 : tensor<1x32000xf32>) -> tensor<1x32000xf32>
      %2136 = tensor.collapse_shape %2135 [[0 : i64, 1 : i64]] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32000xf32> into tensor<32000xf32>
      %2137 = tensor.expand_shape %2136 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32000] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32000xf32> into tensor<1x1x32000xf32>
      %2138 = "tensor.extract_slice"(%2137) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 32000>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<1x1x32000xf32>) -> tensor<32000xf32>
      %2139 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0xff800000 : f32
      %2140 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0 : i64
      %2141 = tensor.splat %2139 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<f32>
      %2142 = tensor.splat %2140 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<i64>
      %2143, %2144 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%2138 : tensor<32000xf32>) outs(%2141, %2142 : tensor<f32>, tensor<i64>) attrs =  {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} {
      ^bb215(%2145: f32, %2146: f32, %2147: i64):
        %2148 = linalg.index 0 : index
        %2149 = arith.index_cast %2148 : index to i64
        %2150 = arith.cmpf ogt, %2145, %2146 : f32
        %2151 = arith.select %2150, %2145, %2146 : f32
        %2152 = arith.select %2150, %2149, %2147 : i64
        linalg.yield %2151, %2152 : f32, i64
      } -> (tensor<f32>, tensor<i64>)
      %2153 = tensor.extract %2143[] : tensor<f32>
      %2154 = tensor.from_elements %2153 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xf32>
      %2155 = tensor.extract %2144[] : tensor<i64>
      %2156 = tensor.from_elements %2155 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xi64>
      %2157 = arith.constant {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} 1 : i64
      %2158 = tensor.splat %2157 {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} : tensor<i64>
      %2159 = tensor.empty() : tensor<i64>
      %2160 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1109, %2158 : tensor<i64>, tensor<i64>) outs(%2159 : tensor<i64>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb216(%2161: i64, %2162: i64, %2163: i64):
        %2164 = arith.addi %2161, %2162 : i64
        linalg.yield %2164 : i64
      } -> tensor<i64>
      scf.yield %2160, %2156, %1116, %2077, %2080 : tensor<i64>, tensor<1xi64>, tensor<1x7xi64>, tensor<2x1x4x15x64xf32>, tensor<2x1x4x15x64xf32>
    }
    func.return %1105 : tensor<1x7xi64>
  }
}
