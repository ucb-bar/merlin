builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func private @aten_zeros_default() -> tensor<2x1x4x27x128xf32>
  func.func private @aten_zeros_default_1() -> tensor<i64>
  func.func private @aten_index_copy_default(tensor<4x27x128xf32>, tensor<20xi64>, tensor<1x4x20x128xf32>) -> tensor<1x4x27x128xf32>
  func.func private @aten_masked_fill_Scalar(tensor<1x4x20x27xf32>, tensor<1x1x20x27xi1>) -> tensor<1x4x20x27xf32>
  func.func private @aten_stack_default(tensor<1x4x27x128xf32>, tensor<1x4x27x128xf32>) -> tensor<2x1x4x27x128xf32>
  func.func private @aten_zeros_default_2() -> tensor<1x7xi64>
  func.func private @aten_index_copy_default_wl0(tensor<1x7xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x7xi64>
  func.func private @wrap_with_set_grad_enabled_wl1(tensor<64xf32>, tensor<1x1xi64>) -> tensor<1x1x128xf32>
  func.func private @aten_unsqueeze_default_wl2() -> tensor<1x1x1x128xf32>
  func.func private @aten_index_copy_default_1_wl3(tensor<4x27x128xf32>, tensor<1xi64>, tensor<1x4x1x128xf32>) -> tensor<1x4x27x128xf32>
  func.func private @aten_masked_fill_Scalar_wl4(tensor<1x4x1x27xf32>, tensor<1x1x1x27xi1>) -> tensor<1x4x1x27xf32>
  func.func private @aten_stack_default_wl5(tensor<1x4x27x128xf32>, tensor<1x4x27x128xf32>) -> tensor<2x1x4x27x128xf32>
  func.func @forward(%0: tensor<512x128xf32>, %1: tensor<512x128xf32>, %2: tensor<512x128xf32>, %3: tensor<512x128xf32>, %4: tensor<128x512xf32>, %5: tensor<256x128xf32>, %6: tensor<256x128xf32>, %7: tensor<128x256xf32>, %8: tensor<128xf32>, %9: tensor<128xf32>, %10: tensor<512x128xf32>, %11: tensor<512x128xf32>, %12: tensor<512x128xf32>, %13: tensor<128x512xf32>, %14: tensor<256x128xf32>, %15: tensor<256x128xf32>, %16: tensor<128x256xf32>, %17: tensor<128xf32>, %18: tensor<128xf32>, %19: tensor<128xf32>, %20: tensor<512x128xf32>, %21: tensor<512x128xf32>, %22: tensor<512x128xf32>, %23: tensor<512x128xf32>, %24: tensor<128x512xf32>, %25: tensor<256x128xf32>, %26: tensor<256x128xf32>, %27: tensor<128x256xf32>, %28: tensor<128xf32>, %29: tensor<128xf32>, %30: tensor<512x128xf32>, %31: tensor<512x128xf32>, %32: tensor<512x128xf32>, %33: tensor<128x512xf32>, %34: tensor<256x128xf32>, %35: tensor<256x128xf32>, %36: tensor<128x256xf32>, %37: tensor<128xf32>, %38: tensor<128xf32>, %39: tensor<128xf32>, %40: tensor<512x128xf32>, %41: tensor<512x128xf32>, %42: tensor<64xf32>, %43: tensor<64xf32>, %44: tensor<64xf32>, %45: tensor<64xf32>, %46: tensor<i64>, %47: tensor<1x20x128xf32>) -> tensor<1x7xi64> {
    %48 = func.call @aten_zeros_default() {prov.region_id = "aten_zeros_default_0", prov.dispatch_id = "aten_zeros_default_0"} : () -> tensor<2x1x4x27x128xf32>
    %49 = func.call @aten_zeros_default() {prov.region_id = "aten_zeros_default_1", prov.dispatch_id = "aten_zeros_default_1"} : () -> tensor<2x1x4x27x128xf32>
    %50 = func.call @aten_zeros_default_1() {prov.region_id = "aten_zeros_default_1_0", prov.dispatch_id = "aten_zeros_default_1_0"} : () -> tensor<i64>
    %51 = tensor.empty() : tensor<20xi64>
    %52 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%51 : tensor<20xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
    ^bb0(%53: i64):
      %54 = linalg.index 0 : index
      %55 = arith.index_cast %54 : index to i64
      %56 = arith.constant 1 : i64
      %57 = arith.muli %55, %56 : i64
      %58 = arith.constant 0 : i64
      %59 = arith.addi %58, %57 : i64
      linalg.yield %59 : i64
    } -> tensor<20xi64>
    %60 = tensor.expand_shape %52 [[0 : i64, 1 : i64]] output_shape [1, 20] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<20xi64> into tensor<1x20xi64>
    %61 = tensor.expand_shape %44 [[0 : i64, 1 : i64]] output_shape [1, 64] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "rotary", prov.fqn = "rotary"} : tensor<64xf32> into tensor<1x64xf32>
    %62 = tensor.collapse_shape %61 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "rotary", prov.fqn = "rotary"} : tensor<1x64xf32> into tensor<64xf32>
    %63 = tensor.expand_shape %62 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 1] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "rotary", prov.fqn = "rotary"} : tensor<64xf32> into tensor<1x64x1xf32>
    %64 = tensor.empty() : tensor<1x64x1xf32>
    %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63 : tensor<1x64x1xf32>) outs(%64 : tensor<1x64x1xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "rotary", prov.fqn = "rotary"} {
    ^bb1(%66: f32, %67: f32):
      linalg.yield %66 : f32
    } -> tensor<1x64x1xf32>
    %68 = tensor.collapse_shape %60 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "rotary", prov.fqn = "rotary"} : tensor<1x20xi64> into tensor<20xi64>
    %69 = tensor.expand_shape %68 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 20] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "rotary", prov.fqn = "rotary"} : tensor<20xi64> into tensor<1x1x20xi64>
    %70 = tensor.empty() : tensor<1x1x20xf32>
    %71 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%69 : tensor<1x1x20xi64>) outs(%70 : tensor<1x1x20xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "rotary", prov.fqn = "rotary"} {
    ^bb2(%72: i64, %73: f32):
      %74 = arith.sitofp %72 : i64 to f32
      linalg.yield %74 : f32
    } -> tensor<1x1x20xf32>
    %75 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "rotary", prov.fqn = "rotary"} 0.000000e+00 : f32
    %76 = tensor.splat %75 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "rotary", prov.fqn = "rotary"} : tensor<1x64x20xf32>
    %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%65, %71 : tensor<1x64x1xf32>, tensor<1x1x20xf32>) outs(%76 : tensor<1x64x20xf32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "rotary", prov.fqn = "rotary"} {
    ^bb3(%78: f32, %79: f32, %80: f32):
      %81 = arith.mulf %78, %79 : f32
      %82 = arith.addf %80, %81 : f32
      linalg.yield %82 : f32
    } -> tensor<1x64x20xf32>
    %83 = tensor.empty() : tensor<1x20x64xf32>
    %84 = linalg.transpose ins(%77:tensor<1x64x20xf32>) outs(%83:tensor<1x20x64xf32>) permutation = [0, 2, 1]
    %85 = tensor.concat dim(2) %84, %84 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "rotary", prov.fqn = "rotary"} : (tensor<1x20x64xf32>, tensor<1x20x64xf32>) -> tensor<1x20x128xf32>
    %86 = tensor.empty() : tensor<1x20x128xf32>
    %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85 : tensor<1x20x128xf32>) outs(%86 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "rotary", prov.fqn = "rotary"} {
    ^bb4(%88: f32, %89: f32):
      %90 = math.cos %88 : f32
      linalg.yield %90 : f32
    } -> tensor<1x20x128xf32>
    %91 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "rotary", prov.fqn = "rotary"} 1.000000e+00 : f32
    %92 = tensor.splat %91 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "rotary", prov.fqn = "rotary"} : tensor<1x20x128xf32>
    %93 = tensor.empty() : tensor<1x20x128xf32>
    %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %92 : tensor<1x20x128xf32>, tensor<1x20x128xf32>) outs(%93 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "rotary", prov.fqn = "rotary"} {
    ^bb5(%95: f32, %96: f32, %97: f32):
      %98 = arith.mulf %95, %96 : f32
      linalg.yield %98 : f32
    } -> tensor<1x20x128xf32>
    %99 = tensor.empty() : tensor<1x20x128xf32>
    %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85 : tensor<1x20x128xf32>) outs(%99 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "rotary", prov.fqn = "rotary"} {
    ^bb6(%101: f32, %102: f32):
      %103 = math.sin %101 : f32
      linalg.yield %103 : f32
    } -> tensor<1x20x128xf32>
    %104 = arith.constant {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "rotary", prov.fqn = "rotary"} 1.000000e+00 : f32
    %105 = tensor.splat %104 {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "rotary", prov.fqn = "rotary"} : tensor<1x20x128xf32>
    %106 = tensor.empty() : tensor<1x20x128xf32>
    %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%100, %105 : tensor<1x20x128xf32>, tensor<1x20x128xf32>) outs(%106 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "rotary", prov.fqn = "rotary"} {
    ^bb7(%108: f32, %109: f32, %110: f32):
      %111 = arith.mulf %108, %109 : f32
      linalg.yield %111 : f32
    } -> tensor<1x20x128xf32>
    %112 = tensor.empty() : tensor<1x20x128xf32>
    %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47 : tensor<1x20x128xf32>) outs(%112 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb8(%114: f32, %115: f32):
      %116 = arith.constant 2.000000e+00 : f32
      %117 = math.powf %114, %116 : f32
      linalg.yield %117 : f32
    } -> tensor<1x20x128xf32>
    %118 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %119 = tensor.splat %118 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x20xf32>
    %120 = linalg.reduce ins(%113:tensor<1x20x128xf32>) outs(%119:tensor<1x20xf32>) dimensions = [2]
    (%121: f32, %122: f32) {
      %123 = arith.addf %121, %122 : f32
      linalg.yield %123 : f32
    }
    %124 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
    %125 = tensor.splat %124 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x20xf32>
    %126 = tensor.empty() : tensor<1x20xf32>
    %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%120, %125 : tensor<1x20xf32>, tensor<1x20xf32>) outs(%126 : tensor<1x20xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb9(%128: f32, %129: f32, %130: f32):
      %131 = arith.divf %128, %129 : f32
      linalg.yield %131 : f32
    } -> tensor<1x20xf32>
    %132 = tensor.collapse_shape %127 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x20xf32> into tensor<20xf32>
    %133 = tensor.expand_shape %132 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<20xf32> into tensor<1x20x1xf32>
    %134 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %135 = tensor.splat %134 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x20x1xf32>
    %136 = tensor.empty() : tensor<1x20x1xf32>
    %137 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%133, %135 : tensor<1x20x1xf32>, tensor<1x20x1xf32>) outs(%136 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb10(%138: f32, %139: f32, %140: f32):
      %141 = arith.addf %138, %139 : f32
      linalg.yield %141 : f32
    } -> tensor<1x20x1xf32>
    %142 = tensor.empty() : tensor<1x20x1xf32>
    %143 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%137 : tensor<1x20x1xf32>) outs(%142 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb11(%144: f32, %145: f32):
      %146 = math.rsqrt %144 : f32
      linalg.yield %146 : f32
    } -> tensor<1x20x1xf32>
    %147 = tensor.empty() : tensor<1x20x128xf32>
    %148 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %143 : tensor<1x20x128xf32>, tensor<1x20x1xf32>) outs(%147 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb12(%149: f32, %150: f32, %151: f32):
      %152 = arith.mulf %149, %150 : f32
      linalg.yield %152 : f32
    } -> tensor<1x20x128xf32>
    %153 = tensor.empty() : tensor<1x20x128xf32>
    %154 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %148 : tensor<128xf32>, tensor<1x20x128xf32>) outs(%153 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb13(%155: f32, %156: f32, %157: f32):
      %158 = arith.mulf %155, %156 : f32
      linalg.yield %158 : f32
    } -> tensor<1x20x128xf32>
    %159 = tensor.empty() : tensor<128x512xf32>
    %160 = linalg.transpose ins(%21:tensor<512x128xf32>) outs(%159:tensor<128x512xf32>) permutation = [1, 0]
    %161 = tensor.empty() : tensor<1x20x512xf32>
    %162 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %163 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%162 : f32) outs(%161 : tensor<1x20x512xf32>) -> tensor<1x20x512xf32>
    %164 = linalg.matmul {prov.region_id = "matmul_1", prov.dispatch_id = "matmul_1", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} ins(%154, %160 : tensor<1x20x128xf32>, tensor<128x512xf32>) outs(%163 : tensor<1x20x512xf32>) -> tensor<1x20x512xf32>
    %165 = tensor.collapse_shape %164 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x20x512xf32> into tensor<10240xf32>
    %166 = tensor.expand_shape %165 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 20, 4, 128] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<10240xf32> into tensor<1x20x4x128xf32>
    %167 = tensor.empty() : tensor<1x4x20x128xf32>
    %168 = linalg.transpose ins(%166:tensor<1x20x4x128xf32>) outs(%167:tensor<1x4x20x128xf32>) permutation = [0, 2, 1, 3]
    %169 = tensor.empty() : tensor<128x512xf32>
    %170 = linalg.transpose ins(%22:tensor<512x128xf32>) outs(%169:tensor<128x512xf32>) permutation = [1, 0]
    %171 = tensor.empty() : tensor<1x20x512xf32>
    %172 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %173 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%172 : f32) outs(%171 : tensor<1x20x512xf32>) -> tensor<1x20x512xf32>
    %174 = linalg.matmul {prov.region_id = "matmul_2", prov.dispatch_id = "matmul_2", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} ins(%154, %170 : tensor<1x20x128xf32>, tensor<128x512xf32>) outs(%173 : tensor<1x20x512xf32>) -> tensor<1x20x512xf32>
    %175 = tensor.collapse_shape %174 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x20x512xf32> into tensor<10240xf32>
    %176 = tensor.expand_shape %175 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 20, 4, 128] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<10240xf32> into tensor<1x20x4x128xf32>
    %177 = tensor.empty() : tensor<1x4x20x128xf32>
    %178 = linalg.transpose ins(%176:tensor<1x20x4x128xf32>) outs(%177:tensor<1x4x20x128xf32>) permutation = [0, 2, 1, 3]
    %179 = tensor.empty() : tensor<128x512xf32>
    %180 = linalg.transpose ins(%23:tensor<512x128xf32>) outs(%179:tensor<128x512xf32>) permutation = [1, 0]
    %181 = tensor.empty() : tensor<1x20x512xf32>
    %182 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %183 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%182 : f32) outs(%181 : tensor<1x20x512xf32>) -> tensor<1x20x512xf32>
    %184 = linalg.matmul {prov.region_id = "matmul_3", prov.dispatch_id = "matmul_3", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} ins(%154, %180 : tensor<1x20x128xf32>, tensor<128x512xf32>) outs(%183 : tensor<1x20x512xf32>) -> tensor<1x20x512xf32>
    %185 = tensor.collapse_shape %184 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x20x512xf32> into tensor<10240xf32>
    %186 = tensor.expand_shape %185 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 20, 4, 128] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<10240xf32> into tensor<1x20x4x128xf32>
    %187 = tensor.empty() : tensor<1x4x20x128xf32>
    %188 = linalg.transpose ins(%186:tensor<1x20x4x128xf32>) outs(%187:tensor<1x4x20x128xf32>) permutation = [0, 2, 1, 3]
    %189 = tensor.collapse_shape %94 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %190 = tensor.expand_shape %189 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 20, 128] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<2560xf32> into tensor<1x1x20x128xf32>
    %191 = tensor.collapse_shape %107 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %192 = tensor.expand_shape %191 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 20, 128] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<2560xf32> into tensor<1x1x20x128xf32>
    %193 = tensor.empty() : tensor<1x4x20x128xf32>
    %194 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%168, %190 : tensor<1x4x20x128xf32>, tensor<1x1x20x128xf32>) outs(%193 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb14(%195: f32, %196: f32, %197: f32):
      %198 = arith.mulf %195, %196 : f32
      linalg.yield %198 : f32
    } -> tensor<1x4x20x128xf32>
    %199 = "tensor.extract_slice"(%168) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 20, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x20x128xf32>) -> tensor<1x4x20x64xf32>
    %200 = "tensor.extract_slice"(%168) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 20, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x20x128xf32>) -> tensor<1x4x20x64xf32>
    %201 = tensor.empty() : tensor<1x4x20x64xf32>
    %202 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%200 : tensor<1x4x20x64xf32>) outs(%201 : tensor<1x4x20x64xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb15(%203: f32, %204: f32):
      %205 = arith.negf %203 : f32
      linalg.yield %205 : f32
    } -> tensor<1x4x20x64xf32>
    %206 = tensor.concat dim(3) %202, %199 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x20x64xf32>, tensor<1x4x20x64xf32>) -> tensor<1x4x20x128xf32>
    %207 = tensor.empty() : tensor<1x4x20x128xf32>
    %208 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%206, %192 : tensor<1x4x20x128xf32>, tensor<1x1x20x128xf32>) outs(%207 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb16(%209: f32, %210: f32, %211: f32):
      %212 = arith.mulf %209, %210 : f32
      linalg.yield %212 : f32
    } -> tensor<1x4x20x128xf32>
    %213 = tensor.empty() : tensor<1x4x20x128xf32>
    %214 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%194, %208 : tensor<1x4x20x128xf32>, tensor<1x4x20x128xf32>) outs(%213 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb17(%215: f32, %216: f32, %217: f32):
      %218 = arith.addf %215, %216 : f32
      linalg.yield %218 : f32
    } -> tensor<1x4x20x128xf32>
    %219 = tensor.empty() : tensor<1x4x20x128xf32>
    %220 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%178, %190 : tensor<1x4x20x128xf32>, tensor<1x1x20x128xf32>) outs(%219 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb18(%221: f32, %222: f32, %223: f32):
      %224 = arith.mulf %221, %222 : f32
      linalg.yield %224 : f32
    } -> tensor<1x4x20x128xf32>
    %225 = "tensor.extract_slice"(%178) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 20, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x20x128xf32>) -> tensor<1x4x20x64xf32>
    %226 = "tensor.extract_slice"(%178) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 20, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x20x128xf32>) -> tensor<1x4x20x64xf32>
    %227 = tensor.empty() : tensor<1x4x20x64xf32>
    %228 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%226 : tensor<1x4x20x64xf32>) outs(%227 : tensor<1x4x20x64xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb19(%229: f32, %230: f32):
      %231 = arith.negf %229 : f32
      linalg.yield %231 : f32
    } -> tensor<1x4x20x64xf32>
    %232 = tensor.concat dim(3) %228, %225 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x20x64xf32>, tensor<1x4x20x64xf32>) -> tensor<1x4x20x128xf32>
    %233 = tensor.empty() : tensor<1x4x20x128xf32>
    %234 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%232, %192 : tensor<1x4x20x128xf32>, tensor<1x1x20x128xf32>) outs(%233 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb20(%235: f32, %236: f32, %237: f32):
      %238 = arith.mulf %235, %236 : f32
      linalg.yield %238 : f32
    } -> tensor<1x4x20x128xf32>
    %239 = tensor.empty() : tensor<1x4x20x128xf32>
    %240 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%220, %234 : tensor<1x4x20x128xf32>, tensor<1x4x20x128xf32>) outs(%239 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb21(%241: f32, %242: f32, %243: f32):
      %244 = arith.addf %241, %242 : f32
      linalg.yield %244 : f32
    } -> tensor<1x4x20x128xf32>
    %245 = "tensor.extract_slice"(%48) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 27, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x27x128xf32>) -> tensor<4x27x128xf32>
    %246 = tensor.empty() : tensor<20xi64>
    %247 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%246 : tensor<20xi64>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
    ^bb22(%248: i64):
      %249 = linalg.index 0 : index
      %250 = arith.index_cast %249 : index to i64
      %251 = arith.constant 1 : i64
      %252 = arith.muli %250, %251 : i64
      %253 = arith.constant 0 : i64
      %254 = arith.addi %253, %252 : i64
      linalg.yield %254 : i64
    } -> tensor<20xi64>
    %255 = tensor.empty() : tensor<20xi64>
    %256 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%50, %247 : tensor<i64>, tensor<20xi64>) outs(%255 : tensor<20xi64>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb23(%257: i64, %258: i64, %259: i64):
      %260 = arith.addi %257, %258 : i64
      linalg.yield %260 : i64
    } -> tensor<20xi64>
    %261 = func.call @aten_index_copy_default(%245, %256, %240) {prov.region_id = "aten_index_copy_default_0", prov.dispatch_id = "aten_index_copy_default_0"} : (tensor<4x27x128xf32>, tensor<20xi64>, tensor<1x4x20x128xf32>) -> tensor<1x4x27x128xf32>
    %262 = "tensor.extract_slice"(%49) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 27, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x27x128xf32>) -> tensor<4x27x128xf32>
    %263 = tensor.empty() : tensor<20xi64>
    %264 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%263 : tensor<20xi64>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
    ^bb24(%265: i64):
      %266 = linalg.index 0 : index
      %267 = arith.index_cast %266 : index to i64
      %268 = arith.constant 1 : i64
      %269 = arith.muli %267, %268 : i64
      %270 = arith.constant 0 : i64
      %271 = arith.addi %270, %269 : i64
      linalg.yield %271 : i64
    } -> tensor<20xi64>
    %272 = tensor.empty() : tensor<20xi64>
    %273 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%50, %264 : tensor<i64>, tensor<20xi64>) outs(%272 : tensor<20xi64>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb25(%274: i64, %275: i64, %276: i64):
      %277 = arith.addi %274, %275 : i64
      linalg.yield %277 : i64
    } -> tensor<20xi64>
    %278 = func.call @aten_index_copy_default(%262, %273, %188) {prov.region_id = "aten_index_copy_default_1", prov.dispatch_id = "aten_index_copy_default_1"} : (tensor<4x27x128xf32>, tensor<20xi64>, tensor<1x4x20x128xf32>) -> tensor<1x4x27x128xf32>
    %279 = tensor.empty() : tensor<1x4x128x27xf32>
    %280 = linalg.transpose ins(%261:tensor<1x4x27x128xf32>) outs(%279:tensor<1x4x128x27xf32>) permutation = [0, 1, 3, 2]
    %281 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %282 = tensor.splat %281 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x4x20x27xf32>
    %283 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%214, %280 : tensor<1x4x20x128xf32>, tensor<1x4x128x27xf32>) outs(%282 : tensor<1x4x20x27xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb26(%284: f32, %285: f32, %286: f32):
      %287 = arith.mulf %284, %285 : f32
      %288 = arith.addf %286, %287 : f32
      linalg.yield %288 : f32
    } -> tensor<1x4x20x27xf32>
    %289 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 11.3137083 : f32
    %290 = tensor.splat %289 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x4x20x27xf32>
    %291 = tensor.empty() : tensor<1x4x20x27xf32>
    %292 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%283, %290 : tensor<1x4x20x27xf32>, tensor<1x4x20x27xf32>) outs(%291 : tensor<1x4x20x27xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb27(%293: f32, %294: f32, %295: f32):
      %296 = arith.divf %293, %294 : f32
      linalg.yield %296 : f32
    } -> tensor<1x4x20x27xf32>
    %297 = tensor.empty() : tensor<27xi64>
    %298 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%297 : tensor<27xi64>) attrs =  {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
    ^bb28(%299: i64):
      %300 = linalg.index 0 : index
      %301 = arith.index_cast %300 : index to i64
      %302 = arith.constant 1 : i64
      %303 = arith.muli %301, %302 : i64
      %304 = arith.constant 0 : i64
      %305 = arith.addi %304, %303 : i64
      linalg.yield %305 : i64
    } -> tensor<27xi64>
    %306 = tensor.empty() : tensor<20xi64>
    %307 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%306 : tensor<20xi64>) attrs =  {prov.region_id = "iota_4", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
    ^bb29(%308: i64):
      %309 = linalg.index 0 : index
      %310 = arith.index_cast %309 : index to i64
      %311 = arith.constant 1 : i64
      %312 = arith.muli %310, %311 : i64
      %313 = arith.constant 0 : i64
      %314 = arith.addi %313, %312 : i64
      linalg.yield %314 : i64
    } -> tensor<20xi64>
    %315 = tensor.empty() : tensor<20xi64>
    %316 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%50, %307 : tensor<i64>, tensor<20xi64>) outs(%315 : tensor<20xi64>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb30(%317: i64, %318: i64, %319: i64):
      %320 = arith.addi %317, %318 : i64
      linalg.yield %320 : i64
    } -> tensor<20xi64>
    %321 = tensor.expand_shape %316 [[0 : i64, 1 : i64]] output_shape [20, 1] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<20xi64> into tensor<20x1xi64>
    %322 = tensor.expand_shape %298 [[0 : i64, 1 : i64]] output_shape [1, 27] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<27xi64> into tensor<1x27xi64>
    %323 = tensor.empty() : tensor<20x27xi1>
    %324 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%322, %321 : tensor<1x27xi64>, tensor<20x1xi64>) outs(%323 : tensor<20x27xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
    ^bb31(%325: i64, %326: i64, %327: i1):
      %328 = arith.cmpi sle, %325, %326 : i64
      linalg.yield %328 : i1
    } -> tensor<20x27xi1>
    %329 = tensor.collapse_shape %324 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<20x27xi1> into tensor<540xi1>
    %330 = tensor.expand_shape %329 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 27] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<540xi1> into tensor<1x20x27xi1>
    %331 = tensor.collapse_shape %330 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x20x27xi1> into tensor<540xi1>
    %332 = tensor.expand_shape %331 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 20, 27] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<540xi1> into tensor<1x1x20x27xi1>
    %333 = tensor.empty() : tensor<1x1x20x27xi1>
    %334 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%332 : tensor<1x1x20x27xi1>) outs(%333 : tensor<1x1x20x27xi1>) attrs =  {prov.region_id = "bitwise_0", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
    ^bb32(%335: i1, %336: i1):
      %337 = arith.constant true
      %338 = arith.xori %335, %337 : i1
      linalg.yield %338 : i1
    } -> tensor<1x1x20x27xi1>
    %339 = func.call @aten_masked_fill_Scalar(%292, %334) {prov.region_id = "aten_masked_fill_Scalar_0", prov.dispatch_id = "aten_masked_fill_Scalar_0"} : (tensor<1x4x20x27xf32>, tensor<1x1x20x27xi1>) -> tensor<1x4x20x27xf32>
    %340 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0xff800000 : f32
    %341 = tensor.splat %340 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x20xf32>
    %342 = linalg.reduce ins(%339:tensor<1x4x20x27xf32>) outs(%341:tensor<1x4x20xf32>) dimensions = [3]
    (%343: f32, %344: f32) {
      %345 = arith.maximumf %343, %344 : f32
      linalg.yield %345 : f32
    }
    %346 = tensor.collapse_shape %342 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x20xf32> into tensor<80xf32>
    %347 = tensor.expand_shape %346 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 20, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<80xf32> into tensor<1x4x20x1xf32>
    %348 = tensor.empty() : tensor<1x4x20x27xf32>
    %349 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%339, %347 : tensor<1x4x20x27xf32>, tensor<1x4x20x1xf32>) outs(%348 : tensor<1x4x20x27xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb33(%350: f32, %351: f32, %352: f32):
      %353 = arith.subf %350, %351 : f32
      linalg.yield %353 : f32
    } -> tensor<1x4x20x27xf32>
    %354 = tensor.empty() : tensor<1x4x20x27xf32>
    %355 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%349 : tensor<1x4x20x27xf32>) outs(%354 : tensor<1x4x20x27xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb34(%356: f32, %357: f32):
      %358 = math.exp %356 : f32
      linalg.yield %358 : f32
    } -> tensor<1x4x20x27xf32>
    %359 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %360 = tensor.splat %359 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x20xf32>
    %361 = linalg.reduce ins(%355:tensor<1x4x20x27xf32>) outs(%360:tensor<1x4x20xf32>) dimensions = [3]
    (%362: f32, %363: f32) {
      %364 = arith.addf %362, %363 : f32
      linalg.yield %364 : f32
    }
    %365 = tensor.collapse_shape %361 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x20xf32> into tensor<80xf32>
    %366 = tensor.expand_shape %365 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 20, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<80xf32> into tensor<1x4x20x1xf32>
    %367 = tensor.empty() : tensor<1x4x20x27xf32>
    %368 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%355, %366 : tensor<1x4x20x27xf32>, tensor<1x4x20x1xf32>) outs(%367 : tensor<1x4x20x27xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb35(%369: f32, %370: f32, %371: f32):
      %372 = arith.divf %369, %370 : f32
      linalg.yield %372 : f32
    } -> tensor<1x4x20x27xf32>
    %373 = arith.constant {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %374 = tensor.splat %373 {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x4x20x128xf32>
    %375 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%368, %278 : tensor<1x4x20x27xf32>, tensor<1x4x27x128xf32>) outs(%374 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb36(%376: f32, %377: f32, %378: f32):
      %379 = arith.mulf %376, %377 : f32
      %380 = arith.addf %378, %379 : f32
      linalg.yield %380 : f32
    } -> tensor<1x4x20x128xf32>
    %381 = tensor.empty() : tensor<1x20x4x128xf32>
    %382 = linalg.transpose ins(%375:tensor<1x4x20x128xf32>) outs(%381:tensor<1x20x4x128xf32>) permutation = [0, 2, 1, 3]
    %383 = tensor.collapse_shape %382 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x20x4x128xf32> into tensor<10240xf32>
    %384 = tensor.expand_shape %383 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 512] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<10240xf32> into tensor<1x20x512xf32>
    %385 = tensor.empty() : tensor<512x128xf32>
    %386 = linalg.transpose ins(%24:tensor<128x512xf32>) outs(%385:tensor<512x128xf32>) permutation = [1, 0]
    %387 = tensor.empty() : tensor<1x20x128xf32>
    %388 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %389 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%388 : f32) outs(%387 : tensor<1x20x128xf32>) -> tensor<1x20x128xf32>
    %390 = linalg.matmul {prov.region_id = "matmul_6", prov.dispatch_id = "matmul_6", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} ins(%384, %386 : tensor<1x20x512xf32>, tensor<512x128xf32>) outs(%389 : tensor<1x20x128xf32>) -> tensor<1x20x128xf32>
    %391 = tensor.empty() : tensor<1x20x128xf32>
    %392 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %390 : tensor<1x20x128xf32>, tensor<1x20x128xf32>) outs(%391 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb37(%393: f32, %394: f32, %395: f32):
      %396 = arith.addf %393, %394 : f32
      linalg.yield %396 : f32
    } -> tensor<1x20x128xf32>
    %397 = tensor.empty() : tensor<1x20x128xf32>
    %398 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%392 : tensor<1x20x128xf32>) outs(%397 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb38(%399: f32, %400: f32):
      %401 = arith.constant 2.000000e+00 : f32
      %402 = math.powf %399, %401 : f32
      linalg.yield %402 : f32
    } -> tensor<1x20x128xf32>
    %403 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %404 = tensor.splat %403 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x20xf32>
    %405 = linalg.reduce ins(%398:tensor<1x20x128xf32>) outs(%404:tensor<1x20xf32>) dimensions = [2]
    (%406: f32, %407: f32) {
      %408 = arith.addf %406, %407 : f32
      linalg.yield %408 : f32
    }
    %409 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
    %410 = tensor.splat %409 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x20xf32>
    %411 = tensor.empty() : tensor<1x20xf32>
    %412 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%405, %410 : tensor<1x20xf32>, tensor<1x20xf32>) outs(%411 : tensor<1x20xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb39(%413: f32, %414: f32, %415: f32):
      %416 = arith.divf %413, %414 : f32
      linalg.yield %416 : f32
    } -> tensor<1x20xf32>
    %417 = tensor.collapse_shape %412 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x20xf32> into tensor<20xf32>
    %418 = tensor.expand_shape %417 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<20xf32> into tensor<1x20x1xf32>
    %419 = arith.constant {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %420 = tensor.splat %419 {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x20x1xf32>
    %421 = tensor.empty() : tensor<1x20x1xf32>
    %422 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%418, %420 : tensor<1x20x1xf32>, tensor<1x20x1xf32>) outs(%421 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb40(%423: f32, %424: f32, %425: f32):
      %426 = arith.addf %423, %424 : f32
      linalg.yield %426 : f32
    } -> tensor<1x20x1xf32>
    %427 = tensor.empty() : tensor<1x20x1xf32>
    %428 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%422 : tensor<1x20x1xf32>) outs(%427 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb41(%429: f32, %430: f32):
      %431 = math.rsqrt %429 : f32
      linalg.yield %431 : f32
    } -> tensor<1x20x1xf32>
    %432 = tensor.empty() : tensor<1x20x128xf32>
    %433 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%392, %428 : tensor<1x20x128xf32>, tensor<1x20x1xf32>) outs(%432 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb42(%434: f32, %435: f32, %436: f32):
      %437 = arith.mulf %434, %435 : f32
      linalg.yield %437 : f32
    } -> tensor<1x20x128xf32>
    %438 = tensor.empty() : tensor<1x20x128xf32>
    %439 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %433 : tensor<128xf32>, tensor<1x20x128xf32>) outs(%438 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb43(%440: f32, %441: f32, %442: f32):
      %443 = arith.mulf %440, %441 : f32
      linalg.yield %443 : f32
    } -> tensor<1x20x128xf32>
    %444 = tensor.empty() : tensor<128x256xf32>
    %445 = linalg.transpose ins(%25:tensor<256x128xf32>) outs(%444:tensor<128x256xf32>) permutation = [1, 0]
    %446 = tensor.empty() : tensor<1x20x256xf32>
    %447 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %448 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%447 : f32) outs(%446 : tensor<1x20x256xf32>) -> tensor<1x20x256xf32>
    %449 = linalg.matmul {prov.region_id = "matmul_7", prov.dispatch_id = "matmul_7", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} ins(%439, %445 : tensor<1x20x128xf32>, tensor<128x256xf32>) outs(%448 : tensor<1x20x256xf32>) -> tensor<1x20x256xf32>
    %450 = tensor.empty() : tensor<1x20x256xf32>
    %451 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%449 : tensor<1x20x256xf32>) outs(%450 : tensor<1x20x256xf32>) attrs =  {prov.region_id = "silu_0", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
    ^bb44(%452: f32, %453: f32):
      %454 = arith.constant 1.000000e+00 : f32
      %455 = arith.negf %452 : f32
      %456 = math.exp %455 : f32
      %457 = arith.addf %454, %456 : f32
      %458 = arith.divf %454, %457 : f32
      %459 = arith.mulf %452, %458 : f32
      linalg.yield %459 : f32
    } -> tensor<1x20x256xf32>
    %460 = tensor.empty() : tensor<128x256xf32>
    %461 = linalg.transpose ins(%26:tensor<256x128xf32>) outs(%460:tensor<128x256xf32>) permutation = [1, 0]
    %462 = tensor.empty() : tensor<1x20x256xf32>
    %463 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %464 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%463 : f32) outs(%462 : tensor<1x20x256xf32>) -> tensor<1x20x256xf32>
    %465 = linalg.matmul {prov.region_id = "matmul_8", prov.dispatch_id = "matmul_8", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} ins(%439, %461 : tensor<1x20x128xf32>, tensor<128x256xf32>) outs(%464 : tensor<1x20x256xf32>) -> tensor<1x20x256xf32>
    %466 = tensor.empty() : tensor<1x20x256xf32>
    %467 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%451, %465 : tensor<1x20x256xf32>, tensor<1x20x256xf32>) outs(%466 : tensor<1x20x256xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb45(%468: f32, %469: f32, %470: f32):
      %471 = arith.mulf %468, %469 : f32
      linalg.yield %471 : f32
    } -> tensor<1x20x256xf32>
    %472 = tensor.empty() : tensor<256x128xf32>
    %473 = linalg.transpose ins(%27:tensor<128x256xf32>) outs(%472:tensor<256x128xf32>) permutation = [1, 0]
    %474 = tensor.empty() : tensor<1x20x128xf32>
    %475 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %476 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%475 : f32) outs(%474 : tensor<1x20x128xf32>) -> tensor<1x20x128xf32>
    %477 = linalg.matmul {prov.region_id = "matmul_9", prov.dispatch_id = "matmul_9", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} ins(%467, %473 : tensor<1x20x256xf32>, tensor<256x128xf32>) outs(%476 : tensor<1x20x128xf32>) -> tensor<1x20x128xf32>
    %478 = tensor.empty() : tensor<1x20x128xf32>
    %479 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%392, %477 : tensor<1x20x128xf32>, tensor<1x20x128xf32>) outs(%478 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb46(%480: f32, %481: f32, %482: f32):
      %483 = arith.addf %480, %481 : f32
      linalg.yield %483 : f32
    } -> tensor<1x20x128xf32>
    %484 = tensor.empty() : tensor<1x20x128xf32>
    %485 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%479 : tensor<1x20x128xf32>) outs(%484 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb47(%486: f32, %487: f32):
      %488 = arith.constant 2.000000e+00 : f32
      %489 = math.powf %486, %488 : f32
      linalg.yield %489 : f32
    } -> tensor<1x20x128xf32>
    %490 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %491 = tensor.splat %490 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x20xf32>
    %492 = linalg.reduce ins(%485:tensor<1x20x128xf32>) outs(%491:tensor<1x20xf32>) dimensions = [2]
    (%493: f32, %494: f32) {
      %495 = arith.addf %493, %494 : f32
      linalg.yield %495 : f32
    }
    %496 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
    %497 = tensor.splat %496 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x20xf32>
    %498 = tensor.empty() : tensor<1x20xf32>
    %499 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%492, %497 : tensor<1x20xf32>, tensor<1x20xf32>) outs(%498 : tensor<1x20xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb48(%500: f32, %501: f32, %502: f32):
      %503 = arith.divf %500, %501 : f32
      linalg.yield %503 : f32
    } -> tensor<1x20xf32>
    %504 = tensor.collapse_shape %499 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x20xf32> into tensor<20xf32>
    %505 = tensor.expand_shape %504 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<20xf32> into tensor<1x20x1xf32>
    %506 = arith.constant {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %507 = tensor.splat %506 {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x20x1xf32>
    %508 = tensor.empty() : tensor<1x20x1xf32>
    %509 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%505, %507 : tensor<1x20x1xf32>, tensor<1x20x1xf32>) outs(%508 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb49(%510: f32, %511: f32, %512: f32):
      %513 = arith.addf %510, %511 : f32
      linalg.yield %513 : f32
    } -> tensor<1x20x1xf32>
    %514 = tensor.empty() : tensor<1x20x1xf32>
    %515 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%509 : tensor<1x20x1xf32>) outs(%514 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb50(%516: f32, %517: f32):
      %518 = math.rsqrt %516 : f32
      linalg.yield %518 : f32
    } -> tensor<1x20x1xf32>
    %519 = tensor.empty() : tensor<1x20x128xf32>
    %520 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%479, %515 : tensor<1x20x128xf32>, tensor<1x20x1xf32>) outs(%519 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb51(%521: f32, %522: f32, %523: f32):
      %524 = arith.mulf %521, %522 : f32
      linalg.yield %524 : f32
    } -> tensor<1x20x128xf32>
    %525 = tensor.empty() : tensor<1x20x128xf32>
    %526 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %520 : tensor<128xf32>, tensor<1x20x128xf32>) outs(%525 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb52(%527: f32, %528: f32, %529: f32):
      %530 = arith.mulf %527, %528 : f32
      linalg.yield %530 : f32
    } -> tensor<1x20x128xf32>
    %531 = tensor.empty() : tensor<128x512xf32>
    %532 = linalg.transpose ins(%30:tensor<512x128xf32>) outs(%531:tensor<128x512xf32>) permutation = [1, 0]
    %533 = tensor.empty() : tensor<1x20x512xf32>
    %534 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %535 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%534 : f32) outs(%533 : tensor<1x20x512xf32>) -> tensor<1x20x512xf32>
    %536 = linalg.matmul {prov.region_id = "matmul_10", prov.dispatch_id = "matmul_10", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} ins(%526, %532 : tensor<1x20x128xf32>, tensor<128x512xf32>) outs(%535 : tensor<1x20x512xf32>) -> tensor<1x20x512xf32>
    %537 = tensor.collapse_shape %536 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x20x512xf32> into tensor<10240xf32>
    %538 = tensor.expand_shape %537 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 20, 4, 128] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<10240xf32> into tensor<1x20x4x128xf32>
    %539 = tensor.empty() : tensor<1x4x20x128xf32>
    %540 = linalg.transpose ins(%538:tensor<1x20x4x128xf32>) outs(%539:tensor<1x4x20x128xf32>) permutation = [0, 2, 1, 3]
    %541 = tensor.empty() : tensor<128x512xf32>
    %542 = linalg.transpose ins(%31:tensor<512x128xf32>) outs(%541:tensor<128x512xf32>) permutation = [1, 0]
    %543 = tensor.empty() : tensor<1x20x512xf32>
    %544 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %545 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%544 : f32) outs(%543 : tensor<1x20x512xf32>) -> tensor<1x20x512xf32>
    %546 = linalg.matmul {prov.region_id = "matmul_11", prov.dispatch_id = "matmul_11", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} ins(%526, %542 : tensor<1x20x128xf32>, tensor<128x512xf32>) outs(%545 : tensor<1x20x512xf32>) -> tensor<1x20x512xf32>
    %547 = tensor.collapse_shape %546 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x20x512xf32> into tensor<10240xf32>
    %548 = tensor.expand_shape %547 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 20, 4, 128] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<10240xf32> into tensor<1x20x4x128xf32>
    %549 = tensor.empty() : tensor<1x4x20x128xf32>
    %550 = linalg.transpose ins(%548:tensor<1x20x4x128xf32>) outs(%549:tensor<1x4x20x128xf32>) permutation = [0, 2, 1, 3]
    %551 = tensor.empty() : tensor<128x512xf32>
    %552 = linalg.transpose ins(%32:tensor<512x128xf32>) outs(%551:tensor<128x512xf32>) permutation = [1, 0]
    %553 = tensor.empty() : tensor<1x20x512xf32>
    %554 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %555 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%554 : f32) outs(%553 : tensor<1x20x512xf32>) -> tensor<1x20x512xf32>
    %556 = linalg.matmul {prov.region_id = "matmul_12", prov.dispatch_id = "matmul_12", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} ins(%526, %552 : tensor<1x20x128xf32>, tensor<128x512xf32>) outs(%555 : tensor<1x20x512xf32>) -> tensor<1x20x512xf32>
    %557 = tensor.collapse_shape %556 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x20x512xf32> into tensor<10240xf32>
    %558 = tensor.expand_shape %557 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 20, 4, 128] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<10240xf32> into tensor<1x20x4x128xf32>
    %559 = tensor.empty() : tensor<1x4x20x128xf32>
    %560 = linalg.transpose ins(%558:tensor<1x20x4x128xf32>) outs(%559:tensor<1x4x20x128xf32>) permutation = [0, 2, 1, 3]
    %561 = tensor.collapse_shape %94 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %562 = tensor.expand_shape %561 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 20, 128] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<2560xf32> into tensor<1x1x20x128xf32>
    %563 = tensor.collapse_shape %107 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x20x128xf32> into tensor<2560xf32>
    %564 = tensor.expand_shape %563 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 20, 128] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<2560xf32> into tensor<1x1x20x128xf32>
    %565 = tensor.empty() : tensor<1x4x20x128xf32>
    %566 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%540, %562 : tensor<1x4x20x128xf32>, tensor<1x1x20x128xf32>) outs(%565 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb53(%567: f32, %568: f32, %569: f32):
      %570 = arith.mulf %567, %568 : f32
      linalg.yield %570 : f32
    } -> tensor<1x4x20x128xf32>
    %571 = "tensor.extract_slice"(%540) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 20, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x20x128xf32>) -> tensor<1x4x20x64xf32>
    %572 = "tensor.extract_slice"(%540) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 20, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x20x128xf32>) -> tensor<1x4x20x64xf32>
    %573 = tensor.empty() : tensor<1x4x20x64xf32>
    %574 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%572 : tensor<1x4x20x64xf32>) outs(%573 : tensor<1x4x20x64xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb54(%575: f32, %576: f32):
      %577 = arith.negf %575 : f32
      linalg.yield %577 : f32
    } -> tensor<1x4x20x64xf32>
    %578 = tensor.concat dim(3) %574, %571 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x20x64xf32>, tensor<1x4x20x64xf32>) -> tensor<1x4x20x128xf32>
    %579 = tensor.empty() : tensor<1x4x20x128xf32>
    %580 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%578, %564 : tensor<1x4x20x128xf32>, tensor<1x1x20x128xf32>) outs(%579 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb55(%581: f32, %582: f32, %583: f32):
      %584 = arith.mulf %581, %582 : f32
      linalg.yield %584 : f32
    } -> tensor<1x4x20x128xf32>
    %585 = tensor.empty() : tensor<1x4x20x128xf32>
    %586 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%566, %580 : tensor<1x4x20x128xf32>, tensor<1x4x20x128xf32>) outs(%585 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb56(%587: f32, %588: f32, %589: f32):
      %590 = arith.addf %587, %588 : f32
      linalg.yield %590 : f32
    } -> tensor<1x4x20x128xf32>
    %591 = tensor.empty() : tensor<1x4x20x128xf32>
    %592 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%550, %562 : tensor<1x4x20x128xf32>, tensor<1x1x20x128xf32>) outs(%591 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb57(%593: f32, %594: f32, %595: f32):
      %596 = arith.mulf %593, %594 : f32
      linalg.yield %596 : f32
    } -> tensor<1x4x20x128xf32>
    %597 = "tensor.extract_slice"(%550) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 20, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x20x128xf32>) -> tensor<1x4x20x64xf32>
    %598 = "tensor.extract_slice"(%550) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 20, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x20x128xf32>) -> tensor<1x4x20x64xf32>
    %599 = tensor.empty() : tensor<1x4x20x64xf32>
    %600 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%598 : tensor<1x4x20x64xf32>) outs(%599 : tensor<1x4x20x64xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb58(%601: f32, %602: f32):
      %603 = arith.negf %601 : f32
      linalg.yield %603 : f32
    } -> tensor<1x4x20x64xf32>
    %604 = tensor.concat dim(3) %600, %597 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x20x64xf32>, tensor<1x4x20x64xf32>) -> tensor<1x4x20x128xf32>
    %605 = tensor.empty() : tensor<1x4x20x128xf32>
    %606 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%604, %564 : tensor<1x4x20x128xf32>, tensor<1x1x20x128xf32>) outs(%605 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb59(%607: f32, %608: f32, %609: f32):
      %610 = arith.mulf %607, %608 : f32
      linalg.yield %610 : f32
    } -> tensor<1x4x20x128xf32>
    %611 = tensor.empty() : tensor<1x4x20x128xf32>
    %612 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%592, %606 : tensor<1x4x20x128xf32>, tensor<1x4x20x128xf32>) outs(%611 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb60(%613: f32, %614: f32, %615: f32):
      %616 = arith.addf %613, %614 : f32
      linalg.yield %616 : f32
    } -> tensor<1x4x20x128xf32>
    %617 = "tensor.extract_slice"(%48) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 27, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x27x128xf32>) -> tensor<4x27x128xf32>
    %618 = tensor.empty() : tensor<20xi64>
    %619 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%618 : tensor<20xi64>) attrs =  {prov.region_id = "iota_5", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
    ^bb61(%620: i64):
      %621 = linalg.index 0 : index
      %622 = arith.index_cast %621 : index to i64
      %623 = arith.constant 1 : i64
      %624 = arith.muli %622, %623 : i64
      %625 = arith.constant 0 : i64
      %626 = arith.addi %625, %624 : i64
      linalg.yield %626 : i64
    } -> tensor<20xi64>
    %627 = tensor.empty() : tensor<20xi64>
    %628 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%50, %619 : tensor<i64>, tensor<20xi64>) outs(%627 : tensor<20xi64>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb62(%629: i64, %630: i64, %631: i64):
      %632 = arith.addi %629, %630 : i64
      linalg.yield %632 : i64
    } -> tensor<20xi64>
    %633 = func.call @aten_index_copy_default(%617, %628, %612) {prov.region_id = "aten_index_copy_default_2", prov.dispatch_id = "aten_index_copy_default_2"} : (tensor<4x27x128xf32>, tensor<20xi64>, tensor<1x4x20x128xf32>) -> tensor<1x4x27x128xf32>
    %634 = "tensor.extract_slice"(%49) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 27, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x27x128xf32>) -> tensor<4x27x128xf32>
    %635 = tensor.empty() : tensor<20xi64>
    %636 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%635 : tensor<20xi64>) attrs =  {prov.region_id = "iota_6", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
    ^bb63(%637: i64):
      %638 = linalg.index 0 : index
      %639 = arith.index_cast %638 : index to i64
      %640 = arith.constant 1 : i64
      %641 = arith.muli %639, %640 : i64
      %642 = arith.constant 0 : i64
      %643 = arith.addi %642, %641 : i64
      linalg.yield %643 : i64
    } -> tensor<20xi64>
    %644 = tensor.empty() : tensor<20xi64>
    %645 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%50, %636 : tensor<i64>, tensor<20xi64>) outs(%644 : tensor<20xi64>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb64(%646: i64, %647: i64, %648: i64):
      %649 = arith.addi %646, %647 : i64
      linalg.yield %649 : i64
    } -> tensor<20xi64>
    %650 = func.call @aten_index_copy_default(%634, %645, %560) {prov.region_id = "aten_index_copy_default_3", prov.dispatch_id = "aten_index_copy_default_3"} : (tensor<4x27x128xf32>, tensor<20xi64>, tensor<1x4x20x128xf32>) -> tensor<1x4x27x128xf32>
    %651 = tensor.empty() : tensor<1x4x128x27xf32>
    %652 = linalg.transpose ins(%633:tensor<1x4x27x128xf32>) outs(%651:tensor<1x4x128x27xf32>) permutation = [0, 1, 3, 2]
    %653 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %654 = tensor.splat %653 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x4x20x27xf32>
    %655 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%586, %652 : tensor<1x4x20x128xf32>, tensor<1x4x128x27xf32>) outs(%654 : tensor<1x4x20x27xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb65(%656: f32, %657: f32, %658: f32):
      %659 = arith.mulf %656, %657 : f32
      %660 = arith.addf %658, %659 : f32
      linalg.yield %660 : f32
    } -> tensor<1x4x20x27xf32>
    %661 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 11.3137083 : f32
    %662 = tensor.splat %661 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x4x20x27xf32>
    %663 = tensor.empty() : tensor<1x4x20x27xf32>
    %664 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%655, %662 : tensor<1x4x20x27xf32>, tensor<1x4x20x27xf32>) outs(%663 : tensor<1x4x20x27xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb66(%665: f32, %666: f32, %667: f32):
      %668 = arith.divf %665, %666 : f32
      linalg.yield %668 : f32
    } -> tensor<1x4x20x27xf32>
    %669 = tensor.empty() : tensor<27xi64>
    %670 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%669 : tensor<27xi64>) attrs =  {prov.region_id = "iota_7", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
    ^bb67(%671: i64):
      %672 = linalg.index 0 : index
      %673 = arith.index_cast %672 : index to i64
      %674 = arith.constant 1 : i64
      %675 = arith.muli %673, %674 : i64
      %676 = arith.constant 0 : i64
      %677 = arith.addi %676, %675 : i64
      linalg.yield %677 : i64
    } -> tensor<27xi64>
    %678 = tensor.empty() : tensor<20xi64>
    %679 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%678 : tensor<20xi64>) attrs =  {prov.region_id = "iota_8", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
    ^bb68(%680: i64):
      %681 = linalg.index 0 : index
      %682 = arith.index_cast %681 : index to i64
      %683 = arith.constant 1 : i64
      %684 = arith.muli %682, %683 : i64
      %685 = arith.constant 0 : i64
      %686 = arith.addi %685, %684 : i64
      linalg.yield %686 : i64
    } -> tensor<20xi64>
    %687 = tensor.empty() : tensor<20xi64>
    %688 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%50, %679 : tensor<i64>, tensor<20xi64>) outs(%687 : tensor<20xi64>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb69(%689: i64, %690: i64, %691: i64):
      %692 = arith.addi %689, %690 : i64
      linalg.yield %692 : i64
    } -> tensor<20xi64>
    %693 = tensor.expand_shape %688 [[0 : i64, 1 : i64]] output_shape [20, 1] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<20xi64> into tensor<20x1xi64>
    %694 = tensor.expand_shape %670 [[0 : i64, 1 : i64]] output_shape [1, 27] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<27xi64> into tensor<1x27xi64>
    %695 = tensor.empty() : tensor<20x27xi1>
    %696 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%694, %693 : tensor<1x27xi64>, tensor<20x1xi64>) outs(%695 : tensor<20x27xi1>) attrs =  {prov.region_id = "compare_1", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
    ^bb70(%697: i64, %698: i64, %699: i1):
      %700 = arith.cmpi sle, %697, %698 : i64
      linalg.yield %700 : i1
    } -> tensor<20x27xi1>
    %701 = tensor.collapse_shape %696 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<20x27xi1> into tensor<540xi1>
    %702 = tensor.expand_shape %701 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 27] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<540xi1> into tensor<1x20x27xi1>
    %703 = tensor.collapse_shape %702 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x20x27xi1> into tensor<540xi1>
    %704 = tensor.expand_shape %703 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 20, 27] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<540xi1> into tensor<1x1x20x27xi1>
    %705 = tensor.empty() : tensor<1x1x20x27xi1>
    %706 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%704 : tensor<1x1x20x27xi1>) outs(%705 : tensor<1x1x20x27xi1>) attrs =  {prov.region_id = "bitwise_1", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
    ^bb71(%707: i1, %708: i1):
      %709 = arith.constant true
      %710 = arith.xori %707, %709 : i1
      linalg.yield %710 : i1
    } -> tensor<1x1x20x27xi1>
    %711 = func.call @aten_masked_fill_Scalar(%664, %706) {prov.region_id = "aten_masked_fill_Scalar_1", prov.dispatch_id = "aten_masked_fill_Scalar_1"} : (tensor<1x4x20x27xf32>, tensor<1x1x20x27xi1>) -> tensor<1x4x20x27xf32>
    %712 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0xff800000 : f32
    %713 = tensor.splat %712 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x20xf32>
    %714 = linalg.reduce ins(%711:tensor<1x4x20x27xf32>) outs(%713:tensor<1x4x20xf32>) dimensions = [3]
    (%715: f32, %716: f32) {
      %717 = arith.maximumf %715, %716 : f32
      linalg.yield %717 : f32
    }
    %718 = tensor.collapse_shape %714 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x20xf32> into tensor<80xf32>
    %719 = tensor.expand_shape %718 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 20, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<80xf32> into tensor<1x4x20x1xf32>
    %720 = tensor.empty() : tensor<1x4x20x27xf32>
    %721 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%711, %719 : tensor<1x4x20x27xf32>, tensor<1x4x20x1xf32>) outs(%720 : tensor<1x4x20x27xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb72(%722: f32, %723: f32, %724: f32):
      %725 = arith.subf %722, %723 : f32
      linalg.yield %725 : f32
    } -> tensor<1x4x20x27xf32>
    %726 = tensor.empty() : tensor<1x4x20x27xf32>
    %727 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%721 : tensor<1x4x20x27xf32>) outs(%726 : tensor<1x4x20x27xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb73(%728: f32, %729: f32):
      %730 = math.exp %728 : f32
      linalg.yield %730 : f32
    } -> tensor<1x4x20x27xf32>
    %731 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %732 = tensor.splat %731 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x20xf32>
    %733 = linalg.reduce ins(%727:tensor<1x4x20x27xf32>) outs(%732:tensor<1x4x20xf32>) dimensions = [3]
    (%734: f32, %735: f32) {
      %736 = arith.addf %734, %735 : f32
      linalg.yield %736 : f32
    }
    %737 = tensor.collapse_shape %733 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x20xf32> into tensor<80xf32>
    %738 = tensor.expand_shape %737 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 20, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<80xf32> into tensor<1x4x20x1xf32>
    %739 = tensor.empty() : tensor<1x4x20x27xf32>
    %740 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%727, %738 : tensor<1x4x20x27xf32>, tensor<1x4x20x1xf32>) outs(%739 : tensor<1x4x20x27xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb74(%741: f32, %742: f32, %743: f32):
      %744 = arith.divf %741, %742 : f32
      linalg.yield %744 : f32
    } -> tensor<1x4x20x27xf32>
    %745 = arith.constant {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %746 = tensor.splat %745 {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x4x20x128xf32>
    %747 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%740, %650 : tensor<1x4x20x27xf32>, tensor<1x4x27x128xf32>) outs(%746 : tensor<1x4x20x128xf32>) attrs =  {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb75(%748: f32, %749: f32, %750: f32):
      %751 = arith.mulf %748, %749 : f32
      %752 = arith.addf %750, %751 : f32
      linalg.yield %752 : f32
    } -> tensor<1x4x20x128xf32>
    %753 = tensor.empty() : tensor<1x20x4x128xf32>
    %754 = linalg.transpose ins(%747:tensor<1x4x20x128xf32>) outs(%753:tensor<1x20x4x128xf32>) permutation = [0, 2, 1, 3]
    %755 = tensor.collapse_shape %754 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x20x4x128xf32> into tensor<10240xf32>
    %756 = tensor.expand_shape %755 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 512] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<10240xf32> into tensor<1x20x512xf32>
    %757 = tensor.empty() : tensor<512x128xf32>
    %758 = linalg.transpose ins(%33:tensor<128x512xf32>) outs(%757:tensor<512x128xf32>) permutation = [1, 0]
    %759 = tensor.empty() : tensor<1x20x128xf32>
    %760 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %761 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%760 : f32) outs(%759 : tensor<1x20x128xf32>) -> tensor<1x20x128xf32>
    %762 = linalg.matmul {prov.region_id = "matmul_15", prov.dispatch_id = "matmul_15", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} ins(%756, %758 : tensor<1x20x512xf32>, tensor<512x128xf32>) outs(%761 : tensor<1x20x128xf32>) -> tensor<1x20x128xf32>
    %763 = tensor.empty() : tensor<1x20x128xf32>
    %764 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%479, %762 : tensor<1x20x128xf32>, tensor<1x20x128xf32>) outs(%763 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb76(%765: f32, %766: f32, %767: f32):
      %768 = arith.addf %765, %766 : f32
      linalg.yield %768 : f32
    } -> tensor<1x20x128xf32>
    %769 = tensor.empty() : tensor<1x20x128xf32>
    %770 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%764 : tensor<1x20x128xf32>) outs(%769 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb77(%771: f32, %772: f32):
      %773 = arith.constant 2.000000e+00 : f32
      %774 = math.powf %771, %773 : f32
      linalg.yield %774 : f32
    } -> tensor<1x20x128xf32>
    %775 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %776 = tensor.splat %775 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x20xf32>
    %777 = linalg.reduce ins(%770:tensor<1x20x128xf32>) outs(%776:tensor<1x20xf32>) dimensions = [2]
    (%778: f32, %779: f32) {
      %780 = arith.addf %778, %779 : f32
      linalg.yield %780 : f32
    }
    %781 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
    %782 = tensor.splat %781 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x20xf32>
    %783 = tensor.empty() : tensor<1x20xf32>
    %784 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%777, %782 : tensor<1x20xf32>, tensor<1x20xf32>) outs(%783 : tensor<1x20xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb78(%785: f32, %786: f32, %787: f32):
      %788 = arith.divf %785, %786 : f32
      linalg.yield %788 : f32
    } -> tensor<1x20xf32>
    %789 = tensor.collapse_shape %784 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x20xf32> into tensor<20xf32>
    %790 = tensor.expand_shape %789 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 20, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<20xf32> into tensor<1x20x1xf32>
    %791 = arith.constant {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %792 = tensor.splat %791 {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x20x1xf32>
    %793 = tensor.empty() : tensor<1x20x1xf32>
    %794 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%790, %792 : tensor<1x20x1xf32>, tensor<1x20x1xf32>) outs(%793 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb79(%795: f32, %796: f32, %797: f32):
      %798 = arith.addf %795, %796 : f32
      linalg.yield %798 : f32
    } -> tensor<1x20x1xf32>
    %799 = tensor.empty() : tensor<1x20x1xf32>
    %800 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%794 : tensor<1x20x1xf32>) outs(%799 : tensor<1x20x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb80(%801: f32, %802: f32):
      %803 = math.rsqrt %801 : f32
      linalg.yield %803 : f32
    } -> tensor<1x20x1xf32>
    %804 = tensor.empty() : tensor<1x20x128xf32>
    %805 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%764, %800 : tensor<1x20x128xf32>, tensor<1x20x1xf32>) outs(%804 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb81(%806: f32, %807: f32, %808: f32):
      %809 = arith.mulf %806, %807 : f32
      linalg.yield %809 : f32
    } -> tensor<1x20x128xf32>
    %810 = tensor.empty() : tensor<1x20x128xf32>
    %811 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %805 : tensor<128xf32>, tensor<1x20x128xf32>) outs(%810 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb82(%812: f32, %813: f32, %814: f32):
      %815 = arith.mulf %812, %813 : f32
      linalg.yield %815 : f32
    } -> tensor<1x20x128xf32>
    %816 = tensor.empty() : tensor<128x256xf32>
    %817 = linalg.transpose ins(%34:tensor<256x128xf32>) outs(%816:tensor<128x256xf32>) permutation = [1, 0]
    %818 = tensor.empty() : tensor<1x20x256xf32>
    %819 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %820 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%819 : f32) outs(%818 : tensor<1x20x256xf32>) -> tensor<1x20x256xf32>
    %821 = linalg.matmul {prov.region_id = "matmul_16", prov.dispatch_id = "matmul_16", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} ins(%811, %817 : tensor<1x20x128xf32>, tensor<128x256xf32>) outs(%820 : tensor<1x20x256xf32>) -> tensor<1x20x256xf32>
    %822 = tensor.empty() : tensor<1x20x256xf32>
    %823 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%821 : tensor<1x20x256xf32>) outs(%822 : tensor<1x20x256xf32>) attrs =  {prov.region_id = "silu_1", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
    ^bb83(%824: f32, %825: f32):
      %826 = arith.constant 1.000000e+00 : f32
      %827 = arith.negf %824 : f32
      %828 = math.exp %827 : f32
      %829 = arith.addf %826, %828 : f32
      %830 = arith.divf %826, %829 : f32
      %831 = arith.mulf %824, %830 : f32
      linalg.yield %831 : f32
    } -> tensor<1x20x256xf32>
    %832 = tensor.empty() : tensor<128x256xf32>
    %833 = linalg.transpose ins(%35:tensor<256x128xf32>) outs(%832:tensor<128x256xf32>) permutation = [1, 0]
    %834 = tensor.empty() : tensor<1x20x256xf32>
    %835 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %836 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%835 : f32) outs(%834 : tensor<1x20x256xf32>) -> tensor<1x20x256xf32>
    %837 = linalg.matmul {prov.region_id = "matmul_17", prov.dispatch_id = "matmul_17", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} ins(%811, %833 : tensor<1x20x128xf32>, tensor<128x256xf32>) outs(%836 : tensor<1x20x256xf32>) -> tensor<1x20x256xf32>
    %838 = tensor.empty() : tensor<1x20x256xf32>
    %839 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%823, %837 : tensor<1x20x256xf32>, tensor<1x20x256xf32>) outs(%838 : tensor<1x20x256xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb84(%840: f32, %841: f32, %842: f32):
      %843 = arith.mulf %840, %841 : f32
      linalg.yield %843 : f32
    } -> tensor<1x20x256xf32>
    %844 = tensor.empty() : tensor<256x128xf32>
    %845 = linalg.transpose ins(%36:tensor<128x256xf32>) outs(%844:tensor<256x128xf32>) permutation = [1, 0]
    %846 = tensor.empty() : tensor<1x20x128xf32>
    %847 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %848 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%847 : f32) outs(%846 : tensor<1x20x128xf32>) -> tensor<1x20x128xf32>
    %849 = linalg.matmul {prov.region_id = "matmul_18", prov.dispatch_id = "matmul_18", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} ins(%839, %845 : tensor<1x20x256xf32>, tensor<256x128xf32>) outs(%848 : tensor<1x20x128xf32>) -> tensor<1x20x128xf32>
    %850 = tensor.empty() : tensor<1x20x128xf32>
    %851 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%764, %849 : tensor<1x20x128xf32>, tensor<1x20x128xf32>) outs(%850 : tensor<1x20x128xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb85(%852: f32, %853: f32, %854: f32):
      %855 = arith.addf %852, %853 : f32
      linalg.yield %855 : f32
    } -> tensor<1x20x128xf32>
    %856 = func.call @aten_stack_default(%261, %633) {prov.region_id = "aten_stack_default_0", prov.dispatch_id = "aten_stack_default_0"} : (tensor<1x4x27x128xf32>, tensor<1x4x27x128xf32>) -> tensor<2x1x4x27x128xf32>
    %857 = func.call @aten_stack_default(%278, %650) {prov.region_id = "aten_stack_default_1", prov.dispatch_id = "aten_stack_default_1"} : (tensor<1x4x27x128xf32>, tensor<1x4x27x128xf32>) -> tensor<2x1x4x27x128xf32>
    %858 = "tensor.extract_slice"(%851) <{static_offsets = array<i64: 0, 19, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x20x128xf32>) -> tensor<1x1x128xf32>
    %859 = tensor.empty() : tensor<1x1x128xf32>
    %860 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%858 : tensor<1x1x128xf32>) outs(%859 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} {
    ^bb86(%861: f32, %862: f32):
      %863 = arith.constant 2.000000e+00 : f32
      %864 = math.powf %861, %863 : f32
      linalg.yield %864 : f32
    } -> tensor<1x1x128xf32>
    %865 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} 0.000000e+00 : f32
    %866 = tensor.splat %865 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} : tensor<1x1xf32>
    %867 = linalg.reduce ins(%860:tensor<1x1x128xf32>) outs(%866:tensor<1x1xf32>) dimensions = [2]
    (%868: f32, %869: f32) {
      %870 = arith.addf %868, %869 : f32
      linalg.yield %870 : f32
    }
    %871 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} 1.280000e+02 : f32
    %872 = tensor.splat %871 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} : tensor<1x1xf32>
    %873 = tensor.empty() : tensor<1x1xf32>
    %874 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%867, %872 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%873 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} {
    ^bb87(%875: f32, %876: f32, %877: f32):
      %878 = arith.divf %875, %876 : f32
      linalg.yield %878 : f32
    } -> tensor<1x1xf32>
    %879 = tensor.collapse_shape %874 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} : tensor<1x1xf32> into tensor<1xf32>
    %880 = tensor.expand_shape %879 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} : tensor<1xf32> into tensor<1x1x1xf32>
    %881 = arith.constant {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} 1.000000e-06 : f32
    %882 = tensor.splat %881 {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} : tensor<1x1x1xf32>
    %883 = tensor.empty() : tensor<1x1x1xf32>
    %884 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%880, %882 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%883 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} {
    ^bb88(%885: f32, %886: f32, %887: f32):
      %888 = arith.addf %885, %886 : f32
      linalg.yield %888 : f32
    } -> tensor<1x1x1xf32>
    %889 = tensor.empty() : tensor<1x1x1xf32>
    %890 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%884 : tensor<1x1x1xf32>) outs(%889 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} {
    ^bb89(%891: f32, %892: f32):
      %893 = math.rsqrt %891 : f32
      linalg.yield %893 : f32
    } -> tensor<1x1x1xf32>
    %894 = tensor.empty() : tensor<1x1x128xf32>
    %895 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%858, %890 : tensor<1x1x128xf32>, tensor<1x1x1xf32>) outs(%894 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} {
    ^bb90(%896: f32, %897: f32, %898: f32):
      %899 = arith.mulf %896, %897 : f32
      linalg.yield %899 : f32
    } -> tensor<1x1x128xf32>
    %900 = tensor.empty() : tensor<1x1x128xf32>
    %901 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %895 : tensor<128xf32>, tensor<1x1x128xf32>) outs(%900 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} {
    ^bb91(%902: f32, %903: f32, %904: f32):
      %905 = arith.mulf %902, %903 : f32
      linalg.yield %905 : f32
    } -> tensor<1x1x128xf32>
    %906 = tensor.empty() : tensor<128x512xf32>
    %907 = linalg.transpose ins(%41:tensor<512x128xf32>) outs(%906:tensor<128x512xf32>) permutation = [1, 0]
    %908 = tensor.empty() : tensor<1x1x512xf32>
    %909 = arith.constant {prov.module = "lm_head"} 0.000000e+00 : f32
    %910 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm_head"} ins(%909 : f32) outs(%908 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
    %911 = linalg.matmul {prov.region_id = "matmul_19", prov.dispatch_id = "matmul_19", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "lm_head", prov.fqn = "lm_head"} ins(%901, %907 : tensor<1x1x128xf32>, tensor<128x512xf32>) outs(%910 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
    %912 = "tensor.extract_slice"(%911) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<1x1x512xf32>) -> tensor<512xf32>
    %913 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0xff800000 : f32
    %914 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0 : i64
    %915 = tensor.splat %913 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<f32>
    %916 = tensor.splat %914 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<i64>
    %917, %918 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%912 : tensor<512xf32>) outs(%915, %916 : tensor<f32>, tensor<i64>) attrs =  {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} {
    ^bb92(%919: f32, %920: f32, %921: i64):
      %922 = linalg.index 0 : index
      %923 = arith.index_cast %922 : index to i64
      %924 = arith.cmpf ogt, %919, %920 : f32
      %925 = arith.select %924, %919, %920 : f32
      %926 = arith.select %924, %923, %921 : i64
      linalg.yield %925, %926 : f32, i64
    } -> (tensor<f32>, tensor<i64>)
    %927 = tensor.extract %917[] : tensor<f32>
    %928 = tensor.from_elements %927 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xf32>
    %929 = tensor.extract %918[] : tensor<i64>
    %930 = tensor.from_elements %929 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xi64>
    %931 = func.call @aten_zeros_default_1() {prov.region_id = "aten_zeros_default_1_1", prov.dispatch_id = "aten_zeros_default_1_1"} : () -> tensor<i64>
    %932 = func.call @aten_zeros_default_2() {prov.region_id = "aten_zeros_default_2_0", prov.dispatch_id = "aten_zeros_default_2_0"} : () -> tensor<1x7xi64>
    %933 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 0 : index
    %934 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 7 : index
    %935 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 1 : index
    %936, %937, %938, %939, %940 = scf.for %941 = %933 to %934 step %935 iter_args(%942 = %931, %943 = %930, %944 = %932, %945 = %856, %946 = %857) -> (tensor<i64>, tensor<1xi64>, tensor<1x7xi64>, tensor<2x1x4x27x128xf32>, tensor<2x1x4x27x128xf32>) {
      %947 = tensor.extract %942[] : tensor<i64>
      %948 = tensor.from_elements %947 {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1xi64>
      %949 = func.call @aten_index_copy_default_wl0(%944, %948, %943) {prov.region_id = "aten_index_copy_default_0", prov.dispatch_id = "aten_index_copy_default_0"} : (tensor<1x7xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x7xi64>
      %950 = tensor.empty() : tensor<i64>
      %951 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%46, %942 : tensor<i64>, tensor<i64>) outs(%950 : tensor<i64>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb93(%952: i64, %953: i64, %954: i64):
        %955 = arith.addi %952, %953 : i64
        linalg.yield %955 : i64
      } -> tensor<i64>
      %956 = tensor.empty() : tensor<1x1x128xf32>
      %957 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%943 : tensor<1xi64>) outs(%956 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32"} {
      ^bb94(%958: i64, %959: f32):
        %960 = arith.index_cast %958 : i64 to index
        %961 = linalg.index 2 : index
        %962 = tensor.extract %40[%960, %961] : tensor<512x128xf32>
        linalg.yield %962 : f32
      } -> tensor<1x1x128xf32>
      %963 = tensor.extract %951[] : tensor<i64>
      %964 = tensor.from_elements %963 {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int64"} : tensor<1x1xi64>
      %965 = func.call @wrap_with_set_grad_enabled_wl1(%44, %964) {prov.region_id = "wrap_with_set_grad_enabled_0", prov.dispatch_id = "wrap_with_set_grad_enabled_0"} : (tensor<64xf32>, tensor<1x1xi64>) -> tensor<1x1x128xf32>
      %966 = tensor.empty() : tensor<1x1x128xf32>
      %967 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%957 : tensor<1x1x128xf32>) outs(%966 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb95(%968: f32, %969: f32):
        %970 = arith.constant 2.000000e+00 : f32
        %971 = math.powf %968, %970 : f32
        linalg.yield %971 : f32
      } -> tensor<1x1x128xf32>
      %972 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %973 = tensor.splat %972 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %974 = linalg.reduce ins(%967:tensor<1x1x128xf32>) outs(%973:tensor<1x1xf32>) dimensions = [2]
      (%975: f32, %976: f32) {
        %977 = arith.addf %975, %976 : f32
        linalg.yield %977 : f32
      }
      %978 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %979 = tensor.splat %978 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %980 = tensor.empty() : tensor<1x1xf32>
      %981 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%974, %979 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%980 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb96(%982: f32, %983: f32, %984: f32):
        %985 = arith.divf %982, %983 : f32
        linalg.yield %985 : f32
      } -> tensor<1x1xf32>
      %986 = tensor.collapse_shape %981 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %987 = tensor.expand_shape %986 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %988 = arith.constant {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %989 = tensor.splat %988 {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %990 = tensor.empty() : tensor<1x1x1xf32>
      %991 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%987, %989 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%990 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb97(%992: f32, %993: f32, %994: f32):
        %995 = arith.addf %992, %993 : f32
        linalg.yield %995 : f32
      } -> tensor<1x1x1xf32>
      %996 = tensor.empty() : tensor<1x1x1xf32>
      %997 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%991 : tensor<1x1x1xf32>) outs(%996 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb98(%998: f32, %999: f32):
        %1000 = math.rsqrt %998 : f32
        linalg.yield %1000 : f32
      } -> tensor<1x1x1xf32>
      %1001 = tensor.empty() : tensor<1x1x128xf32>
      %1002 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%957, %997 : tensor<1x1x128xf32>, tensor<1x1x1xf32>) outs(%1001 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb99(%1003: f32, %1004: f32, %1005: f32):
        %1006 = arith.mulf %1003, %1004 : f32
        linalg.yield %1006 : f32
      } -> tensor<1x1x128xf32>
      %1007 = tensor.empty() : tensor<1x1x128xf32>
      %1008 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %1002 : tensor<128xf32>, tensor<1x1x128xf32>) outs(%1007 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb100(%1009: f32, %1010: f32, %1011: f32):
        %1012 = arith.mulf %1009, %1010 : f32
        linalg.yield %1012 : f32
      } -> tensor<1x1x128xf32>
      %1013 = tensor.empty() : tensor<128x512xf32>
      %1014 = linalg.transpose ins(%21:tensor<512x128xf32>) outs(%1013:tensor<128x512xf32>) permutation = [1, 0]
      %1015 = tensor.empty() : tensor<1x1x512xf32>
      %1016 = arith.constant 0.000000e+00 : f32
      %1017 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1016 : f32) outs(%1015 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %1018 = linalg.matmul {prov.region_id = "matmul_0", prov.dispatch_id = "matmul_0", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1008, %1014 : tensor<1x1x128xf32>, tensor<128x512xf32>) outs(%1017 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %1019 = tensor.collapse_shape %1018 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x512xf32> into tensor<512xf32>
      %1020 = tensor.expand_shape %1019 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 128] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x4x128xf32>
      %1021 = tensor.empty() : tensor<1x4x1x128xf32>
      %1022 = linalg.transpose ins(%1020:tensor<1x1x4x128xf32>) outs(%1021:tensor<1x4x1x128xf32>) permutation = [0, 2, 1, 3]
      %1023 = tensor.empty() : tensor<128x512xf32>
      %1024 = linalg.transpose ins(%22:tensor<512x128xf32>) outs(%1023:tensor<128x512xf32>) permutation = [1, 0]
      %1025 = tensor.empty() : tensor<1x1x512xf32>
      %1026 = arith.constant 0.000000e+00 : f32
      %1027 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1026 : f32) outs(%1025 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %1028 = linalg.matmul {prov.region_id = "matmul_1", prov.dispatch_id = "matmul_1", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1008, %1024 : tensor<1x1x128xf32>, tensor<128x512xf32>) outs(%1027 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %1029 = tensor.collapse_shape %1028 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x512xf32> into tensor<512xf32>
      %1030 = tensor.expand_shape %1029 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 128] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x4x128xf32>
      %1031 = tensor.empty() : tensor<1x4x1x128xf32>
      %1032 = linalg.transpose ins(%1030:tensor<1x1x4x128xf32>) outs(%1031:tensor<1x4x1x128xf32>) permutation = [0, 2, 1, 3]
      %1033 = tensor.empty() : tensor<128x512xf32>
      %1034 = linalg.transpose ins(%23:tensor<512x128xf32>) outs(%1033:tensor<128x512xf32>) permutation = [1, 0]
      %1035 = tensor.empty() : tensor<1x1x512xf32>
      %1036 = arith.constant 0.000000e+00 : f32
      %1037 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1036 : f32) outs(%1035 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %1038 = linalg.matmul {prov.region_id = "matmul_2", prov.dispatch_id = "matmul_2", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1008, %1034 : tensor<1x1x128xf32>, tensor<128x512xf32>) outs(%1037 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %1039 = tensor.collapse_shape %1038 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x512xf32> into tensor<512xf32>
      %1040 = tensor.expand_shape %1039 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 128] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x4x128xf32>
      %1041 = tensor.empty() : tensor<1x4x1x128xf32>
      %1042 = linalg.transpose ins(%1040:tensor<1x1x4x128xf32>) outs(%1041:tensor<1x4x1x128xf32>) permutation = [0, 2, 1, 3]
      %1043 = tensor.collapse_shape %965 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %1044 = tensor.expand_shape %1043 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 128] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x1x128xf32>
      %1045 = func.call @aten_unsqueeze_default_wl2() {prov.region_id = "aten_unsqueeze_default_0", prov.dispatch_id = "aten_unsqueeze_default_0"} : () -> tensor<1x1x1x128xf32>
      %1046 = tensor.empty() : tensor<1x4x1x128xf32>
      %1047 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1022, %1044 : tensor<1x4x1x128xf32>, tensor<1x1x1x128xf32>) outs(%1046 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb101(%1048: f32, %1049: f32, %1050: f32):
        %1051 = arith.mulf %1048, %1049 : f32
        linalg.yield %1051 : f32
      } -> tensor<1x4x1x128xf32>
      %1052 = "tensor.extract_slice"(%1022) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x128xf32>) -> tensor<1x4x1x64xf32>
      %1053 = "tensor.extract_slice"(%1022) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x128xf32>) -> tensor<1x4x1x64xf32>
      %1054 = tensor.empty() : tensor<1x4x1x64xf32>
      %1055 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1053 : tensor<1x4x1x64xf32>) outs(%1054 : tensor<1x4x1x64xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb102(%1056: f32, %1057: f32):
        %1058 = arith.negf %1056 : f32
        linalg.yield %1058 : f32
      } -> tensor<1x4x1x64xf32>
      %1059 = tensor.concat dim(3) %1055, %1052 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>) -> tensor<1x4x1x128xf32>
      %1060 = tensor.empty() : tensor<1x4x1x128xf32>
      %1061 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1059, %1045 : tensor<1x4x1x128xf32>, tensor<1x1x1x128xf32>) outs(%1060 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb103(%1062: f32, %1063: f32, %1064: f32):
        %1065 = arith.mulf %1062, %1063 : f32
        linalg.yield %1065 : f32
      } -> tensor<1x4x1x128xf32>
      %1066 = tensor.empty() : tensor<1x4x1x128xf32>
      %1067 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1047, %1061 : tensor<1x4x1x128xf32>, tensor<1x4x1x128xf32>) outs(%1066 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb104(%1068: f32, %1069: f32, %1070: f32):
        %1071 = arith.addf %1068, %1069 : f32
        linalg.yield %1071 : f32
      } -> tensor<1x4x1x128xf32>
      %1072 = tensor.empty() : tensor<1x4x1x128xf32>
      %1073 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1032, %1044 : tensor<1x4x1x128xf32>, tensor<1x1x1x128xf32>) outs(%1072 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb105(%1074: f32, %1075: f32, %1076: f32):
        %1077 = arith.mulf %1074, %1075 : f32
        linalg.yield %1077 : f32
      } -> tensor<1x4x1x128xf32>
      %1078 = "tensor.extract_slice"(%1032) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x128xf32>) -> tensor<1x4x1x64xf32>
      %1079 = "tensor.extract_slice"(%1032) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x128xf32>) -> tensor<1x4x1x64xf32>
      %1080 = tensor.empty() : tensor<1x4x1x64xf32>
      %1081 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1079 : tensor<1x4x1x64xf32>) outs(%1080 : tensor<1x4x1x64xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb106(%1082: f32, %1083: f32):
        %1084 = arith.negf %1082 : f32
        linalg.yield %1084 : f32
      } -> tensor<1x4x1x64xf32>
      %1085 = tensor.concat dim(3) %1081, %1078 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>) -> tensor<1x4x1x128xf32>
      %1086 = tensor.empty() : tensor<1x4x1x128xf32>
      %1087 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1085, %1045 : tensor<1x4x1x128xf32>, tensor<1x1x1x128xf32>) outs(%1086 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb107(%1088: f32, %1089: f32, %1090: f32):
        %1091 = arith.mulf %1088, %1089 : f32
        linalg.yield %1091 : f32
      } -> tensor<1x4x1x128xf32>
      %1092 = tensor.empty() : tensor<1x4x1x128xf32>
      %1093 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1073, %1087 : tensor<1x4x1x128xf32>, tensor<1x4x1x128xf32>) outs(%1092 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb108(%1094: f32, %1095: f32, %1096: f32):
        %1097 = arith.addf %1094, %1095 : f32
        linalg.yield %1097 : f32
      } -> tensor<1x4x1x128xf32>
      %1098 = "tensor.extract_slice"(%945) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 27, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x27x128xf32>) -> tensor<4x27x128xf32>
      %1099 = tensor.empty() : tensor<1xi64>
      %1100 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1099 : tensor<1xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
      ^bb109(%1101: i64):
        %1102 = linalg.index 0 : index
        %1103 = arith.index_cast %1102 : index to i64
        %1104 = arith.constant 1 : i64
        %1105 = arith.muli %1103, %1104 : i64
        %1106 = arith.constant 0 : i64
        %1107 = arith.addi %1106, %1105 : i64
        linalg.yield %1107 : i64
      } -> tensor<1xi64>
      %1108 = tensor.empty() : tensor<1xi64>
      %1109 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%951, %1100 : tensor<i64>, tensor<1xi64>) outs(%1108 : tensor<1xi64>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb110(%1110: i64, %1111: i64, %1112: i64):
        %1113 = arith.addi %1110, %1111 : i64
        linalg.yield %1113 : i64
      } -> tensor<1xi64>
      %1114 = func.call @aten_index_copy_default_1_wl3(%1098, %1109, %1093) {prov.region_id = "aten_index_copy_default_1_0", prov.dispatch_id = "aten_index_copy_default_1_0"} : (tensor<4x27x128xf32>, tensor<1xi64>, tensor<1x4x1x128xf32>) -> tensor<1x4x27x128xf32>
      %1115 = "tensor.extract_slice"(%946) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 27, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x27x128xf32>) -> tensor<4x27x128xf32>
      %1116 = tensor.empty() : tensor<1xi64>
      %1117 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1116 : tensor<1xi64>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
      ^bb111(%1118: i64):
        %1119 = linalg.index 0 : index
        %1120 = arith.index_cast %1119 : index to i64
        %1121 = arith.constant 1 : i64
        %1122 = arith.muli %1120, %1121 : i64
        %1123 = arith.constant 0 : i64
        %1124 = arith.addi %1123, %1122 : i64
        linalg.yield %1124 : i64
      } -> tensor<1xi64>
      %1125 = tensor.empty() : tensor<1xi64>
      %1126 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%951, %1117 : tensor<i64>, tensor<1xi64>) outs(%1125 : tensor<1xi64>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb112(%1127: i64, %1128: i64, %1129: i64):
        %1130 = arith.addi %1127, %1128 : i64
        linalg.yield %1130 : i64
      } -> tensor<1xi64>
      %1131 = func.call @aten_index_copy_default_1_wl3(%1115, %1126, %1042) {prov.region_id = "aten_index_copy_default_1_1", prov.dispatch_id = "aten_index_copy_default_1_1"} : (tensor<4x27x128xf32>, tensor<1xi64>, tensor<1x4x1x128xf32>) -> tensor<1x4x27x128xf32>
      %1132 = tensor.empty() : tensor<1x4x128x27xf32>
      %1133 = linalg.transpose ins(%1114:tensor<1x4x27x128xf32>) outs(%1132:tensor<1x4x128x27xf32>) permutation = [0, 1, 3, 2]
      %1134 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1135 = tensor.splat %1134 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x4x1x27xf32>
      %1136 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1067, %1133 : tensor<1x4x1x128xf32>, tensor<1x4x128x27xf32>) outs(%1135 : tensor<1x4x1x27xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb113(%1137: f32, %1138: f32, %1139: f32):
        %1140 = arith.mulf %1137, %1138 : f32
        %1141 = arith.addf %1139, %1140 : f32
        linalg.yield %1141 : f32
      } -> tensor<1x4x1x27xf32>
      %1142 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 11.3137083 : f32
      %1143 = tensor.splat %1142 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x4x1x27xf32>
      %1144 = tensor.empty() : tensor<1x4x1x27xf32>
      %1145 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1136, %1143 : tensor<1x4x1x27xf32>, tensor<1x4x1x27xf32>) outs(%1144 : tensor<1x4x1x27xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
      ^bb114(%1146: f32, %1147: f32, %1148: f32):
        %1149 = arith.divf %1146, %1147 : f32
        linalg.yield %1149 : f32
      } -> tensor<1x4x1x27xf32>
      %1150 = tensor.empty() : tensor<27xi64>
      %1151 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1150 : tensor<27xi64>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
      ^bb115(%1152: i64):
        %1153 = linalg.index 0 : index
        %1154 = arith.index_cast %1153 : index to i64
        %1155 = arith.constant 1 : i64
        %1156 = arith.muli %1154, %1155 : i64
        %1157 = arith.constant 0 : i64
        %1158 = arith.addi %1157, %1156 : i64
        linalg.yield %1158 : i64
      } -> tensor<27xi64>
      %1159 = tensor.empty() : tensor<1xi64>
      %1160 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1159 : tensor<1xi64>) attrs =  {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
      ^bb116(%1161: i64):
        %1162 = linalg.index 0 : index
        %1163 = arith.index_cast %1162 : index to i64
        %1164 = arith.constant 1 : i64
        %1165 = arith.muli %1163, %1164 : i64
        %1166 = arith.constant 0 : i64
        %1167 = arith.addi %1166, %1165 : i64
        linalg.yield %1167 : i64
      } -> tensor<1xi64>
      %1168 = tensor.empty() : tensor<1xi64>
      %1169 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%951, %1160 : tensor<i64>, tensor<1xi64>) outs(%1168 : tensor<1xi64>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb117(%1170: i64, %1171: i64, %1172: i64):
        %1173 = arith.addi %1170, %1171 : i64
        linalg.yield %1173 : i64
      } -> tensor<1xi64>
      %1174 = tensor.expand_shape %1169 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1xi64> into tensor<1x1xi64>
      %1175 = tensor.expand_shape %1151 [[0 : i64, 1 : i64]] output_shape [1, 27] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<27xi64> into tensor<1x27xi64>
      %1176 = tensor.empty() : tensor<1x27xi1>
      %1177 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1175, %1174 : tensor<1x27xi64>, tensor<1x1xi64>) outs(%1176 : tensor<1x27xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
      ^bb118(%1178: i64, %1179: i64, %1180: i1):
        %1181 = arith.cmpi sle, %1178, %1179 : i64
        linalg.yield %1181 : i1
      } -> tensor<1x27xi1>
      %1182 = tensor.collapse_shape %1177 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x27xi1> into tensor<27xi1>
      %1183 = tensor.expand_shape %1182 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 27] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<27xi1> into tensor<1x1x27xi1>
      %1184 = tensor.collapse_shape %1183 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x1x27xi1> into tensor<27xi1>
      %1185 = tensor.expand_shape %1184 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 27] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<27xi1> into tensor<1x1x1x27xi1>
      %1186 = tensor.empty() : tensor<1x1x1x27xi1>
      %1187 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1185 : tensor<1x1x1x27xi1>) outs(%1186 : tensor<1x1x1x27xi1>) attrs =  {prov.region_id = "bitwise_0", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
      ^bb119(%1188: i1, %1189: i1):
        %1190 = arith.constant true
        %1191 = arith.xori %1188, %1190 : i1
        linalg.yield %1191 : i1
      } -> tensor<1x1x1x27xi1>
      %1192 = func.call @aten_masked_fill_Scalar_wl4(%1145, %1187) {prov.region_id = "aten_masked_fill_Scalar_0", prov.dispatch_id = "aten_masked_fill_Scalar_0"} : (tensor<1x4x1x27xf32>, tensor<1x1x1x27xi1>) -> tensor<1x4x1x27xf32>
      %1193 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0xff800000 : f32
      %1194 = tensor.splat %1193 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x1xf32>
      %1195 = linalg.reduce ins(%1192:tensor<1x4x1x27xf32>) outs(%1194:tensor<1x4x1xf32>) dimensions = [3]
      (%1196: f32, %1197: f32) {
        %1198 = arith.maximumf %1196, %1197 : f32
        linalg.yield %1198 : f32
      }
      %1199 = tensor.collapse_shape %1195 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x1xf32> into tensor<4xf32>
      %1200 = tensor.expand_shape %1199 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 1, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<4xf32> into tensor<1x4x1x1xf32>
      %1201 = tensor.empty() : tensor<1x4x1x27xf32>
      %1202 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1192, %1200 : tensor<1x4x1x27xf32>, tensor<1x4x1x1xf32>) outs(%1201 : tensor<1x4x1x27xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb120(%1203: f32, %1204: f32, %1205: f32):
        %1206 = arith.subf %1203, %1204 : f32
        linalg.yield %1206 : f32
      } -> tensor<1x4x1x27xf32>
      %1207 = tensor.empty() : tensor<1x4x1x27xf32>
      %1208 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1202 : tensor<1x4x1x27xf32>) outs(%1207 : tensor<1x4x1x27xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb121(%1209: f32, %1210: f32):
        %1211 = math.exp %1209 : f32
        linalg.yield %1211 : f32
      } -> tensor<1x4x1x27xf32>
      %1212 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1213 = tensor.splat %1212 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x1xf32>
      %1214 = linalg.reduce ins(%1208:tensor<1x4x1x27xf32>) outs(%1213:tensor<1x4x1xf32>) dimensions = [3]
      (%1215: f32, %1216: f32) {
        %1217 = arith.addf %1215, %1216 : f32
        linalg.yield %1217 : f32
      }
      %1218 = tensor.collapse_shape %1214 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x1xf32> into tensor<4xf32>
      %1219 = tensor.expand_shape %1218 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 1, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<4xf32> into tensor<1x4x1x1xf32>
      %1220 = tensor.empty() : tensor<1x4x1x27xf32>
      %1221 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1208, %1219 : tensor<1x4x1x27xf32>, tensor<1x4x1x1xf32>) outs(%1220 : tensor<1x4x1x27xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb122(%1222: f32, %1223: f32, %1224: f32):
        %1225 = arith.divf %1222, %1223 : f32
        linalg.yield %1225 : f32
      } -> tensor<1x4x1x27xf32>
      %1226 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1227 = tensor.splat %1226 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x4x1x128xf32>
      %1228 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1221, %1131 : tensor<1x4x1x27xf32>, tensor<1x4x27x128xf32>) outs(%1227 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb123(%1229: f32, %1230: f32, %1231: f32):
        %1232 = arith.mulf %1229, %1230 : f32
        %1233 = arith.addf %1231, %1232 : f32
        linalg.yield %1233 : f32
      } -> tensor<1x4x1x128xf32>
      %1234 = tensor.empty() : tensor<1x1x4x128xf32>
      %1235 = linalg.transpose ins(%1228:tensor<1x4x1x128xf32>) outs(%1234:tensor<1x1x4x128xf32>) permutation = [0, 2, 1, 3]
      %1236 = tensor.collapse_shape %1235 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x1x4x128xf32> into tensor<512xf32>
      %1237 = tensor.expand_shape %1236 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x512xf32>
      %1238 = tensor.empty() : tensor<512x128xf32>
      %1239 = linalg.transpose ins(%24:tensor<128x512xf32>) outs(%1238:tensor<512x128xf32>) permutation = [1, 0]
      %1240 = tensor.empty() : tensor<1x1x128xf32>
      %1241 = arith.constant 0.000000e+00 : f32
      %1242 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1241 : f32) outs(%1240 : tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
      %1243 = linalg.matmul {prov.region_id = "matmul_5", prov.dispatch_id = "matmul_5", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1237, %1239 : tensor<1x1x512xf32>, tensor<512x128xf32>) outs(%1242 : tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
      %1244 = tensor.empty() : tensor<1x1x128xf32>
      %1245 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%957, %1243 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%1244 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb124(%1246: f32, %1247: f32, %1248: f32):
        %1249 = arith.addf %1246, %1247 : f32
        linalg.yield %1249 : f32
      } -> tensor<1x1x128xf32>
      %1250 = tensor.empty() : tensor<1x1x128xf32>
      %1251 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1245 : tensor<1x1x128xf32>) outs(%1250 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb125(%1252: f32, %1253: f32):
        %1254 = arith.constant 2.000000e+00 : f32
        %1255 = math.powf %1252, %1254 : f32
        linalg.yield %1255 : f32
      } -> tensor<1x1x128xf32>
      %1256 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1257 = tensor.splat %1256 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1258 = linalg.reduce ins(%1251:tensor<1x1x128xf32>) outs(%1257:tensor<1x1xf32>) dimensions = [2]
      (%1259: f32, %1260: f32) {
        %1261 = arith.addf %1259, %1260 : f32
        linalg.yield %1261 : f32
      }
      %1262 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %1263 = tensor.splat %1262 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1264 = tensor.empty() : tensor<1x1xf32>
      %1265 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1258, %1263 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%1264 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb126(%1266: f32, %1267: f32, %1268: f32):
        %1269 = arith.divf %1266, %1267 : f32
        linalg.yield %1269 : f32
      } -> tensor<1x1xf32>
      %1270 = tensor.collapse_shape %1265 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %1271 = tensor.expand_shape %1270 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %1272 = arith.constant {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %1273 = tensor.splat %1272 {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %1274 = tensor.empty() : tensor<1x1x1xf32>
      %1275 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1271, %1273 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%1274 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb127(%1276: f32, %1277: f32, %1278: f32):
        %1279 = arith.addf %1276, %1277 : f32
        linalg.yield %1279 : f32
      } -> tensor<1x1x1xf32>
      %1280 = tensor.empty() : tensor<1x1x1xf32>
      %1281 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1275 : tensor<1x1x1xf32>) outs(%1280 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb128(%1282: f32, %1283: f32):
        %1284 = math.rsqrt %1282 : f32
        linalg.yield %1284 : f32
      } -> tensor<1x1x1xf32>
      %1285 = tensor.empty() : tensor<1x1x128xf32>
      %1286 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1245, %1281 : tensor<1x1x128xf32>, tensor<1x1x1xf32>) outs(%1285 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb129(%1287: f32, %1288: f32, %1289: f32):
        %1290 = arith.mulf %1287, %1288 : f32
        linalg.yield %1290 : f32
      } -> tensor<1x1x128xf32>
      %1291 = tensor.empty() : tensor<1x1x128xf32>
      %1292 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %1286 : tensor<128xf32>, tensor<1x1x128xf32>) outs(%1291 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb130(%1293: f32, %1294: f32, %1295: f32):
        %1296 = arith.mulf %1293, %1294 : f32
        linalg.yield %1296 : f32
      } -> tensor<1x1x128xf32>
      %1297 = tensor.empty() : tensor<128x256xf32>
      %1298 = linalg.transpose ins(%25:tensor<256x128xf32>) outs(%1297:tensor<128x256xf32>) permutation = [1, 0]
      %1299 = tensor.empty() : tensor<1x1x256xf32>
      %1300 = arith.constant 0.000000e+00 : f32
      %1301 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1300 : f32) outs(%1299 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %1302 = linalg.matmul {prov.region_id = "matmul_6", prov.dispatch_id = "matmul_6", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1292, %1298 : tensor<1x1x128xf32>, tensor<128x256xf32>) outs(%1301 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %1303 = tensor.empty() : tensor<1x1x256xf32>
      %1304 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1302 : tensor<1x1x256xf32>) outs(%1303 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "silu_0", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb131(%1305: f32, %1306: f32):
        %1307 = arith.constant 1.000000e+00 : f32
        %1308 = arith.negf %1305 : f32
        %1309 = math.exp %1308 : f32
        %1310 = arith.addf %1307, %1309 : f32
        %1311 = arith.divf %1307, %1310 : f32
        %1312 = arith.mulf %1305, %1311 : f32
        linalg.yield %1312 : f32
      } -> tensor<1x1x256xf32>
      %1313 = tensor.empty() : tensor<128x256xf32>
      %1314 = linalg.transpose ins(%26:tensor<256x128xf32>) outs(%1313:tensor<128x256xf32>) permutation = [1, 0]
      %1315 = tensor.empty() : tensor<1x1x256xf32>
      %1316 = arith.constant 0.000000e+00 : f32
      %1317 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1316 : f32) outs(%1315 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %1318 = linalg.matmul {prov.region_id = "matmul_7", prov.dispatch_id = "matmul_7", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1292, %1314 : tensor<1x1x128xf32>, tensor<128x256xf32>) outs(%1317 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %1319 = tensor.empty() : tensor<1x1x256xf32>
      %1320 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1304, %1318 : tensor<1x1x256xf32>, tensor<1x1x256xf32>) outs(%1319 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb132(%1321: f32, %1322: f32, %1323: f32):
        %1324 = arith.mulf %1321, %1322 : f32
        linalg.yield %1324 : f32
      } -> tensor<1x1x256xf32>
      %1325 = tensor.empty() : tensor<256x128xf32>
      %1326 = linalg.transpose ins(%27:tensor<128x256xf32>) outs(%1325:tensor<256x128xf32>) permutation = [1, 0]
      %1327 = tensor.empty() : tensor<1x1x128xf32>
      %1328 = arith.constant 0.000000e+00 : f32
      %1329 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1328 : f32) outs(%1327 : tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
      %1330 = linalg.matmul {prov.region_id = "matmul_8", prov.dispatch_id = "matmul_8", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1320, %1326 : tensor<1x1x256xf32>, tensor<256x128xf32>) outs(%1329 : tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
      %1331 = tensor.empty() : tensor<1x1x128xf32>
      %1332 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1245, %1330 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%1331 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb133(%1333: f32, %1334: f32, %1335: f32):
        %1336 = arith.addf %1333, %1334 : f32
        linalg.yield %1336 : f32
      } -> tensor<1x1x128xf32>
      %1337 = tensor.empty() : tensor<1x1x128xf32>
      %1338 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1332 : tensor<1x1x128xf32>) outs(%1337 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb134(%1339: f32, %1340: f32):
        %1341 = arith.constant 2.000000e+00 : f32
        %1342 = math.powf %1339, %1341 : f32
        linalg.yield %1342 : f32
      } -> tensor<1x1x128xf32>
      %1343 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1344 = tensor.splat %1343 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1345 = linalg.reduce ins(%1338:tensor<1x1x128xf32>) outs(%1344:tensor<1x1xf32>) dimensions = [2]
      (%1346: f32, %1347: f32) {
        %1348 = arith.addf %1346, %1347 : f32
        linalg.yield %1348 : f32
      }
      %1349 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %1350 = tensor.splat %1349 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1351 = tensor.empty() : tensor<1x1xf32>
      %1352 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1345, %1350 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%1351 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb135(%1353: f32, %1354: f32, %1355: f32):
        %1356 = arith.divf %1353, %1354 : f32
        linalg.yield %1356 : f32
      } -> tensor<1x1xf32>
      %1357 = tensor.collapse_shape %1352 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %1358 = tensor.expand_shape %1357 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %1359 = arith.constant {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %1360 = tensor.splat %1359 {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %1361 = tensor.empty() : tensor<1x1x1xf32>
      %1362 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1358, %1360 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%1361 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb136(%1363: f32, %1364: f32, %1365: f32):
        %1366 = arith.addf %1363, %1364 : f32
        linalg.yield %1366 : f32
      } -> tensor<1x1x1xf32>
      %1367 = tensor.empty() : tensor<1x1x1xf32>
      %1368 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1362 : tensor<1x1x1xf32>) outs(%1367 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb137(%1369: f32, %1370: f32):
        %1371 = math.rsqrt %1369 : f32
        linalg.yield %1371 : f32
      } -> tensor<1x1x1xf32>
      %1372 = tensor.empty() : tensor<1x1x128xf32>
      %1373 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1332, %1368 : tensor<1x1x128xf32>, tensor<1x1x1xf32>) outs(%1372 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb138(%1374: f32, %1375: f32, %1376: f32):
        %1377 = arith.mulf %1374, %1375 : f32
        linalg.yield %1377 : f32
      } -> tensor<1x1x128xf32>
      %1378 = tensor.empty() : tensor<1x1x128xf32>
      %1379 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %1373 : tensor<128xf32>, tensor<1x1x128xf32>) outs(%1378 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb139(%1380: f32, %1381: f32, %1382: f32):
        %1383 = arith.mulf %1380, %1381 : f32
        linalg.yield %1383 : f32
      } -> tensor<1x1x128xf32>
      %1384 = tensor.empty() : tensor<128x512xf32>
      %1385 = linalg.transpose ins(%30:tensor<512x128xf32>) outs(%1384:tensor<128x512xf32>) permutation = [1, 0]
      %1386 = tensor.empty() : tensor<1x1x512xf32>
      %1387 = arith.constant 0.000000e+00 : f32
      %1388 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1387 : f32) outs(%1386 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %1389 = linalg.matmul {prov.region_id = "matmul_9", prov.dispatch_id = "matmul_9", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1379, %1385 : tensor<1x1x128xf32>, tensor<128x512xf32>) outs(%1388 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %1390 = tensor.collapse_shape %1389 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x512xf32> into tensor<512xf32>
      %1391 = tensor.expand_shape %1390 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 128] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x4x128xf32>
      %1392 = tensor.empty() : tensor<1x4x1x128xf32>
      %1393 = linalg.transpose ins(%1391:tensor<1x1x4x128xf32>) outs(%1392:tensor<1x4x1x128xf32>) permutation = [0, 2, 1, 3]
      %1394 = tensor.empty() : tensor<128x512xf32>
      %1395 = linalg.transpose ins(%31:tensor<512x128xf32>) outs(%1394:tensor<128x512xf32>) permutation = [1, 0]
      %1396 = tensor.empty() : tensor<1x1x512xf32>
      %1397 = arith.constant 0.000000e+00 : f32
      %1398 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1397 : f32) outs(%1396 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %1399 = linalg.matmul {prov.region_id = "matmul_10", prov.dispatch_id = "matmul_10", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1379, %1395 : tensor<1x1x128xf32>, tensor<128x512xf32>) outs(%1398 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %1400 = tensor.collapse_shape %1399 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x512xf32> into tensor<512xf32>
      %1401 = tensor.expand_shape %1400 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 128] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x4x128xf32>
      %1402 = tensor.empty() : tensor<1x4x1x128xf32>
      %1403 = linalg.transpose ins(%1401:tensor<1x1x4x128xf32>) outs(%1402:tensor<1x4x1x128xf32>) permutation = [0, 2, 1, 3]
      %1404 = tensor.empty() : tensor<128x512xf32>
      %1405 = linalg.transpose ins(%32:tensor<512x128xf32>) outs(%1404:tensor<128x512xf32>) permutation = [1, 0]
      %1406 = tensor.empty() : tensor<1x1x512xf32>
      %1407 = arith.constant 0.000000e+00 : f32
      %1408 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1407 : f32) outs(%1406 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %1409 = linalg.matmul {prov.region_id = "matmul_11", prov.dispatch_id = "matmul_11", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1379, %1405 : tensor<1x1x128xf32>, tensor<128x512xf32>) outs(%1408 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %1410 = tensor.collapse_shape %1409 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x512xf32> into tensor<512xf32>
      %1411 = tensor.expand_shape %1410 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 128] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x4x128xf32>
      %1412 = tensor.empty() : tensor<1x4x1x128xf32>
      %1413 = linalg.transpose ins(%1411:tensor<1x1x4x128xf32>) outs(%1412:tensor<1x4x1x128xf32>) permutation = [0, 2, 1, 3]
      %1414 = tensor.collapse_shape %965 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %1415 = tensor.expand_shape %1414 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 128] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x1x128xf32>
      %1416 = func.call @aten_unsqueeze_default_wl2() {prov.region_id = "aten_unsqueeze_default_1", prov.dispatch_id = "aten_unsqueeze_default_1"} : () -> tensor<1x1x1x128xf32>
      %1417 = tensor.empty() : tensor<1x4x1x128xf32>
      %1418 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1393, %1415 : tensor<1x4x1x128xf32>, tensor<1x1x1x128xf32>) outs(%1417 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb140(%1419: f32, %1420: f32, %1421: f32):
        %1422 = arith.mulf %1419, %1420 : f32
        linalg.yield %1422 : f32
      } -> tensor<1x4x1x128xf32>
      %1423 = "tensor.extract_slice"(%1393) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x128xf32>) -> tensor<1x4x1x64xf32>
      %1424 = "tensor.extract_slice"(%1393) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x128xf32>) -> tensor<1x4x1x64xf32>
      %1425 = tensor.empty() : tensor<1x4x1x64xf32>
      %1426 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1424 : tensor<1x4x1x64xf32>) outs(%1425 : tensor<1x4x1x64xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb141(%1427: f32, %1428: f32):
        %1429 = arith.negf %1427 : f32
        linalg.yield %1429 : f32
      } -> tensor<1x4x1x64xf32>
      %1430 = tensor.concat dim(3) %1426, %1423 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>) -> tensor<1x4x1x128xf32>
      %1431 = tensor.empty() : tensor<1x4x1x128xf32>
      %1432 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1430, %1416 : tensor<1x4x1x128xf32>, tensor<1x1x1x128xf32>) outs(%1431 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb142(%1433: f32, %1434: f32, %1435: f32):
        %1436 = arith.mulf %1433, %1434 : f32
        linalg.yield %1436 : f32
      } -> tensor<1x4x1x128xf32>
      %1437 = tensor.empty() : tensor<1x4x1x128xf32>
      %1438 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1418, %1432 : tensor<1x4x1x128xf32>, tensor<1x4x1x128xf32>) outs(%1437 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb143(%1439: f32, %1440: f32, %1441: f32):
        %1442 = arith.addf %1439, %1440 : f32
        linalg.yield %1442 : f32
      } -> tensor<1x4x1x128xf32>
      %1443 = tensor.empty() : tensor<1x4x1x128xf32>
      %1444 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1403, %1415 : tensor<1x4x1x128xf32>, tensor<1x1x1x128xf32>) outs(%1443 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb144(%1445: f32, %1446: f32, %1447: f32):
        %1448 = arith.mulf %1445, %1446 : f32
        linalg.yield %1448 : f32
      } -> tensor<1x4x1x128xf32>
      %1449 = "tensor.extract_slice"(%1403) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x128xf32>) -> tensor<1x4x1x64xf32>
      %1450 = "tensor.extract_slice"(%1403) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x128xf32>) -> tensor<1x4x1x64xf32>
      %1451 = tensor.empty() : tensor<1x4x1x64xf32>
      %1452 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1450 : tensor<1x4x1x64xf32>) outs(%1451 : tensor<1x4x1x64xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb145(%1453: f32, %1454: f32):
        %1455 = arith.negf %1453 : f32
        linalg.yield %1455 : f32
      } -> tensor<1x4x1x64xf32>
      %1456 = tensor.concat dim(3) %1452, %1449 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>) -> tensor<1x4x1x128xf32>
      %1457 = tensor.empty() : tensor<1x4x1x128xf32>
      %1458 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1456, %1416 : tensor<1x4x1x128xf32>, tensor<1x1x1x128xf32>) outs(%1457 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb146(%1459: f32, %1460: f32, %1461: f32):
        %1462 = arith.mulf %1459, %1460 : f32
        linalg.yield %1462 : f32
      } -> tensor<1x4x1x128xf32>
      %1463 = tensor.empty() : tensor<1x4x1x128xf32>
      %1464 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1444, %1458 : tensor<1x4x1x128xf32>, tensor<1x4x1x128xf32>) outs(%1463 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb147(%1465: f32, %1466: f32, %1467: f32):
        %1468 = arith.addf %1465, %1466 : f32
        linalg.yield %1468 : f32
      } -> tensor<1x4x1x128xf32>
      %1469 = "tensor.extract_slice"(%945) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 27, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x27x128xf32>) -> tensor<4x27x128xf32>
      %1470 = tensor.empty() : tensor<1xi64>
      %1471 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1470 : tensor<1xi64>) attrs =  {prov.region_id = "iota_4", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
      ^bb148(%1472: i64):
        %1473 = linalg.index 0 : index
        %1474 = arith.index_cast %1473 : index to i64
        %1475 = arith.constant 1 : i64
        %1476 = arith.muli %1474, %1475 : i64
        %1477 = arith.constant 0 : i64
        %1478 = arith.addi %1477, %1476 : i64
        linalg.yield %1478 : i64
      } -> tensor<1xi64>
      %1479 = tensor.empty() : tensor<1xi64>
      %1480 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%951, %1471 : tensor<i64>, tensor<1xi64>) outs(%1479 : tensor<1xi64>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb149(%1481: i64, %1482: i64, %1483: i64):
        %1484 = arith.addi %1481, %1482 : i64
        linalg.yield %1484 : i64
      } -> tensor<1xi64>
      %1485 = func.call @aten_index_copy_default_1_wl3(%1469, %1480, %1464) {prov.region_id = "aten_index_copy_default_1_2", prov.dispatch_id = "aten_index_copy_default_1_2"} : (tensor<4x27x128xf32>, tensor<1xi64>, tensor<1x4x1x128xf32>) -> tensor<1x4x27x128xf32>
      %1486 = "tensor.extract_slice"(%946) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 27, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x27x128xf32>) -> tensor<4x27x128xf32>
      %1487 = tensor.empty() : tensor<1xi64>
      %1488 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1487 : tensor<1xi64>) attrs =  {prov.region_id = "iota_5", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
      ^bb150(%1489: i64):
        %1490 = linalg.index 0 : index
        %1491 = arith.index_cast %1490 : index to i64
        %1492 = arith.constant 1 : i64
        %1493 = arith.muli %1491, %1492 : i64
        %1494 = arith.constant 0 : i64
        %1495 = arith.addi %1494, %1493 : i64
        linalg.yield %1495 : i64
      } -> tensor<1xi64>
      %1496 = tensor.empty() : tensor<1xi64>
      %1497 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%951, %1488 : tensor<i64>, tensor<1xi64>) outs(%1496 : tensor<1xi64>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb151(%1498: i64, %1499: i64, %1500: i64):
        %1501 = arith.addi %1498, %1499 : i64
        linalg.yield %1501 : i64
      } -> tensor<1xi64>
      %1502 = func.call @aten_index_copy_default_1_wl3(%1486, %1497, %1413) {prov.region_id = "aten_index_copy_default_1_3", prov.dispatch_id = "aten_index_copy_default_1_3"} : (tensor<4x27x128xf32>, tensor<1xi64>, tensor<1x4x1x128xf32>) -> tensor<1x4x27x128xf32>
      %1503 = tensor.empty() : tensor<1x4x128x27xf32>
      %1504 = linalg.transpose ins(%1485:tensor<1x4x27x128xf32>) outs(%1503:tensor<1x4x128x27xf32>) permutation = [0, 1, 3, 2]
      %1505 = arith.constant {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1506 = tensor.splat %1505 {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x4x1x27xf32>
      %1507 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1438, %1504 : tensor<1x4x1x128xf32>, tensor<1x4x128x27xf32>) outs(%1506 : tensor<1x4x1x27xf32>) attrs =  {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb152(%1508: f32, %1509: f32, %1510: f32):
        %1511 = arith.mulf %1508, %1509 : f32
        %1512 = arith.addf %1510, %1511 : f32
        linalg.yield %1512 : f32
      } -> tensor<1x4x1x27xf32>
      %1513 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 11.3137083 : f32
      %1514 = tensor.splat %1513 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x4x1x27xf32>
      %1515 = tensor.empty() : tensor<1x4x1x27xf32>
      %1516 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1507, %1514 : tensor<1x4x1x27xf32>, tensor<1x4x1x27xf32>) outs(%1515 : tensor<1x4x1x27xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
      ^bb153(%1517: f32, %1518: f32, %1519: f32):
        %1520 = arith.divf %1517, %1518 : f32
        linalg.yield %1520 : f32
      } -> tensor<1x4x1x27xf32>
      %1521 = tensor.empty() : tensor<27xi64>
      %1522 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1521 : tensor<27xi64>) attrs =  {prov.region_id = "iota_6", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
      ^bb154(%1523: i64):
        %1524 = linalg.index 0 : index
        %1525 = arith.index_cast %1524 : index to i64
        %1526 = arith.constant 1 : i64
        %1527 = arith.muli %1525, %1526 : i64
        %1528 = arith.constant 0 : i64
        %1529 = arith.addi %1528, %1527 : i64
        linalg.yield %1529 : i64
      } -> tensor<27xi64>
      %1530 = tensor.empty() : tensor<1xi64>
      %1531 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1530 : tensor<1xi64>) attrs =  {prov.region_id = "iota_7", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
      ^bb155(%1532: i64):
        %1533 = linalg.index 0 : index
        %1534 = arith.index_cast %1533 : index to i64
        %1535 = arith.constant 1 : i64
        %1536 = arith.muli %1534, %1535 : i64
        %1537 = arith.constant 0 : i64
        %1538 = arith.addi %1537, %1536 : i64
        linalg.yield %1538 : i64
      } -> tensor<1xi64>
      %1539 = tensor.empty() : tensor<1xi64>
      %1540 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%951, %1531 : tensor<i64>, tensor<1xi64>) outs(%1539 : tensor<1xi64>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb156(%1541: i64, %1542: i64, %1543: i64):
        %1544 = arith.addi %1541, %1542 : i64
        linalg.yield %1544 : i64
      } -> tensor<1xi64>
      %1545 = tensor.expand_shape %1540 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1xi64> into tensor<1x1xi64>
      %1546 = tensor.expand_shape %1522 [[0 : i64, 1 : i64]] output_shape [1, 27] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<27xi64> into tensor<1x27xi64>
      %1547 = tensor.empty() : tensor<1x27xi1>
      %1548 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1546, %1545 : tensor<1x27xi64>, tensor<1x1xi64>) outs(%1547 : tensor<1x27xi1>) attrs =  {prov.region_id = "compare_1", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
      ^bb157(%1549: i64, %1550: i64, %1551: i1):
        %1552 = arith.cmpi sle, %1549, %1550 : i64
        linalg.yield %1552 : i1
      } -> tensor<1x27xi1>
      %1553 = tensor.collapse_shape %1548 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x27xi1> into tensor<27xi1>
      %1554 = tensor.expand_shape %1553 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 27] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<27xi1> into tensor<1x1x27xi1>
      %1555 = tensor.collapse_shape %1554 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x1x27xi1> into tensor<27xi1>
      %1556 = tensor.expand_shape %1555 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 27] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<27xi1> into tensor<1x1x1x27xi1>
      %1557 = tensor.empty() : tensor<1x1x1x27xi1>
      %1558 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1556 : tensor<1x1x1x27xi1>) outs(%1557 : tensor<1x1x1x27xi1>) attrs =  {prov.region_id = "bitwise_1", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
      ^bb158(%1559: i1, %1560: i1):
        %1561 = arith.constant true
        %1562 = arith.xori %1559, %1561 : i1
        linalg.yield %1562 : i1
      } -> tensor<1x1x1x27xi1>
      %1563 = func.call @aten_masked_fill_Scalar_wl4(%1516, %1558) {prov.region_id = "aten_masked_fill_Scalar_1", prov.dispatch_id = "aten_masked_fill_Scalar_1"} : (tensor<1x4x1x27xf32>, tensor<1x1x1x27xi1>) -> tensor<1x4x1x27xf32>
      %1564 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0xff800000 : f32
      %1565 = tensor.splat %1564 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x1xf32>
      %1566 = linalg.reduce ins(%1563:tensor<1x4x1x27xf32>) outs(%1565:tensor<1x4x1xf32>) dimensions = [3]
      (%1567: f32, %1568: f32) {
        %1569 = arith.maximumf %1567, %1568 : f32
        linalg.yield %1569 : f32
      }
      %1570 = tensor.collapse_shape %1566 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x1xf32> into tensor<4xf32>
      %1571 = tensor.expand_shape %1570 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 1, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<4xf32> into tensor<1x4x1x1xf32>
      %1572 = tensor.empty() : tensor<1x4x1x27xf32>
      %1573 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1563, %1571 : tensor<1x4x1x27xf32>, tensor<1x4x1x1xf32>) outs(%1572 : tensor<1x4x1x27xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb159(%1574: f32, %1575: f32, %1576: f32):
        %1577 = arith.subf %1574, %1575 : f32
        linalg.yield %1577 : f32
      } -> tensor<1x4x1x27xf32>
      %1578 = tensor.empty() : tensor<1x4x1x27xf32>
      %1579 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1573 : tensor<1x4x1x27xf32>) outs(%1578 : tensor<1x4x1x27xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb160(%1580: f32, %1581: f32):
        %1582 = math.exp %1580 : f32
        linalg.yield %1582 : f32
      } -> tensor<1x4x1x27xf32>
      %1583 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1584 = tensor.splat %1583 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x1xf32>
      %1585 = linalg.reduce ins(%1579:tensor<1x4x1x27xf32>) outs(%1584:tensor<1x4x1xf32>) dimensions = [3]
      (%1586: f32, %1587: f32) {
        %1588 = arith.addf %1586, %1587 : f32
        linalg.yield %1588 : f32
      }
      %1589 = tensor.collapse_shape %1585 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x1xf32> into tensor<4xf32>
      %1590 = tensor.expand_shape %1589 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 1, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<4xf32> into tensor<1x4x1x1xf32>
      %1591 = tensor.empty() : tensor<1x4x1x27xf32>
      %1592 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1579, %1590 : tensor<1x4x1x27xf32>, tensor<1x4x1x1xf32>) outs(%1591 : tensor<1x4x1x27xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb161(%1593: f32, %1594: f32, %1595: f32):
        %1596 = arith.divf %1593, %1594 : f32
        linalg.yield %1596 : f32
      } -> tensor<1x4x1x27xf32>
      %1597 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1598 = tensor.splat %1597 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x4x1x128xf32>
      %1599 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1592, %1502 : tensor<1x4x1x27xf32>, tensor<1x4x27x128xf32>) outs(%1598 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb162(%1600: f32, %1601: f32, %1602: f32):
        %1603 = arith.mulf %1600, %1601 : f32
        %1604 = arith.addf %1602, %1603 : f32
        linalg.yield %1604 : f32
      } -> tensor<1x4x1x128xf32>
      %1605 = tensor.empty() : tensor<1x1x4x128xf32>
      %1606 = linalg.transpose ins(%1599:tensor<1x4x1x128xf32>) outs(%1605:tensor<1x1x4x128xf32>) permutation = [0, 2, 1, 3]
      %1607 = tensor.collapse_shape %1606 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x1x4x128xf32> into tensor<512xf32>
      %1608 = tensor.expand_shape %1607 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 512] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x512xf32>
      %1609 = tensor.empty() : tensor<512x128xf32>
      %1610 = linalg.transpose ins(%33:tensor<128x512xf32>) outs(%1609:tensor<512x128xf32>) permutation = [1, 0]
      %1611 = tensor.empty() : tensor<1x1x128xf32>
      %1612 = arith.constant 0.000000e+00 : f32
      %1613 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1612 : f32) outs(%1611 : tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
      %1614 = linalg.matmul {prov.region_id = "matmul_14", prov.dispatch_id = "matmul_14", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1608, %1610 : tensor<1x1x512xf32>, tensor<512x128xf32>) outs(%1613 : tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
      %1615 = tensor.empty() : tensor<1x1x128xf32>
      %1616 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1332, %1614 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%1615 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb163(%1617: f32, %1618: f32, %1619: f32):
        %1620 = arith.addf %1617, %1618 : f32
        linalg.yield %1620 : f32
      } -> tensor<1x1x128xf32>
      %1621 = tensor.empty() : tensor<1x1x128xf32>
      %1622 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1616 : tensor<1x1x128xf32>) outs(%1621 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb164(%1623: f32, %1624: f32):
        %1625 = arith.constant 2.000000e+00 : f32
        %1626 = math.powf %1623, %1625 : f32
        linalg.yield %1626 : f32
      } -> tensor<1x1x128xf32>
      %1627 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1628 = tensor.splat %1627 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1629 = linalg.reduce ins(%1622:tensor<1x1x128xf32>) outs(%1628:tensor<1x1xf32>) dimensions = [2]
      (%1630: f32, %1631: f32) {
        %1632 = arith.addf %1630, %1631 : f32
        linalg.yield %1632 : f32
      }
      %1633 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %1634 = tensor.splat %1633 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1635 = tensor.empty() : tensor<1x1xf32>
      %1636 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1629, %1634 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%1635 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb165(%1637: f32, %1638: f32, %1639: f32):
        %1640 = arith.divf %1637, %1638 : f32
        linalg.yield %1640 : f32
      } -> tensor<1x1xf32>
      %1641 = tensor.collapse_shape %1636 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %1642 = tensor.expand_shape %1641 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %1643 = arith.constant {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %1644 = tensor.splat %1643 {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %1645 = tensor.empty() : tensor<1x1x1xf32>
      %1646 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1642, %1644 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%1645 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb166(%1647: f32, %1648: f32, %1649: f32):
        %1650 = arith.addf %1647, %1648 : f32
        linalg.yield %1650 : f32
      } -> tensor<1x1x1xf32>
      %1651 = tensor.empty() : tensor<1x1x1xf32>
      %1652 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1646 : tensor<1x1x1xf32>) outs(%1651 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb167(%1653: f32, %1654: f32):
        %1655 = math.rsqrt %1653 : f32
        linalg.yield %1655 : f32
      } -> tensor<1x1x1xf32>
      %1656 = tensor.empty() : tensor<1x1x128xf32>
      %1657 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1616, %1652 : tensor<1x1x128xf32>, tensor<1x1x1xf32>) outs(%1656 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb168(%1658: f32, %1659: f32, %1660: f32):
        %1661 = arith.mulf %1658, %1659 : f32
        linalg.yield %1661 : f32
      } -> tensor<1x1x128xf32>
      %1662 = tensor.empty() : tensor<1x1x128xf32>
      %1663 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %1657 : tensor<128xf32>, tensor<1x1x128xf32>) outs(%1662 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb169(%1664: f32, %1665: f32, %1666: f32):
        %1667 = arith.mulf %1664, %1665 : f32
        linalg.yield %1667 : f32
      } -> tensor<1x1x128xf32>
      %1668 = tensor.empty() : tensor<128x256xf32>
      %1669 = linalg.transpose ins(%34:tensor<256x128xf32>) outs(%1668:tensor<128x256xf32>) permutation = [1, 0]
      %1670 = tensor.empty() : tensor<1x1x256xf32>
      %1671 = arith.constant 0.000000e+00 : f32
      %1672 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1671 : f32) outs(%1670 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %1673 = linalg.matmul {prov.region_id = "matmul_15", prov.dispatch_id = "matmul_15", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1663, %1669 : tensor<1x1x128xf32>, tensor<128x256xf32>) outs(%1672 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %1674 = tensor.empty() : tensor<1x1x256xf32>
      %1675 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1673 : tensor<1x1x256xf32>) outs(%1674 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "silu_1", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb170(%1676: f32, %1677: f32):
        %1678 = arith.constant 1.000000e+00 : f32
        %1679 = arith.negf %1676 : f32
        %1680 = math.exp %1679 : f32
        %1681 = arith.addf %1678, %1680 : f32
        %1682 = arith.divf %1678, %1681 : f32
        %1683 = arith.mulf %1676, %1682 : f32
        linalg.yield %1683 : f32
      } -> tensor<1x1x256xf32>
      %1684 = tensor.empty() : tensor<128x256xf32>
      %1685 = linalg.transpose ins(%35:tensor<256x128xf32>) outs(%1684:tensor<128x256xf32>) permutation = [1, 0]
      %1686 = tensor.empty() : tensor<1x1x256xf32>
      %1687 = arith.constant 0.000000e+00 : f32
      %1688 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1687 : f32) outs(%1686 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %1689 = linalg.matmul {prov.region_id = "matmul_16", prov.dispatch_id = "matmul_16", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1663, %1685 : tensor<1x1x128xf32>, tensor<128x256xf32>) outs(%1688 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %1690 = tensor.empty() : tensor<1x1x256xf32>
      %1691 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1675, %1689 : tensor<1x1x256xf32>, tensor<1x1x256xf32>) outs(%1690 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb171(%1692: f32, %1693: f32, %1694: f32):
        %1695 = arith.mulf %1692, %1693 : f32
        linalg.yield %1695 : f32
      } -> tensor<1x1x256xf32>
      %1696 = tensor.empty() : tensor<256x128xf32>
      %1697 = linalg.transpose ins(%36:tensor<128x256xf32>) outs(%1696:tensor<256x128xf32>) permutation = [1, 0]
      %1698 = tensor.empty() : tensor<1x1x128xf32>
      %1699 = arith.constant 0.000000e+00 : f32
      %1700 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1699 : f32) outs(%1698 : tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
      %1701 = linalg.matmul {prov.region_id = "matmul_17", prov.dispatch_id = "matmul_17", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1691, %1697 : tensor<1x1x256xf32>, tensor<256x128xf32>) outs(%1700 : tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
      %1702 = tensor.empty() : tensor<1x1x128xf32>
      %1703 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1616, %1701 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%1702 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb172(%1704: f32, %1705: f32, %1706: f32):
        %1707 = arith.addf %1704, %1705 : f32
        linalg.yield %1707 : f32
      } -> tensor<1x1x128xf32>
      %1708 = func.call @aten_stack_default_wl5(%1114, %1485) {prov.region_id = "aten_stack_default_0", prov.dispatch_id = "aten_stack_default_0"} : (tensor<1x4x27x128xf32>, tensor<1x4x27x128xf32>) -> tensor<2x1x4x27x128xf32>
      %1709 = func.call @aten_stack_default_wl5(%1131, %1502) {prov.region_id = "aten_stack_default_1", prov.dispatch_id = "aten_stack_default_1"} : (tensor<1x4x27x128xf32>, tensor<1x4x27x128xf32>) -> tensor<2x1x4x27x128xf32>
      %1710 = tensor.empty() : tensor<1x1x128xf32>
      %1711 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1703 : tensor<1x1x128xf32>) outs(%1710 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb173(%1712: f32, %1713: f32):
        %1714 = arith.constant 2.000000e+00 : f32
        %1715 = math.powf %1712, %1714 : f32
        linalg.yield %1715 : f32
      } -> tensor<1x1x128xf32>
      %1716 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1717 = tensor.splat %1716 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1718 = linalg.reduce ins(%1711:tensor<1x1x128xf32>) outs(%1717:tensor<1x1xf32>) dimensions = [2]
      (%1719: f32, %1720: f32) {
        %1721 = arith.addf %1719, %1720 : f32
        linalg.yield %1721 : f32
      }
      %1722 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %1723 = tensor.splat %1722 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1724 = tensor.empty() : tensor<1x1xf32>
      %1725 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1718, %1723 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%1724 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb174(%1726: f32, %1727: f32, %1728: f32):
        %1729 = arith.divf %1726, %1727 : f32
        linalg.yield %1729 : f32
      } -> tensor<1x1xf32>
      %1730 = tensor.collapse_shape %1725 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %1731 = tensor.expand_shape %1730 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %1732 = arith.constant {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %1733 = tensor.splat %1732 {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %1734 = tensor.empty() : tensor<1x1x1xf32>
      %1735 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1731, %1733 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%1734 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb175(%1736: f32, %1737: f32, %1738: f32):
        %1739 = arith.addf %1736, %1737 : f32
        linalg.yield %1739 : f32
      } -> tensor<1x1x1xf32>
      %1740 = tensor.empty() : tensor<1x1x1xf32>
      %1741 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1735 : tensor<1x1x1xf32>) outs(%1740 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb176(%1742: f32, %1743: f32):
        %1744 = math.rsqrt %1742 : f32
        linalg.yield %1744 : f32
      } -> tensor<1x1x1xf32>
      %1745 = tensor.empty() : tensor<1x1x128xf32>
      %1746 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1703, %1741 : tensor<1x1x128xf32>, tensor<1x1x1xf32>) outs(%1745 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb177(%1747: f32, %1748: f32, %1749: f32):
        %1750 = arith.mulf %1747, %1748 : f32
        linalg.yield %1750 : f32
      } -> tensor<1x1x128xf32>
      %1751 = tensor.empty() : tensor<1x1x128xf32>
      %1752 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %1746 : tensor<128xf32>, tensor<1x1x128xf32>) outs(%1751 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb178(%1753: f32, %1754: f32, %1755: f32):
        %1756 = arith.mulf %1753, %1754 : f32
        linalg.yield %1756 : f32
      } -> tensor<1x1x128xf32>
      %1757 = tensor.empty() : tensor<128x512xf32>
      %1758 = linalg.transpose ins(%41:tensor<512x128xf32>) outs(%1757:tensor<128x512xf32>) permutation = [1, 0]
      %1759 = tensor.empty() : tensor<1x1x512xf32>
      %1760 = arith.constant 0.000000e+00 : f32
      %1761 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1760 : f32) outs(%1759 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %1762 = linalg.matmul {prov.region_id = "matmul_18", prov.dispatch_id = "matmul_18", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1752, %1758 : tensor<1x1x128xf32>, tensor<128x512xf32>) outs(%1761 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %1763 = "tensor.extract_slice"(%1762) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<1x1x512xf32>) -> tensor<512xf32>
      %1764 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0xff800000 : f32
      %1765 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0 : i64
      %1766 = tensor.splat %1764 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<f32>
      %1767 = tensor.splat %1765 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<i64>
      %1768, %1769 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%1763 : tensor<512xf32>) outs(%1766, %1767 : tensor<f32>, tensor<i64>) attrs =  {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} {
      ^bb179(%1770: f32, %1771: f32, %1772: i64):
        %1773 = linalg.index 0 : index
        %1774 = arith.index_cast %1773 : index to i64
        %1775 = arith.cmpf ogt, %1770, %1771 : f32
        %1776 = arith.select %1775, %1770, %1771 : f32
        %1777 = arith.select %1775, %1774, %1772 : i64
        linalg.yield %1776, %1777 : f32, i64
      } -> (tensor<f32>, tensor<i64>)
      %1778 = tensor.extract %1768[] : tensor<f32>
      %1779 = tensor.from_elements %1778 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xf32>
      %1780 = tensor.extract %1769[] : tensor<i64>
      %1781 = tensor.from_elements %1780 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xi64>
      %1782 = arith.constant {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} 1 : i64
      %1783 = tensor.splat %1782 {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} : tensor<i64>
      %1784 = tensor.empty() : tensor<i64>
      %1785 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%942, %1783 : tensor<i64>, tensor<i64>) outs(%1784 : tensor<i64>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb180(%1786: i64, %1787: i64, %1788: i64):
        %1789 = arith.addi %1786, %1787 : i64
        linalg.yield %1789 : i64
      } -> tensor<i64>
      scf.yield %1785, %1781, %949, %1708, %1709 : tensor<i64>, tensor<1xi64>, tensor<1x7xi64>, tensor<2x1x4x27x128xf32>, tensor<2x1x4x27x128xf32>
    }
    func.return %938 : tensor<1x7xi64>
  }
}
