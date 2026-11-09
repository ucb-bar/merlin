builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func private @aten_zeros_default() -> tensor<2x1x4x15x32xf32>
  func.func private @aten_index_select_default(tensor<15x32xf32>, tensor<8xi64>) -> tensor<8x32xf32>
  func.func private @aten_index_copy_default(tensor<4x15x32xf32>, tensor<8xi64>, tensor<1x4x8x32xf32>) -> tensor<1x4x15x32xf32>
  func.func private @aten_masked_fill_Scalar(tensor<1x4x8x15xf32>, tensor<1x1x8x15xi1>) -> tensor<1x4x8x15xf32>
  func.func private @aten_stack_default(tensor<1x4x15x32xf32>, tensor<1x4x15x32xf32>) -> tensor<2x1x4x15x32xf32>
  func.func private @aten_zeros_default_1() -> tensor<i64>
  func.func private @aten_zeros_default_2() -> tensor<1x7xi64>
  func.func private @aten_index_copy_default_wl0(tensor<1x7xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x7xi64>
  func.func private @aten_index_select_default_wl1(tensor<15x32xf32>, tensor<1xi64>) -> tensor<1x32xf32>
  func.func private @aten_index_copy_default_1_wl2(tensor<4x15x32xf32>, tensor<1xi64>, tensor<1x4x1x32xf32>) -> tensor<1x4x15x32xf32>
  func.func private @aten_masked_fill_Scalar_wl3(tensor<1x4x1x15xf32>, tensor<1x1x1x15xi1>) -> tensor<1x4x1x15xf32>
  func.func private @aten_stack_default_wl4(tensor<1x4x15x32xf32>, tensor<1x4x15x32xf32>) -> tensor<2x1x4x15x32xf32>
  func.func @forward(%0: tensor<256x128xf32>, %1: tensor<128xf32>, %2: tensor<128x128xf32>, %3: tensor<128x128xf32>, %4: tensor<128x128xf32>, %5: tensor<128x128xf32>, %6: tensor<128xf32>, %7: tensor<344x128xf32>, %8: tensor<344x128xf32>, %9: tensor<128x344xf32>, %10: tensor<128xf32>, %11: tensor<128x128xf32>, %12: tensor<128x128xf32>, %13: tensor<128x128xf32>, %14: tensor<128x128xf32>, %15: tensor<128xf32>, %16: tensor<344x128xf32>, %17: tensor<344x128xf32>, %18: tensor<128x344xf32>, %19: tensor<128xf32>, %20: tensor<256x128xf32>, %21: tensor<128xf32>, %22: tensor<128x128xf32>, %23: tensor<128x128xf32>, %24: tensor<128x128xf32>, %25: tensor<128x128xf32>, %26: tensor<128xf32>, %27: tensor<344x128xf32>, %28: tensor<344x128xf32>, %29: tensor<128x344xf32>, %30: tensor<128xf32>, %31: tensor<128x128xf32>, %32: tensor<128x128xf32>, %33: tensor<128x128xf32>, %34: tensor<128x128xf32>, %35: tensor<128xf32>, %36: tensor<344x128xf32>, %37: tensor<344x128xf32>, %38: tensor<128x344xf32>, %39: tensor<128xf32>, %40: tensor<256x128xf32>, %41: tensor<256x128xf32>, %42: tensor<i64>, %43: tensor<1x8xi64>) -> tensor<1x7xi64> {
    %44 = tensor.empty() : tensor<16xf32>
    %45 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%44 : tensor<16xf32>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start", prov.orig_dtype = "float32"} {
    ^bb0(%46: f32):
      %47 = linalg.index 0 : index
      %48 = arith.index_cast %47 : index to i64
      %49 = arith.sitofp %48 : i64 to f32
      %50 = arith.constant 1.000000e+00 : f32
      %51 = arith.mulf %49, %50 : f32
      %52 = arith.constant 0.000000e+00 : f32
      %53 = arith.addf %52, %51 : f32
      linalg.yield %53 : f32
    } -> tensor<16xf32>
    %54 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 1.600000e+01 : f32
    %55 = tensor.splat %54 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<16xf32>
    %56 = tensor.empty() : tensor<16xf32>
    %57 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%45, %55 : tensor<16xf32>, tensor<16xf32>) outs(%56 : tensor<16xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb1(%58: f32, %59: f32, %60: f32):
      %61 = arith.divf %58, %59 : f32
      linalg.yield %61 : f32
    } -> tensor<16xf32>
    %62 = tensor.empty() : tensor<16xf32>
    %63 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%57 : tensor<16xf32>) outs(%62 : tensor<16xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Scalar", prov.orig_dtype = "float32"} {
    ^bb2(%64: f32, %65: f32):
      %66 = arith.constant 1.000000e+04 : f32
      %67 = math.powf %66, %64 : f32
      linalg.yield %67 : f32
    } -> tensor<16xf32>
    %68 = tensor.empty() : tensor<16xf32>
    %69 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%63 : tensor<16xf32>) outs(%68 : tensor<16xf32>) attrs =  {prov.region_id = "elementwise_0", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32"} {
    ^bb3(%70: f32, %71: f32):
      %72 = arith.constant 1.000000e+00 : f32
      %73 = arith.divf %72, %70 : f32
      linalg.yield %73 : f32
    } -> tensor<16xf32>
    %74 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
    %75 = tensor.splat %74 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16xf32>
    %76 = tensor.empty() : tensor<16xf32>
    %77 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%69, %75 : tensor<16xf32>, tensor<16xf32>) outs(%76 : tensor<16xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb4(%78: f32, %79: f32, %80: f32):
      %81 = arith.mulf %78, %79 : f32
      linalg.yield %81 : f32
    } -> tensor<16xf32>
    %82 = tensor.empty() : tensor<15xf32>
    %83 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%82 : tensor<15xf32>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "float32"} {
    ^bb5(%84: f32):
      %85 = linalg.index 0 : index
      %86 = arith.index_cast %85 : index to i64
      %87 = arith.sitofp %86 : i64 to f32
      %88 = arith.constant 1.000000e+00 : f32
      %89 = arith.mulf %87, %88 : f32
      %90 = arith.constant 0.000000e+00 : f32
      %91 = arith.addf %90, %89 : f32
      linalg.yield %91 : f32
    } -> tensor<15xf32>
    %92 = tensor.expand_shape %83 [[0 : i64, 1 : i64]] output_shape [15, 1] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<15xf32> into tensor<15x1xf32>
    %93 = tensor.expand_shape %77 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<16xf32> into tensor<1x16xf32>
    %94 = tensor.empty() : tensor<15x16xf32>
    %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%92, %93 : tensor<15x1xf32>, tensor<1x16xf32>) outs(%94 : tensor<15x16xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb6(%96: f32, %97: f32, %98: f32):
      %99 = arith.mulf %96, %97 : f32
      linalg.yield %99 : f32
    } -> tensor<15x16xf32>
    %100 = tensor.empty() : tensor<15x16xf32>
    %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%95 : tensor<15x16xf32>) outs(%100 : tensor<15x16xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32"} {
    ^bb7(%102: f32, %103: f32):
      %104 = math.cos %102 : f32
      linalg.yield %104 : f32
    } -> tensor<15x16xf32>
    %105 = tensor.empty() : tensor<15x16xf32>
    %106 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%95 : tensor<15x16xf32>) outs(%105 : tensor<15x16xf32>) attrs =  {prov.region_id = "cos_1", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32"} {
    ^bb8(%107: f32, %108: f32):
      %109 = math.cos %107 : f32
      linalg.yield %109 : f32
    } -> tensor<15x16xf32>
    %110 = tensor.concat dim(1) %101, %106 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<15x16xf32>, tensor<15x16xf32>) -> tensor<15x32xf32>
    %111 = tensor.empty() : tensor<15x16xf32>
    %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%95 : tensor<15x16xf32>) outs(%111 : tensor<15x16xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32"} {
    ^bb9(%113: f32, %114: f32):
      %115 = math.sin %113 : f32
      linalg.yield %115 : f32
    } -> tensor<15x16xf32>
    %116 = tensor.empty() : tensor<15x16xf32>
    %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%95 : tensor<15x16xf32>) outs(%116 : tensor<15x16xf32>) attrs =  {prov.region_id = "sin_1", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32"} {
    ^bb10(%118: f32, %119: f32):
      %120 = math.sin %118 : f32
      linalg.yield %120 : f32
    } -> tensor<15x16xf32>
    %121 = tensor.concat dim(1) %112, %117 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<15x16xf32>, tensor<15x16xf32>) -> tensor<15x32xf32>
    %122 = tensor.empty() : tensor<15xi64>
    %123 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%122 : tensor<15xi64>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
    ^bb11(%124: i64):
      %125 = linalg.index 0 : index
      %126 = arith.index_cast %125 : index to i64
      %127 = arith.constant 1 : i64
      %128 = arith.muli %126, %127 : i64
      %129 = arith.constant 0 : i64
      %130 = arith.addi %129, %128 : i64
      linalg.yield %130 : i64
    } -> tensor<15xi64>
    %131 = func.call @aten_zeros_default() {prov.region_id = "aten_zeros_default_0", prov.dispatch_id = "aten_zeros_default_0"} : () -> tensor<2x1x4x15x32xf32>
    %132 = func.call @aten_zeros_default() {prov.region_id = "aten_zeros_default_1", prov.dispatch_id = "aten_zeros_default_1"} : () -> tensor<2x1x4x15x32xf32>
    %133 = tensor.empty() : tensor<8xi64>
    %134 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%133 : tensor<8xi64>) attrs =  {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
    ^bb12(%135: i64):
      %136 = linalg.index 0 : index
      %137 = arith.index_cast %136 : index to i64
      %138 = arith.constant 1 : i64
      %139 = arith.muli %137, %138 : i64
      %140 = arith.constant 0 : i64
      %141 = arith.addi %140, %139 : i64
      linalg.yield %141 : i64
    } -> tensor<8xi64>
    %142 = tensor.empty() : tensor<1x8x128xf32>
    %143 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43 : tensor<1x8xi64>) outs(%142 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32"} {
    ^bb13(%144: i64, %145: f32):
      %146 = arith.index_cast %144 : i64 to index
      %147 = linalg.index 2 : index
      %148 = tensor.extract %40[%146, %147] : tensor<256x128xf32>
      linalg.yield %148 : f32
    } -> tensor<1x8x128xf32>
    %149 = func.call @aten_index_select_default(%110, %134) {prov.region_id = "aten_index_select_default_0", prov.dispatch_id = "aten_index_select_default_0"} : (tensor<15x32xf32>, tensor<8xi64>) -> tensor<8x32xf32>
    %150 = func.call @aten_index_select_default(%121, %134) {prov.region_id = "aten_index_select_default_1", prov.dispatch_id = "aten_index_select_default_1"} : (tensor<15x32xf32>, tensor<8xi64>) -> tensor<8x32xf32>
    %151 = tensor.empty() : tensor<1x8x128xf32>
    %152 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%143 : tensor<1x8x128xf32>) outs(%151 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb14(%153: f32, %154: f32):
      %155 = arith.constant 2.000000e+00 : f32
      %156 = math.powf %153, %155 : f32
      linalg.yield %156 : f32
    } -> tensor<1x8x128xf32>
    %157 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %158 = tensor.splat %157 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %159 = linalg.reduce ins(%152:tensor<1x8x128xf32>) outs(%158:tensor<1x8xf32>) dimensions = [2]
    (%160: f32, %161: f32) {
      %162 = arith.addf %160, %161 : f32
      linalg.yield %162 : f32
    }
    %163 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
    %164 = tensor.splat %163 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %165 = tensor.empty() : tensor<1x8xf32>
    %166 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%159, %164 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%165 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb15(%167: f32, %168: f32, %169: f32):
      %170 = arith.divf %167, %168 : f32
      linalg.yield %170 : f32
    } -> tensor<1x8xf32>
    %171 = tensor.collapse_shape %166 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %172 = tensor.expand_shape %171 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %173 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
    %174 = tensor.splat %173 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %175 = tensor.empty() : tensor<1x8x1xf32>
    %176 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%172, %174 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%175 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb16(%177: f32, %178: f32, %179: f32):
      %180 = arith.addf %177, %178 : f32
      linalg.yield %180 : f32
    } -> tensor<1x8x1xf32>
    %181 = tensor.empty() : tensor<1x8x1xf32>
    %182 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%176 : tensor<1x8x1xf32>) outs(%181 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb17(%183: f32, %184: f32):
      %185 = math.rsqrt %183 : f32
      linalg.yield %185 : f32
    } -> tensor<1x8x1xf32>
    %186 = tensor.empty() : tensor<1x8x128xf32>
    %187 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%143, %182 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%186 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb18(%188: f32, %189: f32, %190: f32):
      %191 = arith.mulf %188, %189 : f32
      linalg.yield %191 : f32
    } -> tensor<1x8x128xf32>
    %192 = tensor.empty() : tensor<1x8x128xf32>
    %193 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%187, %21 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%192 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb19(%194: f32, %195: f32, %196: f32):
      %197 = arith.mulf %194, %195 : f32
      linalg.yield %197 : f32
    } -> tensor<1x8x128xf32>
    %198 = tensor.empty() : tensor<128x128xf32>
    %199 = linalg.transpose ins(%22:tensor<128x128xf32>) outs(%198:tensor<128x128xf32>) permutation = [1, 0]
    %200 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %201 = tensor.splat %200 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32>
    %202 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%193, %199 : tensor<1x8x128xf32>, tensor<128x128xf32>) outs(%201 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb20(%203: f32, %204: f32, %205: f32):
      %206 = arith.mulf %203, %204 : f32
      %207 = arith.addf %205, %206 : f32
      linalg.yield %207 : f32
    } -> tensor<1x8x128xf32>
    %208 = tensor.collapse_shape %202 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %209 = tensor.expand_shape %208 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %210 = tensor.empty() : tensor<1x4x8x32xf32>
    %211 = linalg.transpose ins(%209:tensor<1x8x4x32xf32>) outs(%210:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %212 = tensor.empty() : tensor<128x128xf32>
    %213 = linalg.transpose ins(%23:tensor<128x128xf32>) outs(%212:tensor<128x128xf32>) permutation = [1, 0]
    %214 = arith.constant {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %215 = tensor.splat %214 {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32>
    %216 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%193, %213 : tensor<1x8x128xf32>, tensor<128x128xf32>) outs(%215 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb21(%217: f32, %218: f32, %219: f32):
      %220 = arith.mulf %217, %218 : f32
      %221 = arith.addf %219, %220 : f32
      linalg.yield %221 : f32
    } -> tensor<1x8x128xf32>
    %222 = tensor.collapse_shape %216 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %223 = tensor.expand_shape %222 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %224 = tensor.empty() : tensor<1x4x8x32xf32>
    %225 = linalg.transpose ins(%223:tensor<1x8x4x32xf32>) outs(%224:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %226 = tensor.empty() : tensor<128x128xf32>
    %227 = linalg.transpose ins(%24:tensor<128x128xf32>) outs(%226:tensor<128x128xf32>) permutation = [1, 0]
    %228 = arith.constant {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %229 = tensor.splat %228 {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32>
    %230 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%193, %227 : tensor<1x8x128xf32>, tensor<128x128xf32>) outs(%229 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb22(%231: f32, %232: f32, %233: f32):
      %234 = arith.mulf %231, %232 : f32
      %235 = arith.addf %233, %234 : f32
      linalg.yield %235 : f32
    } -> tensor<1x8x128xf32>
    %236 = tensor.collapse_shape %230 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %237 = tensor.expand_shape %236 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %238 = tensor.empty() : tensor<1x4x8x32xf32>
    %239 = linalg.transpose ins(%237:tensor<1x8x4x32xf32>) outs(%238:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %240 = tensor.collapse_shape %149 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8x32xf32> into tensor<256xf32>
    %241 = tensor.expand_shape %240 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x8x32xf32>
    %242 = tensor.collapse_shape %241 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x32xf32> into tensor<256xf32>
    %243 = tensor.expand_shape %242 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %244 = tensor.collapse_shape %150 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8x32xf32> into tensor<256xf32>
    %245 = tensor.expand_shape %244 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x8x32xf32>
    %246 = tensor.collapse_shape %245 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x32xf32> into tensor<256xf32>
    %247 = tensor.expand_shape %246 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %248 = "tensor.extract_slice"(%211) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %249 = "tensor.extract_slice"(%211) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %250 = tensor.empty() : tensor<1x4x8x16xf32>
    %251 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%249 : tensor<1x4x8x16xf32>) outs(%250 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb23(%252: f32, %253: f32):
      %254 = arith.negf %252 : f32
      linalg.yield %254 : f32
    } -> tensor<1x4x8x16xf32>
    %255 = tensor.concat dim(3) %251, %248 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %256 = tensor.empty() : tensor<1x4x8x32xf32>
    %257 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%211, %243 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%256 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb24(%258: f32, %259: f32, %260: f32):
      %261 = arith.mulf %258, %259 : f32
      linalg.yield %261 : f32
    } -> tensor<1x4x8x32xf32>
    %262 = tensor.empty() : tensor<1x4x8x32xf32>
    %263 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%255, %247 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%262 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb25(%264: f32, %265: f32, %266: f32):
      %267 = arith.mulf %264, %265 : f32
      linalg.yield %267 : f32
    } -> tensor<1x4x8x32xf32>
    %268 = tensor.empty() : tensor<1x4x8x32xf32>
    %269 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%257, %263 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%268 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb26(%270: f32, %271: f32, %272: f32):
      %273 = arith.addf %270, %271 : f32
      linalg.yield %273 : f32
    } -> tensor<1x4x8x32xf32>
    %274 = tensor.collapse_shape %149 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8x32xf32> into tensor<256xf32>
    %275 = tensor.expand_shape %274 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x8x32xf32>
    %276 = tensor.collapse_shape %275 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x32xf32> into tensor<256xf32>
    %277 = tensor.expand_shape %276 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %278 = tensor.collapse_shape %150 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8x32xf32> into tensor<256xf32>
    %279 = tensor.expand_shape %278 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x8x32xf32>
    %280 = tensor.collapse_shape %279 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x32xf32> into tensor<256xf32>
    %281 = tensor.expand_shape %280 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %282 = "tensor.extract_slice"(%225) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %283 = "tensor.extract_slice"(%225) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %284 = tensor.empty() : tensor<1x4x8x16xf32>
    %285 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%283 : tensor<1x4x8x16xf32>) outs(%284 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb27(%286: f32, %287: f32):
      %288 = arith.negf %286 : f32
      linalg.yield %288 : f32
    } -> tensor<1x4x8x16xf32>
    %289 = tensor.concat dim(3) %285, %282 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %290 = tensor.empty() : tensor<1x4x8x32xf32>
    %291 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%225, %277 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%290 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb28(%292: f32, %293: f32, %294: f32):
      %295 = arith.mulf %292, %293 : f32
      linalg.yield %295 : f32
    } -> tensor<1x4x8x32xf32>
    %296 = tensor.empty() : tensor<1x4x8x32xf32>
    %297 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%289, %281 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%296 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb29(%298: f32, %299: f32, %300: f32):
      %301 = arith.mulf %298, %299 : f32
      linalg.yield %301 : f32
    } -> tensor<1x4x8x32xf32>
    %302 = tensor.empty() : tensor<1x4x8x32xf32>
    %303 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%291, %297 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%302 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb30(%304: f32, %305: f32, %306: f32):
      %307 = arith.addf %304, %305 : f32
      linalg.yield %307 : f32
    } -> tensor<1x4x8x32xf32>
    %308 = "tensor.extract_slice"(%131) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 15, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x15x32xf32>) -> tensor<4x15x32xf32>
    %309 = func.call @aten_index_copy_default(%308, %134, %303) {prov.region_id = "aten_index_copy_default_0", prov.dispatch_id = "aten_index_copy_default_0"} : (tensor<4x15x32xf32>, tensor<8xi64>, tensor<1x4x8x32xf32>) -> tensor<1x4x15x32xf32>
    %310 = "tensor.extract_slice"(%132) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 15, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x15x32xf32>) -> tensor<4x15x32xf32>
    %311 = func.call @aten_index_copy_default(%310, %134, %239) {prov.region_id = "aten_index_copy_default_1", prov.dispatch_id = "aten_index_copy_default_1"} : (tensor<4x15x32xf32>, tensor<8xi64>, tensor<1x4x8x32xf32>) -> tensor<1x4x15x32xf32>
    %312 = tensor.empty() : tensor<1x4x32x15xf32>
    %313 = linalg.transpose ins(%309:tensor<1x4x15x32xf32>) outs(%312:tensor<1x4x32x15xf32>) permutation = [0, 1, 3, 2]
    %314 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %315 = tensor.splat %314 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x4x8x15xf32>
    %316 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%269, %313 : tensor<1x4x8x32xf32>, tensor<1x4x32x15xf32>) outs(%315 : tensor<1x4x8x15xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb31(%317: f32, %318: f32, %319: f32):
      %320 = arith.mulf %317, %318 : f32
      %321 = arith.addf %319, %320 : f32
      linalg.yield %321 : f32
    } -> tensor<1x4x8x15xf32>
    %322 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 5.65685415 : f32
    %323 = tensor.splat %322 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x4x8x15xf32>
    %324 = tensor.empty() : tensor<1x4x8x15xf32>
    %325 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%316, %323 : tensor<1x4x8x15xf32>, tensor<1x4x8x15xf32>) outs(%324 : tensor<1x4x8x15xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb32(%326: f32, %327: f32, %328: f32):
      %329 = arith.divf %326, %327 : f32
      linalg.yield %329 : f32
    } -> tensor<1x4x8x15xf32>
    %330 = tensor.expand_shape %134 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<8xi64> into tensor<8x1xi64>
    %331 = tensor.expand_shape %123 [[0 : i64, 1 : i64]] output_shape [1, 15] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<15xi64> into tensor<1x15xi64>
    %332 = tensor.empty() : tensor<8x15xi1>
    %333 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%331, %330 : tensor<1x15xi64>, tensor<8x1xi64>) outs(%332 : tensor<8x15xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
    ^bb33(%334: i64, %335: i64, %336: i1):
      %337 = arith.cmpi sle, %334, %335 : i64
      linalg.yield %337 : i1
    } -> tensor<8x15xi1>
    %338 = tensor.collapse_shape %333 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<8x15xi1> into tensor<120xi1>
    %339 = tensor.expand_shape %338 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 15] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<120xi1> into tensor<1x8x15xi1>
    %340 = tensor.collapse_shape %339 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x8x15xi1> into tensor<120xi1>
    %341 = tensor.expand_shape %340 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 15] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<120xi1> into tensor<1x1x8x15xi1>
    %342 = tensor.empty() : tensor<1x1x8x15xi1>
    %343 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%341 : tensor<1x1x8x15xi1>) outs(%342 : tensor<1x1x8x15xi1>) attrs =  {prov.region_id = "bitwise_0", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
    ^bb34(%344: i1, %345: i1):
      %346 = arith.constant true
      %347 = arith.xori %344, %346 : i1
      linalg.yield %347 : i1
    } -> tensor<1x1x8x15xi1>
    %348 = func.call @aten_masked_fill_Scalar(%325, %343) {prov.region_id = "aten_masked_fill_Scalar_0", prov.dispatch_id = "aten_masked_fill_Scalar_0"} : (tensor<1x4x8x15xf32>, tensor<1x1x8x15xi1>) -> tensor<1x4x8x15xf32>
    %349 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0xff800000 : f32
    %350 = tensor.splat %349 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x8xf32>
    %351 = linalg.reduce ins(%348:tensor<1x4x8x15xf32>) outs(%350:tensor<1x4x8xf32>) dimensions = [3]
    (%352: f32, %353: f32) {
      %354 = arith.maximumf %352, %353 : f32
      linalg.yield %354 : f32
    }
    %355 = tensor.collapse_shape %351 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x8xf32> into tensor<32xf32>
    %356 = tensor.expand_shape %355 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %357 = tensor.empty() : tensor<1x4x8x15xf32>
    %358 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%348, %356 : tensor<1x4x8x15xf32>, tensor<1x4x8x1xf32>) outs(%357 : tensor<1x4x8x15xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb35(%359: f32, %360: f32, %361: f32):
      %362 = arith.subf %359, %360 : f32
      linalg.yield %362 : f32
    } -> tensor<1x4x8x15xf32>
    %363 = tensor.empty() : tensor<1x4x8x15xf32>
    %364 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%358 : tensor<1x4x8x15xf32>) outs(%363 : tensor<1x4x8x15xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb36(%365: f32, %366: f32):
      %367 = math.exp %365 : f32
      linalg.yield %367 : f32
    } -> tensor<1x4x8x15xf32>
    %368 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %369 = tensor.splat %368 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x8xf32>
    %370 = linalg.reduce ins(%364:tensor<1x4x8x15xf32>) outs(%369:tensor<1x4x8xf32>) dimensions = [3]
    (%371: f32, %372: f32) {
      %373 = arith.addf %371, %372 : f32
      linalg.yield %373 : f32
    }
    %374 = tensor.collapse_shape %370 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x8xf32> into tensor<32xf32>
    %375 = tensor.expand_shape %374 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %376 = tensor.empty() : tensor<1x4x8x15xf32>
    %377 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%364, %375 : tensor<1x4x8x15xf32>, tensor<1x4x8x1xf32>) outs(%376 : tensor<1x4x8x15xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb37(%378: f32, %379: f32, %380: f32):
      %381 = arith.divf %378, %379 : f32
      linalg.yield %381 : f32
    } -> tensor<1x4x8x15xf32>
    %382 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %383 = tensor.splat %382 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x4x8x32xf32>
    %384 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%377, %311 : tensor<1x4x8x15xf32>, tensor<1x4x15x32xf32>) outs(%383 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb38(%385: f32, %386: f32, %387: f32):
      %388 = arith.mulf %385, %386 : f32
      %389 = arith.addf %387, %388 : f32
      linalg.yield %389 : f32
    } -> tensor<1x4x8x32xf32>
    %390 = tensor.empty() : tensor<1x8x4x32xf32>
    %391 = linalg.transpose ins(%384:tensor<1x4x8x32xf32>) outs(%390:tensor<1x8x4x32xf32>) permutation = [0, 2, 1, 3]
    %392 = tensor.collapse_shape %391 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x8x4x32xf32> into tensor<1024xf32>
    %393 = tensor.expand_shape %392 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %394 = tensor.empty() : tensor<128x128xf32>
    %395 = linalg.transpose ins(%25:tensor<128x128xf32>) outs(%394:tensor<128x128xf32>) permutation = [1, 0]
    %396 = arith.constant {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %397 = tensor.splat %396 {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32>
    %398 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%393, %395 : tensor<1x8x128xf32>, tensor<128x128xf32>) outs(%397 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb39(%399: f32, %400: f32, %401: f32):
      %402 = arith.mulf %399, %400 : f32
      %403 = arith.addf %401, %402 : f32
      linalg.yield %403 : f32
    } -> tensor<1x8x128xf32>
    %404 = tensor.empty() : tensor<1x8x128xf32>
    %405 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%143, %398 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%404 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb40(%406: f32, %407: f32, %408: f32):
      %409 = arith.addf %406, %407 : f32
      linalg.yield %409 : f32
    } -> tensor<1x8x128xf32>
    %410 = tensor.empty() : tensor<1x8x128xf32>
    %411 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%405 : tensor<1x8x128xf32>) outs(%410 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb41(%412: f32, %413: f32):
      %414 = arith.constant 2.000000e+00 : f32
      %415 = math.powf %412, %414 : f32
      linalg.yield %415 : f32
    } -> tensor<1x8x128xf32>
    %416 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %417 = tensor.splat %416 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %418 = linalg.reduce ins(%411:tensor<1x8x128xf32>) outs(%417:tensor<1x8xf32>) dimensions = [2]
    (%419: f32, %420: f32) {
      %421 = arith.addf %419, %420 : f32
      linalg.yield %421 : f32
    }
    %422 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
    %423 = tensor.splat %422 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %424 = tensor.empty() : tensor<1x8xf32>
    %425 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%418, %423 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%424 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb42(%426: f32, %427: f32, %428: f32):
      %429 = arith.divf %426, %427 : f32
      linalg.yield %429 : f32
    } -> tensor<1x8xf32>
    %430 = tensor.collapse_shape %425 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %431 = tensor.expand_shape %430 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %432 = arith.constant {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
    %433 = tensor.splat %432 {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %434 = tensor.empty() : tensor<1x8x1xf32>
    %435 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%431, %433 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%434 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb43(%436: f32, %437: f32, %438: f32):
      %439 = arith.addf %436, %437 : f32
      linalg.yield %439 : f32
    } -> tensor<1x8x1xf32>
    %440 = tensor.empty() : tensor<1x8x1xf32>
    %441 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%435 : tensor<1x8x1xf32>) outs(%440 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb44(%442: f32, %443: f32):
      %444 = math.rsqrt %442 : f32
      linalg.yield %444 : f32
    } -> tensor<1x8x1xf32>
    %445 = tensor.empty() : tensor<1x8x128xf32>
    %446 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%405, %441 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%445 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb45(%447: f32, %448: f32, %449: f32):
      %450 = arith.mulf %447, %448 : f32
      linalg.yield %450 : f32
    } -> tensor<1x8x128xf32>
    %451 = tensor.empty() : tensor<1x8x128xf32>
    %452 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%446, %26 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%451 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb46(%453: f32, %454: f32, %455: f32):
      %456 = arith.mulf %453, %454 : f32
      linalg.yield %456 : f32
    } -> tensor<1x8x128xf32>
    %457 = tensor.empty() : tensor<128x344xf32>
    %458 = linalg.transpose ins(%27:tensor<344x128xf32>) outs(%457:tensor<128x344xf32>) permutation = [1, 0]
    %459 = arith.constant {prov.region_id = "matmul_6", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %460 = tensor.splat %459 {prov.region_id = "matmul_6", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x344xf32>
    %461 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%452, %458 : tensor<1x8x128xf32>, tensor<128x344xf32>) outs(%460 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "matmul_6", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb47(%462: f32, %463: f32, %464: f32):
      %465 = arith.mulf %462, %463 : f32
      %466 = arith.addf %464, %465 : f32
      linalg.yield %466 : f32
    } -> tensor<1x8x344xf32>
    %467 = tensor.empty() : tensor<128x344xf32>
    %468 = linalg.transpose ins(%28:tensor<344x128xf32>) outs(%467:tensor<128x344xf32>) permutation = [1, 0]
    %469 = arith.constant {prov.region_id = "matmul_7", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %470 = tensor.splat %469 {prov.region_id = "matmul_7", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x344xf32>
    %471 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%452, %468 : tensor<1x8x128xf32>, tensor<128x344xf32>) outs(%470 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "matmul_7", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb48(%472: f32, %473: f32, %474: f32):
      %475 = arith.mulf %472, %473 : f32
      %476 = arith.addf %474, %475 : f32
      linalg.yield %476 : f32
    } -> tensor<1x8x344xf32>
    %477 = tensor.empty() : tensor<1x8x344xf32>
    %478 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%461 : tensor<1x8x344xf32>) outs(%477 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "silu_0", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
    ^bb49(%479: f32, %480: f32):
      %481 = arith.constant 1.000000e+00 : f32
      %482 = arith.negf %479 : f32
      %483 = math.exp %482 : f32
      %484 = arith.addf %481, %483 : f32
      %485 = arith.divf %481, %484 : f32
      %486 = arith.mulf %479, %485 : f32
      linalg.yield %486 : f32
    } -> tensor<1x8x344xf32>
    %487 = tensor.empty() : tensor<1x8x344xf32>
    %488 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%478, %471 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%487 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb50(%489: f32, %490: f32, %491: f32):
      %492 = arith.mulf %489, %490 : f32
      linalg.yield %492 : f32
    } -> tensor<1x8x344xf32>
    %493 = tensor.empty() : tensor<344x128xf32>
    %494 = linalg.transpose ins(%29:tensor<128x344xf32>) outs(%493:tensor<344x128xf32>) permutation = [1, 0]
    %495 = arith.constant {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %496 = tensor.splat %495 {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32>
    %497 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%488, %494 : tensor<1x8x344xf32>, tensor<344x128xf32>) outs(%496 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb51(%498: f32, %499: f32, %500: f32):
      %501 = arith.mulf %498, %499 : f32
      %502 = arith.addf %500, %501 : f32
      linalg.yield %502 : f32
    } -> tensor<1x8x128xf32>
    %503 = tensor.empty() : tensor<1x8x128xf32>
    %504 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%405, %497 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%503 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb52(%505: f32, %506: f32, %507: f32):
      %508 = arith.addf %505, %506 : f32
      linalg.yield %508 : f32
    } -> tensor<1x8x128xf32>
    %509 = tensor.empty() : tensor<1x8x128xf32>
    %510 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%504 : tensor<1x8x128xf32>) outs(%509 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb53(%511: f32, %512: f32):
      %513 = arith.constant 2.000000e+00 : f32
      %514 = math.powf %511, %513 : f32
      linalg.yield %514 : f32
    } -> tensor<1x8x128xf32>
    %515 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %516 = tensor.splat %515 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %517 = linalg.reduce ins(%510:tensor<1x8x128xf32>) outs(%516:tensor<1x8xf32>) dimensions = [2]
    (%518: f32, %519: f32) {
      %520 = arith.addf %518, %519 : f32
      linalg.yield %520 : f32
    }
    %521 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
    %522 = tensor.splat %521 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %523 = tensor.empty() : tensor<1x8xf32>
    %524 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%517, %522 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%523 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb54(%525: f32, %526: f32, %527: f32):
      %528 = arith.divf %525, %526 : f32
      linalg.yield %528 : f32
    } -> tensor<1x8xf32>
    %529 = tensor.collapse_shape %524 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %530 = tensor.expand_shape %529 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %531 = arith.constant {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
    %532 = tensor.splat %531 {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %533 = tensor.empty() : tensor<1x8x1xf32>
    %534 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%530, %532 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%533 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb55(%535: f32, %536: f32, %537: f32):
      %538 = arith.addf %535, %536 : f32
      linalg.yield %538 : f32
    } -> tensor<1x8x1xf32>
    %539 = tensor.empty() : tensor<1x8x1xf32>
    %540 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%534 : tensor<1x8x1xf32>) outs(%539 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb56(%541: f32, %542: f32):
      %543 = math.rsqrt %541 : f32
      linalg.yield %543 : f32
    } -> tensor<1x8x1xf32>
    %544 = tensor.empty() : tensor<1x8x128xf32>
    %545 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%504, %540 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%544 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb57(%546: f32, %547: f32, %548: f32):
      %549 = arith.mulf %546, %547 : f32
      linalg.yield %549 : f32
    } -> tensor<1x8x128xf32>
    %550 = tensor.empty() : tensor<1x8x128xf32>
    %551 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%545, %30 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%550 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb58(%552: f32, %553: f32, %554: f32):
      %555 = arith.mulf %552, %553 : f32
      linalg.yield %555 : f32
    } -> tensor<1x8x128xf32>
    %556 = tensor.empty() : tensor<128x128xf32>
    %557 = linalg.transpose ins(%31:tensor<128x128xf32>) outs(%556:tensor<128x128xf32>) permutation = [1, 0]
    %558 = arith.constant {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %559 = tensor.splat %558 {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32>
    %560 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%551, %557 : tensor<1x8x128xf32>, tensor<128x128xf32>) outs(%559 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb59(%561: f32, %562: f32, %563: f32):
      %564 = arith.mulf %561, %562 : f32
      %565 = arith.addf %563, %564 : f32
      linalg.yield %565 : f32
    } -> tensor<1x8x128xf32>
    %566 = tensor.collapse_shape %560 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %567 = tensor.expand_shape %566 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %568 = tensor.empty() : tensor<1x4x8x32xf32>
    %569 = linalg.transpose ins(%567:tensor<1x8x4x32xf32>) outs(%568:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %570 = tensor.empty() : tensor<128x128xf32>
    %571 = linalg.transpose ins(%32:tensor<128x128xf32>) outs(%570:tensor<128x128xf32>) permutation = [1, 0]
    %572 = arith.constant {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %573 = tensor.splat %572 {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32>
    %574 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%551, %571 : tensor<1x8x128xf32>, tensor<128x128xf32>) outs(%573 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb60(%575: f32, %576: f32, %577: f32):
      %578 = arith.mulf %575, %576 : f32
      %579 = arith.addf %577, %578 : f32
      linalg.yield %579 : f32
    } -> tensor<1x8x128xf32>
    %580 = tensor.collapse_shape %574 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %581 = tensor.expand_shape %580 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %582 = tensor.empty() : tensor<1x4x8x32xf32>
    %583 = linalg.transpose ins(%581:tensor<1x8x4x32xf32>) outs(%582:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %584 = tensor.empty() : tensor<128x128xf32>
    %585 = linalg.transpose ins(%33:tensor<128x128xf32>) outs(%584:tensor<128x128xf32>) permutation = [1, 0]
    %586 = arith.constant {prov.region_id = "matmul_11", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %587 = tensor.splat %586 {prov.region_id = "matmul_11", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32>
    %588 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%551, %585 : tensor<1x8x128xf32>, tensor<128x128xf32>) outs(%587 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "matmul_11", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb61(%589: f32, %590: f32, %591: f32):
      %592 = arith.mulf %589, %590 : f32
      %593 = arith.addf %591, %592 : f32
      linalg.yield %593 : f32
    } -> tensor<1x8x128xf32>
    %594 = tensor.collapse_shape %588 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %595 = tensor.expand_shape %594 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 32] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x8x4x32xf32>
    %596 = tensor.empty() : tensor<1x4x8x32xf32>
    %597 = linalg.transpose ins(%595:tensor<1x8x4x32xf32>) outs(%596:tensor<1x4x8x32xf32>) permutation = [0, 2, 1, 3]
    %598 = tensor.collapse_shape %149 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8x32xf32> into tensor<256xf32>
    %599 = tensor.expand_shape %598 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x8x32xf32>
    %600 = tensor.collapse_shape %599 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x32xf32> into tensor<256xf32>
    %601 = tensor.expand_shape %600 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %602 = tensor.collapse_shape %150 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8x32xf32> into tensor<256xf32>
    %603 = tensor.expand_shape %602 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x8x32xf32>
    %604 = tensor.collapse_shape %603 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x32xf32> into tensor<256xf32>
    %605 = tensor.expand_shape %604 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %606 = "tensor.extract_slice"(%569) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %607 = "tensor.extract_slice"(%569) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %608 = tensor.empty() : tensor<1x4x8x16xf32>
    %609 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%607 : tensor<1x4x8x16xf32>) outs(%608 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb62(%610: f32, %611: f32):
      %612 = arith.negf %610 : f32
      linalg.yield %612 : f32
    } -> tensor<1x4x8x16xf32>
    %613 = tensor.concat dim(3) %609, %606 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %614 = tensor.empty() : tensor<1x4x8x32xf32>
    %615 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%569, %601 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%614 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb63(%616: f32, %617: f32, %618: f32):
      %619 = arith.mulf %616, %617 : f32
      linalg.yield %619 : f32
    } -> tensor<1x4x8x32xf32>
    %620 = tensor.empty() : tensor<1x4x8x32xf32>
    %621 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%613, %605 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%620 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb64(%622: f32, %623: f32, %624: f32):
      %625 = arith.mulf %622, %623 : f32
      linalg.yield %625 : f32
    } -> tensor<1x4x8x32xf32>
    %626 = tensor.empty() : tensor<1x4x8x32xf32>
    %627 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%615, %621 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%626 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb65(%628: f32, %629: f32, %630: f32):
      %631 = arith.addf %628, %629 : f32
      linalg.yield %631 : f32
    } -> tensor<1x4x8x32xf32>
    %632 = tensor.collapse_shape %149 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8x32xf32> into tensor<256xf32>
    %633 = tensor.expand_shape %632 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x8x32xf32>
    %634 = tensor.collapse_shape %633 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x32xf32> into tensor<256xf32>
    %635 = tensor.expand_shape %634 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %636 = tensor.collapse_shape %150 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8x32xf32> into tensor<256xf32>
    %637 = tensor.expand_shape %636 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 32] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x8x32xf32>
    %638 = tensor.collapse_shape %637 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x32xf32> into tensor<256xf32>
    %639 = tensor.expand_shape %638 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x8x32xf32>
    %640 = "tensor.extract_slice"(%583) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %641 = "tensor.extract_slice"(%583) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 8, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x32xf32>) -> tensor<1x4x8x16xf32>
    %642 = tensor.empty() : tensor<1x4x8x16xf32>
    %643 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%641 : tensor<1x4x8x16xf32>) outs(%642 : tensor<1x4x8x16xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb66(%644: f32, %645: f32):
      %646 = arith.negf %644 : f32
      linalg.yield %646 : f32
    } -> tensor<1x4x8x16xf32>
    %647 = tensor.concat dim(3) %643, %640 {prov.region_id = "cat_5", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x8x16xf32>, tensor<1x4x8x16xf32>) -> tensor<1x4x8x32xf32>
    %648 = tensor.empty() : tensor<1x4x8x32xf32>
    %649 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%583, %635 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%648 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb67(%650: f32, %651: f32, %652: f32):
      %653 = arith.mulf %650, %651 : f32
      linalg.yield %653 : f32
    } -> tensor<1x4x8x32xf32>
    %654 = tensor.empty() : tensor<1x4x8x32xf32>
    %655 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%647, %639 : tensor<1x4x8x32xf32>, tensor<1x1x8x32xf32>) outs(%654 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb68(%656: f32, %657: f32, %658: f32):
      %659 = arith.mulf %656, %657 : f32
      linalg.yield %659 : f32
    } -> tensor<1x4x8x32xf32>
    %660 = tensor.empty() : tensor<1x4x8x32xf32>
    %661 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%649, %655 : tensor<1x4x8x32xf32>, tensor<1x4x8x32xf32>) outs(%660 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb69(%662: f32, %663: f32, %664: f32):
      %665 = arith.addf %662, %663 : f32
      linalg.yield %665 : f32
    } -> tensor<1x4x8x32xf32>
    %666 = "tensor.extract_slice"(%131) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 15, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x15x32xf32>) -> tensor<4x15x32xf32>
    %667 = func.call @aten_index_copy_default(%666, %134, %661) {prov.region_id = "aten_index_copy_default_2", prov.dispatch_id = "aten_index_copy_default_2"} : (tensor<4x15x32xf32>, tensor<8xi64>, tensor<1x4x8x32xf32>) -> tensor<1x4x15x32xf32>
    %668 = "tensor.extract_slice"(%132) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 15, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x15x32xf32>) -> tensor<4x15x32xf32>
    %669 = func.call @aten_index_copy_default(%668, %134, %597) {prov.region_id = "aten_index_copy_default_3", prov.dispatch_id = "aten_index_copy_default_3"} : (tensor<4x15x32xf32>, tensor<8xi64>, tensor<1x4x8x32xf32>) -> tensor<1x4x15x32xf32>
    %670 = tensor.empty() : tensor<1x4x32x15xf32>
    %671 = linalg.transpose ins(%667:tensor<1x4x15x32xf32>) outs(%670:tensor<1x4x32x15xf32>) permutation = [0, 1, 3, 2]
    %672 = arith.constant {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %673 = tensor.splat %672 {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x4x8x15xf32>
    %674 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%627, %671 : tensor<1x4x8x32xf32>, tensor<1x4x32x15xf32>) outs(%673 : tensor<1x4x8x15xf32>) attrs =  {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb70(%675: f32, %676: f32, %677: f32):
      %678 = arith.mulf %675, %676 : f32
      %679 = arith.addf %677, %678 : f32
      linalg.yield %679 : f32
    } -> tensor<1x4x8x15xf32>
    %680 = arith.constant {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 5.65685415 : f32
    %681 = tensor.splat %680 {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x4x8x15xf32>
    %682 = tensor.empty() : tensor<1x4x8x15xf32>
    %683 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%674, %681 : tensor<1x4x8x15xf32>, tensor<1x4x8x15xf32>) outs(%682 : tensor<1x4x8x15xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb71(%684: f32, %685: f32, %686: f32):
      %687 = arith.divf %684, %685 : f32
      linalg.yield %687 : f32
    } -> tensor<1x4x8x15xf32>
    %688 = tensor.expand_shape %134 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<8xi64> into tensor<8x1xi64>
    %689 = tensor.expand_shape %123 [[0 : i64, 1 : i64]] output_shape [1, 15] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<15xi64> into tensor<1x15xi64>
    %690 = tensor.empty() : tensor<8x15xi1>
    %691 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%689, %688 : tensor<1x15xi64>, tensor<8x1xi64>) outs(%690 : tensor<8x15xi1>) attrs =  {prov.region_id = "compare_1", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
    ^bb72(%692: i64, %693: i64, %694: i1):
      %695 = arith.cmpi sle, %692, %693 : i64
      linalg.yield %695 : i1
    } -> tensor<8x15xi1>
    %696 = tensor.collapse_shape %691 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<8x15xi1> into tensor<120xi1>
    %697 = tensor.expand_shape %696 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 15] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<120xi1> into tensor<1x8x15xi1>
    %698 = tensor.collapse_shape %697 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x8x15xi1> into tensor<120xi1>
    %699 = tensor.expand_shape %698 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 15] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<120xi1> into tensor<1x1x8x15xi1>
    %700 = tensor.empty() : tensor<1x1x8x15xi1>
    %701 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%699 : tensor<1x1x8x15xi1>) outs(%700 : tensor<1x1x8x15xi1>) attrs =  {prov.region_id = "bitwise_1", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
    ^bb73(%702: i1, %703: i1):
      %704 = arith.constant true
      %705 = arith.xori %702, %704 : i1
      linalg.yield %705 : i1
    } -> tensor<1x1x8x15xi1>
    %706 = func.call @aten_masked_fill_Scalar(%683, %701) {prov.region_id = "aten_masked_fill_Scalar_1", prov.dispatch_id = "aten_masked_fill_Scalar_1"} : (tensor<1x4x8x15xf32>, tensor<1x1x8x15xi1>) -> tensor<1x4x8x15xf32>
    %707 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0xff800000 : f32
    %708 = tensor.splat %707 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x8xf32>
    %709 = linalg.reduce ins(%706:tensor<1x4x8x15xf32>) outs(%708:tensor<1x4x8xf32>) dimensions = [3]
    (%710: f32, %711: f32) {
      %712 = arith.maximumf %710, %711 : f32
      linalg.yield %712 : f32
    }
    %713 = tensor.collapse_shape %709 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x8xf32> into tensor<32xf32>
    %714 = tensor.expand_shape %713 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %715 = tensor.empty() : tensor<1x4x8x15xf32>
    %716 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%706, %714 : tensor<1x4x8x15xf32>, tensor<1x4x8x1xf32>) outs(%715 : tensor<1x4x8x15xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb74(%717: f32, %718: f32, %719: f32):
      %720 = arith.subf %717, %718 : f32
      linalg.yield %720 : f32
    } -> tensor<1x4x8x15xf32>
    %721 = tensor.empty() : tensor<1x4x8x15xf32>
    %722 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%716 : tensor<1x4x8x15xf32>) outs(%721 : tensor<1x4x8x15xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb75(%723: f32, %724: f32):
      %725 = math.exp %723 : f32
      linalg.yield %725 : f32
    } -> tensor<1x4x8x15xf32>
    %726 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %727 = tensor.splat %726 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x8xf32>
    %728 = linalg.reduce ins(%722:tensor<1x4x8x15xf32>) outs(%727:tensor<1x4x8xf32>) dimensions = [3]
    (%729: f32, %730: f32) {
      %731 = arith.addf %729, %730 : f32
      linalg.yield %731 : f32
    }
    %732 = tensor.collapse_shape %728 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x8xf32> into tensor<32xf32>
    %733 = tensor.expand_shape %732 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 8, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x4x8x1xf32>
    %734 = tensor.empty() : tensor<1x4x8x15xf32>
    %735 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%722, %733 : tensor<1x4x8x15xf32>, tensor<1x4x8x1xf32>) outs(%734 : tensor<1x4x8x15xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb76(%736: f32, %737: f32, %738: f32):
      %739 = arith.divf %736, %737 : f32
      linalg.yield %739 : f32
    } -> tensor<1x4x8x15xf32>
    %740 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %741 = tensor.splat %740 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x4x8x32xf32>
    %742 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%735, %669 : tensor<1x4x8x15xf32>, tensor<1x4x15x32xf32>) outs(%741 : tensor<1x4x8x32xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb77(%743: f32, %744: f32, %745: f32):
      %746 = arith.mulf %743, %744 : f32
      %747 = arith.addf %745, %746 : f32
      linalg.yield %747 : f32
    } -> tensor<1x4x8x32xf32>
    %748 = tensor.empty() : tensor<1x8x4x32xf32>
    %749 = linalg.transpose ins(%742:tensor<1x4x8x32xf32>) outs(%748:tensor<1x8x4x32xf32>) permutation = [0, 2, 1, 3]
    %750 = tensor.collapse_shape %749 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x8x4x32xf32> into tensor<1024xf32>
    %751 = tensor.expand_shape %750 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 128] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x8x128xf32>
    %752 = tensor.empty() : tensor<128x128xf32>
    %753 = linalg.transpose ins(%34:tensor<128x128xf32>) outs(%752:tensor<128x128xf32>) permutation = [1, 0]
    %754 = arith.constant {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %755 = tensor.splat %754 {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32>
    %756 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%751, %753 : tensor<1x8x128xf32>, tensor<128x128xf32>) outs(%755 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb78(%757: f32, %758: f32, %759: f32):
      %760 = arith.mulf %757, %758 : f32
      %761 = arith.addf %759, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<1x8x128xf32>
    %762 = tensor.empty() : tensor<1x8x128xf32>
    %763 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%504, %756 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%762 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb79(%764: f32, %765: f32, %766: f32):
      %767 = arith.addf %764, %765 : f32
      linalg.yield %767 : f32
    } -> tensor<1x8x128xf32>
    %768 = tensor.empty() : tensor<1x8x128xf32>
    %769 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%763 : tensor<1x8x128xf32>) outs(%768 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb80(%770: f32, %771: f32):
      %772 = arith.constant 2.000000e+00 : f32
      %773 = math.powf %770, %772 : f32
      linalg.yield %773 : f32
    } -> tensor<1x8x128xf32>
    %774 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %775 = tensor.splat %774 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %776 = linalg.reduce ins(%769:tensor<1x8x128xf32>) outs(%775:tensor<1x8xf32>) dimensions = [2]
    (%777: f32, %778: f32) {
      %779 = arith.addf %777, %778 : f32
      linalg.yield %779 : f32
    }
    %780 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
    %781 = tensor.splat %780 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %782 = tensor.empty() : tensor<1x8xf32>
    %783 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%776, %781 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%782 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb81(%784: f32, %785: f32, %786: f32):
      %787 = arith.divf %784, %785 : f32
      linalg.yield %787 : f32
    } -> tensor<1x8xf32>
    %788 = tensor.collapse_shape %783 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %789 = tensor.expand_shape %788 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %790 = arith.constant {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
    %791 = tensor.splat %790 {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %792 = tensor.empty() : tensor<1x8x1xf32>
    %793 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%789, %791 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%792 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb82(%794: f32, %795: f32, %796: f32):
      %797 = arith.addf %794, %795 : f32
      linalg.yield %797 : f32
    } -> tensor<1x8x1xf32>
    %798 = tensor.empty() : tensor<1x8x1xf32>
    %799 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%793 : tensor<1x8x1xf32>) outs(%798 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb83(%800: f32, %801: f32):
      %802 = math.rsqrt %800 : f32
      linalg.yield %802 : f32
    } -> tensor<1x8x1xf32>
    %803 = tensor.empty() : tensor<1x8x128xf32>
    %804 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%763, %799 : tensor<1x8x128xf32>, tensor<1x8x1xf32>) outs(%803 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb84(%805: f32, %806: f32, %807: f32):
      %808 = arith.mulf %805, %806 : f32
      linalg.yield %808 : f32
    } -> tensor<1x8x128xf32>
    %809 = tensor.empty() : tensor<1x8x128xf32>
    %810 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%804, %35 : tensor<1x8x128xf32>, tensor<128xf32>) outs(%809 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb85(%811: f32, %812: f32, %813: f32):
      %814 = arith.mulf %811, %812 : f32
      linalg.yield %814 : f32
    } -> tensor<1x8x128xf32>
    %815 = tensor.empty() : tensor<128x344xf32>
    %816 = linalg.transpose ins(%36:tensor<344x128xf32>) outs(%815:tensor<128x344xf32>) permutation = [1, 0]
    %817 = arith.constant {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %818 = tensor.splat %817 {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x344xf32>
    %819 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%810, %816 : tensor<1x8x128xf32>, tensor<128x344xf32>) outs(%818 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb86(%820: f32, %821: f32, %822: f32):
      %823 = arith.mulf %820, %821 : f32
      %824 = arith.addf %822, %823 : f32
      linalg.yield %824 : f32
    } -> tensor<1x8x344xf32>
    %825 = tensor.empty() : tensor<128x344xf32>
    %826 = linalg.transpose ins(%37:tensor<344x128xf32>) outs(%825:tensor<128x344xf32>) permutation = [1, 0]
    %827 = arith.constant {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %828 = tensor.splat %827 {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x344xf32>
    %829 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%810, %826 : tensor<1x8x128xf32>, tensor<128x344xf32>) outs(%828 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb87(%830: f32, %831: f32, %832: f32):
      %833 = arith.mulf %830, %831 : f32
      %834 = arith.addf %832, %833 : f32
      linalg.yield %834 : f32
    } -> tensor<1x8x344xf32>
    %835 = tensor.empty() : tensor<1x8x344xf32>
    %836 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%819 : tensor<1x8x344xf32>) outs(%835 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "silu_1", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
    ^bb88(%837: f32, %838: f32):
      %839 = arith.constant 1.000000e+00 : f32
      %840 = arith.negf %837 : f32
      %841 = math.exp %840 : f32
      %842 = arith.addf %839, %841 : f32
      %843 = arith.divf %839, %842 : f32
      %844 = arith.mulf %837, %843 : f32
      linalg.yield %844 : f32
    } -> tensor<1x8x344xf32>
    %845 = tensor.empty() : tensor<1x8x344xf32>
    %846 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%836, %829 : tensor<1x8x344xf32>, tensor<1x8x344xf32>) outs(%845 : tensor<1x8x344xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb89(%847: f32, %848: f32, %849: f32):
      %850 = arith.mulf %847, %848 : f32
      linalg.yield %850 : f32
    } -> tensor<1x8x344xf32>
    %851 = tensor.empty() : tensor<344x128xf32>
    %852 = linalg.transpose ins(%38:tensor<128x344xf32>) outs(%851:tensor<344x128xf32>) permutation = [1, 0]
    %853 = arith.constant {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %854 = tensor.splat %853 {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32>
    %855 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%846, %852 : tensor<1x8x344xf32>, tensor<344x128xf32>) outs(%854 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb90(%856: f32, %857: f32, %858: f32):
      %859 = arith.mulf %856, %857 : f32
      %860 = arith.addf %858, %859 : f32
      linalg.yield %860 : f32
    } -> tensor<1x8x128xf32>
    %861 = tensor.empty() : tensor<1x8x128xf32>
    %862 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%763, %855 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%861 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb91(%863: f32, %864: f32, %865: f32):
      %866 = arith.addf %863, %864 : f32
      linalg.yield %866 : f32
    } -> tensor<1x8x128xf32>
    %867 = func.call @aten_stack_default(%309, %667) {prov.region_id = "aten_stack_default_0", prov.dispatch_id = "aten_stack_default_0"} : (tensor<1x4x15x32xf32>, tensor<1x4x15x32xf32>) -> tensor<2x1x4x15x32xf32>
    %868 = func.call @aten_stack_default(%311, %669) {prov.region_id = "aten_stack_default_1", prov.dispatch_id = "aten_stack_default_1"} : (tensor<1x4x15x32xf32>, tensor<1x4x15x32xf32>) -> tensor<2x1x4x15x32xf32>
    %869 = "tensor.extract_slice"(%862) <{static_offsets = array<i64: 0, 7, 0>, static_sizes = array<i64: 1, 1, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x128xf32>) -> tensor<1x1x128xf32>
    %870 = tensor.empty() : tensor<1x1x128xf32>
    %871 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%869 : tensor<1x1x128xf32>) outs(%870 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb92(%872: f32, %873: f32):
      %874 = arith.constant 2.000000e+00 : f32
      %875 = math.powf %872, %874 : f32
      linalg.yield %875 : f32
    } -> tensor<1x1x128xf32>
    %876 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %877 = tensor.splat %876 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
    %878 = linalg.reduce ins(%871:tensor<1x1x128xf32>) outs(%877:tensor<1x1xf32>) dimensions = [2]
    (%879: f32, %880: f32) {
      %881 = arith.addf %879, %880 : f32
      linalg.yield %881 : f32
    }
    %882 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
    %883 = tensor.splat %882 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
    %884 = tensor.empty() : tensor<1x1xf32>
    %885 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%878, %883 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%884 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb93(%886: f32, %887: f32, %888: f32):
      %889 = arith.divf %886, %887 : f32
      linalg.yield %889 : f32
    } -> tensor<1x1xf32>
    %890 = tensor.collapse_shape %885 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
    %891 = tensor.expand_shape %890 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
    %892 = arith.constant {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
    %893 = tensor.splat %892 {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
    %894 = tensor.empty() : tensor<1x1x1xf32>
    %895 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%891, %893 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%894 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb94(%896: f32, %897: f32, %898: f32):
      %899 = arith.addf %896, %897 : f32
      linalg.yield %899 : f32
    } -> tensor<1x1x1xf32>
    %900 = tensor.empty() : tensor<1x1x1xf32>
    %901 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%895 : tensor<1x1x1xf32>) outs(%900 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb95(%902: f32, %903: f32):
      %904 = math.rsqrt %902 : f32
      linalg.yield %904 : f32
    } -> tensor<1x1x1xf32>
    %905 = tensor.empty() : tensor<1x1x128xf32>
    %906 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%869, %901 : tensor<1x1x128xf32>, tensor<1x1x1xf32>) outs(%905 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb96(%907: f32, %908: f32, %909: f32):
      %910 = arith.mulf %907, %908 : f32
      linalg.yield %910 : f32
    } -> tensor<1x1x128xf32>
    %911 = tensor.empty() : tensor<1x1x128xf32>
    %912 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%906, %39 : tensor<1x1x128xf32>, tensor<128xf32>) outs(%911 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb97(%913: f32, %914: f32, %915: f32):
      %916 = arith.mulf %913, %914 : f32
      linalg.yield %916 : f32
    } -> tensor<1x1x128xf32>
    %917 = tensor.empty() : tensor<128x256xf32>
    %918 = linalg.transpose ins(%41:tensor<256x128xf32>) outs(%917:tensor<128x256xf32>) permutation = [1, 0]
    %919 = arith.constant {prov.region_id = "matmul_18", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %920 = tensor.splat %919 {prov.region_id = "matmul_18", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x1x256xf32>
    %921 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%912, %918 : tensor<1x1x128xf32>, tensor<128x256xf32>) outs(%920 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "matmul_18", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb98(%922: f32, %923: f32, %924: f32):
      %925 = arith.mulf %922, %923 : f32
      %926 = arith.addf %924, %925 : f32
      linalg.yield %926 : f32
    } -> tensor<1x1x256xf32>
    %927 = "tensor.extract_slice"(%921) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 256>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<1x1x256xf32>) -> tensor<256xf32>
    %928 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0xff800000 : f32
    %929 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0 : i64
    %930 = tensor.splat %928 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<f32>
    %931 = tensor.splat %929 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<i64>
    %932, %933 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%927 : tensor<256xf32>) outs(%930, %931 : tensor<f32>, tensor<i64>) attrs =  {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} {
    ^bb99(%934: f32, %935: f32, %936: i64):
      %937 = linalg.index 0 : index
      %938 = arith.index_cast %937 : index to i64
      %939 = arith.cmpf ogt, %934, %935 : f32
      %940 = arith.select %939, %934, %935 : f32
      %941 = arith.select %939, %938, %936 : i64
      linalg.yield %940, %941 : f32, i64
    } -> (tensor<f32>, tensor<i64>)
    %942 = tensor.extract %932[] : tensor<f32>
    %943 = tensor.from_elements %942 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xf32>
    %944 = tensor.extract %933[] : tensor<i64>
    %945 = tensor.from_elements %944 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xi64>
    %946 = func.call @aten_zeros_default_1() {prov.region_id = "aten_zeros_default_1_0", prov.dispatch_id = "aten_zeros_default_1_0"} : () -> tensor<i64>
    %947 = func.call @aten_zeros_default_2() {prov.region_id = "aten_zeros_default_2_0", prov.dispatch_id = "aten_zeros_default_2_0"} : () -> tensor<1x7xi64>
    %948 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 0 : index
    %949 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 7 : index
    %950 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 1 : index
    %951, %952, %953, %954, %955 = scf.for %956 = %948 to %949 step %950 iter_args(%957 = %946, %958 = %945, %959 = %947, %960 = %867, %961 = %868) -> (tensor<i64>, tensor<1xi64>, tensor<1x7xi64>, tensor<2x1x4x15x32xf32>, tensor<2x1x4x15x32xf32>) {
      %962 = tensor.extract %957[] : tensor<i64>
      %963 = tensor.from_elements %962 {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1xi64>
      %964 = func.call @aten_index_copy_default_wl0(%959, %963, %958) {prov.region_id = "aten_index_copy_default_0", prov.dispatch_id = "aten_index_copy_default_0"} : (tensor<1x7xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x7xi64>
      %965 = tensor.empty() : tensor<i64>
      %966 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%42, %957 : tensor<i64>, tensor<i64>) outs(%965 : tensor<i64>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb100(%967: i64, %968: i64, %969: i64):
        %970 = arith.addi %967, %968 : i64
        linalg.yield %970 : i64
      } -> tensor<i64>
      %971 = tensor.extract %966[] : tensor<i64>
      %972 = tensor.from_elements %971 {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1xi64>
      %973 = tensor.empty() : tensor<1x1x128xf32>
      %974 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%958 : tensor<1xi64>) outs(%973 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32"} {
      ^bb101(%975: i64, %976: f32):
        %977 = arith.index_cast %975 : i64 to index
        %978 = linalg.index 2 : index
        %979 = tensor.extract %40[%977, %978] : tensor<256x128xf32>
        linalg.yield %979 : f32
      } -> tensor<1x1x128xf32>
      %980 = func.call @aten_index_select_default_wl1(%110, %972) {prov.region_id = "aten_index_select_default_0", prov.dispatch_id = "aten_index_select_default_0"} : (tensor<15x32xf32>, tensor<1xi64>) -> tensor<1x32xf32>
      %981 = func.call @aten_index_select_default_wl1(%121, %972) {prov.region_id = "aten_index_select_default_1", prov.dispatch_id = "aten_index_select_default_1"} : (tensor<15x32xf32>, tensor<1xi64>) -> tensor<1x32xf32>
      %982 = tensor.empty() : tensor<1x1x128xf32>
      %983 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%974 : tensor<1x1x128xf32>) outs(%982 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb102(%984: f32, %985: f32):
        %986 = arith.constant 2.000000e+00 : f32
        %987 = math.powf %984, %986 : f32
        linalg.yield %987 : f32
      } -> tensor<1x1x128xf32>
      %988 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %989 = tensor.splat %988 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %990 = linalg.reduce ins(%983:tensor<1x1x128xf32>) outs(%989:tensor<1x1xf32>) dimensions = [2]
      (%991: f32, %992: f32) {
        %993 = arith.addf %991, %992 : f32
        linalg.yield %993 : f32
      }
      %994 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %995 = tensor.splat %994 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %996 = tensor.empty() : tensor<1x1xf32>
      %997 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%990, %995 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%996 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb103(%998: f32, %999: f32, %1000: f32):
        %1001 = arith.divf %998, %999 : f32
        linalg.yield %1001 : f32
      } -> tensor<1x1xf32>
      %1002 = tensor.collapse_shape %997 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %1003 = tensor.expand_shape %1002 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %1004 = arith.constant {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %1005 = tensor.splat %1004 {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %1006 = tensor.empty() : tensor<1x1x1xf32>
      %1007 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1003, %1005 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%1006 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb104(%1008: f32, %1009: f32, %1010: f32):
        %1011 = arith.addf %1008, %1009 : f32
        linalg.yield %1011 : f32
      } -> tensor<1x1x1xf32>
      %1012 = tensor.empty() : tensor<1x1x1xf32>
      %1013 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1007 : tensor<1x1x1xf32>) outs(%1012 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb105(%1014: f32, %1015: f32):
        %1016 = math.rsqrt %1014 : f32
        linalg.yield %1016 : f32
      } -> tensor<1x1x1xf32>
      %1017 = tensor.empty() : tensor<1x1x128xf32>
      %1018 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%974, %1013 : tensor<1x1x128xf32>, tensor<1x1x1xf32>) outs(%1017 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb106(%1019: f32, %1020: f32, %1021: f32):
        %1022 = arith.mulf %1019, %1020 : f32
        linalg.yield %1022 : f32
      } -> tensor<1x1x128xf32>
      %1023 = tensor.empty() : tensor<1x1x128xf32>
      %1024 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1018, %21 : tensor<1x1x128xf32>, tensor<128xf32>) outs(%1023 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb107(%1025: f32, %1026: f32, %1027: f32):
        %1028 = arith.mulf %1025, %1026 : f32
        linalg.yield %1028 : f32
      } -> tensor<1x1x128xf32>
      %1029 = tensor.empty() : tensor<128x128xf32>
      %1030 = linalg.transpose ins(%22:tensor<128x128xf32>) outs(%1029:tensor<128x128xf32>) permutation = [1, 0]
      %1031 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1032 = tensor.splat %1031 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32>
      %1033 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1024, %1030 : tensor<1x1x128xf32>, tensor<128x128xf32>) outs(%1032 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb108(%1034: f32, %1035: f32, %1036: f32):
        %1037 = arith.mulf %1034, %1035 : f32
        %1038 = arith.addf %1036, %1037 : f32
        linalg.yield %1038 : f32
      } -> tensor<1x1x128xf32>
      %1039 = tensor.collapse_shape %1033 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %1040 = tensor.expand_shape %1039 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 32] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x4x32xf32>
      %1041 = tensor.empty() : tensor<1x4x1x32xf32>
      %1042 = linalg.transpose ins(%1040:tensor<1x1x4x32xf32>) outs(%1041:tensor<1x4x1x32xf32>) permutation = [0, 2, 1, 3]
      %1043 = tensor.empty() : tensor<128x128xf32>
      %1044 = linalg.transpose ins(%23:tensor<128x128xf32>) outs(%1043:tensor<128x128xf32>) permutation = [1, 0]
      %1045 = arith.constant {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1046 = tensor.splat %1045 {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32>
      %1047 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1024, %1044 : tensor<1x1x128xf32>, tensor<128x128xf32>) outs(%1046 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "matmul_1", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb109(%1048: f32, %1049: f32, %1050: f32):
        %1051 = arith.mulf %1048, %1049 : f32
        %1052 = arith.addf %1050, %1051 : f32
        linalg.yield %1052 : f32
      } -> tensor<1x1x128xf32>
      %1053 = tensor.collapse_shape %1047 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %1054 = tensor.expand_shape %1053 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 32] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x4x32xf32>
      %1055 = tensor.empty() : tensor<1x4x1x32xf32>
      %1056 = linalg.transpose ins(%1054:tensor<1x1x4x32xf32>) outs(%1055:tensor<1x4x1x32xf32>) permutation = [0, 2, 1, 3]
      %1057 = tensor.empty() : tensor<128x128xf32>
      %1058 = linalg.transpose ins(%24:tensor<128x128xf32>) outs(%1057:tensor<128x128xf32>) permutation = [1, 0]
      %1059 = arith.constant {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1060 = tensor.splat %1059 {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32>
      %1061 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1024, %1058 : tensor<1x1x128xf32>, tensor<128x128xf32>) outs(%1060 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb110(%1062: f32, %1063: f32, %1064: f32):
        %1065 = arith.mulf %1062, %1063 : f32
        %1066 = arith.addf %1064, %1065 : f32
        linalg.yield %1066 : f32
      } -> tensor<1x1x128xf32>
      %1067 = tensor.collapse_shape %1061 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %1068 = tensor.expand_shape %1067 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 32] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x4x32xf32>
      %1069 = tensor.empty() : tensor<1x4x1x32xf32>
      %1070 = linalg.transpose ins(%1068:tensor<1x1x4x32xf32>) outs(%1069:tensor<1x4x1x32xf32>) permutation = [0, 2, 1, 3]
      %1071 = tensor.collapse_shape %980 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
      %1072 = tensor.expand_shape %1071 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x32xf32>
      %1073 = tensor.collapse_shape %1072 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x32xf32> into tensor<32xf32>
      %1074 = tensor.expand_shape %1073 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x1x32xf32>
      %1075 = tensor.collapse_shape %981 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
      %1076 = tensor.expand_shape %1075 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x32xf32>
      %1077 = tensor.collapse_shape %1076 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x32xf32> into tensor<32xf32>
      %1078 = tensor.expand_shape %1077 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x1x32xf32>
      %1079 = "tensor.extract_slice"(%1042) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x32xf32>) -> tensor<1x4x1x16xf32>
      %1080 = "tensor.extract_slice"(%1042) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 1, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x32xf32>) -> tensor<1x4x1x16xf32>
      %1081 = tensor.empty() : tensor<1x4x1x16xf32>
      %1082 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1080 : tensor<1x4x1x16xf32>) outs(%1081 : tensor<1x4x1x16xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb111(%1083: f32, %1084: f32):
        %1085 = arith.negf %1083 : f32
        linalg.yield %1085 : f32
      } -> tensor<1x4x1x16xf32>
      %1086 = tensor.concat dim(3) %1082, %1079 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x1x16xf32>, tensor<1x4x1x16xf32>) -> tensor<1x4x1x32xf32>
      %1087 = tensor.empty() : tensor<1x4x1x32xf32>
      %1088 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1042, %1074 : tensor<1x4x1x32xf32>, tensor<1x1x1x32xf32>) outs(%1087 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb112(%1089: f32, %1090: f32, %1091: f32):
        %1092 = arith.mulf %1089, %1090 : f32
        linalg.yield %1092 : f32
      } -> tensor<1x4x1x32xf32>
      %1093 = tensor.empty() : tensor<1x4x1x32xf32>
      %1094 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1086, %1078 : tensor<1x4x1x32xf32>, tensor<1x1x1x32xf32>) outs(%1093 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb113(%1095: f32, %1096: f32, %1097: f32):
        %1098 = arith.mulf %1095, %1096 : f32
        linalg.yield %1098 : f32
      } -> tensor<1x4x1x32xf32>
      %1099 = tensor.empty() : tensor<1x4x1x32xf32>
      %1100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1088, %1094 : tensor<1x4x1x32xf32>, tensor<1x4x1x32xf32>) outs(%1099 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb114(%1101: f32, %1102: f32, %1103: f32):
        %1104 = arith.addf %1101, %1102 : f32
        linalg.yield %1104 : f32
      } -> tensor<1x4x1x32xf32>
      %1105 = tensor.collapse_shape %980 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
      %1106 = tensor.expand_shape %1105 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x32xf32>
      %1107 = tensor.collapse_shape %1106 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x32xf32> into tensor<32xf32>
      %1108 = tensor.expand_shape %1107 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x1x32xf32>
      %1109 = tensor.collapse_shape %981 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
      %1110 = tensor.expand_shape %1109 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x32xf32>
      %1111 = tensor.collapse_shape %1110 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x32xf32> into tensor<32xf32>
      %1112 = tensor.expand_shape %1111 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x1x32xf32>
      %1113 = "tensor.extract_slice"(%1056) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x32xf32>) -> tensor<1x4x1x16xf32>
      %1114 = "tensor.extract_slice"(%1056) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 1, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x32xf32>) -> tensor<1x4x1x16xf32>
      %1115 = tensor.empty() : tensor<1x4x1x16xf32>
      %1116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1114 : tensor<1x4x1x16xf32>) outs(%1115 : tensor<1x4x1x16xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb115(%1117: f32, %1118: f32):
        %1119 = arith.negf %1117 : f32
        linalg.yield %1119 : f32
      } -> tensor<1x4x1x16xf32>
      %1120 = tensor.concat dim(3) %1116, %1113 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x1x16xf32>, tensor<1x4x1x16xf32>) -> tensor<1x4x1x32xf32>
      %1121 = tensor.empty() : tensor<1x4x1x32xf32>
      %1122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1056, %1108 : tensor<1x4x1x32xf32>, tensor<1x1x1x32xf32>) outs(%1121 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb116(%1123: f32, %1124: f32, %1125: f32):
        %1126 = arith.mulf %1123, %1124 : f32
        linalg.yield %1126 : f32
      } -> tensor<1x4x1x32xf32>
      %1127 = tensor.empty() : tensor<1x4x1x32xf32>
      %1128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1120, %1112 : tensor<1x4x1x32xf32>, tensor<1x1x1x32xf32>) outs(%1127 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb117(%1129: f32, %1130: f32, %1131: f32):
        %1132 = arith.mulf %1129, %1130 : f32
        linalg.yield %1132 : f32
      } -> tensor<1x4x1x32xf32>
      %1133 = tensor.empty() : tensor<1x4x1x32xf32>
      %1134 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1122, %1128 : tensor<1x4x1x32xf32>, tensor<1x4x1x32xf32>) outs(%1133 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb118(%1135: f32, %1136: f32, %1137: f32):
        %1138 = arith.addf %1135, %1136 : f32
        linalg.yield %1138 : f32
      } -> tensor<1x4x1x32xf32>
      %1139 = "tensor.extract_slice"(%960) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 15, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x15x32xf32>) -> tensor<4x15x32xf32>
      %1140 = func.call @aten_index_copy_default_1_wl2(%1139, %972, %1134) {prov.region_id = "aten_index_copy_default_1_0", prov.dispatch_id = "aten_index_copy_default_1_0"} : (tensor<4x15x32xf32>, tensor<1xi64>, tensor<1x4x1x32xf32>) -> tensor<1x4x15x32xf32>
      %1141 = "tensor.extract_slice"(%961) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 15, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x15x32xf32>) -> tensor<4x15x32xf32>
      %1142 = func.call @aten_index_copy_default_1_wl2(%1141, %972, %1070) {prov.region_id = "aten_index_copy_default_1_1", prov.dispatch_id = "aten_index_copy_default_1_1"} : (tensor<4x15x32xf32>, tensor<1xi64>, tensor<1x4x1x32xf32>) -> tensor<1x4x15x32xf32>
      %1143 = tensor.empty() : tensor<1x4x32x15xf32>
      %1144 = linalg.transpose ins(%1140:tensor<1x4x15x32xf32>) outs(%1143:tensor<1x4x32x15xf32>) permutation = [0, 1, 3, 2]
      %1145 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1146 = tensor.splat %1145 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x4x1x15xf32>
      %1147 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1100, %1144 : tensor<1x4x1x32xf32>, tensor<1x4x32x15xf32>) outs(%1146 : tensor<1x4x1x15xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb119(%1148: f32, %1149: f32, %1150: f32):
        %1151 = arith.mulf %1148, %1149 : f32
        %1152 = arith.addf %1150, %1151 : f32
        linalg.yield %1152 : f32
      } -> tensor<1x4x1x15xf32>
      %1153 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 5.65685415 : f32
      %1154 = tensor.splat %1153 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x4x1x15xf32>
      %1155 = tensor.empty() : tensor<1x4x1x15xf32>
      %1156 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1147, %1154 : tensor<1x4x1x15xf32>, tensor<1x4x1x15xf32>) outs(%1155 : tensor<1x4x1x15xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
      ^bb120(%1157: f32, %1158: f32, %1159: f32):
        %1160 = arith.divf %1157, %1158 : f32
        linalg.yield %1160 : f32
      } -> tensor<1x4x1x15xf32>
      %1161 = tensor.expand_shape %972 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1xi64> into tensor<1x1xi64>
      %1162 = tensor.expand_shape %123 [[0 : i64, 1 : i64]] output_shape [1, 15] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<15xi64> into tensor<1x15xi64>
      %1163 = tensor.empty() : tensor<1x15xi1>
      %1164 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1162, %1161 : tensor<1x15xi64>, tensor<1x1xi64>) outs(%1163 : tensor<1x15xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
      ^bb121(%1165: i64, %1166: i64, %1167: i1):
        %1168 = arith.cmpi sle, %1165, %1166 : i64
        linalg.yield %1168 : i1
      } -> tensor<1x15xi1>
      %1169 = tensor.collapse_shape %1164 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x15xi1> into tensor<15xi1>
      %1170 = tensor.expand_shape %1169 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 15] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<15xi1> into tensor<1x1x15xi1>
      %1171 = tensor.collapse_shape %1170 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x1x15xi1> into tensor<15xi1>
      %1172 = tensor.expand_shape %1171 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 15] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<15xi1> into tensor<1x1x1x15xi1>
      %1173 = tensor.empty() : tensor<1x1x1x15xi1>
      %1174 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1172 : tensor<1x1x1x15xi1>) outs(%1173 : tensor<1x1x1x15xi1>) attrs =  {prov.region_id = "bitwise_0", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
      ^bb122(%1175: i1, %1176: i1):
        %1177 = arith.constant true
        %1178 = arith.xori %1175, %1177 : i1
        linalg.yield %1178 : i1
      } -> tensor<1x1x1x15xi1>
      %1179 = func.call @aten_masked_fill_Scalar_wl3(%1156, %1174) {prov.region_id = "aten_masked_fill_Scalar_0", prov.dispatch_id = "aten_masked_fill_Scalar_0"} : (tensor<1x4x1x15xf32>, tensor<1x1x1x15xi1>) -> tensor<1x4x1x15xf32>
      %1180 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0xff800000 : f32
      %1181 = tensor.splat %1180 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x1xf32>
      %1182 = linalg.reduce ins(%1179:tensor<1x4x1x15xf32>) outs(%1181:tensor<1x4x1xf32>) dimensions = [3]
      (%1183: f32, %1184: f32) {
        %1185 = arith.maximumf %1183, %1184 : f32
        linalg.yield %1185 : f32
      }
      %1186 = tensor.collapse_shape %1182 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x1xf32> into tensor<4xf32>
      %1187 = tensor.expand_shape %1186 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 1, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<4xf32> into tensor<1x4x1x1xf32>
      %1188 = tensor.empty() : tensor<1x4x1x15xf32>
      %1189 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1179, %1187 : tensor<1x4x1x15xf32>, tensor<1x4x1x1xf32>) outs(%1188 : tensor<1x4x1x15xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb123(%1190: f32, %1191: f32, %1192: f32):
        %1193 = arith.subf %1190, %1191 : f32
        linalg.yield %1193 : f32
      } -> tensor<1x4x1x15xf32>
      %1194 = tensor.empty() : tensor<1x4x1x15xf32>
      %1195 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1189 : tensor<1x4x1x15xf32>) outs(%1194 : tensor<1x4x1x15xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb124(%1196: f32, %1197: f32):
        %1198 = math.exp %1196 : f32
        linalg.yield %1198 : f32
      } -> tensor<1x4x1x15xf32>
      %1199 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1200 = tensor.splat %1199 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x1xf32>
      %1201 = linalg.reduce ins(%1195:tensor<1x4x1x15xf32>) outs(%1200:tensor<1x4x1xf32>) dimensions = [3]
      (%1202: f32, %1203: f32) {
        %1204 = arith.addf %1202, %1203 : f32
        linalg.yield %1204 : f32
      }
      %1205 = tensor.collapse_shape %1201 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x1xf32> into tensor<4xf32>
      %1206 = tensor.expand_shape %1205 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 1, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<4xf32> into tensor<1x4x1x1xf32>
      %1207 = tensor.empty() : tensor<1x4x1x15xf32>
      %1208 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1195, %1206 : tensor<1x4x1x15xf32>, tensor<1x4x1x1xf32>) outs(%1207 : tensor<1x4x1x15xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb125(%1209: f32, %1210: f32, %1211: f32):
        %1212 = arith.divf %1209, %1210 : f32
        linalg.yield %1212 : f32
      } -> tensor<1x4x1x15xf32>
      %1213 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1214 = tensor.splat %1213 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x4x1x32xf32>
      %1215 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1208, %1142 : tensor<1x4x1x15xf32>, tensor<1x4x15x32xf32>) outs(%1214 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb126(%1216: f32, %1217: f32, %1218: f32):
        %1219 = arith.mulf %1216, %1217 : f32
        %1220 = arith.addf %1218, %1219 : f32
        linalg.yield %1220 : f32
      } -> tensor<1x4x1x32xf32>
      %1221 = tensor.empty() : tensor<1x1x4x32xf32>
      %1222 = linalg.transpose ins(%1215:tensor<1x4x1x32xf32>) outs(%1221:tensor<1x1x4x32xf32>) permutation = [0, 2, 1, 3]
      %1223 = tensor.collapse_shape %1222 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x1x4x32xf32> into tensor<128xf32>
      %1224 = tensor.expand_shape %1223 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x128xf32>
      %1225 = tensor.empty() : tensor<128x128xf32>
      %1226 = linalg.transpose ins(%25:tensor<128x128xf32>) outs(%1225:tensor<128x128xf32>) permutation = [1, 0]
      %1227 = arith.constant {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1228 = tensor.splat %1227 {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32>
      %1229 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1224, %1226 : tensor<1x1x128xf32>, tensor<128x128xf32>) outs(%1228 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb127(%1230: f32, %1231: f32, %1232: f32):
        %1233 = arith.mulf %1230, %1231 : f32
        %1234 = arith.addf %1232, %1233 : f32
        linalg.yield %1234 : f32
      } -> tensor<1x1x128xf32>
      %1235 = tensor.empty() : tensor<1x1x128xf32>
      %1236 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%974, %1229 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%1235 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb128(%1237: f32, %1238: f32, %1239: f32):
        %1240 = arith.addf %1237, %1238 : f32
        linalg.yield %1240 : f32
      } -> tensor<1x1x128xf32>
      %1241 = tensor.empty() : tensor<1x1x128xf32>
      %1242 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1236 : tensor<1x1x128xf32>) outs(%1241 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb129(%1243: f32, %1244: f32):
        %1245 = arith.constant 2.000000e+00 : f32
        %1246 = math.powf %1243, %1245 : f32
        linalg.yield %1246 : f32
      } -> tensor<1x1x128xf32>
      %1247 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1248 = tensor.splat %1247 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1249 = linalg.reduce ins(%1242:tensor<1x1x128xf32>) outs(%1248:tensor<1x1xf32>) dimensions = [2]
      (%1250: f32, %1251: f32) {
        %1252 = arith.addf %1250, %1251 : f32
        linalg.yield %1252 : f32
      }
      %1253 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %1254 = tensor.splat %1253 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1255 = tensor.empty() : tensor<1x1xf32>
      %1256 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1249, %1254 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%1255 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb130(%1257: f32, %1258: f32, %1259: f32):
        %1260 = arith.divf %1257, %1258 : f32
        linalg.yield %1260 : f32
      } -> tensor<1x1xf32>
      %1261 = tensor.collapse_shape %1256 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %1262 = tensor.expand_shape %1261 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %1263 = arith.constant {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %1264 = tensor.splat %1263 {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %1265 = tensor.empty() : tensor<1x1x1xf32>
      %1266 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1262, %1264 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%1265 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb131(%1267: f32, %1268: f32, %1269: f32):
        %1270 = arith.addf %1267, %1268 : f32
        linalg.yield %1270 : f32
      } -> tensor<1x1x1xf32>
      %1271 = tensor.empty() : tensor<1x1x1xf32>
      %1272 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1266 : tensor<1x1x1xf32>) outs(%1271 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb132(%1273: f32, %1274: f32):
        %1275 = math.rsqrt %1273 : f32
        linalg.yield %1275 : f32
      } -> tensor<1x1x1xf32>
      %1276 = tensor.empty() : tensor<1x1x128xf32>
      %1277 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1236, %1272 : tensor<1x1x128xf32>, tensor<1x1x1xf32>) outs(%1276 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb133(%1278: f32, %1279: f32, %1280: f32):
        %1281 = arith.mulf %1278, %1279 : f32
        linalg.yield %1281 : f32
      } -> tensor<1x1x128xf32>
      %1282 = tensor.empty() : tensor<1x1x128xf32>
      %1283 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1277, %26 : tensor<1x1x128xf32>, tensor<128xf32>) outs(%1282 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb134(%1284: f32, %1285: f32, %1286: f32):
        %1287 = arith.mulf %1284, %1285 : f32
        linalg.yield %1287 : f32
      } -> tensor<1x1x128xf32>
      %1288 = tensor.empty() : tensor<128x344xf32>
      %1289 = linalg.transpose ins(%27:tensor<344x128xf32>) outs(%1288:tensor<128x344xf32>) permutation = [1, 0]
      %1290 = arith.constant {prov.region_id = "matmul_6", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1291 = tensor.splat %1290 {prov.region_id = "matmul_6", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x1x344xf32>
      %1292 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1283, %1289 : tensor<1x1x128xf32>, tensor<128x344xf32>) outs(%1291 : tensor<1x1x344xf32>) attrs =  {prov.region_id = "matmul_6", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb135(%1293: f32, %1294: f32, %1295: f32):
        %1296 = arith.mulf %1293, %1294 : f32
        %1297 = arith.addf %1295, %1296 : f32
        linalg.yield %1297 : f32
      } -> tensor<1x1x344xf32>
      %1298 = tensor.empty() : tensor<128x344xf32>
      %1299 = linalg.transpose ins(%28:tensor<344x128xf32>) outs(%1298:tensor<128x344xf32>) permutation = [1, 0]
      %1300 = arith.constant {prov.region_id = "matmul_7", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1301 = tensor.splat %1300 {prov.region_id = "matmul_7", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x1x344xf32>
      %1302 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1283, %1299 : tensor<1x1x128xf32>, tensor<128x344xf32>) outs(%1301 : tensor<1x1x344xf32>) attrs =  {prov.region_id = "matmul_7", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb136(%1303: f32, %1304: f32, %1305: f32):
        %1306 = arith.mulf %1303, %1304 : f32
        %1307 = arith.addf %1305, %1306 : f32
        linalg.yield %1307 : f32
      } -> tensor<1x1x344xf32>
      %1308 = tensor.empty() : tensor<1x1x344xf32>
      %1309 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1292 : tensor<1x1x344xf32>) outs(%1308 : tensor<1x1x344xf32>) attrs =  {prov.region_id = "silu_0", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb137(%1310: f32, %1311: f32):
        %1312 = arith.constant 1.000000e+00 : f32
        %1313 = arith.negf %1310 : f32
        %1314 = math.exp %1313 : f32
        %1315 = arith.addf %1312, %1314 : f32
        %1316 = arith.divf %1312, %1315 : f32
        %1317 = arith.mulf %1310, %1316 : f32
        linalg.yield %1317 : f32
      } -> tensor<1x1x344xf32>
      %1318 = tensor.empty() : tensor<1x1x344xf32>
      %1319 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1309, %1302 : tensor<1x1x344xf32>, tensor<1x1x344xf32>) outs(%1318 : tensor<1x1x344xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb138(%1320: f32, %1321: f32, %1322: f32):
        %1323 = arith.mulf %1320, %1321 : f32
        linalg.yield %1323 : f32
      } -> tensor<1x1x344xf32>
      %1324 = tensor.empty() : tensor<344x128xf32>
      %1325 = linalg.transpose ins(%29:tensor<128x344xf32>) outs(%1324:tensor<344x128xf32>) permutation = [1, 0]
      %1326 = arith.constant {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1327 = tensor.splat %1326 {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32>
      %1328 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1319, %1325 : tensor<1x1x344xf32>, tensor<344x128xf32>) outs(%1327 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb139(%1329: f32, %1330: f32, %1331: f32):
        %1332 = arith.mulf %1329, %1330 : f32
        %1333 = arith.addf %1331, %1332 : f32
        linalg.yield %1333 : f32
      } -> tensor<1x1x128xf32>
      %1334 = tensor.empty() : tensor<1x1x128xf32>
      %1335 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1236, %1328 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%1334 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb140(%1336: f32, %1337: f32, %1338: f32):
        %1339 = arith.addf %1336, %1337 : f32
        linalg.yield %1339 : f32
      } -> tensor<1x1x128xf32>
      %1340 = tensor.empty() : tensor<1x1x128xf32>
      %1341 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1335 : tensor<1x1x128xf32>) outs(%1340 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb141(%1342: f32, %1343: f32):
        %1344 = arith.constant 2.000000e+00 : f32
        %1345 = math.powf %1342, %1344 : f32
        linalg.yield %1345 : f32
      } -> tensor<1x1x128xf32>
      %1346 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1347 = tensor.splat %1346 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1348 = linalg.reduce ins(%1341:tensor<1x1x128xf32>) outs(%1347:tensor<1x1xf32>) dimensions = [2]
      (%1349: f32, %1350: f32) {
        %1351 = arith.addf %1349, %1350 : f32
        linalg.yield %1351 : f32
      }
      %1352 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %1353 = tensor.splat %1352 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1354 = tensor.empty() : tensor<1x1xf32>
      %1355 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1348, %1353 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%1354 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb142(%1356: f32, %1357: f32, %1358: f32):
        %1359 = arith.divf %1356, %1357 : f32
        linalg.yield %1359 : f32
      } -> tensor<1x1xf32>
      %1360 = tensor.collapse_shape %1355 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %1361 = tensor.expand_shape %1360 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %1362 = arith.constant {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %1363 = tensor.splat %1362 {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %1364 = tensor.empty() : tensor<1x1x1xf32>
      %1365 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1361, %1363 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%1364 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb143(%1366: f32, %1367: f32, %1368: f32):
        %1369 = arith.addf %1366, %1367 : f32
        linalg.yield %1369 : f32
      } -> tensor<1x1x1xf32>
      %1370 = tensor.empty() : tensor<1x1x1xf32>
      %1371 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1365 : tensor<1x1x1xf32>) outs(%1370 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb144(%1372: f32, %1373: f32):
        %1374 = math.rsqrt %1372 : f32
        linalg.yield %1374 : f32
      } -> tensor<1x1x1xf32>
      %1375 = tensor.empty() : tensor<1x1x128xf32>
      %1376 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1335, %1371 : tensor<1x1x128xf32>, tensor<1x1x1xf32>) outs(%1375 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb145(%1377: f32, %1378: f32, %1379: f32):
        %1380 = arith.mulf %1377, %1378 : f32
        linalg.yield %1380 : f32
      } -> tensor<1x1x128xf32>
      %1381 = tensor.empty() : tensor<1x1x128xf32>
      %1382 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1376, %30 : tensor<1x1x128xf32>, tensor<128xf32>) outs(%1381 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb146(%1383: f32, %1384: f32, %1385: f32):
        %1386 = arith.mulf %1383, %1384 : f32
        linalg.yield %1386 : f32
      } -> tensor<1x1x128xf32>
      %1387 = tensor.empty() : tensor<128x128xf32>
      %1388 = linalg.transpose ins(%31:tensor<128x128xf32>) outs(%1387:tensor<128x128xf32>) permutation = [1, 0]
      %1389 = arith.constant {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1390 = tensor.splat %1389 {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32>
      %1391 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1382, %1388 : tensor<1x1x128xf32>, tensor<128x128xf32>) outs(%1390 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb147(%1392: f32, %1393: f32, %1394: f32):
        %1395 = arith.mulf %1392, %1393 : f32
        %1396 = arith.addf %1394, %1395 : f32
        linalg.yield %1396 : f32
      } -> tensor<1x1x128xf32>
      %1397 = tensor.collapse_shape %1391 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %1398 = tensor.expand_shape %1397 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 32] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x4x32xf32>
      %1399 = tensor.empty() : tensor<1x4x1x32xf32>
      %1400 = linalg.transpose ins(%1398:tensor<1x1x4x32xf32>) outs(%1399:tensor<1x4x1x32xf32>) permutation = [0, 2, 1, 3]
      %1401 = tensor.empty() : tensor<128x128xf32>
      %1402 = linalg.transpose ins(%32:tensor<128x128xf32>) outs(%1401:tensor<128x128xf32>) permutation = [1, 0]
      %1403 = arith.constant {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1404 = tensor.splat %1403 {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32>
      %1405 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1382, %1402 : tensor<1x1x128xf32>, tensor<128x128xf32>) outs(%1404 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "matmul_10", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb148(%1406: f32, %1407: f32, %1408: f32):
        %1409 = arith.mulf %1406, %1407 : f32
        %1410 = arith.addf %1408, %1409 : f32
        linalg.yield %1410 : f32
      } -> tensor<1x1x128xf32>
      %1411 = tensor.collapse_shape %1405 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %1412 = tensor.expand_shape %1411 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 32] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x4x32xf32>
      %1413 = tensor.empty() : tensor<1x4x1x32xf32>
      %1414 = linalg.transpose ins(%1412:tensor<1x1x4x32xf32>) outs(%1413:tensor<1x4x1x32xf32>) permutation = [0, 2, 1, 3]
      %1415 = tensor.empty() : tensor<128x128xf32>
      %1416 = linalg.transpose ins(%33:tensor<128x128xf32>) outs(%1415:tensor<128x128xf32>) permutation = [1, 0]
      %1417 = arith.constant {prov.region_id = "matmul_11", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1418 = tensor.splat %1417 {prov.region_id = "matmul_11", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32>
      %1419 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1382, %1416 : tensor<1x1x128xf32>, tensor<128x128xf32>) outs(%1418 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "matmul_11", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb149(%1420: f32, %1421: f32, %1422: f32):
        %1423 = arith.mulf %1420, %1421 : f32
        %1424 = arith.addf %1422, %1423 : f32
        linalg.yield %1424 : f32
      } -> tensor<1x1x128xf32>
      %1425 = tensor.collapse_shape %1419 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %1426 = tensor.expand_shape %1425 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 32] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x4x32xf32>
      %1427 = tensor.empty() : tensor<1x4x1x32xf32>
      %1428 = linalg.transpose ins(%1426:tensor<1x1x4x32xf32>) outs(%1427:tensor<1x4x1x32xf32>) permutation = [0, 2, 1, 3]
      %1429 = tensor.collapse_shape %980 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
      %1430 = tensor.expand_shape %1429 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x32xf32>
      %1431 = tensor.collapse_shape %1430 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x32xf32> into tensor<32xf32>
      %1432 = tensor.expand_shape %1431 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x1x32xf32>
      %1433 = tensor.collapse_shape %981 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
      %1434 = tensor.expand_shape %1433 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x32xf32>
      %1435 = tensor.collapse_shape %1434 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x32xf32> into tensor<32xf32>
      %1436 = tensor.expand_shape %1435 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x1x32xf32>
      %1437 = "tensor.extract_slice"(%1400) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x32xf32>) -> tensor<1x4x1x16xf32>
      %1438 = "tensor.extract_slice"(%1400) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 1, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x32xf32>) -> tensor<1x4x1x16xf32>
      %1439 = tensor.empty() : tensor<1x4x1x16xf32>
      %1440 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1438 : tensor<1x4x1x16xf32>) outs(%1439 : tensor<1x4x1x16xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb150(%1441: f32, %1442: f32):
        %1443 = arith.negf %1441 : f32
        linalg.yield %1443 : f32
      } -> tensor<1x4x1x16xf32>
      %1444 = tensor.concat dim(3) %1440, %1437 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x1x16xf32>, tensor<1x4x1x16xf32>) -> tensor<1x4x1x32xf32>
      %1445 = tensor.empty() : tensor<1x4x1x32xf32>
      %1446 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1400, %1432 : tensor<1x4x1x32xf32>, tensor<1x1x1x32xf32>) outs(%1445 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb151(%1447: f32, %1448: f32, %1449: f32):
        %1450 = arith.mulf %1447, %1448 : f32
        linalg.yield %1450 : f32
      } -> tensor<1x4x1x32xf32>
      %1451 = tensor.empty() : tensor<1x4x1x32xf32>
      %1452 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1444, %1436 : tensor<1x4x1x32xf32>, tensor<1x1x1x32xf32>) outs(%1451 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb152(%1453: f32, %1454: f32, %1455: f32):
        %1456 = arith.mulf %1453, %1454 : f32
        linalg.yield %1456 : f32
      } -> tensor<1x4x1x32xf32>
      %1457 = tensor.empty() : tensor<1x4x1x32xf32>
      %1458 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1446, %1452 : tensor<1x4x1x32xf32>, tensor<1x4x1x32xf32>) outs(%1457 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb153(%1459: f32, %1460: f32, %1461: f32):
        %1462 = arith.addf %1459, %1460 : f32
        linalg.yield %1462 : f32
      } -> tensor<1x4x1x32xf32>
      %1463 = tensor.collapse_shape %980 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
      %1464 = tensor.expand_shape %1463 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x32xf32>
      %1465 = tensor.collapse_shape %1464 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x32xf32> into tensor<32xf32>
      %1466 = tensor.expand_shape %1465 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x1x32xf32>
      %1467 = tensor.collapse_shape %981 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
      %1468 = tensor.expand_shape %1467 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x32xf32>
      %1469 = tensor.collapse_shape %1468 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x32xf32> into tensor<32xf32>
      %1470 = tensor.expand_shape %1469 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x1x32xf32>
      %1471 = "tensor.extract_slice"(%1414) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x32xf32>) -> tensor<1x4x1x16xf32>
      %1472 = "tensor.extract_slice"(%1414) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 1, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x32xf32>) -> tensor<1x4x1x16xf32>
      %1473 = tensor.empty() : tensor<1x4x1x16xf32>
      %1474 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1472 : tensor<1x4x1x16xf32>) outs(%1473 : tensor<1x4x1x16xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb154(%1475: f32, %1476: f32):
        %1477 = arith.negf %1475 : f32
        linalg.yield %1477 : f32
      } -> tensor<1x4x1x16xf32>
      %1478 = tensor.concat dim(3) %1474, %1471 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x1x16xf32>, tensor<1x4x1x16xf32>) -> tensor<1x4x1x32xf32>
      %1479 = tensor.empty() : tensor<1x4x1x32xf32>
      %1480 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1414, %1466 : tensor<1x4x1x32xf32>, tensor<1x1x1x32xf32>) outs(%1479 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb155(%1481: f32, %1482: f32, %1483: f32):
        %1484 = arith.mulf %1481, %1482 : f32
        linalg.yield %1484 : f32
      } -> tensor<1x4x1x32xf32>
      %1485 = tensor.empty() : tensor<1x4x1x32xf32>
      %1486 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1478, %1470 : tensor<1x4x1x32xf32>, tensor<1x1x1x32xf32>) outs(%1485 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb156(%1487: f32, %1488: f32, %1489: f32):
        %1490 = arith.mulf %1487, %1488 : f32
        linalg.yield %1490 : f32
      } -> tensor<1x4x1x32xf32>
      %1491 = tensor.empty() : tensor<1x4x1x32xf32>
      %1492 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1480, %1486 : tensor<1x4x1x32xf32>, tensor<1x4x1x32xf32>) outs(%1491 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb157(%1493: f32, %1494: f32, %1495: f32):
        %1496 = arith.addf %1493, %1494 : f32
        linalg.yield %1496 : f32
      } -> tensor<1x4x1x32xf32>
      %1497 = "tensor.extract_slice"(%960) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 15, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x15x32xf32>) -> tensor<4x15x32xf32>
      %1498 = func.call @aten_index_copy_default_1_wl2(%1497, %972, %1492) {prov.region_id = "aten_index_copy_default_1_2", prov.dispatch_id = "aten_index_copy_default_1_2"} : (tensor<4x15x32xf32>, tensor<1xi64>, tensor<1x4x1x32xf32>) -> tensor<1x4x15x32xf32>
      %1499 = "tensor.extract_slice"(%961) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 15, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x15x32xf32>) -> tensor<4x15x32xf32>
      %1500 = func.call @aten_index_copy_default_1_wl2(%1499, %972, %1428) {prov.region_id = "aten_index_copy_default_1_3", prov.dispatch_id = "aten_index_copy_default_1_3"} : (tensor<4x15x32xf32>, tensor<1xi64>, tensor<1x4x1x32xf32>) -> tensor<1x4x15x32xf32>
      %1501 = tensor.empty() : tensor<1x4x32x15xf32>
      %1502 = linalg.transpose ins(%1498:tensor<1x4x15x32xf32>) outs(%1501:tensor<1x4x32x15xf32>) permutation = [0, 1, 3, 2]
      %1503 = arith.constant {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1504 = tensor.splat %1503 {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x4x1x15xf32>
      %1505 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1458, %1502 : tensor<1x4x1x32xf32>, tensor<1x4x32x15xf32>) outs(%1504 : tensor<1x4x1x15xf32>) attrs =  {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb158(%1506: f32, %1507: f32, %1508: f32):
        %1509 = arith.mulf %1506, %1507 : f32
        %1510 = arith.addf %1508, %1509 : f32
        linalg.yield %1510 : f32
      } -> tensor<1x4x1x15xf32>
      %1511 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 5.65685415 : f32
      %1512 = tensor.splat %1511 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x4x1x15xf32>
      %1513 = tensor.empty() : tensor<1x4x1x15xf32>
      %1514 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1505, %1512 : tensor<1x4x1x15xf32>, tensor<1x4x1x15xf32>) outs(%1513 : tensor<1x4x1x15xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
      ^bb159(%1515: f32, %1516: f32, %1517: f32):
        %1518 = arith.divf %1515, %1516 : f32
        linalg.yield %1518 : f32
      } -> tensor<1x4x1x15xf32>
      %1519 = tensor.expand_shape %972 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1xi64> into tensor<1x1xi64>
      %1520 = tensor.expand_shape %123 [[0 : i64, 1 : i64]] output_shape [1, 15] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<15xi64> into tensor<1x15xi64>
      %1521 = tensor.empty() : tensor<1x15xi1>
      %1522 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1520, %1519 : tensor<1x15xi64>, tensor<1x1xi64>) outs(%1521 : tensor<1x15xi1>) attrs =  {prov.region_id = "compare_1", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
      ^bb160(%1523: i64, %1524: i64, %1525: i1):
        %1526 = arith.cmpi sle, %1523, %1524 : i64
        linalg.yield %1526 : i1
      } -> tensor<1x15xi1>
      %1527 = tensor.collapse_shape %1522 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x15xi1> into tensor<15xi1>
      %1528 = tensor.expand_shape %1527 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 15] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<15xi1> into tensor<1x1x15xi1>
      %1529 = tensor.collapse_shape %1528 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x1x15xi1> into tensor<15xi1>
      %1530 = tensor.expand_shape %1529 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 15] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<15xi1> into tensor<1x1x1x15xi1>
      %1531 = tensor.empty() : tensor<1x1x1x15xi1>
      %1532 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1530 : tensor<1x1x1x15xi1>) outs(%1531 : tensor<1x1x1x15xi1>) attrs =  {prov.region_id = "bitwise_1", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
      ^bb161(%1533: i1, %1534: i1):
        %1535 = arith.constant true
        %1536 = arith.xori %1533, %1535 : i1
        linalg.yield %1536 : i1
      } -> tensor<1x1x1x15xi1>
      %1537 = func.call @aten_masked_fill_Scalar_wl3(%1514, %1532) {prov.region_id = "aten_masked_fill_Scalar_1", prov.dispatch_id = "aten_masked_fill_Scalar_1"} : (tensor<1x4x1x15xf32>, tensor<1x1x1x15xi1>) -> tensor<1x4x1x15xf32>
      %1538 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0xff800000 : f32
      %1539 = tensor.splat %1538 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x1xf32>
      %1540 = linalg.reduce ins(%1537:tensor<1x4x1x15xf32>) outs(%1539:tensor<1x4x1xf32>) dimensions = [3]
      (%1541: f32, %1542: f32) {
        %1543 = arith.maximumf %1541, %1542 : f32
        linalg.yield %1543 : f32
      }
      %1544 = tensor.collapse_shape %1540 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x1xf32> into tensor<4xf32>
      %1545 = tensor.expand_shape %1544 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 1, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<4xf32> into tensor<1x4x1x1xf32>
      %1546 = tensor.empty() : tensor<1x4x1x15xf32>
      %1547 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1537, %1545 : tensor<1x4x1x15xf32>, tensor<1x4x1x1xf32>) outs(%1546 : tensor<1x4x1x15xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb162(%1548: f32, %1549: f32, %1550: f32):
        %1551 = arith.subf %1548, %1549 : f32
        linalg.yield %1551 : f32
      } -> tensor<1x4x1x15xf32>
      %1552 = tensor.empty() : tensor<1x4x1x15xf32>
      %1553 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1547 : tensor<1x4x1x15xf32>) outs(%1552 : tensor<1x4x1x15xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb163(%1554: f32, %1555: f32):
        %1556 = math.exp %1554 : f32
        linalg.yield %1556 : f32
      } -> tensor<1x4x1x15xf32>
      %1557 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1558 = tensor.splat %1557 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x1xf32>
      %1559 = linalg.reduce ins(%1553:tensor<1x4x1x15xf32>) outs(%1558:tensor<1x4x1xf32>) dimensions = [3]
      (%1560: f32, %1561: f32) {
        %1562 = arith.addf %1560, %1561 : f32
        linalg.yield %1562 : f32
      }
      %1563 = tensor.collapse_shape %1559 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x4x1xf32> into tensor<4xf32>
      %1564 = tensor.expand_shape %1563 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 1, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<4xf32> into tensor<1x4x1x1xf32>
      %1565 = tensor.empty() : tensor<1x4x1x15xf32>
      %1566 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1553, %1564 : tensor<1x4x1x15xf32>, tensor<1x4x1x1xf32>) outs(%1565 : tensor<1x4x1x15xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb164(%1567: f32, %1568: f32, %1569: f32):
        %1570 = arith.divf %1567, %1568 : f32
        linalg.yield %1570 : f32
      } -> tensor<1x4x1x15xf32>
      %1571 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1572 = tensor.splat %1571 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x4x1x32xf32>
      %1573 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1566, %1500 : tensor<1x4x1x15xf32>, tensor<1x4x15x32xf32>) outs(%1572 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb165(%1574: f32, %1575: f32, %1576: f32):
        %1577 = arith.mulf %1574, %1575 : f32
        %1578 = arith.addf %1576, %1577 : f32
        linalg.yield %1578 : f32
      } -> tensor<1x4x1x32xf32>
      %1579 = tensor.empty() : tensor<1x1x4x32xf32>
      %1580 = linalg.transpose ins(%1573:tensor<1x4x1x32xf32>) outs(%1579:tensor<1x1x4x32xf32>) permutation = [0, 2, 1, 3]
      %1581 = tensor.collapse_shape %1580 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x1x4x32xf32> into tensor<128xf32>
      %1582 = tensor.expand_shape %1581 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 128] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x128xf32>
      %1583 = tensor.empty() : tensor<128x128xf32>
      %1584 = linalg.transpose ins(%34:tensor<128x128xf32>) outs(%1583:tensor<128x128xf32>) permutation = [1, 0]
      %1585 = arith.constant {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1586 = tensor.splat %1585 {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32>
      %1587 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1582, %1584 : tensor<1x1x128xf32>, tensor<128x128xf32>) outs(%1586 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb166(%1588: f32, %1589: f32, %1590: f32):
        %1591 = arith.mulf %1588, %1589 : f32
        %1592 = arith.addf %1590, %1591 : f32
        linalg.yield %1592 : f32
      } -> tensor<1x1x128xf32>
      %1593 = tensor.empty() : tensor<1x1x128xf32>
      %1594 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1335, %1587 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%1593 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb167(%1595: f32, %1596: f32, %1597: f32):
        %1598 = arith.addf %1595, %1596 : f32
        linalg.yield %1598 : f32
      } -> tensor<1x1x128xf32>
      %1599 = tensor.empty() : tensor<1x1x128xf32>
      %1600 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1594 : tensor<1x1x128xf32>) outs(%1599 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb168(%1601: f32, %1602: f32):
        %1603 = arith.constant 2.000000e+00 : f32
        %1604 = math.powf %1601, %1603 : f32
        linalg.yield %1604 : f32
      } -> tensor<1x1x128xf32>
      %1605 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1606 = tensor.splat %1605 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1607 = linalg.reduce ins(%1600:tensor<1x1x128xf32>) outs(%1606:tensor<1x1xf32>) dimensions = [2]
      (%1608: f32, %1609: f32) {
        %1610 = arith.addf %1608, %1609 : f32
        linalg.yield %1610 : f32
      }
      %1611 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %1612 = tensor.splat %1611 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1613 = tensor.empty() : tensor<1x1xf32>
      %1614 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1607, %1612 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%1613 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb169(%1615: f32, %1616: f32, %1617: f32):
        %1618 = arith.divf %1615, %1616 : f32
        linalg.yield %1618 : f32
      } -> tensor<1x1xf32>
      %1619 = tensor.collapse_shape %1614 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %1620 = tensor.expand_shape %1619 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %1621 = arith.constant {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %1622 = tensor.splat %1621 {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %1623 = tensor.empty() : tensor<1x1x1xf32>
      %1624 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1620, %1622 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%1623 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb170(%1625: f32, %1626: f32, %1627: f32):
        %1628 = arith.addf %1625, %1626 : f32
        linalg.yield %1628 : f32
      } -> tensor<1x1x1xf32>
      %1629 = tensor.empty() : tensor<1x1x1xf32>
      %1630 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1624 : tensor<1x1x1xf32>) outs(%1629 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb171(%1631: f32, %1632: f32):
        %1633 = math.rsqrt %1631 : f32
        linalg.yield %1633 : f32
      } -> tensor<1x1x1xf32>
      %1634 = tensor.empty() : tensor<1x1x128xf32>
      %1635 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1594, %1630 : tensor<1x1x128xf32>, tensor<1x1x1xf32>) outs(%1634 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb172(%1636: f32, %1637: f32, %1638: f32):
        %1639 = arith.mulf %1636, %1637 : f32
        linalg.yield %1639 : f32
      } -> tensor<1x1x128xf32>
      %1640 = tensor.empty() : tensor<1x1x128xf32>
      %1641 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1635, %35 : tensor<1x1x128xf32>, tensor<128xf32>) outs(%1640 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb173(%1642: f32, %1643: f32, %1644: f32):
        %1645 = arith.mulf %1642, %1643 : f32
        linalg.yield %1645 : f32
      } -> tensor<1x1x128xf32>
      %1646 = tensor.empty() : tensor<128x344xf32>
      %1647 = linalg.transpose ins(%36:tensor<344x128xf32>) outs(%1646:tensor<128x344xf32>) permutation = [1, 0]
      %1648 = arith.constant {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1649 = tensor.splat %1648 {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x1x344xf32>
      %1650 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1641, %1647 : tensor<1x1x128xf32>, tensor<128x344xf32>) outs(%1649 : tensor<1x1x344xf32>) attrs =  {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb174(%1651: f32, %1652: f32, %1653: f32):
        %1654 = arith.mulf %1651, %1652 : f32
        %1655 = arith.addf %1653, %1654 : f32
        linalg.yield %1655 : f32
      } -> tensor<1x1x344xf32>
      %1656 = tensor.empty() : tensor<128x344xf32>
      %1657 = linalg.transpose ins(%37:tensor<344x128xf32>) outs(%1656:tensor<128x344xf32>) permutation = [1, 0]
      %1658 = arith.constant {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1659 = tensor.splat %1658 {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x1x344xf32>
      %1660 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1641, %1657 : tensor<1x1x128xf32>, tensor<128x344xf32>) outs(%1659 : tensor<1x1x344xf32>) attrs =  {prov.region_id = "matmul_16", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb175(%1661: f32, %1662: f32, %1663: f32):
        %1664 = arith.mulf %1661, %1662 : f32
        %1665 = arith.addf %1663, %1664 : f32
        linalg.yield %1665 : f32
      } -> tensor<1x1x344xf32>
      %1666 = tensor.empty() : tensor<1x1x344xf32>
      %1667 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1650 : tensor<1x1x344xf32>) outs(%1666 : tensor<1x1x344xf32>) attrs =  {prov.region_id = "silu_1", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb176(%1668: f32, %1669: f32):
        %1670 = arith.constant 1.000000e+00 : f32
        %1671 = arith.negf %1668 : f32
        %1672 = math.exp %1671 : f32
        %1673 = arith.addf %1670, %1672 : f32
        %1674 = arith.divf %1670, %1673 : f32
        %1675 = arith.mulf %1668, %1674 : f32
        linalg.yield %1675 : f32
      } -> tensor<1x1x344xf32>
      %1676 = tensor.empty() : tensor<1x1x344xf32>
      %1677 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1667, %1660 : tensor<1x1x344xf32>, tensor<1x1x344xf32>) outs(%1676 : tensor<1x1x344xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb177(%1678: f32, %1679: f32, %1680: f32):
        %1681 = arith.mulf %1678, %1679 : f32
        linalg.yield %1681 : f32
      } -> tensor<1x1x344xf32>
      %1682 = tensor.empty() : tensor<344x128xf32>
      %1683 = linalg.transpose ins(%38:tensor<128x344xf32>) outs(%1682:tensor<344x128xf32>) permutation = [1, 0]
      %1684 = arith.constant {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1685 = tensor.splat %1684 {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32>
      %1686 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1677, %1683 : tensor<1x1x344xf32>, tensor<344x128xf32>) outs(%1685 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "matmul_17", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb178(%1687: f32, %1688: f32, %1689: f32):
        %1690 = arith.mulf %1687, %1688 : f32
        %1691 = arith.addf %1689, %1690 : f32
        linalg.yield %1691 : f32
      } -> tensor<1x1x128xf32>
      %1692 = tensor.empty() : tensor<1x1x128xf32>
      %1693 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1594, %1686 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%1692 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb179(%1694: f32, %1695: f32, %1696: f32):
        %1697 = arith.addf %1694, %1695 : f32
        linalg.yield %1697 : f32
      } -> tensor<1x1x128xf32>
      %1698 = func.call @aten_stack_default_wl4(%1140, %1498) {prov.region_id = "aten_stack_default_0", prov.dispatch_id = "aten_stack_default_0"} : (tensor<1x4x15x32xf32>, tensor<1x4x15x32xf32>) -> tensor<2x1x4x15x32xf32>
      %1699 = func.call @aten_stack_default_wl4(%1142, %1500) {prov.region_id = "aten_stack_default_1", prov.dispatch_id = "aten_stack_default_1"} : (tensor<1x4x15x32xf32>, tensor<1x4x15x32xf32>) -> tensor<2x1x4x15x32xf32>
      %1700 = tensor.empty() : tensor<1x1x128xf32>
      %1701 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1693 : tensor<1x1x128xf32>) outs(%1700 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb180(%1702: f32, %1703: f32):
        %1704 = arith.constant 2.000000e+00 : f32
        %1705 = math.powf %1702, %1704 : f32
        linalg.yield %1705 : f32
      } -> tensor<1x1x128xf32>
      %1706 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1707 = tensor.splat %1706 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1708 = linalg.reduce ins(%1701:tensor<1x1x128xf32>) outs(%1707:tensor<1x1xf32>) dimensions = [2]
      (%1709: f32, %1710: f32) {
        %1711 = arith.addf %1709, %1710 : f32
        linalg.yield %1711 : f32
      }
      %1712 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %1713 = tensor.splat %1712 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1714 = tensor.empty() : tensor<1x1xf32>
      %1715 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1708, %1713 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%1714 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb181(%1716: f32, %1717: f32, %1718: f32):
        %1719 = arith.divf %1716, %1717 : f32
        linalg.yield %1719 : f32
      } -> tensor<1x1xf32>
      %1720 = tensor.collapse_shape %1715 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %1721 = tensor.expand_shape %1720 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %1722 = arith.constant {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %1723 = tensor.splat %1722 {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %1724 = tensor.empty() : tensor<1x1x1xf32>
      %1725 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1721, %1723 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%1724 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb182(%1726: f32, %1727: f32, %1728: f32):
        %1729 = arith.addf %1726, %1727 : f32
        linalg.yield %1729 : f32
      } -> tensor<1x1x1xf32>
      %1730 = tensor.empty() : tensor<1x1x1xf32>
      %1731 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1725 : tensor<1x1x1xf32>) outs(%1730 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb183(%1732: f32, %1733: f32):
        %1734 = math.rsqrt %1732 : f32
        linalg.yield %1734 : f32
      } -> tensor<1x1x1xf32>
      %1735 = tensor.empty() : tensor<1x1x128xf32>
      %1736 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1693, %1731 : tensor<1x1x128xf32>, tensor<1x1x1xf32>) outs(%1735 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb184(%1737: f32, %1738: f32, %1739: f32):
        %1740 = arith.mulf %1737, %1738 : f32
        linalg.yield %1740 : f32
      } -> tensor<1x1x128xf32>
      %1741 = tensor.empty() : tensor<1x1x128xf32>
      %1742 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1736, %39 : tensor<1x1x128xf32>, tensor<128xf32>) outs(%1741 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb185(%1743: f32, %1744: f32, %1745: f32):
        %1746 = arith.mulf %1743, %1744 : f32
        linalg.yield %1746 : f32
      } -> tensor<1x1x128xf32>
      %1747 = tensor.empty() : tensor<128x256xf32>
      %1748 = linalg.transpose ins(%41:tensor<256x128xf32>) outs(%1747:tensor<128x256xf32>) permutation = [1, 0]
      %1749 = arith.constant {prov.region_id = "matmul_18", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1750 = tensor.splat %1749 {prov.region_id = "matmul_18", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x1x256xf32>
      %1751 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1742, %1748 : tensor<1x1x128xf32>, tensor<128x256xf32>) outs(%1750 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "matmul_18", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb186(%1752: f32, %1753: f32, %1754: f32):
        %1755 = arith.mulf %1752, %1753 : f32
        %1756 = arith.addf %1754, %1755 : f32
        linalg.yield %1756 : f32
      } -> tensor<1x1x256xf32>
      %1757 = "tensor.extract_slice"(%1751) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 256>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<1x1x256xf32>) -> tensor<256xf32>
      %1758 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0xff800000 : f32
      %1759 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0 : i64
      %1760 = tensor.splat %1758 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<f32>
      %1761 = tensor.splat %1759 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<i64>
      %1762, %1763 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%1757 : tensor<256xf32>) outs(%1760, %1761 : tensor<f32>, tensor<i64>) attrs =  {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} {
      ^bb187(%1764: f32, %1765: f32, %1766: i64):
        %1767 = linalg.index 0 : index
        %1768 = arith.index_cast %1767 : index to i64
        %1769 = arith.cmpf ogt, %1764, %1765 : f32
        %1770 = arith.select %1769, %1764, %1765 : f32
        %1771 = arith.select %1769, %1768, %1766 : i64
        linalg.yield %1770, %1771 : f32, i64
      } -> (tensor<f32>, tensor<i64>)
      %1772 = tensor.extract %1762[] : tensor<f32>
      %1773 = tensor.from_elements %1772 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xf32>
      %1774 = tensor.extract %1763[] : tensor<i64>
      %1775 = tensor.from_elements %1774 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xi64>
      %1776 = arith.constant {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} 1 : i64
      %1777 = tensor.splat %1776 {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} : tensor<i64>
      %1778 = tensor.empty() : tensor<i64>
      %1779 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%957, %1777 : tensor<i64>, tensor<i64>) outs(%1778 : tensor<i64>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb188(%1780: i64, %1781: i64, %1782: i64):
        %1783 = arith.addi %1780, %1781 : i64
        linalg.yield %1783 : i64
      } -> tensor<i64>
      scf.yield %1779, %1775, %964, %1698, %1699 : tensor<i64>, tensor<1xi64>, tensor<1x7xi64>, tensor<2x1x4x15x32xf32>, tensor<2x1x4x15x32xf32>
    }
    func.return %953 : tensor<1x7xi64>
  }
}
