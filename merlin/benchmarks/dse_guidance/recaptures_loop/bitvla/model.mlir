builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func private @aten_zeros_default() -> tensor<2x1x4x39x32xf32>
  func.func private @aten_zeros_default_1() -> tensor<i64>
  func.func private @aten_clamp__default(tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  func.func private @aten_clamp__default_1(tensor<f32>) -> tensor<f32>
  func.func private @aten_index_copy_default(tensor<4x39x32xf32>, tensor<32xi64>, tensor<1x4x32x32xf32>) -> tensor<1x4x39x32xf32>
  func.func private @aten_masked_fill_Scalar(tensor<1x8x32x39xf32>, tensor<1x1x32x39xi1>) -> tensor<1x8x32x39xf32>
  func.func private @aten_stack_default(tensor<1x4x39x32xf32>, tensor<1x4x39x32xf32>) -> tensor<2x1x4x39x32xf32>
  func.func private @aten_zeros_default_2() -> tensor<1x7xi64>
  func.func private @aten_index_copy_default_wl0(tensor<1x7xi64>, tensor<1xi64>, tensor<1x1xi64>) -> tensor<1x7xi64>
  func.func private @wrap_with_set_grad_enabled_wl1(tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
  func.func private @wrap_with_set_grad_enabled_1_wl2(tensor<256x256xf32>) -> tensor<256x256xf32>
  func.func private @wrap_with_set_grad_enabled_2_wl3() -> tensor<1x1x256xf32>
  func.func private @wrap_with_set_grad_enabled_3_wl4(tensor<128x256xf32>) -> tensor<128x256xf32>
  func.func private @aten_index_copy_default_1_wl5(tensor<4x39x32xf32>, tensor<1xi64>, tensor<1x4x1x32xf32>) -> tensor<1x4x39x32xf32>
  func.func private @aten_masked_fill_Scalar_wl6(tensor<1x8x1x39xf32>, tensor<1x1x1x39xi1>) -> tensor<1x8x1x39xf32>
  func.func private @wrap_with_set_grad_enabled_4_wl7(tensor<512x256xf32>) -> tensor<512x256xf32>
  func.func private @wrap_with_set_grad_enabled_5_wl8(tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
  func.func private @wrap_with_set_grad_enabled_6_wl9(tensor<256x512xf32>) -> tensor<256x512xf32>
  func.func private @aten_stack_default_wl10(tensor<1x4x39x32xf32>, tensor<1x4x39x32xf32>) -> tensor<2x1x4x39x32xf32>
  func.func @forward(%0: tensor<256x256xf32>, %1: tensor<128x256xf32>, %2: tensor<128x256xf32>, %3: tensor<256x256xf32>, %4: tensor<256xf32>, %5: tensor<512x256xf32>, %6: tensor<512x256xf32>, %7: tensor<256x512xf32>, %8: tensor<512xf32>, %9: tensor<256xf32>, %10: tensor<256xf32>, %11: tensor<256x256xf32>, %12: tensor<128x256xf32>, %13: tensor<128x256xf32>, %14: tensor<256x256xf32>, %15: tensor<256xf32>, %16: tensor<512x256xf32>, %17: tensor<512x256xf32>, %18: tensor<256x512xf32>, %19: tensor<512xf32>, %20: tensor<256xf32>, %21: tensor<256xf32>, %22: tensor<256xf32>, %23: tensor<1024x256xf32>, %24: tensor<1024x256xf32>, %25: tensor<16xf32>, %26: tensor<2048x32xf32>, %27: tensor<2048x32xf32>, %28: tensor<16xf32>, %29: tensor<2048x32xf32>, %30: tensor<2048x32xf32>, %31: tensor<i64>, %32: tensor<1x32x256xf32>) -> tensor<1x7xi64> {
    %33 = func.call @aten_zeros_default() {prov.region_id = "aten_zeros_default_0", prov.dispatch_id = "aten_zeros_default_0"} : () -> tensor<2x1x4x39x32xf32>
    %34 = func.call @aten_zeros_default() {prov.region_id = "aten_zeros_default_1", prov.dispatch_id = "aten_zeros_default_1"} : () -> tensor<2x1x4x39x32xf32>
    %35 = tensor.empty() : tensor<39xi64>
    %36 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%35 : tensor<39xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
    ^bb0(%37: i64):
      %38 = linalg.index 0 : index
      %39 = arith.index_cast %38 : index to i64
      %40 = arith.constant 1 : i64
      %41 = arith.muli %39, %40 : i64
      %42 = arith.constant 0 : i64
      %43 = arith.addi %42, %41 : i64
      linalg.yield %43 : i64
    } -> tensor<39xi64>
    %44 = tensor.empty() : tensor<32xi64>
    %45 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%44 : tensor<32xi64>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
    ^bb1(%46: i64):
      %47 = linalg.index 0 : index
      %48 = arith.index_cast %47 : index to i64
      %49 = arith.constant 1 : i64
      %50 = arith.muli %48, %49 : i64
      %51 = arith.constant 0 : i64
      %52 = arith.addi %51, %50 : i64
      linalg.yield %52 : i64
    } -> tensor<32xi64>
    %53 = tensor.empty() : tensor<1xi64>
    %54 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%53 : tensor<1xi64>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64"} {
    ^bb2(%55: i64):
      %56 = linalg.index 0 : index
      %57 = arith.index_cast %56 : index to i64
      %58 = arith.constant 1 : i64
      %59 = arith.muli %57, %58 : i64
      %60 = arith.constant 0 : i64
      %61 = arith.addi %60, %59 : i64
      linalg.yield %61 : i64
    } -> tensor<1xi64>
    %62 = func.call @aten_zeros_default_1() {prov.region_id = "aten_zeros_default_1_0", prov.dispatch_id = "aten_zeros_default_1_0"} : () -> tensor<i64>
    %63 = tensor.expand_shape %45 [[0 : i64, 1 : i64]] output_shape [1, 32] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<32xi64> into tensor<1x32xi64>
    %64 = tensor.empty() : tensor<1x32x256xf32>
    %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32 : tensor<1x32x256xf32>) outs(%64 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.input_layernorm"} {
    ^bb3(%66: f32, %67: f32):
      %68 = arith.constant 2.000000e+00 : f32
      %69 = math.powf %66, %68 : f32
      linalg.yield %69 : f32
    } -> tensor<1x32x256xf32>
    %70 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.input_layernorm"} 0.000000e+00 : f32
    %71 = tensor.splat %70 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.input_layernorm"} : tensor<1x32xf32>
    %72 = linalg.reduce ins(%65:tensor<1x32x256xf32>) outs(%71:tensor<1x32xf32>) dimensions = [2]
    (%73: f32, %74: f32) {
      %75 = arith.addf %73, %74 : f32
      linalg.yield %75 : f32
    }
    %76 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.input_layernorm"} 2.560000e+02 : f32
    %77 = tensor.splat %76 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.input_layernorm"} : tensor<1x32xf32>
    %78 = tensor.empty() : tensor<1x32xf32>
    %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %77 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%78 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.input_layernorm"} {
    ^bb4(%80: f32, %81: f32, %82: f32):
      %83 = arith.divf %80, %81 : f32
      linalg.yield %83 : f32
    } -> tensor<1x32xf32>
    %84 = tensor.collapse_shape %79 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.input_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %85 = tensor.expand_shape %84 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.input_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %86 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.input_layernorm"} 1.000000e-05 : f32
    %87 = tensor.splat %86 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.input_layernorm"} : tensor<1x32x1xf32>
    %88 = tensor.empty() : tensor<1x32x1xf32>
    %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85, %87 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%88 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.input_layernorm"} {
    ^bb5(%90: f32, %91: f32, %92: f32):
      %93 = arith.addf %90, %91 : f32
      linalg.yield %93 : f32
    } -> tensor<1x32x1xf32>
    %94 = tensor.empty() : tensor<1x32x1xf32>
    %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89 : tensor<1x32x1xf32>) outs(%94 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.input_layernorm"} {
    ^bb6(%96: f32, %97: f32):
      %98 = math.rsqrt %96 : f32
      linalg.yield %98 : f32
    } -> tensor<1x32x1xf32>
    %99 = tensor.empty() : tensor<1x32x256xf32>
    %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %95 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%99 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.input_layernorm"} {
    ^bb7(%101: f32, %102: f32, %103: f32):
      %104 = arith.mulf %101, %102 : f32
      linalg.yield %104 : f32
    } -> tensor<1x32x256xf32>
    %105 = tensor.empty() : tensor<1x32x256xf32>
    %106 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %100 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%105 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.input_layernorm"} {
    ^bb8(%107: f32, %108: f32, %109: f32):
      %110 = arith.mulf %107, %108 : f32
      linalg.yield %110 : f32
    } -> tensor<1x32x256xf32>
    %111 = tensor.empty() : tensor<1x32x256xf32>
    %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%106 : tensor<1x32x256xf32>) outs(%111 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_0", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} {
    ^bb9(%113: f32, %114: f32):
      %115 = math.absf %113 : f32
      linalg.yield %115 : f32
    } -> tensor<1x32x256xf32>
    %116 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} 0xff800000 : f32
    %117 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} 0 : i64
    %118 = tensor.splat %116 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} : tensor<1x32xf32>
    %119 = tensor.splat %117 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} : tensor<1x32xi64>
    %120, %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%112 : tensor<1x32x256xf32>) outs(%118, %119 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} {
    ^bb10(%122: f32, %123: f32, %124: i64):
      %125 = linalg.index 2 : index
      %126 = arith.index_cast %125 : index to i64
      %127 = arith.cmpf ogt, %122, %123 : f32
      %128 = arith.select %127, %122, %123 : f32
      %129 = arith.select %127, %126, %124 : i64
      linalg.yield %128, %129 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %130 = tensor.collapse_shape %120 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %131 = tensor.expand_shape %130 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %132 = tensor.collapse_shape %121 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %133 = tensor.expand_shape %132 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %134 = func.call @aten_clamp__default(%131) {prov.region_id = "aten_clamp__default_0", prov.dispatch_id = "aten_clamp__default_0"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %135 = tensor.empty() : tensor<1x32x1xf32>
    %136 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%134 : tensor<1x32x1xf32>) outs(%135 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_0", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} {
    ^bb11(%137: f32, %138: f32):
      %139 = arith.constant 1.000000e+00 : f32
      %140 = arith.divf %139, %137 : f32
      linalg.yield %140 : f32
    } -> tensor<1x32x1xf32>
    %141 = arith.constant {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} 1.270000e+02 : f32
    %142 = tensor.splat %141 {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} : tensor<1x32x1xf32>
    %143 = tensor.empty() : tensor<1x32x1xf32>
    %144 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%136, %142 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%143 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} {
    ^bb12(%145: f32, %146: f32, %147: f32):
      %148 = arith.mulf %145, %146 : f32
      linalg.yield %148 : f32
    } -> tensor<1x32x1xf32>
    %149 = tensor.empty() : tensor<1x32x256xf32>
    %150 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%106, %144 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%149 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} {
    ^bb13(%151: f32, %152: f32, %153: f32):
      %154 = arith.mulf %151, %152 : f32
      linalg.yield %154 : f32
    } -> tensor<1x32x256xf32>
    %155 = tensor.empty() : tensor<1x32x256xf32>
    %156 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%150 : tensor<1x32x256xf32>) outs(%155 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_0", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} {
    ^bb14(%157: f32, %158: f32):
      %159 = math.roundeven %157 : f32
      linalg.yield %159 : f32
    } -> tensor<1x32x256xf32>
    %160 = tensor.empty() : tensor<1x32x256xf32>
    %161 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%156 : tensor<1x32x256xf32>) outs(%160 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_0", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} {
    ^bb15(%162: f32, %163: f32):
      %164 = arith.constant -1.280000e+02 : f32
      %165 = arith.maximumf %162, %164 : f32
      %166 = arith.constant 1.270000e+02 : f32
      %167 = arith.minimumf %165, %166 : f32
      linalg.yield %167 : f32
    } -> tensor<1x32x256xf32>
    %168 = tensor.empty() : tensor<1x32x256xf32>
    %169 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%161, %144 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%168 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} {
    ^bb16(%170: f32, %171: f32, %172: f32):
      %173 = arith.divf %170, %171 : f32
      linalg.yield %173 : f32
    } -> tensor<1x32x256xf32>
    %174 = tensor.empty() : tensor<256x256xf32>
    %175 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<256x256xf32>) outs(%174 : tensor<256x256xf32>) attrs =  {prov.region_id = "abs_1", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} {
    ^bb17(%176: f32, %177: f32):
      %178 = math.absf %176 : f32
      linalg.yield %178 : f32
    } -> tensor<256x256xf32>
    %179 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} 0.000000e+00 : f32
    %180 = tensor.splat %179 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} : tensor<f32>
    %181 = linalg.reduce ins(%175:tensor<256x256xf32>) outs(%180:tensor<f32>) dimensions = [0, 1]
    (%182: f32, %183: f32) {
      %184 = arith.addf %182, %183 : f32
      linalg.yield %184 : f32
    }
    %185 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} 6.553600e+04 : f32
    %186 = tensor.splat %185 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} : tensor<f32>
    %187 = tensor.empty() : tensor<f32>
    %188 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%181, %186 : tensor<f32>, tensor<f32>) outs(%187 : tensor<f32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} {
    ^bb18(%189: f32, %190: f32, %191: f32):
      %192 = arith.divf %189, %190 : f32
      linalg.yield %192 : f32
    } -> tensor<f32>
    %193 = func.call @aten_clamp__default_1(%188) {prov.region_id = "aten_clamp__default_1_0", prov.dispatch_id = "aten_clamp__default_1_0"} : (tensor<f32>) -> tensor<f32>
    %194 = tensor.empty() : tensor<f32>
    %195 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%193 : tensor<f32>) outs(%194 : tensor<f32>) attrs =  {prov.region_id = "elementwise_1", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} {
    ^bb19(%196: f32, %197: f32):
      %198 = arith.constant 1.000000e+00 : f32
      %199 = arith.divf %198, %196 : f32
      linalg.yield %199 : f32
    } -> tensor<f32>
    %200 = arith.constant {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} 1.000000e+00 : f32
    %201 = tensor.splat %200 {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} : tensor<f32>
    %202 = tensor.empty() : tensor<f32>
    %203 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%195, %201 : tensor<f32>, tensor<f32>) outs(%202 : tensor<f32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} {
    ^bb20(%204: f32, %205: f32, %206: f32):
      %207 = arith.mulf %204, %205 : f32
      linalg.yield %207 : f32
    } -> tensor<f32>
    %208 = tensor.empty() : tensor<256x256xf32>
    %209 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %203 : tensor<256x256xf32>, tensor<f32>) outs(%208 : tensor<256x256xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} {
    ^bb21(%210: f32, %211: f32, %212: f32):
      %213 = arith.mulf %210, %211 : f32
      linalg.yield %213 : f32
    } -> tensor<256x256xf32>
    %214 = tensor.empty() : tensor<256x256xf32>
    %215 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%209 : tensor<256x256xf32>) outs(%214 : tensor<256x256xf32>) attrs =  {prov.region_id = "round_1", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} {
    ^bb22(%216: f32, %217: f32):
      %218 = math.roundeven %216 : f32
      linalg.yield %218 : f32
    } -> tensor<256x256xf32>
    %219 = tensor.empty() : tensor<256x256xf32>
    %220 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%215 : tensor<256x256xf32>) outs(%219 : tensor<256x256xf32>) attrs =  {prov.region_id = "minmax_1", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} {
    ^bb23(%221: f32, %222: f32):
      %223 = arith.constant -1.000000e+00 : f32
      %224 = arith.maximumf %221, %223 : f32
      %225 = arith.constant 1.000000e+00 : f32
      %226 = arith.minimumf %224, %225 : f32
      linalg.yield %226 : f32
    } -> tensor<256x256xf32>
    %227 = tensor.empty() : tensor<256x256xf32>
    %228 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%220, %203 : tensor<256x256xf32>, tensor<f32>) outs(%227 : tensor<256x256xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} {
    ^bb24(%229: f32, %230: f32, %231: f32):
      %232 = arith.divf %229, %230 : f32
      linalg.yield %232 : f32
    } -> tensor<256x256xf32>
    %233 = tensor.empty() : tensor<256x256xf32>
    %234 = linalg.transpose ins(%228:tensor<256x256xf32>) outs(%233:tensor<256x256xf32>) permutation = [1, 0]
    %235 = tensor.empty() : tensor<1x32x256xf32>
    %236 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %237 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%236 : f32) outs(%235 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %238 = linalg.matmul {prov.region_id = "matmul_0", prov.dispatch_id = "matmul_0", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.q_proj"} ins(%169, %234 : tensor<1x32x256xf32>, tensor<256x256xf32>) outs(%237 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %239 = tensor.collapse_shape %238 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %240 = tensor.expand_shape %239 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 32] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x32x8x32xf32>
    %241 = tensor.empty() : tensor<1x8x32x32xf32>
    %242 = linalg.transpose ins(%240:tensor<1x32x8x32xf32>) outs(%241:tensor<1x8x32x32xf32>) permutation = [0, 2, 1, 3]
    %243 = tensor.empty() : tensor<1x32x256xf32>
    %244 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%106 : tensor<1x32x256xf32>) outs(%243 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_2", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} {
    ^bb25(%245: f32, %246: f32):
      %247 = math.absf %245 : f32
      linalg.yield %247 : f32
    } -> tensor<1x32x256xf32>
    %248 = arith.constant {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} 0xff800000 : f32
    %249 = arith.constant {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} 0 : i64
    %250 = tensor.splat %248 {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} : tensor<1x32xf32>
    %251 = tensor.splat %249 {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} : tensor<1x32xi64>
    %252, %253 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%244 : tensor<1x32x256xf32>) outs(%250, %251 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} {
    ^bb26(%254: f32, %255: f32, %256: i64):
      %257 = linalg.index 2 : index
      %258 = arith.index_cast %257 : index to i64
      %259 = arith.cmpf ogt, %254, %255 : f32
      %260 = arith.select %259, %254, %255 : f32
      %261 = arith.select %259, %258, %256 : i64
      linalg.yield %260, %261 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %262 = tensor.collapse_shape %252 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %263 = tensor.expand_shape %262 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %264 = tensor.collapse_shape %253 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %265 = tensor.expand_shape %264 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %266 = func.call @aten_clamp__default(%263) {prov.region_id = "aten_clamp__default_1", prov.dispatch_id = "aten_clamp__default_1"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %267 = tensor.empty() : tensor<1x32x1xf32>
    %268 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%266 : tensor<1x32x1xf32>) outs(%267 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_2", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} {
    ^bb27(%269: f32, %270: f32):
      %271 = arith.constant 1.000000e+00 : f32
      %272 = arith.divf %271, %269 : f32
      linalg.yield %272 : f32
    } -> tensor<1x32x1xf32>
    %273 = arith.constant {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} 1.270000e+02 : f32
    %274 = tensor.splat %273 {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} : tensor<1x32x1xf32>
    %275 = tensor.empty() : tensor<1x32x1xf32>
    %276 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%268, %274 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%275 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} {
    ^bb28(%277: f32, %278: f32, %279: f32):
      %280 = arith.mulf %277, %278 : f32
      linalg.yield %280 : f32
    } -> tensor<1x32x1xf32>
    %281 = tensor.empty() : tensor<1x32x256xf32>
    %282 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%106, %276 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%281 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} {
    ^bb29(%283: f32, %284: f32, %285: f32):
      %286 = arith.mulf %283, %284 : f32
      linalg.yield %286 : f32
    } -> tensor<1x32x256xf32>
    %287 = tensor.empty() : tensor<1x32x256xf32>
    %288 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%282 : tensor<1x32x256xf32>) outs(%287 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_2", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} {
    ^bb30(%289: f32, %290: f32):
      %291 = math.roundeven %289 : f32
      linalg.yield %291 : f32
    } -> tensor<1x32x256xf32>
    %292 = tensor.empty() : tensor<1x32x256xf32>
    %293 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%288 : tensor<1x32x256xf32>) outs(%292 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_2", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} {
    ^bb31(%294: f32, %295: f32):
      %296 = arith.constant -1.280000e+02 : f32
      %297 = arith.maximumf %294, %296 : f32
      %298 = arith.constant 1.270000e+02 : f32
      %299 = arith.minimumf %297, %298 : f32
      linalg.yield %299 : f32
    } -> tensor<1x32x256xf32>
    %300 = tensor.empty() : tensor<1x32x256xf32>
    %301 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%293, %276 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%300 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} {
    ^bb32(%302: f32, %303: f32, %304: f32):
      %305 = arith.divf %302, %303 : f32
      linalg.yield %305 : f32
    } -> tensor<1x32x256xf32>
    %306 = tensor.empty() : tensor<128x256xf32>
    %307 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<128x256xf32>) outs(%306 : tensor<128x256xf32>) attrs =  {prov.region_id = "abs_3", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} {
    ^bb33(%308: f32, %309: f32):
      %310 = math.absf %308 : f32
      linalg.yield %310 : f32
    } -> tensor<128x256xf32>
    %311 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} 0.000000e+00 : f32
    %312 = tensor.splat %311 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} : tensor<f32>
    %313 = linalg.reduce ins(%307:tensor<128x256xf32>) outs(%312:tensor<f32>) dimensions = [0, 1]
    (%314: f32, %315: f32) {
      %316 = arith.addf %314, %315 : f32
      linalg.yield %316 : f32
    }
    %317 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} 3.276800e+04 : f32
    %318 = tensor.splat %317 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} : tensor<f32>
    %319 = tensor.empty() : tensor<f32>
    %320 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%313, %318 : tensor<f32>, tensor<f32>) outs(%319 : tensor<f32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} {
    ^bb34(%321: f32, %322: f32, %323: f32):
      %324 = arith.divf %321, %322 : f32
      linalg.yield %324 : f32
    } -> tensor<f32>
    %325 = func.call @aten_clamp__default_1(%320) {prov.region_id = "aten_clamp__default_1_1", prov.dispatch_id = "aten_clamp__default_1_1"} : (tensor<f32>) -> tensor<f32>
    %326 = tensor.empty() : tensor<f32>
    %327 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%325 : tensor<f32>) outs(%326 : tensor<f32>) attrs =  {prov.region_id = "elementwise_3", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} {
    ^bb35(%328: f32, %329: f32):
      %330 = arith.constant 1.000000e+00 : f32
      %331 = arith.divf %330, %328 : f32
      linalg.yield %331 : f32
    } -> tensor<f32>
    %332 = arith.constant {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} 1.000000e+00 : f32
    %333 = tensor.splat %332 {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} : tensor<f32>
    %334 = tensor.empty() : tensor<f32>
    %335 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%327, %333 : tensor<f32>, tensor<f32>) outs(%334 : tensor<f32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} {
    ^bb36(%336: f32, %337: f32, %338: f32):
      %339 = arith.mulf %336, %337 : f32
      linalg.yield %339 : f32
    } -> tensor<f32>
    %340 = tensor.empty() : tensor<128x256xf32>
    %341 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1, %335 : tensor<128x256xf32>, tensor<f32>) outs(%340 : tensor<128x256xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} {
    ^bb37(%342: f32, %343: f32, %344: f32):
      %345 = arith.mulf %342, %343 : f32
      linalg.yield %345 : f32
    } -> tensor<128x256xf32>
    %346 = tensor.empty() : tensor<128x256xf32>
    %347 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%341 : tensor<128x256xf32>) outs(%346 : tensor<128x256xf32>) attrs =  {prov.region_id = "round_3", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} {
    ^bb38(%348: f32, %349: f32):
      %350 = math.roundeven %348 : f32
      linalg.yield %350 : f32
    } -> tensor<128x256xf32>
    %351 = tensor.empty() : tensor<128x256xf32>
    %352 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%347 : tensor<128x256xf32>) outs(%351 : tensor<128x256xf32>) attrs =  {prov.region_id = "minmax_3", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} {
    ^bb39(%353: f32, %354: f32):
      %355 = arith.constant -1.000000e+00 : f32
      %356 = arith.maximumf %353, %355 : f32
      %357 = arith.constant 1.000000e+00 : f32
      %358 = arith.minimumf %356, %357 : f32
      linalg.yield %358 : f32
    } -> tensor<128x256xf32>
    %359 = tensor.empty() : tensor<128x256xf32>
    %360 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%352, %335 : tensor<128x256xf32>, tensor<f32>) outs(%359 : tensor<128x256xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} {
    ^bb40(%361: f32, %362: f32, %363: f32):
      %364 = arith.divf %361, %362 : f32
      linalg.yield %364 : f32
    } -> tensor<128x256xf32>
    %365 = tensor.empty() : tensor<256x128xf32>
    %366 = linalg.transpose ins(%360:tensor<128x256xf32>) outs(%365:tensor<256x128xf32>) permutation = [1, 0]
    %367 = tensor.empty() : tensor<1x32x128xf32>
    %368 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %369 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%368 : f32) outs(%367 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %370 = linalg.matmul {prov.region_id = "matmul_1", prov.dispatch_id = "matmul_1", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.k_proj"} ins(%301, %366 : tensor<1x32x256xf32>, tensor<256x128xf32>) outs(%369 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %371 = tensor.collapse_shape %370 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %372 = tensor.expand_shape %371 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 32] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x32x4x32xf32>
    %373 = tensor.empty() : tensor<1x4x32x32xf32>
    %374 = linalg.transpose ins(%372:tensor<1x32x4x32xf32>) outs(%373:tensor<1x4x32x32xf32>) permutation = [0, 2, 1, 3]
    %375 = tensor.empty() : tensor<1x32x256xf32>
    %376 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%106 : tensor<1x32x256xf32>) outs(%375 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_4", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} {
    ^bb41(%377: f32, %378: f32):
      %379 = math.absf %377 : f32
      linalg.yield %379 : f32
    } -> tensor<1x32x256xf32>
    %380 = arith.constant {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} 0xff800000 : f32
    %381 = arith.constant {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} 0 : i64
    %382 = tensor.splat %380 {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} : tensor<1x32xf32>
    %383 = tensor.splat %381 {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} : tensor<1x32xi64>
    %384, %385 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%376 : tensor<1x32x256xf32>) outs(%382, %383 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} {
    ^bb42(%386: f32, %387: f32, %388: i64):
      %389 = linalg.index 2 : index
      %390 = arith.index_cast %389 : index to i64
      %391 = arith.cmpf ogt, %386, %387 : f32
      %392 = arith.select %391, %386, %387 : f32
      %393 = arith.select %391, %390, %388 : i64
      linalg.yield %392, %393 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %394 = tensor.collapse_shape %384 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %395 = tensor.expand_shape %394 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %396 = tensor.collapse_shape %385 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %397 = tensor.expand_shape %396 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %398 = func.call @aten_clamp__default(%395) {prov.region_id = "aten_clamp__default_2", prov.dispatch_id = "aten_clamp__default_2"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %399 = tensor.empty() : tensor<1x32x1xf32>
    %400 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%398 : tensor<1x32x1xf32>) outs(%399 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_4", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} {
    ^bb43(%401: f32, %402: f32):
      %403 = arith.constant 1.000000e+00 : f32
      %404 = arith.divf %403, %401 : f32
      linalg.yield %404 : f32
    } -> tensor<1x32x1xf32>
    %405 = arith.constant {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} 1.270000e+02 : f32
    %406 = tensor.splat %405 {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} : tensor<1x32x1xf32>
    %407 = tensor.empty() : tensor<1x32x1xf32>
    %408 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%400, %406 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%407 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} {
    ^bb44(%409: f32, %410: f32, %411: f32):
      %412 = arith.mulf %409, %410 : f32
      linalg.yield %412 : f32
    } -> tensor<1x32x1xf32>
    %413 = tensor.empty() : tensor<1x32x256xf32>
    %414 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%106, %408 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%413 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} {
    ^bb45(%415: f32, %416: f32, %417: f32):
      %418 = arith.mulf %415, %416 : f32
      linalg.yield %418 : f32
    } -> tensor<1x32x256xf32>
    %419 = tensor.empty() : tensor<1x32x256xf32>
    %420 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%414 : tensor<1x32x256xf32>) outs(%419 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_4", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} {
    ^bb46(%421: f32, %422: f32):
      %423 = math.roundeven %421 : f32
      linalg.yield %423 : f32
    } -> tensor<1x32x256xf32>
    %424 = tensor.empty() : tensor<1x32x256xf32>
    %425 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%420 : tensor<1x32x256xf32>) outs(%424 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_4", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} {
    ^bb47(%426: f32, %427: f32):
      %428 = arith.constant -1.280000e+02 : f32
      %429 = arith.maximumf %426, %428 : f32
      %430 = arith.constant 1.270000e+02 : f32
      %431 = arith.minimumf %429, %430 : f32
      linalg.yield %431 : f32
    } -> tensor<1x32x256xf32>
    %432 = tensor.empty() : tensor<1x32x256xf32>
    %433 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%425, %408 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%432 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} {
    ^bb48(%434: f32, %435: f32, %436: f32):
      %437 = arith.divf %434, %435 : f32
      linalg.yield %437 : f32
    } -> tensor<1x32x256xf32>
    %438 = tensor.empty() : tensor<128x256xf32>
    %439 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<128x256xf32>) outs(%438 : tensor<128x256xf32>) attrs =  {prov.region_id = "abs_5", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} {
    ^bb49(%440: f32, %441: f32):
      %442 = math.absf %440 : f32
      linalg.yield %442 : f32
    } -> tensor<128x256xf32>
    %443 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} 0.000000e+00 : f32
    %444 = tensor.splat %443 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} : tensor<f32>
    %445 = linalg.reduce ins(%439:tensor<128x256xf32>) outs(%444:tensor<f32>) dimensions = [0, 1]
    (%446: f32, %447: f32) {
      %448 = arith.addf %446, %447 : f32
      linalg.yield %448 : f32
    }
    %449 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} 3.276800e+04 : f32
    %450 = tensor.splat %449 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} : tensor<f32>
    %451 = tensor.empty() : tensor<f32>
    %452 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%445, %450 : tensor<f32>, tensor<f32>) outs(%451 : tensor<f32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} {
    ^bb50(%453: f32, %454: f32, %455: f32):
      %456 = arith.divf %453, %454 : f32
      linalg.yield %456 : f32
    } -> tensor<f32>
    %457 = func.call @aten_clamp__default_1(%452) {prov.region_id = "aten_clamp__default_1_2", prov.dispatch_id = "aten_clamp__default_1_2"} : (tensor<f32>) -> tensor<f32>
    %458 = tensor.empty() : tensor<f32>
    %459 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%457 : tensor<f32>) outs(%458 : tensor<f32>) attrs =  {prov.region_id = "elementwise_5", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} {
    ^bb51(%460: f32, %461: f32):
      %462 = arith.constant 1.000000e+00 : f32
      %463 = arith.divf %462, %460 : f32
      linalg.yield %463 : f32
    } -> tensor<f32>
    %464 = arith.constant {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} 1.000000e+00 : f32
    %465 = tensor.splat %464 {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} : tensor<f32>
    %466 = tensor.empty() : tensor<f32>
    %467 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%459, %465 : tensor<f32>, tensor<f32>) outs(%466 : tensor<f32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} {
    ^bb52(%468: f32, %469: f32, %470: f32):
      %471 = arith.mulf %468, %469 : f32
      linalg.yield %471 : f32
    } -> tensor<f32>
    %472 = tensor.empty() : tensor<128x256xf32>
    %473 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %467 : tensor<128x256xf32>, tensor<f32>) outs(%472 : tensor<128x256xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} {
    ^bb53(%474: f32, %475: f32, %476: f32):
      %477 = arith.mulf %474, %475 : f32
      linalg.yield %477 : f32
    } -> tensor<128x256xf32>
    %478 = tensor.empty() : tensor<128x256xf32>
    %479 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%473 : tensor<128x256xf32>) outs(%478 : tensor<128x256xf32>) attrs =  {prov.region_id = "round_5", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} {
    ^bb54(%480: f32, %481: f32):
      %482 = math.roundeven %480 : f32
      linalg.yield %482 : f32
    } -> tensor<128x256xf32>
    %483 = tensor.empty() : tensor<128x256xf32>
    %484 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%479 : tensor<128x256xf32>) outs(%483 : tensor<128x256xf32>) attrs =  {prov.region_id = "minmax_5", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} {
    ^bb55(%485: f32, %486: f32):
      %487 = arith.constant -1.000000e+00 : f32
      %488 = arith.maximumf %485, %487 : f32
      %489 = arith.constant 1.000000e+00 : f32
      %490 = arith.minimumf %488, %489 : f32
      linalg.yield %490 : f32
    } -> tensor<128x256xf32>
    %491 = tensor.empty() : tensor<128x256xf32>
    %492 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%484, %467 : tensor<128x256xf32>, tensor<f32>) outs(%491 : tensor<128x256xf32>) attrs =  {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} {
    ^bb56(%493: f32, %494: f32, %495: f32):
      %496 = arith.divf %493, %494 : f32
      linalg.yield %496 : f32
    } -> tensor<128x256xf32>
    %497 = tensor.empty() : tensor<256x128xf32>
    %498 = linalg.transpose ins(%492:tensor<128x256xf32>) outs(%497:tensor<256x128xf32>) permutation = [1, 0]
    %499 = tensor.empty() : tensor<1x32x128xf32>
    %500 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %501 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%500 : f32) outs(%499 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %502 = linalg.matmul {prov.region_id = "matmul_2", prov.dispatch_id = "matmul_2", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.v_proj"} ins(%433, %498 : tensor<1x32x256xf32>, tensor<256x128xf32>) outs(%501 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %503 = tensor.collapse_shape %502 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %504 = tensor.expand_shape %503 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 32] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x32x4x32xf32>
    %505 = tensor.empty() : tensor<1x4x32x32xf32>
    %506 = linalg.transpose ins(%504:tensor<1x32x4x32xf32>) outs(%505:tensor<1x4x32x32xf32>) permutation = [0, 2, 1, 3]
    %507 = "tensor.extract_slice"(%26) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 39, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.rotary_emb"} : (tensor<2048x32xf32>) -> tensor<39x32xf32>
    %508 = "tensor.extract_slice"(%27) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 39, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.rotary_emb"} : (tensor<2048x32xf32>) -> tensor<39x32xf32>
    %509 = tensor.empty() : tensor<1x32x32xf32>
    %510 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63 : tensor<1x32xi64>) outs(%509 : tensor<1x32x32xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32"} {
    ^bb57(%511: i64, %512: f32):
      %513 = arith.index_cast %511 : i64 to index
      %514 = linalg.index 2 : index
      %515 = tensor.extract %507[%513, %514] : tensor<39x32xf32>
      linalg.yield %515 : f32
    } -> tensor<1x32x32xf32>
    %516 = tensor.collapse_shape %510 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %517 = tensor.expand_shape %516 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %518 = tensor.empty() : tensor<1x32x32xf32>
    %519 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63 : tensor<1x32xi64>) outs(%518 : tensor<1x32x32xf32>) attrs =  {prov.region_id = "gather_1", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32"} {
    ^bb58(%520: i64, %521: f32):
      %522 = arith.index_cast %520 : i64 to index
      %523 = linalg.index 2 : index
      %524 = tensor.extract %508[%522, %523] : tensor<39x32xf32>
      linalg.yield %524 : f32
    } -> tensor<1x32x32xf32>
    %525 = tensor.collapse_shape %519 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %526 = tensor.expand_shape %525 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %527 = tensor.empty() : tensor<1x8x32x32xf32>
    %528 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%242, %517 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%527 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb59(%529: f32, %530: f32, %531: f32):
      %532 = arith.mulf %529, %530 : f32
      linalg.yield %532 : f32
    } -> tensor<1x8x32x32xf32>
    %533 = "tensor.extract_slice"(%242) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x32x32xf32>) -> tensor<1x8x32x16xf32>
    %534 = "tensor.extract_slice"(%242) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 8, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x32x32xf32>) -> tensor<1x8x32x16xf32>
    %535 = tensor.empty() : tensor<1x8x32x16xf32>
    %536 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%534 : tensor<1x8x32x16xf32>) outs(%535 : tensor<1x8x32x16xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb60(%537: f32, %538: f32):
      %539 = arith.negf %537 : f32
      linalg.yield %539 : f32
    } -> tensor<1x8x32x16xf32>
    %540 = tensor.concat dim(3) %536, %533 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x8x32x16xf32>, tensor<1x8x32x16xf32>) -> tensor<1x8x32x32xf32>
    %541 = tensor.empty() : tensor<1x8x32x32xf32>
    %542 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%540, %526 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%541 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb61(%543: f32, %544: f32, %545: f32):
      %546 = arith.mulf %543, %544 : f32
      linalg.yield %546 : f32
    } -> tensor<1x8x32x32xf32>
    %547 = tensor.empty() : tensor<1x8x32x32xf32>
    %548 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%528, %542 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%547 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb62(%549: f32, %550: f32, %551: f32):
      %552 = arith.addf %549, %550 : f32
      linalg.yield %552 : f32
    } -> tensor<1x8x32x32xf32>
    %553 = tensor.empty() : tensor<1x4x32x32xf32>
    %554 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%374, %517 : tensor<1x4x32x32xf32>, tensor<1x1x32x32xf32>) outs(%553 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb63(%555: f32, %556: f32, %557: f32):
      %558 = arith.mulf %555, %556 : f32
      linalg.yield %558 : f32
    } -> tensor<1x4x32x32xf32>
    %559 = "tensor.extract_slice"(%374) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x16xf32>
    %560 = "tensor.extract_slice"(%374) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x16xf32>
    %561 = tensor.empty() : tensor<1x4x32x16xf32>
    %562 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%560 : tensor<1x4x32x16xf32>) outs(%561 : tensor<1x4x32x16xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb64(%563: f32, %564: f32):
      %565 = arith.negf %563 : f32
      linalg.yield %565 : f32
    } -> tensor<1x4x32x16xf32>
    %566 = tensor.concat dim(3) %562, %559 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x32x16xf32>, tensor<1x4x32x16xf32>) -> tensor<1x4x32x32xf32>
    %567 = tensor.empty() : tensor<1x4x32x32xf32>
    %568 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%566, %526 : tensor<1x4x32x32xf32>, tensor<1x1x32x32xf32>) outs(%567 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb65(%569: f32, %570: f32, %571: f32):
      %572 = arith.mulf %569, %570 : f32
      linalg.yield %572 : f32
    } -> tensor<1x4x32x32xf32>
    %573 = tensor.empty() : tensor<1x4x32x32xf32>
    %574 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%554, %568 : tensor<1x4x32x32xf32>, tensor<1x4x32x32xf32>) outs(%573 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb66(%575: f32, %576: f32, %577: f32):
      %578 = arith.addf %575, %576 : f32
      linalg.yield %578 : f32
    } -> tensor<1x4x32x32xf32>
    %579 = "tensor.extract_slice"(%33) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x39x32xf32>) -> tensor<4x39x32xf32>
    %580 = tensor.empty() : tensor<32xi64>
    %581 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%62, %45 : tensor<i64>, tensor<32xi64>) outs(%580 : tensor<32xi64>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb67(%582: i64, %583: i64, %584: i64):
      %585 = arith.addi %582, %583 : i64
      linalg.yield %585 : i64
    } -> tensor<32xi64>
    %586 = func.call @aten_index_copy_default(%579, %581, %574) {prov.region_id = "aten_index_copy_default_0", prov.dispatch_id = "aten_index_copy_default_0"} : (tensor<4x39x32xf32>, tensor<32xi64>, tensor<1x4x32x32xf32>) -> tensor<1x4x39x32xf32>
    %587 = "tensor.extract_slice"(%34) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x39x32xf32>) -> tensor<4x39x32xf32>
    %588 = tensor.empty() : tensor<32xi64>
    %589 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%62, %45 : tensor<i64>, tensor<32xi64>) outs(%588 : tensor<32xi64>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb68(%590: i64, %591: i64, %592: i64):
      %593 = arith.addi %590, %591 : i64
      linalg.yield %593 : i64
    } -> tensor<32xi64>
    %594 = func.call @aten_index_copy_default(%587, %589, %506) {prov.region_id = "aten_index_copy_default_1", prov.dispatch_id = "aten_index_copy_default_1"} : (tensor<4x39x32xf32>, tensor<32xi64>, tensor<1x4x32x32xf32>) -> tensor<1x4x39x32xf32>
    %595 = "tensor.extract_slice"(%586) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x39x32xf32>) -> tensor<1x4x39x32xf32>
    %596 = "tensor.extract_slice"(%595) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x39x32xf32>) -> tensor<1x4x39x32xf32>
    %597 = tensor.collapse_shape %596 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x39x32xf32> into tensor<4992xf32>
    %598 = tensor.expand_shape %597 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 39, 32] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<4992xf32> into tensor<1x4x1x39x32xf32>
    %599 = "tensor.extract_slice"(%598) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x39x32xf32>) -> tensor<1x4x1x39x32xf32>
    %600 = "tensor.extract_slice"(%599) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_9", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x39x32xf32>) -> tensor<1x4x1x39x32xf32>
    %601 = tensor.empty() : tensor<1x4x2x39x32xf32>
    %602 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%600 : tensor<1x4x1x39x32xf32>) outs(%601 : tensor<1x4x2x39x32xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb69(%603: f32, %604: f32):
      linalg.yield %603 : f32
    } -> tensor<1x4x2x39x32xf32>
    %605 = tensor.collapse_shape %602 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x4x2x39x32xf32> into tensor<9984xf32>
    %606 = tensor.expand_shape %605 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 39, 32] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<9984xf32> into tensor<1x8x39x32xf32>
    %607 = "tensor.extract_slice"(%594) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_10", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x39x32xf32>) -> tensor<1x4x39x32xf32>
    %608 = "tensor.extract_slice"(%607) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_11", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x39x32xf32>) -> tensor<1x4x39x32xf32>
    %609 = tensor.collapse_shape %608 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x39x32xf32> into tensor<4992xf32>
    %610 = tensor.expand_shape %609 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 39, 32] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<4992xf32> into tensor<1x4x1x39x32xf32>
    %611 = "tensor.extract_slice"(%610) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_12", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x39x32xf32>) -> tensor<1x4x1x39x32xf32>
    %612 = "tensor.extract_slice"(%611) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_13", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x39x32xf32>) -> tensor<1x4x1x39x32xf32>
    %613 = tensor.empty() : tensor<1x4x2x39x32xf32>
    %614 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%612 : tensor<1x4x1x39x32xf32>) outs(%613 : tensor<1x4x2x39x32xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb70(%615: f32, %616: f32):
      linalg.yield %615 : f32
    } -> tensor<1x4x2x39x32xf32>
    %617 = tensor.collapse_shape %614 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x4x2x39x32xf32> into tensor<9984xf32>
    %618 = tensor.expand_shape %617 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 39, 32] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<9984xf32> into tensor<1x8x39x32xf32>
    %619 = tensor.empty() : tensor<1x8x32x39xf32>
    %620 = linalg.transpose ins(%606:tensor<1x8x39x32xf32>) outs(%619:tensor<1x8x32x39xf32>) permutation = [0, 1, 3, 2]
    %621 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %622 = tensor.splat %621 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x32x39xf32>
    %623 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%548, %620 : tensor<1x8x32x32xf32>, tensor<1x8x32x39xf32>) outs(%622 : tensor<1x8x32x39xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb71(%624: f32, %625: f32, %626: f32):
      %627 = arith.mulf %624, %625 : f32
      %628 = arith.addf %626, %627 : f32
      linalg.yield %628 : f32
    } -> tensor<1x8x32x39xf32>
    %629 = arith.constant {prov.region_id = "div_6", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 5.65685415 : f32
    %630 = tensor.splat %629 {prov.region_id = "div_6", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x32x39xf32>
    %631 = tensor.empty() : tensor<1x8x32x39xf32>
    %632 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%623, %630 : tensor<1x8x32x39xf32>, tensor<1x8x32x39xf32>) outs(%631 : tensor<1x8x32x39xf32>) attrs =  {prov.region_id = "div_6", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb72(%633: f32, %634: f32, %635: f32):
      %636 = arith.divf %633, %634 : f32
      linalg.yield %636 : f32
    } -> tensor<1x8x32x39xf32>
    %637 = tensor.empty() : tensor<32xi64>
    %638 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%62, %45 : tensor<i64>, tensor<32xi64>) outs(%637 : tensor<32xi64>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb73(%639: i64, %640: i64, %641: i64):
      %642 = arith.addi %639, %640 : i64
      linalg.yield %642 : i64
    } -> tensor<32xi64>
    %643 = tensor.expand_shape %638 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<32xi64> into tensor<32x1xi64>
    %644 = tensor.expand_shape %36 [[0 : i64, 1 : i64]] output_shape [1, 39] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<39xi64> into tensor<1x39xi64>
    %645 = tensor.empty() : tensor<32x39xi1>
    %646 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%644, %643 : tensor<1x39xi64>, tensor<32x1xi64>) outs(%645 : tensor<32x39xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
    ^bb74(%647: i64, %648: i64, %649: i1):
      %650 = arith.cmpi sle, %647, %648 : i64
      linalg.yield %650 : i1
    } -> tensor<32x39xi1>
    %651 = tensor.collapse_shape %646 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<32x39xi1> into tensor<1248xi1>
    %652 = tensor.expand_shape %651 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 39] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1248xi1> into tensor<1x32x39xi1>
    %653 = tensor.collapse_shape %652 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x32x39xi1> into tensor<1248xi1>
    %654 = tensor.expand_shape %653 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 39] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1248xi1> into tensor<1x1x32x39xi1>
    %655 = tensor.empty() : tensor<1x1x32x39xi1>
    %656 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%654 : tensor<1x1x32x39xi1>) outs(%655 : tensor<1x1x32x39xi1>) attrs =  {prov.region_id = "bitwise_0", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
    ^bb75(%657: i1, %658: i1):
      %659 = arith.constant true
      %660 = arith.xori %657, %659 : i1
      linalg.yield %660 : i1
    } -> tensor<1x1x32x39xi1>
    %661 = func.call @aten_masked_fill_Scalar(%632, %656) {prov.region_id = "aten_masked_fill_Scalar_0", prov.dispatch_id = "aten_masked_fill_Scalar_0"} : (tensor<1x8x32x39xf32>, tensor<1x1x32x39xi1>) -> tensor<1x8x32x39xf32>
    %662 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0xff800000 : f32
    %663 = tensor.splat %662 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x32xf32>
    %664 = linalg.reduce ins(%661:tensor<1x8x32x39xf32>) outs(%663:tensor<1x8x32xf32>) dimensions = [3]
    (%665: f32, %666: f32) {
      %667 = arith.maximumf %665, %666 : f32
      linalg.yield %667 : f32
    }
    %668 = tensor.collapse_shape %664 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x32xf32> into tensor<256xf32>
    %669 = tensor.expand_shape %668 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %670 = tensor.empty() : tensor<1x8x32x39xf32>
    %671 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%661, %669 : tensor<1x8x32x39xf32>, tensor<1x8x32x1xf32>) outs(%670 : tensor<1x8x32x39xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb76(%672: f32, %673: f32, %674: f32):
      %675 = arith.subf %672, %673 : f32
      linalg.yield %675 : f32
    } -> tensor<1x8x32x39xf32>
    %676 = tensor.empty() : tensor<1x8x32x39xf32>
    %677 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%671 : tensor<1x8x32x39xf32>) outs(%676 : tensor<1x8x32x39xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb77(%678: f32, %679: f32):
      %680 = math.exp %678 : f32
      linalg.yield %680 : f32
    } -> tensor<1x8x32x39xf32>
    %681 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %682 = tensor.splat %681 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x32xf32>
    %683 = linalg.reduce ins(%677:tensor<1x8x32x39xf32>) outs(%682:tensor<1x8x32xf32>) dimensions = [3]
    (%684: f32, %685: f32) {
      %686 = arith.addf %684, %685 : f32
      linalg.yield %686 : f32
    }
    %687 = tensor.collapse_shape %683 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x32xf32> into tensor<256xf32>
    %688 = tensor.expand_shape %687 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %689 = tensor.empty() : tensor<1x8x32x39xf32>
    %690 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%677, %688 : tensor<1x8x32x39xf32>, tensor<1x8x32x1xf32>) outs(%689 : tensor<1x8x32x39xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb78(%691: f32, %692: f32, %693: f32):
      %694 = arith.divf %691, %692 : f32
      linalg.yield %694 : f32
    } -> tensor<1x8x32x39xf32>
    %695 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %696 = tensor.splat %695 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x32x32xf32>
    %697 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%690, %618 : tensor<1x8x32x39xf32>, tensor<1x8x39x32xf32>) outs(%696 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb79(%698: f32, %699: f32, %700: f32):
      %701 = arith.mulf %698, %699 : f32
      %702 = arith.addf %700, %701 : f32
      linalg.yield %702 : f32
    } -> tensor<1x8x32x32xf32>
    %703 = tensor.empty() : tensor<1x32x8x32xf32>
    %704 = linalg.transpose ins(%697:tensor<1x8x32x32xf32>) outs(%703:tensor<1x32x8x32xf32>) permutation = [0, 2, 1, 3]
    %705 = tensor.collapse_shape %704 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x32x8x32xf32> into tensor<8192xf32>
    %706 = tensor.expand_shape %705 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 256] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x32x256xf32>
    %707 = tensor.empty() : tensor<1x32x256xf32>
    %708 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%706 : tensor<1x32x256xf32>) outs(%707 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.attn_sub_norm"} {
    ^bb80(%709: f32, %710: f32):
      %711 = arith.constant 2.000000e+00 : f32
      %712 = math.powf %709, %711 : f32
      linalg.yield %712 : f32
    } -> tensor<1x32x256xf32>
    %713 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.attn_sub_norm"} 0.000000e+00 : f32
    %714 = tensor.splat %713 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.attn_sub_norm"} : tensor<1x32xf32>
    %715 = linalg.reduce ins(%708:tensor<1x32x256xf32>) outs(%714:tensor<1x32xf32>) dimensions = [2]
    (%716: f32, %717: f32) {
      %718 = arith.addf %716, %717 : f32
      linalg.yield %718 : f32
    }
    %719 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.attn_sub_norm"} 2.560000e+02 : f32
    %720 = tensor.splat %719 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.attn_sub_norm"} : tensor<1x32xf32>
    %721 = tensor.empty() : tensor<1x32xf32>
    %722 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%715, %720 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%721 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.attn_sub_norm"} {
    ^bb81(%723: f32, %724: f32, %725: f32):
      %726 = arith.divf %723, %724 : f32
      linalg.yield %726 : f32
    } -> tensor<1x32xf32>
    %727 = tensor.collapse_shape %722 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.attn_sub_norm"} : tensor<1x32xf32> into tensor<32xf32>
    %728 = tensor.expand_shape %727 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.attn_sub_norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %729 = arith.constant {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.attn_sub_norm"} 1.000000e-05 : f32
    %730 = tensor.splat %729 {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.attn_sub_norm"} : tensor<1x32x1xf32>
    %731 = tensor.empty() : tensor<1x32x1xf32>
    %732 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%728, %730 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%731 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.attn_sub_norm"} {
    ^bb82(%733: f32, %734: f32, %735: f32):
      %736 = arith.addf %733, %734 : f32
      linalg.yield %736 : f32
    } -> tensor<1x32x1xf32>
    %737 = tensor.empty() : tensor<1x32x1xf32>
    %738 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%732 : tensor<1x32x1xf32>) outs(%737 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.attn_sub_norm"} {
    ^bb83(%739: f32, %740: f32):
      %741 = math.rsqrt %739 : f32
      linalg.yield %741 : f32
    } -> tensor<1x32x1xf32>
    %742 = tensor.empty() : tensor<1x32x256xf32>
    %743 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%706, %738 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%742 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.attn_sub_norm"} {
    ^bb84(%744: f32, %745: f32, %746: f32):
      %747 = arith.mulf %744, %745 : f32
      linalg.yield %747 : f32
    } -> tensor<1x32x256xf32>
    %748 = tensor.empty() : tensor<1x32x256xf32>
    %749 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %743 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%748 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.attn_sub_norm"} {
    ^bb85(%750: f32, %751: f32, %752: f32):
      %753 = arith.mulf %750, %751 : f32
      linalg.yield %753 : f32
    } -> tensor<1x32x256xf32>
    %754 = tensor.empty() : tensor<1x32x256xf32>
    %755 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%749 : tensor<1x32x256xf32>) outs(%754 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_6", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} {
    ^bb86(%756: f32, %757: f32):
      %758 = math.absf %756 : f32
      linalg.yield %758 : f32
    } -> tensor<1x32x256xf32>
    %759 = arith.constant {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} 0xff800000 : f32
    %760 = arith.constant {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} 0 : i64
    %761 = tensor.splat %759 {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} : tensor<1x32xf32>
    %762 = tensor.splat %760 {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} : tensor<1x32xi64>
    %763, %764 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%755 : tensor<1x32x256xf32>) outs(%761, %762 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} {
    ^bb87(%765: f32, %766: f32, %767: i64):
      %768 = linalg.index 2 : index
      %769 = arith.index_cast %768 : index to i64
      %770 = arith.cmpf ogt, %765, %766 : f32
      %771 = arith.select %770, %765, %766 : f32
      %772 = arith.select %770, %769, %767 : i64
      linalg.yield %771, %772 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %773 = tensor.collapse_shape %763 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %774 = tensor.expand_shape %773 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %775 = tensor.collapse_shape %764 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %776 = tensor.expand_shape %775 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %777 = func.call @aten_clamp__default(%774) {prov.region_id = "aten_clamp__default_3", prov.dispatch_id = "aten_clamp__default_3"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %778 = tensor.empty() : tensor<1x32x1xf32>
    %779 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%777 : tensor<1x32x1xf32>) outs(%778 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_6", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} {
    ^bb88(%780: f32, %781: f32):
      %782 = arith.constant 1.000000e+00 : f32
      %783 = arith.divf %782, %780 : f32
      linalg.yield %783 : f32
    } -> tensor<1x32x1xf32>
    %784 = arith.constant {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} 1.270000e+02 : f32
    %785 = tensor.splat %784 {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} : tensor<1x32x1xf32>
    %786 = tensor.empty() : tensor<1x32x1xf32>
    %787 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%779, %785 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%786 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} {
    ^bb89(%788: f32, %789: f32, %790: f32):
      %791 = arith.mulf %788, %789 : f32
      linalg.yield %791 : f32
    } -> tensor<1x32x1xf32>
    %792 = tensor.empty() : tensor<1x32x256xf32>
    %793 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%749, %787 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%792 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} {
    ^bb90(%794: f32, %795: f32, %796: f32):
      %797 = arith.mulf %794, %795 : f32
      linalg.yield %797 : f32
    } -> tensor<1x32x256xf32>
    %798 = tensor.empty() : tensor<1x32x256xf32>
    %799 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%793 : tensor<1x32x256xf32>) outs(%798 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_6", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} {
    ^bb91(%800: f32, %801: f32):
      %802 = math.roundeven %800 : f32
      linalg.yield %802 : f32
    } -> tensor<1x32x256xf32>
    %803 = tensor.empty() : tensor<1x32x256xf32>
    %804 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%799 : tensor<1x32x256xf32>) outs(%803 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_6", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} {
    ^bb92(%805: f32, %806: f32):
      %807 = arith.constant -1.280000e+02 : f32
      %808 = arith.maximumf %805, %807 : f32
      %809 = arith.constant 1.270000e+02 : f32
      %810 = arith.minimumf %808, %809 : f32
      linalg.yield %810 : f32
    } -> tensor<1x32x256xf32>
    %811 = tensor.empty() : tensor<1x32x256xf32>
    %812 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%804, %787 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%811 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_7", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} {
    ^bb93(%813: f32, %814: f32, %815: f32):
      %816 = arith.divf %813, %814 : f32
      linalg.yield %816 : f32
    } -> tensor<1x32x256xf32>
    %817 = tensor.empty() : tensor<256x256xf32>
    %818 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<256x256xf32>) outs(%817 : tensor<256x256xf32>) attrs =  {prov.region_id = "abs_7", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} {
    ^bb94(%819: f32, %820: f32):
      %821 = math.absf %819 : f32
      linalg.yield %821 : f32
    } -> tensor<256x256xf32>
    %822 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} 0.000000e+00 : f32
    %823 = tensor.splat %822 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} : tensor<f32>
    %824 = linalg.reduce ins(%818:tensor<256x256xf32>) outs(%823:tensor<f32>) dimensions = [0, 1]
    (%825: f32, %826: f32) {
      %827 = arith.addf %825, %826 : f32
      linalg.yield %827 : f32
    }
    %828 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} 6.553600e+04 : f32
    %829 = tensor.splat %828 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} : tensor<f32>
    %830 = tensor.empty() : tensor<f32>
    %831 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%824, %829 : tensor<f32>, tensor<f32>) outs(%830 : tensor<f32>) attrs =  {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} {
    ^bb95(%832: f32, %833: f32, %834: f32):
      %835 = arith.divf %832, %833 : f32
      linalg.yield %835 : f32
    } -> tensor<f32>
    %836 = func.call @aten_clamp__default_1(%831) {prov.region_id = "aten_clamp__default_1_3", prov.dispatch_id = "aten_clamp__default_1_3"} : (tensor<f32>) -> tensor<f32>
    %837 = tensor.empty() : tensor<f32>
    %838 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%836 : tensor<f32>) outs(%837 : tensor<f32>) attrs =  {prov.region_id = "elementwise_7", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} {
    ^bb96(%839: f32, %840: f32):
      %841 = arith.constant 1.000000e+00 : f32
      %842 = arith.divf %841, %839 : f32
      linalg.yield %842 : f32
    } -> tensor<f32>
    %843 = arith.constant {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} 1.000000e+00 : f32
    %844 = tensor.splat %843 {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} : tensor<f32>
    %845 = tensor.empty() : tensor<f32>
    %846 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%838, %844 : tensor<f32>, tensor<f32>) outs(%845 : tensor<f32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} {
    ^bb97(%847: f32, %848: f32, %849: f32):
      %850 = arith.mulf %847, %848 : f32
      linalg.yield %850 : f32
    } -> tensor<f32>
    %851 = tensor.empty() : tensor<256x256xf32>
    %852 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3, %846 : tensor<256x256xf32>, tensor<f32>) outs(%851 : tensor<256x256xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} {
    ^bb98(%853: f32, %854: f32, %855: f32):
      %856 = arith.mulf %853, %854 : f32
      linalg.yield %856 : f32
    } -> tensor<256x256xf32>
    %857 = tensor.empty() : tensor<256x256xf32>
    %858 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%852 : tensor<256x256xf32>) outs(%857 : tensor<256x256xf32>) attrs =  {prov.region_id = "round_7", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} {
    ^bb99(%859: f32, %860: f32):
      %861 = math.roundeven %859 : f32
      linalg.yield %861 : f32
    } -> tensor<256x256xf32>
    %862 = tensor.empty() : tensor<256x256xf32>
    %863 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%858 : tensor<256x256xf32>) outs(%862 : tensor<256x256xf32>) attrs =  {prov.region_id = "minmax_7", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} {
    ^bb100(%864: f32, %865: f32):
      %866 = arith.constant -1.000000e+00 : f32
      %867 = arith.maximumf %864, %866 : f32
      %868 = arith.constant 1.000000e+00 : f32
      %869 = arith.minimumf %867, %868 : f32
      linalg.yield %869 : f32
    } -> tensor<256x256xf32>
    %870 = tensor.empty() : tensor<256x256xf32>
    %871 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%863, %846 : tensor<256x256xf32>, tensor<f32>) outs(%870 : tensor<256x256xf32>) attrs =  {prov.region_id = "div_8", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} {
    ^bb101(%872: f32, %873: f32, %874: f32):
      %875 = arith.divf %872, %873 : f32
      linalg.yield %875 : f32
    } -> tensor<256x256xf32>
    %876 = tensor.empty() : tensor<256x256xf32>
    %877 = linalg.transpose ins(%871:tensor<256x256xf32>) outs(%876:tensor<256x256xf32>) permutation = [1, 0]
    %878 = tensor.empty() : tensor<1x32x256xf32>
    %879 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %880 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%879 : f32) outs(%878 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %881 = linalg.matmul {prov.region_id = "matmul_5", prov.dispatch_id = "matmul_5", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.self_attn.o_proj"} ins(%812, %877 : tensor<1x32x256xf32>, tensor<256x256xf32>) outs(%880 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %882 = tensor.empty() : tensor<1x32x256xf32>
    %883 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %881 : tensor<1x32x256xf32>, tensor<1x32x256xf32>) outs(%882 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb102(%884: f32, %885: f32, %886: f32):
      %887 = arith.addf %884, %885 : f32
      linalg.yield %887 : f32
    } -> tensor<1x32x256xf32>
    %888 = tensor.empty() : tensor<1x32x256xf32>
    %889 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%883 : tensor<1x32x256xf32>) outs(%888 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.post_attention_layernorm"} {
    ^bb103(%890: f32, %891: f32):
      %892 = arith.constant 2.000000e+00 : f32
      %893 = math.powf %890, %892 : f32
      linalg.yield %893 : f32
    } -> tensor<1x32x256xf32>
    %894 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.post_attention_layernorm"} 0.000000e+00 : f32
    %895 = tensor.splat %894 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.post_attention_layernorm"} : tensor<1x32xf32>
    %896 = linalg.reduce ins(%889:tensor<1x32x256xf32>) outs(%895:tensor<1x32xf32>) dimensions = [2]
    (%897: f32, %898: f32) {
      %899 = arith.addf %897, %898 : f32
      linalg.yield %899 : f32
    }
    %900 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.post_attention_layernorm"} 2.560000e+02 : f32
    %901 = tensor.splat %900 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.post_attention_layernorm"} : tensor<1x32xf32>
    %902 = tensor.empty() : tensor<1x32xf32>
    %903 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%896, %901 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%902 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.post_attention_layernorm"} {
    ^bb104(%904: f32, %905: f32, %906: f32):
      %907 = arith.divf %904, %905 : f32
      linalg.yield %907 : f32
    } -> tensor<1x32xf32>
    %908 = tensor.collapse_shape %903 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.post_attention_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %909 = tensor.expand_shape %908 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.post_attention_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %910 = arith.constant {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.post_attention_layernorm"} 1.000000e-05 : f32
    %911 = tensor.splat %910 {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.post_attention_layernorm"} : tensor<1x32x1xf32>
    %912 = tensor.empty() : tensor<1x32x1xf32>
    %913 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%909, %911 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%912 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.post_attention_layernorm"} {
    ^bb105(%914: f32, %915: f32, %916: f32):
      %917 = arith.addf %914, %915 : f32
      linalg.yield %917 : f32
    } -> tensor<1x32x1xf32>
    %918 = tensor.empty() : tensor<1x32x1xf32>
    %919 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%913 : tensor<1x32x1xf32>) outs(%918 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.post_attention_layernorm"} {
    ^bb106(%920: f32, %921: f32):
      %922 = math.rsqrt %920 : f32
      linalg.yield %922 : f32
    } -> tensor<1x32x1xf32>
    %923 = tensor.empty() : tensor<1x32x256xf32>
    %924 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%883, %919 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%923 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.post_attention_layernorm"} {
    ^bb107(%925: f32, %926: f32, %927: f32):
      %928 = arith.mulf %925, %926 : f32
      linalg.yield %928 : f32
    } -> tensor<1x32x256xf32>
    %929 = tensor.empty() : tensor<1x32x256xf32>
    %930 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %924 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%929 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.post_attention_layernorm"} {
    ^bb108(%931: f32, %932: f32, %933: f32):
      %934 = arith.mulf %931, %932 : f32
      linalg.yield %934 : f32
    } -> tensor<1x32x256xf32>
    %935 = tensor.empty() : tensor<1x32x256xf32>
    %936 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%930 : tensor<1x32x256xf32>) outs(%935 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_8", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} {
    ^bb109(%937: f32, %938: f32):
      %939 = math.absf %937 : f32
      linalg.yield %939 : f32
    } -> tensor<1x32x256xf32>
    %940 = arith.constant {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} 0xff800000 : f32
    %941 = arith.constant {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} 0 : i64
    %942 = tensor.splat %940 {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} : tensor<1x32xf32>
    %943 = tensor.splat %941 {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} : tensor<1x32xi64>
    %944, %945 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%936 : tensor<1x32x256xf32>) outs(%942, %943 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} {
    ^bb110(%946: f32, %947: f32, %948: i64):
      %949 = linalg.index 2 : index
      %950 = arith.index_cast %949 : index to i64
      %951 = arith.cmpf ogt, %946, %947 : f32
      %952 = arith.select %951, %946, %947 : f32
      %953 = arith.select %951, %950, %948 : i64
      linalg.yield %952, %953 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %954 = tensor.collapse_shape %944 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %955 = tensor.expand_shape %954 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %956 = tensor.collapse_shape %945 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %957 = tensor.expand_shape %956 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %958 = func.call @aten_clamp__default(%955) {prov.region_id = "aten_clamp__default_4", prov.dispatch_id = "aten_clamp__default_4"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %959 = tensor.empty() : tensor<1x32x1xf32>
    %960 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%958 : tensor<1x32x1xf32>) outs(%959 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_8", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} {
    ^bb111(%961: f32, %962: f32):
      %963 = arith.constant 1.000000e+00 : f32
      %964 = arith.divf %963, %961 : f32
      linalg.yield %964 : f32
    } -> tensor<1x32x1xf32>
    %965 = arith.constant {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} 1.270000e+02 : f32
    %966 = tensor.splat %965 {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} : tensor<1x32x1xf32>
    %967 = tensor.empty() : tensor<1x32x1xf32>
    %968 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%960, %966 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%967 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} {
    ^bb112(%969: f32, %970: f32, %971: f32):
      %972 = arith.mulf %969, %970 : f32
      linalg.yield %972 : f32
    } -> tensor<1x32x1xf32>
    %973 = tensor.empty() : tensor<1x32x256xf32>
    %974 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%930, %968 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%973 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} {
    ^bb113(%975: f32, %976: f32, %977: f32):
      %978 = arith.mulf %975, %976 : f32
      linalg.yield %978 : f32
    } -> tensor<1x32x256xf32>
    %979 = tensor.empty() : tensor<1x32x256xf32>
    %980 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%974 : tensor<1x32x256xf32>) outs(%979 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_8", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} {
    ^bb114(%981: f32, %982: f32):
      %983 = math.roundeven %981 : f32
      linalg.yield %983 : f32
    } -> tensor<1x32x256xf32>
    %984 = tensor.empty() : tensor<1x32x256xf32>
    %985 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%980 : tensor<1x32x256xf32>) outs(%984 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_8", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} {
    ^bb115(%986: f32, %987: f32):
      %988 = arith.constant -1.280000e+02 : f32
      %989 = arith.maximumf %986, %988 : f32
      %990 = arith.constant 1.270000e+02 : f32
      %991 = arith.minimumf %989, %990 : f32
      linalg.yield %991 : f32
    } -> tensor<1x32x256xf32>
    %992 = tensor.empty() : tensor<1x32x256xf32>
    %993 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%985, %968 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%992 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_9", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} {
    ^bb116(%994: f32, %995: f32, %996: f32):
      %997 = arith.divf %994, %995 : f32
      linalg.yield %997 : f32
    } -> tensor<1x32x256xf32>
    %998 = tensor.empty() : tensor<512x256xf32>
    %999 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<512x256xf32>) outs(%998 : tensor<512x256xf32>) attrs =  {prov.region_id = "abs_9", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} {
    ^bb117(%1000: f32, %1001: f32):
      %1002 = math.absf %1000 : f32
      linalg.yield %1002 : f32
    } -> tensor<512x256xf32>
    %1003 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} 0.000000e+00 : f32
    %1004 = tensor.splat %1003 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} : tensor<f32>
    %1005 = linalg.reduce ins(%999:tensor<512x256xf32>) outs(%1004:tensor<f32>) dimensions = [0, 1]
    (%1006: f32, %1007: f32) {
      %1008 = arith.addf %1006, %1007 : f32
      linalg.yield %1008 : f32
    }
    %1009 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} 1.310720e+05 : f32
    %1010 = tensor.splat %1009 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} : tensor<f32>
    %1011 = tensor.empty() : tensor<f32>
    %1012 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1005, %1010 : tensor<f32>, tensor<f32>) outs(%1011 : tensor<f32>) attrs =  {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} {
    ^bb118(%1013: f32, %1014: f32, %1015: f32):
      %1016 = arith.divf %1013, %1014 : f32
      linalg.yield %1016 : f32
    } -> tensor<f32>
    %1017 = func.call @aten_clamp__default_1(%1012) {prov.region_id = "aten_clamp__default_1_4", prov.dispatch_id = "aten_clamp__default_1_4"} : (tensor<f32>) -> tensor<f32>
    %1018 = tensor.empty() : tensor<f32>
    %1019 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1017 : tensor<f32>) outs(%1018 : tensor<f32>) attrs =  {prov.region_id = "elementwise_9", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} {
    ^bb119(%1020: f32, %1021: f32):
      %1022 = arith.constant 1.000000e+00 : f32
      %1023 = arith.divf %1022, %1020 : f32
      linalg.yield %1023 : f32
    } -> tensor<f32>
    %1024 = arith.constant {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} 1.000000e+00 : f32
    %1025 = tensor.splat %1024 {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} : tensor<f32>
    %1026 = tensor.empty() : tensor<f32>
    %1027 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1019, %1025 : tensor<f32>, tensor<f32>) outs(%1026 : tensor<f32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} {
    ^bb120(%1028: f32, %1029: f32, %1030: f32):
      %1031 = arith.mulf %1028, %1029 : f32
      linalg.yield %1031 : f32
    } -> tensor<f32>
    %1032 = tensor.empty() : tensor<512x256xf32>
    %1033 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %1027 : tensor<512x256xf32>, tensor<f32>) outs(%1032 : tensor<512x256xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} {
    ^bb121(%1034: f32, %1035: f32, %1036: f32):
      %1037 = arith.mulf %1034, %1035 : f32
      linalg.yield %1037 : f32
    } -> tensor<512x256xf32>
    %1038 = tensor.empty() : tensor<512x256xf32>
    %1039 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1033 : tensor<512x256xf32>) outs(%1038 : tensor<512x256xf32>) attrs =  {prov.region_id = "round_9", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} {
    ^bb122(%1040: f32, %1041: f32):
      %1042 = math.roundeven %1040 : f32
      linalg.yield %1042 : f32
    } -> tensor<512x256xf32>
    %1043 = tensor.empty() : tensor<512x256xf32>
    %1044 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1039 : tensor<512x256xf32>) outs(%1043 : tensor<512x256xf32>) attrs =  {prov.region_id = "minmax_9", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} {
    ^bb123(%1045: f32, %1046: f32):
      %1047 = arith.constant -1.000000e+00 : f32
      %1048 = arith.maximumf %1045, %1047 : f32
      %1049 = arith.constant 1.000000e+00 : f32
      %1050 = arith.minimumf %1048, %1049 : f32
      linalg.yield %1050 : f32
    } -> tensor<512x256xf32>
    %1051 = tensor.empty() : tensor<512x256xf32>
    %1052 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1044, %1027 : tensor<512x256xf32>, tensor<f32>) outs(%1051 : tensor<512x256xf32>) attrs =  {prov.region_id = "div_10", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} {
    ^bb124(%1053: f32, %1054: f32, %1055: f32):
      %1056 = arith.divf %1053, %1054 : f32
      linalg.yield %1056 : f32
    } -> tensor<512x256xf32>
    %1057 = tensor.empty() : tensor<256x512xf32>
    %1058 = linalg.transpose ins(%1052:tensor<512x256xf32>) outs(%1057:tensor<256x512xf32>) permutation = [1, 0]
    %1059 = tensor.empty() : tensor<1x32x512xf32>
    %1060 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %1061 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%1060 : f32) outs(%1059 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %1062 = linalg.matmul {prov.region_id = "matmul_6", prov.dispatch_id = "matmul_6", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.gate_proj"} ins(%993, %1058 : tensor<1x32x256xf32>, tensor<256x512xf32>) outs(%1061 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %1063 = tensor.empty() : tensor<1x32x512xf32>
    %1064 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1062 : tensor<1x32x512xf32>) outs(%1063 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "minmax_10", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu.default", prov.orig_dtype = "float32"} {
    ^bb125(%1065: f32, %1066: f32):
      %1067 = arith.constant 0.000000e+00 : f32
      %1068 = arith.maximumf %1065, %1067 : f32
      linalg.yield %1068 : f32
    } -> tensor<1x32x512xf32>
    %1069 = tensor.empty() : tensor<1x32x512xf32>
    %1070 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1064 : tensor<1x32x512xf32>) outs(%1069 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb126(%1071: f32, %1072: f32):
      %1073 = arith.constant 2.000000e+00 : f32
      %1074 = math.powf %1071, %1073 : f32
      linalg.yield %1074 : f32
    } -> tensor<1x32x512xf32>
    %1075 = tensor.empty() : tensor<1x32x256xf32>
    %1076 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%930 : tensor<1x32x256xf32>) outs(%1075 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_10", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} {
    ^bb127(%1077: f32, %1078: f32):
      %1079 = math.absf %1077 : f32
      linalg.yield %1079 : f32
    } -> tensor<1x32x256xf32>
    %1080 = arith.constant {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} 0xff800000 : f32
    %1081 = arith.constant {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} 0 : i64
    %1082 = tensor.splat %1080 {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} : tensor<1x32xf32>
    %1083 = tensor.splat %1081 {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} : tensor<1x32xi64>
    %1084, %1085 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1076 : tensor<1x32x256xf32>) outs(%1082, %1083 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} {
    ^bb128(%1086: f32, %1087: f32, %1088: i64):
      %1089 = linalg.index 2 : index
      %1090 = arith.index_cast %1089 : index to i64
      %1091 = arith.cmpf ogt, %1086, %1087 : f32
      %1092 = arith.select %1091, %1086, %1087 : f32
      %1093 = arith.select %1091, %1090, %1088 : i64
      linalg.yield %1092, %1093 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1094 = tensor.collapse_shape %1084 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1095 = tensor.expand_shape %1094 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1096 = tensor.collapse_shape %1085 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1097 = tensor.expand_shape %1096 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1098 = func.call @aten_clamp__default(%1095) {prov.region_id = "aten_clamp__default_5", prov.dispatch_id = "aten_clamp__default_5"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1099 = tensor.empty() : tensor<1x32x1xf32>
    %1100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1098 : tensor<1x32x1xf32>) outs(%1099 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_10", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} {
    ^bb129(%1101: f32, %1102: f32):
      %1103 = arith.constant 1.000000e+00 : f32
      %1104 = arith.divf %1103, %1101 : f32
      linalg.yield %1104 : f32
    } -> tensor<1x32x1xf32>
    %1105 = arith.constant {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} 1.270000e+02 : f32
    %1106 = tensor.splat %1105 {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} : tensor<1x32x1xf32>
    %1107 = tensor.empty() : tensor<1x32x1xf32>
    %1108 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1100, %1106 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1107 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} {
    ^bb130(%1109: f32, %1110: f32, %1111: f32):
      %1112 = arith.mulf %1109, %1110 : f32
      linalg.yield %1112 : f32
    } -> tensor<1x32x1xf32>
    %1113 = tensor.empty() : tensor<1x32x256xf32>
    %1114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%930, %1108 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1113 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} {
    ^bb131(%1115: f32, %1116: f32, %1117: f32):
      %1118 = arith.mulf %1115, %1116 : f32
      linalg.yield %1118 : f32
    } -> tensor<1x32x256xf32>
    %1119 = tensor.empty() : tensor<1x32x256xf32>
    %1120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1114 : tensor<1x32x256xf32>) outs(%1119 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_10", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} {
    ^bb132(%1121: f32, %1122: f32):
      %1123 = math.roundeven %1121 : f32
      linalg.yield %1123 : f32
    } -> tensor<1x32x256xf32>
    %1124 = tensor.empty() : tensor<1x32x256xf32>
    %1125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1120 : tensor<1x32x256xf32>) outs(%1124 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_11", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} {
    ^bb133(%1126: f32, %1127: f32):
      %1128 = arith.constant -1.280000e+02 : f32
      %1129 = arith.maximumf %1126, %1128 : f32
      %1130 = arith.constant 1.270000e+02 : f32
      %1131 = arith.minimumf %1129, %1130 : f32
      linalg.yield %1131 : f32
    } -> tensor<1x32x256xf32>
    %1132 = tensor.empty() : tensor<1x32x256xf32>
    %1133 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1125, %1108 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1132 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_11", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} {
    ^bb134(%1134: f32, %1135: f32, %1136: f32):
      %1137 = arith.divf %1134, %1135 : f32
      linalg.yield %1137 : f32
    } -> tensor<1x32x256xf32>
    %1138 = tensor.empty() : tensor<512x256xf32>
    %1139 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<512x256xf32>) outs(%1138 : tensor<512x256xf32>) attrs =  {prov.region_id = "abs_11", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} {
    ^bb135(%1140: f32, %1141: f32):
      %1142 = math.absf %1140 : f32
      linalg.yield %1142 : f32
    } -> tensor<512x256xf32>
    %1143 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} 0.000000e+00 : f32
    %1144 = tensor.splat %1143 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} : tensor<f32>
    %1145 = linalg.reduce ins(%1139:tensor<512x256xf32>) outs(%1144:tensor<f32>) dimensions = [0, 1]
    (%1146: f32, %1147: f32) {
      %1148 = arith.addf %1146, %1147 : f32
      linalg.yield %1148 : f32
    }
    %1149 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} 1.310720e+05 : f32
    %1150 = tensor.splat %1149 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} : tensor<f32>
    %1151 = tensor.empty() : tensor<f32>
    %1152 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1145, %1150 : tensor<f32>, tensor<f32>) outs(%1151 : tensor<f32>) attrs =  {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} {
    ^bb136(%1153: f32, %1154: f32, %1155: f32):
      %1156 = arith.divf %1153, %1154 : f32
      linalg.yield %1156 : f32
    } -> tensor<f32>
    %1157 = func.call @aten_clamp__default_1(%1152) {prov.region_id = "aten_clamp__default_1_5", prov.dispatch_id = "aten_clamp__default_1_5"} : (tensor<f32>) -> tensor<f32>
    %1158 = tensor.empty() : tensor<f32>
    %1159 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1157 : tensor<f32>) outs(%1158 : tensor<f32>) attrs =  {prov.region_id = "elementwise_11", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} {
    ^bb137(%1160: f32, %1161: f32):
      %1162 = arith.constant 1.000000e+00 : f32
      %1163 = arith.divf %1162, %1160 : f32
      linalg.yield %1163 : f32
    } -> tensor<f32>
    %1164 = arith.constant {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} 1.000000e+00 : f32
    %1165 = tensor.splat %1164 {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} : tensor<f32>
    %1166 = tensor.empty() : tensor<f32>
    %1167 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1159, %1165 : tensor<f32>, tensor<f32>) outs(%1166 : tensor<f32>) attrs =  {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} {
    ^bb138(%1168: f32, %1169: f32, %1170: f32):
      %1171 = arith.mulf %1168, %1169 : f32
      linalg.yield %1171 : f32
    } -> tensor<f32>
    %1172 = tensor.empty() : tensor<512x256xf32>
    %1173 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6, %1167 : tensor<512x256xf32>, tensor<f32>) outs(%1172 : tensor<512x256xf32>) attrs =  {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} {
    ^bb139(%1174: f32, %1175: f32, %1176: f32):
      %1177 = arith.mulf %1174, %1175 : f32
      linalg.yield %1177 : f32
    } -> tensor<512x256xf32>
    %1178 = tensor.empty() : tensor<512x256xf32>
    %1179 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1173 : tensor<512x256xf32>) outs(%1178 : tensor<512x256xf32>) attrs =  {prov.region_id = "round_11", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} {
    ^bb140(%1180: f32, %1181: f32):
      %1182 = math.roundeven %1180 : f32
      linalg.yield %1182 : f32
    } -> tensor<512x256xf32>
    %1183 = tensor.empty() : tensor<512x256xf32>
    %1184 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1179 : tensor<512x256xf32>) outs(%1183 : tensor<512x256xf32>) attrs =  {prov.region_id = "minmax_12", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} {
    ^bb141(%1185: f32, %1186: f32):
      %1187 = arith.constant -1.000000e+00 : f32
      %1188 = arith.maximumf %1185, %1187 : f32
      %1189 = arith.constant 1.000000e+00 : f32
      %1190 = arith.minimumf %1188, %1189 : f32
      linalg.yield %1190 : f32
    } -> tensor<512x256xf32>
    %1191 = tensor.empty() : tensor<512x256xf32>
    %1192 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1184, %1167 : tensor<512x256xf32>, tensor<f32>) outs(%1191 : tensor<512x256xf32>) attrs =  {prov.region_id = "div_12", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} {
    ^bb142(%1193: f32, %1194: f32, %1195: f32):
      %1196 = arith.divf %1193, %1194 : f32
      linalg.yield %1196 : f32
    } -> tensor<512x256xf32>
    %1197 = tensor.empty() : tensor<256x512xf32>
    %1198 = linalg.transpose ins(%1192:tensor<512x256xf32>) outs(%1197:tensor<256x512xf32>) permutation = [1, 0]
    %1199 = tensor.empty() : tensor<1x32x512xf32>
    %1200 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %1201 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%1200 : f32) outs(%1199 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %1202 = linalg.matmul {prov.region_id = "matmul_7", prov.dispatch_id = "matmul_7", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.up_proj"} ins(%1133, %1198 : tensor<1x32x256xf32>, tensor<256x512xf32>) outs(%1201 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %1203 = tensor.empty() : tensor<1x32x512xf32>
    %1204 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1070, %1202 : tensor<1x32x512xf32>, tensor<1x32x512xf32>) outs(%1203 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_34", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb143(%1205: f32, %1206: f32, %1207: f32):
      %1208 = arith.mulf %1205, %1206 : f32
      linalg.yield %1208 : f32
    } -> tensor<1x32x512xf32>
    %1209 = tensor.empty() : tensor<1x32x512xf32>
    %1210 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1204 : tensor<1x32x512xf32>) outs(%1209 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.ffn_sub_norm"} {
    ^bb144(%1211: f32, %1212: f32):
      %1213 = arith.constant 2.000000e+00 : f32
      %1214 = math.powf %1211, %1213 : f32
      linalg.yield %1214 : f32
    } -> tensor<1x32x512xf32>
    %1215 = arith.constant {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.ffn_sub_norm"} 0.000000e+00 : f32
    %1216 = tensor.splat %1215 {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.ffn_sub_norm"} : tensor<1x32xf32>
    %1217 = linalg.reduce ins(%1210:tensor<1x32x512xf32>) outs(%1216:tensor<1x32xf32>) dimensions = [2]
    (%1218: f32, %1219: f32) {
      %1220 = arith.addf %1218, %1219 : f32
      linalg.yield %1220 : f32
    }
    %1221 = arith.constant {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.ffn_sub_norm"} 5.120000e+02 : f32
    %1222 = tensor.splat %1221 {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.ffn_sub_norm"} : tensor<1x32xf32>
    %1223 = tensor.empty() : tensor<1x32xf32>
    %1224 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1217, %1222 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1223 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.ffn_sub_norm"} {
    ^bb145(%1225: f32, %1226: f32, %1227: f32):
      %1228 = arith.divf %1225, %1226 : f32
      linalg.yield %1228 : f32
    } -> tensor<1x32xf32>
    %1229 = tensor.collapse_shape %1224 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.ffn_sub_norm"} : tensor<1x32xf32> into tensor<32xf32>
    %1230 = tensor.expand_shape %1229 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.ffn_sub_norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1231 = arith.constant {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.ffn_sub_norm"} 1.000000e-05 : f32
    %1232 = tensor.splat %1231 {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.ffn_sub_norm"} : tensor<1x32x1xf32>
    %1233 = tensor.empty() : tensor<1x32x1xf32>
    %1234 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1230, %1232 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1233 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.ffn_sub_norm"} {
    ^bb146(%1235: f32, %1236: f32, %1237: f32):
      %1238 = arith.addf %1235, %1236 : f32
      linalg.yield %1238 : f32
    } -> tensor<1x32x1xf32>
    %1239 = tensor.empty() : tensor<1x32x1xf32>
    %1240 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1234 : tensor<1x32x1xf32>) outs(%1239 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.ffn_sub_norm"} {
    ^bb147(%1241: f32, %1242: f32):
      %1243 = math.rsqrt %1241 : f32
      linalg.yield %1243 : f32
    } -> tensor<1x32x1xf32>
    %1244 = tensor.empty() : tensor<1x32x512xf32>
    %1245 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1204, %1240 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%1244 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_35", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.ffn_sub_norm"} {
    ^bb148(%1246: f32, %1247: f32, %1248: f32):
      %1249 = arith.mulf %1246, %1247 : f32
      linalg.yield %1249 : f32
    } -> tensor<1x32x512xf32>
    %1250 = tensor.empty() : tensor<1x32x512xf32>
    %1251 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %1245 : tensor<512xf32>, tensor<1x32x512xf32>) outs(%1250 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_36", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.ffn_sub_norm"} {
    ^bb149(%1252: f32, %1253: f32, %1254: f32):
      %1255 = arith.mulf %1252, %1253 : f32
      linalg.yield %1255 : f32
    } -> tensor<1x32x512xf32>
    %1256 = tensor.empty() : tensor<1x32x512xf32>
    %1257 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1251 : tensor<1x32x512xf32>) outs(%1256 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "abs_12", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} {
    ^bb150(%1258: f32, %1259: f32):
      %1260 = math.absf %1258 : f32
      linalg.yield %1260 : f32
    } -> tensor<1x32x512xf32>
    %1261 = arith.constant {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} 0xff800000 : f32
    %1262 = arith.constant {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} 0 : i64
    %1263 = tensor.splat %1261 {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} : tensor<1x32xf32>
    %1264 = tensor.splat %1262 {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} : tensor<1x32xi64>
    %1265, %1266 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1257 : tensor<1x32x512xf32>) outs(%1263, %1264 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} {
    ^bb151(%1267: f32, %1268: f32, %1269: i64):
      %1270 = linalg.index 2 : index
      %1271 = arith.index_cast %1270 : index to i64
      %1272 = arith.cmpf ogt, %1267, %1268 : f32
      %1273 = arith.select %1272, %1267, %1268 : f32
      %1274 = arith.select %1272, %1271, %1269 : i64
      linalg.yield %1273, %1274 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1275 = tensor.collapse_shape %1265 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1276 = tensor.expand_shape %1275 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1277 = tensor.collapse_shape %1266 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1278 = tensor.expand_shape %1277 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1279 = func.call @aten_clamp__default(%1276) {prov.region_id = "aten_clamp__default_6", prov.dispatch_id = "aten_clamp__default_6"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1280 = tensor.empty() : tensor<1x32x1xf32>
    %1281 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1279 : tensor<1x32x1xf32>) outs(%1280 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_12", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} {
    ^bb152(%1282: f32, %1283: f32):
      %1284 = arith.constant 1.000000e+00 : f32
      %1285 = arith.divf %1284, %1282 : f32
      linalg.yield %1285 : f32
    } -> tensor<1x32x1xf32>
    %1286 = arith.constant {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} 1.270000e+02 : f32
    %1287 = tensor.splat %1286 {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} : tensor<1x32x1xf32>
    %1288 = tensor.empty() : tensor<1x32x1xf32>
    %1289 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1281, %1287 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1288 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} {
    ^bb153(%1290: f32, %1291: f32, %1292: f32):
      %1293 = arith.mulf %1290, %1291 : f32
      linalg.yield %1293 : f32
    } -> tensor<1x32x1xf32>
    %1294 = tensor.empty() : tensor<1x32x512xf32>
    %1295 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1251, %1289 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%1294 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_38", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} {
    ^bb154(%1296: f32, %1297: f32, %1298: f32):
      %1299 = arith.mulf %1296, %1297 : f32
      linalg.yield %1299 : f32
    } -> tensor<1x32x512xf32>
    %1300 = tensor.empty() : tensor<1x32x512xf32>
    %1301 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1295 : tensor<1x32x512xf32>) outs(%1300 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "round_12", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} {
    ^bb155(%1302: f32, %1303: f32):
      %1304 = math.roundeven %1302 : f32
      linalg.yield %1304 : f32
    } -> tensor<1x32x512xf32>
    %1305 = tensor.empty() : tensor<1x32x512xf32>
    %1306 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1301 : tensor<1x32x512xf32>) outs(%1305 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "minmax_13", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} {
    ^bb156(%1307: f32, %1308: f32):
      %1309 = arith.constant -1.280000e+02 : f32
      %1310 = arith.maximumf %1307, %1309 : f32
      %1311 = arith.constant 1.270000e+02 : f32
      %1312 = arith.minimumf %1310, %1311 : f32
      linalg.yield %1312 : f32
    } -> tensor<1x32x512xf32>
    %1313 = tensor.empty() : tensor<1x32x512xf32>
    %1314 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1306, %1289 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%1313 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "div_13", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} {
    ^bb157(%1315: f32, %1316: f32, %1317: f32):
      %1318 = arith.divf %1315, %1316 : f32
      linalg.yield %1318 : f32
    } -> tensor<1x32x512xf32>
    %1319 = tensor.empty() : tensor<256x512xf32>
    %1320 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<256x512xf32>) outs(%1319 : tensor<256x512xf32>) attrs =  {prov.region_id = "abs_13", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} {
    ^bb158(%1321: f32, %1322: f32):
      %1323 = math.absf %1321 : f32
      linalg.yield %1323 : f32
    } -> tensor<256x512xf32>
    %1324 = arith.constant {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} 0.000000e+00 : f32
    %1325 = tensor.splat %1324 {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} : tensor<f32>
    %1326 = linalg.reduce ins(%1320:tensor<256x512xf32>) outs(%1325:tensor<f32>) dimensions = [0, 1]
    (%1327: f32, %1328: f32) {
      %1329 = arith.addf %1327, %1328 : f32
      linalg.yield %1329 : f32
    }
    %1330 = arith.constant {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} 1.310720e+05 : f32
    %1331 = tensor.splat %1330 {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} : tensor<f32>
    %1332 = tensor.empty() : tensor<f32>
    %1333 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1326, %1331 : tensor<f32>, tensor<f32>) outs(%1332 : tensor<f32>) attrs =  {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} {
    ^bb159(%1334: f32, %1335: f32, %1336: f32):
      %1337 = arith.divf %1334, %1335 : f32
      linalg.yield %1337 : f32
    } -> tensor<f32>
    %1338 = func.call @aten_clamp__default_1(%1333) {prov.region_id = "aten_clamp__default_1_6", prov.dispatch_id = "aten_clamp__default_1_6"} : (tensor<f32>) -> tensor<f32>
    %1339 = tensor.empty() : tensor<f32>
    %1340 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1338 : tensor<f32>) outs(%1339 : tensor<f32>) attrs =  {prov.region_id = "elementwise_13", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} {
    ^bb160(%1341: f32, %1342: f32):
      %1343 = arith.constant 1.000000e+00 : f32
      %1344 = arith.divf %1343, %1341 : f32
      linalg.yield %1344 : f32
    } -> tensor<f32>
    %1345 = arith.constant {prov.region_id = "mul_39", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} 1.000000e+00 : f32
    %1346 = tensor.splat %1345 {prov.region_id = "mul_39", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} : tensor<f32>
    %1347 = tensor.empty() : tensor<f32>
    %1348 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1340, %1346 : tensor<f32>, tensor<f32>) outs(%1347 : tensor<f32>) attrs =  {prov.region_id = "mul_39", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} {
    ^bb161(%1349: f32, %1350: f32, %1351: f32):
      %1352 = arith.mulf %1349, %1350 : f32
      linalg.yield %1352 : f32
    } -> tensor<f32>
    %1353 = tensor.empty() : tensor<256x512xf32>
    %1354 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7, %1348 : tensor<256x512xf32>, tensor<f32>) outs(%1353 : tensor<256x512xf32>) attrs =  {prov.region_id = "mul_40", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} {
    ^bb162(%1355: f32, %1356: f32, %1357: f32):
      %1358 = arith.mulf %1355, %1356 : f32
      linalg.yield %1358 : f32
    } -> tensor<256x512xf32>
    %1359 = tensor.empty() : tensor<256x512xf32>
    %1360 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1354 : tensor<256x512xf32>) outs(%1359 : tensor<256x512xf32>) attrs =  {prov.region_id = "round_13", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} {
    ^bb163(%1361: f32, %1362: f32):
      %1363 = math.roundeven %1361 : f32
      linalg.yield %1363 : f32
    } -> tensor<256x512xf32>
    %1364 = tensor.empty() : tensor<256x512xf32>
    %1365 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1360 : tensor<256x512xf32>) outs(%1364 : tensor<256x512xf32>) attrs =  {prov.region_id = "minmax_14", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} {
    ^bb164(%1366: f32, %1367: f32):
      %1368 = arith.constant -1.000000e+00 : f32
      %1369 = arith.maximumf %1366, %1368 : f32
      %1370 = arith.constant 1.000000e+00 : f32
      %1371 = arith.minimumf %1369, %1370 : f32
      linalg.yield %1371 : f32
    } -> tensor<256x512xf32>
    %1372 = tensor.empty() : tensor<256x512xf32>
    %1373 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1365, %1348 : tensor<256x512xf32>, tensor<f32>) outs(%1372 : tensor<256x512xf32>) attrs =  {prov.region_id = "div_14", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} {
    ^bb165(%1374: f32, %1375: f32, %1376: f32):
      %1377 = arith.divf %1374, %1375 : f32
      linalg.yield %1377 : f32
    } -> tensor<256x512xf32>
    %1378 = tensor.empty() : tensor<512x256xf32>
    %1379 = linalg.transpose ins(%1373:tensor<256x512xf32>) outs(%1378:tensor<512x256xf32>) permutation = [1, 0]
    %1380 = tensor.empty() : tensor<1x32x256xf32>
    %1381 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %1382 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%1381 : f32) outs(%1380 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %1383 = linalg.matmul {prov.region_id = "matmul_8", prov.dispatch_id = "matmul_8", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.0.mlp.down_proj"} ins(%1314, %1379 : tensor<1x32x512xf32>, tensor<512x256xf32>) outs(%1382 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %1384 = tensor.empty() : tensor<1x32x256xf32>
    %1385 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%883, %1383 : tensor<1x32x256xf32>, tensor<1x32x256xf32>) outs(%1384 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb166(%1386: f32, %1387: f32, %1388: f32):
      %1389 = arith.addf %1386, %1387 : f32
      linalg.yield %1389 : f32
    } -> tensor<1x32x256xf32>
    %1390 = tensor.empty() : tensor<1x32x256xf32>
    %1391 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1385 : tensor<1x32x256xf32>) outs(%1390 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.input_layernorm"} {
    ^bb167(%1392: f32, %1393: f32):
      %1394 = arith.constant 2.000000e+00 : f32
      %1395 = math.powf %1392, %1394 : f32
      linalg.yield %1395 : f32
    } -> tensor<1x32x256xf32>
    %1396 = arith.constant {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.input_layernorm"} 0.000000e+00 : f32
    %1397 = tensor.splat %1396 {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.input_layernorm"} : tensor<1x32xf32>
    %1398 = linalg.reduce ins(%1391:tensor<1x32x256xf32>) outs(%1397:tensor<1x32xf32>) dimensions = [2]
    (%1399: f32, %1400: f32) {
      %1401 = arith.addf %1399, %1400 : f32
      linalg.yield %1401 : f32
    }
    %1402 = arith.constant {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.input_layernorm"} 2.560000e+02 : f32
    %1403 = tensor.splat %1402 {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.input_layernorm"} : tensor<1x32xf32>
    %1404 = tensor.empty() : tensor<1x32xf32>
    %1405 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1398, %1403 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1404 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.input_layernorm"} {
    ^bb168(%1406: f32, %1407: f32, %1408: f32):
      %1409 = arith.divf %1406, %1407 : f32
      linalg.yield %1409 : f32
    } -> tensor<1x32xf32>
    %1410 = tensor.collapse_shape %1405 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.input_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %1411 = tensor.expand_shape %1410 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.input_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1412 = arith.constant {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.input_layernorm"} 1.000000e-05 : f32
    %1413 = tensor.splat %1412 {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.input_layernorm"} : tensor<1x32x1xf32>
    %1414 = tensor.empty() : tensor<1x32x1xf32>
    %1415 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1411, %1413 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1414 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.input_layernorm"} {
    ^bb169(%1416: f32, %1417: f32, %1418: f32):
      %1419 = arith.addf %1416, %1417 : f32
      linalg.yield %1419 : f32
    } -> tensor<1x32x1xf32>
    %1420 = tensor.empty() : tensor<1x32x1xf32>
    %1421 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1415 : tensor<1x32x1xf32>) outs(%1420 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.input_layernorm"} {
    ^bb170(%1422: f32, %1423: f32):
      %1424 = math.rsqrt %1422 : f32
      linalg.yield %1424 : f32
    } -> tensor<1x32x1xf32>
    %1425 = tensor.empty() : tensor<1x32x256xf32>
    %1426 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1385, %1421 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1425 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.input_layernorm"} {
    ^bb171(%1427: f32, %1428: f32, %1429: f32):
      %1430 = arith.mulf %1427, %1428 : f32
      linalg.yield %1430 : f32
    } -> tensor<1x32x256xf32>
    %1431 = tensor.empty() : tensor<1x32x256xf32>
    %1432 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20, %1426 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%1431 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_42", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.input_layernorm"} {
    ^bb172(%1433: f32, %1434: f32, %1435: f32):
      %1436 = arith.mulf %1433, %1434 : f32
      linalg.yield %1436 : f32
    } -> tensor<1x32x256xf32>
    %1437 = tensor.empty() : tensor<1x32x256xf32>
    %1438 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1432 : tensor<1x32x256xf32>) outs(%1437 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_14", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} {
    ^bb173(%1439: f32, %1440: f32):
      %1441 = math.absf %1439 : f32
      linalg.yield %1441 : f32
    } -> tensor<1x32x256xf32>
    %1442 = arith.constant {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} 0xff800000 : f32
    %1443 = arith.constant {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} 0 : i64
    %1444 = tensor.splat %1442 {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} : tensor<1x32xf32>
    %1445 = tensor.splat %1443 {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} : tensor<1x32xi64>
    %1446, %1447 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1438 : tensor<1x32x256xf32>) outs(%1444, %1445 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} {
    ^bb174(%1448: f32, %1449: f32, %1450: i64):
      %1451 = linalg.index 2 : index
      %1452 = arith.index_cast %1451 : index to i64
      %1453 = arith.cmpf ogt, %1448, %1449 : f32
      %1454 = arith.select %1453, %1448, %1449 : f32
      %1455 = arith.select %1453, %1452, %1450 : i64
      linalg.yield %1454, %1455 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1456 = tensor.collapse_shape %1446 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1457 = tensor.expand_shape %1456 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1458 = tensor.collapse_shape %1447 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1459 = tensor.expand_shape %1458 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1460 = func.call @aten_clamp__default(%1457) {prov.region_id = "aten_clamp__default_7", prov.dispatch_id = "aten_clamp__default_7"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1461 = tensor.empty() : tensor<1x32x1xf32>
    %1462 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1460 : tensor<1x32x1xf32>) outs(%1461 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_14", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} {
    ^bb175(%1463: f32, %1464: f32):
      %1465 = arith.constant 1.000000e+00 : f32
      %1466 = arith.divf %1465, %1463 : f32
      linalg.yield %1466 : f32
    } -> tensor<1x32x1xf32>
    %1467 = arith.constant {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} 1.270000e+02 : f32
    %1468 = tensor.splat %1467 {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} : tensor<1x32x1xf32>
    %1469 = tensor.empty() : tensor<1x32x1xf32>
    %1470 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1462, %1468 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1469 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} {
    ^bb176(%1471: f32, %1472: f32, %1473: f32):
      %1474 = arith.mulf %1471, %1472 : f32
      linalg.yield %1474 : f32
    } -> tensor<1x32x1xf32>
    %1475 = tensor.empty() : tensor<1x32x256xf32>
    %1476 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1432, %1470 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1475 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_44", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} {
    ^bb177(%1477: f32, %1478: f32, %1479: f32):
      %1480 = arith.mulf %1477, %1478 : f32
      linalg.yield %1480 : f32
    } -> tensor<1x32x256xf32>
    %1481 = tensor.empty() : tensor<1x32x256xf32>
    %1482 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1476 : tensor<1x32x256xf32>) outs(%1481 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_14", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} {
    ^bb178(%1483: f32, %1484: f32):
      %1485 = math.roundeven %1483 : f32
      linalg.yield %1485 : f32
    } -> tensor<1x32x256xf32>
    %1486 = tensor.empty() : tensor<1x32x256xf32>
    %1487 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1482 : tensor<1x32x256xf32>) outs(%1486 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_15", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} {
    ^bb179(%1488: f32, %1489: f32):
      %1490 = arith.constant -1.280000e+02 : f32
      %1491 = arith.maximumf %1488, %1490 : f32
      %1492 = arith.constant 1.270000e+02 : f32
      %1493 = arith.minimumf %1491, %1492 : f32
      linalg.yield %1493 : f32
    } -> tensor<1x32x256xf32>
    %1494 = tensor.empty() : tensor<1x32x256xf32>
    %1495 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1487, %1470 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1494 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_15", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} {
    ^bb180(%1496: f32, %1497: f32, %1498: f32):
      %1499 = arith.divf %1496, %1497 : f32
      linalg.yield %1499 : f32
    } -> tensor<1x32x256xf32>
    %1500 = tensor.empty() : tensor<256x256xf32>
    %1501 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<256x256xf32>) outs(%1500 : tensor<256x256xf32>) attrs =  {prov.region_id = "abs_15", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} {
    ^bb181(%1502: f32, %1503: f32):
      %1504 = math.absf %1502 : f32
      linalg.yield %1504 : f32
    } -> tensor<256x256xf32>
    %1505 = arith.constant {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} 0.000000e+00 : f32
    %1506 = tensor.splat %1505 {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} : tensor<f32>
    %1507 = linalg.reduce ins(%1501:tensor<256x256xf32>) outs(%1506:tensor<f32>) dimensions = [0, 1]
    (%1508: f32, %1509: f32) {
      %1510 = arith.addf %1508, %1509 : f32
      linalg.yield %1510 : f32
    }
    %1511 = arith.constant {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} 6.553600e+04 : f32
    %1512 = tensor.splat %1511 {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} : tensor<f32>
    %1513 = tensor.empty() : tensor<f32>
    %1514 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1507, %1512 : tensor<f32>, tensor<f32>) outs(%1513 : tensor<f32>) attrs =  {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} {
    ^bb182(%1515: f32, %1516: f32, %1517: f32):
      %1518 = arith.divf %1515, %1516 : f32
      linalg.yield %1518 : f32
    } -> tensor<f32>
    %1519 = func.call @aten_clamp__default_1(%1514) {prov.region_id = "aten_clamp__default_1_7", prov.dispatch_id = "aten_clamp__default_1_7"} : (tensor<f32>) -> tensor<f32>
    %1520 = tensor.empty() : tensor<f32>
    %1521 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1519 : tensor<f32>) outs(%1520 : tensor<f32>) attrs =  {prov.region_id = "elementwise_15", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} {
    ^bb183(%1522: f32, %1523: f32):
      %1524 = arith.constant 1.000000e+00 : f32
      %1525 = arith.divf %1524, %1522 : f32
      linalg.yield %1525 : f32
    } -> tensor<f32>
    %1526 = arith.constant {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} 1.000000e+00 : f32
    %1527 = tensor.splat %1526 {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} : tensor<f32>
    %1528 = tensor.empty() : tensor<f32>
    %1529 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1521, %1527 : tensor<f32>, tensor<f32>) outs(%1528 : tensor<f32>) attrs =  {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} {
    ^bb184(%1530: f32, %1531: f32, %1532: f32):
      %1533 = arith.mulf %1530, %1531 : f32
      linalg.yield %1533 : f32
    } -> tensor<f32>
    %1534 = tensor.empty() : tensor<256x256xf32>
    %1535 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%11, %1529 : tensor<256x256xf32>, tensor<f32>) outs(%1534 : tensor<256x256xf32>) attrs =  {prov.region_id = "mul_46", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} {
    ^bb185(%1536: f32, %1537: f32, %1538: f32):
      %1539 = arith.mulf %1536, %1537 : f32
      linalg.yield %1539 : f32
    } -> tensor<256x256xf32>
    %1540 = tensor.empty() : tensor<256x256xf32>
    %1541 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1535 : tensor<256x256xf32>) outs(%1540 : tensor<256x256xf32>) attrs =  {prov.region_id = "round_15", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} {
    ^bb186(%1542: f32, %1543: f32):
      %1544 = math.roundeven %1542 : f32
      linalg.yield %1544 : f32
    } -> tensor<256x256xf32>
    %1545 = tensor.empty() : tensor<256x256xf32>
    %1546 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1541 : tensor<256x256xf32>) outs(%1545 : tensor<256x256xf32>) attrs =  {prov.region_id = "minmax_16", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} {
    ^bb187(%1547: f32, %1548: f32):
      %1549 = arith.constant -1.000000e+00 : f32
      %1550 = arith.maximumf %1547, %1549 : f32
      %1551 = arith.constant 1.000000e+00 : f32
      %1552 = arith.minimumf %1550, %1551 : f32
      linalg.yield %1552 : f32
    } -> tensor<256x256xf32>
    %1553 = tensor.empty() : tensor<256x256xf32>
    %1554 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1546, %1529 : tensor<256x256xf32>, tensor<f32>) outs(%1553 : tensor<256x256xf32>) attrs =  {prov.region_id = "div_16", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} {
    ^bb188(%1555: f32, %1556: f32, %1557: f32):
      %1558 = arith.divf %1555, %1556 : f32
      linalg.yield %1558 : f32
    } -> tensor<256x256xf32>
    %1559 = tensor.empty() : tensor<256x256xf32>
    %1560 = linalg.transpose ins(%1554:tensor<256x256xf32>) outs(%1559:tensor<256x256xf32>) permutation = [1, 0]
    %1561 = tensor.empty() : tensor<1x32x256xf32>
    %1562 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %1563 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%1562 : f32) outs(%1561 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %1564 = linalg.matmul {prov.region_id = "matmul_9", prov.dispatch_id = "matmul_9", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.q_proj"} ins(%1495, %1560 : tensor<1x32x256xf32>, tensor<256x256xf32>) outs(%1563 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %1565 = tensor.collapse_shape %1564 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %1566 = tensor.expand_shape %1565 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 32] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x32x8x32xf32>
    %1567 = tensor.empty() : tensor<1x8x32x32xf32>
    %1568 = linalg.transpose ins(%1566:tensor<1x32x8x32xf32>) outs(%1567:tensor<1x8x32x32xf32>) permutation = [0, 2, 1, 3]
    %1569 = tensor.empty() : tensor<1x32x256xf32>
    %1570 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1432 : tensor<1x32x256xf32>) outs(%1569 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_16", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} {
    ^bb189(%1571: f32, %1572: f32):
      %1573 = math.absf %1571 : f32
      linalg.yield %1573 : f32
    } -> tensor<1x32x256xf32>
    %1574 = arith.constant {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} 0xff800000 : f32
    %1575 = arith.constant {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} 0 : i64
    %1576 = tensor.splat %1574 {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} : tensor<1x32xf32>
    %1577 = tensor.splat %1575 {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} : tensor<1x32xi64>
    %1578, %1579 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1570 : tensor<1x32x256xf32>) outs(%1576, %1577 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} {
    ^bb190(%1580: f32, %1581: f32, %1582: i64):
      %1583 = linalg.index 2 : index
      %1584 = arith.index_cast %1583 : index to i64
      %1585 = arith.cmpf ogt, %1580, %1581 : f32
      %1586 = arith.select %1585, %1580, %1581 : f32
      %1587 = arith.select %1585, %1584, %1582 : i64
      linalg.yield %1586, %1587 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1588 = tensor.collapse_shape %1578 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1589 = tensor.expand_shape %1588 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1590 = tensor.collapse_shape %1579 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1591 = tensor.expand_shape %1590 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1592 = func.call @aten_clamp__default(%1589) {prov.region_id = "aten_clamp__default_8", prov.dispatch_id = "aten_clamp__default_8"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1593 = tensor.empty() : tensor<1x32x1xf32>
    %1594 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1592 : tensor<1x32x1xf32>) outs(%1593 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_16", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} {
    ^bb191(%1595: f32, %1596: f32):
      %1597 = arith.constant 1.000000e+00 : f32
      %1598 = arith.divf %1597, %1595 : f32
      linalg.yield %1598 : f32
    } -> tensor<1x32x1xf32>
    %1599 = arith.constant {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} 1.270000e+02 : f32
    %1600 = tensor.splat %1599 {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} : tensor<1x32x1xf32>
    %1601 = tensor.empty() : tensor<1x32x1xf32>
    %1602 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1594, %1600 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1601 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} {
    ^bb192(%1603: f32, %1604: f32, %1605: f32):
      %1606 = arith.mulf %1603, %1604 : f32
      linalg.yield %1606 : f32
    } -> tensor<1x32x1xf32>
    %1607 = tensor.empty() : tensor<1x32x256xf32>
    %1608 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1432, %1602 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1607 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_48", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} {
    ^bb193(%1609: f32, %1610: f32, %1611: f32):
      %1612 = arith.mulf %1609, %1610 : f32
      linalg.yield %1612 : f32
    } -> tensor<1x32x256xf32>
    %1613 = tensor.empty() : tensor<1x32x256xf32>
    %1614 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1608 : tensor<1x32x256xf32>) outs(%1613 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_16", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} {
    ^bb194(%1615: f32, %1616: f32):
      %1617 = math.roundeven %1615 : f32
      linalg.yield %1617 : f32
    } -> tensor<1x32x256xf32>
    %1618 = tensor.empty() : tensor<1x32x256xf32>
    %1619 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1614 : tensor<1x32x256xf32>) outs(%1618 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_17", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} {
    ^bb195(%1620: f32, %1621: f32):
      %1622 = arith.constant -1.280000e+02 : f32
      %1623 = arith.maximumf %1620, %1622 : f32
      %1624 = arith.constant 1.270000e+02 : f32
      %1625 = arith.minimumf %1623, %1624 : f32
      linalg.yield %1625 : f32
    } -> tensor<1x32x256xf32>
    %1626 = tensor.empty() : tensor<1x32x256xf32>
    %1627 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1619, %1602 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1626 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_17", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} {
    ^bb196(%1628: f32, %1629: f32, %1630: f32):
      %1631 = arith.divf %1628, %1629 : f32
      linalg.yield %1631 : f32
    } -> tensor<1x32x256xf32>
    %1632 = tensor.empty() : tensor<128x256xf32>
    %1633 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<128x256xf32>) outs(%1632 : tensor<128x256xf32>) attrs =  {prov.region_id = "abs_17", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} {
    ^bb197(%1634: f32, %1635: f32):
      %1636 = math.absf %1634 : f32
      linalg.yield %1636 : f32
    } -> tensor<128x256xf32>
    %1637 = arith.constant {prov.region_id = "reduce_13", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} 0.000000e+00 : f32
    %1638 = tensor.splat %1637 {prov.region_id = "reduce_13", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} : tensor<f32>
    %1639 = linalg.reduce ins(%1633:tensor<128x256xf32>) outs(%1638:tensor<f32>) dimensions = [0, 1]
    (%1640: f32, %1641: f32) {
      %1642 = arith.addf %1640, %1641 : f32
      linalg.yield %1642 : f32
    }
    %1643 = arith.constant {prov.region_id = "reduce_13", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} 3.276800e+04 : f32
    %1644 = tensor.splat %1643 {prov.region_id = "reduce_13", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} : tensor<f32>
    %1645 = tensor.empty() : tensor<f32>
    %1646 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1639, %1644 : tensor<f32>, tensor<f32>) outs(%1645 : tensor<f32>) attrs =  {prov.region_id = "reduce_13", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} {
    ^bb198(%1647: f32, %1648: f32, %1649: f32):
      %1650 = arith.divf %1647, %1648 : f32
      linalg.yield %1650 : f32
    } -> tensor<f32>
    %1651 = func.call @aten_clamp__default_1(%1646) {prov.region_id = "aten_clamp__default_1_8", prov.dispatch_id = "aten_clamp__default_1_8"} : (tensor<f32>) -> tensor<f32>
    %1652 = tensor.empty() : tensor<f32>
    %1653 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1651 : tensor<f32>) outs(%1652 : tensor<f32>) attrs =  {prov.region_id = "elementwise_17", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} {
    ^bb199(%1654: f32, %1655: f32):
      %1656 = arith.constant 1.000000e+00 : f32
      %1657 = arith.divf %1656, %1654 : f32
      linalg.yield %1657 : f32
    } -> tensor<f32>
    %1658 = arith.constant {prov.region_id = "mul_49", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} 1.000000e+00 : f32
    %1659 = tensor.splat %1658 {prov.region_id = "mul_49", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} : tensor<f32>
    %1660 = tensor.empty() : tensor<f32>
    %1661 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1653, %1659 : tensor<f32>, tensor<f32>) outs(%1660 : tensor<f32>) attrs =  {prov.region_id = "mul_49", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} {
    ^bb200(%1662: f32, %1663: f32, %1664: f32):
      %1665 = arith.mulf %1662, %1663 : f32
      linalg.yield %1665 : f32
    } -> tensor<f32>
    %1666 = tensor.empty() : tensor<128x256xf32>
    %1667 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12, %1661 : tensor<128x256xf32>, tensor<f32>) outs(%1666 : tensor<128x256xf32>) attrs =  {prov.region_id = "mul_50", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} {
    ^bb201(%1668: f32, %1669: f32, %1670: f32):
      %1671 = arith.mulf %1668, %1669 : f32
      linalg.yield %1671 : f32
    } -> tensor<128x256xf32>
    %1672 = tensor.empty() : tensor<128x256xf32>
    %1673 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1667 : tensor<128x256xf32>) outs(%1672 : tensor<128x256xf32>) attrs =  {prov.region_id = "round_17", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} {
    ^bb202(%1674: f32, %1675: f32):
      %1676 = math.roundeven %1674 : f32
      linalg.yield %1676 : f32
    } -> tensor<128x256xf32>
    %1677 = tensor.empty() : tensor<128x256xf32>
    %1678 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1673 : tensor<128x256xf32>) outs(%1677 : tensor<128x256xf32>) attrs =  {prov.region_id = "minmax_18", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} {
    ^bb203(%1679: f32, %1680: f32):
      %1681 = arith.constant -1.000000e+00 : f32
      %1682 = arith.maximumf %1679, %1681 : f32
      %1683 = arith.constant 1.000000e+00 : f32
      %1684 = arith.minimumf %1682, %1683 : f32
      linalg.yield %1684 : f32
    } -> tensor<128x256xf32>
    %1685 = tensor.empty() : tensor<128x256xf32>
    %1686 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1678, %1661 : tensor<128x256xf32>, tensor<f32>) outs(%1685 : tensor<128x256xf32>) attrs =  {prov.region_id = "div_18", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} {
    ^bb204(%1687: f32, %1688: f32, %1689: f32):
      %1690 = arith.divf %1687, %1688 : f32
      linalg.yield %1690 : f32
    } -> tensor<128x256xf32>
    %1691 = tensor.empty() : tensor<256x128xf32>
    %1692 = linalg.transpose ins(%1686:tensor<128x256xf32>) outs(%1691:tensor<256x128xf32>) permutation = [1, 0]
    %1693 = tensor.empty() : tensor<1x32x128xf32>
    %1694 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %1695 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%1694 : f32) outs(%1693 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %1696 = linalg.matmul {prov.region_id = "matmul_10", prov.dispatch_id = "matmul_10", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.k_proj"} ins(%1627, %1692 : tensor<1x32x256xf32>, tensor<256x128xf32>) outs(%1695 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %1697 = tensor.collapse_shape %1696 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %1698 = tensor.expand_shape %1697 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 32] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x32x4x32xf32>
    %1699 = tensor.empty() : tensor<1x4x32x32xf32>
    %1700 = linalg.transpose ins(%1698:tensor<1x32x4x32xf32>) outs(%1699:tensor<1x4x32x32xf32>) permutation = [0, 2, 1, 3]
    %1701 = tensor.empty() : tensor<1x32x256xf32>
    %1702 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1432 : tensor<1x32x256xf32>) outs(%1701 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_18", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} {
    ^bb205(%1703: f32, %1704: f32):
      %1705 = math.absf %1703 : f32
      linalg.yield %1705 : f32
    } -> tensor<1x32x256xf32>
    %1706 = arith.constant {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} 0xff800000 : f32
    %1707 = arith.constant {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} 0 : i64
    %1708 = tensor.splat %1706 {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} : tensor<1x32xf32>
    %1709 = tensor.splat %1707 {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} : tensor<1x32xi64>
    %1710, %1711 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1702 : tensor<1x32x256xf32>) outs(%1708, %1709 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} {
    ^bb206(%1712: f32, %1713: f32, %1714: i64):
      %1715 = linalg.index 2 : index
      %1716 = arith.index_cast %1715 : index to i64
      %1717 = arith.cmpf ogt, %1712, %1713 : f32
      %1718 = arith.select %1717, %1712, %1713 : f32
      %1719 = arith.select %1717, %1716, %1714 : i64
      linalg.yield %1718, %1719 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1720 = tensor.collapse_shape %1710 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1721 = tensor.expand_shape %1720 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1722 = tensor.collapse_shape %1711 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1723 = tensor.expand_shape %1722 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1724 = func.call @aten_clamp__default(%1721) {prov.region_id = "aten_clamp__default_9", prov.dispatch_id = "aten_clamp__default_9"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1725 = tensor.empty() : tensor<1x32x1xf32>
    %1726 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1724 : tensor<1x32x1xf32>) outs(%1725 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_18", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} {
    ^bb207(%1727: f32, %1728: f32):
      %1729 = arith.constant 1.000000e+00 : f32
      %1730 = arith.divf %1729, %1727 : f32
      linalg.yield %1730 : f32
    } -> tensor<1x32x1xf32>
    %1731 = arith.constant {prov.region_id = "mul_51", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} 1.270000e+02 : f32
    %1732 = tensor.splat %1731 {prov.region_id = "mul_51", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} : tensor<1x32x1xf32>
    %1733 = tensor.empty() : tensor<1x32x1xf32>
    %1734 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1726, %1732 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1733 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_51", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} {
    ^bb208(%1735: f32, %1736: f32, %1737: f32):
      %1738 = arith.mulf %1735, %1736 : f32
      linalg.yield %1738 : f32
    } -> tensor<1x32x1xf32>
    %1739 = tensor.empty() : tensor<1x32x256xf32>
    %1740 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1432, %1734 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1739 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_52", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} {
    ^bb209(%1741: f32, %1742: f32, %1743: f32):
      %1744 = arith.mulf %1741, %1742 : f32
      linalg.yield %1744 : f32
    } -> tensor<1x32x256xf32>
    %1745 = tensor.empty() : tensor<1x32x256xf32>
    %1746 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1740 : tensor<1x32x256xf32>) outs(%1745 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_18", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} {
    ^bb210(%1747: f32, %1748: f32):
      %1749 = math.roundeven %1747 : f32
      linalg.yield %1749 : f32
    } -> tensor<1x32x256xf32>
    %1750 = tensor.empty() : tensor<1x32x256xf32>
    %1751 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1746 : tensor<1x32x256xf32>) outs(%1750 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_19", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} {
    ^bb211(%1752: f32, %1753: f32):
      %1754 = arith.constant -1.280000e+02 : f32
      %1755 = arith.maximumf %1752, %1754 : f32
      %1756 = arith.constant 1.270000e+02 : f32
      %1757 = arith.minimumf %1755, %1756 : f32
      linalg.yield %1757 : f32
    } -> tensor<1x32x256xf32>
    %1758 = tensor.empty() : tensor<1x32x256xf32>
    %1759 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1751, %1734 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1758 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_19", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} {
    ^bb212(%1760: f32, %1761: f32, %1762: f32):
      %1763 = arith.divf %1760, %1761 : f32
      linalg.yield %1763 : f32
    } -> tensor<1x32x256xf32>
    %1764 = tensor.empty() : tensor<128x256xf32>
    %1765 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<128x256xf32>) outs(%1764 : tensor<128x256xf32>) attrs =  {prov.region_id = "abs_19", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} {
    ^bb213(%1766: f32, %1767: f32):
      %1768 = math.absf %1766 : f32
      linalg.yield %1768 : f32
    } -> tensor<128x256xf32>
    %1769 = arith.constant {prov.region_id = "reduce_14", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} 0.000000e+00 : f32
    %1770 = tensor.splat %1769 {prov.region_id = "reduce_14", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} : tensor<f32>
    %1771 = linalg.reduce ins(%1765:tensor<128x256xf32>) outs(%1770:tensor<f32>) dimensions = [0, 1]
    (%1772: f32, %1773: f32) {
      %1774 = arith.addf %1772, %1773 : f32
      linalg.yield %1774 : f32
    }
    %1775 = arith.constant {prov.region_id = "reduce_14", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} 3.276800e+04 : f32
    %1776 = tensor.splat %1775 {prov.region_id = "reduce_14", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} : tensor<f32>
    %1777 = tensor.empty() : tensor<f32>
    %1778 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1771, %1776 : tensor<f32>, tensor<f32>) outs(%1777 : tensor<f32>) attrs =  {prov.region_id = "reduce_14", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} {
    ^bb214(%1779: f32, %1780: f32, %1781: f32):
      %1782 = arith.divf %1779, %1780 : f32
      linalg.yield %1782 : f32
    } -> tensor<f32>
    %1783 = func.call @aten_clamp__default_1(%1778) {prov.region_id = "aten_clamp__default_1_9", prov.dispatch_id = "aten_clamp__default_1_9"} : (tensor<f32>) -> tensor<f32>
    %1784 = tensor.empty() : tensor<f32>
    %1785 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1783 : tensor<f32>) outs(%1784 : tensor<f32>) attrs =  {prov.region_id = "elementwise_19", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} {
    ^bb215(%1786: f32, %1787: f32):
      %1788 = arith.constant 1.000000e+00 : f32
      %1789 = arith.divf %1788, %1786 : f32
      linalg.yield %1789 : f32
    } -> tensor<f32>
    %1790 = arith.constant {prov.region_id = "mul_53", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} 1.000000e+00 : f32
    %1791 = tensor.splat %1790 {prov.region_id = "mul_53", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} : tensor<f32>
    %1792 = tensor.empty() : tensor<f32>
    %1793 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1785, %1791 : tensor<f32>, tensor<f32>) outs(%1792 : tensor<f32>) attrs =  {prov.region_id = "mul_53", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} {
    ^bb216(%1794: f32, %1795: f32, %1796: f32):
      %1797 = arith.mulf %1794, %1795 : f32
      linalg.yield %1797 : f32
    } -> tensor<f32>
    %1798 = tensor.empty() : tensor<128x256xf32>
    %1799 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%13, %1793 : tensor<128x256xf32>, tensor<f32>) outs(%1798 : tensor<128x256xf32>) attrs =  {prov.region_id = "mul_54", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} {
    ^bb217(%1800: f32, %1801: f32, %1802: f32):
      %1803 = arith.mulf %1800, %1801 : f32
      linalg.yield %1803 : f32
    } -> tensor<128x256xf32>
    %1804 = tensor.empty() : tensor<128x256xf32>
    %1805 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1799 : tensor<128x256xf32>) outs(%1804 : tensor<128x256xf32>) attrs =  {prov.region_id = "round_19", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} {
    ^bb218(%1806: f32, %1807: f32):
      %1808 = math.roundeven %1806 : f32
      linalg.yield %1808 : f32
    } -> tensor<128x256xf32>
    %1809 = tensor.empty() : tensor<128x256xf32>
    %1810 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1805 : tensor<128x256xf32>) outs(%1809 : tensor<128x256xf32>) attrs =  {prov.region_id = "minmax_20", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} {
    ^bb219(%1811: f32, %1812: f32):
      %1813 = arith.constant -1.000000e+00 : f32
      %1814 = arith.maximumf %1811, %1813 : f32
      %1815 = arith.constant 1.000000e+00 : f32
      %1816 = arith.minimumf %1814, %1815 : f32
      linalg.yield %1816 : f32
    } -> tensor<128x256xf32>
    %1817 = tensor.empty() : tensor<128x256xf32>
    %1818 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1810, %1793 : tensor<128x256xf32>, tensor<f32>) outs(%1817 : tensor<128x256xf32>) attrs =  {prov.region_id = "div_20", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} {
    ^bb220(%1819: f32, %1820: f32, %1821: f32):
      %1822 = arith.divf %1819, %1820 : f32
      linalg.yield %1822 : f32
    } -> tensor<128x256xf32>
    %1823 = tensor.empty() : tensor<256x128xf32>
    %1824 = linalg.transpose ins(%1818:tensor<128x256xf32>) outs(%1823:tensor<256x128xf32>) permutation = [1, 0]
    %1825 = tensor.empty() : tensor<1x32x128xf32>
    %1826 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %1827 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%1826 : f32) outs(%1825 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %1828 = linalg.matmul {prov.region_id = "matmul_11", prov.dispatch_id = "matmul_11", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.v_proj"} ins(%1759, %1824 : tensor<1x32x256xf32>, tensor<256x128xf32>) outs(%1827 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %1829 = tensor.collapse_shape %1828 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %1830 = tensor.expand_shape %1829 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 32] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x32x4x32xf32>
    %1831 = tensor.empty() : tensor<1x4x32x32xf32>
    %1832 = linalg.transpose ins(%1830:tensor<1x32x4x32xf32>) outs(%1831:tensor<1x4x32x32xf32>) permutation = [0, 2, 1, 3]
    %1833 = "tensor.extract_slice"(%29) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 39, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_14", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.rotary_emb"} : (tensor<2048x32xf32>) -> tensor<39x32xf32>
    %1834 = "tensor.extract_slice"(%30) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 39, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_15", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.rotary_emb"} : (tensor<2048x32xf32>) -> tensor<39x32xf32>
    %1835 = tensor.empty() : tensor<1x32x32xf32>
    %1836 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63 : tensor<1x32xi64>) outs(%1835 : tensor<1x32x32xf32>) attrs =  {prov.region_id = "gather_2", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32"} {
    ^bb221(%1837: i64, %1838: f32):
      %1839 = arith.index_cast %1837 : i64 to index
      %1840 = linalg.index 2 : index
      %1841 = tensor.extract %1833[%1839, %1840] : tensor<39x32xf32>
      linalg.yield %1841 : f32
    } -> tensor<1x32x32xf32>
    %1842 = tensor.collapse_shape %1836 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %1843 = tensor.expand_shape %1842 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %1844 = tensor.empty() : tensor<1x32x32xf32>
    %1845 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63 : tensor<1x32xi64>) outs(%1844 : tensor<1x32x32xf32>) attrs =  {prov.region_id = "gather_3", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32"} {
    ^bb222(%1846: i64, %1847: f32):
      %1848 = arith.index_cast %1846 : i64 to index
      %1849 = linalg.index 2 : index
      %1850 = tensor.extract %1834[%1848, %1849] : tensor<39x32xf32>
      linalg.yield %1850 : f32
    } -> tensor<1x32x32xf32>
    %1851 = tensor.collapse_shape %1845 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %1852 = tensor.expand_shape %1851 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %1853 = tensor.empty() : tensor<1x8x32x32xf32>
    %1854 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1568, %1843 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%1853 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "mul_55", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb223(%1855: f32, %1856: f32, %1857: f32):
      %1858 = arith.mulf %1855, %1856 : f32
      linalg.yield %1858 : f32
    } -> tensor<1x8x32x32xf32>
    %1859 = "tensor.extract_slice"(%1568) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_16", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x32x32xf32>) -> tensor<1x8x32x16xf32>
    %1860 = "tensor.extract_slice"(%1568) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 8, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_17", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x32x32xf32>) -> tensor<1x8x32x16xf32>
    %1861 = tensor.empty() : tensor<1x8x32x16xf32>
    %1862 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1860 : tensor<1x8x32x16xf32>) outs(%1861 : tensor<1x8x32x16xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb224(%1863: f32, %1864: f32):
      %1865 = arith.negf %1863 : f32
      linalg.yield %1865 : f32
    } -> tensor<1x8x32x16xf32>
    %1866 = tensor.concat dim(3) %1862, %1859 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x8x32x16xf32>, tensor<1x8x32x16xf32>) -> tensor<1x8x32x32xf32>
    %1867 = tensor.empty() : tensor<1x8x32x32xf32>
    %1868 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1866, %1852 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%1867 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "mul_56", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb225(%1869: f32, %1870: f32, %1871: f32):
      %1872 = arith.mulf %1869, %1870 : f32
      linalg.yield %1872 : f32
    } -> tensor<1x8x32x32xf32>
    %1873 = tensor.empty() : tensor<1x8x32x32xf32>
    %1874 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1854, %1868 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%1873 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb226(%1875: f32, %1876: f32, %1877: f32):
      %1878 = arith.addf %1875, %1876 : f32
      linalg.yield %1878 : f32
    } -> tensor<1x8x32x32xf32>
    %1879 = tensor.empty() : tensor<1x4x32x32xf32>
    %1880 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1700, %1843 : tensor<1x4x32x32xf32>, tensor<1x1x32x32xf32>) outs(%1879 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "mul_57", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb227(%1881: f32, %1882: f32, %1883: f32):
      %1884 = arith.mulf %1881, %1882 : f32
      linalg.yield %1884 : f32
    } -> tensor<1x4x32x32xf32>
    %1885 = "tensor.extract_slice"(%1700) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_18", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x16xf32>
    %1886 = "tensor.extract_slice"(%1700) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_19", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x16xf32>
    %1887 = tensor.empty() : tensor<1x4x32x16xf32>
    %1888 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1886 : tensor<1x4x32x16xf32>) outs(%1887 : tensor<1x4x32x16xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb228(%1889: f32, %1890: f32):
      %1891 = arith.negf %1889 : f32
      linalg.yield %1891 : f32
    } -> tensor<1x4x32x16xf32>
    %1892 = tensor.concat dim(3) %1888, %1885 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x32x16xf32>, tensor<1x4x32x16xf32>) -> tensor<1x4x32x32xf32>
    %1893 = tensor.empty() : tensor<1x4x32x32xf32>
    %1894 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1892, %1852 : tensor<1x4x32x32xf32>, tensor<1x1x32x32xf32>) outs(%1893 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "mul_58", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb229(%1895: f32, %1896: f32, %1897: f32):
      %1898 = arith.mulf %1895, %1896 : f32
      linalg.yield %1898 : f32
    } -> tensor<1x4x32x32xf32>
    %1899 = tensor.empty() : tensor<1x4x32x32xf32>
    %1900 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1880, %1894 : tensor<1x4x32x32xf32>, tensor<1x4x32x32xf32>) outs(%1899 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb230(%1901: f32, %1902: f32, %1903: f32):
      %1904 = arith.addf %1901, %1902 : f32
      linalg.yield %1904 : f32
    } -> tensor<1x4x32x32xf32>
    %1905 = "tensor.extract_slice"(%33) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x39x32xf32>) -> tensor<4x39x32xf32>
    %1906 = tensor.empty() : tensor<32xi64>
    %1907 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%62, %45 : tensor<i64>, tensor<32xi64>) outs(%1906 : tensor<32xi64>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb231(%1908: i64, %1909: i64, %1910: i64):
      %1911 = arith.addi %1908, %1909 : i64
      linalg.yield %1911 : i64
    } -> tensor<32xi64>
    %1912 = func.call @aten_index_copy_default(%1905, %1907, %1900) {prov.region_id = "aten_index_copy_default_2", prov.dispatch_id = "aten_index_copy_default_2"} : (tensor<4x39x32xf32>, tensor<32xi64>, tensor<1x4x32x32xf32>) -> tensor<1x4x39x32xf32>
    %1913 = "tensor.extract_slice"(%34) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x39x32xf32>) -> tensor<4x39x32xf32>
    %1914 = tensor.empty() : tensor<32xi64>
    %1915 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%62, %45 : tensor<i64>, tensor<32xi64>) outs(%1914 : tensor<32xi64>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb232(%1916: i64, %1917: i64, %1918: i64):
      %1919 = arith.addi %1916, %1917 : i64
      linalg.yield %1919 : i64
    } -> tensor<32xi64>
    %1920 = func.call @aten_index_copy_default(%1913, %1915, %1832) {prov.region_id = "aten_index_copy_default_3", prov.dispatch_id = "aten_index_copy_default_3"} : (tensor<4x39x32xf32>, tensor<32xi64>, tensor<1x4x32x32xf32>) -> tensor<1x4x39x32xf32>
    %1921 = "tensor.extract_slice"(%1912) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_20", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x39x32xf32>) -> tensor<1x4x39x32xf32>
    %1922 = "tensor.extract_slice"(%1921) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_21", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x39x32xf32>) -> tensor<1x4x39x32xf32>
    %1923 = tensor.collapse_shape %1922 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x39x32xf32> into tensor<4992xf32>
    %1924 = tensor.expand_shape %1923 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 39, 32] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<4992xf32> into tensor<1x4x1x39x32xf32>
    %1925 = "tensor.extract_slice"(%1924) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_22", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x39x32xf32>) -> tensor<1x4x1x39x32xf32>
    %1926 = "tensor.extract_slice"(%1925) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_23", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x39x32xf32>) -> tensor<1x4x1x39x32xf32>
    %1927 = tensor.empty() : tensor<1x4x2x39x32xf32>
    %1928 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1926 : tensor<1x4x1x39x32xf32>) outs(%1927 : tensor<1x4x2x39x32xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb233(%1929: f32, %1930: f32):
      linalg.yield %1929 : f32
    } -> tensor<1x4x2x39x32xf32>
    %1931 = tensor.collapse_shape %1928 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x4x2x39x32xf32> into tensor<9984xf32>
    %1932 = tensor.expand_shape %1931 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 39, 32] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<9984xf32> into tensor<1x8x39x32xf32>
    %1933 = "tensor.extract_slice"(%1920) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_24", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x39x32xf32>) -> tensor<1x4x39x32xf32>
    %1934 = "tensor.extract_slice"(%1933) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_25", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x39x32xf32>) -> tensor<1x4x39x32xf32>
    %1935 = tensor.collapse_shape %1934 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x39x32xf32> into tensor<4992xf32>
    %1936 = tensor.expand_shape %1935 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 39, 32] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<4992xf32> into tensor<1x4x1x39x32xf32>
    %1937 = "tensor.extract_slice"(%1936) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_26", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x39x32xf32>) -> tensor<1x4x1x39x32xf32>
    %1938 = "tensor.extract_slice"(%1937) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_27", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x39x32xf32>) -> tensor<1x4x1x39x32xf32>
    %1939 = tensor.empty() : tensor<1x4x2x39x32xf32>
    %1940 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1938 : tensor<1x4x1x39x32xf32>) outs(%1939 : tensor<1x4x2x39x32xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb234(%1941: f32, %1942: f32):
      linalg.yield %1941 : f32
    } -> tensor<1x4x2x39x32xf32>
    %1943 = tensor.collapse_shape %1940 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x4x2x39x32xf32> into tensor<9984xf32>
    %1944 = tensor.expand_shape %1943 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 39, 32] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<9984xf32> into tensor<1x8x39x32xf32>
    %1945 = tensor.empty() : tensor<1x8x32x39xf32>
    %1946 = linalg.transpose ins(%1932:tensor<1x8x39x32xf32>) outs(%1945:tensor<1x8x32x39xf32>) permutation = [0, 1, 3, 2]
    %1947 = arith.constant {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %1948 = tensor.splat %1947 {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x32x39xf32>
    %1949 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1874, %1946 : tensor<1x8x32x32xf32>, tensor<1x8x32x39xf32>) outs(%1948 : tensor<1x8x32x39xf32>) attrs =  {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb235(%1950: f32, %1951: f32, %1952: f32):
      %1953 = arith.mulf %1950, %1951 : f32
      %1954 = arith.addf %1952, %1953 : f32
      linalg.yield %1954 : f32
    } -> tensor<1x8x32x39xf32>
    %1955 = arith.constant {prov.region_id = "div_21", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 5.65685415 : f32
    %1956 = tensor.splat %1955 {prov.region_id = "div_21", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x32x39xf32>
    %1957 = tensor.empty() : tensor<1x8x32x39xf32>
    %1958 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1949, %1956 : tensor<1x8x32x39xf32>, tensor<1x8x32x39xf32>) outs(%1957 : tensor<1x8x32x39xf32>) attrs =  {prov.region_id = "div_21", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb236(%1959: f32, %1960: f32, %1961: f32):
      %1962 = arith.divf %1959, %1960 : f32
      linalg.yield %1962 : f32
    } -> tensor<1x8x32x39xf32>
    %1963 = tensor.empty() : tensor<32xi64>
    %1964 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%62, %45 : tensor<i64>, tensor<32xi64>) outs(%1963 : tensor<32xi64>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb237(%1965: i64, %1966: i64, %1967: i64):
      %1968 = arith.addi %1965, %1966 : i64
      linalg.yield %1968 : i64
    } -> tensor<32xi64>
    %1969 = tensor.expand_shape %1964 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<32xi64> into tensor<32x1xi64>
    %1970 = tensor.expand_shape %36 [[0 : i64, 1 : i64]] output_shape [1, 39] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<39xi64> into tensor<1x39xi64>
    %1971 = tensor.empty() : tensor<32x39xi1>
    %1972 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1970, %1969 : tensor<1x39xi64>, tensor<32x1xi64>) outs(%1971 : tensor<32x39xi1>) attrs =  {prov.region_id = "compare_1", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
    ^bb238(%1973: i64, %1974: i64, %1975: i1):
      %1976 = arith.cmpi sle, %1973, %1974 : i64
      linalg.yield %1976 : i1
    } -> tensor<32x39xi1>
    %1977 = tensor.collapse_shape %1972 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<32x39xi1> into tensor<1248xi1>
    %1978 = tensor.expand_shape %1977 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 39] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1248xi1> into tensor<1x32x39xi1>
    %1979 = tensor.collapse_shape %1978 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x32x39xi1> into tensor<1248xi1>
    %1980 = tensor.expand_shape %1979 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 39] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1248xi1> into tensor<1x1x32x39xi1>
    %1981 = tensor.empty() : tensor<1x1x32x39xi1>
    %1982 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1980 : tensor<1x1x32x39xi1>) outs(%1981 : tensor<1x1x32x39xi1>) attrs =  {prov.region_id = "bitwise_1", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
    ^bb239(%1983: i1, %1984: i1):
      %1985 = arith.constant true
      %1986 = arith.xori %1983, %1985 : i1
      linalg.yield %1986 : i1
    } -> tensor<1x1x32x39xi1>
    %1987 = func.call @aten_masked_fill_Scalar(%1958, %1982) {prov.region_id = "aten_masked_fill_Scalar_1", prov.dispatch_id = "aten_masked_fill_Scalar_1"} : (tensor<1x8x32x39xf32>, tensor<1x1x32x39xi1>) -> tensor<1x8x32x39xf32>
    %1988 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0xff800000 : f32
    %1989 = tensor.splat %1988 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x32xf32>
    %1990 = linalg.reduce ins(%1987:tensor<1x8x32x39xf32>) outs(%1989:tensor<1x8x32xf32>) dimensions = [3]
    (%1991: f32, %1992: f32) {
      %1993 = arith.maximumf %1991, %1992 : f32
      linalg.yield %1993 : f32
    }
    %1994 = tensor.collapse_shape %1990 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x32xf32> into tensor<256xf32>
    %1995 = tensor.expand_shape %1994 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %1996 = tensor.empty() : tensor<1x8x32x39xf32>
    %1997 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1987, %1995 : tensor<1x8x32x39xf32>, tensor<1x8x32x1xf32>) outs(%1996 : tensor<1x8x32x39xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb240(%1998: f32, %1999: f32, %2000: f32):
      %2001 = arith.subf %1998, %1999 : f32
      linalg.yield %2001 : f32
    } -> tensor<1x8x32x39xf32>
    %2002 = tensor.empty() : tensor<1x8x32x39xf32>
    %2003 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1997 : tensor<1x8x32x39xf32>) outs(%2002 : tensor<1x8x32x39xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb241(%2004: f32, %2005: f32):
      %2006 = math.exp %2004 : f32
      linalg.yield %2006 : f32
    } -> tensor<1x8x32x39xf32>
    %2007 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %2008 = tensor.splat %2007 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x32xf32>
    %2009 = linalg.reduce ins(%2003:tensor<1x8x32x39xf32>) outs(%2008:tensor<1x8x32xf32>) dimensions = [3]
    (%2010: f32, %2011: f32) {
      %2012 = arith.addf %2010, %2011 : f32
      linalg.yield %2012 : f32
    }
    %2013 = tensor.collapse_shape %2009 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x32xf32> into tensor<256xf32>
    %2014 = tensor.expand_shape %2013 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %2015 = tensor.empty() : tensor<1x8x32x39xf32>
    %2016 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2003, %2014 : tensor<1x8x32x39xf32>, tensor<1x8x32x1xf32>) outs(%2015 : tensor<1x8x32x39xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
    ^bb242(%2017: f32, %2018: f32, %2019: f32):
      %2020 = arith.divf %2017, %2018 : f32
      linalg.yield %2020 : f32
    } -> tensor<1x8x32x39xf32>
    %2021 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %2022 = tensor.splat %2021 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x32x32xf32>
    %2023 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%2016, %1944 : tensor<1x8x32x39xf32>, tensor<1x8x39x32xf32>) outs(%2022 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
    ^bb243(%2024: f32, %2025: f32, %2026: f32):
      %2027 = arith.mulf %2024, %2025 : f32
      %2028 = arith.addf %2026, %2027 : f32
      linalg.yield %2028 : f32
    } -> tensor<1x8x32x32xf32>
    %2029 = tensor.empty() : tensor<1x32x8x32xf32>
    %2030 = linalg.transpose ins(%2023:tensor<1x8x32x32xf32>) outs(%2029:tensor<1x32x8x32xf32>) permutation = [0, 2, 1, 3]
    %2031 = tensor.collapse_shape %2030 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x32x8x32xf32> into tensor<8192xf32>
    %2032 = tensor.expand_shape %2031 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 256] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x32x256xf32>
    %2033 = tensor.empty() : tensor<1x32x256xf32>
    %2034 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2032 : tensor<1x32x256xf32>) outs(%2033 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.attn_sub_norm"} {
    ^bb244(%2035: f32, %2036: f32):
      %2037 = arith.constant 2.000000e+00 : f32
      %2038 = math.powf %2035, %2037 : f32
      linalg.yield %2038 : f32
    } -> tensor<1x32x256xf32>
    %2039 = arith.constant {prov.region_id = "reduce_15", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.attn_sub_norm"} 0.000000e+00 : f32
    %2040 = tensor.splat %2039 {prov.region_id = "reduce_15", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.attn_sub_norm"} : tensor<1x32xf32>
    %2041 = linalg.reduce ins(%2034:tensor<1x32x256xf32>) outs(%2040:tensor<1x32xf32>) dimensions = [2]
    (%2042: f32, %2043: f32) {
      %2044 = arith.addf %2042, %2043 : f32
      linalg.yield %2044 : f32
    }
    %2045 = arith.constant {prov.region_id = "reduce_15", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.attn_sub_norm"} 2.560000e+02 : f32
    %2046 = tensor.splat %2045 {prov.region_id = "reduce_15", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.attn_sub_norm"} : tensor<1x32xf32>
    %2047 = tensor.empty() : tensor<1x32xf32>
    %2048 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2041, %2046 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%2047 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_15", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.attn_sub_norm"} {
    ^bb245(%2049: f32, %2050: f32, %2051: f32):
      %2052 = arith.divf %2049, %2050 : f32
      linalg.yield %2052 : f32
    } -> tensor<1x32xf32>
    %2053 = tensor.collapse_shape %2048 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_15", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.attn_sub_norm"} : tensor<1x32xf32> into tensor<32xf32>
    %2054 = tensor.expand_shape %2053 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_15", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.attn_sub_norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2055 = arith.constant {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.attn_sub_norm"} 1.000000e-05 : f32
    %2056 = tensor.splat %2055 {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.attn_sub_norm"} : tensor<1x32x1xf32>
    %2057 = tensor.empty() : tensor<1x32x1xf32>
    %2058 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2054, %2056 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2057 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.attn_sub_norm"} {
    ^bb246(%2059: f32, %2060: f32, %2061: f32):
      %2062 = arith.addf %2059, %2060 : f32
      linalg.yield %2062 : f32
    } -> tensor<1x32x1xf32>
    %2063 = tensor.empty() : tensor<1x32x1xf32>
    %2064 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2058 : tensor<1x32x1xf32>) outs(%2063 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_5", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.attn_sub_norm"} {
    ^bb247(%2065: f32, %2066: f32):
      %2067 = math.rsqrt %2065 : f32
      linalg.yield %2067 : f32
    } -> tensor<1x32x1xf32>
    %2068 = tensor.empty() : tensor<1x32x256xf32>
    %2069 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2032, %2064 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2068 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_59", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.attn_sub_norm"} {
    ^bb248(%2070: f32, %2071: f32, %2072: f32):
      %2073 = arith.mulf %2070, %2071 : f32
      linalg.yield %2073 : f32
    } -> tensor<1x32x256xf32>
    %2074 = tensor.empty() : tensor<1x32x256xf32>
    %2075 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %2069 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%2074 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_60", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.attn_sub_norm"} {
    ^bb249(%2076: f32, %2077: f32, %2078: f32):
      %2079 = arith.mulf %2076, %2077 : f32
      linalg.yield %2079 : f32
    } -> tensor<1x32x256xf32>
    %2080 = tensor.empty() : tensor<1x32x256xf32>
    %2081 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2075 : tensor<1x32x256xf32>) outs(%2080 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_20", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} {
    ^bb250(%2082: f32, %2083: f32):
      %2084 = math.absf %2082 : f32
      linalg.yield %2084 : f32
    } -> tensor<1x32x256xf32>
    %2085 = arith.constant {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} 0xff800000 : f32
    %2086 = arith.constant {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} 0 : i64
    %2087 = tensor.splat %2085 {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} : tensor<1x32xf32>
    %2088 = tensor.splat %2086 {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} : tensor<1x32xi64>
    %2089, %2090 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2081 : tensor<1x32x256xf32>) outs(%2087, %2088 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} {
    ^bb251(%2091: f32, %2092: f32, %2093: i64):
      %2094 = linalg.index 2 : index
      %2095 = arith.index_cast %2094 : index to i64
      %2096 = arith.cmpf ogt, %2091, %2092 : f32
      %2097 = arith.select %2096, %2091, %2092 : f32
      %2098 = arith.select %2096, %2095, %2093 : i64
      linalg.yield %2097, %2098 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %2099 = tensor.collapse_shape %2089 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %2100 = tensor.expand_shape %2099 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2101 = tensor.collapse_shape %2090 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %2102 = tensor.expand_shape %2101 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %2103 = func.call @aten_clamp__default(%2100) {prov.region_id = "aten_clamp__default_10", prov.dispatch_id = "aten_clamp__default_10"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %2104 = tensor.empty() : tensor<1x32x1xf32>
    %2105 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2103 : tensor<1x32x1xf32>) outs(%2104 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_20", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} {
    ^bb252(%2106: f32, %2107: f32):
      %2108 = arith.constant 1.000000e+00 : f32
      %2109 = arith.divf %2108, %2106 : f32
      linalg.yield %2109 : f32
    } -> tensor<1x32x1xf32>
    %2110 = arith.constant {prov.region_id = "mul_61", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} 1.270000e+02 : f32
    %2111 = tensor.splat %2110 {prov.region_id = "mul_61", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} : tensor<1x32x1xf32>
    %2112 = tensor.empty() : tensor<1x32x1xf32>
    %2113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2105, %2111 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2112 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_61", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} {
    ^bb253(%2114: f32, %2115: f32, %2116: f32):
      %2117 = arith.mulf %2114, %2115 : f32
      linalg.yield %2117 : f32
    } -> tensor<1x32x1xf32>
    %2118 = tensor.empty() : tensor<1x32x256xf32>
    %2119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2075, %2113 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2118 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_62", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} {
    ^bb254(%2120: f32, %2121: f32, %2122: f32):
      %2123 = arith.mulf %2120, %2121 : f32
      linalg.yield %2123 : f32
    } -> tensor<1x32x256xf32>
    %2124 = tensor.empty() : tensor<1x32x256xf32>
    %2125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2119 : tensor<1x32x256xf32>) outs(%2124 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_20", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} {
    ^bb255(%2126: f32, %2127: f32):
      %2128 = math.roundeven %2126 : f32
      linalg.yield %2128 : f32
    } -> tensor<1x32x256xf32>
    %2129 = tensor.empty() : tensor<1x32x256xf32>
    %2130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2125 : tensor<1x32x256xf32>) outs(%2129 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_21", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} {
    ^bb256(%2131: f32, %2132: f32):
      %2133 = arith.constant -1.280000e+02 : f32
      %2134 = arith.maximumf %2131, %2133 : f32
      %2135 = arith.constant 1.270000e+02 : f32
      %2136 = arith.minimumf %2134, %2135 : f32
      linalg.yield %2136 : f32
    } -> tensor<1x32x256xf32>
    %2137 = tensor.empty() : tensor<1x32x256xf32>
    %2138 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2130, %2113 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2137 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_22", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} {
    ^bb257(%2139: f32, %2140: f32, %2141: f32):
      %2142 = arith.divf %2139, %2140 : f32
      linalg.yield %2142 : f32
    } -> tensor<1x32x256xf32>
    %2143 = tensor.empty() : tensor<256x256xf32>
    %2144 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<256x256xf32>) outs(%2143 : tensor<256x256xf32>) attrs =  {prov.region_id = "abs_21", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} {
    ^bb258(%2145: f32, %2146: f32):
      %2147 = math.absf %2145 : f32
      linalg.yield %2147 : f32
    } -> tensor<256x256xf32>
    %2148 = arith.constant {prov.region_id = "reduce_16", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} 0.000000e+00 : f32
    %2149 = tensor.splat %2148 {prov.region_id = "reduce_16", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} : tensor<f32>
    %2150 = linalg.reduce ins(%2144:tensor<256x256xf32>) outs(%2149:tensor<f32>) dimensions = [0, 1]
    (%2151: f32, %2152: f32) {
      %2153 = arith.addf %2151, %2152 : f32
      linalg.yield %2153 : f32
    }
    %2154 = arith.constant {prov.region_id = "reduce_16", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} 6.553600e+04 : f32
    %2155 = tensor.splat %2154 {prov.region_id = "reduce_16", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} : tensor<f32>
    %2156 = tensor.empty() : tensor<f32>
    %2157 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2150, %2155 : tensor<f32>, tensor<f32>) outs(%2156 : tensor<f32>) attrs =  {prov.region_id = "reduce_16", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} {
    ^bb259(%2158: f32, %2159: f32, %2160: f32):
      %2161 = arith.divf %2158, %2159 : f32
      linalg.yield %2161 : f32
    } -> tensor<f32>
    %2162 = func.call @aten_clamp__default_1(%2157) {prov.region_id = "aten_clamp__default_1_10", prov.dispatch_id = "aten_clamp__default_1_10"} : (tensor<f32>) -> tensor<f32>
    %2163 = tensor.empty() : tensor<f32>
    %2164 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2162 : tensor<f32>) outs(%2163 : tensor<f32>) attrs =  {prov.region_id = "elementwise_21", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} {
    ^bb260(%2165: f32, %2166: f32):
      %2167 = arith.constant 1.000000e+00 : f32
      %2168 = arith.divf %2167, %2165 : f32
      linalg.yield %2168 : f32
    } -> tensor<f32>
    %2169 = arith.constant {prov.region_id = "mul_63", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} 1.000000e+00 : f32
    %2170 = tensor.splat %2169 {prov.region_id = "mul_63", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} : tensor<f32>
    %2171 = tensor.empty() : tensor<f32>
    %2172 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2164, %2170 : tensor<f32>, tensor<f32>) outs(%2171 : tensor<f32>) attrs =  {prov.region_id = "mul_63", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} {
    ^bb261(%2173: f32, %2174: f32, %2175: f32):
      %2176 = arith.mulf %2173, %2174 : f32
      linalg.yield %2176 : f32
    } -> tensor<f32>
    %2177 = tensor.empty() : tensor<256x256xf32>
    %2178 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14, %2172 : tensor<256x256xf32>, tensor<f32>) outs(%2177 : tensor<256x256xf32>) attrs =  {prov.region_id = "mul_64", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} {
    ^bb262(%2179: f32, %2180: f32, %2181: f32):
      %2182 = arith.mulf %2179, %2180 : f32
      linalg.yield %2182 : f32
    } -> tensor<256x256xf32>
    %2183 = tensor.empty() : tensor<256x256xf32>
    %2184 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2178 : tensor<256x256xf32>) outs(%2183 : tensor<256x256xf32>) attrs =  {prov.region_id = "round_21", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} {
    ^bb263(%2185: f32, %2186: f32):
      %2187 = math.roundeven %2185 : f32
      linalg.yield %2187 : f32
    } -> tensor<256x256xf32>
    %2188 = tensor.empty() : tensor<256x256xf32>
    %2189 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2184 : tensor<256x256xf32>) outs(%2188 : tensor<256x256xf32>) attrs =  {prov.region_id = "minmax_22", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} {
    ^bb264(%2190: f32, %2191: f32):
      %2192 = arith.constant -1.000000e+00 : f32
      %2193 = arith.maximumf %2190, %2192 : f32
      %2194 = arith.constant 1.000000e+00 : f32
      %2195 = arith.minimumf %2193, %2194 : f32
      linalg.yield %2195 : f32
    } -> tensor<256x256xf32>
    %2196 = tensor.empty() : tensor<256x256xf32>
    %2197 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2189, %2172 : tensor<256x256xf32>, tensor<f32>) outs(%2196 : tensor<256x256xf32>) attrs =  {prov.region_id = "div_23", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} {
    ^bb265(%2198: f32, %2199: f32, %2200: f32):
      %2201 = arith.divf %2198, %2199 : f32
      linalg.yield %2201 : f32
    } -> tensor<256x256xf32>
    %2202 = tensor.empty() : tensor<256x256xf32>
    %2203 = linalg.transpose ins(%2197:tensor<256x256xf32>) outs(%2202:tensor<256x256xf32>) permutation = [1, 0]
    %2204 = tensor.empty() : tensor<1x32x256xf32>
    %2205 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %2206 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%2205 : f32) outs(%2204 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %2207 = linalg.matmul {prov.region_id = "matmul_14", prov.dispatch_id = "matmul_14", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.self_attn.o_proj"} ins(%2138, %2203 : tensor<1x32x256xf32>, tensor<256x256xf32>) outs(%2206 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %2208 = tensor.empty() : tensor<1x32x256xf32>
    %2209 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1385, %2207 : tensor<1x32x256xf32>, tensor<1x32x256xf32>) outs(%2208 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb266(%2210: f32, %2211: f32, %2212: f32):
      %2213 = arith.addf %2210, %2211 : f32
      linalg.yield %2213 : f32
    } -> tensor<1x32x256xf32>
    %2214 = tensor.empty() : tensor<1x32x256xf32>
    %2215 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2209 : tensor<1x32x256xf32>) outs(%2214 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.post_attention_layernorm"} {
    ^bb267(%2216: f32, %2217: f32):
      %2218 = arith.constant 2.000000e+00 : f32
      %2219 = math.powf %2216, %2218 : f32
      linalg.yield %2219 : f32
    } -> tensor<1x32x256xf32>
    %2220 = arith.constant {prov.region_id = "reduce_17", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.post_attention_layernorm"} 0.000000e+00 : f32
    %2221 = tensor.splat %2220 {prov.region_id = "reduce_17", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.post_attention_layernorm"} : tensor<1x32xf32>
    %2222 = linalg.reduce ins(%2215:tensor<1x32x256xf32>) outs(%2221:tensor<1x32xf32>) dimensions = [2]
    (%2223: f32, %2224: f32) {
      %2225 = arith.addf %2223, %2224 : f32
      linalg.yield %2225 : f32
    }
    %2226 = arith.constant {prov.region_id = "reduce_17", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.post_attention_layernorm"} 2.560000e+02 : f32
    %2227 = tensor.splat %2226 {prov.region_id = "reduce_17", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.post_attention_layernorm"} : tensor<1x32xf32>
    %2228 = tensor.empty() : tensor<1x32xf32>
    %2229 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2222, %2227 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%2228 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_17", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.post_attention_layernorm"} {
    ^bb268(%2230: f32, %2231: f32, %2232: f32):
      %2233 = arith.divf %2230, %2231 : f32
      linalg.yield %2233 : f32
    } -> tensor<1x32xf32>
    %2234 = tensor.collapse_shape %2229 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_17", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.post_attention_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %2235 = tensor.expand_shape %2234 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_17", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.post_attention_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2236 = arith.constant {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.post_attention_layernorm"} 1.000000e-05 : f32
    %2237 = tensor.splat %2236 {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.post_attention_layernorm"} : tensor<1x32x1xf32>
    %2238 = tensor.empty() : tensor<1x32x1xf32>
    %2239 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2235, %2237 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2238 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.post_attention_layernorm"} {
    ^bb269(%2240: f32, %2241: f32, %2242: f32):
      %2243 = arith.addf %2240, %2241 : f32
      linalg.yield %2243 : f32
    } -> tensor<1x32x1xf32>
    %2244 = tensor.empty() : tensor<1x32x1xf32>
    %2245 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2239 : tensor<1x32x1xf32>) outs(%2244 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_6", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.post_attention_layernorm"} {
    ^bb270(%2246: f32, %2247: f32):
      %2248 = math.rsqrt %2246 : f32
      linalg.yield %2248 : f32
    } -> tensor<1x32x1xf32>
    %2249 = tensor.empty() : tensor<1x32x256xf32>
    %2250 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2209, %2245 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2249 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_65", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.post_attention_layernorm"} {
    ^bb271(%2251: f32, %2252: f32, %2253: f32):
      %2254 = arith.mulf %2251, %2252 : f32
      linalg.yield %2254 : f32
    } -> tensor<1x32x256xf32>
    %2255 = tensor.empty() : tensor<1x32x256xf32>
    %2256 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%21, %2250 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%2255 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_66", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.post_attention_layernorm"} {
    ^bb272(%2257: f32, %2258: f32, %2259: f32):
      %2260 = arith.mulf %2257, %2258 : f32
      linalg.yield %2260 : f32
    } -> tensor<1x32x256xf32>
    %2261 = tensor.empty() : tensor<1x32x256xf32>
    %2262 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2256 : tensor<1x32x256xf32>) outs(%2261 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_22", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} {
    ^bb273(%2263: f32, %2264: f32):
      %2265 = math.absf %2263 : f32
      linalg.yield %2265 : f32
    } -> tensor<1x32x256xf32>
    %2266 = arith.constant {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} 0xff800000 : f32
    %2267 = arith.constant {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} 0 : i64
    %2268 = tensor.splat %2266 {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} : tensor<1x32xf32>
    %2269 = tensor.splat %2267 {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} : tensor<1x32xi64>
    %2270, %2271 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2262 : tensor<1x32x256xf32>) outs(%2268, %2269 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} {
    ^bb274(%2272: f32, %2273: f32, %2274: i64):
      %2275 = linalg.index 2 : index
      %2276 = arith.index_cast %2275 : index to i64
      %2277 = arith.cmpf ogt, %2272, %2273 : f32
      %2278 = arith.select %2277, %2272, %2273 : f32
      %2279 = arith.select %2277, %2276, %2274 : i64
      linalg.yield %2278, %2279 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %2280 = tensor.collapse_shape %2270 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %2281 = tensor.expand_shape %2280 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2282 = tensor.collapse_shape %2271 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %2283 = tensor.expand_shape %2282 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %2284 = func.call @aten_clamp__default(%2281) {prov.region_id = "aten_clamp__default_11", prov.dispatch_id = "aten_clamp__default_11"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %2285 = tensor.empty() : tensor<1x32x1xf32>
    %2286 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2284 : tensor<1x32x1xf32>) outs(%2285 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_22", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} {
    ^bb275(%2287: f32, %2288: f32):
      %2289 = arith.constant 1.000000e+00 : f32
      %2290 = arith.divf %2289, %2287 : f32
      linalg.yield %2290 : f32
    } -> tensor<1x32x1xf32>
    %2291 = arith.constant {prov.region_id = "mul_67", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} 1.270000e+02 : f32
    %2292 = tensor.splat %2291 {prov.region_id = "mul_67", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} : tensor<1x32x1xf32>
    %2293 = tensor.empty() : tensor<1x32x1xf32>
    %2294 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2286, %2292 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2293 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_67", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} {
    ^bb276(%2295: f32, %2296: f32, %2297: f32):
      %2298 = arith.mulf %2295, %2296 : f32
      linalg.yield %2298 : f32
    } -> tensor<1x32x1xf32>
    %2299 = tensor.empty() : tensor<1x32x256xf32>
    %2300 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2256, %2294 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2299 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_68", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} {
    ^bb277(%2301: f32, %2302: f32, %2303: f32):
      %2304 = arith.mulf %2301, %2302 : f32
      linalg.yield %2304 : f32
    } -> tensor<1x32x256xf32>
    %2305 = tensor.empty() : tensor<1x32x256xf32>
    %2306 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2300 : tensor<1x32x256xf32>) outs(%2305 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_22", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} {
    ^bb278(%2307: f32, %2308: f32):
      %2309 = math.roundeven %2307 : f32
      linalg.yield %2309 : f32
    } -> tensor<1x32x256xf32>
    %2310 = tensor.empty() : tensor<1x32x256xf32>
    %2311 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2306 : tensor<1x32x256xf32>) outs(%2310 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_23", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} {
    ^bb279(%2312: f32, %2313: f32):
      %2314 = arith.constant -1.280000e+02 : f32
      %2315 = arith.maximumf %2312, %2314 : f32
      %2316 = arith.constant 1.270000e+02 : f32
      %2317 = arith.minimumf %2315, %2316 : f32
      linalg.yield %2317 : f32
    } -> tensor<1x32x256xf32>
    %2318 = tensor.empty() : tensor<1x32x256xf32>
    %2319 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2311, %2294 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2318 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_24", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} {
    ^bb280(%2320: f32, %2321: f32, %2322: f32):
      %2323 = arith.divf %2320, %2321 : f32
      linalg.yield %2323 : f32
    } -> tensor<1x32x256xf32>
    %2324 = tensor.empty() : tensor<512x256xf32>
    %2325 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<512x256xf32>) outs(%2324 : tensor<512x256xf32>) attrs =  {prov.region_id = "abs_23", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} {
    ^bb281(%2326: f32, %2327: f32):
      %2328 = math.absf %2326 : f32
      linalg.yield %2328 : f32
    } -> tensor<512x256xf32>
    %2329 = arith.constant {prov.region_id = "reduce_18", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} 0.000000e+00 : f32
    %2330 = tensor.splat %2329 {prov.region_id = "reduce_18", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} : tensor<f32>
    %2331 = linalg.reduce ins(%2325:tensor<512x256xf32>) outs(%2330:tensor<f32>) dimensions = [0, 1]
    (%2332: f32, %2333: f32) {
      %2334 = arith.addf %2332, %2333 : f32
      linalg.yield %2334 : f32
    }
    %2335 = arith.constant {prov.region_id = "reduce_18", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} 1.310720e+05 : f32
    %2336 = tensor.splat %2335 {prov.region_id = "reduce_18", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} : tensor<f32>
    %2337 = tensor.empty() : tensor<f32>
    %2338 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2331, %2336 : tensor<f32>, tensor<f32>) outs(%2337 : tensor<f32>) attrs =  {prov.region_id = "reduce_18", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} {
    ^bb282(%2339: f32, %2340: f32, %2341: f32):
      %2342 = arith.divf %2339, %2340 : f32
      linalg.yield %2342 : f32
    } -> tensor<f32>
    %2343 = func.call @aten_clamp__default_1(%2338) {prov.region_id = "aten_clamp__default_1_11", prov.dispatch_id = "aten_clamp__default_1_11"} : (tensor<f32>) -> tensor<f32>
    %2344 = tensor.empty() : tensor<f32>
    %2345 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2343 : tensor<f32>) outs(%2344 : tensor<f32>) attrs =  {prov.region_id = "elementwise_23", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} {
    ^bb283(%2346: f32, %2347: f32):
      %2348 = arith.constant 1.000000e+00 : f32
      %2349 = arith.divf %2348, %2346 : f32
      linalg.yield %2349 : f32
    } -> tensor<f32>
    %2350 = arith.constant {prov.region_id = "mul_69", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} 1.000000e+00 : f32
    %2351 = tensor.splat %2350 {prov.region_id = "mul_69", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} : tensor<f32>
    %2352 = tensor.empty() : tensor<f32>
    %2353 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2345, %2351 : tensor<f32>, tensor<f32>) outs(%2352 : tensor<f32>) attrs =  {prov.region_id = "mul_69", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} {
    ^bb284(%2354: f32, %2355: f32, %2356: f32):
      %2357 = arith.mulf %2354, %2355 : f32
      linalg.yield %2357 : f32
    } -> tensor<f32>
    %2358 = tensor.empty() : tensor<512x256xf32>
    %2359 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%16, %2353 : tensor<512x256xf32>, tensor<f32>) outs(%2358 : tensor<512x256xf32>) attrs =  {prov.region_id = "mul_70", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} {
    ^bb285(%2360: f32, %2361: f32, %2362: f32):
      %2363 = arith.mulf %2360, %2361 : f32
      linalg.yield %2363 : f32
    } -> tensor<512x256xf32>
    %2364 = tensor.empty() : tensor<512x256xf32>
    %2365 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2359 : tensor<512x256xf32>) outs(%2364 : tensor<512x256xf32>) attrs =  {prov.region_id = "round_23", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} {
    ^bb286(%2366: f32, %2367: f32):
      %2368 = math.roundeven %2366 : f32
      linalg.yield %2368 : f32
    } -> tensor<512x256xf32>
    %2369 = tensor.empty() : tensor<512x256xf32>
    %2370 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2365 : tensor<512x256xf32>) outs(%2369 : tensor<512x256xf32>) attrs =  {prov.region_id = "minmax_24", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} {
    ^bb287(%2371: f32, %2372: f32):
      %2373 = arith.constant -1.000000e+00 : f32
      %2374 = arith.maximumf %2371, %2373 : f32
      %2375 = arith.constant 1.000000e+00 : f32
      %2376 = arith.minimumf %2374, %2375 : f32
      linalg.yield %2376 : f32
    } -> tensor<512x256xf32>
    %2377 = tensor.empty() : tensor<512x256xf32>
    %2378 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2370, %2353 : tensor<512x256xf32>, tensor<f32>) outs(%2377 : tensor<512x256xf32>) attrs =  {prov.region_id = "div_25", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} {
    ^bb288(%2379: f32, %2380: f32, %2381: f32):
      %2382 = arith.divf %2379, %2380 : f32
      linalg.yield %2382 : f32
    } -> tensor<512x256xf32>
    %2383 = tensor.empty() : tensor<256x512xf32>
    %2384 = linalg.transpose ins(%2378:tensor<512x256xf32>) outs(%2383:tensor<256x512xf32>) permutation = [1, 0]
    %2385 = tensor.empty() : tensor<1x32x512xf32>
    %2386 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %2387 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%2386 : f32) outs(%2385 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %2388 = linalg.matmul {prov.region_id = "matmul_15", prov.dispatch_id = "matmul_15", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.gate_proj"} ins(%2319, %2384 : tensor<1x32x256xf32>, tensor<256x512xf32>) outs(%2387 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %2389 = tensor.empty() : tensor<1x32x512xf32>
    %2390 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2388 : tensor<1x32x512xf32>) outs(%2389 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "minmax_25", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu.default", prov.orig_dtype = "float32"} {
    ^bb289(%2391: f32, %2392: f32):
      %2393 = arith.constant 0.000000e+00 : f32
      %2394 = arith.maximumf %2391, %2393 : f32
      linalg.yield %2394 : f32
    } -> tensor<1x32x512xf32>
    %2395 = tensor.empty() : tensor<1x32x512xf32>
    %2396 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2390 : tensor<1x32x512xf32>) outs(%2395 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb290(%2397: f32, %2398: f32):
      %2399 = arith.constant 2.000000e+00 : f32
      %2400 = math.powf %2397, %2399 : f32
      linalg.yield %2400 : f32
    } -> tensor<1x32x512xf32>
    %2401 = tensor.empty() : tensor<1x32x256xf32>
    %2402 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2256 : tensor<1x32x256xf32>) outs(%2401 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_24", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} {
    ^bb291(%2403: f32, %2404: f32):
      %2405 = math.absf %2403 : f32
      linalg.yield %2405 : f32
    } -> tensor<1x32x256xf32>
    %2406 = arith.constant {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} 0xff800000 : f32
    %2407 = arith.constant {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} 0 : i64
    %2408 = tensor.splat %2406 {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} : tensor<1x32xf32>
    %2409 = tensor.splat %2407 {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} : tensor<1x32xi64>
    %2410, %2411 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2402 : tensor<1x32x256xf32>) outs(%2408, %2409 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} {
    ^bb292(%2412: f32, %2413: f32, %2414: i64):
      %2415 = linalg.index 2 : index
      %2416 = arith.index_cast %2415 : index to i64
      %2417 = arith.cmpf ogt, %2412, %2413 : f32
      %2418 = arith.select %2417, %2412, %2413 : f32
      %2419 = arith.select %2417, %2416, %2414 : i64
      linalg.yield %2418, %2419 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %2420 = tensor.collapse_shape %2410 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %2421 = tensor.expand_shape %2420 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2422 = tensor.collapse_shape %2411 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %2423 = tensor.expand_shape %2422 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %2424 = func.call @aten_clamp__default(%2421) {prov.region_id = "aten_clamp__default_12", prov.dispatch_id = "aten_clamp__default_12"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %2425 = tensor.empty() : tensor<1x32x1xf32>
    %2426 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2424 : tensor<1x32x1xf32>) outs(%2425 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_24", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} {
    ^bb293(%2427: f32, %2428: f32):
      %2429 = arith.constant 1.000000e+00 : f32
      %2430 = arith.divf %2429, %2427 : f32
      linalg.yield %2430 : f32
    } -> tensor<1x32x1xf32>
    %2431 = arith.constant {prov.region_id = "mul_71", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} 1.270000e+02 : f32
    %2432 = tensor.splat %2431 {prov.region_id = "mul_71", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} : tensor<1x32x1xf32>
    %2433 = tensor.empty() : tensor<1x32x1xf32>
    %2434 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2426, %2432 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2433 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_71", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} {
    ^bb294(%2435: f32, %2436: f32, %2437: f32):
      %2438 = arith.mulf %2435, %2436 : f32
      linalg.yield %2438 : f32
    } -> tensor<1x32x1xf32>
    %2439 = tensor.empty() : tensor<1x32x256xf32>
    %2440 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2256, %2434 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2439 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_72", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} {
    ^bb295(%2441: f32, %2442: f32, %2443: f32):
      %2444 = arith.mulf %2441, %2442 : f32
      linalg.yield %2444 : f32
    } -> tensor<1x32x256xf32>
    %2445 = tensor.empty() : tensor<1x32x256xf32>
    %2446 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2440 : tensor<1x32x256xf32>) outs(%2445 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_24", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} {
    ^bb296(%2447: f32, %2448: f32):
      %2449 = math.roundeven %2447 : f32
      linalg.yield %2449 : f32
    } -> tensor<1x32x256xf32>
    %2450 = tensor.empty() : tensor<1x32x256xf32>
    %2451 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2446 : tensor<1x32x256xf32>) outs(%2450 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_26", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} {
    ^bb297(%2452: f32, %2453: f32):
      %2454 = arith.constant -1.280000e+02 : f32
      %2455 = arith.maximumf %2452, %2454 : f32
      %2456 = arith.constant 1.270000e+02 : f32
      %2457 = arith.minimumf %2455, %2456 : f32
      linalg.yield %2457 : f32
    } -> tensor<1x32x256xf32>
    %2458 = tensor.empty() : tensor<1x32x256xf32>
    %2459 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2451, %2434 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2458 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_26", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} {
    ^bb298(%2460: f32, %2461: f32, %2462: f32):
      %2463 = arith.divf %2460, %2461 : f32
      linalg.yield %2463 : f32
    } -> tensor<1x32x256xf32>
    %2464 = tensor.empty() : tensor<512x256xf32>
    %2465 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17 : tensor<512x256xf32>) outs(%2464 : tensor<512x256xf32>) attrs =  {prov.region_id = "abs_25", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} {
    ^bb299(%2466: f32, %2467: f32):
      %2468 = math.absf %2466 : f32
      linalg.yield %2468 : f32
    } -> tensor<512x256xf32>
    %2469 = arith.constant {prov.region_id = "reduce_19", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} 0.000000e+00 : f32
    %2470 = tensor.splat %2469 {prov.region_id = "reduce_19", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} : tensor<f32>
    %2471 = linalg.reduce ins(%2465:tensor<512x256xf32>) outs(%2470:tensor<f32>) dimensions = [0, 1]
    (%2472: f32, %2473: f32) {
      %2474 = arith.addf %2472, %2473 : f32
      linalg.yield %2474 : f32
    }
    %2475 = arith.constant {prov.region_id = "reduce_19", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} 1.310720e+05 : f32
    %2476 = tensor.splat %2475 {prov.region_id = "reduce_19", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} : tensor<f32>
    %2477 = tensor.empty() : tensor<f32>
    %2478 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2471, %2476 : tensor<f32>, tensor<f32>) outs(%2477 : tensor<f32>) attrs =  {prov.region_id = "reduce_19", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} {
    ^bb300(%2479: f32, %2480: f32, %2481: f32):
      %2482 = arith.divf %2479, %2480 : f32
      linalg.yield %2482 : f32
    } -> tensor<f32>
    %2483 = func.call @aten_clamp__default_1(%2478) {prov.region_id = "aten_clamp__default_1_12", prov.dispatch_id = "aten_clamp__default_1_12"} : (tensor<f32>) -> tensor<f32>
    %2484 = tensor.empty() : tensor<f32>
    %2485 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2483 : tensor<f32>) outs(%2484 : tensor<f32>) attrs =  {prov.region_id = "elementwise_25", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} {
    ^bb301(%2486: f32, %2487: f32):
      %2488 = arith.constant 1.000000e+00 : f32
      %2489 = arith.divf %2488, %2486 : f32
      linalg.yield %2489 : f32
    } -> tensor<f32>
    %2490 = arith.constant {prov.region_id = "mul_73", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} 1.000000e+00 : f32
    %2491 = tensor.splat %2490 {prov.region_id = "mul_73", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} : tensor<f32>
    %2492 = tensor.empty() : tensor<f32>
    %2493 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2485, %2491 : tensor<f32>, tensor<f32>) outs(%2492 : tensor<f32>) attrs =  {prov.region_id = "mul_73", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} {
    ^bb302(%2494: f32, %2495: f32, %2496: f32):
      %2497 = arith.mulf %2494, %2495 : f32
      linalg.yield %2497 : f32
    } -> tensor<f32>
    %2498 = tensor.empty() : tensor<512x256xf32>
    %2499 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %2493 : tensor<512x256xf32>, tensor<f32>) outs(%2498 : tensor<512x256xf32>) attrs =  {prov.region_id = "mul_74", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} {
    ^bb303(%2500: f32, %2501: f32, %2502: f32):
      %2503 = arith.mulf %2500, %2501 : f32
      linalg.yield %2503 : f32
    } -> tensor<512x256xf32>
    %2504 = tensor.empty() : tensor<512x256xf32>
    %2505 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2499 : tensor<512x256xf32>) outs(%2504 : tensor<512x256xf32>) attrs =  {prov.region_id = "round_25", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} {
    ^bb304(%2506: f32, %2507: f32):
      %2508 = math.roundeven %2506 : f32
      linalg.yield %2508 : f32
    } -> tensor<512x256xf32>
    %2509 = tensor.empty() : tensor<512x256xf32>
    %2510 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2505 : tensor<512x256xf32>) outs(%2509 : tensor<512x256xf32>) attrs =  {prov.region_id = "minmax_27", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} {
    ^bb305(%2511: f32, %2512: f32):
      %2513 = arith.constant -1.000000e+00 : f32
      %2514 = arith.maximumf %2511, %2513 : f32
      %2515 = arith.constant 1.000000e+00 : f32
      %2516 = arith.minimumf %2514, %2515 : f32
      linalg.yield %2516 : f32
    } -> tensor<512x256xf32>
    %2517 = tensor.empty() : tensor<512x256xf32>
    %2518 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2510, %2493 : tensor<512x256xf32>, tensor<f32>) outs(%2517 : tensor<512x256xf32>) attrs =  {prov.region_id = "div_27", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} {
    ^bb306(%2519: f32, %2520: f32, %2521: f32):
      %2522 = arith.divf %2519, %2520 : f32
      linalg.yield %2522 : f32
    } -> tensor<512x256xf32>
    %2523 = tensor.empty() : tensor<256x512xf32>
    %2524 = linalg.transpose ins(%2518:tensor<512x256xf32>) outs(%2523:tensor<256x512xf32>) permutation = [1, 0]
    %2525 = tensor.empty() : tensor<1x32x512xf32>
    %2526 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %2527 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%2526 : f32) outs(%2525 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %2528 = linalg.matmul {prov.region_id = "matmul_16", prov.dispatch_id = "matmul_16", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.up_proj"} ins(%2459, %2524 : tensor<1x32x256xf32>, tensor<256x512xf32>) outs(%2527 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %2529 = tensor.empty() : tensor<1x32x512xf32>
    %2530 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2396, %2528 : tensor<1x32x512xf32>, tensor<1x32x512xf32>) outs(%2529 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_75", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb307(%2531: f32, %2532: f32, %2533: f32):
      %2534 = arith.mulf %2531, %2532 : f32
      linalg.yield %2534 : f32
    } -> tensor<1x32x512xf32>
    %2535 = tensor.empty() : tensor<1x32x512xf32>
    %2536 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2530 : tensor<1x32x512xf32>) outs(%2535 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "pow_9", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.ffn_sub_norm"} {
    ^bb308(%2537: f32, %2538: f32):
      %2539 = arith.constant 2.000000e+00 : f32
      %2540 = math.powf %2537, %2539 : f32
      linalg.yield %2540 : f32
    } -> tensor<1x32x512xf32>
    %2541 = arith.constant {prov.region_id = "reduce_20", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.ffn_sub_norm"} 0.000000e+00 : f32
    %2542 = tensor.splat %2541 {prov.region_id = "reduce_20", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.ffn_sub_norm"} : tensor<1x32xf32>
    %2543 = linalg.reduce ins(%2536:tensor<1x32x512xf32>) outs(%2542:tensor<1x32xf32>) dimensions = [2]
    (%2544: f32, %2545: f32) {
      %2546 = arith.addf %2544, %2545 : f32
      linalg.yield %2546 : f32
    }
    %2547 = arith.constant {prov.region_id = "reduce_20", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.ffn_sub_norm"} 5.120000e+02 : f32
    %2548 = tensor.splat %2547 {prov.region_id = "reduce_20", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.ffn_sub_norm"} : tensor<1x32xf32>
    %2549 = tensor.empty() : tensor<1x32xf32>
    %2550 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2543, %2548 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%2549 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_20", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.ffn_sub_norm"} {
    ^bb309(%2551: f32, %2552: f32, %2553: f32):
      %2554 = arith.divf %2551, %2552 : f32
      linalg.yield %2554 : f32
    } -> tensor<1x32xf32>
    %2555 = tensor.collapse_shape %2550 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_20", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.ffn_sub_norm"} : tensor<1x32xf32> into tensor<32xf32>
    %2556 = tensor.expand_shape %2555 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_20", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.ffn_sub_norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2557 = arith.constant {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.ffn_sub_norm"} 1.000000e-05 : f32
    %2558 = tensor.splat %2557 {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.ffn_sub_norm"} : tensor<1x32x1xf32>
    %2559 = tensor.empty() : tensor<1x32x1xf32>
    %2560 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2556, %2558 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2559 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.ffn_sub_norm"} {
    ^bb310(%2561: f32, %2562: f32, %2563: f32):
      %2564 = arith.addf %2561, %2562 : f32
      linalg.yield %2564 : f32
    } -> tensor<1x32x1xf32>
    %2565 = tensor.empty() : tensor<1x32x1xf32>
    %2566 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2560 : tensor<1x32x1xf32>) outs(%2565 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_7", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.ffn_sub_norm"} {
    ^bb311(%2567: f32, %2568: f32):
      %2569 = math.rsqrt %2567 : f32
      linalg.yield %2569 : f32
    } -> tensor<1x32x1xf32>
    %2570 = tensor.empty() : tensor<1x32x512xf32>
    %2571 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2530, %2566 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%2570 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_76", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.ffn_sub_norm"} {
    ^bb312(%2572: f32, %2573: f32, %2574: f32):
      %2575 = arith.mulf %2572, %2573 : f32
      linalg.yield %2575 : f32
    } -> tensor<1x32x512xf32>
    %2576 = tensor.empty() : tensor<1x32x512xf32>
    %2577 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %2571 : tensor<512xf32>, tensor<1x32x512xf32>) outs(%2576 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_77", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.ffn_sub_norm"} {
    ^bb313(%2578: f32, %2579: f32, %2580: f32):
      %2581 = arith.mulf %2578, %2579 : f32
      linalg.yield %2581 : f32
    } -> tensor<1x32x512xf32>
    %2582 = tensor.empty() : tensor<1x32x512xf32>
    %2583 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2577 : tensor<1x32x512xf32>) outs(%2582 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "abs_26", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} {
    ^bb314(%2584: f32, %2585: f32):
      %2586 = math.absf %2584 : f32
      linalg.yield %2586 : f32
    } -> tensor<1x32x512xf32>
    %2587 = arith.constant {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} 0xff800000 : f32
    %2588 = arith.constant {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} 0 : i64
    %2589 = tensor.splat %2587 {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} : tensor<1x32xf32>
    %2590 = tensor.splat %2588 {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} : tensor<1x32xi64>
    %2591, %2592 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2583 : tensor<1x32x512xf32>) outs(%2589, %2590 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} {
    ^bb315(%2593: f32, %2594: f32, %2595: i64):
      %2596 = linalg.index 2 : index
      %2597 = arith.index_cast %2596 : index to i64
      %2598 = arith.cmpf ogt, %2593, %2594 : f32
      %2599 = arith.select %2598, %2593, %2594 : f32
      %2600 = arith.select %2598, %2597, %2595 : i64
      linalg.yield %2599, %2600 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %2601 = tensor.collapse_shape %2591 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %2602 = tensor.expand_shape %2601 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2603 = tensor.collapse_shape %2592 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %2604 = tensor.expand_shape %2603 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %2605 = func.call @aten_clamp__default(%2602) {prov.region_id = "aten_clamp__default_13", prov.dispatch_id = "aten_clamp__default_13"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %2606 = tensor.empty() : tensor<1x32x1xf32>
    %2607 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2605 : tensor<1x32x1xf32>) outs(%2606 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_26", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} {
    ^bb316(%2608: f32, %2609: f32):
      %2610 = arith.constant 1.000000e+00 : f32
      %2611 = arith.divf %2610, %2608 : f32
      linalg.yield %2611 : f32
    } -> tensor<1x32x1xf32>
    %2612 = arith.constant {prov.region_id = "mul_78", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} 1.270000e+02 : f32
    %2613 = tensor.splat %2612 {prov.region_id = "mul_78", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} : tensor<1x32x1xf32>
    %2614 = tensor.empty() : tensor<1x32x1xf32>
    %2615 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2607, %2613 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2614 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_78", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} {
    ^bb317(%2616: f32, %2617: f32, %2618: f32):
      %2619 = arith.mulf %2616, %2617 : f32
      linalg.yield %2619 : f32
    } -> tensor<1x32x1xf32>
    %2620 = tensor.empty() : tensor<1x32x512xf32>
    %2621 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2577, %2615 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%2620 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_79", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} {
    ^bb318(%2622: f32, %2623: f32, %2624: f32):
      %2625 = arith.mulf %2622, %2623 : f32
      linalg.yield %2625 : f32
    } -> tensor<1x32x512xf32>
    %2626 = tensor.empty() : tensor<1x32x512xf32>
    %2627 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2621 : tensor<1x32x512xf32>) outs(%2626 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "round_26", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} {
    ^bb319(%2628: f32, %2629: f32):
      %2630 = math.roundeven %2628 : f32
      linalg.yield %2630 : f32
    } -> tensor<1x32x512xf32>
    %2631 = tensor.empty() : tensor<1x32x512xf32>
    %2632 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2627 : tensor<1x32x512xf32>) outs(%2631 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "minmax_28", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} {
    ^bb320(%2633: f32, %2634: f32):
      %2635 = arith.constant -1.280000e+02 : f32
      %2636 = arith.maximumf %2633, %2635 : f32
      %2637 = arith.constant 1.270000e+02 : f32
      %2638 = arith.minimumf %2636, %2637 : f32
      linalg.yield %2638 : f32
    } -> tensor<1x32x512xf32>
    %2639 = tensor.empty() : tensor<1x32x512xf32>
    %2640 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2632, %2615 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%2639 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "div_28", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} {
    ^bb321(%2641: f32, %2642: f32, %2643: f32):
      %2644 = arith.divf %2641, %2642 : f32
      linalg.yield %2644 : f32
    } -> tensor<1x32x512xf32>
    %2645 = tensor.empty() : tensor<256x512xf32>
    %2646 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18 : tensor<256x512xf32>) outs(%2645 : tensor<256x512xf32>) attrs =  {prov.region_id = "abs_27", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} {
    ^bb322(%2647: f32, %2648: f32):
      %2649 = math.absf %2647 : f32
      linalg.yield %2649 : f32
    } -> tensor<256x512xf32>
    %2650 = arith.constant {prov.region_id = "reduce_21", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} 0.000000e+00 : f32
    %2651 = tensor.splat %2650 {prov.region_id = "reduce_21", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} : tensor<f32>
    %2652 = linalg.reduce ins(%2646:tensor<256x512xf32>) outs(%2651:tensor<f32>) dimensions = [0, 1]
    (%2653: f32, %2654: f32) {
      %2655 = arith.addf %2653, %2654 : f32
      linalg.yield %2655 : f32
    }
    %2656 = arith.constant {prov.region_id = "reduce_21", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} 1.310720e+05 : f32
    %2657 = tensor.splat %2656 {prov.region_id = "reduce_21", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} : tensor<f32>
    %2658 = tensor.empty() : tensor<f32>
    %2659 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2652, %2657 : tensor<f32>, tensor<f32>) outs(%2658 : tensor<f32>) attrs =  {prov.region_id = "reduce_21", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} {
    ^bb323(%2660: f32, %2661: f32, %2662: f32):
      %2663 = arith.divf %2660, %2661 : f32
      linalg.yield %2663 : f32
    } -> tensor<f32>
    %2664 = func.call @aten_clamp__default_1(%2659) {prov.region_id = "aten_clamp__default_1_13", prov.dispatch_id = "aten_clamp__default_1_13"} : (tensor<f32>) -> tensor<f32>
    %2665 = tensor.empty() : tensor<f32>
    %2666 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2664 : tensor<f32>) outs(%2665 : tensor<f32>) attrs =  {prov.region_id = "elementwise_27", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} {
    ^bb324(%2667: f32, %2668: f32):
      %2669 = arith.constant 1.000000e+00 : f32
      %2670 = arith.divf %2669, %2667 : f32
      linalg.yield %2670 : f32
    } -> tensor<f32>
    %2671 = arith.constant {prov.region_id = "mul_80", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} 1.000000e+00 : f32
    %2672 = tensor.splat %2671 {prov.region_id = "mul_80", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} : tensor<f32>
    %2673 = tensor.empty() : tensor<f32>
    %2674 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2666, %2672 : tensor<f32>, tensor<f32>) outs(%2673 : tensor<f32>) attrs =  {prov.region_id = "mul_80", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} {
    ^bb325(%2675: f32, %2676: f32, %2677: f32):
      %2678 = arith.mulf %2675, %2676 : f32
      linalg.yield %2678 : f32
    } -> tensor<f32>
    %2679 = tensor.empty() : tensor<256x512xf32>
    %2680 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %2674 : tensor<256x512xf32>, tensor<f32>) outs(%2679 : tensor<256x512xf32>) attrs =  {prov.region_id = "mul_81", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} {
    ^bb326(%2681: f32, %2682: f32, %2683: f32):
      %2684 = arith.mulf %2681, %2682 : f32
      linalg.yield %2684 : f32
    } -> tensor<256x512xf32>
    %2685 = tensor.empty() : tensor<256x512xf32>
    %2686 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2680 : tensor<256x512xf32>) outs(%2685 : tensor<256x512xf32>) attrs =  {prov.region_id = "round_27", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} {
    ^bb327(%2687: f32, %2688: f32):
      %2689 = math.roundeven %2687 : f32
      linalg.yield %2689 : f32
    } -> tensor<256x512xf32>
    %2690 = tensor.empty() : tensor<256x512xf32>
    %2691 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2686 : tensor<256x512xf32>) outs(%2690 : tensor<256x512xf32>) attrs =  {prov.region_id = "minmax_29", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} {
    ^bb328(%2692: f32, %2693: f32):
      %2694 = arith.constant -1.000000e+00 : f32
      %2695 = arith.maximumf %2692, %2694 : f32
      %2696 = arith.constant 1.000000e+00 : f32
      %2697 = arith.minimumf %2695, %2696 : f32
      linalg.yield %2697 : f32
    } -> tensor<256x512xf32>
    %2698 = tensor.empty() : tensor<256x512xf32>
    %2699 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2691, %2674 : tensor<256x512xf32>, tensor<f32>) outs(%2698 : tensor<256x512xf32>) attrs =  {prov.region_id = "div_29", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} {
    ^bb329(%2700: f32, %2701: f32, %2702: f32):
      %2703 = arith.divf %2700, %2701 : f32
      linalg.yield %2703 : f32
    } -> tensor<256x512xf32>
    %2704 = tensor.empty() : tensor<512x256xf32>
    %2705 = linalg.transpose ins(%2699:tensor<256x512xf32>) outs(%2704:tensor<512x256xf32>) permutation = [1, 0]
    %2706 = tensor.empty() : tensor<1x32x256xf32>
    %2707 = arith.constant {prov.module = "layers"} 0.000000e+00 : f32
    %2708 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "layers"} ins(%2707 : f32) outs(%2706 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %2709 = linalg.matmul {prov.region_id = "matmul_17", prov.dispatch_id = "matmul_17", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "layers", prov.fqn = "layers.1.mlp.down_proj"} ins(%2640, %2705 : tensor<1x32x512xf32>, tensor<512x256xf32>) outs(%2708 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %2710 = tensor.empty() : tensor<1x32x256xf32>
    %2711 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2209, %2709 : tensor<1x32x256xf32>, tensor<1x32x256xf32>) outs(%2710 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb330(%2712: f32, %2713: f32, %2714: f32):
      %2715 = arith.addf %2712, %2713 : f32
      linalg.yield %2715 : f32
    } -> tensor<1x32x256xf32>
    %2716 = func.call @aten_stack_default(%586, %1912) {prov.region_id = "aten_stack_default_0", prov.dispatch_id = "aten_stack_default_0"} : (tensor<1x4x39x32xf32>, tensor<1x4x39x32xf32>) -> tensor<2x1x4x39x32xf32>
    %2717 = func.call @aten_stack_default(%594, %1920) {prov.region_id = "aten_stack_default_1", prov.dispatch_id = "aten_stack_default_1"} : (tensor<1x4x39x32xf32>, tensor<1x4x39x32xf32>) -> tensor<2x1x4x39x32xf32>
    %2718 = "tensor.extract_slice"(%2711) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 256>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_28", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %2719 = "tensor.extract_slice"(%2718) <{static_offsets = array<i64: 0, 31, 0>, static_sizes = array<i64: 1, 1, 256>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_29", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x32x256xf32>) -> tensor<1x1x256xf32>
    %2720 = "tensor.extract_slice"(%2719) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 256>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_30", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
    %2721 = tensor.empty() : tensor<1x1x256xf32>
    %2722 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2720 : tensor<1x1x256xf32>) outs(%2721 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "pow_10", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} {
    ^bb331(%2723: f32, %2724: f32):
      %2725 = arith.constant 2.000000e+00 : f32
      %2726 = math.powf %2723, %2725 : f32
      linalg.yield %2726 : f32
    } -> tensor<1x1x256xf32>
    %2727 = arith.constant {prov.region_id = "reduce_22", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} 0.000000e+00 : f32
    %2728 = tensor.splat %2727 {prov.region_id = "reduce_22", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} : tensor<1x1xf32>
    %2729 = linalg.reduce ins(%2722:tensor<1x1x256xf32>) outs(%2728:tensor<1x1xf32>) dimensions = [2]
    (%2730: f32, %2731: f32) {
      %2732 = arith.addf %2730, %2731 : f32
      linalg.yield %2732 : f32
    }
    %2733 = arith.constant {prov.region_id = "reduce_22", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} 2.560000e+02 : f32
    %2734 = tensor.splat %2733 {prov.region_id = "reduce_22", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} : tensor<1x1xf32>
    %2735 = tensor.empty() : tensor<1x1xf32>
    %2736 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2729, %2734 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%2735 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_22", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} {
    ^bb332(%2737: f32, %2738: f32, %2739: f32):
      %2740 = arith.divf %2737, %2738 : f32
      linalg.yield %2740 : f32
    } -> tensor<1x1xf32>
    %2741 = tensor.collapse_shape %2736 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_22", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} : tensor<1x1xf32> into tensor<1xf32>
    %2742 = tensor.expand_shape %2741 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_22", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} : tensor<1xf32> into tensor<1x1x1xf32>
    %2743 = arith.constant {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} 1.000000e-05 : f32
    %2744 = tensor.splat %2743 {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} : tensor<1x1x1xf32>
    %2745 = tensor.empty() : tensor<1x1x1xf32>
    %2746 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2742, %2744 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%2745 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} {
    ^bb333(%2747: f32, %2748: f32, %2749: f32):
      %2750 = arith.addf %2747, %2748 : f32
      linalg.yield %2750 : f32
    } -> tensor<1x1x1xf32>
    %2751 = tensor.empty() : tensor<1x1x1xf32>
    %2752 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2746 : tensor<1x1x1xf32>) outs(%2751 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_8", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} {
    ^bb334(%2753: f32, %2754: f32):
      %2755 = math.rsqrt %2753 : f32
      linalg.yield %2755 : f32
    } -> tensor<1x1x1xf32>
    %2756 = tensor.empty() : tensor<1x1x256xf32>
    %2757 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2720, %2752 : tensor<1x1x256xf32>, tensor<1x1x1xf32>) outs(%2756 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_82", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} {
    ^bb335(%2758: f32, %2759: f32, %2760: f32):
      %2761 = arith.mulf %2758, %2759 : f32
      linalg.yield %2761 : f32
    } -> tensor<1x1x256xf32>
    %2762 = tensor.empty() : tensor<1x1x256xf32>
    %2763 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22, %2757 : tensor<256xf32>, tensor<1x1x256xf32>) outs(%2762 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_83", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} {
    ^bb336(%2764: f32, %2765: f32, %2766: f32):
      %2767 = arith.mulf %2764, %2765 : f32
      linalg.yield %2767 : f32
    } -> tensor<1x1x256xf32>
    %2768 = tensor.empty() : tensor<256x1024xf32>
    %2769 = linalg.transpose ins(%24:tensor<1024x256xf32>) outs(%2768:tensor<256x1024xf32>) permutation = [1, 0]
    %2770 = tensor.empty() : tensor<1x1x1024xf32>
    %2771 = arith.constant {prov.module = "lm_head"} 0.000000e+00 : f32
    %2772 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm_head"} ins(%2771 : f32) outs(%2770 : tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
    %2773 = linalg.matmul {prov.region_id = "matmul_18", prov.dispatch_id = "matmul_18", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "lm_head", prov.fqn = "lm_head"} ins(%2763, %2769 : tensor<1x1x256xf32>, tensor<256x1024xf32>) outs(%2772 : tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
    %2774 = "tensor.extract_slice"(%2773) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_31", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
    %2775 = "tensor.extract_slice"(%2774) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<1x1x1024xf32>) -> tensor<1024xf32>
    %2776 = tensor.expand_shape %2775 [[0 : i64, 1 : i64]] output_shape [1, 1024] {prov.region_id = "slice_32", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1024xf32>
    %2777 = arith.constant {prov.region_id = "arg_reduce_14", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0xff800000 : f32
    %2778 = arith.constant {prov.region_id = "arg_reduce_14", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0 : i64
    %2779 = tensor.splat %2777 {prov.region_id = "arg_reduce_14", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xf32>
    %2780 = tensor.splat %2778 {prov.region_id = "arg_reduce_14", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xi64>
    %2781, %2782 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2776 : tensor<1x1024xf32>) outs(%2779, %2780 : tensor<1xf32>, tensor<1xi64>) attrs =  {prov.region_id = "arg_reduce_14", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} {
    ^bb337(%2783: f32, %2784: f32, %2785: i64):
      %2786 = linalg.index 1 : index
      %2787 = arith.index_cast %2786 : index to i64
      %2788 = arith.cmpf ogt, %2783, %2784 : f32
      %2789 = arith.select %2788, %2783, %2784 : f32
      %2790 = arith.select %2788, %2787, %2785 : i64
      linalg.yield %2789, %2790 : f32, i64
    } -> (tensor<1xf32>, tensor<1xi64>)
    %2791 = tensor.expand_shape %2781 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "arg_reduce_14", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xf32> into tensor<1x1xf32>
    %2792 = tensor.expand_shape %2782 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "arg_reduce_14", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xi64> into tensor<1x1xi64>
    %2793 = func.call @aten_zeros_default_1() {prov.region_id = "aten_zeros_default_1_1", prov.dispatch_id = "aten_zeros_default_1_1"} : () -> tensor<i64>
    %2794 = func.call @aten_zeros_default_2() {prov.region_id = "aten_zeros_default_2_0", prov.dispatch_id = "aten_zeros_default_2_0"} : () -> tensor<1x7xi64>
    %2795 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 0 : index
    %2796 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 7 : index
    %2797 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 1 : index
    %2798, %2799, %2800, %2801, %2802 = scf.for %2803 = %2795 to %2796 step %2797 iter_args(%2804 = %2793, %2805 = %2792, %2806 = %2794, %2807 = %2716, %2808 = %2717) -> (tensor<i64>, tensor<1x1xi64>, tensor<1x7xi64>, tensor<2x1x4x39x32xf32>, tensor<2x1x4x39x32xf32>) {
      %2809 = tensor.extract %2804[] : tensor<i64>
      %2810 = tensor.from_elements %2809 {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1xi64>
      %2811 = func.call @aten_index_copy_default_wl0(%2806, %2810, %2805) {prov.region_id = "aten_index_copy_default_0", prov.dispatch_id = "aten_index_copy_default_0"} : (tensor<1x7xi64>, tensor<1xi64>, tensor<1x1xi64>) -> tensor<1x7xi64>
      %2812 = tensor.empty() : tensor<i64>
      %2813 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%31, %2804 : tensor<i64>, tensor<i64>) outs(%2812 : tensor<i64>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb338(%2814: i64, %2815: i64, %2816: i64):
        %2817 = arith.addi %2814, %2815 : i64
        linalg.yield %2817 : i64
      } -> tensor<i64>
      %2818 = tensor.empty() : tensor<1x1x256xf32>
      %2819 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2805 : tensor<1x1xi64>) outs(%2818 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32"} {
      ^bb339(%2820: i64, %2821: f32):
        %2822 = arith.index_cast %2820 : i64 to index
        %2823 = linalg.index 2 : index
        %2824 = tensor.extract %23[%2822, %2823] : tensor<1024x256xf32>
        linalg.yield %2824 : f32
      } -> tensor<1x1x256xf32>
      %2825 = tensor.extract %2813[] : tensor<i64>
      %2826 = tensor.from_elements %2825 {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int64"} : tensor<1x1xi64>
      %2827 = tensor.empty() : tensor<1x1x256xf32>
      %2828 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2819 : tensor<1x1x256xf32>) outs(%2827 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb340(%2829: f32, %2830: f32):
        %2831 = arith.constant 2.000000e+00 : f32
        %2832 = math.powf %2829, %2831 : f32
        linalg.yield %2832 : f32
      } -> tensor<1x1x256xf32>
      %2833 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %2834 = tensor.splat %2833 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %2835 = linalg.reduce ins(%2828:tensor<1x1x256xf32>) outs(%2834:tensor<1x1xf32>) dimensions = [2]
      (%2836: f32, %2837: f32) {
        %2838 = arith.addf %2836, %2837 : f32
        linalg.yield %2838 : f32
      }
      %2839 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 2.560000e+02 : f32
      %2840 = tensor.splat %2839 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %2841 = tensor.empty() : tensor<1x1xf32>
      %2842 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2835, %2840 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%2841 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb341(%2843: f32, %2844: f32, %2845: f32):
        %2846 = arith.divf %2843, %2844 : f32
        linalg.yield %2846 : f32
      } -> tensor<1x1xf32>
      %2847 = tensor.collapse_shape %2842 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %2848 = tensor.expand_shape %2847 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %2849 = arith.constant {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %2850 = tensor.splat %2849 {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %2851 = tensor.empty() : tensor<1x1x1xf32>
      %2852 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2848, %2850 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%2851 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb342(%2853: f32, %2854: f32, %2855: f32):
        %2856 = arith.addf %2853, %2854 : f32
        linalg.yield %2856 : f32
      } -> tensor<1x1x1xf32>
      %2857 = tensor.empty() : tensor<1x1x1xf32>
      %2858 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2852 : tensor<1x1x1xf32>) outs(%2857 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb343(%2859: f32, %2860: f32):
        %2861 = math.rsqrt %2859 : f32
        linalg.yield %2861 : f32
      } -> tensor<1x1x1xf32>
      %2862 = tensor.empty() : tensor<1x1x256xf32>
      %2863 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2819, %2858 : tensor<1x1x256xf32>, tensor<1x1x1xf32>) outs(%2862 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb344(%2864: f32, %2865: f32, %2866: f32):
        %2867 = arith.mulf %2864, %2865 : f32
        linalg.yield %2867 : f32
      } -> tensor<1x1x256xf32>
      %2868 = tensor.empty() : tensor<1x1x256xf32>
      %2869 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %2863 : tensor<256xf32>, tensor<1x1x256xf32>) outs(%2868 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb345(%2870: f32, %2871: f32, %2872: f32):
        %2873 = arith.mulf %2870, %2871 : f32
        linalg.yield %2873 : f32
      } -> tensor<1x1x256xf32>
      %2874 = func.call @wrap_with_set_grad_enabled_wl1(%2869) {prov.region_id = "wrap_with_set_grad_enabled_0", prov.dispatch_id = "wrap_with_set_grad_enabled_0"} : (tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %2875 = func.call @wrap_with_set_grad_enabled_1_wl2(%0) {prov.region_id = "wrap_with_set_grad_enabled_1_0", prov.dispatch_id = "wrap_with_set_grad_enabled_1_0"} : (tensor<256x256xf32>) -> tensor<256x256xf32>
      %2876 = tensor.empty() : tensor<256x256xf32>
      %2877 = linalg.transpose ins(%2875:tensor<256x256xf32>) outs(%2876:tensor<256x256xf32>) permutation = [1, 0]
      %2878 = tensor.empty() : tensor<1x1x256xf32>
      %2879 = arith.constant 0.000000e+00 : f32
      %2880 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%2879 : f32) outs(%2878 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %2881 = linalg.matmul {prov.region_id = "matmul_0", prov.dispatch_id = "matmul_0", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%2874, %2877 : tensor<1x1x256xf32>, tensor<256x256xf32>) outs(%2880 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %2882 = tensor.collapse_shape %2881 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x256xf32> into tensor<256xf32>
      %2883 = tensor.expand_shape %2882 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x8x32xf32>
      %2884 = tensor.empty() : tensor<1x8x1x32xf32>
      %2885 = linalg.transpose ins(%2883:tensor<1x1x8x32xf32>) outs(%2884:tensor<1x8x1x32xf32>) permutation = [0, 2, 1, 3]
      %2886 = func.call @wrap_with_set_grad_enabled_2_wl3() {prov.region_id = "wrap_with_set_grad_enabled_2_0", prov.dispatch_id = "wrap_with_set_grad_enabled_2_0"} : () -> tensor<1x1x256xf32>
      %2887 = func.call @wrap_with_set_grad_enabled_3_wl4(%1) {prov.region_id = "wrap_with_set_grad_enabled_3_0", prov.dispatch_id = "wrap_with_set_grad_enabled_3_0"} : (tensor<128x256xf32>) -> tensor<128x256xf32>
      %2888 = tensor.empty() : tensor<256x128xf32>
      %2889 = linalg.transpose ins(%2887:tensor<128x256xf32>) outs(%2888:tensor<256x128xf32>) permutation = [1, 0]
      %2890 = tensor.empty() : tensor<1x1x128xf32>
      %2891 = arith.constant 0.000000e+00 : f32
      %2892 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%2891 : f32) outs(%2890 : tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
      %2893 = linalg.matmul {prov.region_id = "matmul_1", prov.dispatch_id = "matmul_1", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%2886, %2889 : tensor<1x1x256xf32>, tensor<256x128xf32>) outs(%2892 : tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
      %2894 = tensor.collapse_shape %2893 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %2895 = tensor.expand_shape %2894 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 32] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x4x32xf32>
      %2896 = tensor.empty() : tensor<1x4x1x32xf32>
      %2897 = linalg.transpose ins(%2895:tensor<1x1x4x32xf32>) outs(%2896:tensor<1x4x1x32xf32>) permutation = [0, 2, 1, 3]
      %2898 = func.call @wrap_with_set_grad_enabled_2_wl3() {prov.region_id = "wrap_with_set_grad_enabled_2_1", prov.dispatch_id = "wrap_with_set_grad_enabled_2_1"} : () -> tensor<1x1x256xf32>
      %2899 = func.call @wrap_with_set_grad_enabled_3_wl4(%2) {prov.region_id = "wrap_with_set_grad_enabled_3_1", prov.dispatch_id = "wrap_with_set_grad_enabled_3_1"} : (tensor<128x256xf32>) -> tensor<128x256xf32>
      %2900 = tensor.empty() : tensor<256x128xf32>
      %2901 = linalg.transpose ins(%2899:tensor<128x256xf32>) outs(%2900:tensor<256x128xf32>) permutation = [1, 0]
      %2902 = tensor.empty() : tensor<1x1x128xf32>
      %2903 = arith.constant 0.000000e+00 : f32
      %2904 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%2903 : f32) outs(%2902 : tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
      %2905 = linalg.matmul {prov.region_id = "matmul_2", prov.dispatch_id = "matmul_2", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%2898, %2901 : tensor<1x1x256xf32>, tensor<256x128xf32>) outs(%2904 : tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
      %2906 = tensor.collapse_shape %2905 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %2907 = tensor.expand_shape %2906 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 32] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x4x32xf32>
      %2908 = tensor.empty() : tensor<1x4x1x32xf32>
      %2909 = linalg.transpose ins(%2907:tensor<1x1x4x32xf32>) outs(%2908:tensor<1x4x1x32xf32>) permutation = [0, 2, 1, 3]
      %2910 = "tensor.extract_slice"(%26) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 39, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<2048x32xf32>) -> tensor<39x32xf32>
      %2911 = "tensor.extract_slice"(%27) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 39, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<2048x32xf32>) -> tensor<39x32xf32>
      %2912 = tensor.empty() : tensor<1x1x32xf32>
      %2913 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2826 : tensor<1x1xi64>) outs(%2912 : tensor<1x1x32xf32>) attrs =  {prov.region_id = "gather_1", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32"} {
      ^bb346(%2914: i64, %2915: f32):
        %2916 = arith.index_cast %2914 : i64 to index
        %2917 = linalg.index 2 : index
        %2918 = tensor.extract %2910[%2916, %2917] : tensor<39x32xf32>
        linalg.yield %2918 : f32
      } -> tensor<1x1x32xf32>
      %2919 = tensor.collapse_shape %2913 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x32xf32> into tensor<32xf32>
      %2920 = tensor.expand_shape %2919 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x1x32xf32>
      %2921 = tensor.empty() : tensor<1x1x32xf32>
      %2922 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2826 : tensor<1x1xi64>) outs(%2921 : tensor<1x1x32xf32>) attrs =  {prov.region_id = "gather_2", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32"} {
      ^bb347(%2923: i64, %2924: f32):
        %2925 = arith.index_cast %2923 : i64 to index
        %2926 = linalg.index 2 : index
        %2927 = tensor.extract %2911[%2925, %2926] : tensor<39x32xf32>
        linalg.yield %2927 : f32
      } -> tensor<1x1x32xf32>
      %2928 = tensor.collapse_shape %2922 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x32xf32> into tensor<32xf32>
      %2929 = tensor.expand_shape %2928 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x1x32xf32>
      %2930 = tensor.empty() : tensor<1x8x1x32xf32>
      %2931 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2885, %2920 : tensor<1x8x1x32xf32>, tensor<1x1x1x32xf32>) outs(%2930 : tensor<1x8x1x32xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb348(%2932: f32, %2933: f32, %2934: f32):
        %2935 = arith.mulf %2932, %2933 : f32
        linalg.yield %2935 : f32
      } -> tensor<1x8x1x32xf32>
      %2936 = "tensor.extract_slice"(%2885) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 1, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x1x32xf32>) -> tensor<1x8x1x16xf32>
      %2937 = "tensor.extract_slice"(%2885) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 8, 1, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x1x32xf32>) -> tensor<1x8x1x16xf32>
      %2938 = tensor.empty() : tensor<1x8x1x16xf32>
      %2939 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2937 : tensor<1x8x1x16xf32>) outs(%2938 : tensor<1x8x1x16xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb349(%2940: f32, %2941: f32):
        %2942 = arith.negf %2940 : f32
        linalg.yield %2942 : f32
      } -> tensor<1x8x1x16xf32>
      %2943 = tensor.concat dim(3) %2939, %2936 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x8x1x16xf32>, tensor<1x8x1x16xf32>) -> tensor<1x8x1x32xf32>
      %2944 = tensor.empty() : tensor<1x8x1x32xf32>
      %2945 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2943, %2929 : tensor<1x8x1x32xf32>, tensor<1x1x1x32xf32>) outs(%2944 : tensor<1x8x1x32xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb350(%2946: f32, %2947: f32, %2948: f32):
        %2949 = arith.mulf %2946, %2947 : f32
        linalg.yield %2949 : f32
      } -> tensor<1x8x1x32xf32>
      %2950 = tensor.empty() : tensor<1x8x1x32xf32>
      %2951 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2931, %2945 : tensor<1x8x1x32xf32>, tensor<1x8x1x32xf32>) outs(%2950 : tensor<1x8x1x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb351(%2952: f32, %2953: f32, %2954: f32):
        %2955 = arith.addf %2952, %2953 : f32
        linalg.yield %2955 : f32
      } -> tensor<1x8x1x32xf32>
      %2956 = tensor.empty() : tensor<1x4x1x32xf32>
      %2957 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2897, %2920 : tensor<1x4x1x32xf32>, tensor<1x1x1x32xf32>) outs(%2956 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb352(%2958: f32, %2959: f32, %2960: f32):
        %2961 = arith.mulf %2958, %2959 : f32
        linalg.yield %2961 : f32
      } -> tensor<1x4x1x32xf32>
      %2962 = "tensor.extract_slice"(%2897) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x32xf32>) -> tensor<1x4x1x16xf32>
      %2963 = "tensor.extract_slice"(%2897) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 1, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x32xf32>) -> tensor<1x4x1x16xf32>
      %2964 = tensor.empty() : tensor<1x4x1x16xf32>
      %2965 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2963 : tensor<1x4x1x16xf32>) outs(%2964 : tensor<1x4x1x16xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb353(%2966: f32, %2967: f32):
        %2968 = arith.negf %2966 : f32
        linalg.yield %2968 : f32
      } -> tensor<1x4x1x16xf32>
      %2969 = tensor.concat dim(3) %2965, %2962 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x1x16xf32>, tensor<1x4x1x16xf32>) -> tensor<1x4x1x32xf32>
      %2970 = tensor.empty() : tensor<1x4x1x32xf32>
      %2971 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2969, %2929 : tensor<1x4x1x32xf32>, tensor<1x1x1x32xf32>) outs(%2970 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb354(%2972: f32, %2973: f32, %2974: f32):
        %2975 = arith.mulf %2972, %2973 : f32
        linalg.yield %2975 : f32
      } -> tensor<1x4x1x32xf32>
      %2976 = tensor.empty() : tensor<1x4x1x32xf32>
      %2977 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2957, %2971 : tensor<1x4x1x32xf32>, tensor<1x4x1x32xf32>) outs(%2976 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb355(%2978: f32, %2979: f32, %2980: f32):
        %2981 = arith.addf %2978, %2979 : f32
        linalg.yield %2981 : f32
      } -> tensor<1x4x1x32xf32>
      %2982 = "tensor.extract_slice"(%2807) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x39x32xf32>) -> tensor<4x39x32xf32>
      %2983 = tensor.empty() : tensor<1xi64>
      %2984 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2813, %54 : tensor<i64>, tensor<1xi64>) outs(%2983 : tensor<1xi64>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb356(%2985: i64, %2986: i64, %2987: i64):
        %2988 = arith.addi %2985, %2986 : i64
        linalg.yield %2988 : i64
      } -> tensor<1xi64>
      %2989 = func.call @aten_index_copy_default_1_wl5(%2982, %2984, %2977) {prov.region_id = "aten_index_copy_default_1_0", prov.dispatch_id = "aten_index_copy_default_1_0"} : (tensor<4x39x32xf32>, tensor<1xi64>, tensor<1x4x1x32xf32>) -> tensor<1x4x39x32xf32>
      %2990 = "tensor.extract_slice"(%2808) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x39x32xf32>) -> tensor<4x39x32xf32>
      %2991 = tensor.empty() : tensor<1xi64>
      %2992 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2813, %54 : tensor<i64>, tensor<1xi64>) outs(%2991 : tensor<1xi64>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb357(%2993: i64, %2994: i64, %2995: i64):
        %2996 = arith.addi %2993, %2994 : i64
        linalg.yield %2996 : i64
      } -> tensor<1xi64>
      %2997 = func.call @aten_index_copy_default_1_wl5(%2990, %2992, %2909) {prov.region_id = "aten_index_copy_default_1_1", prov.dispatch_id = "aten_index_copy_default_1_1"} : (tensor<4x39x32xf32>, tensor<1xi64>, tensor<1x4x1x32xf32>) -> tensor<1x4x39x32xf32>
      %2998 = "tensor.extract_slice"(%2989) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x39x32xf32>) -> tensor<1x4x39x32xf32>
      %2999 = "tensor.extract_slice"(%2998) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x39x32xf32>) -> tensor<1x4x39x32xf32>
      %3000 = tensor.collapse_shape %2999 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x39x32xf32> into tensor<4992xf32>
      %3001 = tensor.expand_shape %3000 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 39, 32] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<4992xf32> into tensor<1x4x1x39x32xf32>
      %3002 = "tensor.extract_slice"(%3001) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x39x32xf32>) -> tensor<1x4x1x39x32xf32>
      %3003 = "tensor.extract_slice"(%3002) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_9", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x39x32xf32>) -> tensor<1x4x1x39x32xf32>
      %3004 = tensor.empty() : tensor<1x4x2x39x32xf32>
      %3005 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3003 : tensor<1x4x1x39x32xf32>) outs(%3004 : tensor<1x4x2x39x32xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb358(%3006: f32, %3007: f32):
        linalg.yield %3006 : f32
      } -> tensor<1x4x2x39x32xf32>
      %3008 = tensor.collapse_shape %3005 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x4x2x39x32xf32> into tensor<9984xf32>
      %3009 = tensor.expand_shape %3008 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 39, 32] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<9984xf32> into tensor<1x8x39x32xf32>
      %3010 = "tensor.extract_slice"(%2997) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_10", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x39x32xf32>) -> tensor<1x4x39x32xf32>
      %3011 = "tensor.extract_slice"(%3010) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_11", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x39x32xf32>) -> tensor<1x4x39x32xf32>
      %3012 = tensor.collapse_shape %3011 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x39x32xf32> into tensor<4992xf32>
      %3013 = tensor.expand_shape %3012 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 39, 32] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<4992xf32> into tensor<1x4x1x39x32xf32>
      %3014 = "tensor.extract_slice"(%3013) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_12", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x39x32xf32>) -> tensor<1x4x1x39x32xf32>
      %3015 = "tensor.extract_slice"(%3014) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_13", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x39x32xf32>) -> tensor<1x4x1x39x32xf32>
      %3016 = tensor.empty() : tensor<1x4x2x39x32xf32>
      %3017 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3015 : tensor<1x4x1x39x32xf32>) outs(%3016 : tensor<1x4x2x39x32xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb359(%3018: f32, %3019: f32):
        linalg.yield %3018 : f32
      } -> tensor<1x4x2x39x32xf32>
      %3020 = tensor.collapse_shape %3017 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x4x2x39x32xf32> into tensor<9984xf32>
      %3021 = tensor.expand_shape %3020 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 39, 32] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<9984xf32> into tensor<1x8x39x32xf32>
      %3022 = tensor.empty() : tensor<1x8x32x39xf32>
      %3023 = linalg.transpose ins(%3009:tensor<1x8x39x32xf32>) outs(%3022:tensor<1x8x32x39xf32>) permutation = [0, 1, 3, 2]
      %3024 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3025 = tensor.splat %3024 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x1x39xf32>
      %3026 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%2951, %3023 : tensor<1x8x1x32xf32>, tensor<1x8x32x39xf32>) outs(%3025 : tensor<1x8x1x39xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb360(%3027: f32, %3028: f32, %3029: f32):
        %3030 = arith.mulf %3027, %3028 : f32
        %3031 = arith.addf %3029, %3030 : f32
        linalg.yield %3031 : f32
      } -> tensor<1x8x1x39xf32>
      %3032 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 5.65685415 : f32
      %3033 = tensor.splat %3032 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1x39xf32>
      %3034 = tensor.empty() : tensor<1x8x1x39xf32>
      %3035 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3026, %3033 : tensor<1x8x1x39xf32>, tensor<1x8x1x39xf32>) outs(%3034 : tensor<1x8x1x39xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
      ^bb361(%3036: f32, %3037: f32, %3038: f32):
        %3039 = arith.divf %3036, %3037 : f32
        linalg.yield %3039 : f32
      } -> tensor<1x8x1x39xf32>
      %3040 = tensor.empty() : tensor<1xi64>
      %3041 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2813, %54 : tensor<i64>, tensor<1xi64>) outs(%3040 : tensor<1xi64>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb362(%3042: i64, %3043: i64, %3044: i64):
        %3045 = arith.addi %3042, %3043 : i64
        linalg.yield %3045 : i64
      } -> tensor<1xi64>
      %3046 = tensor.expand_shape %3041 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1xi64> into tensor<1x1xi64>
      %3047 = tensor.expand_shape %36 [[0 : i64, 1 : i64]] output_shape [1, 39] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<39xi64> into tensor<1x39xi64>
      %3048 = tensor.empty() : tensor<1x39xi1>
      %3049 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3047, %3046 : tensor<1x39xi64>, tensor<1x1xi64>) outs(%3048 : tensor<1x39xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
      ^bb363(%3050: i64, %3051: i64, %3052: i1):
        %3053 = arith.cmpi sle, %3050, %3051 : i64
        linalg.yield %3053 : i1
      } -> tensor<1x39xi1>
      %3054 = tensor.collapse_shape %3049 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x39xi1> into tensor<39xi1>
      %3055 = tensor.expand_shape %3054 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 39] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<39xi1> into tensor<1x1x39xi1>
      %3056 = tensor.collapse_shape %3055 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x1x39xi1> into tensor<39xi1>
      %3057 = tensor.expand_shape %3056 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 39] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<39xi1> into tensor<1x1x1x39xi1>
      %3058 = tensor.empty() : tensor<1x1x1x39xi1>
      %3059 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3057 : tensor<1x1x1x39xi1>) outs(%3058 : tensor<1x1x1x39xi1>) attrs =  {prov.region_id = "bitwise_0", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
      ^bb364(%3060: i1, %3061: i1):
        %3062 = arith.constant true
        %3063 = arith.xori %3060, %3062 : i1
        linalg.yield %3063 : i1
      } -> tensor<1x1x1x39xi1>
      %3064 = func.call @aten_masked_fill_Scalar_wl6(%3035, %3059) {prov.region_id = "aten_masked_fill_Scalar_0", prov.dispatch_id = "aten_masked_fill_Scalar_0"} : (tensor<1x8x1x39xf32>, tensor<1x1x1x39xi1>) -> tensor<1x8x1x39xf32>
      %3065 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0xff800000 : f32
      %3066 = tensor.splat %3065 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
      %3067 = linalg.reduce ins(%3064:tensor<1x8x1x39xf32>) outs(%3066:tensor<1x8x1xf32>) dimensions = [3]
      (%3068: f32, %3069: f32) {
        %3070 = arith.maximumf %3068, %3069 : f32
        linalg.yield %3070 : f32
      }
      %3071 = tensor.collapse_shape %3067 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x1xf32> into tensor<8xf32>
      %3072 = tensor.expand_shape %3071 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 1, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1x1xf32>
      %3073 = tensor.empty() : tensor<1x8x1x39xf32>
      %3074 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3064, %3072 : tensor<1x8x1x39xf32>, tensor<1x8x1x1xf32>) outs(%3073 : tensor<1x8x1x39xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb365(%3075: f32, %3076: f32, %3077: f32):
        %3078 = arith.subf %3075, %3076 : f32
        linalg.yield %3078 : f32
      } -> tensor<1x8x1x39xf32>
      %3079 = tensor.empty() : tensor<1x8x1x39xf32>
      %3080 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3074 : tensor<1x8x1x39xf32>) outs(%3079 : tensor<1x8x1x39xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb366(%3081: f32, %3082: f32):
        %3083 = math.exp %3081 : f32
        linalg.yield %3083 : f32
      } -> tensor<1x8x1x39xf32>
      %3084 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3085 = tensor.splat %3084 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
      %3086 = linalg.reduce ins(%3080:tensor<1x8x1x39xf32>) outs(%3085:tensor<1x8x1xf32>) dimensions = [3]
      (%3087: f32, %3088: f32) {
        %3089 = arith.addf %3087, %3088 : f32
        linalg.yield %3089 : f32
      }
      %3090 = tensor.collapse_shape %3086 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x1xf32> into tensor<8xf32>
      %3091 = tensor.expand_shape %3090 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 1, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1x1xf32>
      %3092 = tensor.empty() : tensor<1x8x1x39xf32>
      %3093 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3080, %3091 : tensor<1x8x1x39xf32>, tensor<1x8x1x1xf32>) outs(%3092 : tensor<1x8x1x39xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb367(%3094: f32, %3095: f32, %3096: f32):
        %3097 = arith.divf %3094, %3095 : f32
        linalg.yield %3097 : f32
      } -> tensor<1x8x1x39xf32>
      %3098 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3099 = tensor.splat %3098 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x1x32xf32>
      %3100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%3093, %3021 : tensor<1x8x1x39xf32>, tensor<1x8x39x32xf32>) outs(%3099 : tensor<1x8x1x32xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb368(%3101: f32, %3102: f32, %3103: f32):
        %3104 = arith.mulf %3101, %3102 : f32
        %3105 = arith.addf %3103, %3104 : f32
        linalg.yield %3105 : f32
      } -> tensor<1x8x1x32xf32>
      %3106 = tensor.empty() : tensor<1x1x8x32xf32>
      %3107 = linalg.transpose ins(%3100:tensor<1x8x1x32xf32>) outs(%3106:tensor<1x1x8x32xf32>) permutation = [0, 2, 1, 3]
      %3108 = tensor.collapse_shape %3107 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x1x8x32xf32> into tensor<256xf32>
      %3109 = tensor.expand_shape %3108 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 256] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x256xf32>
      %3110 = tensor.empty() : tensor<1x1x256xf32>
      %3111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3109 : tensor<1x1x256xf32>) outs(%3110 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb369(%3112: f32, %3113: f32):
        %3114 = arith.constant 2.000000e+00 : f32
        %3115 = math.powf %3112, %3114 : f32
        linalg.yield %3115 : f32
      } -> tensor<1x1x256xf32>
      %3116 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3117 = tensor.splat %3116 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3118 = linalg.reduce ins(%3111:tensor<1x1x256xf32>) outs(%3117:tensor<1x1xf32>) dimensions = [2]
      (%3119: f32, %3120: f32) {
        %3121 = arith.addf %3119, %3120 : f32
        linalg.yield %3121 : f32
      }
      %3122 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 2.560000e+02 : f32
      %3123 = tensor.splat %3122 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3124 = tensor.empty() : tensor<1x1xf32>
      %3125 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3118, %3123 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%3124 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb370(%3126: f32, %3127: f32, %3128: f32):
        %3129 = arith.divf %3126, %3127 : f32
        linalg.yield %3129 : f32
      } -> tensor<1x1xf32>
      %3130 = tensor.collapse_shape %3125 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %3131 = tensor.expand_shape %3130 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %3132 = arith.constant {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %3133 = tensor.splat %3132 {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %3134 = tensor.empty() : tensor<1x1x1xf32>
      %3135 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3131, %3133 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%3134 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb371(%3136: f32, %3137: f32, %3138: f32):
        %3139 = arith.addf %3136, %3137 : f32
        linalg.yield %3139 : f32
      } -> tensor<1x1x1xf32>
      %3140 = tensor.empty() : tensor<1x1x1xf32>
      %3141 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3135 : tensor<1x1x1xf32>) outs(%3140 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb372(%3142: f32, %3143: f32):
        %3144 = math.rsqrt %3142 : f32
        linalg.yield %3144 : f32
      } -> tensor<1x1x1xf32>
      %3145 = tensor.empty() : tensor<1x1x256xf32>
      %3146 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3109, %3141 : tensor<1x1x256xf32>, tensor<1x1x1xf32>) outs(%3145 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb373(%3147: f32, %3148: f32, %3149: f32):
        %3150 = arith.mulf %3147, %3148 : f32
        linalg.yield %3150 : f32
      } -> tensor<1x1x256xf32>
      %3151 = tensor.empty() : tensor<1x1x256xf32>
      %3152 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %3146 : tensor<256xf32>, tensor<1x1x256xf32>) outs(%3151 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb374(%3153: f32, %3154: f32, %3155: f32):
        %3156 = arith.mulf %3153, %3154 : f32
        linalg.yield %3156 : f32
      } -> tensor<1x1x256xf32>
      %3157 = func.call @wrap_with_set_grad_enabled_wl1(%3152) {prov.region_id = "wrap_with_set_grad_enabled_1", prov.dispatch_id = "wrap_with_set_grad_enabled_1"} : (tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %3158 = func.call @wrap_with_set_grad_enabled_1_wl2(%3) {prov.region_id = "wrap_with_set_grad_enabled_1_1", prov.dispatch_id = "wrap_with_set_grad_enabled_1_1"} : (tensor<256x256xf32>) -> tensor<256x256xf32>
      %3159 = tensor.empty() : tensor<256x256xf32>
      %3160 = linalg.transpose ins(%3158:tensor<256x256xf32>) outs(%3159:tensor<256x256xf32>) permutation = [1, 0]
      %3161 = tensor.empty() : tensor<1x1x256xf32>
      %3162 = arith.constant 0.000000e+00 : f32
      %3163 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%3162 : f32) outs(%3161 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %3164 = linalg.matmul {prov.region_id = "matmul_5", prov.dispatch_id = "matmul_5", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%3157, %3160 : tensor<1x1x256xf32>, tensor<256x256xf32>) outs(%3163 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %3165 = tensor.empty() : tensor<1x1x256xf32>
      %3166 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2819, %3164 : tensor<1x1x256xf32>, tensor<1x1x256xf32>) outs(%3165 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb375(%3167: f32, %3168: f32, %3169: f32):
        %3170 = arith.addf %3167, %3168 : f32
        linalg.yield %3170 : f32
      } -> tensor<1x1x256xf32>
      %3171 = tensor.empty() : tensor<1x1x256xf32>
      %3172 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3166 : tensor<1x1x256xf32>) outs(%3171 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb376(%3173: f32, %3174: f32):
        %3175 = arith.constant 2.000000e+00 : f32
        %3176 = math.powf %3173, %3175 : f32
        linalg.yield %3176 : f32
      } -> tensor<1x1x256xf32>
      %3177 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3178 = tensor.splat %3177 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3179 = linalg.reduce ins(%3172:tensor<1x1x256xf32>) outs(%3178:tensor<1x1xf32>) dimensions = [2]
      (%3180: f32, %3181: f32) {
        %3182 = arith.addf %3180, %3181 : f32
        linalg.yield %3182 : f32
      }
      %3183 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 2.560000e+02 : f32
      %3184 = tensor.splat %3183 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3185 = tensor.empty() : tensor<1x1xf32>
      %3186 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3179, %3184 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%3185 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb377(%3187: f32, %3188: f32, %3189: f32):
        %3190 = arith.divf %3187, %3188 : f32
        linalg.yield %3190 : f32
      } -> tensor<1x1xf32>
      %3191 = tensor.collapse_shape %3186 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %3192 = tensor.expand_shape %3191 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %3193 = arith.constant {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %3194 = tensor.splat %3193 {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %3195 = tensor.empty() : tensor<1x1x1xf32>
      %3196 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3192, %3194 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%3195 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb378(%3197: f32, %3198: f32, %3199: f32):
        %3200 = arith.addf %3197, %3198 : f32
        linalg.yield %3200 : f32
      } -> tensor<1x1x1xf32>
      %3201 = tensor.empty() : tensor<1x1x1xf32>
      %3202 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3196 : tensor<1x1x1xf32>) outs(%3201 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb379(%3203: f32, %3204: f32):
        %3205 = math.rsqrt %3203 : f32
        linalg.yield %3205 : f32
      } -> tensor<1x1x1xf32>
      %3206 = tensor.empty() : tensor<1x1x256xf32>
      %3207 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3166, %3202 : tensor<1x1x256xf32>, tensor<1x1x1xf32>) outs(%3206 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb380(%3208: f32, %3209: f32, %3210: f32):
        %3211 = arith.mulf %3208, %3209 : f32
        linalg.yield %3211 : f32
      } -> tensor<1x1x256xf32>
      %3212 = tensor.empty() : tensor<1x1x256xf32>
      %3213 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %3207 : tensor<256xf32>, tensor<1x1x256xf32>) outs(%3212 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb381(%3214: f32, %3215: f32, %3216: f32):
        %3217 = arith.mulf %3214, %3215 : f32
        linalg.yield %3217 : f32
      } -> tensor<1x1x256xf32>
      %3218 = func.call @wrap_with_set_grad_enabled_wl1(%3213) {prov.region_id = "wrap_with_set_grad_enabled_2", prov.dispatch_id = "wrap_with_set_grad_enabled_2"} : (tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %3219 = func.call @wrap_with_set_grad_enabled_4_wl7(%5) {prov.region_id = "wrap_with_set_grad_enabled_4_0", prov.dispatch_id = "wrap_with_set_grad_enabled_4_0"} : (tensor<512x256xf32>) -> tensor<512x256xf32>
      %3220 = tensor.empty() : tensor<256x512xf32>
      %3221 = linalg.transpose ins(%3219:tensor<512x256xf32>) outs(%3220:tensor<256x512xf32>) permutation = [1, 0]
      %3222 = tensor.empty() : tensor<1x1x512xf32>
      %3223 = arith.constant 0.000000e+00 : f32
      %3224 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%3223 : f32) outs(%3222 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %3225 = linalg.matmul {prov.region_id = "matmul_6", prov.dispatch_id = "matmul_6", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%3218, %3221 : tensor<1x1x256xf32>, tensor<256x512xf32>) outs(%3224 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %3226 = tensor.empty() : tensor<1x1x512xf32>
      %3227 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3225 : tensor<1x1x512xf32>) outs(%3226 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "minmax_0", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu.default", prov.orig_dtype = "float32"} {
      ^bb382(%3228: f32, %3229: f32):
        %3230 = arith.constant 0.000000e+00 : f32
        %3231 = arith.maximumf %3228, %3230 : f32
        linalg.yield %3231 : f32
      } -> tensor<1x1x512xf32>
      %3232 = tensor.empty() : tensor<1x1x512xf32>
      %3233 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3227 : tensor<1x1x512xf32>) outs(%3232 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb383(%3234: f32, %3235: f32):
        %3236 = arith.constant 2.000000e+00 : f32
        %3237 = math.powf %3234, %3236 : f32
        linalg.yield %3237 : f32
      } -> tensor<1x1x512xf32>
      %3238 = func.call @wrap_with_set_grad_enabled_2_wl3() {prov.region_id = "wrap_with_set_grad_enabled_2_2", prov.dispatch_id = "wrap_with_set_grad_enabled_2_2"} : () -> tensor<1x1x256xf32>
      %3239 = func.call @wrap_with_set_grad_enabled_4_wl7(%6) {prov.region_id = "wrap_with_set_grad_enabled_4_1", prov.dispatch_id = "wrap_with_set_grad_enabled_4_1"} : (tensor<512x256xf32>) -> tensor<512x256xf32>
      %3240 = tensor.empty() : tensor<256x512xf32>
      %3241 = linalg.transpose ins(%3239:tensor<512x256xf32>) outs(%3240:tensor<256x512xf32>) permutation = [1, 0]
      %3242 = tensor.empty() : tensor<1x1x512xf32>
      %3243 = arith.constant 0.000000e+00 : f32
      %3244 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%3243 : f32) outs(%3242 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %3245 = linalg.matmul {prov.region_id = "matmul_7", prov.dispatch_id = "matmul_7", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%3238, %3241 : tensor<1x1x256xf32>, tensor<256x512xf32>) outs(%3244 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %3246 = tensor.empty() : tensor<1x1x512xf32>
      %3247 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3233, %3245 : tensor<1x1x512xf32>, tensor<1x1x512xf32>) outs(%3246 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb384(%3248: f32, %3249: f32, %3250: f32):
        %3251 = arith.mulf %3248, %3249 : f32
        linalg.yield %3251 : f32
      } -> tensor<1x1x512xf32>
      %3252 = tensor.empty() : tensor<1x1x512xf32>
      %3253 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3247 : tensor<1x1x512xf32>) outs(%3252 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb385(%3254: f32, %3255: f32):
        %3256 = arith.constant 2.000000e+00 : f32
        %3257 = math.powf %3254, %3256 : f32
        linalg.yield %3257 : f32
      } -> tensor<1x1x512xf32>
      %3258 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3259 = tensor.splat %3258 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3260 = linalg.reduce ins(%3253:tensor<1x1x512xf32>) outs(%3259:tensor<1x1xf32>) dimensions = [2]
      (%3261: f32, %3262: f32) {
        %3263 = arith.addf %3261, %3262 : f32
        linalg.yield %3263 : f32
      }
      %3264 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 5.120000e+02 : f32
      %3265 = tensor.splat %3264 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3266 = tensor.empty() : tensor<1x1xf32>
      %3267 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3260, %3265 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%3266 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb386(%3268: f32, %3269: f32, %3270: f32):
        %3271 = arith.divf %3268, %3269 : f32
        linalg.yield %3271 : f32
      } -> tensor<1x1xf32>
      %3272 = tensor.collapse_shape %3267 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %3273 = tensor.expand_shape %3272 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %3274 = arith.constant {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %3275 = tensor.splat %3274 {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %3276 = tensor.empty() : tensor<1x1x1xf32>
      %3277 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3273, %3275 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%3276 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb387(%3278: f32, %3279: f32, %3280: f32):
        %3281 = arith.addf %3278, %3279 : f32
        linalg.yield %3281 : f32
      } -> tensor<1x1x1xf32>
      %3282 = tensor.empty() : tensor<1x1x1xf32>
      %3283 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3277 : tensor<1x1x1xf32>) outs(%3282 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb388(%3284: f32, %3285: f32):
        %3286 = math.rsqrt %3284 : f32
        linalg.yield %3286 : f32
      } -> tensor<1x1x1xf32>
      %3287 = tensor.empty() : tensor<1x1x512xf32>
      %3288 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3247, %3283 : tensor<1x1x512xf32>, tensor<1x1x1xf32>) outs(%3287 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb389(%3289: f32, %3290: f32, %3291: f32):
        %3292 = arith.mulf %3289, %3290 : f32
        linalg.yield %3292 : f32
      } -> tensor<1x1x512xf32>
      %3293 = tensor.empty() : tensor<1x1x512xf32>
      %3294 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %3288 : tensor<512xf32>, tensor<1x1x512xf32>) outs(%3293 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb390(%3295: f32, %3296: f32, %3297: f32):
        %3298 = arith.mulf %3295, %3296 : f32
        linalg.yield %3298 : f32
      } -> tensor<1x1x512xf32>
      %3299 = func.call @wrap_with_set_grad_enabled_5_wl8(%3294) {prov.region_id = "wrap_with_set_grad_enabled_5_0", prov.dispatch_id = "wrap_with_set_grad_enabled_5_0"} : (tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %3300 = func.call @wrap_with_set_grad_enabled_6_wl9(%7) {prov.region_id = "wrap_with_set_grad_enabled_6_0", prov.dispatch_id = "wrap_with_set_grad_enabled_6_0"} : (tensor<256x512xf32>) -> tensor<256x512xf32>
      %3301 = tensor.empty() : tensor<512x256xf32>
      %3302 = linalg.transpose ins(%3300:tensor<256x512xf32>) outs(%3301:tensor<512x256xf32>) permutation = [1, 0]
      %3303 = tensor.empty() : tensor<1x1x256xf32>
      %3304 = arith.constant 0.000000e+00 : f32
      %3305 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%3304 : f32) outs(%3303 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %3306 = linalg.matmul {prov.region_id = "matmul_8", prov.dispatch_id = "matmul_8", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%3299, %3302 : tensor<1x1x512xf32>, tensor<512x256xf32>) outs(%3305 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %3307 = tensor.empty() : tensor<1x1x256xf32>
      %3308 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3166, %3306 : tensor<1x1x256xf32>, tensor<1x1x256xf32>) outs(%3307 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb391(%3309: f32, %3310: f32, %3311: f32):
        %3312 = arith.addf %3309, %3310 : f32
        linalg.yield %3312 : f32
      } -> tensor<1x1x256xf32>
      %3313 = tensor.empty() : tensor<1x1x256xf32>
      %3314 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3308 : tensor<1x1x256xf32>) outs(%3313 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb392(%3315: f32, %3316: f32):
        %3317 = arith.constant 2.000000e+00 : f32
        %3318 = math.powf %3315, %3317 : f32
        linalg.yield %3318 : f32
      } -> tensor<1x1x256xf32>
      %3319 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3320 = tensor.splat %3319 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3321 = linalg.reduce ins(%3314:tensor<1x1x256xf32>) outs(%3320:tensor<1x1xf32>) dimensions = [2]
      (%3322: f32, %3323: f32) {
        %3324 = arith.addf %3322, %3323 : f32
        linalg.yield %3324 : f32
      }
      %3325 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 2.560000e+02 : f32
      %3326 = tensor.splat %3325 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3327 = tensor.empty() : tensor<1x1xf32>
      %3328 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3321, %3326 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%3327 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb393(%3329: f32, %3330: f32, %3331: f32):
        %3332 = arith.divf %3329, %3330 : f32
        linalg.yield %3332 : f32
      } -> tensor<1x1xf32>
      %3333 = tensor.collapse_shape %3328 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %3334 = tensor.expand_shape %3333 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %3335 = arith.constant {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %3336 = tensor.splat %3335 {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %3337 = tensor.empty() : tensor<1x1x1xf32>
      %3338 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3334, %3336 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%3337 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb394(%3339: f32, %3340: f32, %3341: f32):
        %3342 = arith.addf %3339, %3340 : f32
        linalg.yield %3342 : f32
      } -> tensor<1x1x1xf32>
      %3343 = tensor.empty() : tensor<1x1x1xf32>
      %3344 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3338 : tensor<1x1x1xf32>) outs(%3343 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb395(%3345: f32, %3346: f32):
        %3347 = math.rsqrt %3345 : f32
        linalg.yield %3347 : f32
      } -> tensor<1x1x1xf32>
      %3348 = tensor.empty() : tensor<1x1x256xf32>
      %3349 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3308, %3344 : tensor<1x1x256xf32>, tensor<1x1x1xf32>) outs(%3348 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb396(%3350: f32, %3351: f32, %3352: f32):
        %3353 = arith.mulf %3350, %3351 : f32
        linalg.yield %3353 : f32
      } -> tensor<1x1x256xf32>
      %3354 = tensor.empty() : tensor<1x1x256xf32>
      %3355 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20, %3349 : tensor<256xf32>, tensor<1x1x256xf32>) outs(%3354 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb397(%3356: f32, %3357: f32, %3358: f32):
        %3359 = arith.mulf %3356, %3357 : f32
        linalg.yield %3359 : f32
      } -> tensor<1x1x256xf32>
      %3360 = func.call @wrap_with_set_grad_enabled_wl1(%3355) {prov.region_id = "wrap_with_set_grad_enabled_3", prov.dispatch_id = "wrap_with_set_grad_enabled_3"} : (tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %3361 = func.call @wrap_with_set_grad_enabled_1_wl2(%11) {prov.region_id = "wrap_with_set_grad_enabled_1_2", prov.dispatch_id = "wrap_with_set_grad_enabled_1_2"} : (tensor<256x256xf32>) -> tensor<256x256xf32>
      %3362 = tensor.empty() : tensor<256x256xf32>
      %3363 = linalg.transpose ins(%3361:tensor<256x256xf32>) outs(%3362:tensor<256x256xf32>) permutation = [1, 0]
      %3364 = tensor.empty() : tensor<1x1x256xf32>
      %3365 = arith.constant 0.000000e+00 : f32
      %3366 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%3365 : f32) outs(%3364 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %3367 = linalg.matmul {prov.region_id = "matmul_9", prov.dispatch_id = "matmul_9", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%3360, %3363 : tensor<1x1x256xf32>, tensor<256x256xf32>) outs(%3366 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %3368 = tensor.collapse_shape %3367 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x256xf32> into tensor<256xf32>
      %3369 = tensor.expand_shape %3368 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 32] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x8x32xf32>
      %3370 = tensor.empty() : tensor<1x8x1x32xf32>
      %3371 = linalg.transpose ins(%3369:tensor<1x1x8x32xf32>) outs(%3370:tensor<1x8x1x32xf32>) permutation = [0, 2, 1, 3]
      %3372 = func.call @wrap_with_set_grad_enabled_2_wl3() {prov.region_id = "wrap_with_set_grad_enabled_2_3", prov.dispatch_id = "wrap_with_set_grad_enabled_2_3"} : () -> tensor<1x1x256xf32>
      %3373 = func.call @wrap_with_set_grad_enabled_3_wl4(%12) {prov.region_id = "wrap_with_set_grad_enabled_3_2", prov.dispatch_id = "wrap_with_set_grad_enabled_3_2"} : (tensor<128x256xf32>) -> tensor<128x256xf32>
      %3374 = tensor.empty() : tensor<256x128xf32>
      %3375 = linalg.transpose ins(%3373:tensor<128x256xf32>) outs(%3374:tensor<256x128xf32>) permutation = [1, 0]
      %3376 = tensor.empty() : tensor<1x1x128xf32>
      %3377 = arith.constant 0.000000e+00 : f32
      %3378 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%3377 : f32) outs(%3376 : tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
      %3379 = linalg.matmul {prov.region_id = "matmul_10", prov.dispatch_id = "matmul_10", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%3372, %3375 : tensor<1x1x256xf32>, tensor<256x128xf32>) outs(%3378 : tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
      %3380 = tensor.collapse_shape %3379 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %3381 = tensor.expand_shape %3380 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 32] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x4x32xf32>
      %3382 = tensor.empty() : tensor<1x4x1x32xf32>
      %3383 = linalg.transpose ins(%3381:tensor<1x1x4x32xf32>) outs(%3382:tensor<1x4x1x32xf32>) permutation = [0, 2, 1, 3]
      %3384 = func.call @wrap_with_set_grad_enabled_2_wl3() {prov.region_id = "wrap_with_set_grad_enabled_2_4", prov.dispatch_id = "wrap_with_set_grad_enabled_2_4"} : () -> tensor<1x1x256xf32>
      %3385 = func.call @wrap_with_set_grad_enabled_3_wl4(%13) {prov.region_id = "wrap_with_set_grad_enabled_3_3", prov.dispatch_id = "wrap_with_set_grad_enabled_3_3"} : (tensor<128x256xf32>) -> tensor<128x256xf32>
      %3386 = tensor.empty() : tensor<256x128xf32>
      %3387 = linalg.transpose ins(%3385:tensor<128x256xf32>) outs(%3386:tensor<256x128xf32>) permutation = [1, 0]
      %3388 = tensor.empty() : tensor<1x1x128xf32>
      %3389 = arith.constant 0.000000e+00 : f32
      %3390 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%3389 : f32) outs(%3388 : tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
      %3391 = linalg.matmul {prov.region_id = "matmul_11", prov.dispatch_id = "matmul_11", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%3384, %3387 : tensor<1x1x256xf32>, tensor<256x128xf32>) outs(%3390 : tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
      %3392 = tensor.collapse_shape %3391 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %3393 = tensor.expand_shape %3392 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 32] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x4x32xf32>
      %3394 = tensor.empty() : tensor<1x4x1x32xf32>
      %3395 = linalg.transpose ins(%3393:tensor<1x1x4x32xf32>) outs(%3394:tensor<1x4x1x32xf32>) permutation = [0, 2, 1, 3]
      %3396 = "tensor.extract_slice"(%29) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 39, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_14", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<2048x32xf32>) -> tensor<39x32xf32>
      %3397 = "tensor.extract_slice"(%30) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 39, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_15", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<2048x32xf32>) -> tensor<39x32xf32>
      %3398 = tensor.empty() : tensor<1x1x32xf32>
      %3399 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2826 : tensor<1x1xi64>) outs(%3398 : tensor<1x1x32xf32>) attrs =  {prov.region_id = "gather_3", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32"} {
      ^bb398(%3400: i64, %3401: f32):
        %3402 = arith.index_cast %3400 : i64 to index
        %3403 = linalg.index 2 : index
        %3404 = tensor.extract %3396[%3402, %3403] : tensor<39x32xf32>
        linalg.yield %3404 : f32
      } -> tensor<1x1x32xf32>
      %3405 = tensor.collapse_shape %3399 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x32xf32> into tensor<32xf32>
      %3406 = tensor.expand_shape %3405 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x1x32xf32>
      %3407 = tensor.empty() : tensor<1x1x32xf32>
      %3408 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2826 : tensor<1x1xi64>) outs(%3407 : tensor<1x1x32xf32>) attrs =  {prov.region_id = "gather_4", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32"} {
      ^bb399(%3409: i64, %3410: f32):
        %3411 = arith.index_cast %3409 : i64 to index
        %3412 = linalg.index 2 : index
        %3413 = tensor.extract %3397[%3411, %3412] : tensor<39x32xf32>
        linalg.yield %3413 : f32
      } -> tensor<1x1x32xf32>
      %3414 = tensor.collapse_shape %3408 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x32xf32> into tensor<32xf32>
      %3415 = tensor.expand_shape %3414 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x1x1x32xf32>
      %3416 = tensor.empty() : tensor<1x8x1x32xf32>
      %3417 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3371, %3406 : tensor<1x8x1x32xf32>, tensor<1x1x1x32xf32>) outs(%3416 : tensor<1x8x1x32xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb400(%3418: f32, %3419: f32, %3420: f32):
        %3421 = arith.mulf %3418, %3419 : f32
        linalg.yield %3421 : f32
      } -> tensor<1x8x1x32xf32>
      %3422 = "tensor.extract_slice"(%3371) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 1, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_16", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x1x32xf32>) -> tensor<1x8x1x16xf32>
      %3423 = "tensor.extract_slice"(%3371) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 8, 1, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_17", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x1x32xf32>) -> tensor<1x8x1x16xf32>
      %3424 = tensor.empty() : tensor<1x8x1x16xf32>
      %3425 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3423 : tensor<1x8x1x16xf32>) outs(%3424 : tensor<1x8x1x16xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb401(%3426: f32, %3427: f32):
        %3428 = arith.negf %3426 : f32
        linalg.yield %3428 : f32
      } -> tensor<1x8x1x16xf32>
      %3429 = tensor.concat dim(3) %3425, %3422 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x8x1x16xf32>, tensor<1x8x1x16xf32>) -> tensor<1x8x1x32xf32>
      %3430 = tensor.empty() : tensor<1x8x1x32xf32>
      %3431 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3429, %3415 : tensor<1x8x1x32xf32>, tensor<1x1x1x32xf32>) outs(%3430 : tensor<1x8x1x32xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb402(%3432: f32, %3433: f32, %3434: f32):
        %3435 = arith.mulf %3432, %3433 : f32
        linalg.yield %3435 : f32
      } -> tensor<1x8x1x32xf32>
      %3436 = tensor.empty() : tensor<1x8x1x32xf32>
      %3437 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3417, %3431 : tensor<1x8x1x32xf32>, tensor<1x8x1x32xf32>) outs(%3436 : tensor<1x8x1x32xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb403(%3438: f32, %3439: f32, %3440: f32):
        %3441 = arith.addf %3438, %3439 : f32
        linalg.yield %3441 : f32
      } -> tensor<1x8x1x32xf32>
      %3442 = tensor.empty() : tensor<1x4x1x32xf32>
      %3443 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3383, %3406 : tensor<1x4x1x32xf32>, tensor<1x1x1x32xf32>) outs(%3442 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb404(%3444: f32, %3445: f32, %3446: f32):
        %3447 = arith.mulf %3444, %3445 : f32
        linalg.yield %3447 : f32
      } -> tensor<1x4x1x32xf32>
      %3448 = "tensor.extract_slice"(%3383) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_18", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x32xf32>) -> tensor<1x4x1x16xf32>
      %3449 = "tensor.extract_slice"(%3383) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 1, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_19", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x32xf32>) -> tensor<1x4x1x16xf32>
      %3450 = tensor.empty() : tensor<1x4x1x16xf32>
      %3451 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3449 : tensor<1x4x1x16xf32>) outs(%3450 : tensor<1x4x1x16xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb405(%3452: f32, %3453: f32):
        %3454 = arith.negf %3452 : f32
        linalg.yield %3454 : f32
      } -> tensor<1x4x1x16xf32>
      %3455 = tensor.concat dim(3) %3451, %3448 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x1x16xf32>, tensor<1x4x1x16xf32>) -> tensor<1x4x1x32xf32>
      %3456 = tensor.empty() : tensor<1x4x1x32xf32>
      %3457 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3455, %3415 : tensor<1x4x1x32xf32>, tensor<1x1x1x32xf32>) outs(%3456 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb406(%3458: f32, %3459: f32, %3460: f32):
        %3461 = arith.mulf %3458, %3459 : f32
        linalg.yield %3461 : f32
      } -> tensor<1x4x1x32xf32>
      %3462 = tensor.empty() : tensor<1x4x1x32xf32>
      %3463 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3443, %3457 : tensor<1x4x1x32xf32>, tensor<1x4x1x32xf32>) outs(%3462 : tensor<1x4x1x32xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb407(%3464: f32, %3465: f32, %3466: f32):
        %3467 = arith.addf %3464, %3465 : f32
        linalg.yield %3467 : f32
      } -> tensor<1x4x1x32xf32>
      %3468 = "tensor.extract_slice"(%2807) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x39x32xf32>) -> tensor<4x39x32xf32>
      %3469 = tensor.empty() : tensor<1xi64>
      %3470 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2813, %54 : tensor<i64>, tensor<1xi64>) outs(%3469 : tensor<1xi64>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb408(%3471: i64, %3472: i64, %3473: i64):
        %3474 = arith.addi %3471, %3472 : i64
        linalg.yield %3474 : i64
      } -> tensor<1xi64>
      %3475 = func.call @aten_index_copy_default_1_wl5(%3468, %3470, %3463) {prov.region_id = "aten_index_copy_default_1_2", prov.dispatch_id = "aten_index_copy_default_1_2"} : (tensor<4x39x32xf32>, tensor<1xi64>, tensor<1x4x1x32xf32>) -> tensor<1x4x39x32xf32>
      %3476 = "tensor.extract_slice"(%2808) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<2x1x4x39x32xf32>) -> tensor<4x39x32xf32>
      %3477 = tensor.empty() : tensor<1xi64>
      %3478 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2813, %54 : tensor<i64>, tensor<1xi64>) outs(%3477 : tensor<1xi64>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb409(%3479: i64, %3480: i64, %3481: i64):
        %3482 = arith.addi %3479, %3480 : i64
        linalg.yield %3482 : i64
      } -> tensor<1xi64>
      %3483 = func.call @aten_index_copy_default_1_wl5(%3476, %3478, %3395) {prov.region_id = "aten_index_copy_default_1_3", prov.dispatch_id = "aten_index_copy_default_1_3"} : (tensor<4x39x32xf32>, tensor<1xi64>, tensor<1x4x1x32xf32>) -> tensor<1x4x39x32xf32>
      %3484 = "tensor.extract_slice"(%3475) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_20", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x39x32xf32>) -> tensor<1x4x39x32xf32>
      %3485 = "tensor.extract_slice"(%3484) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_21", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x39x32xf32>) -> tensor<1x4x39x32xf32>
      %3486 = tensor.collapse_shape %3485 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x39x32xf32> into tensor<4992xf32>
      %3487 = tensor.expand_shape %3486 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 39, 32] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<4992xf32> into tensor<1x4x1x39x32xf32>
      %3488 = "tensor.extract_slice"(%3487) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_22", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x39x32xf32>) -> tensor<1x4x1x39x32xf32>
      %3489 = "tensor.extract_slice"(%3488) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_23", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x39x32xf32>) -> tensor<1x4x1x39x32xf32>
      %3490 = tensor.empty() : tensor<1x4x2x39x32xf32>
      %3491 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3489 : tensor<1x4x1x39x32xf32>) outs(%3490 : tensor<1x4x2x39x32xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb410(%3492: f32, %3493: f32):
        linalg.yield %3492 : f32
      } -> tensor<1x4x2x39x32xf32>
      %3494 = tensor.collapse_shape %3491 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x4x2x39x32xf32> into tensor<9984xf32>
      %3495 = tensor.expand_shape %3494 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 39, 32] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<9984xf32> into tensor<1x8x39x32xf32>
      %3496 = "tensor.extract_slice"(%3483) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_24", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x39x32xf32>) -> tensor<1x4x39x32xf32>
      %3497 = "tensor.extract_slice"(%3496) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 39, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_25", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x39x32xf32>) -> tensor<1x4x39x32xf32>
      %3498 = tensor.collapse_shape %3497 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x39x32xf32> into tensor<4992xf32>
      %3499 = tensor.expand_shape %3498 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 39, 32] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<4992xf32> into tensor<1x4x1x39x32xf32>
      %3500 = "tensor.extract_slice"(%3499) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_26", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x39x32xf32>) -> tensor<1x4x1x39x32xf32>
      %3501 = "tensor.extract_slice"(%3500) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 39, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_27", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x39x32xf32>) -> tensor<1x4x1x39x32xf32>
      %3502 = tensor.empty() : tensor<1x4x2x39x32xf32>
      %3503 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3501 : tensor<1x4x1x39x32xf32>) outs(%3502 : tensor<1x4x2x39x32xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb411(%3504: f32, %3505: f32):
        linalg.yield %3504 : f32
      } -> tensor<1x4x2x39x32xf32>
      %3506 = tensor.collapse_shape %3503 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x4x2x39x32xf32> into tensor<9984xf32>
      %3507 = tensor.expand_shape %3506 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 39, 32] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<9984xf32> into tensor<1x8x39x32xf32>
      %3508 = tensor.empty() : tensor<1x8x32x39xf32>
      %3509 = linalg.transpose ins(%3495:tensor<1x8x39x32xf32>) outs(%3508:tensor<1x8x32x39xf32>) permutation = [0, 1, 3, 2]
      %3510 = arith.constant {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3511 = tensor.splat %3510 {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x1x39xf32>
      %3512 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%3437, %3509 : tensor<1x8x1x32xf32>, tensor<1x8x32x39xf32>) outs(%3511 : tensor<1x8x1x39xf32>) attrs =  {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb412(%3513: f32, %3514: f32, %3515: f32):
        %3516 = arith.mulf %3513, %3514 : f32
        %3517 = arith.addf %3515, %3516 : f32
        linalg.yield %3517 : f32
      } -> tensor<1x8x1x39xf32>
      %3518 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 5.65685415 : f32
      %3519 = tensor.splat %3518 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1x39xf32>
      %3520 = tensor.empty() : tensor<1x8x1x39xf32>
      %3521 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3512, %3519 : tensor<1x8x1x39xf32>, tensor<1x8x1x39xf32>) outs(%3520 : tensor<1x8x1x39xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
      ^bb413(%3522: f32, %3523: f32, %3524: f32):
        %3525 = arith.divf %3522, %3523 : f32
        linalg.yield %3525 : f32
      } -> tensor<1x8x1x39xf32>
      %3526 = tensor.empty() : tensor<1xi64>
      %3527 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2813, %54 : tensor<i64>, tensor<1xi64>) outs(%3526 : tensor<1xi64>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb414(%3528: i64, %3529: i64, %3530: i64):
        %3531 = arith.addi %3528, %3529 : i64
        linalg.yield %3531 : i64
      } -> tensor<1xi64>
      %3532 = tensor.expand_shape %3527 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1xi64> into tensor<1x1xi64>
      %3533 = tensor.expand_shape %36 [[0 : i64, 1 : i64]] output_shape [1, 39] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<39xi64> into tensor<1x39xi64>
      %3534 = tensor.empty() : tensor<1x39xi1>
      %3535 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3533, %3532 : tensor<1x39xi64>, tensor<1x1xi64>) outs(%3534 : tensor<1x39xi1>) attrs =  {prov.region_id = "compare_1", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
      ^bb415(%3536: i64, %3537: i64, %3538: i1):
        %3539 = arith.cmpi sle, %3536, %3537 : i64
        linalg.yield %3539 : i1
      } -> tensor<1x39xi1>
      %3540 = tensor.collapse_shape %3535 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x39xi1> into tensor<39xi1>
      %3541 = tensor.expand_shape %3540 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 39] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<39xi1> into tensor<1x1x39xi1>
      %3542 = tensor.collapse_shape %3541 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x1x39xi1> into tensor<39xi1>
      %3543 = tensor.expand_shape %3542 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 39] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<39xi1> into tensor<1x1x1x39xi1>
      %3544 = tensor.empty() : tensor<1x1x1x39xi1>
      %3545 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3543 : tensor<1x1x1x39xi1>) outs(%3544 : tensor<1x1x1x39xi1>) attrs =  {prov.region_id = "bitwise_1", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
      ^bb416(%3546: i1, %3547: i1):
        %3548 = arith.constant true
        %3549 = arith.xori %3546, %3548 : i1
        linalg.yield %3549 : i1
      } -> tensor<1x1x1x39xi1>
      %3550 = func.call @aten_masked_fill_Scalar_wl6(%3521, %3545) {prov.region_id = "aten_masked_fill_Scalar_1", prov.dispatch_id = "aten_masked_fill_Scalar_1"} : (tensor<1x8x1x39xf32>, tensor<1x1x1x39xi1>) -> tensor<1x8x1x39xf32>
      %3551 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0xff800000 : f32
      %3552 = tensor.splat %3551 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
      %3553 = linalg.reduce ins(%3550:tensor<1x8x1x39xf32>) outs(%3552:tensor<1x8x1xf32>) dimensions = [3]
      (%3554: f32, %3555: f32) {
        %3556 = arith.maximumf %3554, %3555 : f32
        linalg.yield %3556 : f32
      }
      %3557 = tensor.collapse_shape %3553 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x1xf32> into tensor<8xf32>
      %3558 = tensor.expand_shape %3557 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 1, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1x1xf32>
      %3559 = tensor.empty() : tensor<1x8x1x39xf32>
      %3560 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3550, %3558 : tensor<1x8x1x39xf32>, tensor<1x8x1x1xf32>) outs(%3559 : tensor<1x8x1x39xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb417(%3561: f32, %3562: f32, %3563: f32):
        %3564 = arith.subf %3561, %3562 : f32
        linalg.yield %3564 : f32
      } -> tensor<1x8x1x39xf32>
      %3565 = tensor.empty() : tensor<1x8x1x39xf32>
      %3566 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3560 : tensor<1x8x1x39xf32>) outs(%3565 : tensor<1x8x1x39xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb418(%3567: f32, %3568: f32):
        %3569 = math.exp %3567 : f32
        linalg.yield %3569 : f32
      } -> tensor<1x8x1x39xf32>
      %3570 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3571 = tensor.splat %3570 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
      %3572 = linalg.reduce ins(%3566:tensor<1x8x1x39xf32>) outs(%3571:tensor<1x8x1xf32>) dimensions = [3]
      (%3573: f32, %3574: f32) {
        %3575 = arith.addf %3573, %3574 : f32
        linalg.yield %3575 : f32
      }
      %3576 = tensor.collapse_shape %3572 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x1xf32> into tensor<8xf32>
      %3577 = tensor.expand_shape %3576 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 1, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1x1xf32>
      %3578 = tensor.empty() : tensor<1x8x1x39xf32>
      %3579 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3566, %3577 : tensor<1x8x1x39xf32>, tensor<1x8x1x1xf32>) outs(%3578 : tensor<1x8x1x39xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb419(%3580: f32, %3581: f32, %3582: f32):
        %3583 = arith.divf %3580, %3581 : f32
        linalg.yield %3583 : f32
      } -> tensor<1x8x1x39xf32>
      %3584 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3585 = tensor.splat %3584 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x1x32xf32>
      %3586 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%3579, %3507 : tensor<1x8x1x39xf32>, tensor<1x8x39x32xf32>) outs(%3585 : tensor<1x8x1x32xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb420(%3587: f32, %3588: f32, %3589: f32):
        %3590 = arith.mulf %3587, %3588 : f32
        %3591 = arith.addf %3589, %3590 : f32
        linalg.yield %3591 : f32
      } -> tensor<1x8x1x32xf32>
      %3592 = tensor.empty() : tensor<1x1x8x32xf32>
      %3593 = linalg.transpose ins(%3586:tensor<1x8x1x32xf32>) outs(%3592:tensor<1x1x8x32xf32>) permutation = [0, 2, 1, 3]
      %3594 = tensor.collapse_shape %3593 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x1x8x32xf32> into tensor<256xf32>
      %3595 = tensor.expand_shape %3594 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 256] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x1x256xf32>
      %3596 = tensor.empty() : tensor<1x1x256xf32>
      %3597 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3595 : tensor<1x1x256xf32>) outs(%3596 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb421(%3598: f32, %3599: f32):
        %3600 = arith.constant 2.000000e+00 : f32
        %3601 = math.powf %3598, %3600 : f32
        linalg.yield %3601 : f32
      } -> tensor<1x1x256xf32>
      %3602 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3603 = tensor.splat %3602 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3604 = linalg.reduce ins(%3597:tensor<1x1x256xf32>) outs(%3603:tensor<1x1xf32>) dimensions = [2]
      (%3605: f32, %3606: f32) {
        %3607 = arith.addf %3605, %3606 : f32
        linalg.yield %3607 : f32
      }
      %3608 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 2.560000e+02 : f32
      %3609 = tensor.splat %3608 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3610 = tensor.empty() : tensor<1x1xf32>
      %3611 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3604, %3609 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%3610 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb422(%3612: f32, %3613: f32, %3614: f32):
        %3615 = arith.divf %3612, %3613 : f32
        linalg.yield %3615 : f32
      } -> tensor<1x1xf32>
      %3616 = tensor.collapse_shape %3611 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %3617 = tensor.expand_shape %3616 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %3618 = arith.constant {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %3619 = tensor.splat %3618 {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %3620 = tensor.empty() : tensor<1x1x1xf32>
      %3621 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3617, %3619 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%3620 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb423(%3622: f32, %3623: f32, %3624: f32):
        %3625 = arith.addf %3622, %3623 : f32
        linalg.yield %3625 : f32
      } -> tensor<1x1x1xf32>
      %3626 = tensor.empty() : tensor<1x1x1xf32>
      %3627 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3621 : tensor<1x1x1xf32>) outs(%3626 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_5", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb424(%3628: f32, %3629: f32):
        %3630 = math.rsqrt %3628 : f32
        linalg.yield %3630 : f32
      } -> tensor<1x1x1xf32>
      %3631 = tensor.empty() : tensor<1x1x256xf32>
      %3632 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3595, %3627 : tensor<1x1x256xf32>, tensor<1x1x1xf32>) outs(%3631 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb425(%3633: f32, %3634: f32, %3635: f32):
        %3636 = arith.mulf %3633, %3634 : f32
        linalg.yield %3636 : f32
      } -> tensor<1x1x256xf32>
      %3637 = tensor.empty() : tensor<1x1x256xf32>
      %3638 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %3632 : tensor<256xf32>, tensor<1x1x256xf32>) outs(%3637 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb426(%3639: f32, %3640: f32, %3641: f32):
        %3642 = arith.mulf %3639, %3640 : f32
        linalg.yield %3642 : f32
      } -> tensor<1x1x256xf32>
      %3643 = func.call @wrap_with_set_grad_enabled_wl1(%3638) {prov.region_id = "wrap_with_set_grad_enabled_4", prov.dispatch_id = "wrap_with_set_grad_enabled_4"} : (tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %3644 = func.call @wrap_with_set_grad_enabled_1_wl2(%14) {prov.region_id = "wrap_with_set_grad_enabled_1_3", prov.dispatch_id = "wrap_with_set_grad_enabled_1_3"} : (tensor<256x256xf32>) -> tensor<256x256xf32>
      %3645 = tensor.empty() : tensor<256x256xf32>
      %3646 = linalg.transpose ins(%3644:tensor<256x256xf32>) outs(%3645:tensor<256x256xf32>) permutation = [1, 0]
      %3647 = tensor.empty() : tensor<1x1x256xf32>
      %3648 = arith.constant 0.000000e+00 : f32
      %3649 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%3648 : f32) outs(%3647 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %3650 = linalg.matmul {prov.region_id = "matmul_14", prov.dispatch_id = "matmul_14", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%3643, %3646 : tensor<1x1x256xf32>, tensor<256x256xf32>) outs(%3649 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %3651 = tensor.empty() : tensor<1x1x256xf32>
      %3652 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3308, %3650 : tensor<1x1x256xf32>, tensor<1x1x256xf32>) outs(%3651 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb427(%3653: f32, %3654: f32, %3655: f32):
        %3656 = arith.addf %3653, %3654 : f32
        linalg.yield %3656 : f32
      } -> tensor<1x1x256xf32>
      %3657 = tensor.empty() : tensor<1x1x256xf32>
      %3658 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3652 : tensor<1x1x256xf32>) outs(%3657 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb428(%3659: f32, %3660: f32):
        %3661 = arith.constant 2.000000e+00 : f32
        %3662 = math.powf %3659, %3661 : f32
        linalg.yield %3662 : f32
      } -> tensor<1x1x256xf32>
      %3663 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3664 = tensor.splat %3663 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3665 = linalg.reduce ins(%3658:tensor<1x1x256xf32>) outs(%3664:tensor<1x1xf32>) dimensions = [2]
      (%3666: f32, %3667: f32) {
        %3668 = arith.addf %3666, %3667 : f32
        linalg.yield %3668 : f32
      }
      %3669 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 2.560000e+02 : f32
      %3670 = tensor.splat %3669 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3671 = tensor.empty() : tensor<1x1xf32>
      %3672 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3665, %3670 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%3671 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb429(%3673: f32, %3674: f32, %3675: f32):
        %3676 = arith.divf %3673, %3674 : f32
        linalg.yield %3676 : f32
      } -> tensor<1x1xf32>
      %3677 = tensor.collapse_shape %3672 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %3678 = tensor.expand_shape %3677 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %3679 = arith.constant {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %3680 = tensor.splat %3679 {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %3681 = tensor.empty() : tensor<1x1x1xf32>
      %3682 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3678, %3680 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%3681 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb430(%3683: f32, %3684: f32, %3685: f32):
        %3686 = arith.addf %3683, %3684 : f32
        linalg.yield %3686 : f32
      } -> tensor<1x1x1xf32>
      %3687 = tensor.empty() : tensor<1x1x1xf32>
      %3688 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3682 : tensor<1x1x1xf32>) outs(%3687 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_6", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb431(%3689: f32, %3690: f32):
        %3691 = math.rsqrt %3689 : f32
        linalg.yield %3691 : f32
      } -> tensor<1x1x1xf32>
      %3692 = tensor.empty() : tensor<1x1x256xf32>
      %3693 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3652, %3688 : tensor<1x1x256xf32>, tensor<1x1x1xf32>) outs(%3692 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb432(%3694: f32, %3695: f32, %3696: f32):
        %3697 = arith.mulf %3694, %3695 : f32
        linalg.yield %3697 : f32
      } -> tensor<1x1x256xf32>
      %3698 = tensor.empty() : tensor<1x1x256xf32>
      %3699 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%21, %3693 : tensor<256xf32>, tensor<1x1x256xf32>) outs(%3698 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb433(%3700: f32, %3701: f32, %3702: f32):
        %3703 = arith.mulf %3700, %3701 : f32
        linalg.yield %3703 : f32
      } -> tensor<1x1x256xf32>
      %3704 = func.call @wrap_with_set_grad_enabled_wl1(%3699) {prov.region_id = "wrap_with_set_grad_enabled_5", prov.dispatch_id = "wrap_with_set_grad_enabled_5"} : (tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %3705 = func.call @wrap_with_set_grad_enabled_4_wl7(%16) {prov.region_id = "wrap_with_set_grad_enabled_4_2", prov.dispatch_id = "wrap_with_set_grad_enabled_4_2"} : (tensor<512x256xf32>) -> tensor<512x256xf32>
      %3706 = tensor.empty() : tensor<256x512xf32>
      %3707 = linalg.transpose ins(%3705:tensor<512x256xf32>) outs(%3706:tensor<256x512xf32>) permutation = [1, 0]
      %3708 = tensor.empty() : tensor<1x1x512xf32>
      %3709 = arith.constant 0.000000e+00 : f32
      %3710 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%3709 : f32) outs(%3708 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %3711 = linalg.matmul {prov.region_id = "matmul_15", prov.dispatch_id = "matmul_15", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%3704, %3707 : tensor<1x1x256xf32>, tensor<256x512xf32>) outs(%3710 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %3712 = tensor.empty() : tensor<1x1x512xf32>
      %3713 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3711 : tensor<1x1x512xf32>) outs(%3712 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "minmax_1", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu.default", prov.orig_dtype = "float32"} {
      ^bb434(%3714: f32, %3715: f32):
        %3716 = arith.constant 0.000000e+00 : f32
        %3717 = arith.maximumf %3714, %3716 : f32
        linalg.yield %3717 : f32
      } -> tensor<1x1x512xf32>
      %3718 = tensor.empty() : tensor<1x1x512xf32>
      %3719 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3713 : tensor<1x1x512xf32>) outs(%3718 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb435(%3720: f32, %3721: f32):
        %3722 = arith.constant 2.000000e+00 : f32
        %3723 = math.powf %3720, %3722 : f32
        linalg.yield %3723 : f32
      } -> tensor<1x1x512xf32>
      %3724 = func.call @wrap_with_set_grad_enabled_2_wl3() {prov.region_id = "wrap_with_set_grad_enabled_2_5", prov.dispatch_id = "wrap_with_set_grad_enabled_2_5"} : () -> tensor<1x1x256xf32>
      %3725 = func.call @wrap_with_set_grad_enabled_4_wl7(%17) {prov.region_id = "wrap_with_set_grad_enabled_4_3", prov.dispatch_id = "wrap_with_set_grad_enabled_4_3"} : (tensor<512x256xf32>) -> tensor<512x256xf32>
      %3726 = tensor.empty() : tensor<256x512xf32>
      %3727 = linalg.transpose ins(%3725:tensor<512x256xf32>) outs(%3726:tensor<256x512xf32>) permutation = [1, 0]
      %3728 = tensor.empty() : tensor<1x1x512xf32>
      %3729 = arith.constant 0.000000e+00 : f32
      %3730 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%3729 : f32) outs(%3728 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %3731 = linalg.matmul {prov.region_id = "matmul_16", prov.dispatch_id = "matmul_16", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%3724, %3727 : tensor<1x1x256xf32>, tensor<256x512xf32>) outs(%3730 : tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %3732 = tensor.empty() : tensor<1x1x512xf32>
      %3733 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3719, %3731 : tensor<1x1x512xf32>, tensor<1x1x512xf32>) outs(%3732 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb436(%3734: f32, %3735: f32, %3736: f32):
        %3737 = arith.mulf %3734, %3735 : f32
        linalg.yield %3737 : f32
      } -> tensor<1x1x512xf32>
      %3738 = tensor.empty() : tensor<1x1x512xf32>
      %3739 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3733 : tensor<1x1x512xf32>) outs(%3738 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "pow_9", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb437(%3740: f32, %3741: f32):
        %3742 = arith.constant 2.000000e+00 : f32
        %3743 = math.powf %3740, %3742 : f32
        linalg.yield %3743 : f32
      } -> tensor<1x1x512xf32>
      %3744 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3745 = tensor.splat %3744 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3746 = linalg.reduce ins(%3739:tensor<1x1x512xf32>) outs(%3745:tensor<1x1xf32>) dimensions = [2]
      (%3747: f32, %3748: f32) {
        %3749 = arith.addf %3747, %3748 : f32
        linalg.yield %3749 : f32
      }
      %3750 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 5.120000e+02 : f32
      %3751 = tensor.splat %3750 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3752 = tensor.empty() : tensor<1x1xf32>
      %3753 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3746, %3751 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%3752 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb438(%3754: f32, %3755: f32, %3756: f32):
        %3757 = arith.divf %3754, %3755 : f32
        linalg.yield %3757 : f32
      } -> tensor<1x1xf32>
      %3758 = tensor.collapse_shape %3753 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %3759 = tensor.expand_shape %3758 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %3760 = arith.constant {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %3761 = tensor.splat %3760 {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %3762 = tensor.empty() : tensor<1x1x1xf32>
      %3763 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3759, %3761 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%3762 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb439(%3764: f32, %3765: f32, %3766: f32):
        %3767 = arith.addf %3764, %3765 : f32
        linalg.yield %3767 : f32
      } -> tensor<1x1x1xf32>
      %3768 = tensor.empty() : tensor<1x1x1xf32>
      %3769 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3763 : tensor<1x1x1xf32>) outs(%3768 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_7", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb440(%3770: f32, %3771: f32):
        %3772 = math.rsqrt %3770 : f32
        linalg.yield %3772 : f32
      } -> tensor<1x1x1xf32>
      %3773 = tensor.empty() : tensor<1x1x512xf32>
      %3774 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3733, %3769 : tensor<1x1x512xf32>, tensor<1x1x1xf32>) outs(%3773 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb441(%3775: f32, %3776: f32, %3777: f32):
        %3778 = arith.mulf %3775, %3776 : f32
        linalg.yield %3778 : f32
      } -> tensor<1x1x512xf32>
      %3779 = tensor.empty() : tensor<1x1x512xf32>
      %3780 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %3774 : tensor<512xf32>, tensor<1x1x512xf32>) outs(%3779 : tensor<1x1x512xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb442(%3781: f32, %3782: f32, %3783: f32):
        %3784 = arith.mulf %3781, %3782 : f32
        linalg.yield %3784 : f32
      } -> tensor<1x1x512xf32>
      %3785 = func.call @wrap_with_set_grad_enabled_5_wl8(%3780) {prov.region_id = "wrap_with_set_grad_enabled_5_1", prov.dispatch_id = "wrap_with_set_grad_enabled_5_1"} : (tensor<1x1x512xf32>) -> tensor<1x1x512xf32>
      %3786 = func.call @wrap_with_set_grad_enabled_6_wl9(%18) {prov.region_id = "wrap_with_set_grad_enabled_6_1", prov.dispatch_id = "wrap_with_set_grad_enabled_6_1"} : (tensor<256x512xf32>) -> tensor<256x512xf32>
      %3787 = tensor.empty() : tensor<512x256xf32>
      %3788 = linalg.transpose ins(%3786:tensor<256x512xf32>) outs(%3787:tensor<512x256xf32>) permutation = [1, 0]
      %3789 = tensor.empty() : tensor<1x1x256xf32>
      %3790 = arith.constant 0.000000e+00 : f32
      %3791 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%3790 : f32) outs(%3789 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %3792 = linalg.matmul {prov.region_id = "matmul_17", prov.dispatch_id = "matmul_17", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%3785, %3788 : tensor<1x1x512xf32>, tensor<512x256xf32>) outs(%3791 : tensor<1x1x256xf32>) -> tensor<1x1x256xf32>
      %3793 = tensor.empty() : tensor<1x1x256xf32>
      %3794 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3652, %3792 : tensor<1x1x256xf32>, tensor<1x1x256xf32>) outs(%3793 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb443(%3795: f32, %3796: f32, %3797: f32):
        %3798 = arith.addf %3795, %3796 : f32
        linalg.yield %3798 : f32
      } -> tensor<1x1x256xf32>
      %3799 = func.call @aten_stack_default_wl10(%2989, %3475) {prov.region_id = "aten_stack_default_0", prov.dispatch_id = "aten_stack_default_0"} : (tensor<1x4x39x32xf32>, tensor<1x4x39x32xf32>) -> tensor<2x1x4x39x32xf32>
      %3800 = func.call @aten_stack_default_wl10(%2997, %3483) {prov.region_id = "aten_stack_default_1", prov.dispatch_id = "aten_stack_default_1"} : (tensor<1x4x39x32xf32>, tensor<1x4x39x32xf32>) -> tensor<2x1x4x39x32xf32>
      %3801 = tensor.empty() : tensor<1x1x256xf32>
      %3802 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3794 : tensor<1x1x256xf32>) outs(%3801 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "pow_10", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb444(%3803: f32, %3804: f32):
        %3805 = arith.constant 2.000000e+00 : f32
        %3806 = math.powf %3803, %3805 : f32
        linalg.yield %3806 : f32
      } -> tensor<1x1x256xf32>
      %3807 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3808 = tensor.splat %3807 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3809 = linalg.reduce ins(%3802:tensor<1x1x256xf32>) outs(%3808:tensor<1x1xf32>) dimensions = [2]
      (%3810: f32, %3811: f32) {
        %3812 = arith.addf %3810, %3811 : f32
        linalg.yield %3812 : f32
      }
      %3813 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 2.560000e+02 : f32
      %3814 = tensor.splat %3813 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3815 = tensor.empty() : tensor<1x1xf32>
      %3816 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3809, %3814 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%3815 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb445(%3817: f32, %3818: f32, %3819: f32):
        %3820 = arith.divf %3817, %3818 : f32
        linalg.yield %3820 : f32
      } -> tensor<1x1xf32>
      %3821 = tensor.collapse_shape %3816 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %3822 = tensor.expand_shape %3821 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %3823 = arith.constant {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %3824 = tensor.splat %3823 {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %3825 = tensor.empty() : tensor<1x1x1xf32>
      %3826 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3822, %3824 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%3825 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb446(%3827: f32, %3828: f32, %3829: f32):
        %3830 = arith.addf %3827, %3828 : f32
        linalg.yield %3830 : f32
      } -> tensor<1x1x1xf32>
      %3831 = tensor.empty() : tensor<1x1x1xf32>
      %3832 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3826 : tensor<1x1x1xf32>) outs(%3831 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_8", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb447(%3833: f32, %3834: f32):
        %3835 = math.rsqrt %3833 : f32
        linalg.yield %3835 : f32
      } -> tensor<1x1x1xf32>
      %3836 = tensor.empty() : tensor<1x1x256xf32>
      %3837 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3794, %3832 : tensor<1x1x256xf32>, tensor<1x1x1xf32>) outs(%3836 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb448(%3838: f32, %3839: f32, %3840: f32):
        %3841 = arith.mulf %3838, %3839 : f32
        linalg.yield %3841 : f32
      } -> tensor<1x1x256xf32>
      %3842 = tensor.empty() : tensor<1x1x256xf32>
      %3843 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22, %3837 : tensor<256xf32>, tensor<1x1x256xf32>) outs(%3842 : tensor<1x1x256xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb449(%3844: f32, %3845: f32, %3846: f32):
        %3847 = arith.mulf %3844, %3845 : f32
        linalg.yield %3847 : f32
      } -> tensor<1x1x256xf32>
      %3848 = tensor.empty() : tensor<256x1024xf32>
      %3849 = linalg.transpose ins(%24:tensor<1024x256xf32>) outs(%3848:tensor<256x1024xf32>) permutation = [1, 0]
      %3850 = tensor.empty() : tensor<1x1x1024xf32>
      %3851 = arith.constant 0.000000e+00 : f32
      %3852 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%3851 : f32) outs(%3850 : tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
      %3853 = linalg.matmul {prov.region_id = "matmul_18", prov.dispatch_id = "matmul_18", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%3843, %3849 : tensor<1x1x256xf32>, tensor<256x1024xf32>) outs(%3852 : tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
      %3854 = "tensor.extract_slice"(%3853) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_28", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
      %3855 = "tensor.extract_slice"(%3854) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<1x1x1024xf32>) -> tensor<1024xf32>
      %3856 = tensor.expand_shape %3855 [[0 : i64, 1 : i64]] output_shape [1, 1024] {prov.region_id = "slice_29", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1024xf32>
      %3857 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0xff800000 : f32
      %3858 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0 : i64
      %3859 = tensor.splat %3857 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xf32>
      %3860 = tensor.splat %3858 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xi64>
      %3861, %3862 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%3856 : tensor<1x1024xf32>) outs(%3859, %3860 : tensor<1xf32>, tensor<1xi64>) attrs =  {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} {
      ^bb450(%3863: f32, %3864: f32, %3865: i64):
        %3866 = linalg.index 1 : index
        %3867 = arith.index_cast %3866 : index to i64
        %3868 = arith.cmpf ogt, %3863, %3864 : f32
        %3869 = arith.select %3868, %3863, %3864 : f32
        %3870 = arith.select %3868, %3867, %3865 : i64
        linalg.yield %3869, %3870 : f32, i64
      } -> (tensor<1xf32>, tensor<1xi64>)
      %3871 = tensor.expand_shape %3861 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xf32> into tensor<1x1xf32>
      %3872 = tensor.expand_shape %3862 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xi64> into tensor<1x1xi64>
      %3873 = arith.constant {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} 1 : i64
      %3874 = tensor.splat %3873 {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} : tensor<i64>
      %3875 = tensor.empty() : tensor<i64>
      %3876 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2804, %3874 : tensor<i64>, tensor<i64>) outs(%3875 : tensor<i64>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb451(%3877: i64, %3878: i64, %3879: i64):
        %3880 = arith.addi %3877, %3878 : i64
        linalg.yield %3880 : i64
      } -> tensor<i64>
      scf.yield %3876, %3872, %2811, %3799, %3800 : tensor<i64>, tensor<1x1xi64>, tensor<1x7xi64>, tensor<2x1x4x39x32xf32>, tensor<2x1x4x39x32xf32>
    }
    func.return %2800 : tensor<1x7xi64>
  }
}
