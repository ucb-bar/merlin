builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func private @aten_masked_fill__Scalar(tensor<32x32xf32>, tensor<32x32xi1>) -> tensor<32x32xf32>
  func.func private @aten_rsub_Scalar(tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
  func.func private @aten_masked_fill_Scalar(tensor<1x1x32x32xf32>, tensor<1x1x32x32xi1>) -> tensor<1x1x32x32xf32>
  func.func private @aten_clamp__default(tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  func.func private @aten___and___Scalar(tensor<16384xi8>) -> tensor<16384xi8>
  func.func private @aten___rshift___Scalar(tensor<16384xi8>) -> tensor<16384xi8>
  func.func private @aten_stack_default(tensor<16384xi8>, tensor<16384xi8>, tensor<16384xi8>, tensor<16384xi8>) -> tensor<16384x4xi8>
  func.func private @aten_mul_Tensor(tensor<65536xf32>) -> tensor<65536xf32>
  func.func private @aten___and___Scalar_1(tensor<8192xi8>) -> tensor<8192xi8>
  func.func private @aten___rshift___Scalar_1(tensor<8192xi8>) -> tensor<8192xi8>
  func.func private @aten_stack_default_1(tensor<8192xi8>, tensor<8192xi8>, tensor<8192xi8>, tensor<8192xi8>) -> tensor<8192x4xi8>
  func.func private @aten_mul_Tensor_1(tensor<32768xf32>) -> tensor<32768xf32>
  func.func private @aten___and___Scalar_2(tensor<32768xi8>) -> tensor<32768xi8>
  func.func private @aten___rshift___Scalar_2(tensor<32768xi8>) -> tensor<32768xi8>
  func.func private @aten_stack_default_2(tensor<32768xi8>, tensor<32768xi8>, tensor<32768xi8>, tensor<32768xi8>) -> tensor<32768x4xi8>
  func.func private @aten_mul_Tensor_2(tensor<131072xf32>) -> tensor<131072xf32>
  func.func @forward(%0: tensor<128x3x16x16xf32>, %1: tensor<128xf32>, %2: tensor<196x128xf32>, %3: tensor<128xf32>, %4: tensor<128xf32>, %5: tensor<128xf32>, %6: tensor<128xf32>, %7: tensor<128xf32>, %8: tensor<128xf32>, %9: tensor<256xf32>, %10: tensor<128xf32>, %11: tensor<128xf32>, %12: tensor<128xf32>, %13: tensor<128xf32>, %14: tensor<128xf32>, %15: tensor<128xf32>, %16: tensor<128xf32>, %17: tensor<128xf32>, %18: tensor<128xf32>, %19: tensor<256xf32>, %20: tensor<128xf32>, %21: tensor<128xf32>, %22: tensor<128xf32>, %23: tensor<128xf32>, %24: tensor<128xf32>, %25: tensor<1x1x128xf32>, %26: tensor<384x128xf32>, %27: tensor<384xf32>, %28: tensor<128x128xf32>, %29: tensor<128xf32>, %30: tensor<128xf32>, %31: tensor<128xf32>, %32: tensor<256xf32>, %33: tensor<128xf32>, %34: tensor<256x128xf32>, %35: tensor<256xf32>, %36: tensor<256x256xf32>, %37: tensor<256xf32>, %38: tensor<1024x256xf32>, %39: tensor<256xf32>, %40: tensor<512xf32>, %41: tensor<256xf32>, %42: tensor<256xf32>, %43: tensor<256xf32>, %44: tensor<512xf32>, %45: tensor<256xf32>, %46: tensor<256xf32>, %47: tensor<256xf32>, %48: tensor<1024x256xf32>, %49: tensor<1x196xi64>, %50: tensor<4096xi8>, %51: tensor<f32>, %52: tensor<4096xi8>, %53: tensor<f32>, %54: tensor<4096xi8>, %55: tensor<f32>, %56: tensor<4096xi8>, %57: tensor<f32>, %58: tensor<8192xi8>, %59: tensor<f32>, %60: tensor<8192xi8>, %61: tensor<f32>, %62: tensor<4096xi8>, %63: tensor<f32>, %64: tensor<4096xi8>, %65: tensor<f32>, %66: tensor<4096xi8>, %67: tensor<f32>, %68: tensor<4096xi8>, %69: tensor<f32>, %70: tensor<8192xi8>, %71: tensor<f32>, %72: tensor<8192xi8>, %73: tensor<f32>, %74: tensor<8192xi8>, %75: tensor<f32>, %76: tensor<8192xi8>, %77: tensor<f32>, %78: tensor<16384xi8>, %79: tensor<f32>, %80: tensor<8192xi8>, %81: tensor<f32>, %82: tensor<8192xi8>, %83: tensor<f32>, %84: tensor<16384xi8>, %85: tensor<f32>, %86: tensor<16xf32>, %87: tensor<2048x32xf32>, %88: tensor<2048x32xf32>, %89: tensor<32768xi8>, %90: tensor<f32>, %91: tensor<32768xi8>, %92: tensor<f32>, %93: tensor<32768xi8>, %94: tensor<f32>, %95: tensor<16384xi8>, %96: tensor<f32>, %97: tensor<8192xi8>, %98: tensor<f32>, %99: tensor<8192xi8>, %100: tensor<f32>, %101: tensor<16384xi8>, %102: tensor<f32>, %103: tensor<16xf32>, %104: tensor<2048x32xf32>, %105: tensor<2048x32xf32>, %106: tensor<32768xi8>, %107: tensor<f32>, %108: tensor<32768xi8>, %109: tensor<f32>, %110: tensor<32768xi8>, %111: tensor<f32>, %112: tensor<1x32x256xf32>, %113: tensor<1x32xi64>) -> tensor<1x32x1024xf32> {
    %114 = tensor.empty() : tensor<32xi64>
    %115 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%114 : tensor<32xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb0(%116: i64):
      %117 = linalg.index 0 : index
      %118 = arith.index_cast %117 : index to i64
      %119 = arith.constant 1 : i64
      %120 = arith.muli %118, %119 : i64
      %121 = arith.constant 0 : i64
      %122 = arith.addi %121, %120 : i64
      linalg.yield %122 : i64
    } -> tensor<32xi64>
    %123 = tensor.expand_shape %115 [[0 : i64, 1 : i64]] output_shape [1, 32] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<32xi64> into tensor<1x32xi64>
    %124 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} -3.40282347e+38 : f32
    %125 = tensor.splat %124 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<32x32xf32>
    %126 = tensor.empty() : tensor<32xi64>
    %127 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%126 : tensor<32xi64>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb1(%128: i64):
      %129 = linalg.index 0 : index
      %130 = arith.index_cast %129 : index to i64
      %131 = arith.constant 1 : i64
      %132 = arith.muli %130, %131 : i64
      %133 = arith.constant 0 : i64
      %134 = arith.addi %133, %132 : i64
      linalg.yield %134 : i64
    } -> tensor<32xi64>
    %135 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} 1 : i64
    %136 = tensor.splat %135 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<32xi64>
    %137 = tensor.empty() : tensor<32xi64>
    %138 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%127, %136 : tensor<32xi64>, tensor<32xi64>) outs(%137 : tensor<32xi64>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb2(%139: i64, %140: i64, %141: i64):
      %142 = arith.addi %139, %140 : i64
      linalg.yield %142 : i64
    } -> tensor<32xi64>
    %143 = tensor.expand_shape %138 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<32xi64> into tensor<32x1xi64>
    %144 = tensor.empty() : tensor<32x32xi1>
    %145 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%127, %143 : tensor<32xi64>, tensor<32x1xi64>) outs(%144 : tensor<32x32xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.lt.Tensor", prov.orig_dtype = "bool", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb3(%146: i64, %147: i64, %148: i1):
      %149 = arith.cmpi slt, %146, %147 : i64
      linalg.yield %149 : i1
    } -> tensor<32x32xi1>
    %150 = func.call @aten_masked_fill__Scalar(%125, %145) {prov.region_id = "aten_masked_fill__Scalar_0", prov.dispatch_id = "aten_masked_fill__Scalar_0"} : (tensor<32x32xf32>, tensor<32x32xi1>) -> tensor<32x32xf32>
    %151 = tensor.collapse_shape %150 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<32x32xf32> into tensor<1024xf32>
    %152 = tensor.expand_shape %151 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 32] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1024xf32> into tensor<1x32x32xf32>
    %153 = tensor.collapse_shape %152 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %154 = tensor.expand_shape %153 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %155 = "tensor.extract_slice"(%154) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    %156 = "tensor.extract_slice"(%155) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    %157 = tensor.empty() : tensor<1x1x32x32xf32>
    %158 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%156 : tensor<1x1x32x32xf32>) outs(%157 : tensor<1x1x32x32xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb4(%159: f32, %160: f32):
      linalg.yield %159 : f32
    } -> tensor<1x1x32x32xf32>
    %161 = "tensor.extract_slice"(%113) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : (tensor<1x32xi64>) -> tensor<1x32xi64>
    %162 = tensor.collapse_shape %161 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1x32xi64> into tensor<32xi64>
    %163 = tensor.expand_shape %162 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<32xi64> into tensor<1x1x32xi64>
    %164 = tensor.collapse_shape %163 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1x1x32xi64> into tensor<32xi64>
    %165 = tensor.expand_shape %164 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<32xi64> into tensor<1x1x1x32xi64>
    %166 = "tensor.extract_slice"(%165) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : (tensor<1x1x1x32xi64>) -> tensor<1x1x1x32xi64>
    %167 = tensor.empty() : tensor<1x1x32x32xi64>
    %168 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%166 : tensor<1x1x1x32xi64>) outs(%167 : tensor<1x1x32x32xi64>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb5(%169: i64, %170: i64):
      linalg.yield %169 : i64
    } -> tensor<1x1x32x32xi64>
    %171 = tensor.empty() : tensor<1x1x32x32xf32>
    %172 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%168 : tensor<1x1x32x32xi64>) outs(%171 : tensor<1x1x32x32xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb6(%173: i64, %174: f32):
      %175 = arith.sitofp %173 : i64 to f32
      linalg.yield %175 : f32
    } -> tensor<1x1x32x32xf32>
    %176 = func.call @aten_rsub_Scalar(%172) {prov.region_id = "aten_rsub_Scalar_0", prov.dispatch_id = "aten_rsub_Scalar_0"} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    %177 = tensor.empty() : tensor<1x1x32x32xi1>
    %178 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%176 : tensor<1x1x32x32xf32>) outs(%177 : tensor<1x1x32x32xi1>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "bool", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb7(%179: f32, %180: i1):
      %181 = arith.fptosi %179 : f32 to i1
      linalg.yield %181 : i1
    } -> tensor<1x1x32x32xi1>
    %182 = func.call @aten_masked_fill_Scalar(%176, %178) {prov.region_id = "aten_masked_fill_Scalar_0", prov.dispatch_id = "aten_masked_fill_Scalar_0"} : (tensor<1x1x32x32xf32>, tensor<1x1x32x32xi1>) -> tensor<1x1x32x32xf32>
    %183 = tensor.empty() : tensor<1x1x32x32xi1>
    %184 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%182 : tensor<1x1x32x32xf32>) outs(%183 : tensor<1x1x32x32xi1>) attrs =  {prov.region_id = "dtype_cast_2", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "bool", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb8(%185: f32, %186: i1):
      %187 = arith.fptosi %185 : f32 to i1
      linalg.yield %187 : i1
    } -> tensor<1x1x32x32xi1>
    %188 = func.call @aten_masked_fill_Scalar(%158, %184) {prov.region_id = "aten_masked_fill_Scalar_1", prov.dispatch_id = "aten_masked_fill_Scalar_1"} : (tensor<1x1x32x32xf32>, tensor<1x1x32x32xi1>) -> tensor<1x1x32x32xf32>
    %189 = tensor.empty() : tensor<1x32x256xf32>
    %190 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%112 : tensor<1x32x256xf32>) outs(%189 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb9(%191: f32, %192: f32):
      %193 = arith.constant 2.000000e+00 : f32
      %194 = math.powf %191, %193 : f32
      linalg.yield %194 : f32
    } -> tensor<1x32x256xf32>
    %195 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} 0.000000e+00 : f32
    %196 = tensor.splat %195 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} : tensor<1x32xf32>
    %197 = linalg.reduce ins(%190:tensor<1x32x256xf32>) outs(%196:tensor<1x32xf32>) dimensions = [2]
    (%198: f32, %199: f32) {
      %200 = arith.addf %198, %199 : f32
      linalg.yield %200 : f32
    }
    %201 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} 2.560000e+02 : f32
    %202 = tensor.splat %201 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} : tensor<1x32xf32>
    %203 = tensor.empty() : tensor<1x32xf32>
    %204 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%197, %202 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%203 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb10(%205: f32, %206: f32, %207: f32):
      %208 = arith.divf %205, %206 : f32
      linalg.yield %208 : f32
    } -> tensor<1x32xf32>
    %209 = tensor.collapse_shape %204 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %210 = tensor.expand_shape %209 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %211 = arith.constant {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} 1.000000e-05 : f32
    %212 = tensor.splat %211 {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} : tensor<1x32x1xf32>
    %213 = tensor.empty() : tensor<1x32x1xf32>
    %214 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%210, %212 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%213 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb11(%215: f32, %216: f32, %217: f32):
      %218 = arith.addf %215, %216 : f32
      linalg.yield %218 : f32
    } -> tensor<1x32x1xf32>
    %219 = tensor.empty() : tensor<1x32x1xf32>
    %220 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%214 : tensor<1x32x1xf32>) outs(%219 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb12(%221: f32, %222: f32):
      %223 = math.rsqrt %221 : f32
      linalg.yield %223 : f32
    } -> tensor<1x32x1xf32>
    %224 = tensor.empty() : tensor<1x32x256xf32>
    %225 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%112, %220 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%224 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb13(%226: f32, %227: f32, %228: f32):
      %229 = arith.mulf %226, %227 : f32
      linalg.yield %229 : f32
    } -> tensor<1x32x256xf32>
    %230 = tensor.empty() : tensor<1x32x256xf32>
    %231 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %225 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%230 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb14(%232: f32, %233: f32, %234: f32):
      %235 = arith.mulf %232, %233 : f32
      linalg.yield %235 : f32
    } -> tensor<1x32x256xf32>
    %236 = tensor.empty() : tensor<1x32x256xf32>
    %237 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%231 : tensor<1x32x256xf32>) outs(%236 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_0", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb15(%238: f32, %239: f32):
      %240 = math.absf %238 : f32
      linalg.yield %240 : f32
    } -> tensor<1x32x256xf32>
    %241 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} 0xff800000 : f32
    %242 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} 0 : i64
    %243 = tensor.splat %241 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<1x32xf32>
    %244 = tensor.splat %242 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<1x32xi64>
    %245, %246 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%237 : tensor<1x32x256xf32>) outs(%243, %244 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb16(%247: f32, %248: f32, %249: i64):
      %250 = linalg.index 2 : index
      %251 = arith.index_cast %250 : index to i64
      %252 = arith.cmpf ogt, %247, %248 : f32
      %253 = arith.select %252, %247, %248 : f32
      %254 = arith.select %252, %251, %249 : i64
      linalg.yield %253, %254 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %255 = tensor.collapse_shape %245 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %256 = tensor.expand_shape %255 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %257 = tensor.collapse_shape %246 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %258 = tensor.expand_shape %257 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %259 = func.call @aten_clamp__default(%256) {prov.region_id = "aten_clamp__default_0", prov.dispatch_id = "aten_clamp__default_0"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %260 = tensor.empty() : tensor<1x32x1xf32>
    %261 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%259 : tensor<1x32x1xf32>) outs(%260 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_0", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb17(%262: f32, %263: f32):
      %264 = arith.constant 1.000000e+00 : f32
      %265 = arith.divf %264, %262 : f32
      linalg.yield %265 : f32
    } -> tensor<1x32x1xf32>
    %266 = arith.constant {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} 1.270000e+02 : f32
    %267 = tensor.splat %266 {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<1x32x1xf32>
    %268 = tensor.empty() : tensor<1x32x1xf32>
    %269 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%261, %267 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%268 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb18(%270: f32, %271: f32, %272: f32):
      %273 = arith.mulf %270, %271 : f32
      linalg.yield %273 : f32
    } -> tensor<1x32x1xf32>
    %274 = tensor.empty() : tensor<1x32x256xf32>
    %275 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%231, %269 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%274 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb19(%276: f32, %277: f32, %278: f32):
      %279 = arith.mulf %276, %277 : f32
      linalg.yield %279 : f32
    } -> tensor<1x32x256xf32>
    %280 = tensor.empty() : tensor<1x32x256xf32>
    %281 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%275 : tensor<1x32x256xf32>) outs(%280 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_0", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb20(%282: f32, %283: f32):
      %284 = math.roundeven %282 : f32
      linalg.yield %284 : f32
    } -> tensor<1x32x256xf32>
    %285 = tensor.empty() : tensor<1x32x256xf32>
    %286 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%281 : tensor<1x32x256xf32>) outs(%285 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_0", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb21(%287: f32, %288: f32):
      %289 = arith.constant -1.280000e+02 : f32
      %290 = arith.maximumf %287, %289 : f32
      %291 = arith.constant 1.270000e+02 : f32
      %292 = arith.minimumf %290, %291 : f32
      linalg.yield %292 : f32
    } -> tensor<1x32x256xf32>
    %293 = tensor.empty() : tensor<1x32x256xf32>
    %294 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%286, %269 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%293 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb22(%295: f32, %296: f32, %297: f32):
      %298 = arith.divf %295, %296 : f32
      linalg.yield %298 : f32
    } -> tensor<1x32x256xf32>
    %299 = "quant_ext.unpack_int2"(%78) <{bits = 2 : i64, lanes = 4 : i64}> {prov.op = "unpack_int2", prov.family = "quantize"} : (tensor<16384xi8>) -> tensor<16384x4xi8>
    %300 = tensor.collapse_shape %299 [[0 : i64, 1 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<16384x4xi8> into tensor<65536xi8>
    %301 = "tensor.extract_slice"(%300) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 65536>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : (tensor<65536xi8>) -> tensor<65536xi8>
    %302 = tensor.empty() : tensor<65536xf32>
    %303 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%301 : tensor<65536xi8>) outs(%302 : tensor<65536xf32>) attrs =  {prov.region_id = "dtype_cast_3", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb23(%304: i8, %305: f32):
      %306 = arith.sitofp %304 : i8 to f32
      linalg.yield %306 : f32
    } -> tensor<65536xf32>
    %307 = arith.constant {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} 1.000000e+00 : f32
    %308 = tensor.splat %307 {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<65536xf32>
    %309 = tensor.empty() : tensor<65536xf32>
    %310 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%303, %308 : tensor<65536xf32>, tensor<65536xf32>) outs(%309 : tensor<65536xf32>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb24(%311: f32, %312: f32, %313: f32):
      %314 = arith.subf %311, %312 : f32
      linalg.yield %314 : f32
    } -> tensor<65536xf32>
    %315 = func.call @aten_mul_Tensor(%310) {prov.region_id = "aten_mul_Tensor_0", prov.dispatch_id = "aten_mul_Tensor_0"} : (tensor<65536xf32>) -> tensor<65536xf32>
    %316 = tensor.expand_shape %315 [[0 : i64, 1 : i64]] output_shape [256, 256] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<65536xf32> into tensor<256x256xf32>
    %317 = tensor.empty() : tensor<256x256xf32>
    %318 = linalg.transpose ins(%316:tensor<256x256xf32>) outs(%317:tensor<256x256xf32>) permutation = [1, 0]
    %319 = tensor.empty() : tensor<1x32x256xf32>
    %320 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %321 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%320 : f32) outs(%319 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %322 = linalg.matmul {prov.region_id = "matmul_0", prov.dispatch_id = "matmul_0", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} ins(%294, %318 : tensor<1x32x256xf32>, tensor<256x256xf32>) outs(%321 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %323 = tensor.empty() : tensor<1x32x256xf32>
    %324 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%231 : tensor<1x32x256xf32>) outs(%323 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_1", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb25(%325: f32, %326: f32):
      %327 = math.absf %325 : f32
      linalg.yield %327 : f32
    } -> tensor<1x32x256xf32>
    %328 = arith.constant {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} 0xff800000 : f32
    %329 = arith.constant {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} 0 : i64
    %330 = tensor.splat %328 {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<1x32xf32>
    %331 = tensor.splat %329 {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<1x32xi64>
    %332, %333 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%324 : tensor<1x32x256xf32>) outs(%330, %331 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb26(%334: f32, %335: f32, %336: i64):
      %337 = linalg.index 2 : index
      %338 = arith.index_cast %337 : index to i64
      %339 = arith.cmpf ogt, %334, %335 : f32
      %340 = arith.select %339, %334, %335 : f32
      %341 = arith.select %339, %338, %336 : i64
      linalg.yield %340, %341 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %342 = tensor.collapse_shape %332 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %343 = tensor.expand_shape %342 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %344 = tensor.collapse_shape %333 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %345 = tensor.expand_shape %344 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %346 = func.call @aten_clamp__default(%343) {prov.region_id = "aten_clamp__default_1", prov.dispatch_id = "aten_clamp__default_1"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %347 = tensor.empty() : tensor<1x32x1xf32>
    %348 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%346 : tensor<1x32x1xf32>) outs(%347 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_1", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb27(%349: f32, %350: f32):
      %351 = arith.constant 1.000000e+00 : f32
      %352 = arith.divf %351, %349 : f32
      linalg.yield %352 : f32
    } -> tensor<1x32x1xf32>
    %353 = arith.constant {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} 1.270000e+02 : f32
    %354 = tensor.splat %353 {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<1x32x1xf32>
    %355 = tensor.empty() : tensor<1x32x1xf32>
    %356 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%348, %354 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%355 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb28(%357: f32, %358: f32, %359: f32):
      %360 = arith.mulf %357, %358 : f32
      linalg.yield %360 : f32
    } -> tensor<1x32x1xf32>
    %361 = tensor.empty() : tensor<1x32x256xf32>
    %362 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%231, %356 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%361 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb29(%363: f32, %364: f32, %365: f32):
      %366 = arith.mulf %363, %364 : f32
      linalg.yield %366 : f32
    } -> tensor<1x32x256xf32>
    %367 = tensor.empty() : tensor<1x32x256xf32>
    %368 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%362 : tensor<1x32x256xf32>) outs(%367 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_1", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb30(%369: f32, %370: f32):
      %371 = math.roundeven %369 : f32
      linalg.yield %371 : f32
    } -> tensor<1x32x256xf32>
    %372 = tensor.empty() : tensor<1x32x256xf32>
    %373 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%368 : tensor<1x32x256xf32>) outs(%372 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_1", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb31(%374: f32, %375: f32):
      %376 = arith.constant -1.280000e+02 : f32
      %377 = arith.maximumf %374, %376 : f32
      %378 = arith.constant 1.270000e+02 : f32
      %379 = arith.minimumf %377, %378 : f32
      linalg.yield %379 : f32
    } -> tensor<1x32x256xf32>
    %380 = tensor.empty() : tensor<1x32x256xf32>
    %381 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%373, %356 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%380 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb32(%382: f32, %383: f32, %384: f32):
      %385 = arith.divf %382, %383 : f32
      linalg.yield %385 : f32
    } -> tensor<1x32x256xf32>
    %386 = "quant_ext.unpack_int2"(%80) <{bits = 2 : i64, lanes = 4 : i64}> {prov.op = "unpack_int2", prov.family = "quantize"} : (tensor<8192xi8>) -> tensor<8192x4xi8>
    %387 = tensor.collapse_shape %386 [[0 : i64, 1 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<8192x4xi8> into tensor<32768xi8>
    %388 = "tensor.extract_slice"(%387) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 32768>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %389 = tensor.empty() : tensor<32768xf32>
    %390 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%388 : tensor<32768xi8>) outs(%389 : tensor<32768xf32>) attrs =  {prov.region_id = "dtype_cast_4", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb33(%391: i8, %392: f32):
      %393 = arith.sitofp %391 : i8 to f32
      linalg.yield %393 : f32
    } -> tensor<32768xf32>
    %394 = arith.constant {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} 1.000000e+00 : f32
    %395 = tensor.splat %394 {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<32768xf32>
    %396 = tensor.empty() : tensor<32768xf32>
    %397 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%390, %395 : tensor<32768xf32>, tensor<32768xf32>) outs(%396 : tensor<32768xf32>) attrs =  {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb34(%398: f32, %399: f32, %400: f32):
      %401 = arith.subf %398, %399 : f32
      linalg.yield %401 : f32
    } -> tensor<32768xf32>
    %402 = func.call @aten_mul_Tensor_1(%397) {prov.region_id = "aten_mul_Tensor_1_0", prov.dispatch_id = "aten_mul_Tensor_1_0"} : (tensor<32768xf32>) -> tensor<32768xf32>
    %403 = tensor.expand_shape %402 [[0 : i64, 1 : i64]] output_shape [128, 256] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<32768xf32> into tensor<128x256xf32>
    %404 = tensor.empty() : tensor<256x128xf32>
    %405 = linalg.transpose ins(%403:tensor<128x256xf32>) outs(%404:tensor<256x128xf32>) permutation = [1, 0]
    %406 = tensor.empty() : tensor<1x32x128xf32>
    %407 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %408 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%407 : f32) outs(%406 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %409 = linalg.matmul {prov.region_id = "matmul_1", prov.dispatch_id = "matmul_1", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} ins(%381, %405 : tensor<1x32x256xf32>, tensor<256x128xf32>) outs(%408 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %410 = tensor.empty() : tensor<1x32x256xf32>
    %411 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%231 : tensor<1x32x256xf32>) outs(%410 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_2", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb35(%412: f32, %413: f32):
      %414 = math.absf %412 : f32
      linalg.yield %414 : f32
    } -> tensor<1x32x256xf32>
    %415 = arith.constant {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} 0xff800000 : f32
    %416 = arith.constant {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} 0 : i64
    %417 = tensor.splat %415 {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<1x32xf32>
    %418 = tensor.splat %416 {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<1x32xi64>
    %419, %420 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%411 : tensor<1x32x256xf32>) outs(%417, %418 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb36(%421: f32, %422: f32, %423: i64):
      %424 = linalg.index 2 : index
      %425 = arith.index_cast %424 : index to i64
      %426 = arith.cmpf ogt, %421, %422 : f32
      %427 = arith.select %426, %421, %422 : f32
      %428 = arith.select %426, %425, %423 : i64
      linalg.yield %427, %428 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %429 = tensor.collapse_shape %419 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %430 = tensor.expand_shape %429 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %431 = tensor.collapse_shape %420 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %432 = tensor.expand_shape %431 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %433 = func.call @aten_clamp__default(%430) {prov.region_id = "aten_clamp__default_2", prov.dispatch_id = "aten_clamp__default_2"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %434 = tensor.empty() : tensor<1x32x1xf32>
    %435 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%433 : tensor<1x32x1xf32>) outs(%434 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_2", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb37(%436: f32, %437: f32):
      %438 = arith.constant 1.000000e+00 : f32
      %439 = arith.divf %438, %436 : f32
      linalg.yield %439 : f32
    } -> tensor<1x32x1xf32>
    %440 = arith.constant {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} 1.270000e+02 : f32
    %441 = tensor.splat %440 {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<1x32x1xf32>
    %442 = tensor.empty() : tensor<1x32x1xf32>
    %443 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%435, %441 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%442 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb38(%444: f32, %445: f32, %446: f32):
      %447 = arith.mulf %444, %445 : f32
      linalg.yield %447 : f32
    } -> tensor<1x32x1xf32>
    %448 = tensor.empty() : tensor<1x32x256xf32>
    %449 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%231, %443 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%448 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb39(%450: f32, %451: f32, %452: f32):
      %453 = arith.mulf %450, %451 : f32
      linalg.yield %453 : f32
    } -> tensor<1x32x256xf32>
    %454 = tensor.empty() : tensor<1x32x256xf32>
    %455 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%449 : tensor<1x32x256xf32>) outs(%454 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_2", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb40(%456: f32, %457: f32):
      %458 = math.roundeven %456 : f32
      linalg.yield %458 : f32
    } -> tensor<1x32x256xf32>
    %459 = tensor.empty() : tensor<1x32x256xf32>
    %460 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%455 : tensor<1x32x256xf32>) outs(%459 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_2", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb41(%461: f32, %462: f32):
      %463 = arith.constant -1.280000e+02 : f32
      %464 = arith.maximumf %461, %463 : f32
      %465 = arith.constant 1.270000e+02 : f32
      %466 = arith.minimumf %464, %465 : f32
      linalg.yield %466 : f32
    } -> tensor<1x32x256xf32>
    %467 = tensor.empty() : tensor<1x32x256xf32>
    %468 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%460, %443 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%467 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb42(%469: f32, %470: f32, %471: f32):
      %472 = arith.divf %469, %470 : f32
      linalg.yield %472 : f32
    } -> tensor<1x32x256xf32>
    %473 = "quant_ext.unpack_int2"(%82) <{bits = 2 : i64, lanes = 4 : i64}> {prov.op = "unpack_int2", prov.family = "quantize"} : (tensor<8192xi8>) -> tensor<8192x4xi8>
    %474 = tensor.collapse_shape %473 [[0 : i64, 1 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<8192x4xi8> into tensor<32768xi8>
    %475 = "tensor.extract_slice"(%474) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 32768>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %476 = tensor.empty() : tensor<32768xf32>
    %477 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%475 : tensor<32768xi8>) outs(%476 : tensor<32768xf32>) attrs =  {prov.region_id = "dtype_cast_5", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb43(%478: i8, %479: f32):
      %480 = arith.sitofp %478 : i8 to f32
      linalg.yield %480 : f32
    } -> tensor<32768xf32>
    %481 = arith.constant {prov.region_id = "sub_2", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} 1.000000e+00 : f32
    %482 = tensor.splat %481 {prov.region_id = "sub_2", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<32768xf32>
    %483 = tensor.empty() : tensor<32768xf32>
    %484 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%477, %482 : tensor<32768xf32>, tensor<32768xf32>) outs(%483 : tensor<32768xf32>) attrs =  {prov.region_id = "sub_2", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb44(%485: f32, %486: f32, %487: f32):
      %488 = arith.subf %485, %486 : f32
      linalg.yield %488 : f32
    } -> tensor<32768xf32>
    %489 = func.call @aten_mul_Tensor_1(%484) {prov.region_id = "aten_mul_Tensor_1_1", prov.dispatch_id = "aten_mul_Tensor_1_1"} : (tensor<32768xf32>) -> tensor<32768xf32>
    %490 = tensor.expand_shape %489 [[0 : i64, 1 : i64]] output_shape [128, 256] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<32768xf32> into tensor<128x256xf32>
    %491 = tensor.empty() : tensor<256x128xf32>
    %492 = linalg.transpose ins(%490:tensor<128x256xf32>) outs(%491:tensor<256x128xf32>) permutation = [1, 0]
    %493 = tensor.empty() : tensor<1x32x128xf32>
    %494 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %495 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%494 : f32) outs(%493 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %496 = linalg.matmul {prov.region_id = "matmul_2", prov.dispatch_id = "matmul_2", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} ins(%468, %492 : tensor<1x32x256xf32>, tensor<256x128xf32>) outs(%495 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %497 = tensor.collapse_shape %322 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %498 = tensor.expand_shape %497 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 32] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x32x8x32xf32>
    %499 = tensor.empty() : tensor<1x8x32x32xf32>
    %500 = linalg.transpose ins(%498:tensor<1x32x8x32xf32>) outs(%499:tensor<1x8x32x32xf32>) permutation = [0, 2, 1, 3]
    %501 = tensor.collapse_shape %409 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %502 = tensor.expand_shape %501 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 32] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<4096xf32> into tensor<1x32x4x32xf32>
    %503 = tensor.empty() : tensor<1x4x32x32xf32>
    %504 = linalg.transpose ins(%502:tensor<1x32x4x32xf32>) outs(%503:tensor<1x4x32x32xf32>) permutation = [0, 2, 1, 3]
    %505 = tensor.collapse_shape %496 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %506 = tensor.expand_shape %505 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 32] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<4096xf32> into tensor<1x32x4x32xf32>
    %507 = tensor.empty() : tensor<1x4x32x32xf32>
    %508 = linalg.transpose ins(%506:tensor<1x32x4x32xf32>) outs(%507:tensor<1x4x32x32xf32>) permutation = [0, 2, 1, 3]
    %509 = "tensor.extract_slice"(%87) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.rotary_emb"} : (tensor<2048x32xf32>) -> tensor<32x32xf32>
    %510 = "tensor.extract_slice"(%88) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.rotary_emb"} : (tensor<2048x32xf32>) -> tensor<32x32xf32>
    %511 = tensor.empty() : tensor<1x32x32xf32>
    %512 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%123 : tensor<1x32xi64>) outs(%511 : tensor<1x32x32xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb45(%513: i64, %514: f32):
      %515 = arith.index_cast %513 : i64 to index
      %516 = linalg.index 2 : index
      %517 = tensor.extract %509[%515, %516] : tensor<32x32xf32>
      linalg.yield %517 : f32
    } -> tensor<1x32x32xf32>
    %518 = tensor.collapse_shape %512 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %519 = tensor.expand_shape %518 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %520 = tensor.empty() : tensor<1x32x32xf32>
    %521 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%123 : tensor<1x32xi64>) outs(%520 : tensor<1x32x32xf32>) attrs =  {prov.region_id = "gather_1", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb46(%522: i64, %523: f32):
      %524 = arith.index_cast %522 : i64 to index
      %525 = linalg.index 2 : index
      %526 = tensor.extract %510[%524, %525] : tensor<32x32xf32>
      linalg.yield %526 : f32
    } -> tensor<1x32x32xf32>
    %527 = tensor.collapse_shape %521 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %528 = tensor.expand_shape %527 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %529 = tensor.empty() : tensor<1x8x32x32xf32>
    %530 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%500, %519 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%529 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb47(%531: f32, %532: f32, %533: f32):
      %534 = arith.mulf %531, %532 : f32
      linalg.yield %534 : f32
    } -> tensor<1x8x32x32xf32>
    %535 = "tensor.extract_slice"(%500) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_9", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x8x32x32xf32>) -> tensor<1x8x32x16xf32>
    %536 = "tensor.extract_slice"(%500) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 8, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_10", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x8x32x32xf32>) -> tensor<1x8x32x16xf32>
    %537 = tensor.empty() : tensor<1x8x32x16xf32>
    %538 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%536 : tensor<1x8x32x16xf32>) outs(%537 : tensor<1x8x32x16xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb48(%539: f32, %540: f32):
      %541 = arith.negf %539 : f32
      linalg.yield %541 : f32
    } -> tensor<1x8x32x16xf32>
    %542 = tensor.concat dim(3) %538, %535 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x8x32x16xf32>, tensor<1x8x32x16xf32>) -> tensor<1x8x32x32xf32>
    %543 = tensor.empty() : tensor<1x8x32x32xf32>
    %544 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%542, %528 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%543 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb49(%545: f32, %546: f32, %547: f32):
      %548 = arith.mulf %545, %546 : f32
      linalg.yield %548 : f32
    } -> tensor<1x8x32x32xf32>
    %549 = tensor.empty() : tensor<1x8x32x32xf32>
    %550 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%530, %544 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%549 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb50(%551: f32, %552: f32, %553: f32):
      %554 = arith.addf %551, %552 : f32
      linalg.yield %554 : f32
    } -> tensor<1x8x32x32xf32>
    %555 = tensor.empty() : tensor<1x4x32x32xf32>
    %556 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%504, %519 : tensor<1x4x32x32xf32>, tensor<1x1x32x32xf32>) outs(%555 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb51(%557: f32, %558: f32, %559: f32):
      %560 = arith.mulf %557, %558 : f32
      linalg.yield %560 : f32
    } -> tensor<1x4x32x32xf32>
    %561 = "tensor.extract_slice"(%504) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_11", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x16xf32>
    %562 = "tensor.extract_slice"(%504) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_12", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x16xf32>
    %563 = tensor.empty() : tensor<1x4x32x16xf32>
    %564 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%562 : tensor<1x4x32x16xf32>) outs(%563 : tensor<1x4x32x16xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb52(%565: f32, %566: f32):
      %567 = arith.negf %565 : f32
      linalg.yield %567 : f32
    } -> tensor<1x4x32x16xf32>
    %568 = tensor.concat dim(3) %564, %561 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x16xf32>, tensor<1x4x32x16xf32>) -> tensor<1x4x32x32xf32>
    %569 = tensor.empty() : tensor<1x4x32x32xf32>
    %570 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%568, %528 : tensor<1x4x32x32xf32>, tensor<1x1x32x32xf32>) outs(%569 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb53(%571: f32, %572: f32, %573: f32):
      %574 = arith.mulf %571, %572 : f32
      linalg.yield %574 : f32
    } -> tensor<1x4x32x32xf32>
    %575 = tensor.empty() : tensor<1x4x32x32xf32>
    %576 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%556, %570 : tensor<1x4x32x32xf32>, tensor<1x4x32x32xf32>) outs(%575 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb54(%577: f32, %578: f32, %579: f32):
      %580 = arith.addf %577, %578 : f32
      linalg.yield %580 : f32
    } -> tensor<1x4x32x32xf32>
    %581 = "tensor.extract_slice"(%576) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_13", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %582 = "tensor.extract_slice"(%581) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_14", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %583 = tensor.collapse_shape %582 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x32x32xf32> into tensor<4096xf32>
    %584 = tensor.expand_shape %583 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 32, 32] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<4096xf32> into tensor<1x4x1x32x32xf32>
    %585 = "tensor.extract_slice"(%584) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_15", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %586 = "tensor.extract_slice"(%585) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_16", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %587 = tensor.empty() : tensor<1x4x2x32x32xf32>
    %588 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%586 : tensor<1x4x1x32x32xf32>) outs(%587 : tensor<1x4x2x32x32xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb55(%589: f32, %590: f32):
      linalg.yield %589 : f32
    } -> tensor<1x4x2x32x32xf32>
    %591 = tensor.collapse_shape %588 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x2x32x32xf32> into tensor<8192xf32>
    %592 = tensor.expand_shape %591 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 32] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x8x32x32xf32>
    %593 = "tensor.extract_slice"(%508) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_17", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %594 = "tensor.extract_slice"(%593) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_18", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %595 = tensor.collapse_shape %594 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x32x32xf32> into tensor<4096xf32>
    %596 = tensor.expand_shape %595 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 32, 32] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<4096xf32> into tensor<1x4x1x32x32xf32>
    %597 = "tensor.extract_slice"(%596) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_19", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %598 = "tensor.extract_slice"(%597) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_20", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %599 = tensor.empty() : tensor<1x4x2x32x32xf32>
    %600 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%598 : tensor<1x4x1x32x32xf32>) outs(%599 : tensor<1x4x2x32x32xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb56(%601: f32, %602: f32):
      linalg.yield %601 : f32
    } -> tensor<1x4x2x32x32xf32>
    %603 = tensor.collapse_shape %600 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x2x32x32xf32> into tensor<8192xf32>
    %604 = tensor.expand_shape %603 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 32] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x8x32x32xf32>
    %605 = tensor.empty() : tensor<1x8x32x32xf32>
    %606 = linalg.transpose ins(%592:tensor<1x8x32x32xf32>) outs(%605:tensor<1x8x32x32xf32>) permutation = [0, 1, 3, 2]
    %607 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 0.000000e+00 : f32
    %608 = tensor.splat %607 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32x32xf32>
    %609 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%550, %606 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%608 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb57(%610: f32, %611: f32, %612: f32):
      %613 = arith.mulf %610, %611 : f32
      %614 = arith.addf %612, %613 : f32
      linalg.yield %614 : f32
    } -> tensor<1x8x32x32xf32>
    %615 = arith.constant {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 5.65685415 : f32
    %616 = tensor.splat %615 {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32x32xf32>
    %617 = tensor.empty() : tensor<1x8x32x32xf32>
    %618 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%609, %616 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%617 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb58(%619: f32, %620: f32, %621: f32):
      %622 = arith.divf %619, %620 : f32
      linalg.yield %622 : f32
    } -> tensor<1x8x32x32xf32>
    %623 = "tensor.extract_slice"(%188) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_21", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    %624 = "tensor.extract_slice"(%623) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_22", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    %625 = "tensor.extract_slice"(%624) <{static_offsets = array<i64: 0, 0, 31, 0>, static_sizes = array<i64: 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<32xf32>
    %626 = tensor.expand_shape %625 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "slice_23", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<32xf32> into tensor<1x1x32xf32>
    %627 = tensor.collapse_shape %626 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x1x32xf32> into tensor<32xf32>
    %628 = tensor.expand_shape %627 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<32xf32> into tensor<1x1x1x32xf32>
    %629 = tensor.empty() : tensor<1x1x32x32xf32>
    %630 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%628 : tensor<1x1x1x32xf32>) outs(%629 : tensor<1x1x32x32xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb59(%631: f32, %632: f32):
      linalg.yield %631 : f32
    } -> tensor<1x1x32x32xf32>
    %633 = tensor.empty() : tensor<1x8x32x32xf32>
    %634 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%618, %630 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%633 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb60(%635: f32, %636: f32, %637: f32):
      %638 = arith.addf %635, %636 : f32
      linalg.yield %638 : f32
    } -> tensor<1x8x32x32xf32>
    %639 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 0xff800000 : f32
    %640 = tensor.splat %639 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32xf32>
    %641 = linalg.reduce ins(%634:tensor<1x8x32x32xf32>) outs(%640:tensor<1x8x32xf32>) dimensions = [3]
    (%642: f32, %643: f32) {
      %644 = arith.maximumf %642, %643 : f32
      linalg.yield %644 : f32
    }
    %645 = tensor.collapse_shape %641 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %646 = tensor.expand_shape %645 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %647 = tensor.empty() : tensor<1x8x32x32xf32>
    %648 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%634, %646 : tensor<1x8x32x32xf32>, tensor<1x8x32x1xf32>) outs(%647 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb61(%649: f32, %650: f32, %651: f32):
      %652 = arith.subf %649, %650 : f32
      linalg.yield %652 : f32
    } -> tensor<1x8x32x32xf32>
    %653 = tensor.empty() : tensor<1x8x32x32xf32>
    %654 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%648 : tensor<1x8x32x32xf32>) outs(%653 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb62(%655: f32, %656: f32):
      %657 = math.exp %655 : f32
      linalg.yield %657 : f32
    } -> tensor<1x8x32x32xf32>
    %658 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 0.000000e+00 : f32
    %659 = tensor.splat %658 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32xf32>
    %660 = linalg.reduce ins(%654:tensor<1x8x32x32xf32>) outs(%659:tensor<1x8x32xf32>) dimensions = [3]
    (%661: f32, %662: f32) {
      %663 = arith.addf %661, %662 : f32
      linalg.yield %663 : f32
    }
    %664 = tensor.collapse_shape %660 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %665 = tensor.expand_shape %664 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %666 = tensor.empty() : tensor<1x8x32x32xf32>
    %667 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%654, %665 : tensor<1x8x32x32xf32>, tensor<1x8x32x1xf32>) outs(%666 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb63(%668: f32, %669: f32, %670: f32):
      %671 = arith.divf %668, %669 : f32
      linalg.yield %671 : f32
    } -> tensor<1x8x32x32xf32>
    %672 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 0.000000e+00 : f32
    %673 = tensor.splat %672 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32x32xf32>
    %674 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%667, %604 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%673 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb64(%675: f32, %676: f32, %677: f32):
      %678 = arith.mulf %675, %676 : f32
      %679 = arith.addf %677, %678 : f32
      linalg.yield %679 : f32
    } -> tensor<1x8x32x32xf32>
    %680 = tensor.empty() : tensor<1x32x8x32xf32>
    %681 = linalg.transpose ins(%674:tensor<1x8x32x32xf32>) outs(%680:tensor<1x32x8x32xf32>) permutation = [0, 2, 1, 3]
    %682 = tensor.collapse_shape %681 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x8x32xf32> into tensor<8192xf32>
    %683 = tensor.expand_shape %682 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 256] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x32x256xf32>
    %684 = tensor.empty() : tensor<1x32x256xf32>
    %685 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%683 : tensor<1x32x256xf32>) outs(%684 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb65(%686: f32, %687: f32):
      %688 = arith.constant 2.000000e+00 : f32
      %689 = math.powf %686, %688 : f32
      linalg.yield %689 : f32
    } -> tensor<1x32x256xf32>
    %690 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} 0.000000e+00 : f32
    %691 = tensor.splat %690 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} : tensor<1x32xf32>
    %692 = linalg.reduce ins(%685:tensor<1x32x256xf32>) outs(%691:tensor<1x32xf32>) dimensions = [2]
    (%693: f32, %694: f32) {
      %695 = arith.addf %693, %694 : f32
      linalg.yield %695 : f32
    }
    %696 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} 2.560000e+02 : f32
    %697 = tensor.splat %696 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} : tensor<1x32xf32>
    %698 = tensor.empty() : tensor<1x32xf32>
    %699 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%692, %697 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%698 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb66(%700: f32, %701: f32, %702: f32):
      %703 = arith.divf %700, %701 : f32
      linalg.yield %703 : f32
    } -> tensor<1x32xf32>
    %704 = tensor.collapse_shape %699 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} : tensor<1x32xf32> into tensor<32xf32>
    %705 = tensor.expand_shape %704 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %706 = arith.constant {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} 1.000000e-05 : f32
    %707 = tensor.splat %706 {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} : tensor<1x32x1xf32>
    %708 = tensor.empty() : tensor<1x32x1xf32>
    %709 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%705, %707 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%708 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb67(%710: f32, %711: f32, %712: f32):
      %713 = arith.addf %710, %711 : f32
      linalg.yield %713 : f32
    } -> tensor<1x32x1xf32>
    %714 = tensor.empty() : tensor<1x32x1xf32>
    %715 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%709 : tensor<1x32x1xf32>) outs(%714 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb68(%716: f32, %717: f32):
      %718 = math.rsqrt %716 : f32
      linalg.yield %718 : f32
    } -> tensor<1x32x1xf32>
    %719 = tensor.empty() : tensor<1x32x256xf32>
    %720 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%683, %715 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%719 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb69(%721: f32, %722: f32, %723: f32):
      %724 = arith.mulf %721, %722 : f32
      linalg.yield %724 : f32
    } -> tensor<1x32x256xf32>
    %725 = tensor.empty() : tensor<1x32x256xf32>
    %726 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %720 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%725 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb70(%727: f32, %728: f32, %729: f32):
      %730 = arith.mulf %727, %728 : f32
      linalg.yield %730 : f32
    } -> tensor<1x32x256xf32>
    %731 = tensor.empty() : tensor<1x32x256xf32>
    %732 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%726 : tensor<1x32x256xf32>) outs(%731 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_3", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb71(%733: f32, %734: f32):
      %735 = math.absf %733 : f32
      linalg.yield %735 : f32
    } -> tensor<1x32x256xf32>
    %736 = arith.constant {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} 0xff800000 : f32
    %737 = arith.constant {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} 0 : i64
    %738 = tensor.splat %736 {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<1x32xf32>
    %739 = tensor.splat %737 {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<1x32xi64>
    %740, %741 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%732 : tensor<1x32x256xf32>) outs(%738, %739 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb72(%742: f32, %743: f32, %744: i64):
      %745 = linalg.index 2 : index
      %746 = arith.index_cast %745 : index to i64
      %747 = arith.cmpf ogt, %742, %743 : f32
      %748 = arith.select %747, %742, %743 : f32
      %749 = arith.select %747, %746, %744 : i64
      linalg.yield %748, %749 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %750 = tensor.collapse_shape %740 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %751 = tensor.expand_shape %750 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %752 = tensor.collapse_shape %741 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %753 = tensor.expand_shape %752 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %754 = func.call @aten_clamp__default(%751) {prov.region_id = "aten_clamp__default_3", prov.dispatch_id = "aten_clamp__default_3"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %755 = tensor.empty() : tensor<1x32x1xf32>
    %756 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%754 : tensor<1x32x1xf32>) outs(%755 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_3", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb73(%757: f32, %758: f32):
      %759 = arith.constant 1.000000e+00 : f32
      %760 = arith.divf %759, %757 : f32
      linalg.yield %760 : f32
    } -> tensor<1x32x1xf32>
    %761 = arith.constant {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} 1.270000e+02 : f32
    %762 = tensor.splat %761 {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<1x32x1xf32>
    %763 = tensor.empty() : tensor<1x32x1xf32>
    %764 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%756, %762 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%763 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb74(%765: f32, %766: f32, %767: f32):
      %768 = arith.mulf %765, %766 : f32
      linalg.yield %768 : f32
    } -> tensor<1x32x1xf32>
    %769 = tensor.empty() : tensor<1x32x256xf32>
    %770 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%726, %764 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%769 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb75(%771: f32, %772: f32, %773: f32):
      %774 = arith.mulf %771, %772 : f32
      linalg.yield %774 : f32
    } -> tensor<1x32x256xf32>
    %775 = tensor.empty() : tensor<1x32x256xf32>
    %776 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%770 : tensor<1x32x256xf32>) outs(%775 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_3", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb76(%777: f32, %778: f32):
      %779 = math.roundeven %777 : f32
      linalg.yield %779 : f32
    } -> tensor<1x32x256xf32>
    %780 = tensor.empty() : tensor<1x32x256xf32>
    %781 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%776 : tensor<1x32x256xf32>) outs(%780 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_3", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb77(%782: f32, %783: f32):
      %784 = arith.constant -1.280000e+02 : f32
      %785 = arith.maximumf %782, %784 : f32
      %786 = arith.constant 1.270000e+02 : f32
      %787 = arith.minimumf %785, %786 : f32
      linalg.yield %787 : f32
    } -> tensor<1x32x256xf32>
    %788 = tensor.empty() : tensor<1x32x256xf32>
    %789 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%781, %764 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%788 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb78(%790: f32, %791: f32, %792: f32):
      %793 = arith.divf %790, %791 : f32
      linalg.yield %793 : f32
    } -> tensor<1x32x256xf32>
    %794 = "quant_ext.unpack_int2"(%84) <{bits = 2 : i64, lanes = 4 : i64}> {prov.op = "unpack_int2", prov.family = "quantize"} : (tensor<16384xi8>) -> tensor<16384x4xi8>
    %795 = tensor.collapse_shape %794 [[0 : i64, 1 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<16384x4xi8> into tensor<65536xi8>
    %796 = "tensor.extract_slice"(%795) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 65536>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_24", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : (tensor<65536xi8>) -> tensor<65536xi8>
    %797 = tensor.empty() : tensor<65536xf32>
    %798 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%796 : tensor<65536xi8>) outs(%797 : tensor<65536xf32>) attrs =  {prov.region_id = "dtype_cast_6", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb79(%799: i8, %800: f32):
      %801 = arith.sitofp %799 : i8 to f32
      linalg.yield %801 : f32
    } -> tensor<65536xf32>
    %802 = arith.constant {prov.region_id = "sub_3", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} 1.000000e+00 : f32
    %803 = tensor.splat %802 {prov.region_id = "sub_3", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<65536xf32>
    %804 = tensor.empty() : tensor<65536xf32>
    %805 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%798, %803 : tensor<65536xf32>, tensor<65536xf32>) outs(%804 : tensor<65536xf32>) attrs =  {prov.region_id = "sub_3", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb80(%806: f32, %807: f32, %808: f32):
      %809 = arith.subf %806, %807 : f32
      linalg.yield %809 : f32
    } -> tensor<65536xf32>
    %810 = func.call @aten_mul_Tensor(%805) {prov.region_id = "aten_mul_Tensor_1", prov.dispatch_id = "aten_mul_Tensor_1"} : (tensor<65536xf32>) -> tensor<65536xf32>
    %811 = tensor.expand_shape %810 [[0 : i64, 1 : i64]] output_shape [256, 256] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<65536xf32> into tensor<256x256xf32>
    %812 = tensor.empty() : tensor<256x256xf32>
    %813 = linalg.transpose ins(%811:tensor<256x256xf32>) outs(%812:tensor<256x256xf32>) permutation = [1, 0]
    %814 = tensor.empty() : tensor<1x32x256xf32>
    %815 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %816 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%815 : f32) outs(%814 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %817 = linalg.matmul {prov.region_id = "matmul_5", prov.dispatch_id = "matmul_5", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} ins(%789, %813 : tensor<1x32x256xf32>, tensor<256x256xf32>) outs(%816 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %818 = tensor.empty() : tensor<1x32x256xf32>
    %819 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%112, %817 : tensor<1x32x256xf32>, tensor<1x32x256xf32>) outs(%818 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0"} {
    ^bb81(%820: f32, %821: f32, %822: f32):
      %823 = arith.addf %820, %821 : f32
      linalg.yield %823 : f32
    } -> tensor<1x32x256xf32>
    %824 = tensor.empty() : tensor<1x32x256xf32>
    %825 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%819 : tensor<1x32x256xf32>) outs(%824 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb82(%826: f32, %827: f32):
      %828 = arith.constant 2.000000e+00 : f32
      %829 = math.powf %826, %828 : f32
      linalg.yield %829 : f32
    } -> tensor<1x32x256xf32>
    %830 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} 0.000000e+00 : f32
    %831 = tensor.splat %830 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<1x32xf32>
    %832 = linalg.reduce ins(%825:tensor<1x32x256xf32>) outs(%831:tensor<1x32xf32>) dimensions = [2]
    (%833: f32, %834: f32) {
      %835 = arith.addf %833, %834 : f32
      linalg.yield %835 : f32
    }
    %836 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} 2.560000e+02 : f32
    %837 = tensor.splat %836 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<1x32xf32>
    %838 = tensor.empty() : tensor<1x32xf32>
    %839 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%832, %837 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%838 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb83(%840: f32, %841: f32, %842: f32):
      %843 = arith.divf %840, %841 : f32
      linalg.yield %843 : f32
    } -> tensor<1x32xf32>
    %844 = tensor.collapse_shape %839 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %845 = tensor.expand_shape %844 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %846 = arith.constant {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} 1.000000e-05 : f32
    %847 = tensor.splat %846 {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<1x32x1xf32>
    %848 = tensor.empty() : tensor<1x32x1xf32>
    %849 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%845, %847 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%848 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb84(%850: f32, %851: f32, %852: f32):
      %853 = arith.addf %850, %851 : f32
      linalg.yield %853 : f32
    } -> tensor<1x32x1xf32>
    %854 = tensor.empty() : tensor<1x32x1xf32>
    %855 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%849 : tensor<1x32x1xf32>) outs(%854 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb85(%856: f32, %857: f32):
      %858 = math.rsqrt %856 : f32
      linalg.yield %858 : f32
    } -> tensor<1x32x1xf32>
    %859 = tensor.empty() : tensor<1x32x256xf32>
    %860 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%819, %855 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%859 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb86(%861: f32, %862: f32, %863: f32):
      %864 = arith.mulf %861, %862 : f32
      linalg.yield %864 : f32
    } -> tensor<1x32x256xf32>
    %865 = tensor.empty() : tensor<1x32x256xf32>
    %866 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %860 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%865 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb87(%867: f32, %868: f32, %869: f32):
      %870 = arith.mulf %867, %868 : f32
      linalg.yield %870 : f32
    } -> tensor<1x32x256xf32>
    %871 = tensor.empty() : tensor<1x32x256xf32>
    %872 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%866 : tensor<1x32x256xf32>) outs(%871 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_4", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb88(%873: f32, %874: f32):
      %875 = math.absf %873 : f32
      linalg.yield %875 : f32
    } -> tensor<1x32x256xf32>
    %876 = arith.constant {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} 0xff800000 : f32
    %877 = arith.constant {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} 0 : i64
    %878 = tensor.splat %876 {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<1x32xf32>
    %879 = tensor.splat %877 {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<1x32xi64>
    %880, %881 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%872 : tensor<1x32x256xf32>) outs(%878, %879 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb89(%882: f32, %883: f32, %884: i64):
      %885 = linalg.index 2 : index
      %886 = arith.index_cast %885 : index to i64
      %887 = arith.cmpf ogt, %882, %883 : f32
      %888 = arith.select %887, %882, %883 : f32
      %889 = arith.select %887, %886, %884 : i64
      linalg.yield %888, %889 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %890 = tensor.collapse_shape %880 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %891 = tensor.expand_shape %890 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %892 = tensor.collapse_shape %881 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %893 = tensor.expand_shape %892 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %894 = func.call @aten_clamp__default(%891) {prov.region_id = "aten_clamp__default_4", prov.dispatch_id = "aten_clamp__default_4"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %895 = tensor.empty() : tensor<1x32x1xf32>
    %896 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%894 : tensor<1x32x1xf32>) outs(%895 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_4", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb90(%897: f32, %898: f32):
      %899 = arith.constant 1.000000e+00 : f32
      %900 = arith.divf %899, %897 : f32
      linalg.yield %900 : f32
    } -> tensor<1x32x1xf32>
    %901 = arith.constant {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} 1.270000e+02 : f32
    %902 = tensor.splat %901 {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<1x32x1xf32>
    %903 = tensor.empty() : tensor<1x32x1xf32>
    %904 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%896, %902 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%903 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb91(%905: f32, %906: f32, %907: f32):
      %908 = arith.mulf %905, %906 : f32
      linalg.yield %908 : f32
    } -> tensor<1x32x1xf32>
    %909 = tensor.empty() : tensor<1x32x256xf32>
    %910 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%866, %904 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%909 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb92(%911: f32, %912: f32, %913: f32):
      %914 = arith.mulf %911, %912 : f32
      linalg.yield %914 : f32
    } -> tensor<1x32x256xf32>
    %915 = tensor.empty() : tensor<1x32x256xf32>
    %916 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%910 : tensor<1x32x256xf32>) outs(%915 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_4", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb93(%917: f32, %918: f32):
      %919 = math.roundeven %917 : f32
      linalg.yield %919 : f32
    } -> tensor<1x32x256xf32>
    %920 = tensor.empty() : tensor<1x32x256xf32>
    %921 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%916 : tensor<1x32x256xf32>) outs(%920 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_4", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb94(%922: f32, %923: f32):
      %924 = arith.constant -1.280000e+02 : f32
      %925 = arith.maximumf %922, %924 : f32
      %926 = arith.constant 1.270000e+02 : f32
      %927 = arith.minimumf %925, %926 : f32
      linalg.yield %927 : f32
    } -> tensor<1x32x256xf32>
    %928 = tensor.empty() : tensor<1x32x256xf32>
    %929 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%921, %904 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%928 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb95(%930: f32, %931: f32, %932: f32):
      %933 = arith.divf %930, %931 : f32
      linalg.yield %933 : f32
    } -> tensor<1x32x256xf32>
    %934 = "quant_ext.unpack_int2"(%89) <{bits = 2 : i64, lanes = 4 : i64}> {prov.op = "unpack_int2", prov.family = "quantize"} : (tensor<32768xi8>) -> tensor<32768x4xi8>
    %935 = tensor.collapse_shape %934 [[0 : i64, 1 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<32768x4xi8> into tensor<131072xi8>
    %936 = "tensor.extract_slice"(%935) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 131072>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_25", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : (tensor<131072xi8>) -> tensor<131072xi8>
    %937 = tensor.empty() : tensor<131072xf32>
    %938 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%936 : tensor<131072xi8>) outs(%937 : tensor<131072xf32>) attrs =  {prov.region_id = "dtype_cast_7", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb96(%939: i8, %940: f32):
      %941 = arith.sitofp %939 : i8 to f32
      linalg.yield %941 : f32
    } -> tensor<131072xf32>
    %942 = arith.constant {prov.region_id = "sub_4", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} 1.000000e+00 : f32
    %943 = tensor.splat %942 {prov.region_id = "sub_4", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<131072xf32>
    %944 = tensor.empty() : tensor<131072xf32>
    %945 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%938, %943 : tensor<131072xf32>, tensor<131072xf32>) outs(%944 : tensor<131072xf32>) attrs =  {prov.region_id = "sub_4", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb97(%946: f32, %947: f32, %948: f32):
      %949 = arith.subf %946, %947 : f32
      linalg.yield %949 : f32
    } -> tensor<131072xf32>
    %950 = func.call @aten_mul_Tensor_2(%945) {prov.region_id = "aten_mul_Tensor_2_0", prov.dispatch_id = "aten_mul_Tensor_2_0"} : (tensor<131072xf32>) -> tensor<131072xf32>
    %951 = tensor.expand_shape %950 [[0 : i64, 1 : i64]] output_shape [512, 256] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<131072xf32> into tensor<512x256xf32>
    %952 = tensor.empty() : tensor<256x512xf32>
    %953 = linalg.transpose ins(%951:tensor<512x256xf32>) outs(%952:tensor<256x512xf32>) permutation = [1, 0]
    %954 = tensor.empty() : tensor<1x32x512xf32>
    %955 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %956 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%955 : f32) outs(%954 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %957 = linalg.matmul {prov.region_id = "matmul_6", prov.dispatch_id = "matmul_6", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} ins(%929, %953 : tensor<1x32x256xf32>, tensor<256x512xf32>) outs(%956 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %958 = tensor.empty() : tensor<1x32x512xf32>
    %959 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%957 : tensor<1x32x512xf32>) outs(%958 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "minmax_5", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp"} {
    ^bb98(%960: f32, %961: f32):
      %962 = arith.constant 0.000000e+00 : f32
      %963 = arith.maximumf %960, %962 : f32
      linalg.yield %963 : f32
    } -> tensor<1x32x512xf32>
    %964 = tensor.empty() : tensor<1x32x512xf32>
    %965 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%959 : tensor<1x32x512xf32>) outs(%964 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp"} {
    ^bb99(%966: f32, %967: f32):
      %968 = arith.constant 2.000000e+00 : f32
      %969 = math.powf %966, %968 : f32
      linalg.yield %969 : f32
    } -> tensor<1x32x512xf32>
    %970 = tensor.empty() : tensor<1x32x256xf32>
    %971 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%866 : tensor<1x32x256xf32>) outs(%970 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_5", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb100(%972: f32, %973: f32):
      %974 = math.absf %972 : f32
      linalg.yield %974 : f32
    } -> tensor<1x32x256xf32>
    %975 = arith.constant {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} 0xff800000 : f32
    %976 = arith.constant {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} 0 : i64
    %977 = tensor.splat %975 {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<1x32xf32>
    %978 = tensor.splat %976 {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<1x32xi64>
    %979, %980 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%971 : tensor<1x32x256xf32>) outs(%977, %978 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb101(%981: f32, %982: f32, %983: i64):
      %984 = linalg.index 2 : index
      %985 = arith.index_cast %984 : index to i64
      %986 = arith.cmpf ogt, %981, %982 : f32
      %987 = arith.select %986, %981, %982 : f32
      %988 = arith.select %986, %985, %983 : i64
      linalg.yield %987, %988 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %989 = tensor.collapse_shape %979 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %990 = tensor.expand_shape %989 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %991 = tensor.collapse_shape %980 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %992 = tensor.expand_shape %991 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %993 = func.call @aten_clamp__default(%990) {prov.region_id = "aten_clamp__default_5", prov.dispatch_id = "aten_clamp__default_5"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %994 = tensor.empty() : tensor<1x32x1xf32>
    %995 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%993 : tensor<1x32x1xf32>) outs(%994 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_5", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb102(%996: f32, %997: f32):
      %998 = arith.constant 1.000000e+00 : f32
      %999 = arith.divf %998, %996 : f32
      linalg.yield %999 : f32
    } -> tensor<1x32x1xf32>
    %1000 = arith.constant {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} 1.270000e+02 : f32
    %1001 = tensor.splat %1000 {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<1x32x1xf32>
    %1002 = tensor.empty() : tensor<1x32x1xf32>
    %1003 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%995, %1001 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1002 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb103(%1004: f32, %1005: f32, %1006: f32):
      %1007 = arith.mulf %1004, %1005 : f32
      linalg.yield %1007 : f32
    } -> tensor<1x32x1xf32>
    %1008 = tensor.empty() : tensor<1x32x256xf32>
    %1009 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%866, %1003 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1008 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb104(%1010: f32, %1011: f32, %1012: f32):
      %1013 = arith.mulf %1010, %1011 : f32
      linalg.yield %1013 : f32
    } -> tensor<1x32x256xf32>
    %1014 = tensor.empty() : tensor<1x32x256xf32>
    %1015 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1009 : tensor<1x32x256xf32>) outs(%1014 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_5", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb105(%1016: f32, %1017: f32):
      %1018 = math.roundeven %1016 : f32
      linalg.yield %1018 : f32
    } -> tensor<1x32x256xf32>
    %1019 = tensor.empty() : tensor<1x32x256xf32>
    %1020 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1015 : tensor<1x32x256xf32>) outs(%1019 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_6", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb106(%1021: f32, %1022: f32):
      %1023 = arith.constant -1.280000e+02 : f32
      %1024 = arith.maximumf %1021, %1023 : f32
      %1025 = arith.constant 1.270000e+02 : f32
      %1026 = arith.minimumf %1024, %1025 : f32
      linalg.yield %1026 : f32
    } -> tensor<1x32x256xf32>
    %1027 = tensor.empty() : tensor<1x32x256xf32>
    %1028 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1020, %1003 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1027 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_6", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb107(%1029: f32, %1030: f32, %1031: f32):
      %1032 = arith.divf %1029, %1030 : f32
      linalg.yield %1032 : f32
    } -> tensor<1x32x256xf32>
    %1033 = "quant_ext.unpack_int2"(%91) <{bits = 2 : i64, lanes = 4 : i64}> {prov.op = "unpack_int2", prov.family = "quantize"} : (tensor<32768xi8>) -> tensor<32768x4xi8>
    %1034 = tensor.collapse_shape %1033 [[0 : i64, 1 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<32768x4xi8> into tensor<131072xi8>
    %1035 = "tensor.extract_slice"(%1034) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 131072>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_26", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : (tensor<131072xi8>) -> tensor<131072xi8>
    %1036 = tensor.empty() : tensor<131072xf32>
    %1037 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1035 : tensor<131072xi8>) outs(%1036 : tensor<131072xf32>) attrs =  {prov.region_id = "dtype_cast_8", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb108(%1038: i8, %1039: f32):
      %1040 = arith.sitofp %1038 : i8 to f32
      linalg.yield %1040 : f32
    } -> tensor<131072xf32>
    %1041 = arith.constant {prov.region_id = "sub_5", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} 1.000000e+00 : f32
    %1042 = tensor.splat %1041 {prov.region_id = "sub_5", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<131072xf32>
    %1043 = tensor.empty() : tensor<131072xf32>
    %1044 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1037, %1042 : tensor<131072xf32>, tensor<131072xf32>) outs(%1043 : tensor<131072xf32>) attrs =  {prov.region_id = "sub_5", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb109(%1045: f32, %1046: f32, %1047: f32):
      %1048 = arith.subf %1045, %1046 : f32
      linalg.yield %1048 : f32
    } -> tensor<131072xf32>
    %1049 = func.call @aten_mul_Tensor_2(%1044) {prov.region_id = "aten_mul_Tensor_2_1", prov.dispatch_id = "aten_mul_Tensor_2_1"} : (tensor<131072xf32>) -> tensor<131072xf32>
    %1050 = tensor.expand_shape %1049 [[0 : i64, 1 : i64]] output_shape [512, 256] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<131072xf32> into tensor<512x256xf32>
    %1051 = tensor.empty() : tensor<256x512xf32>
    %1052 = linalg.transpose ins(%1050:tensor<512x256xf32>) outs(%1051:tensor<256x512xf32>) permutation = [1, 0]
    %1053 = tensor.empty() : tensor<1x32x512xf32>
    %1054 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1055 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1054 : f32) outs(%1053 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %1056 = linalg.matmul {prov.region_id = "matmul_7", prov.dispatch_id = "matmul_7", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} ins(%1028, %1052 : tensor<1x32x256xf32>, tensor<256x512xf32>) outs(%1055 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %1057 = tensor.empty() : tensor<1x32x512xf32>
    %1058 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%965, %1056 : tensor<1x32x512xf32>, tensor<1x32x512xf32>) outs(%1057 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp"} {
    ^bb110(%1059: f32, %1060: f32, %1061: f32):
      %1062 = arith.mulf %1059, %1060 : f32
      linalg.yield %1062 : f32
    } -> tensor<1x32x512xf32>
    %1063 = tensor.empty() : tensor<1x32x512xf32>
    %1064 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1058 : tensor<1x32x512xf32>) outs(%1063 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb111(%1065: f32, %1066: f32):
      %1067 = arith.constant 2.000000e+00 : f32
      %1068 = math.powf %1065, %1067 : f32
      linalg.yield %1068 : f32
    } -> tensor<1x32x512xf32>
    %1069 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} 0.000000e+00 : f32
    %1070 = tensor.splat %1069 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} : tensor<1x32xf32>
    %1071 = linalg.reduce ins(%1064:tensor<1x32x512xf32>) outs(%1070:tensor<1x32xf32>) dimensions = [2]
    (%1072: f32, %1073: f32) {
      %1074 = arith.addf %1072, %1073 : f32
      linalg.yield %1074 : f32
    }
    %1075 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} 5.120000e+02 : f32
    %1076 = tensor.splat %1075 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} : tensor<1x32xf32>
    %1077 = tensor.empty() : tensor<1x32xf32>
    %1078 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1071, %1076 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1077 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb112(%1079: f32, %1080: f32, %1081: f32):
      %1082 = arith.divf %1079, %1080 : f32
      linalg.yield %1082 : f32
    } -> tensor<1x32xf32>
    %1083 = tensor.collapse_shape %1078 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} : tensor<1x32xf32> into tensor<32xf32>
    %1084 = tensor.expand_shape %1083 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1085 = arith.constant {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} 1.000000e-05 : f32
    %1086 = tensor.splat %1085 {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} : tensor<1x32x1xf32>
    %1087 = tensor.empty() : tensor<1x32x1xf32>
    %1088 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1084, %1086 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1087 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb113(%1089: f32, %1090: f32, %1091: f32):
      %1092 = arith.addf %1089, %1090 : f32
      linalg.yield %1092 : f32
    } -> tensor<1x32x1xf32>
    %1093 = tensor.empty() : tensor<1x32x1xf32>
    %1094 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1088 : tensor<1x32x1xf32>) outs(%1093 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb114(%1095: f32, %1096: f32):
      %1097 = math.rsqrt %1095 : f32
      linalg.yield %1097 : f32
    } -> tensor<1x32x1xf32>
    %1098 = tensor.empty() : tensor<1x32x512xf32>
    %1099 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1058, %1094 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%1098 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb115(%1100: f32, %1101: f32, %1102: f32):
      %1103 = arith.mulf %1100, %1101 : f32
      linalg.yield %1103 : f32
    } -> tensor<1x32x512xf32>
    %1104 = tensor.empty() : tensor<1x32x512xf32>
    %1105 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %1099 : tensor<512xf32>, tensor<1x32x512xf32>) outs(%1104 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb116(%1106: f32, %1107: f32, %1108: f32):
      %1109 = arith.mulf %1106, %1107 : f32
      linalg.yield %1109 : f32
    } -> tensor<1x32x512xf32>
    %1110 = tensor.empty() : tensor<1x32x512xf32>
    %1111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1105 : tensor<1x32x512xf32>) outs(%1110 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "abs_6", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb117(%1112: f32, %1113: f32):
      %1114 = math.absf %1112 : f32
      linalg.yield %1114 : f32
    } -> tensor<1x32x512xf32>
    %1115 = arith.constant {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} 0xff800000 : f32
    %1116 = arith.constant {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} 0 : i64
    %1117 = tensor.splat %1115 {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<1x32xf32>
    %1118 = tensor.splat %1116 {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<1x32xi64>
    %1119, %1120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1111 : tensor<1x32x512xf32>) outs(%1117, %1118 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb118(%1121: f32, %1122: f32, %1123: i64):
      %1124 = linalg.index 2 : index
      %1125 = arith.index_cast %1124 : index to i64
      %1126 = arith.cmpf ogt, %1121, %1122 : f32
      %1127 = arith.select %1126, %1121, %1122 : f32
      %1128 = arith.select %1126, %1125, %1123 : i64
      linalg.yield %1127, %1128 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1129 = tensor.collapse_shape %1119 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1130 = tensor.expand_shape %1129 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1131 = tensor.collapse_shape %1120 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1132 = tensor.expand_shape %1131 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1133 = func.call @aten_clamp__default(%1130) {prov.region_id = "aten_clamp__default_6", prov.dispatch_id = "aten_clamp__default_6"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1134 = tensor.empty() : tensor<1x32x1xf32>
    %1135 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1133 : tensor<1x32x1xf32>) outs(%1134 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_6", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb119(%1136: f32, %1137: f32):
      %1138 = arith.constant 1.000000e+00 : f32
      %1139 = arith.divf %1138, %1136 : f32
      linalg.yield %1139 : f32
    } -> tensor<1x32x1xf32>
    %1140 = arith.constant {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} 1.270000e+02 : f32
    %1141 = tensor.splat %1140 {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<1x32x1xf32>
    %1142 = tensor.empty() : tensor<1x32x1xf32>
    %1143 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1135, %1141 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1142 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb120(%1144: f32, %1145: f32, %1146: f32):
      %1147 = arith.mulf %1144, %1145 : f32
      linalg.yield %1147 : f32
    } -> tensor<1x32x1xf32>
    %1148 = tensor.empty() : tensor<1x32x512xf32>
    %1149 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1105, %1143 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%1148 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb121(%1150: f32, %1151: f32, %1152: f32):
      %1153 = arith.mulf %1150, %1151 : f32
      linalg.yield %1153 : f32
    } -> tensor<1x32x512xf32>
    %1154 = tensor.empty() : tensor<1x32x512xf32>
    %1155 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1149 : tensor<1x32x512xf32>) outs(%1154 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "round_6", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb122(%1156: f32, %1157: f32):
      %1158 = math.roundeven %1156 : f32
      linalg.yield %1158 : f32
    } -> tensor<1x32x512xf32>
    %1159 = tensor.empty() : tensor<1x32x512xf32>
    %1160 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1155 : tensor<1x32x512xf32>) outs(%1159 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "minmax_7", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb123(%1161: f32, %1162: f32):
      %1163 = arith.constant -1.280000e+02 : f32
      %1164 = arith.maximumf %1161, %1163 : f32
      %1165 = arith.constant 1.270000e+02 : f32
      %1166 = arith.minimumf %1164, %1165 : f32
      linalg.yield %1166 : f32
    } -> tensor<1x32x512xf32>
    %1167 = tensor.empty() : tensor<1x32x512xf32>
    %1168 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1160, %1143 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%1167 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "div_7", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb124(%1169: f32, %1170: f32, %1171: f32):
      %1172 = arith.divf %1169, %1170 : f32
      linalg.yield %1172 : f32
    } -> tensor<1x32x512xf32>
    %1173 = "quant_ext.unpack_int2"(%93) <{bits = 2 : i64, lanes = 4 : i64}> {prov.op = "unpack_int2", prov.family = "quantize"} : (tensor<32768xi8>) -> tensor<32768x4xi8>
    %1174 = tensor.collapse_shape %1173 [[0 : i64, 1 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<32768x4xi8> into tensor<131072xi8>
    %1175 = "tensor.extract_slice"(%1174) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 131072>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_27", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : (tensor<131072xi8>) -> tensor<131072xi8>
    %1176 = tensor.empty() : tensor<131072xf32>
    %1177 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1175 : tensor<131072xi8>) outs(%1176 : tensor<131072xf32>) attrs =  {prov.region_id = "dtype_cast_9", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb125(%1178: i8, %1179: f32):
      %1180 = arith.sitofp %1178 : i8 to f32
      linalg.yield %1180 : f32
    } -> tensor<131072xf32>
    %1181 = arith.constant {prov.region_id = "sub_6", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} 1.000000e+00 : f32
    %1182 = tensor.splat %1181 {prov.region_id = "sub_6", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<131072xf32>
    %1183 = tensor.empty() : tensor<131072xf32>
    %1184 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1177, %1182 : tensor<131072xf32>, tensor<131072xf32>) outs(%1183 : tensor<131072xf32>) attrs =  {prov.region_id = "sub_6", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb126(%1185: f32, %1186: f32, %1187: f32):
      %1188 = arith.subf %1185, %1186 : f32
      linalg.yield %1188 : f32
    } -> tensor<131072xf32>
    %1189 = func.call @aten_mul_Tensor_2(%1184) {prov.region_id = "aten_mul_Tensor_2_2", prov.dispatch_id = "aten_mul_Tensor_2_2"} : (tensor<131072xf32>) -> tensor<131072xf32>
    %1190 = tensor.expand_shape %1189 [[0 : i64, 1 : i64]] output_shape [256, 512] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<131072xf32> into tensor<256x512xf32>
    %1191 = tensor.empty() : tensor<512x256xf32>
    %1192 = linalg.transpose ins(%1190:tensor<256x512xf32>) outs(%1191:tensor<512x256xf32>) permutation = [1, 0]
    %1193 = tensor.empty() : tensor<1x32x256xf32>
    %1194 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1195 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1194 : f32) outs(%1193 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %1196 = linalg.matmul {prov.region_id = "matmul_8", prov.dispatch_id = "matmul_8", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} ins(%1168, %1192 : tensor<1x32x512xf32>, tensor<512x256xf32>) outs(%1195 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %1197 = tensor.empty() : tensor<1x32x256xf32>
    %1198 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%819, %1196 : tensor<1x32x256xf32>, tensor<1x32x256xf32>) outs(%1197 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0"} {
    ^bb127(%1199: f32, %1200: f32, %1201: f32):
      %1202 = arith.addf %1199, %1200 : f32
      linalg.yield %1202 : f32
    } -> tensor<1x32x256xf32>
    %1203 = tensor.empty() : tensor<1x32x256xf32>
    %1204 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1198 : tensor<1x32x256xf32>) outs(%1203 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb128(%1205: f32, %1206: f32):
      %1207 = arith.constant 2.000000e+00 : f32
      %1208 = math.powf %1205, %1207 : f32
      linalg.yield %1208 : f32
    } -> tensor<1x32x256xf32>
    %1209 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} 0.000000e+00 : f32
    %1210 = tensor.splat %1209 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<1x32xf32>
    %1211 = linalg.reduce ins(%1204:tensor<1x32x256xf32>) outs(%1210:tensor<1x32xf32>) dimensions = [2]
    (%1212: f32, %1213: f32) {
      %1214 = arith.addf %1212, %1213 : f32
      linalg.yield %1214 : f32
    }
    %1215 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} 2.560000e+02 : f32
    %1216 = tensor.splat %1215 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<1x32xf32>
    %1217 = tensor.empty() : tensor<1x32xf32>
    %1218 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1211, %1216 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1217 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb129(%1219: f32, %1220: f32, %1221: f32):
      %1222 = arith.divf %1219, %1220 : f32
      linalg.yield %1222 : f32
    } -> tensor<1x32xf32>
    %1223 = tensor.collapse_shape %1218 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %1224 = tensor.expand_shape %1223 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1225 = arith.constant {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} 1.000000e-05 : f32
    %1226 = tensor.splat %1225 {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<1x32x1xf32>
    %1227 = tensor.empty() : tensor<1x32x1xf32>
    %1228 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1224, %1226 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1227 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb130(%1229: f32, %1230: f32, %1231: f32):
      %1232 = arith.addf %1229, %1230 : f32
      linalg.yield %1232 : f32
    } -> tensor<1x32x1xf32>
    %1233 = tensor.empty() : tensor<1x32x1xf32>
    %1234 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1228 : tensor<1x32x1xf32>) outs(%1233 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb131(%1235: f32, %1236: f32):
      %1237 = math.rsqrt %1235 : f32
      linalg.yield %1237 : f32
    } -> tensor<1x32x1xf32>
    %1238 = tensor.empty() : tensor<1x32x256xf32>
    %1239 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1198, %1234 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1238 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb132(%1240: f32, %1241: f32, %1242: f32):
      %1243 = arith.mulf %1240, %1241 : f32
      linalg.yield %1243 : f32
    } -> tensor<1x32x256xf32>
    %1244 = tensor.empty() : tensor<1x32x256xf32>
    %1245 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %1239 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%1244 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb133(%1246: f32, %1247: f32, %1248: f32):
      %1249 = arith.mulf %1246, %1247 : f32
      linalg.yield %1249 : f32
    } -> tensor<1x32x256xf32>
    %1250 = tensor.empty() : tensor<1x32x256xf32>
    %1251 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1245 : tensor<1x32x256xf32>) outs(%1250 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_7", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb134(%1252: f32, %1253: f32):
      %1254 = math.absf %1252 : f32
      linalg.yield %1254 : f32
    } -> tensor<1x32x256xf32>
    %1255 = arith.constant {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} 0xff800000 : f32
    %1256 = arith.constant {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} 0 : i64
    %1257 = tensor.splat %1255 {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<1x32xf32>
    %1258 = tensor.splat %1256 {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<1x32xi64>
    %1259, %1260 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1251 : tensor<1x32x256xf32>) outs(%1257, %1258 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb135(%1261: f32, %1262: f32, %1263: i64):
      %1264 = linalg.index 2 : index
      %1265 = arith.index_cast %1264 : index to i64
      %1266 = arith.cmpf ogt, %1261, %1262 : f32
      %1267 = arith.select %1266, %1261, %1262 : f32
      %1268 = arith.select %1266, %1265, %1263 : i64
      linalg.yield %1267, %1268 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1269 = tensor.collapse_shape %1259 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1270 = tensor.expand_shape %1269 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1271 = tensor.collapse_shape %1260 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1272 = tensor.expand_shape %1271 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1273 = func.call @aten_clamp__default(%1270) {prov.region_id = "aten_clamp__default_7", prov.dispatch_id = "aten_clamp__default_7"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1274 = tensor.empty() : tensor<1x32x1xf32>
    %1275 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1273 : tensor<1x32x1xf32>) outs(%1274 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_7", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb136(%1276: f32, %1277: f32):
      %1278 = arith.constant 1.000000e+00 : f32
      %1279 = arith.divf %1278, %1276 : f32
      linalg.yield %1279 : f32
    } -> tensor<1x32x1xf32>
    %1280 = arith.constant {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} 1.270000e+02 : f32
    %1281 = tensor.splat %1280 {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<1x32x1xf32>
    %1282 = tensor.empty() : tensor<1x32x1xf32>
    %1283 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1275, %1281 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1282 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb137(%1284: f32, %1285: f32, %1286: f32):
      %1287 = arith.mulf %1284, %1285 : f32
      linalg.yield %1287 : f32
    } -> tensor<1x32x1xf32>
    %1288 = tensor.empty() : tensor<1x32x256xf32>
    %1289 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1245, %1283 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1288 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb138(%1290: f32, %1291: f32, %1292: f32):
      %1293 = arith.mulf %1290, %1291 : f32
      linalg.yield %1293 : f32
    } -> tensor<1x32x256xf32>
    %1294 = tensor.empty() : tensor<1x32x256xf32>
    %1295 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1289 : tensor<1x32x256xf32>) outs(%1294 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_7", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb139(%1296: f32, %1297: f32):
      %1298 = math.roundeven %1296 : f32
      linalg.yield %1298 : f32
    } -> tensor<1x32x256xf32>
    %1299 = tensor.empty() : tensor<1x32x256xf32>
    %1300 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1295 : tensor<1x32x256xf32>) outs(%1299 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_8", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb140(%1301: f32, %1302: f32):
      %1303 = arith.constant -1.280000e+02 : f32
      %1304 = arith.maximumf %1301, %1303 : f32
      %1305 = arith.constant 1.270000e+02 : f32
      %1306 = arith.minimumf %1304, %1305 : f32
      linalg.yield %1306 : f32
    } -> tensor<1x32x256xf32>
    %1307 = tensor.empty() : tensor<1x32x256xf32>
    %1308 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1300, %1283 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1307 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_8", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb141(%1309: f32, %1310: f32, %1311: f32):
      %1312 = arith.divf %1309, %1310 : f32
      linalg.yield %1312 : f32
    } -> tensor<1x32x256xf32>
    %1313 = "quant_ext.unpack_int2"(%95) <{bits = 2 : i64, lanes = 4 : i64}> {prov.op = "unpack_int2", prov.family = "quantize"} : (tensor<16384xi8>) -> tensor<16384x4xi8>
    %1314 = tensor.collapse_shape %1313 [[0 : i64, 1 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<16384x4xi8> into tensor<65536xi8>
    %1315 = "tensor.extract_slice"(%1314) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 65536>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_28", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : (tensor<65536xi8>) -> tensor<65536xi8>
    %1316 = tensor.empty() : tensor<65536xf32>
    %1317 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1315 : tensor<65536xi8>) outs(%1316 : tensor<65536xf32>) attrs =  {prov.region_id = "dtype_cast_10", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb142(%1318: i8, %1319: f32):
      %1320 = arith.sitofp %1318 : i8 to f32
      linalg.yield %1320 : f32
    } -> tensor<65536xf32>
    %1321 = arith.constant {prov.region_id = "sub_7", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} 1.000000e+00 : f32
    %1322 = tensor.splat %1321 {prov.region_id = "sub_7", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<65536xf32>
    %1323 = tensor.empty() : tensor<65536xf32>
    %1324 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1317, %1322 : tensor<65536xf32>, tensor<65536xf32>) outs(%1323 : tensor<65536xf32>) attrs =  {prov.region_id = "sub_7", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb143(%1325: f32, %1326: f32, %1327: f32):
      %1328 = arith.subf %1325, %1326 : f32
      linalg.yield %1328 : f32
    } -> tensor<65536xf32>
    %1329 = func.call @aten_mul_Tensor(%1324) {prov.region_id = "aten_mul_Tensor_2", prov.dispatch_id = "aten_mul_Tensor_2"} : (tensor<65536xf32>) -> tensor<65536xf32>
    %1330 = tensor.expand_shape %1329 [[0 : i64, 1 : i64]] output_shape [256, 256] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<65536xf32> into tensor<256x256xf32>
    %1331 = tensor.empty() : tensor<256x256xf32>
    %1332 = linalg.transpose ins(%1330:tensor<256x256xf32>) outs(%1331:tensor<256x256xf32>) permutation = [1, 0]
    %1333 = tensor.empty() : tensor<1x32x256xf32>
    %1334 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1335 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1334 : f32) outs(%1333 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %1336 = linalg.matmul {prov.region_id = "matmul_9", prov.dispatch_id = "matmul_9", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} ins(%1308, %1332 : tensor<1x32x256xf32>, tensor<256x256xf32>) outs(%1335 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %1337 = tensor.empty() : tensor<1x32x256xf32>
    %1338 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1245 : tensor<1x32x256xf32>) outs(%1337 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_8", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb144(%1339: f32, %1340: f32):
      %1341 = math.absf %1339 : f32
      linalg.yield %1341 : f32
    } -> tensor<1x32x256xf32>
    %1342 = arith.constant {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} 0xff800000 : f32
    %1343 = arith.constant {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} 0 : i64
    %1344 = tensor.splat %1342 {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<1x32xf32>
    %1345 = tensor.splat %1343 {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<1x32xi64>
    %1346, %1347 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1338 : tensor<1x32x256xf32>) outs(%1344, %1345 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb145(%1348: f32, %1349: f32, %1350: i64):
      %1351 = linalg.index 2 : index
      %1352 = arith.index_cast %1351 : index to i64
      %1353 = arith.cmpf ogt, %1348, %1349 : f32
      %1354 = arith.select %1353, %1348, %1349 : f32
      %1355 = arith.select %1353, %1352, %1350 : i64
      linalg.yield %1354, %1355 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1356 = tensor.collapse_shape %1346 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1357 = tensor.expand_shape %1356 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1358 = tensor.collapse_shape %1347 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1359 = tensor.expand_shape %1358 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1360 = func.call @aten_clamp__default(%1357) {prov.region_id = "aten_clamp__default_8", prov.dispatch_id = "aten_clamp__default_8"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1361 = tensor.empty() : tensor<1x32x1xf32>
    %1362 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1360 : tensor<1x32x1xf32>) outs(%1361 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_8", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb146(%1363: f32, %1364: f32):
      %1365 = arith.constant 1.000000e+00 : f32
      %1366 = arith.divf %1365, %1363 : f32
      linalg.yield %1366 : f32
    } -> tensor<1x32x1xf32>
    %1367 = arith.constant {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} 1.270000e+02 : f32
    %1368 = tensor.splat %1367 {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<1x32x1xf32>
    %1369 = tensor.empty() : tensor<1x32x1xf32>
    %1370 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1362, %1368 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1369 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb147(%1371: f32, %1372: f32, %1373: f32):
      %1374 = arith.mulf %1371, %1372 : f32
      linalg.yield %1374 : f32
    } -> tensor<1x32x1xf32>
    %1375 = tensor.empty() : tensor<1x32x256xf32>
    %1376 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1245, %1370 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1375 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb148(%1377: f32, %1378: f32, %1379: f32):
      %1380 = arith.mulf %1377, %1378 : f32
      linalg.yield %1380 : f32
    } -> tensor<1x32x256xf32>
    %1381 = tensor.empty() : tensor<1x32x256xf32>
    %1382 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1376 : tensor<1x32x256xf32>) outs(%1381 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_8", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb149(%1383: f32, %1384: f32):
      %1385 = math.roundeven %1383 : f32
      linalg.yield %1385 : f32
    } -> tensor<1x32x256xf32>
    %1386 = tensor.empty() : tensor<1x32x256xf32>
    %1387 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1382 : tensor<1x32x256xf32>) outs(%1386 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_9", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb150(%1388: f32, %1389: f32):
      %1390 = arith.constant -1.280000e+02 : f32
      %1391 = arith.maximumf %1388, %1390 : f32
      %1392 = arith.constant 1.270000e+02 : f32
      %1393 = arith.minimumf %1391, %1392 : f32
      linalg.yield %1393 : f32
    } -> tensor<1x32x256xf32>
    %1394 = tensor.empty() : tensor<1x32x256xf32>
    %1395 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1387, %1370 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1394 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_9", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb151(%1396: f32, %1397: f32, %1398: f32):
      %1399 = arith.divf %1396, %1397 : f32
      linalg.yield %1399 : f32
    } -> tensor<1x32x256xf32>
    %1400 = "quant_ext.unpack_int2"(%97) <{bits = 2 : i64, lanes = 4 : i64}> {prov.op = "unpack_int2", prov.family = "quantize"} : (tensor<8192xi8>) -> tensor<8192x4xi8>
    %1401 = tensor.collapse_shape %1400 [[0 : i64, 1 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<8192x4xi8> into tensor<32768xi8>
    %1402 = "tensor.extract_slice"(%1401) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 32768>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_29", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1403 = tensor.empty() : tensor<32768xf32>
    %1404 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1402 : tensor<32768xi8>) outs(%1403 : tensor<32768xf32>) attrs =  {prov.region_id = "dtype_cast_11", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb152(%1405: i8, %1406: f32):
      %1407 = arith.sitofp %1405 : i8 to f32
      linalg.yield %1407 : f32
    } -> tensor<32768xf32>
    %1408 = arith.constant {prov.region_id = "sub_8", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} 1.000000e+00 : f32
    %1409 = tensor.splat %1408 {prov.region_id = "sub_8", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<32768xf32>
    %1410 = tensor.empty() : tensor<32768xf32>
    %1411 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1404, %1409 : tensor<32768xf32>, tensor<32768xf32>) outs(%1410 : tensor<32768xf32>) attrs =  {prov.region_id = "sub_8", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb153(%1412: f32, %1413: f32, %1414: f32):
      %1415 = arith.subf %1412, %1413 : f32
      linalg.yield %1415 : f32
    } -> tensor<32768xf32>
    %1416 = func.call @aten_mul_Tensor_1(%1411) {prov.region_id = "aten_mul_Tensor_1_2", prov.dispatch_id = "aten_mul_Tensor_1_2"} : (tensor<32768xf32>) -> tensor<32768xf32>
    %1417 = tensor.expand_shape %1416 [[0 : i64, 1 : i64]] output_shape [128, 256] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<32768xf32> into tensor<128x256xf32>
    %1418 = tensor.empty() : tensor<256x128xf32>
    %1419 = linalg.transpose ins(%1417:tensor<128x256xf32>) outs(%1418:tensor<256x128xf32>) permutation = [1, 0]
    %1420 = tensor.empty() : tensor<1x32x128xf32>
    %1421 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1422 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1421 : f32) outs(%1420 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %1423 = linalg.matmul {prov.region_id = "matmul_10", prov.dispatch_id = "matmul_10", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} ins(%1395, %1419 : tensor<1x32x256xf32>, tensor<256x128xf32>) outs(%1422 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %1424 = tensor.empty() : tensor<1x32x256xf32>
    %1425 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1245 : tensor<1x32x256xf32>) outs(%1424 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_9", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb154(%1426: f32, %1427: f32):
      %1428 = math.absf %1426 : f32
      linalg.yield %1428 : f32
    } -> tensor<1x32x256xf32>
    %1429 = arith.constant {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} 0xff800000 : f32
    %1430 = arith.constant {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} 0 : i64
    %1431 = tensor.splat %1429 {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<1x32xf32>
    %1432 = tensor.splat %1430 {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<1x32xi64>
    %1433, %1434 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1425 : tensor<1x32x256xf32>) outs(%1431, %1432 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb155(%1435: f32, %1436: f32, %1437: i64):
      %1438 = linalg.index 2 : index
      %1439 = arith.index_cast %1438 : index to i64
      %1440 = arith.cmpf ogt, %1435, %1436 : f32
      %1441 = arith.select %1440, %1435, %1436 : f32
      %1442 = arith.select %1440, %1439, %1437 : i64
      linalg.yield %1441, %1442 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1443 = tensor.collapse_shape %1433 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1444 = tensor.expand_shape %1443 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1445 = tensor.collapse_shape %1434 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1446 = tensor.expand_shape %1445 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1447 = func.call @aten_clamp__default(%1444) {prov.region_id = "aten_clamp__default_9", prov.dispatch_id = "aten_clamp__default_9"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1448 = tensor.empty() : tensor<1x32x1xf32>
    %1449 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1447 : tensor<1x32x1xf32>) outs(%1448 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_9", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb156(%1450: f32, %1451: f32):
      %1452 = arith.constant 1.000000e+00 : f32
      %1453 = arith.divf %1452, %1450 : f32
      linalg.yield %1453 : f32
    } -> tensor<1x32x1xf32>
    %1454 = arith.constant {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} 1.270000e+02 : f32
    %1455 = tensor.splat %1454 {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<1x32x1xf32>
    %1456 = tensor.empty() : tensor<1x32x1xf32>
    %1457 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1449, %1455 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1456 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb157(%1458: f32, %1459: f32, %1460: f32):
      %1461 = arith.mulf %1458, %1459 : f32
      linalg.yield %1461 : f32
    } -> tensor<1x32x1xf32>
    %1462 = tensor.empty() : tensor<1x32x256xf32>
    %1463 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1245, %1457 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1462 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_34", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb158(%1464: f32, %1465: f32, %1466: f32):
      %1467 = arith.mulf %1464, %1465 : f32
      linalg.yield %1467 : f32
    } -> tensor<1x32x256xf32>
    %1468 = tensor.empty() : tensor<1x32x256xf32>
    %1469 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1463 : tensor<1x32x256xf32>) outs(%1468 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_9", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb159(%1470: f32, %1471: f32):
      %1472 = math.roundeven %1470 : f32
      linalg.yield %1472 : f32
    } -> tensor<1x32x256xf32>
    %1473 = tensor.empty() : tensor<1x32x256xf32>
    %1474 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1469 : tensor<1x32x256xf32>) outs(%1473 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_10", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb160(%1475: f32, %1476: f32):
      %1477 = arith.constant -1.280000e+02 : f32
      %1478 = arith.maximumf %1475, %1477 : f32
      %1479 = arith.constant 1.270000e+02 : f32
      %1480 = arith.minimumf %1478, %1479 : f32
      linalg.yield %1480 : f32
    } -> tensor<1x32x256xf32>
    %1481 = tensor.empty() : tensor<1x32x256xf32>
    %1482 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1474, %1457 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1481 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_10", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb161(%1483: f32, %1484: f32, %1485: f32):
      %1486 = arith.divf %1483, %1484 : f32
      linalg.yield %1486 : f32
    } -> tensor<1x32x256xf32>
    %1487 = "quant_ext.unpack_int2"(%99) <{bits = 2 : i64, lanes = 4 : i64}> {prov.op = "unpack_int2", prov.family = "quantize"} : (tensor<8192xi8>) -> tensor<8192x4xi8>
    %1488 = tensor.collapse_shape %1487 [[0 : i64, 1 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<8192x4xi8> into tensor<32768xi8>
    %1489 = "tensor.extract_slice"(%1488) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 32768>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_30", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1490 = tensor.empty() : tensor<32768xf32>
    %1491 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1489 : tensor<32768xi8>) outs(%1490 : tensor<32768xf32>) attrs =  {prov.region_id = "dtype_cast_12", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb162(%1492: i8, %1493: f32):
      %1494 = arith.sitofp %1492 : i8 to f32
      linalg.yield %1494 : f32
    } -> tensor<32768xf32>
    %1495 = arith.constant {prov.region_id = "sub_9", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} 1.000000e+00 : f32
    %1496 = tensor.splat %1495 {prov.region_id = "sub_9", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<32768xf32>
    %1497 = tensor.empty() : tensor<32768xf32>
    %1498 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1491, %1496 : tensor<32768xf32>, tensor<32768xf32>) outs(%1497 : tensor<32768xf32>) attrs =  {prov.region_id = "sub_9", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb163(%1499: f32, %1500: f32, %1501: f32):
      %1502 = arith.subf %1499, %1500 : f32
      linalg.yield %1502 : f32
    } -> tensor<32768xf32>
    %1503 = func.call @aten_mul_Tensor_1(%1498) {prov.region_id = "aten_mul_Tensor_1_3", prov.dispatch_id = "aten_mul_Tensor_1_3"} : (tensor<32768xf32>) -> tensor<32768xf32>
    %1504 = tensor.expand_shape %1503 [[0 : i64, 1 : i64]] output_shape [128, 256] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<32768xf32> into tensor<128x256xf32>
    %1505 = tensor.empty() : tensor<256x128xf32>
    %1506 = linalg.transpose ins(%1504:tensor<128x256xf32>) outs(%1505:tensor<256x128xf32>) permutation = [1, 0]
    %1507 = tensor.empty() : tensor<1x32x128xf32>
    %1508 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1509 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1508 : f32) outs(%1507 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %1510 = linalg.matmul {prov.region_id = "matmul_11", prov.dispatch_id = "matmul_11", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} ins(%1482, %1506 : tensor<1x32x256xf32>, tensor<256x128xf32>) outs(%1509 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %1511 = tensor.collapse_shape %1336 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %1512 = tensor.expand_shape %1511 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 32] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x32x8x32xf32>
    %1513 = tensor.empty() : tensor<1x8x32x32xf32>
    %1514 = linalg.transpose ins(%1512:tensor<1x32x8x32xf32>) outs(%1513:tensor<1x8x32x32xf32>) permutation = [0, 2, 1, 3]
    %1515 = tensor.collapse_shape %1423 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %1516 = tensor.expand_shape %1515 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 32] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<4096xf32> into tensor<1x32x4x32xf32>
    %1517 = tensor.empty() : tensor<1x4x32x32xf32>
    %1518 = linalg.transpose ins(%1516:tensor<1x32x4x32xf32>) outs(%1517:tensor<1x4x32x32xf32>) permutation = [0, 2, 1, 3]
    %1519 = tensor.collapse_shape %1510 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %1520 = tensor.expand_shape %1519 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 32] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<4096xf32> into tensor<1x32x4x32xf32>
    %1521 = tensor.empty() : tensor<1x4x32x32xf32>
    %1522 = linalg.transpose ins(%1520:tensor<1x32x4x32xf32>) outs(%1521:tensor<1x4x32x32xf32>) permutation = [0, 2, 1, 3]
    %1523 = "tensor.extract_slice"(%104) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_31", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.rotary_emb"} : (tensor<2048x32xf32>) -> tensor<32x32xf32>
    %1524 = "tensor.extract_slice"(%105) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_32", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.rotary_emb"} : (tensor<2048x32xf32>) -> tensor<32x32xf32>
    %1525 = tensor.empty() : tensor<1x32x32xf32>
    %1526 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%123 : tensor<1x32xi64>) outs(%1525 : tensor<1x32x32xf32>) attrs =  {prov.region_id = "gather_2", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb164(%1527: i64, %1528: f32):
      %1529 = arith.index_cast %1527 : i64 to index
      %1530 = linalg.index 2 : index
      %1531 = tensor.extract %1523[%1529, %1530] : tensor<32x32xf32>
      linalg.yield %1531 : f32
    } -> tensor<1x32x32xf32>
    %1532 = tensor.collapse_shape %1526 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %1533 = tensor.expand_shape %1532 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %1534 = tensor.empty() : tensor<1x32x32xf32>
    %1535 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%123 : tensor<1x32xi64>) outs(%1534 : tensor<1x32x32xf32>) attrs =  {prov.region_id = "gather_3", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb165(%1536: i64, %1537: f32):
      %1538 = arith.index_cast %1536 : i64 to index
      %1539 = linalg.index 2 : index
      %1540 = tensor.extract %1524[%1538, %1539] : tensor<32x32xf32>
      linalg.yield %1540 : f32
    } -> tensor<1x32x32xf32>
    %1541 = tensor.collapse_shape %1535 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %1542 = tensor.expand_shape %1541 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %1543 = tensor.empty() : tensor<1x8x32x32xf32>
    %1544 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1514, %1533 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%1543 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "mul_35", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb166(%1545: f32, %1546: f32, %1547: f32):
      %1548 = arith.mulf %1545, %1546 : f32
      linalg.yield %1548 : f32
    } -> tensor<1x8x32x32xf32>
    %1549 = "tensor.extract_slice"(%1514) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_33", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x8x32x32xf32>) -> tensor<1x8x32x16xf32>
    %1550 = "tensor.extract_slice"(%1514) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 8, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_34", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x8x32x32xf32>) -> tensor<1x8x32x16xf32>
    %1551 = tensor.empty() : tensor<1x8x32x16xf32>
    %1552 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1550 : tensor<1x8x32x16xf32>) outs(%1551 : tensor<1x8x32x16xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb167(%1553: f32, %1554: f32):
      %1555 = arith.negf %1553 : f32
      linalg.yield %1555 : f32
    } -> tensor<1x8x32x16xf32>
    %1556 = tensor.concat dim(3) %1552, %1549 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x8x32x16xf32>, tensor<1x8x32x16xf32>) -> tensor<1x8x32x32xf32>
    %1557 = tensor.empty() : tensor<1x8x32x32xf32>
    %1558 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1556, %1542 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%1557 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "mul_36", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb168(%1559: f32, %1560: f32, %1561: f32):
      %1562 = arith.mulf %1559, %1560 : f32
      linalg.yield %1562 : f32
    } -> tensor<1x8x32x32xf32>
    %1563 = tensor.empty() : tensor<1x8x32x32xf32>
    %1564 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1544, %1558 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%1563 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb169(%1565: f32, %1566: f32, %1567: f32):
      %1568 = arith.addf %1565, %1566 : f32
      linalg.yield %1568 : f32
    } -> tensor<1x8x32x32xf32>
    %1569 = tensor.empty() : tensor<1x4x32x32xf32>
    %1570 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1518, %1533 : tensor<1x4x32x32xf32>, tensor<1x1x32x32xf32>) outs(%1569 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb170(%1571: f32, %1572: f32, %1573: f32):
      %1574 = arith.mulf %1571, %1572 : f32
      linalg.yield %1574 : f32
    } -> tensor<1x4x32x32xf32>
    %1575 = "tensor.extract_slice"(%1518) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_35", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x16xf32>
    %1576 = "tensor.extract_slice"(%1518) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_36", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x16xf32>
    %1577 = tensor.empty() : tensor<1x4x32x16xf32>
    %1578 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1576 : tensor<1x4x32x16xf32>) outs(%1577 : tensor<1x4x32x16xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb171(%1579: f32, %1580: f32):
      %1581 = arith.negf %1579 : f32
      linalg.yield %1581 : f32
    } -> tensor<1x4x32x16xf32>
    %1582 = tensor.concat dim(3) %1578, %1575 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x16xf32>, tensor<1x4x32x16xf32>) -> tensor<1x4x32x32xf32>
    %1583 = tensor.empty() : tensor<1x4x32x32xf32>
    %1584 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1582, %1542 : tensor<1x4x32x32xf32>, tensor<1x1x32x32xf32>) outs(%1583 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "mul_38", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb172(%1585: f32, %1586: f32, %1587: f32):
      %1588 = arith.mulf %1585, %1586 : f32
      linalg.yield %1588 : f32
    } -> tensor<1x4x32x32xf32>
    %1589 = tensor.empty() : tensor<1x4x32x32xf32>
    %1590 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1570, %1584 : tensor<1x4x32x32xf32>, tensor<1x4x32x32xf32>) outs(%1589 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb173(%1591: f32, %1592: f32, %1593: f32):
      %1594 = arith.addf %1591, %1592 : f32
      linalg.yield %1594 : f32
    } -> tensor<1x4x32x32xf32>
    %1595 = "tensor.extract_slice"(%1590) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_37", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %1596 = "tensor.extract_slice"(%1595) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_38", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %1597 = tensor.collapse_shape %1596 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x32x32xf32> into tensor<4096xf32>
    %1598 = tensor.expand_shape %1597 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 32, 32] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<4096xf32> into tensor<1x4x1x32x32xf32>
    %1599 = "tensor.extract_slice"(%1598) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_39", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %1600 = "tensor.extract_slice"(%1599) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_40", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %1601 = tensor.empty() : tensor<1x4x2x32x32xf32>
    %1602 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1600 : tensor<1x4x1x32x32xf32>) outs(%1601 : tensor<1x4x2x32x32xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb174(%1603: f32, %1604: f32):
      linalg.yield %1603 : f32
    } -> tensor<1x4x2x32x32xf32>
    %1605 = tensor.collapse_shape %1602 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x2x32x32xf32> into tensor<8192xf32>
    %1606 = tensor.expand_shape %1605 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 32] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x8x32x32xf32>
    %1607 = "tensor.extract_slice"(%1522) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_41", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %1608 = "tensor.extract_slice"(%1607) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_42", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %1609 = tensor.collapse_shape %1608 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x32x32xf32> into tensor<4096xf32>
    %1610 = tensor.expand_shape %1609 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 32, 32] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<4096xf32> into tensor<1x4x1x32x32xf32>
    %1611 = "tensor.extract_slice"(%1610) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_43", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %1612 = "tensor.extract_slice"(%1611) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_44", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %1613 = tensor.empty() : tensor<1x4x2x32x32xf32>
    %1614 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1612 : tensor<1x4x1x32x32xf32>) outs(%1613 : tensor<1x4x2x32x32xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb175(%1615: f32, %1616: f32):
      linalg.yield %1615 : f32
    } -> tensor<1x4x2x32x32xf32>
    %1617 = tensor.collapse_shape %1614 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x2x32x32xf32> into tensor<8192xf32>
    %1618 = tensor.expand_shape %1617 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 32] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x8x32x32xf32>
    %1619 = tensor.empty() : tensor<1x8x32x32xf32>
    %1620 = linalg.transpose ins(%1606:tensor<1x8x32x32xf32>) outs(%1619:tensor<1x8x32x32xf32>) permutation = [0, 1, 3, 2]
    %1621 = arith.constant {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 0.000000e+00 : f32
    %1622 = tensor.splat %1621 {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32x32xf32>
    %1623 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1564, %1620 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%1622 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb176(%1624: f32, %1625: f32, %1626: f32):
      %1627 = arith.mulf %1624, %1625 : f32
      %1628 = arith.addf %1626, %1627 : f32
      linalg.yield %1628 : f32
    } -> tensor<1x8x32x32xf32>
    %1629 = arith.constant {prov.region_id = "div_11", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 5.65685415 : f32
    %1630 = tensor.splat %1629 {prov.region_id = "div_11", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32x32xf32>
    %1631 = tensor.empty() : tensor<1x8x32x32xf32>
    %1632 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1623, %1630 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%1631 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "div_11", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb177(%1633: f32, %1634: f32, %1635: f32):
      %1636 = arith.divf %1633, %1634 : f32
      linalg.yield %1636 : f32
    } -> tensor<1x8x32x32xf32>
    %1637 = "tensor.extract_slice"(%188) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_45", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    %1638 = "tensor.extract_slice"(%1637) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_46", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    %1639 = "tensor.extract_slice"(%1638) <{static_offsets = array<i64: 0, 0, 31, 0>, static_sizes = array<i64: 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<32xf32>
    %1640 = tensor.expand_shape %1639 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "slice_47", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<32xf32> into tensor<1x1x32xf32>
    %1641 = tensor.collapse_shape %1640 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x1x32xf32> into tensor<32xf32>
    %1642 = tensor.expand_shape %1641 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<32xf32> into tensor<1x1x1x32xf32>
    %1643 = tensor.empty() : tensor<1x1x32x32xf32>
    %1644 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1642 : tensor<1x1x1x32xf32>) outs(%1643 : tensor<1x1x32x32xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb178(%1645: f32, %1646: f32):
      linalg.yield %1645 : f32
    } -> tensor<1x1x32x32xf32>
    %1647 = tensor.empty() : tensor<1x8x32x32xf32>
    %1648 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1632, %1644 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%1647 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb179(%1649: f32, %1650: f32, %1651: f32):
      %1652 = arith.addf %1649, %1650 : f32
      linalg.yield %1652 : f32
    } -> tensor<1x8x32x32xf32>
    %1653 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 0xff800000 : f32
    %1654 = tensor.splat %1653 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32xf32>
    %1655 = linalg.reduce ins(%1648:tensor<1x8x32x32xf32>) outs(%1654:tensor<1x8x32xf32>) dimensions = [3]
    (%1656: f32, %1657: f32) {
      %1658 = arith.maximumf %1656, %1657 : f32
      linalg.yield %1658 : f32
    }
    %1659 = tensor.collapse_shape %1655 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %1660 = tensor.expand_shape %1659 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %1661 = tensor.empty() : tensor<1x8x32x32xf32>
    %1662 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1648, %1660 : tensor<1x8x32x32xf32>, tensor<1x8x32x1xf32>) outs(%1661 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb180(%1663: f32, %1664: f32, %1665: f32):
      %1666 = arith.subf %1663, %1664 : f32
      linalg.yield %1666 : f32
    } -> tensor<1x8x32x32xf32>
    %1667 = tensor.empty() : tensor<1x8x32x32xf32>
    %1668 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1662 : tensor<1x8x32x32xf32>) outs(%1667 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb181(%1669: f32, %1670: f32):
      %1671 = math.exp %1669 : f32
      linalg.yield %1671 : f32
    } -> tensor<1x8x32x32xf32>
    %1672 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 0.000000e+00 : f32
    %1673 = tensor.splat %1672 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32xf32>
    %1674 = linalg.reduce ins(%1668:tensor<1x8x32x32xf32>) outs(%1673:tensor<1x8x32xf32>) dimensions = [3]
    (%1675: f32, %1676: f32) {
      %1677 = arith.addf %1675, %1676 : f32
      linalg.yield %1677 : f32
    }
    %1678 = tensor.collapse_shape %1674 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %1679 = tensor.expand_shape %1678 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %1680 = tensor.empty() : tensor<1x8x32x32xf32>
    %1681 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1668, %1679 : tensor<1x8x32x32xf32>, tensor<1x8x32x1xf32>) outs(%1680 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb182(%1682: f32, %1683: f32, %1684: f32):
      %1685 = arith.divf %1682, %1683 : f32
      linalg.yield %1685 : f32
    } -> tensor<1x8x32x32xf32>
    %1686 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 0.000000e+00 : f32
    %1687 = tensor.splat %1686 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32x32xf32>
    %1688 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1681, %1618 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%1687 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb183(%1689: f32, %1690: f32, %1691: f32):
      %1692 = arith.mulf %1689, %1690 : f32
      %1693 = arith.addf %1691, %1692 : f32
      linalg.yield %1693 : f32
    } -> tensor<1x8x32x32xf32>
    %1694 = tensor.empty() : tensor<1x32x8x32xf32>
    %1695 = linalg.transpose ins(%1688:tensor<1x8x32x32xf32>) outs(%1694:tensor<1x32x8x32xf32>) permutation = [0, 2, 1, 3]
    %1696 = tensor.collapse_shape %1695 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x8x32xf32> into tensor<8192xf32>
    %1697 = tensor.expand_shape %1696 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 256] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x32x256xf32>
    %1698 = tensor.empty() : tensor<1x32x256xf32>
    %1699 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1697 : tensor<1x32x256xf32>) outs(%1698 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb184(%1700: f32, %1701: f32):
      %1702 = arith.constant 2.000000e+00 : f32
      %1703 = math.powf %1700, %1702 : f32
      linalg.yield %1703 : f32
    } -> tensor<1x32x256xf32>
    %1704 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} 0.000000e+00 : f32
    %1705 = tensor.splat %1704 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} : tensor<1x32xf32>
    %1706 = linalg.reduce ins(%1699:tensor<1x32x256xf32>) outs(%1705:tensor<1x32xf32>) dimensions = [2]
    (%1707: f32, %1708: f32) {
      %1709 = arith.addf %1707, %1708 : f32
      linalg.yield %1709 : f32
    }
    %1710 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} 2.560000e+02 : f32
    %1711 = tensor.splat %1710 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} : tensor<1x32xf32>
    %1712 = tensor.empty() : tensor<1x32xf32>
    %1713 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1706, %1711 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1712 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb185(%1714: f32, %1715: f32, %1716: f32):
      %1717 = arith.divf %1714, %1715 : f32
      linalg.yield %1717 : f32
    } -> tensor<1x32xf32>
    %1718 = tensor.collapse_shape %1713 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} : tensor<1x32xf32> into tensor<32xf32>
    %1719 = tensor.expand_shape %1718 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1720 = arith.constant {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} 1.000000e-05 : f32
    %1721 = tensor.splat %1720 {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} : tensor<1x32x1xf32>
    %1722 = tensor.empty() : tensor<1x32x1xf32>
    %1723 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1719, %1721 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1722 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb186(%1724: f32, %1725: f32, %1726: f32):
      %1727 = arith.addf %1724, %1725 : f32
      linalg.yield %1727 : f32
    } -> tensor<1x32x1xf32>
    %1728 = tensor.empty() : tensor<1x32x1xf32>
    %1729 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1723 : tensor<1x32x1xf32>) outs(%1728 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_5", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb187(%1730: f32, %1731: f32):
      %1732 = math.rsqrt %1730 : f32
      linalg.yield %1732 : f32
    } -> tensor<1x32x1xf32>
    %1733 = tensor.empty() : tensor<1x32x256xf32>
    %1734 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1697, %1729 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1733 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_39", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb188(%1735: f32, %1736: f32, %1737: f32):
      %1738 = arith.mulf %1735, %1736 : f32
      linalg.yield %1738 : f32
    } -> tensor<1x32x256xf32>
    %1739 = tensor.empty() : tensor<1x32x256xf32>
    %1740 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43, %1734 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%1739 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_40", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb189(%1741: f32, %1742: f32, %1743: f32):
      %1744 = arith.mulf %1741, %1742 : f32
      linalg.yield %1744 : f32
    } -> tensor<1x32x256xf32>
    %1745 = tensor.empty() : tensor<1x32x256xf32>
    %1746 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1740 : tensor<1x32x256xf32>) outs(%1745 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_10", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb190(%1747: f32, %1748: f32):
      %1749 = math.absf %1747 : f32
      linalg.yield %1749 : f32
    } -> tensor<1x32x256xf32>
    %1750 = arith.constant {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} 0xff800000 : f32
    %1751 = arith.constant {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} 0 : i64
    %1752 = tensor.splat %1750 {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<1x32xf32>
    %1753 = tensor.splat %1751 {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<1x32xi64>
    %1754, %1755 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1746 : tensor<1x32x256xf32>) outs(%1752, %1753 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb191(%1756: f32, %1757: f32, %1758: i64):
      %1759 = linalg.index 2 : index
      %1760 = arith.index_cast %1759 : index to i64
      %1761 = arith.cmpf ogt, %1756, %1757 : f32
      %1762 = arith.select %1761, %1756, %1757 : f32
      %1763 = arith.select %1761, %1760, %1758 : i64
      linalg.yield %1762, %1763 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1764 = tensor.collapse_shape %1754 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1765 = tensor.expand_shape %1764 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1766 = tensor.collapse_shape %1755 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1767 = tensor.expand_shape %1766 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1768 = func.call @aten_clamp__default(%1765) {prov.region_id = "aten_clamp__default_10", prov.dispatch_id = "aten_clamp__default_10"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1769 = tensor.empty() : tensor<1x32x1xf32>
    %1770 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1768 : tensor<1x32x1xf32>) outs(%1769 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_10", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb192(%1771: f32, %1772: f32):
      %1773 = arith.constant 1.000000e+00 : f32
      %1774 = arith.divf %1773, %1771 : f32
      linalg.yield %1774 : f32
    } -> tensor<1x32x1xf32>
    %1775 = arith.constant {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} 1.270000e+02 : f32
    %1776 = tensor.splat %1775 {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<1x32x1xf32>
    %1777 = tensor.empty() : tensor<1x32x1xf32>
    %1778 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1770, %1776 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1777 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb193(%1779: f32, %1780: f32, %1781: f32):
      %1782 = arith.mulf %1779, %1780 : f32
      linalg.yield %1782 : f32
    } -> tensor<1x32x1xf32>
    %1783 = tensor.empty() : tensor<1x32x256xf32>
    %1784 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1740, %1778 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1783 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_42", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb194(%1785: f32, %1786: f32, %1787: f32):
      %1788 = arith.mulf %1785, %1786 : f32
      linalg.yield %1788 : f32
    } -> tensor<1x32x256xf32>
    %1789 = tensor.empty() : tensor<1x32x256xf32>
    %1790 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1784 : tensor<1x32x256xf32>) outs(%1789 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_10", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb195(%1791: f32, %1792: f32):
      %1793 = math.roundeven %1791 : f32
      linalg.yield %1793 : f32
    } -> tensor<1x32x256xf32>
    %1794 = tensor.empty() : tensor<1x32x256xf32>
    %1795 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1790 : tensor<1x32x256xf32>) outs(%1794 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_11", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb196(%1796: f32, %1797: f32):
      %1798 = arith.constant -1.280000e+02 : f32
      %1799 = arith.maximumf %1796, %1798 : f32
      %1800 = arith.constant 1.270000e+02 : f32
      %1801 = arith.minimumf %1799, %1800 : f32
      linalg.yield %1801 : f32
    } -> tensor<1x32x256xf32>
    %1802 = tensor.empty() : tensor<1x32x256xf32>
    %1803 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1795, %1778 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1802 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_12", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb197(%1804: f32, %1805: f32, %1806: f32):
      %1807 = arith.divf %1804, %1805 : f32
      linalg.yield %1807 : f32
    } -> tensor<1x32x256xf32>
    %1808 = "quant_ext.unpack_int2"(%101) <{bits = 2 : i64, lanes = 4 : i64}> {prov.op = "unpack_int2", prov.family = "quantize"} : (tensor<16384xi8>) -> tensor<16384x4xi8>
    %1809 = tensor.collapse_shape %1808 [[0 : i64, 1 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<16384x4xi8> into tensor<65536xi8>
    %1810 = "tensor.extract_slice"(%1809) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 65536>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_48", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : (tensor<65536xi8>) -> tensor<65536xi8>
    %1811 = tensor.empty() : tensor<65536xf32>
    %1812 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1810 : tensor<65536xi8>) outs(%1811 : tensor<65536xf32>) attrs =  {prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb198(%1813: i8, %1814: f32):
      %1815 = arith.sitofp %1813 : i8 to f32
      linalg.yield %1815 : f32
    } -> tensor<65536xf32>
    %1816 = arith.constant {prov.region_id = "sub_10", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} 1.000000e+00 : f32
    %1817 = tensor.splat %1816 {prov.region_id = "sub_10", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<65536xf32>
    %1818 = tensor.empty() : tensor<65536xf32>
    %1819 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1812, %1817 : tensor<65536xf32>, tensor<65536xf32>) outs(%1818 : tensor<65536xf32>) attrs =  {prov.region_id = "sub_10", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb199(%1820: f32, %1821: f32, %1822: f32):
      %1823 = arith.subf %1820, %1821 : f32
      linalg.yield %1823 : f32
    } -> tensor<65536xf32>
    %1824 = func.call @aten_mul_Tensor(%1819) {prov.region_id = "aten_mul_Tensor_3", prov.dispatch_id = "aten_mul_Tensor_3"} : (tensor<65536xf32>) -> tensor<65536xf32>
    %1825 = tensor.expand_shape %1824 [[0 : i64, 1 : i64]] output_shape [256, 256] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<65536xf32> into tensor<256x256xf32>
    %1826 = tensor.empty() : tensor<256x256xf32>
    %1827 = linalg.transpose ins(%1825:tensor<256x256xf32>) outs(%1826:tensor<256x256xf32>) permutation = [1, 0]
    %1828 = tensor.empty() : tensor<1x32x256xf32>
    %1829 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1830 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1829 : f32) outs(%1828 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %1831 = linalg.matmul {prov.region_id = "matmul_14", prov.dispatch_id = "matmul_14", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} ins(%1803, %1827 : tensor<1x32x256xf32>, tensor<256x256xf32>) outs(%1830 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %1832 = tensor.empty() : tensor<1x32x256xf32>
    %1833 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1198, %1831 : tensor<1x32x256xf32>, tensor<1x32x256xf32>) outs(%1832 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1"} {
    ^bb200(%1834: f32, %1835: f32, %1836: f32):
      %1837 = arith.addf %1834, %1835 : f32
      linalg.yield %1837 : f32
    } -> tensor<1x32x256xf32>
    %1838 = tensor.empty() : tensor<1x32x256xf32>
    %1839 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1833 : tensor<1x32x256xf32>) outs(%1838 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb201(%1840: f32, %1841: f32):
      %1842 = arith.constant 2.000000e+00 : f32
      %1843 = math.powf %1840, %1842 : f32
      linalg.yield %1843 : f32
    } -> tensor<1x32x256xf32>
    %1844 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} 0.000000e+00 : f32
    %1845 = tensor.splat %1844 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<1x32xf32>
    %1846 = linalg.reduce ins(%1839:tensor<1x32x256xf32>) outs(%1845:tensor<1x32xf32>) dimensions = [2]
    (%1847: f32, %1848: f32) {
      %1849 = arith.addf %1847, %1848 : f32
      linalg.yield %1849 : f32
    }
    %1850 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} 2.560000e+02 : f32
    %1851 = tensor.splat %1850 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<1x32xf32>
    %1852 = tensor.empty() : tensor<1x32xf32>
    %1853 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1846, %1851 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1852 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb202(%1854: f32, %1855: f32, %1856: f32):
      %1857 = arith.divf %1854, %1855 : f32
      linalg.yield %1857 : f32
    } -> tensor<1x32xf32>
    %1858 = tensor.collapse_shape %1853 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %1859 = tensor.expand_shape %1858 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1860 = arith.constant {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} 1.000000e-05 : f32
    %1861 = tensor.splat %1860 {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<1x32x1xf32>
    %1862 = tensor.empty() : tensor<1x32x1xf32>
    %1863 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1859, %1861 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1862 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb203(%1864: f32, %1865: f32, %1866: f32):
      %1867 = arith.addf %1864, %1865 : f32
      linalg.yield %1867 : f32
    } -> tensor<1x32x1xf32>
    %1868 = tensor.empty() : tensor<1x32x1xf32>
    %1869 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1863 : tensor<1x32x1xf32>) outs(%1868 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_6", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb204(%1870: f32, %1871: f32):
      %1872 = math.rsqrt %1870 : f32
      linalg.yield %1872 : f32
    } -> tensor<1x32x1xf32>
    %1873 = tensor.empty() : tensor<1x32x256xf32>
    %1874 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1833, %1869 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1873 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb205(%1875: f32, %1876: f32, %1877: f32):
      %1878 = arith.mulf %1875, %1876 : f32
      linalg.yield %1878 : f32
    } -> tensor<1x32x256xf32>
    %1879 = tensor.empty() : tensor<1x32x256xf32>
    %1880 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46, %1874 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%1879 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_44", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb206(%1881: f32, %1882: f32, %1883: f32):
      %1884 = arith.mulf %1881, %1882 : f32
      linalg.yield %1884 : f32
    } -> tensor<1x32x256xf32>
    %1885 = tensor.empty() : tensor<1x32x256xf32>
    %1886 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1880 : tensor<1x32x256xf32>) outs(%1885 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_11", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb207(%1887: f32, %1888: f32):
      %1889 = math.absf %1887 : f32
      linalg.yield %1889 : f32
    } -> tensor<1x32x256xf32>
    %1890 = arith.constant {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} 0xff800000 : f32
    %1891 = arith.constant {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} 0 : i64
    %1892 = tensor.splat %1890 {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<1x32xf32>
    %1893 = tensor.splat %1891 {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<1x32xi64>
    %1894, %1895 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1886 : tensor<1x32x256xf32>) outs(%1892, %1893 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb208(%1896: f32, %1897: f32, %1898: i64):
      %1899 = linalg.index 2 : index
      %1900 = arith.index_cast %1899 : index to i64
      %1901 = arith.cmpf ogt, %1896, %1897 : f32
      %1902 = arith.select %1901, %1896, %1897 : f32
      %1903 = arith.select %1901, %1900, %1898 : i64
      linalg.yield %1902, %1903 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1904 = tensor.collapse_shape %1894 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1905 = tensor.expand_shape %1904 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1906 = tensor.collapse_shape %1895 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1907 = tensor.expand_shape %1906 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1908 = func.call @aten_clamp__default(%1905) {prov.region_id = "aten_clamp__default_11", prov.dispatch_id = "aten_clamp__default_11"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1909 = tensor.empty() : tensor<1x32x1xf32>
    %1910 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1908 : tensor<1x32x1xf32>) outs(%1909 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_11", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb209(%1911: f32, %1912: f32):
      %1913 = arith.constant 1.000000e+00 : f32
      %1914 = arith.divf %1913, %1911 : f32
      linalg.yield %1914 : f32
    } -> tensor<1x32x1xf32>
    %1915 = arith.constant {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} 1.270000e+02 : f32
    %1916 = tensor.splat %1915 {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<1x32x1xf32>
    %1917 = tensor.empty() : tensor<1x32x1xf32>
    %1918 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1910, %1916 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1917 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb210(%1919: f32, %1920: f32, %1921: f32):
      %1922 = arith.mulf %1919, %1920 : f32
      linalg.yield %1922 : f32
    } -> tensor<1x32x1xf32>
    %1923 = tensor.empty() : tensor<1x32x256xf32>
    %1924 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1880, %1918 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1923 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_46", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb211(%1925: f32, %1926: f32, %1927: f32):
      %1928 = arith.mulf %1925, %1926 : f32
      linalg.yield %1928 : f32
    } -> tensor<1x32x256xf32>
    %1929 = tensor.empty() : tensor<1x32x256xf32>
    %1930 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1924 : tensor<1x32x256xf32>) outs(%1929 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_11", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb212(%1931: f32, %1932: f32):
      %1933 = math.roundeven %1931 : f32
      linalg.yield %1933 : f32
    } -> tensor<1x32x256xf32>
    %1934 = tensor.empty() : tensor<1x32x256xf32>
    %1935 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1930 : tensor<1x32x256xf32>) outs(%1934 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_12", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb213(%1936: f32, %1937: f32):
      %1938 = arith.constant -1.280000e+02 : f32
      %1939 = arith.maximumf %1936, %1938 : f32
      %1940 = arith.constant 1.270000e+02 : f32
      %1941 = arith.minimumf %1939, %1940 : f32
      linalg.yield %1941 : f32
    } -> tensor<1x32x256xf32>
    %1942 = tensor.empty() : tensor<1x32x256xf32>
    %1943 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1935, %1918 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1942 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_13", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb214(%1944: f32, %1945: f32, %1946: f32):
      %1947 = arith.divf %1944, %1945 : f32
      linalg.yield %1947 : f32
    } -> tensor<1x32x256xf32>
    %1948 = "quant_ext.unpack_int2"(%106) <{bits = 2 : i64, lanes = 4 : i64}> {prov.op = "unpack_int2", prov.family = "quantize"} : (tensor<32768xi8>) -> tensor<32768x4xi8>
    %1949 = tensor.collapse_shape %1948 [[0 : i64, 1 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<32768x4xi8> into tensor<131072xi8>
    %1950 = "tensor.extract_slice"(%1949) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 131072>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_49", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : (tensor<131072xi8>) -> tensor<131072xi8>
    %1951 = tensor.empty() : tensor<131072xf32>
    %1952 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1950 : tensor<131072xi8>) outs(%1951 : tensor<131072xf32>) attrs =  {prov.region_id = "dtype_cast_14", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb215(%1953: i8, %1954: f32):
      %1955 = arith.sitofp %1953 : i8 to f32
      linalg.yield %1955 : f32
    } -> tensor<131072xf32>
    %1956 = arith.constant {prov.region_id = "sub_11", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} 1.000000e+00 : f32
    %1957 = tensor.splat %1956 {prov.region_id = "sub_11", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<131072xf32>
    %1958 = tensor.empty() : tensor<131072xf32>
    %1959 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1952, %1957 : tensor<131072xf32>, tensor<131072xf32>) outs(%1958 : tensor<131072xf32>) attrs =  {prov.region_id = "sub_11", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb216(%1960: f32, %1961: f32, %1962: f32):
      %1963 = arith.subf %1960, %1961 : f32
      linalg.yield %1963 : f32
    } -> tensor<131072xf32>
    %1964 = func.call @aten_mul_Tensor_2(%1959) {prov.region_id = "aten_mul_Tensor_2_3", prov.dispatch_id = "aten_mul_Tensor_2_3"} : (tensor<131072xf32>) -> tensor<131072xf32>
    %1965 = tensor.expand_shape %1964 [[0 : i64, 1 : i64]] output_shape [512, 256] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<131072xf32> into tensor<512x256xf32>
    %1966 = tensor.empty() : tensor<256x512xf32>
    %1967 = linalg.transpose ins(%1965:tensor<512x256xf32>) outs(%1966:tensor<256x512xf32>) permutation = [1, 0]
    %1968 = tensor.empty() : tensor<1x32x512xf32>
    %1969 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1970 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1969 : f32) outs(%1968 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %1971 = linalg.matmul {prov.region_id = "matmul_15", prov.dispatch_id = "matmul_15", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} ins(%1943, %1967 : tensor<1x32x256xf32>, tensor<256x512xf32>) outs(%1970 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %1972 = tensor.empty() : tensor<1x32x512xf32>
    %1973 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1971 : tensor<1x32x512xf32>) outs(%1972 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "minmax_13", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp"} {
    ^bb217(%1974: f32, %1975: f32):
      %1976 = arith.constant 0.000000e+00 : f32
      %1977 = arith.maximumf %1974, %1976 : f32
      linalg.yield %1977 : f32
    } -> tensor<1x32x512xf32>
    %1978 = tensor.empty() : tensor<1x32x512xf32>
    %1979 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1973 : tensor<1x32x512xf32>) outs(%1978 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp"} {
    ^bb218(%1980: f32, %1981: f32):
      %1982 = arith.constant 2.000000e+00 : f32
      %1983 = math.powf %1980, %1982 : f32
      linalg.yield %1983 : f32
    } -> tensor<1x32x512xf32>
    %1984 = tensor.empty() : tensor<1x32x256xf32>
    %1985 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1880 : tensor<1x32x256xf32>) outs(%1984 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_12", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb219(%1986: f32, %1987: f32):
      %1988 = math.absf %1986 : f32
      linalg.yield %1988 : f32
    } -> tensor<1x32x256xf32>
    %1989 = arith.constant {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} 0xff800000 : f32
    %1990 = arith.constant {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} 0 : i64
    %1991 = tensor.splat %1989 {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<1x32xf32>
    %1992 = tensor.splat %1990 {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<1x32xi64>
    %1993, %1994 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1985 : tensor<1x32x256xf32>) outs(%1991, %1992 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb220(%1995: f32, %1996: f32, %1997: i64):
      %1998 = linalg.index 2 : index
      %1999 = arith.index_cast %1998 : index to i64
      %2000 = arith.cmpf ogt, %1995, %1996 : f32
      %2001 = arith.select %2000, %1995, %1996 : f32
      %2002 = arith.select %2000, %1999, %1997 : i64
      linalg.yield %2001, %2002 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %2003 = tensor.collapse_shape %1993 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %2004 = tensor.expand_shape %2003 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2005 = tensor.collapse_shape %1994 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %2006 = tensor.expand_shape %2005 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %2007 = func.call @aten_clamp__default(%2004) {prov.region_id = "aten_clamp__default_12", prov.dispatch_id = "aten_clamp__default_12"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %2008 = tensor.empty() : tensor<1x32x1xf32>
    %2009 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2007 : tensor<1x32x1xf32>) outs(%2008 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_12", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb221(%2010: f32, %2011: f32):
      %2012 = arith.constant 1.000000e+00 : f32
      %2013 = arith.divf %2012, %2010 : f32
      linalg.yield %2013 : f32
    } -> tensor<1x32x1xf32>
    %2014 = arith.constant {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} 1.270000e+02 : f32
    %2015 = tensor.splat %2014 {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<1x32x1xf32>
    %2016 = tensor.empty() : tensor<1x32x1xf32>
    %2017 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2009, %2015 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2016 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb222(%2018: f32, %2019: f32, %2020: f32):
      %2021 = arith.mulf %2018, %2019 : f32
      linalg.yield %2021 : f32
    } -> tensor<1x32x1xf32>
    %2022 = tensor.empty() : tensor<1x32x256xf32>
    %2023 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1880, %2017 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2022 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_48", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb223(%2024: f32, %2025: f32, %2026: f32):
      %2027 = arith.mulf %2024, %2025 : f32
      linalg.yield %2027 : f32
    } -> tensor<1x32x256xf32>
    %2028 = tensor.empty() : tensor<1x32x256xf32>
    %2029 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2023 : tensor<1x32x256xf32>) outs(%2028 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_12", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb224(%2030: f32, %2031: f32):
      %2032 = math.roundeven %2030 : f32
      linalg.yield %2032 : f32
    } -> tensor<1x32x256xf32>
    %2033 = tensor.empty() : tensor<1x32x256xf32>
    %2034 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2029 : tensor<1x32x256xf32>) outs(%2033 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_14", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb225(%2035: f32, %2036: f32):
      %2037 = arith.constant -1.280000e+02 : f32
      %2038 = arith.maximumf %2035, %2037 : f32
      %2039 = arith.constant 1.270000e+02 : f32
      %2040 = arith.minimumf %2038, %2039 : f32
      linalg.yield %2040 : f32
    } -> tensor<1x32x256xf32>
    %2041 = tensor.empty() : tensor<1x32x256xf32>
    %2042 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2034, %2017 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2041 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_14", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb226(%2043: f32, %2044: f32, %2045: f32):
      %2046 = arith.divf %2043, %2044 : f32
      linalg.yield %2046 : f32
    } -> tensor<1x32x256xf32>
    %2047 = "quant_ext.unpack_int2"(%108) <{bits = 2 : i64, lanes = 4 : i64}> {prov.op = "unpack_int2", prov.family = "quantize"} : (tensor<32768xi8>) -> tensor<32768x4xi8>
    %2048 = tensor.collapse_shape %2047 [[0 : i64, 1 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<32768x4xi8> into tensor<131072xi8>
    %2049 = "tensor.extract_slice"(%2048) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 131072>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_50", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : (tensor<131072xi8>) -> tensor<131072xi8>
    %2050 = tensor.empty() : tensor<131072xf32>
    %2051 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2049 : tensor<131072xi8>) outs(%2050 : tensor<131072xf32>) attrs =  {prov.region_id = "dtype_cast_15", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb227(%2052: i8, %2053: f32):
      %2054 = arith.sitofp %2052 : i8 to f32
      linalg.yield %2054 : f32
    } -> tensor<131072xf32>
    %2055 = arith.constant {prov.region_id = "sub_12", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} 1.000000e+00 : f32
    %2056 = tensor.splat %2055 {prov.region_id = "sub_12", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<131072xf32>
    %2057 = tensor.empty() : tensor<131072xf32>
    %2058 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2051, %2056 : tensor<131072xf32>, tensor<131072xf32>) outs(%2057 : tensor<131072xf32>) attrs =  {prov.region_id = "sub_12", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb228(%2059: f32, %2060: f32, %2061: f32):
      %2062 = arith.subf %2059, %2060 : f32
      linalg.yield %2062 : f32
    } -> tensor<131072xf32>
    %2063 = func.call @aten_mul_Tensor_2(%2058) {prov.region_id = "aten_mul_Tensor_2_4", prov.dispatch_id = "aten_mul_Tensor_2_4"} : (tensor<131072xf32>) -> tensor<131072xf32>
    %2064 = tensor.expand_shape %2063 [[0 : i64, 1 : i64]] output_shape [512, 256] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<131072xf32> into tensor<512x256xf32>
    %2065 = tensor.empty() : tensor<256x512xf32>
    %2066 = linalg.transpose ins(%2064:tensor<512x256xf32>) outs(%2065:tensor<256x512xf32>) permutation = [1, 0]
    %2067 = tensor.empty() : tensor<1x32x512xf32>
    %2068 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %2069 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%2068 : f32) outs(%2067 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %2070 = linalg.matmul {prov.region_id = "matmul_16", prov.dispatch_id = "matmul_16", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} ins(%2042, %2066 : tensor<1x32x256xf32>, tensor<256x512xf32>) outs(%2069 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %2071 = tensor.empty() : tensor<1x32x512xf32>
    %2072 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1979, %2070 : tensor<1x32x512xf32>, tensor<1x32x512xf32>) outs(%2071 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_49", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp"} {
    ^bb229(%2073: f32, %2074: f32, %2075: f32):
      %2076 = arith.mulf %2073, %2074 : f32
      linalg.yield %2076 : f32
    } -> tensor<1x32x512xf32>
    %2077 = tensor.empty() : tensor<1x32x512xf32>
    %2078 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2072 : tensor<1x32x512xf32>) outs(%2077 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "pow_9", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb230(%2079: f32, %2080: f32):
      %2081 = arith.constant 2.000000e+00 : f32
      %2082 = math.powf %2079, %2081 : f32
      linalg.yield %2082 : f32
    } -> tensor<1x32x512xf32>
    %2083 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} 0.000000e+00 : f32
    %2084 = tensor.splat %2083 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} : tensor<1x32xf32>
    %2085 = linalg.reduce ins(%2078:tensor<1x32x512xf32>) outs(%2084:tensor<1x32xf32>) dimensions = [2]
    (%2086: f32, %2087: f32) {
      %2088 = arith.addf %2086, %2087 : f32
      linalg.yield %2088 : f32
    }
    %2089 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} 5.120000e+02 : f32
    %2090 = tensor.splat %2089 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} : tensor<1x32xf32>
    %2091 = tensor.empty() : tensor<1x32xf32>
    %2092 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2085, %2090 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%2091 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb231(%2093: f32, %2094: f32, %2095: f32):
      %2096 = arith.divf %2093, %2094 : f32
      linalg.yield %2096 : f32
    } -> tensor<1x32xf32>
    %2097 = tensor.collapse_shape %2092 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} : tensor<1x32xf32> into tensor<32xf32>
    %2098 = tensor.expand_shape %2097 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2099 = arith.constant {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} 1.000000e-05 : f32
    %2100 = tensor.splat %2099 {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} : tensor<1x32x1xf32>
    %2101 = tensor.empty() : tensor<1x32x1xf32>
    %2102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2098, %2100 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2101 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb232(%2103: f32, %2104: f32, %2105: f32):
      %2106 = arith.addf %2103, %2104 : f32
      linalg.yield %2106 : f32
    } -> tensor<1x32x1xf32>
    %2107 = tensor.empty() : tensor<1x32x1xf32>
    %2108 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2102 : tensor<1x32x1xf32>) outs(%2107 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_7", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb233(%2109: f32, %2110: f32):
      %2111 = math.rsqrt %2109 : f32
      linalg.yield %2111 : f32
    } -> tensor<1x32x1xf32>
    %2112 = tensor.empty() : tensor<1x32x512xf32>
    %2113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2072, %2108 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%2112 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_50", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb234(%2114: f32, %2115: f32, %2116: f32):
      %2117 = arith.mulf %2114, %2115 : f32
      linalg.yield %2117 : f32
    } -> tensor<1x32x512xf32>
    %2118 = tensor.empty() : tensor<1x32x512xf32>
    %2119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %2113 : tensor<512xf32>, tensor<1x32x512xf32>) outs(%2118 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_51", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb235(%2120: f32, %2121: f32, %2122: f32):
      %2123 = arith.mulf %2120, %2121 : f32
      linalg.yield %2123 : f32
    } -> tensor<1x32x512xf32>
    %2124 = tensor.empty() : tensor<1x32x512xf32>
    %2125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2119 : tensor<1x32x512xf32>) outs(%2124 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "abs_13", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb236(%2126: f32, %2127: f32):
      %2128 = math.absf %2126 : f32
      linalg.yield %2128 : f32
    } -> tensor<1x32x512xf32>
    %2129 = arith.constant {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} 0xff800000 : f32
    %2130 = arith.constant {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} 0 : i64
    %2131 = tensor.splat %2129 {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<1x32xf32>
    %2132 = tensor.splat %2130 {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<1x32xi64>
    %2133, %2134 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2125 : tensor<1x32x512xf32>) outs(%2131, %2132 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb237(%2135: f32, %2136: f32, %2137: i64):
      %2138 = linalg.index 2 : index
      %2139 = arith.index_cast %2138 : index to i64
      %2140 = arith.cmpf ogt, %2135, %2136 : f32
      %2141 = arith.select %2140, %2135, %2136 : f32
      %2142 = arith.select %2140, %2139, %2137 : i64
      linalg.yield %2141, %2142 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %2143 = tensor.collapse_shape %2133 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %2144 = tensor.expand_shape %2143 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2145 = tensor.collapse_shape %2134 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %2146 = tensor.expand_shape %2145 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %2147 = func.call @aten_clamp__default(%2144) {prov.region_id = "aten_clamp__default_13", prov.dispatch_id = "aten_clamp__default_13"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %2148 = tensor.empty() : tensor<1x32x1xf32>
    %2149 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2147 : tensor<1x32x1xf32>) outs(%2148 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_13", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb238(%2150: f32, %2151: f32):
      %2152 = arith.constant 1.000000e+00 : f32
      %2153 = arith.divf %2152, %2150 : f32
      linalg.yield %2153 : f32
    } -> tensor<1x32x1xf32>
    %2154 = arith.constant {prov.region_id = "mul_52", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} 1.270000e+02 : f32
    %2155 = tensor.splat %2154 {prov.region_id = "mul_52", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<1x32x1xf32>
    %2156 = tensor.empty() : tensor<1x32x1xf32>
    %2157 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2149, %2155 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2156 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_52", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb239(%2158: f32, %2159: f32, %2160: f32):
      %2161 = arith.mulf %2158, %2159 : f32
      linalg.yield %2161 : f32
    } -> tensor<1x32x1xf32>
    %2162 = tensor.empty() : tensor<1x32x512xf32>
    %2163 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2119, %2157 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%2162 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_53", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb240(%2164: f32, %2165: f32, %2166: f32):
      %2167 = arith.mulf %2164, %2165 : f32
      linalg.yield %2167 : f32
    } -> tensor<1x32x512xf32>
    %2168 = tensor.empty() : tensor<1x32x512xf32>
    %2169 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2163 : tensor<1x32x512xf32>) outs(%2168 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "round_13", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb241(%2170: f32, %2171: f32):
      %2172 = math.roundeven %2170 : f32
      linalg.yield %2172 : f32
    } -> tensor<1x32x512xf32>
    %2173 = tensor.empty() : tensor<1x32x512xf32>
    %2174 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2169 : tensor<1x32x512xf32>) outs(%2173 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "minmax_15", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb242(%2175: f32, %2176: f32):
      %2177 = arith.constant -1.280000e+02 : f32
      %2178 = arith.maximumf %2175, %2177 : f32
      %2179 = arith.constant 1.270000e+02 : f32
      %2180 = arith.minimumf %2178, %2179 : f32
      linalg.yield %2180 : f32
    } -> tensor<1x32x512xf32>
    %2181 = tensor.empty() : tensor<1x32x512xf32>
    %2182 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2174, %2157 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%2181 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "div_15", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb243(%2183: f32, %2184: f32, %2185: f32):
      %2186 = arith.divf %2183, %2184 : f32
      linalg.yield %2186 : f32
    } -> tensor<1x32x512xf32>
    %2187 = "quant_ext.unpack_int2"(%110) <{bits = 2 : i64, lanes = 4 : i64}> {prov.op = "unpack_int2", prov.family = "quantize"} : (tensor<32768xi8>) -> tensor<32768x4xi8>
    %2188 = tensor.collapse_shape %2187 [[0 : i64, 1 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<32768x4xi8> into tensor<131072xi8>
    %2189 = "tensor.extract_slice"(%2188) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 131072>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_51", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : (tensor<131072xi8>) -> tensor<131072xi8>
    %2190 = tensor.empty() : tensor<131072xf32>
    %2191 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2189 : tensor<131072xi8>) outs(%2190 : tensor<131072xf32>) attrs =  {prov.region_id = "dtype_cast_16", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb244(%2192: i8, %2193: f32):
      %2194 = arith.sitofp %2192 : i8 to f32
      linalg.yield %2194 : f32
    } -> tensor<131072xf32>
    %2195 = arith.constant {prov.region_id = "sub_13", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} 1.000000e+00 : f32
    %2196 = tensor.splat %2195 {prov.region_id = "sub_13", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<131072xf32>
    %2197 = tensor.empty() : tensor<131072xf32>
    %2198 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2191, %2196 : tensor<131072xf32>, tensor<131072xf32>) outs(%2197 : tensor<131072xf32>) attrs =  {prov.region_id = "sub_13", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb245(%2199: f32, %2200: f32, %2201: f32):
      %2202 = arith.subf %2199, %2200 : f32
      linalg.yield %2202 : f32
    } -> tensor<131072xf32>
    %2203 = func.call @aten_mul_Tensor_2(%2198) {prov.region_id = "aten_mul_Tensor_2_5", prov.dispatch_id = "aten_mul_Tensor_2_5"} : (tensor<131072xf32>) -> tensor<131072xf32>
    %2204 = tensor.expand_shape %2203 [[0 : i64, 1 : i64]] output_shape [256, 512] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<131072xf32> into tensor<256x512xf32>
    %2205 = tensor.empty() : tensor<512x256xf32>
    %2206 = linalg.transpose ins(%2204:tensor<256x512xf32>) outs(%2205:tensor<512x256xf32>) permutation = [1, 0]
    %2207 = tensor.empty() : tensor<1x32x256xf32>
    %2208 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %2209 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%2208 : f32) outs(%2207 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %2210 = linalg.matmul {prov.region_id = "matmul_17", prov.dispatch_id = "matmul_17", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} ins(%2182, %2206 : tensor<1x32x512xf32>, tensor<512x256xf32>) outs(%2209 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %2211 = tensor.empty() : tensor<1x32x256xf32>
    %2212 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1833, %2210 : tensor<1x32x256xf32>, tensor<1x32x256xf32>) outs(%2211 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1"} {
    ^bb246(%2213: f32, %2214: f32, %2215: f32):
      %2216 = arith.addf %2213, %2214 : f32
      linalg.yield %2216 : f32
    } -> tensor<1x32x256xf32>
    %2217 = tensor.empty() : tensor<1x32x256xf32>
    %2218 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2212 : tensor<1x32x256xf32>) outs(%2217 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_10", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb247(%2219: f32, %2220: f32):
      %2221 = arith.constant 2.000000e+00 : f32
      %2222 = math.powf %2219, %2221 : f32
      linalg.yield %2222 : f32
    } -> tensor<1x32x256xf32>
    %2223 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} 0.000000e+00 : f32
    %2224 = tensor.splat %2223 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<1x32xf32>
    %2225 = linalg.reduce ins(%2218:tensor<1x32x256xf32>) outs(%2224:tensor<1x32xf32>) dimensions = [2]
    (%2226: f32, %2227: f32) {
      %2228 = arith.addf %2226, %2227 : f32
      linalg.yield %2228 : f32
    }
    %2229 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} 2.560000e+02 : f32
    %2230 = tensor.splat %2229 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<1x32xf32>
    %2231 = tensor.empty() : tensor<1x32xf32>
    %2232 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2225, %2230 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%2231 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb248(%2233: f32, %2234: f32, %2235: f32):
      %2236 = arith.divf %2233, %2234 : f32
      linalg.yield %2236 : f32
    } -> tensor<1x32xf32>
    %2237 = tensor.collapse_shape %2232 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<1x32xf32> into tensor<32xf32>
    %2238 = tensor.expand_shape %2237 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2239 = arith.constant {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} 1.000000e-05 : f32
    %2240 = tensor.splat %2239 {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<1x32x1xf32>
    %2241 = tensor.empty() : tensor<1x32x1xf32>
    %2242 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2238, %2240 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2241 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb249(%2243: f32, %2244: f32, %2245: f32):
      %2246 = arith.addf %2243, %2244 : f32
      linalg.yield %2246 : f32
    } -> tensor<1x32x1xf32>
    %2247 = tensor.empty() : tensor<1x32x1xf32>
    %2248 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2242 : tensor<1x32x1xf32>) outs(%2247 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_8", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb250(%2249: f32, %2250: f32):
      %2251 = math.rsqrt %2249 : f32
      linalg.yield %2251 : f32
    } -> tensor<1x32x1xf32>
    %2252 = tensor.empty() : tensor<1x32x256xf32>
    %2253 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2212, %2248 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2252 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_54", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb251(%2254: f32, %2255: f32, %2256: f32):
      %2257 = arith.mulf %2254, %2255 : f32
      linalg.yield %2257 : f32
    } -> tensor<1x32x256xf32>
    %2258 = tensor.empty() : tensor<1x32x256xf32>
    %2259 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %2253 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%2258 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_55", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb252(%2260: f32, %2261: f32, %2262: f32):
      %2263 = arith.mulf %2260, %2261 : f32
      linalg.yield %2263 : f32
    } -> tensor<1x32x256xf32>
    %2264 = tensor.empty() : tensor<256x1024xf32>
    %2265 = linalg.transpose ins(%48:tensor<1024x256xf32>) outs(%2264:tensor<256x1024xf32>) permutation = [1, 0]
    %2266 = tensor.empty() : tensor<1x32x1024xf32>
    %2267 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %2268 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%2267 : f32) outs(%2266 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
    %2269 = linalg.matmul {prov.region_id = "matmul_18", prov.dispatch_id = "matmul_18", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.lm_head"} ins(%2259, %2265 : tensor<1x32x256xf32>, tensor<256x1024xf32>) outs(%2268 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
    func.return %2269 : tensor<1x32x1024xf32>
  }
}
