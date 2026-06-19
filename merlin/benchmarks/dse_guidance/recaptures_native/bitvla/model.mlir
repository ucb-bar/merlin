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
    %299 = func.call @aten___and___Scalar(%78) {prov.region_id = "aten___and___Scalar_0", prov.dispatch_id = "aten___and___Scalar_0"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %300 = func.call @aten___rshift___Scalar(%78) {prov.region_id = "aten___rshift___Scalar_0", prov.dispatch_id = "aten___rshift___Scalar_0"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %301 = func.call @aten___and___Scalar(%300) {prov.region_id = "aten___and___Scalar_1", prov.dispatch_id = "aten___and___Scalar_1"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %302 = func.call @aten___rshift___Scalar(%78) {prov.region_id = "aten___rshift___Scalar_1", prov.dispatch_id = "aten___rshift___Scalar_1"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %303 = func.call @aten___and___Scalar(%302) {prov.region_id = "aten___and___Scalar_2", prov.dispatch_id = "aten___and___Scalar_2"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %304 = func.call @aten___rshift___Scalar(%78) {prov.region_id = "aten___rshift___Scalar_2", prov.dispatch_id = "aten___rshift___Scalar_2"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %305 = func.call @aten___and___Scalar(%304) {prov.region_id = "aten___and___Scalar_3", prov.dispatch_id = "aten___and___Scalar_3"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %306 = func.call @aten_stack_default(%299, %301, %303, %305) {prov.region_id = "aten_stack_default_0", prov.dispatch_id = "aten_stack_default_0"} : (tensor<16384xi8>, tensor<16384xi8>, tensor<16384xi8>, tensor<16384xi8>) -> tensor<16384x4xi8>
    %307 = tensor.collapse_shape %306 [[0 : i64, 1 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<16384x4xi8> into tensor<65536xi8>
    %308 = "tensor.extract_slice"(%307) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 65536>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : (tensor<65536xi8>) -> tensor<65536xi8>
    %309 = tensor.empty() : tensor<65536xf32>
    %310 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%308 : tensor<65536xi8>) outs(%309 : tensor<65536xf32>) attrs =  {prov.region_id = "dtype_cast_3", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb23(%311: i8, %312: f32):
      %313 = arith.sitofp %311 : i8 to f32
      linalg.yield %313 : f32
    } -> tensor<65536xf32>
    %314 = arith.constant {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} 1.000000e+00 : f32
    %315 = tensor.splat %314 {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<65536xf32>
    %316 = tensor.empty() : tensor<65536xf32>
    %317 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%310, %315 : tensor<65536xf32>, tensor<65536xf32>) outs(%316 : tensor<65536xf32>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb24(%318: f32, %319: f32, %320: f32):
      %321 = arith.subf %318, %319 : f32
      linalg.yield %321 : f32
    } -> tensor<65536xf32>
    %322 = func.call @aten_mul_Tensor(%317) {prov.region_id = "aten_mul_Tensor_0", prov.dispatch_id = "aten_mul_Tensor_0"} : (tensor<65536xf32>) -> tensor<65536xf32>
    %323 = tensor.expand_shape %322 [[0 : i64, 1 : i64]] output_shape [256, 256] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<65536xf32> into tensor<256x256xf32>
    %324 = tensor.empty() : tensor<256x256xf32>
    %325 = linalg.transpose ins(%323:tensor<256x256xf32>) outs(%324:tensor<256x256xf32>) permutation = [1, 0]
    %326 = tensor.empty() : tensor<1x32x256xf32>
    %327 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %328 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%327 : f32) outs(%326 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %329 = linalg.matmul {prov.region_id = "matmul_0", prov.dispatch_id = "matmul_0", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} ins(%294, %325 : tensor<1x32x256xf32>, tensor<256x256xf32>) outs(%328 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %330 = tensor.empty() : tensor<1x32x256xf32>
    %331 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%231 : tensor<1x32x256xf32>) outs(%330 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_1", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb25(%332: f32, %333: f32):
      %334 = math.absf %332 : f32
      linalg.yield %334 : f32
    } -> tensor<1x32x256xf32>
    %335 = arith.constant {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} 0xff800000 : f32
    %336 = arith.constant {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} 0 : i64
    %337 = tensor.splat %335 {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<1x32xf32>
    %338 = tensor.splat %336 {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<1x32xi64>
    %339, %340 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%331 : tensor<1x32x256xf32>) outs(%337, %338 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb26(%341: f32, %342: f32, %343: i64):
      %344 = linalg.index 2 : index
      %345 = arith.index_cast %344 : index to i64
      %346 = arith.cmpf ogt, %341, %342 : f32
      %347 = arith.select %346, %341, %342 : f32
      %348 = arith.select %346, %345, %343 : i64
      linalg.yield %347, %348 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %349 = tensor.collapse_shape %339 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %350 = tensor.expand_shape %349 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %351 = tensor.collapse_shape %340 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %352 = tensor.expand_shape %351 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %353 = func.call @aten_clamp__default(%350) {prov.region_id = "aten_clamp__default_1", prov.dispatch_id = "aten_clamp__default_1"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %354 = tensor.empty() : tensor<1x32x1xf32>
    %355 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%353 : tensor<1x32x1xf32>) outs(%354 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_1", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb27(%356: f32, %357: f32):
      %358 = arith.constant 1.000000e+00 : f32
      %359 = arith.divf %358, %356 : f32
      linalg.yield %359 : f32
    } -> tensor<1x32x1xf32>
    %360 = arith.constant {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} 1.270000e+02 : f32
    %361 = tensor.splat %360 {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<1x32x1xf32>
    %362 = tensor.empty() : tensor<1x32x1xf32>
    %363 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%355, %361 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%362 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb28(%364: f32, %365: f32, %366: f32):
      %367 = arith.mulf %364, %365 : f32
      linalg.yield %367 : f32
    } -> tensor<1x32x1xf32>
    %368 = tensor.empty() : tensor<1x32x256xf32>
    %369 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%231, %363 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%368 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb29(%370: f32, %371: f32, %372: f32):
      %373 = arith.mulf %370, %371 : f32
      linalg.yield %373 : f32
    } -> tensor<1x32x256xf32>
    %374 = tensor.empty() : tensor<1x32x256xf32>
    %375 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%369 : tensor<1x32x256xf32>) outs(%374 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_1", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb30(%376: f32, %377: f32):
      %378 = math.roundeven %376 : f32
      linalg.yield %378 : f32
    } -> tensor<1x32x256xf32>
    %379 = tensor.empty() : tensor<1x32x256xf32>
    %380 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%375 : tensor<1x32x256xf32>) outs(%379 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_1", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb31(%381: f32, %382: f32):
      %383 = arith.constant -1.280000e+02 : f32
      %384 = arith.maximumf %381, %383 : f32
      %385 = arith.constant 1.270000e+02 : f32
      %386 = arith.minimumf %384, %385 : f32
      linalg.yield %386 : f32
    } -> tensor<1x32x256xf32>
    %387 = tensor.empty() : tensor<1x32x256xf32>
    %388 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%380, %363 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%387 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb32(%389: f32, %390: f32, %391: f32):
      %392 = arith.divf %389, %390 : f32
      linalg.yield %392 : f32
    } -> tensor<1x32x256xf32>
    %393 = func.call @aten___and___Scalar_1(%80) {prov.region_id = "aten___and___Scalar_1_0", prov.dispatch_id = "aten___and___Scalar_1_0"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %394 = func.call @aten___rshift___Scalar_1(%80) {prov.region_id = "aten___rshift___Scalar_1_0", prov.dispatch_id = "aten___rshift___Scalar_1_0"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %395 = func.call @aten___and___Scalar_1(%394) {prov.region_id = "aten___and___Scalar_1_1", prov.dispatch_id = "aten___and___Scalar_1_1"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %396 = func.call @aten___rshift___Scalar_1(%80) {prov.region_id = "aten___rshift___Scalar_1_1", prov.dispatch_id = "aten___rshift___Scalar_1_1"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %397 = func.call @aten___and___Scalar_1(%396) {prov.region_id = "aten___and___Scalar_1_2", prov.dispatch_id = "aten___and___Scalar_1_2"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %398 = func.call @aten___rshift___Scalar_1(%80) {prov.region_id = "aten___rshift___Scalar_1_2", prov.dispatch_id = "aten___rshift___Scalar_1_2"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %399 = func.call @aten___and___Scalar_1(%398) {prov.region_id = "aten___and___Scalar_1_3", prov.dispatch_id = "aten___and___Scalar_1_3"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %400 = func.call @aten_stack_default_1(%393, %395, %397, %399) {prov.region_id = "aten_stack_default_1_0", prov.dispatch_id = "aten_stack_default_1_0"} : (tensor<8192xi8>, tensor<8192xi8>, tensor<8192xi8>, tensor<8192xi8>) -> tensor<8192x4xi8>
    %401 = tensor.collapse_shape %400 [[0 : i64, 1 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<8192x4xi8> into tensor<32768xi8>
    %402 = "tensor.extract_slice"(%401) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 32768>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %403 = tensor.empty() : tensor<32768xf32>
    %404 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%402 : tensor<32768xi8>) outs(%403 : tensor<32768xf32>) attrs =  {prov.region_id = "dtype_cast_4", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb33(%405: i8, %406: f32):
      %407 = arith.sitofp %405 : i8 to f32
      linalg.yield %407 : f32
    } -> tensor<32768xf32>
    %408 = arith.constant {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} 1.000000e+00 : f32
    %409 = tensor.splat %408 {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<32768xf32>
    %410 = tensor.empty() : tensor<32768xf32>
    %411 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%404, %409 : tensor<32768xf32>, tensor<32768xf32>) outs(%410 : tensor<32768xf32>) attrs =  {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb34(%412: f32, %413: f32, %414: f32):
      %415 = arith.subf %412, %413 : f32
      linalg.yield %415 : f32
    } -> tensor<32768xf32>
    %416 = func.call @aten_mul_Tensor_1(%411) {prov.region_id = "aten_mul_Tensor_1_0", prov.dispatch_id = "aten_mul_Tensor_1_0"} : (tensor<32768xf32>) -> tensor<32768xf32>
    %417 = tensor.expand_shape %416 [[0 : i64, 1 : i64]] output_shape [128, 256] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<32768xf32> into tensor<128x256xf32>
    %418 = tensor.empty() : tensor<256x128xf32>
    %419 = linalg.transpose ins(%417:tensor<128x256xf32>) outs(%418:tensor<256x128xf32>) permutation = [1, 0]
    %420 = tensor.empty() : tensor<1x32x128xf32>
    %421 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %422 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%421 : f32) outs(%420 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %423 = linalg.matmul {prov.region_id = "matmul_1", prov.dispatch_id = "matmul_1", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} ins(%388, %419 : tensor<1x32x256xf32>, tensor<256x128xf32>) outs(%422 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %424 = tensor.empty() : tensor<1x32x256xf32>
    %425 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%231 : tensor<1x32x256xf32>) outs(%424 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_2", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb35(%426: f32, %427: f32):
      %428 = math.absf %426 : f32
      linalg.yield %428 : f32
    } -> tensor<1x32x256xf32>
    %429 = arith.constant {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} 0xff800000 : f32
    %430 = arith.constant {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} 0 : i64
    %431 = tensor.splat %429 {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<1x32xf32>
    %432 = tensor.splat %430 {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<1x32xi64>
    %433, %434 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%425 : tensor<1x32x256xf32>) outs(%431, %432 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb36(%435: f32, %436: f32, %437: i64):
      %438 = linalg.index 2 : index
      %439 = arith.index_cast %438 : index to i64
      %440 = arith.cmpf ogt, %435, %436 : f32
      %441 = arith.select %440, %435, %436 : f32
      %442 = arith.select %440, %439, %437 : i64
      linalg.yield %441, %442 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %443 = tensor.collapse_shape %433 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %444 = tensor.expand_shape %443 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %445 = tensor.collapse_shape %434 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %446 = tensor.expand_shape %445 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %447 = func.call @aten_clamp__default(%444) {prov.region_id = "aten_clamp__default_2", prov.dispatch_id = "aten_clamp__default_2"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %448 = tensor.empty() : tensor<1x32x1xf32>
    %449 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%447 : tensor<1x32x1xf32>) outs(%448 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_2", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb37(%450: f32, %451: f32):
      %452 = arith.constant 1.000000e+00 : f32
      %453 = arith.divf %452, %450 : f32
      linalg.yield %453 : f32
    } -> tensor<1x32x1xf32>
    %454 = arith.constant {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} 1.270000e+02 : f32
    %455 = tensor.splat %454 {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<1x32x1xf32>
    %456 = tensor.empty() : tensor<1x32x1xf32>
    %457 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%449, %455 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%456 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb38(%458: f32, %459: f32, %460: f32):
      %461 = arith.mulf %458, %459 : f32
      linalg.yield %461 : f32
    } -> tensor<1x32x1xf32>
    %462 = tensor.empty() : tensor<1x32x256xf32>
    %463 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%231, %457 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%462 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb39(%464: f32, %465: f32, %466: f32):
      %467 = arith.mulf %464, %465 : f32
      linalg.yield %467 : f32
    } -> tensor<1x32x256xf32>
    %468 = tensor.empty() : tensor<1x32x256xf32>
    %469 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%463 : tensor<1x32x256xf32>) outs(%468 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_2", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb40(%470: f32, %471: f32):
      %472 = math.roundeven %470 : f32
      linalg.yield %472 : f32
    } -> tensor<1x32x256xf32>
    %473 = tensor.empty() : tensor<1x32x256xf32>
    %474 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%469 : tensor<1x32x256xf32>) outs(%473 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_2", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb41(%475: f32, %476: f32):
      %477 = arith.constant -1.280000e+02 : f32
      %478 = arith.maximumf %475, %477 : f32
      %479 = arith.constant 1.270000e+02 : f32
      %480 = arith.minimumf %478, %479 : f32
      linalg.yield %480 : f32
    } -> tensor<1x32x256xf32>
    %481 = tensor.empty() : tensor<1x32x256xf32>
    %482 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%474, %457 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%481 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb42(%483: f32, %484: f32, %485: f32):
      %486 = arith.divf %483, %484 : f32
      linalg.yield %486 : f32
    } -> tensor<1x32x256xf32>
    %487 = func.call @aten___and___Scalar_1(%82) {prov.region_id = "aten___and___Scalar_1_4", prov.dispatch_id = "aten___and___Scalar_1_4"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %488 = func.call @aten___rshift___Scalar_1(%82) {prov.region_id = "aten___rshift___Scalar_1_3", prov.dispatch_id = "aten___rshift___Scalar_1_3"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %489 = func.call @aten___and___Scalar_1(%488) {prov.region_id = "aten___and___Scalar_1_5", prov.dispatch_id = "aten___and___Scalar_1_5"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %490 = func.call @aten___rshift___Scalar_1(%82) {prov.region_id = "aten___rshift___Scalar_1_4", prov.dispatch_id = "aten___rshift___Scalar_1_4"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %491 = func.call @aten___and___Scalar_1(%490) {prov.region_id = "aten___and___Scalar_1_6", prov.dispatch_id = "aten___and___Scalar_1_6"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %492 = func.call @aten___rshift___Scalar_1(%82) {prov.region_id = "aten___rshift___Scalar_1_5", prov.dispatch_id = "aten___rshift___Scalar_1_5"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %493 = func.call @aten___and___Scalar_1(%492) {prov.region_id = "aten___and___Scalar_1_7", prov.dispatch_id = "aten___and___Scalar_1_7"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %494 = func.call @aten_stack_default_1(%487, %489, %491, %493) {prov.region_id = "aten_stack_default_1_1", prov.dispatch_id = "aten_stack_default_1_1"} : (tensor<8192xi8>, tensor<8192xi8>, tensor<8192xi8>, tensor<8192xi8>) -> tensor<8192x4xi8>
    %495 = tensor.collapse_shape %494 [[0 : i64, 1 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<8192x4xi8> into tensor<32768xi8>
    %496 = "tensor.extract_slice"(%495) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 32768>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %497 = tensor.empty() : tensor<32768xf32>
    %498 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%496 : tensor<32768xi8>) outs(%497 : tensor<32768xf32>) attrs =  {prov.region_id = "dtype_cast_5", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb43(%499: i8, %500: f32):
      %501 = arith.sitofp %499 : i8 to f32
      linalg.yield %501 : f32
    } -> tensor<32768xf32>
    %502 = arith.constant {prov.region_id = "sub_2", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} 1.000000e+00 : f32
    %503 = tensor.splat %502 {prov.region_id = "sub_2", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<32768xf32>
    %504 = tensor.empty() : tensor<32768xf32>
    %505 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%498, %503 : tensor<32768xf32>, tensor<32768xf32>) outs(%504 : tensor<32768xf32>) attrs =  {prov.region_id = "sub_2", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb44(%506: f32, %507: f32, %508: f32):
      %509 = arith.subf %506, %507 : f32
      linalg.yield %509 : f32
    } -> tensor<32768xf32>
    %510 = func.call @aten_mul_Tensor_1(%505) {prov.region_id = "aten_mul_Tensor_1_1", prov.dispatch_id = "aten_mul_Tensor_1_1"} : (tensor<32768xf32>) -> tensor<32768xf32>
    %511 = tensor.expand_shape %510 [[0 : i64, 1 : i64]] output_shape [128, 256] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<32768xf32> into tensor<128x256xf32>
    %512 = tensor.empty() : tensor<256x128xf32>
    %513 = linalg.transpose ins(%511:tensor<128x256xf32>) outs(%512:tensor<256x128xf32>) permutation = [1, 0]
    %514 = tensor.empty() : tensor<1x32x128xf32>
    %515 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %516 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%515 : f32) outs(%514 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %517 = linalg.matmul {prov.region_id = "matmul_2", prov.dispatch_id = "matmul_2", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} ins(%482, %513 : tensor<1x32x256xf32>, tensor<256x128xf32>) outs(%516 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %518 = tensor.collapse_shape %329 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %519 = tensor.expand_shape %518 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 32] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x32x8x32xf32>
    %520 = tensor.empty() : tensor<1x8x32x32xf32>
    %521 = linalg.transpose ins(%519:tensor<1x32x8x32xf32>) outs(%520:tensor<1x8x32x32xf32>) permutation = [0, 2, 1, 3]
    %522 = tensor.collapse_shape %423 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %523 = tensor.expand_shape %522 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 32] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<4096xf32> into tensor<1x32x4x32xf32>
    %524 = tensor.empty() : tensor<1x4x32x32xf32>
    %525 = linalg.transpose ins(%523:tensor<1x32x4x32xf32>) outs(%524:tensor<1x4x32x32xf32>) permutation = [0, 2, 1, 3]
    %526 = tensor.collapse_shape %517 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %527 = tensor.expand_shape %526 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 32] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<4096xf32> into tensor<1x32x4x32xf32>
    %528 = tensor.empty() : tensor<1x4x32x32xf32>
    %529 = linalg.transpose ins(%527:tensor<1x32x4x32xf32>) outs(%528:tensor<1x4x32x32xf32>) permutation = [0, 2, 1, 3]
    %530 = "tensor.extract_slice"(%87) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.rotary_emb"} : (tensor<2048x32xf32>) -> tensor<32x32xf32>
    %531 = "tensor.extract_slice"(%88) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.rotary_emb"} : (tensor<2048x32xf32>) -> tensor<32x32xf32>
    %532 = tensor.empty() : tensor<1x32x32xf32>
    %533 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%123 : tensor<1x32xi64>) outs(%532 : tensor<1x32x32xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb45(%534: i64, %535: f32):
      %536 = arith.index_cast %534 : i64 to index
      %537 = linalg.index 2 : index
      %538 = tensor.extract %530[%536, %537] : tensor<32x32xf32>
      linalg.yield %538 : f32
    } -> tensor<1x32x32xf32>
    %539 = tensor.collapse_shape %533 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %540 = tensor.expand_shape %539 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %541 = tensor.empty() : tensor<1x32x32xf32>
    %542 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%123 : tensor<1x32xi64>) outs(%541 : tensor<1x32x32xf32>) attrs =  {prov.region_id = "gather_1", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb46(%543: i64, %544: f32):
      %545 = arith.index_cast %543 : i64 to index
      %546 = linalg.index 2 : index
      %547 = tensor.extract %531[%545, %546] : tensor<32x32xf32>
      linalg.yield %547 : f32
    } -> tensor<1x32x32xf32>
    %548 = tensor.collapse_shape %542 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %549 = tensor.expand_shape %548 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %550 = tensor.empty() : tensor<1x8x32x32xf32>
    %551 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%521, %540 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%550 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb47(%552: f32, %553: f32, %554: f32):
      %555 = arith.mulf %552, %553 : f32
      linalg.yield %555 : f32
    } -> tensor<1x8x32x32xf32>
    %556 = "tensor.extract_slice"(%521) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_9", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x8x32x32xf32>) -> tensor<1x8x32x16xf32>
    %557 = "tensor.extract_slice"(%521) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 8, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_10", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x8x32x32xf32>) -> tensor<1x8x32x16xf32>
    %558 = tensor.empty() : tensor<1x8x32x16xf32>
    %559 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%557 : tensor<1x8x32x16xf32>) outs(%558 : tensor<1x8x32x16xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb48(%560: f32, %561: f32):
      %562 = arith.negf %560 : f32
      linalg.yield %562 : f32
    } -> tensor<1x8x32x16xf32>
    %563 = tensor.concat dim(3) %559, %556 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x8x32x16xf32>, tensor<1x8x32x16xf32>) -> tensor<1x8x32x32xf32>
    %564 = tensor.empty() : tensor<1x8x32x32xf32>
    %565 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%563, %549 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%564 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb49(%566: f32, %567: f32, %568: f32):
      %569 = arith.mulf %566, %567 : f32
      linalg.yield %569 : f32
    } -> tensor<1x8x32x32xf32>
    %570 = tensor.empty() : tensor<1x8x32x32xf32>
    %571 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%551, %565 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%570 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb50(%572: f32, %573: f32, %574: f32):
      %575 = arith.addf %572, %573 : f32
      linalg.yield %575 : f32
    } -> tensor<1x8x32x32xf32>
    %576 = tensor.empty() : tensor<1x4x32x32xf32>
    %577 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%525, %540 : tensor<1x4x32x32xf32>, tensor<1x1x32x32xf32>) outs(%576 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb51(%578: f32, %579: f32, %580: f32):
      %581 = arith.mulf %578, %579 : f32
      linalg.yield %581 : f32
    } -> tensor<1x4x32x32xf32>
    %582 = "tensor.extract_slice"(%525) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_11", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x16xf32>
    %583 = "tensor.extract_slice"(%525) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_12", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x16xf32>
    %584 = tensor.empty() : tensor<1x4x32x16xf32>
    %585 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%583 : tensor<1x4x32x16xf32>) outs(%584 : tensor<1x4x32x16xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb52(%586: f32, %587: f32):
      %588 = arith.negf %586 : f32
      linalg.yield %588 : f32
    } -> tensor<1x4x32x16xf32>
    %589 = tensor.concat dim(3) %585, %582 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x16xf32>, tensor<1x4x32x16xf32>) -> tensor<1x4x32x32xf32>
    %590 = tensor.empty() : tensor<1x4x32x32xf32>
    %591 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%589, %549 : tensor<1x4x32x32xf32>, tensor<1x1x32x32xf32>) outs(%590 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb53(%592: f32, %593: f32, %594: f32):
      %595 = arith.mulf %592, %593 : f32
      linalg.yield %595 : f32
    } -> tensor<1x4x32x32xf32>
    %596 = tensor.empty() : tensor<1x4x32x32xf32>
    %597 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%577, %591 : tensor<1x4x32x32xf32>, tensor<1x4x32x32xf32>) outs(%596 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb54(%598: f32, %599: f32, %600: f32):
      %601 = arith.addf %598, %599 : f32
      linalg.yield %601 : f32
    } -> tensor<1x4x32x32xf32>
    %602 = "tensor.extract_slice"(%597) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_13", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %603 = "tensor.extract_slice"(%602) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_14", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %604 = tensor.collapse_shape %603 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x32x32xf32> into tensor<4096xf32>
    %605 = tensor.expand_shape %604 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 32, 32] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<4096xf32> into tensor<1x4x1x32x32xf32>
    %606 = "tensor.extract_slice"(%605) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_15", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %607 = "tensor.extract_slice"(%606) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_16", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %608 = tensor.empty() : tensor<1x4x2x32x32xf32>
    %609 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%607 : tensor<1x4x1x32x32xf32>) outs(%608 : tensor<1x4x2x32x32xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb55(%610: f32, %611: f32):
      linalg.yield %610 : f32
    } -> tensor<1x4x2x32x32xf32>
    %612 = tensor.collapse_shape %609 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x2x32x32xf32> into tensor<8192xf32>
    %613 = tensor.expand_shape %612 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 32] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x8x32x32xf32>
    %614 = "tensor.extract_slice"(%529) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_17", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %615 = "tensor.extract_slice"(%614) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_18", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %616 = tensor.collapse_shape %615 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x32x32xf32> into tensor<4096xf32>
    %617 = tensor.expand_shape %616 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 32, 32] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<4096xf32> into tensor<1x4x1x32x32xf32>
    %618 = "tensor.extract_slice"(%617) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_19", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %619 = "tensor.extract_slice"(%618) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_20", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %620 = tensor.empty() : tensor<1x4x2x32x32xf32>
    %621 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%619 : tensor<1x4x1x32x32xf32>) outs(%620 : tensor<1x4x2x32x32xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb56(%622: f32, %623: f32):
      linalg.yield %622 : f32
    } -> tensor<1x4x2x32x32xf32>
    %624 = tensor.collapse_shape %621 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x2x32x32xf32> into tensor<8192xf32>
    %625 = tensor.expand_shape %624 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 32] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x8x32x32xf32>
    %626 = tensor.empty() : tensor<1x8x32x32xf32>
    %627 = linalg.transpose ins(%613:tensor<1x8x32x32xf32>) outs(%626:tensor<1x8x32x32xf32>) permutation = [0, 1, 3, 2]
    %628 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 0.000000e+00 : f32
    %629 = tensor.splat %628 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32x32xf32>
    %630 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%571, %627 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%629 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb57(%631: f32, %632: f32, %633: f32):
      %634 = arith.mulf %631, %632 : f32
      %635 = arith.addf %633, %634 : f32
      linalg.yield %635 : f32
    } -> tensor<1x8x32x32xf32>
    %636 = arith.constant {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 5.65685415 : f32
    %637 = tensor.splat %636 {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32x32xf32>
    %638 = tensor.empty() : tensor<1x8x32x32xf32>
    %639 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%630, %637 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%638 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb58(%640: f32, %641: f32, %642: f32):
      %643 = arith.divf %640, %641 : f32
      linalg.yield %643 : f32
    } -> tensor<1x8x32x32xf32>
    %644 = "tensor.extract_slice"(%188) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_21", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    %645 = "tensor.extract_slice"(%644) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_22", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    %646 = "tensor.extract_slice"(%645) <{static_offsets = array<i64: 0, 0, 31, 0>, static_sizes = array<i64: 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<32xf32>
    %647 = tensor.expand_shape %646 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "slice_23", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<32xf32> into tensor<1x1x32xf32>
    %648 = tensor.collapse_shape %647 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x1x32xf32> into tensor<32xf32>
    %649 = tensor.expand_shape %648 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<32xf32> into tensor<1x1x1x32xf32>
    %650 = tensor.empty() : tensor<1x1x32x32xf32>
    %651 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%649 : tensor<1x1x1x32xf32>) outs(%650 : tensor<1x1x32x32xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb59(%652: f32, %653: f32):
      linalg.yield %652 : f32
    } -> tensor<1x1x32x32xf32>
    %654 = tensor.empty() : tensor<1x8x32x32xf32>
    %655 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%639, %651 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%654 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb60(%656: f32, %657: f32, %658: f32):
      %659 = arith.addf %656, %657 : f32
      linalg.yield %659 : f32
    } -> tensor<1x8x32x32xf32>
    %660 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 0xff800000 : f32
    %661 = tensor.splat %660 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32xf32>
    %662 = linalg.reduce ins(%655:tensor<1x8x32x32xf32>) outs(%661:tensor<1x8x32xf32>) dimensions = [3]
    (%663: f32, %664: f32) {
      %665 = arith.maximumf %663, %664 : f32
      linalg.yield %665 : f32
    }
    %666 = tensor.collapse_shape %662 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %667 = tensor.expand_shape %666 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %668 = tensor.empty() : tensor<1x8x32x32xf32>
    %669 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%655, %667 : tensor<1x8x32x32xf32>, tensor<1x8x32x1xf32>) outs(%668 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb61(%670: f32, %671: f32, %672: f32):
      %673 = arith.subf %670, %671 : f32
      linalg.yield %673 : f32
    } -> tensor<1x8x32x32xf32>
    %674 = tensor.empty() : tensor<1x8x32x32xf32>
    %675 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%669 : tensor<1x8x32x32xf32>) outs(%674 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb62(%676: f32, %677: f32):
      %678 = math.exp %676 : f32
      linalg.yield %678 : f32
    } -> tensor<1x8x32x32xf32>
    %679 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 0.000000e+00 : f32
    %680 = tensor.splat %679 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32xf32>
    %681 = linalg.reduce ins(%675:tensor<1x8x32x32xf32>) outs(%680:tensor<1x8x32xf32>) dimensions = [3]
    (%682: f32, %683: f32) {
      %684 = arith.addf %682, %683 : f32
      linalg.yield %684 : f32
    }
    %685 = tensor.collapse_shape %681 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %686 = tensor.expand_shape %685 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %687 = tensor.empty() : tensor<1x8x32x32xf32>
    %688 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%675, %686 : tensor<1x8x32x32xf32>, tensor<1x8x32x1xf32>) outs(%687 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb63(%689: f32, %690: f32, %691: f32):
      %692 = arith.divf %689, %690 : f32
      linalg.yield %692 : f32
    } -> tensor<1x8x32x32xf32>
    %693 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 0.000000e+00 : f32
    %694 = tensor.splat %693 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32x32xf32>
    %695 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%688, %625 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%694 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb64(%696: f32, %697: f32, %698: f32):
      %699 = arith.mulf %696, %697 : f32
      %700 = arith.addf %698, %699 : f32
      linalg.yield %700 : f32
    } -> tensor<1x8x32x32xf32>
    %701 = tensor.empty() : tensor<1x32x8x32xf32>
    %702 = linalg.transpose ins(%695:tensor<1x8x32x32xf32>) outs(%701:tensor<1x32x8x32xf32>) permutation = [0, 2, 1, 3]
    %703 = tensor.collapse_shape %702 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x8x32xf32> into tensor<8192xf32>
    %704 = tensor.expand_shape %703 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 256] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x32x256xf32>
    %705 = tensor.empty() : tensor<1x32x256xf32>
    %706 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%704 : tensor<1x32x256xf32>) outs(%705 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb65(%707: f32, %708: f32):
      %709 = arith.constant 2.000000e+00 : f32
      %710 = math.powf %707, %709 : f32
      linalg.yield %710 : f32
    } -> tensor<1x32x256xf32>
    %711 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} 0.000000e+00 : f32
    %712 = tensor.splat %711 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} : tensor<1x32xf32>
    %713 = linalg.reduce ins(%706:tensor<1x32x256xf32>) outs(%712:tensor<1x32xf32>) dimensions = [2]
    (%714: f32, %715: f32) {
      %716 = arith.addf %714, %715 : f32
      linalg.yield %716 : f32
    }
    %717 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} 2.560000e+02 : f32
    %718 = tensor.splat %717 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} : tensor<1x32xf32>
    %719 = tensor.empty() : tensor<1x32xf32>
    %720 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%713, %718 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%719 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb66(%721: f32, %722: f32, %723: f32):
      %724 = arith.divf %721, %722 : f32
      linalg.yield %724 : f32
    } -> tensor<1x32xf32>
    %725 = tensor.collapse_shape %720 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} : tensor<1x32xf32> into tensor<32xf32>
    %726 = tensor.expand_shape %725 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %727 = arith.constant {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} 1.000000e-05 : f32
    %728 = tensor.splat %727 {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} : tensor<1x32x1xf32>
    %729 = tensor.empty() : tensor<1x32x1xf32>
    %730 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%726, %728 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%729 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb67(%731: f32, %732: f32, %733: f32):
      %734 = arith.addf %731, %732 : f32
      linalg.yield %734 : f32
    } -> tensor<1x32x1xf32>
    %735 = tensor.empty() : tensor<1x32x1xf32>
    %736 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%730 : tensor<1x32x1xf32>) outs(%735 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb68(%737: f32, %738: f32):
      %739 = math.rsqrt %737 : f32
      linalg.yield %739 : f32
    } -> tensor<1x32x1xf32>
    %740 = tensor.empty() : tensor<1x32x256xf32>
    %741 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%704, %736 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%740 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb69(%742: f32, %743: f32, %744: f32):
      %745 = arith.mulf %742, %743 : f32
      linalg.yield %745 : f32
    } -> tensor<1x32x256xf32>
    %746 = tensor.empty() : tensor<1x32x256xf32>
    %747 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %741 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%746 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb70(%748: f32, %749: f32, %750: f32):
      %751 = arith.mulf %748, %749 : f32
      linalg.yield %751 : f32
    } -> tensor<1x32x256xf32>
    %752 = tensor.empty() : tensor<1x32x256xf32>
    %753 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%747 : tensor<1x32x256xf32>) outs(%752 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_3", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb71(%754: f32, %755: f32):
      %756 = math.absf %754 : f32
      linalg.yield %756 : f32
    } -> tensor<1x32x256xf32>
    %757 = arith.constant {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} 0xff800000 : f32
    %758 = arith.constant {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} 0 : i64
    %759 = tensor.splat %757 {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<1x32xf32>
    %760 = tensor.splat %758 {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<1x32xi64>
    %761, %762 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%753 : tensor<1x32x256xf32>) outs(%759, %760 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb72(%763: f32, %764: f32, %765: i64):
      %766 = linalg.index 2 : index
      %767 = arith.index_cast %766 : index to i64
      %768 = arith.cmpf ogt, %763, %764 : f32
      %769 = arith.select %768, %763, %764 : f32
      %770 = arith.select %768, %767, %765 : i64
      linalg.yield %769, %770 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %771 = tensor.collapse_shape %761 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %772 = tensor.expand_shape %771 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %773 = tensor.collapse_shape %762 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %774 = tensor.expand_shape %773 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %775 = func.call @aten_clamp__default(%772) {prov.region_id = "aten_clamp__default_3", prov.dispatch_id = "aten_clamp__default_3"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %776 = tensor.empty() : tensor<1x32x1xf32>
    %777 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%775 : tensor<1x32x1xf32>) outs(%776 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_3", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb73(%778: f32, %779: f32):
      %780 = arith.constant 1.000000e+00 : f32
      %781 = arith.divf %780, %778 : f32
      linalg.yield %781 : f32
    } -> tensor<1x32x1xf32>
    %782 = arith.constant {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} 1.270000e+02 : f32
    %783 = tensor.splat %782 {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<1x32x1xf32>
    %784 = tensor.empty() : tensor<1x32x1xf32>
    %785 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%777, %783 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%784 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb74(%786: f32, %787: f32, %788: f32):
      %789 = arith.mulf %786, %787 : f32
      linalg.yield %789 : f32
    } -> tensor<1x32x1xf32>
    %790 = tensor.empty() : tensor<1x32x256xf32>
    %791 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%747, %785 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%790 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb75(%792: f32, %793: f32, %794: f32):
      %795 = arith.mulf %792, %793 : f32
      linalg.yield %795 : f32
    } -> tensor<1x32x256xf32>
    %796 = tensor.empty() : tensor<1x32x256xf32>
    %797 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%791 : tensor<1x32x256xf32>) outs(%796 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_3", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb76(%798: f32, %799: f32):
      %800 = math.roundeven %798 : f32
      linalg.yield %800 : f32
    } -> tensor<1x32x256xf32>
    %801 = tensor.empty() : tensor<1x32x256xf32>
    %802 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%797 : tensor<1x32x256xf32>) outs(%801 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_3", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb77(%803: f32, %804: f32):
      %805 = arith.constant -1.280000e+02 : f32
      %806 = arith.maximumf %803, %805 : f32
      %807 = arith.constant 1.270000e+02 : f32
      %808 = arith.minimumf %806, %807 : f32
      linalg.yield %808 : f32
    } -> tensor<1x32x256xf32>
    %809 = tensor.empty() : tensor<1x32x256xf32>
    %810 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%802, %785 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%809 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb78(%811: f32, %812: f32, %813: f32):
      %814 = arith.divf %811, %812 : f32
      linalg.yield %814 : f32
    } -> tensor<1x32x256xf32>
    %815 = func.call @aten___and___Scalar(%84) {prov.region_id = "aten___and___Scalar_4", prov.dispatch_id = "aten___and___Scalar_4"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %816 = func.call @aten___rshift___Scalar(%84) {prov.region_id = "aten___rshift___Scalar_3", prov.dispatch_id = "aten___rshift___Scalar_3"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %817 = func.call @aten___and___Scalar(%816) {prov.region_id = "aten___and___Scalar_5", prov.dispatch_id = "aten___and___Scalar_5"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %818 = func.call @aten___rshift___Scalar(%84) {prov.region_id = "aten___rshift___Scalar_4", prov.dispatch_id = "aten___rshift___Scalar_4"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %819 = func.call @aten___and___Scalar(%818) {prov.region_id = "aten___and___Scalar_6", prov.dispatch_id = "aten___and___Scalar_6"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %820 = func.call @aten___rshift___Scalar(%84) {prov.region_id = "aten___rshift___Scalar_5", prov.dispatch_id = "aten___rshift___Scalar_5"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %821 = func.call @aten___and___Scalar(%820) {prov.region_id = "aten___and___Scalar_7", prov.dispatch_id = "aten___and___Scalar_7"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %822 = func.call @aten_stack_default(%815, %817, %819, %821) {prov.region_id = "aten_stack_default_1", prov.dispatch_id = "aten_stack_default_1"} : (tensor<16384xi8>, tensor<16384xi8>, tensor<16384xi8>, tensor<16384xi8>) -> tensor<16384x4xi8>
    %823 = tensor.collapse_shape %822 [[0 : i64, 1 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<16384x4xi8> into tensor<65536xi8>
    %824 = "tensor.extract_slice"(%823) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 65536>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_24", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : (tensor<65536xi8>) -> tensor<65536xi8>
    %825 = tensor.empty() : tensor<65536xf32>
    %826 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%824 : tensor<65536xi8>) outs(%825 : tensor<65536xf32>) attrs =  {prov.region_id = "dtype_cast_6", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb79(%827: i8, %828: f32):
      %829 = arith.sitofp %827 : i8 to f32
      linalg.yield %829 : f32
    } -> tensor<65536xf32>
    %830 = arith.constant {prov.region_id = "sub_3", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} 1.000000e+00 : f32
    %831 = tensor.splat %830 {prov.region_id = "sub_3", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<65536xf32>
    %832 = tensor.empty() : tensor<65536xf32>
    %833 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%826, %831 : tensor<65536xf32>, tensor<65536xf32>) outs(%832 : tensor<65536xf32>) attrs =  {prov.region_id = "sub_3", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb80(%834: f32, %835: f32, %836: f32):
      %837 = arith.subf %834, %835 : f32
      linalg.yield %837 : f32
    } -> tensor<65536xf32>
    %838 = func.call @aten_mul_Tensor(%833) {prov.region_id = "aten_mul_Tensor_1", prov.dispatch_id = "aten_mul_Tensor_1"} : (tensor<65536xf32>) -> tensor<65536xf32>
    %839 = tensor.expand_shape %838 [[0 : i64, 1 : i64]] output_shape [256, 256] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<65536xf32> into tensor<256x256xf32>
    %840 = tensor.empty() : tensor<256x256xf32>
    %841 = linalg.transpose ins(%839:tensor<256x256xf32>) outs(%840:tensor<256x256xf32>) permutation = [1, 0]
    %842 = tensor.empty() : tensor<1x32x256xf32>
    %843 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %844 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%843 : f32) outs(%842 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %845 = linalg.matmul {prov.region_id = "matmul_5", prov.dispatch_id = "matmul_5", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} ins(%810, %841 : tensor<1x32x256xf32>, tensor<256x256xf32>) outs(%844 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %846 = tensor.empty() : tensor<1x32x256xf32>
    %847 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%112, %845 : tensor<1x32x256xf32>, tensor<1x32x256xf32>) outs(%846 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0"} {
    ^bb81(%848: f32, %849: f32, %850: f32):
      %851 = arith.addf %848, %849 : f32
      linalg.yield %851 : f32
    } -> tensor<1x32x256xf32>
    %852 = tensor.empty() : tensor<1x32x256xf32>
    %853 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%847 : tensor<1x32x256xf32>) outs(%852 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb82(%854: f32, %855: f32):
      %856 = arith.constant 2.000000e+00 : f32
      %857 = math.powf %854, %856 : f32
      linalg.yield %857 : f32
    } -> tensor<1x32x256xf32>
    %858 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} 0.000000e+00 : f32
    %859 = tensor.splat %858 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<1x32xf32>
    %860 = linalg.reduce ins(%853:tensor<1x32x256xf32>) outs(%859:tensor<1x32xf32>) dimensions = [2]
    (%861: f32, %862: f32) {
      %863 = arith.addf %861, %862 : f32
      linalg.yield %863 : f32
    }
    %864 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} 2.560000e+02 : f32
    %865 = tensor.splat %864 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<1x32xf32>
    %866 = tensor.empty() : tensor<1x32xf32>
    %867 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%860, %865 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%866 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb83(%868: f32, %869: f32, %870: f32):
      %871 = arith.divf %868, %869 : f32
      linalg.yield %871 : f32
    } -> tensor<1x32xf32>
    %872 = tensor.collapse_shape %867 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %873 = tensor.expand_shape %872 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %874 = arith.constant {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} 1.000000e-05 : f32
    %875 = tensor.splat %874 {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<1x32x1xf32>
    %876 = tensor.empty() : tensor<1x32x1xf32>
    %877 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%873, %875 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%876 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb84(%878: f32, %879: f32, %880: f32):
      %881 = arith.addf %878, %879 : f32
      linalg.yield %881 : f32
    } -> tensor<1x32x1xf32>
    %882 = tensor.empty() : tensor<1x32x1xf32>
    %883 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%877 : tensor<1x32x1xf32>) outs(%882 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb85(%884: f32, %885: f32):
      %886 = math.rsqrt %884 : f32
      linalg.yield %886 : f32
    } -> tensor<1x32x1xf32>
    %887 = tensor.empty() : tensor<1x32x256xf32>
    %888 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%847, %883 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%887 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb86(%889: f32, %890: f32, %891: f32):
      %892 = arith.mulf %889, %890 : f32
      linalg.yield %892 : f32
    } -> tensor<1x32x256xf32>
    %893 = tensor.empty() : tensor<1x32x256xf32>
    %894 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %888 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%893 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb87(%895: f32, %896: f32, %897: f32):
      %898 = arith.mulf %895, %896 : f32
      linalg.yield %898 : f32
    } -> tensor<1x32x256xf32>
    %899 = tensor.empty() : tensor<1x32x256xf32>
    %900 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%894 : tensor<1x32x256xf32>) outs(%899 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_4", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb88(%901: f32, %902: f32):
      %903 = math.absf %901 : f32
      linalg.yield %903 : f32
    } -> tensor<1x32x256xf32>
    %904 = arith.constant {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} 0xff800000 : f32
    %905 = arith.constant {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} 0 : i64
    %906 = tensor.splat %904 {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<1x32xf32>
    %907 = tensor.splat %905 {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<1x32xi64>
    %908, %909 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%900 : tensor<1x32x256xf32>) outs(%906, %907 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb89(%910: f32, %911: f32, %912: i64):
      %913 = linalg.index 2 : index
      %914 = arith.index_cast %913 : index to i64
      %915 = arith.cmpf ogt, %910, %911 : f32
      %916 = arith.select %915, %910, %911 : f32
      %917 = arith.select %915, %914, %912 : i64
      linalg.yield %916, %917 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %918 = tensor.collapse_shape %908 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %919 = tensor.expand_shape %918 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %920 = tensor.collapse_shape %909 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %921 = tensor.expand_shape %920 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %922 = func.call @aten_clamp__default(%919) {prov.region_id = "aten_clamp__default_4", prov.dispatch_id = "aten_clamp__default_4"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %923 = tensor.empty() : tensor<1x32x1xf32>
    %924 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%922 : tensor<1x32x1xf32>) outs(%923 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_4", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb90(%925: f32, %926: f32):
      %927 = arith.constant 1.000000e+00 : f32
      %928 = arith.divf %927, %925 : f32
      linalg.yield %928 : f32
    } -> tensor<1x32x1xf32>
    %929 = arith.constant {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} 1.270000e+02 : f32
    %930 = tensor.splat %929 {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<1x32x1xf32>
    %931 = tensor.empty() : tensor<1x32x1xf32>
    %932 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%924, %930 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%931 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb91(%933: f32, %934: f32, %935: f32):
      %936 = arith.mulf %933, %934 : f32
      linalg.yield %936 : f32
    } -> tensor<1x32x1xf32>
    %937 = tensor.empty() : tensor<1x32x256xf32>
    %938 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%894, %932 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%937 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb92(%939: f32, %940: f32, %941: f32):
      %942 = arith.mulf %939, %940 : f32
      linalg.yield %942 : f32
    } -> tensor<1x32x256xf32>
    %943 = tensor.empty() : tensor<1x32x256xf32>
    %944 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%938 : tensor<1x32x256xf32>) outs(%943 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_4", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb93(%945: f32, %946: f32):
      %947 = math.roundeven %945 : f32
      linalg.yield %947 : f32
    } -> tensor<1x32x256xf32>
    %948 = tensor.empty() : tensor<1x32x256xf32>
    %949 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%944 : tensor<1x32x256xf32>) outs(%948 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_4", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb94(%950: f32, %951: f32):
      %952 = arith.constant -1.280000e+02 : f32
      %953 = arith.maximumf %950, %952 : f32
      %954 = arith.constant 1.270000e+02 : f32
      %955 = arith.minimumf %953, %954 : f32
      linalg.yield %955 : f32
    } -> tensor<1x32x256xf32>
    %956 = tensor.empty() : tensor<1x32x256xf32>
    %957 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%949, %932 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%956 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb95(%958: f32, %959: f32, %960: f32):
      %961 = arith.divf %958, %959 : f32
      linalg.yield %961 : f32
    } -> tensor<1x32x256xf32>
    %962 = func.call @aten___and___Scalar_2(%89) {prov.region_id = "aten___and___Scalar_2_0", prov.dispatch_id = "aten___and___Scalar_2_0"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %963 = func.call @aten___rshift___Scalar_2(%89) {prov.region_id = "aten___rshift___Scalar_2_0", prov.dispatch_id = "aten___rshift___Scalar_2_0"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %964 = func.call @aten___and___Scalar_2(%963) {prov.region_id = "aten___and___Scalar_2_1", prov.dispatch_id = "aten___and___Scalar_2_1"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %965 = func.call @aten___rshift___Scalar_2(%89) {prov.region_id = "aten___rshift___Scalar_2_1", prov.dispatch_id = "aten___rshift___Scalar_2_1"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %966 = func.call @aten___and___Scalar_2(%965) {prov.region_id = "aten___and___Scalar_2_2", prov.dispatch_id = "aten___and___Scalar_2_2"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %967 = func.call @aten___rshift___Scalar_2(%89) {prov.region_id = "aten___rshift___Scalar_2_2", prov.dispatch_id = "aten___rshift___Scalar_2_2"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %968 = func.call @aten___and___Scalar_2(%967) {prov.region_id = "aten___and___Scalar_2_3", prov.dispatch_id = "aten___and___Scalar_2_3"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %969 = func.call @aten_stack_default_2(%962, %964, %966, %968) {prov.region_id = "aten_stack_default_2_0", prov.dispatch_id = "aten_stack_default_2_0"} : (tensor<32768xi8>, tensor<32768xi8>, tensor<32768xi8>, tensor<32768xi8>) -> tensor<32768x4xi8>
    %970 = tensor.collapse_shape %969 [[0 : i64, 1 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<32768x4xi8> into tensor<131072xi8>
    %971 = "tensor.extract_slice"(%970) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 131072>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_25", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : (tensor<131072xi8>) -> tensor<131072xi8>
    %972 = tensor.empty() : tensor<131072xf32>
    %973 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%971 : tensor<131072xi8>) outs(%972 : tensor<131072xf32>) attrs =  {prov.region_id = "dtype_cast_7", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb96(%974: i8, %975: f32):
      %976 = arith.sitofp %974 : i8 to f32
      linalg.yield %976 : f32
    } -> tensor<131072xf32>
    %977 = arith.constant {prov.region_id = "sub_4", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} 1.000000e+00 : f32
    %978 = tensor.splat %977 {prov.region_id = "sub_4", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<131072xf32>
    %979 = tensor.empty() : tensor<131072xf32>
    %980 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%973, %978 : tensor<131072xf32>, tensor<131072xf32>) outs(%979 : tensor<131072xf32>) attrs =  {prov.region_id = "sub_4", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb97(%981: f32, %982: f32, %983: f32):
      %984 = arith.subf %981, %982 : f32
      linalg.yield %984 : f32
    } -> tensor<131072xf32>
    %985 = func.call @aten_mul_Tensor_2(%980) {prov.region_id = "aten_mul_Tensor_2_0", prov.dispatch_id = "aten_mul_Tensor_2_0"} : (tensor<131072xf32>) -> tensor<131072xf32>
    %986 = tensor.expand_shape %985 [[0 : i64, 1 : i64]] output_shape [512, 256] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<131072xf32> into tensor<512x256xf32>
    %987 = tensor.empty() : tensor<256x512xf32>
    %988 = linalg.transpose ins(%986:tensor<512x256xf32>) outs(%987:tensor<256x512xf32>) permutation = [1, 0]
    %989 = tensor.empty() : tensor<1x32x512xf32>
    %990 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %991 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%990 : f32) outs(%989 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %992 = linalg.matmul {prov.region_id = "matmul_6", prov.dispatch_id = "matmul_6", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} ins(%957, %988 : tensor<1x32x256xf32>, tensor<256x512xf32>) outs(%991 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %993 = tensor.empty() : tensor<1x32x512xf32>
    %994 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%992 : tensor<1x32x512xf32>) outs(%993 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "minmax_5", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp"} {
    ^bb98(%995: f32, %996: f32):
      %997 = arith.constant 0.000000e+00 : f32
      %998 = arith.maximumf %995, %997 : f32
      linalg.yield %998 : f32
    } -> tensor<1x32x512xf32>
    %999 = tensor.empty() : tensor<1x32x512xf32>
    %1000 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%994 : tensor<1x32x512xf32>) outs(%999 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp"} {
    ^bb99(%1001: f32, %1002: f32):
      %1003 = arith.constant 2.000000e+00 : f32
      %1004 = math.powf %1001, %1003 : f32
      linalg.yield %1004 : f32
    } -> tensor<1x32x512xf32>
    %1005 = tensor.empty() : tensor<1x32x256xf32>
    %1006 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%894 : tensor<1x32x256xf32>) outs(%1005 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_5", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb100(%1007: f32, %1008: f32):
      %1009 = math.absf %1007 : f32
      linalg.yield %1009 : f32
    } -> tensor<1x32x256xf32>
    %1010 = arith.constant {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} 0xff800000 : f32
    %1011 = arith.constant {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} 0 : i64
    %1012 = tensor.splat %1010 {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<1x32xf32>
    %1013 = tensor.splat %1011 {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<1x32xi64>
    %1014, %1015 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1006 : tensor<1x32x256xf32>) outs(%1012, %1013 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb101(%1016: f32, %1017: f32, %1018: i64):
      %1019 = linalg.index 2 : index
      %1020 = arith.index_cast %1019 : index to i64
      %1021 = arith.cmpf ogt, %1016, %1017 : f32
      %1022 = arith.select %1021, %1016, %1017 : f32
      %1023 = arith.select %1021, %1020, %1018 : i64
      linalg.yield %1022, %1023 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1024 = tensor.collapse_shape %1014 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1025 = tensor.expand_shape %1024 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1026 = tensor.collapse_shape %1015 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1027 = tensor.expand_shape %1026 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1028 = func.call @aten_clamp__default(%1025) {prov.region_id = "aten_clamp__default_5", prov.dispatch_id = "aten_clamp__default_5"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1029 = tensor.empty() : tensor<1x32x1xf32>
    %1030 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1028 : tensor<1x32x1xf32>) outs(%1029 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_5", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb102(%1031: f32, %1032: f32):
      %1033 = arith.constant 1.000000e+00 : f32
      %1034 = arith.divf %1033, %1031 : f32
      linalg.yield %1034 : f32
    } -> tensor<1x32x1xf32>
    %1035 = arith.constant {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} 1.270000e+02 : f32
    %1036 = tensor.splat %1035 {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<1x32x1xf32>
    %1037 = tensor.empty() : tensor<1x32x1xf32>
    %1038 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1030, %1036 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1037 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb103(%1039: f32, %1040: f32, %1041: f32):
      %1042 = arith.mulf %1039, %1040 : f32
      linalg.yield %1042 : f32
    } -> tensor<1x32x1xf32>
    %1043 = tensor.empty() : tensor<1x32x256xf32>
    %1044 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%894, %1038 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1043 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb104(%1045: f32, %1046: f32, %1047: f32):
      %1048 = arith.mulf %1045, %1046 : f32
      linalg.yield %1048 : f32
    } -> tensor<1x32x256xf32>
    %1049 = tensor.empty() : tensor<1x32x256xf32>
    %1050 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1044 : tensor<1x32x256xf32>) outs(%1049 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_5", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb105(%1051: f32, %1052: f32):
      %1053 = math.roundeven %1051 : f32
      linalg.yield %1053 : f32
    } -> tensor<1x32x256xf32>
    %1054 = tensor.empty() : tensor<1x32x256xf32>
    %1055 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1050 : tensor<1x32x256xf32>) outs(%1054 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_6", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb106(%1056: f32, %1057: f32):
      %1058 = arith.constant -1.280000e+02 : f32
      %1059 = arith.maximumf %1056, %1058 : f32
      %1060 = arith.constant 1.270000e+02 : f32
      %1061 = arith.minimumf %1059, %1060 : f32
      linalg.yield %1061 : f32
    } -> tensor<1x32x256xf32>
    %1062 = tensor.empty() : tensor<1x32x256xf32>
    %1063 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1055, %1038 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1062 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_6", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb107(%1064: f32, %1065: f32, %1066: f32):
      %1067 = arith.divf %1064, %1065 : f32
      linalg.yield %1067 : f32
    } -> tensor<1x32x256xf32>
    %1068 = func.call @aten___and___Scalar_2(%91) {prov.region_id = "aten___and___Scalar_2_4", prov.dispatch_id = "aten___and___Scalar_2_4"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1069 = func.call @aten___rshift___Scalar_2(%91) {prov.region_id = "aten___rshift___Scalar_2_3", prov.dispatch_id = "aten___rshift___Scalar_2_3"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1070 = func.call @aten___and___Scalar_2(%1069) {prov.region_id = "aten___and___Scalar_2_5", prov.dispatch_id = "aten___and___Scalar_2_5"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1071 = func.call @aten___rshift___Scalar_2(%91) {prov.region_id = "aten___rshift___Scalar_2_4", prov.dispatch_id = "aten___rshift___Scalar_2_4"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1072 = func.call @aten___and___Scalar_2(%1071) {prov.region_id = "aten___and___Scalar_2_6", prov.dispatch_id = "aten___and___Scalar_2_6"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1073 = func.call @aten___rshift___Scalar_2(%91) {prov.region_id = "aten___rshift___Scalar_2_5", prov.dispatch_id = "aten___rshift___Scalar_2_5"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1074 = func.call @aten___and___Scalar_2(%1073) {prov.region_id = "aten___and___Scalar_2_7", prov.dispatch_id = "aten___and___Scalar_2_7"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1075 = func.call @aten_stack_default_2(%1068, %1070, %1072, %1074) {prov.region_id = "aten_stack_default_2_1", prov.dispatch_id = "aten_stack_default_2_1"} : (tensor<32768xi8>, tensor<32768xi8>, tensor<32768xi8>, tensor<32768xi8>) -> tensor<32768x4xi8>
    %1076 = tensor.collapse_shape %1075 [[0 : i64, 1 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<32768x4xi8> into tensor<131072xi8>
    %1077 = "tensor.extract_slice"(%1076) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 131072>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_26", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : (tensor<131072xi8>) -> tensor<131072xi8>
    %1078 = tensor.empty() : tensor<131072xf32>
    %1079 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1077 : tensor<131072xi8>) outs(%1078 : tensor<131072xf32>) attrs =  {prov.region_id = "dtype_cast_8", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb108(%1080: i8, %1081: f32):
      %1082 = arith.sitofp %1080 : i8 to f32
      linalg.yield %1082 : f32
    } -> tensor<131072xf32>
    %1083 = arith.constant {prov.region_id = "sub_5", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} 1.000000e+00 : f32
    %1084 = tensor.splat %1083 {prov.region_id = "sub_5", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<131072xf32>
    %1085 = tensor.empty() : tensor<131072xf32>
    %1086 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1079, %1084 : tensor<131072xf32>, tensor<131072xf32>) outs(%1085 : tensor<131072xf32>) attrs =  {prov.region_id = "sub_5", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb109(%1087: f32, %1088: f32, %1089: f32):
      %1090 = arith.subf %1087, %1088 : f32
      linalg.yield %1090 : f32
    } -> tensor<131072xf32>
    %1091 = func.call @aten_mul_Tensor_2(%1086) {prov.region_id = "aten_mul_Tensor_2_1", prov.dispatch_id = "aten_mul_Tensor_2_1"} : (tensor<131072xf32>) -> tensor<131072xf32>
    %1092 = tensor.expand_shape %1091 [[0 : i64, 1 : i64]] output_shape [512, 256] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<131072xf32> into tensor<512x256xf32>
    %1093 = tensor.empty() : tensor<256x512xf32>
    %1094 = linalg.transpose ins(%1092:tensor<512x256xf32>) outs(%1093:tensor<256x512xf32>) permutation = [1, 0]
    %1095 = tensor.empty() : tensor<1x32x512xf32>
    %1096 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1097 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1096 : f32) outs(%1095 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %1098 = linalg.matmul {prov.region_id = "matmul_7", prov.dispatch_id = "matmul_7", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} ins(%1063, %1094 : tensor<1x32x256xf32>, tensor<256x512xf32>) outs(%1097 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %1099 = tensor.empty() : tensor<1x32x512xf32>
    %1100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1000, %1098 : tensor<1x32x512xf32>, tensor<1x32x512xf32>) outs(%1099 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp"} {
    ^bb110(%1101: f32, %1102: f32, %1103: f32):
      %1104 = arith.mulf %1101, %1102 : f32
      linalg.yield %1104 : f32
    } -> tensor<1x32x512xf32>
    %1105 = tensor.empty() : tensor<1x32x512xf32>
    %1106 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1100 : tensor<1x32x512xf32>) outs(%1105 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb111(%1107: f32, %1108: f32):
      %1109 = arith.constant 2.000000e+00 : f32
      %1110 = math.powf %1107, %1109 : f32
      linalg.yield %1110 : f32
    } -> tensor<1x32x512xf32>
    %1111 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} 0.000000e+00 : f32
    %1112 = tensor.splat %1111 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} : tensor<1x32xf32>
    %1113 = linalg.reduce ins(%1106:tensor<1x32x512xf32>) outs(%1112:tensor<1x32xf32>) dimensions = [2]
    (%1114: f32, %1115: f32) {
      %1116 = arith.addf %1114, %1115 : f32
      linalg.yield %1116 : f32
    }
    %1117 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} 5.120000e+02 : f32
    %1118 = tensor.splat %1117 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} : tensor<1x32xf32>
    %1119 = tensor.empty() : tensor<1x32xf32>
    %1120 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1113, %1118 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1119 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb112(%1121: f32, %1122: f32, %1123: f32):
      %1124 = arith.divf %1121, %1122 : f32
      linalg.yield %1124 : f32
    } -> tensor<1x32xf32>
    %1125 = tensor.collapse_shape %1120 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} : tensor<1x32xf32> into tensor<32xf32>
    %1126 = tensor.expand_shape %1125 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1127 = arith.constant {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} 1.000000e-05 : f32
    %1128 = tensor.splat %1127 {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} : tensor<1x32x1xf32>
    %1129 = tensor.empty() : tensor<1x32x1xf32>
    %1130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1126, %1128 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1129 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb113(%1131: f32, %1132: f32, %1133: f32):
      %1134 = arith.addf %1131, %1132 : f32
      linalg.yield %1134 : f32
    } -> tensor<1x32x1xf32>
    %1135 = tensor.empty() : tensor<1x32x1xf32>
    %1136 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1130 : tensor<1x32x1xf32>) outs(%1135 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb114(%1137: f32, %1138: f32):
      %1139 = math.rsqrt %1137 : f32
      linalg.yield %1139 : f32
    } -> tensor<1x32x1xf32>
    %1140 = tensor.empty() : tensor<1x32x512xf32>
    %1141 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1100, %1136 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%1140 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb115(%1142: f32, %1143: f32, %1144: f32):
      %1145 = arith.mulf %1142, %1143 : f32
      linalg.yield %1145 : f32
    } -> tensor<1x32x512xf32>
    %1146 = tensor.empty() : tensor<1x32x512xf32>
    %1147 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %1141 : tensor<512xf32>, tensor<1x32x512xf32>) outs(%1146 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb116(%1148: f32, %1149: f32, %1150: f32):
      %1151 = arith.mulf %1148, %1149 : f32
      linalg.yield %1151 : f32
    } -> tensor<1x32x512xf32>
    %1152 = tensor.empty() : tensor<1x32x512xf32>
    %1153 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1147 : tensor<1x32x512xf32>) outs(%1152 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "abs_6", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb117(%1154: f32, %1155: f32):
      %1156 = math.absf %1154 : f32
      linalg.yield %1156 : f32
    } -> tensor<1x32x512xf32>
    %1157 = arith.constant {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} 0xff800000 : f32
    %1158 = arith.constant {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} 0 : i64
    %1159 = tensor.splat %1157 {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<1x32xf32>
    %1160 = tensor.splat %1158 {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<1x32xi64>
    %1161, %1162 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1153 : tensor<1x32x512xf32>) outs(%1159, %1160 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb118(%1163: f32, %1164: f32, %1165: i64):
      %1166 = linalg.index 2 : index
      %1167 = arith.index_cast %1166 : index to i64
      %1168 = arith.cmpf ogt, %1163, %1164 : f32
      %1169 = arith.select %1168, %1163, %1164 : f32
      %1170 = arith.select %1168, %1167, %1165 : i64
      linalg.yield %1169, %1170 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1171 = tensor.collapse_shape %1161 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1172 = tensor.expand_shape %1171 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1173 = tensor.collapse_shape %1162 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1174 = tensor.expand_shape %1173 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1175 = func.call @aten_clamp__default(%1172) {prov.region_id = "aten_clamp__default_6", prov.dispatch_id = "aten_clamp__default_6"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1176 = tensor.empty() : tensor<1x32x1xf32>
    %1177 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1175 : tensor<1x32x1xf32>) outs(%1176 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_6", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb119(%1178: f32, %1179: f32):
      %1180 = arith.constant 1.000000e+00 : f32
      %1181 = arith.divf %1180, %1178 : f32
      linalg.yield %1181 : f32
    } -> tensor<1x32x1xf32>
    %1182 = arith.constant {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} 1.270000e+02 : f32
    %1183 = tensor.splat %1182 {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<1x32x1xf32>
    %1184 = tensor.empty() : tensor<1x32x1xf32>
    %1185 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1177, %1183 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1184 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb120(%1186: f32, %1187: f32, %1188: f32):
      %1189 = arith.mulf %1186, %1187 : f32
      linalg.yield %1189 : f32
    } -> tensor<1x32x1xf32>
    %1190 = tensor.empty() : tensor<1x32x512xf32>
    %1191 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1147, %1185 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%1190 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb121(%1192: f32, %1193: f32, %1194: f32):
      %1195 = arith.mulf %1192, %1193 : f32
      linalg.yield %1195 : f32
    } -> tensor<1x32x512xf32>
    %1196 = tensor.empty() : tensor<1x32x512xf32>
    %1197 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1191 : tensor<1x32x512xf32>) outs(%1196 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "round_6", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb122(%1198: f32, %1199: f32):
      %1200 = math.roundeven %1198 : f32
      linalg.yield %1200 : f32
    } -> tensor<1x32x512xf32>
    %1201 = tensor.empty() : tensor<1x32x512xf32>
    %1202 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1197 : tensor<1x32x512xf32>) outs(%1201 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "minmax_7", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb123(%1203: f32, %1204: f32):
      %1205 = arith.constant -1.280000e+02 : f32
      %1206 = arith.maximumf %1203, %1205 : f32
      %1207 = arith.constant 1.270000e+02 : f32
      %1208 = arith.minimumf %1206, %1207 : f32
      linalg.yield %1208 : f32
    } -> tensor<1x32x512xf32>
    %1209 = tensor.empty() : tensor<1x32x512xf32>
    %1210 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1202, %1185 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%1209 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "div_7", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb124(%1211: f32, %1212: f32, %1213: f32):
      %1214 = arith.divf %1211, %1212 : f32
      linalg.yield %1214 : f32
    } -> tensor<1x32x512xf32>
    %1215 = func.call @aten___and___Scalar_2(%93) {prov.region_id = "aten___and___Scalar_2_8", prov.dispatch_id = "aten___and___Scalar_2_8"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1216 = func.call @aten___rshift___Scalar_2(%93) {prov.region_id = "aten___rshift___Scalar_2_6", prov.dispatch_id = "aten___rshift___Scalar_2_6"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1217 = func.call @aten___and___Scalar_2(%1216) {prov.region_id = "aten___and___Scalar_2_9", prov.dispatch_id = "aten___and___Scalar_2_9"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1218 = func.call @aten___rshift___Scalar_2(%93) {prov.region_id = "aten___rshift___Scalar_2_7", prov.dispatch_id = "aten___rshift___Scalar_2_7"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1219 = func.call @aten___and___Scalar_2(%1218) {prov.region_id = "aten___and___Scalar_2_10", prov.dispatch_id = "aten___and___Scalar_2_10"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1220 = func.call @aten___rshift___Scalar_2(%93) {prov.region_id = "aten___rshift___Scalar_2_8", prov.dispatch_id = "aten___rshift___Scalar_2_8"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1221 = func.call @aten___and___Scalar_2(%1220) {prov.region_id = "aten___and___Scalar_2_11", prov.dispatch_id = "aten___and___Scalar_2_11"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1222 = func.call @aten_stack_default_2(%1215, %1217, %1219, %1221) {prov.region_id = "aten_stack_default_2_2", prov.dispatch_id = "aten_stack_default_2_2"} : (tensor<32768xi8>, tensor<32768xi8>, tensor<32768xi8>, tensor<32768xi8>) -> tensor<32768x4xi8>
    %1223 = tensor.collapse_shape %1222 [[0 : i64, 1 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<32768x4xi8> into tensor<131072xi8>
    %1224 = "tensor.extract_slice"(%1223) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 131072>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_27", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : (tensor<131072xi8>) -> tensor<131072xi8>
    %1225 = tensor.empty() : tensor<131072xf32>
    %1226 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1224 : tensor<131072xi8>) outs(%1225 : tensor<131072xf32>) attrs =  {prov.region_id = "dtype_cast_9", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb125(%1227: i8, %1228: f32):
      %1229 = arith.sitofp %1227 : i8 to f32
      linalg.yield %1229 : f32
    } -> tensor<131072xf32>
    %1230 = arith.constant {prov.region_id = "sub_6", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} 1.000000e+00 : f32
    %1231 = tensor.splat %1230 {prov.region_id = "sub_6", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<131072xf32>
    %1232 = tensor.empty() : tensor<131072xf32>
    %1233 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1226, %1231 : tensor<131072xf32>, tensor<131072xf32>) outs(%1232 : tensor<131072xf32>) attrs =  {prov.region_id = "sub_6", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb126(%1234: f32, %1235: f32, %1236: f32):
      %1237 = arith.subf %1234, %1235 : f32
      linalg.yield %1237 : f32
    } -> tensor<131072xf32>
    %1238 = func.call @aten_mul_Tensor_2(%1233) {prov.region_id = "aten_mul_Tensor_2_2", prov.dispatch_id = "aten_mul_Tensor_2_2"} : (tensor<131072xf32>) -> tensor<131072xf32>
    %1239 = tensor.expand_shape %1238 [[0 : i64, 1 : i64]] output_shape [256, 512] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<131072xf32> into tensor<256x512xf32>
    %1240 = tensor.empty() : tensor<512x256xf32>
    %1241 = linalg.transpose ins(%1239:tensor<256x512xf32>) outs(%1240:tensor<512x256xf32>) permutation = [1, 0]
    %1242 = tensor.empty() : tensor<1x32x256xf32>
    %1243 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1244 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1243 : f32) outs(%1242 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %1245 = linalg.matmul {prov.region_id = "matmul_8", prov.dispatch_id = "matmul_8", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} ins(%1210, %1241 : tensor<1x32x512xf32>, tensor<512x256xf32>) outs(%1244 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %1246 = tensor.empty() : tensor<1x32x256xf32>
    %1247 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%847, %1245 : tensor<1x32x256xf32>, tensor<1x32x256xf32>) outs(%1246 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0"} {
    ^bb127(%1248: f32, %1249: f32, %1250: f32):
      %1251 = arith.addf %1248, %1249 : f32
      linalg.yield %1251 : f32
    } -> tensor<1x32x256xf32>
    %1252 = tensor.empty() : tensor<1x32x256xf32>
    %1253 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1247 : tensor<1x32x256xf32>) outs(%1252 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb128(%1254: f32, %1255: f32):
      %1256 = arith.constant 2.000000e+00 : f32
      %1257 = math.powf %1254, %1256 : f32
      linalg.yield %1257 : f32
    } -> tensor<1x32x256xf32>
    %1258 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} 0.000000e+00 : f32
    %1259 = tensor.splat %1258 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<1x32xf32>
    %1260 = linalg.reduce ins(%1253:tensor<1x32x256xf32>) outs(%1259:tensor<1x32xf32>) dimensions = [2]
    (%1261: f32, %1262: f32) {
      %1263 = arith.addf %1261, %1262 : f32
      linalg.yield %1263 : f32
    }
    %1264 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} 2.560000e+02 : f32
    %1265 = tensor.splat %1264 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<1x32xf32>
    %1266 = tensor.empty() : tensor<1x32xf32>
    %1267 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1260, %1265 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1266 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb129(%1268: f32, %1269: f32, %1270: f32):
      %1271 = arith.divf %1268, %1269 : f32
      linalg.yield %1271 : f32
    } -> tensor<1x32xf32>
    %1272 = tensor.collapse_shape %1267 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %1273 = tensor.expand_shape %1272 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1274 = arith.constant {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} 1.000000e-05 : f32
    %1275 = tensor.splat %1274 {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<1x32x1xf32>
    %1276 = tensor.empty() : tensor<1x32x1xf32>
    %1277 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1273, %1275 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1276 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb130(%1278: f32, %1279: f32, %1280: f32):
      %1281 = arith.addf %1278, %1279 : f32
      linalg.yield %1281 : f32
    } -> tensor<1x32x1xf32>
    %1282 = tensor.empty() : tensor<1x32x1xf32>
    %1283 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1277 : tensor<1x32x1xf32>) outs(%1282 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb131(%1284: f32, %1285: f32):
      %1286 = math.rsqrt %1284 : f32
      linalg.yield %1286 : f32
    } -> tensor<1x32x1xf32>
    %1287 = tensor.empty() : tensor<1x32x256xf32>
    %1288 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1247, %1283 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1287 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb132(%1289: f32, %1290: f32, %1291: f32):
      %1292 = arith.mulf %1289, %1290 : f32
      linalg.yield %1292 : f32
    } -> tensor<1x32x256xf32>
    %1293 = tensor.empty() : tensor<1x32x256xf32>
    %1294 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %1288 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%1293 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb133(%1295: f32, %1296: f32, %1297: f32):
      %1298 = arith.mulf %1295, %1296 : f32
      linalg.yield %1298 : f32
    } -> tensor<1x32x256xf32>
    %1299 = tensor.empty() : tensor<1x32x256xf32>
    %1300 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1294 : tensor<1x32x256xf32>) outs(%1299 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_7", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb134(%1301: f32, %1302: f32):
      %1303 = math.absf %1301 : f32
      linalg.yield %1303 : f32
    } -> tensor<1x32x256xf32>
    %1304 = arith.constant {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} 0xff800000 : f32
    %1305 = arith.constant {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} 0 : i64
    %1306 = tensor.splat %1304 {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<1x32xf32>
    %1307 = tensor.splat %1305 {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<1x32xi64>
    %1308, %1309 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1300 : tensor<1x32x256xf32>) outs(%1306, %1307 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb135(%1310: f32, %1311: f32, %1312: i64):
      %1313 = linalg.index 2 : index
      %1314 = arith.index_cast %1313 : index to i64
      %1315 = arith.cmpf ogt, %1310, %1311 : f32
      %1316 = arith.select %1315, %1310, %1311 : f32
      %1317 = arith.select %1315, %1314, %1312 : i64
      linalg.yield %1316, %1317 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1318 = tensor.collapse_shape %1308 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1319 = tensor.expand_shape %1318 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1320 = tensor.collapse_shape %1309 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1321 = tensor.expand_shape %1320 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1322 = func.call @aten_clamp__default(%1319) {prov.region_id = "aten_clamp__default_7", prov.dispatch_id = "aten_clamp__default_7"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1323 = tensor.empty() : tensor<1x32x1xf32>
    %1324 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1322 : tensor<1x32x1xf32>) outs(%1323 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_7", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb136(%1325: f32, %1326: f32):
      %1327 = arith.constant 1.000000e+00 : f32
      %1328 = arith.divf %1327, %1325 : f32
      linalg.yield %1328 : f32
    } -> tensor<1x32x1xf32>
    %1329 = arith.constant {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} 1.270000e+02 : f32
    %1330 = tensor.splat %1329 {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<1x32x1xf32>
    %1331 = tensor.empty() : tensor<1x32x1xf32>
    %1332 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1324, %1330 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1331 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb137(%1333: f32, %1334: f32, %1335: f32):
      %1336 = arith.mulf %1333, %1334 : f32
      linalg.yield %1336 : f32
    } -> tensor<1x32x1xf32>
    %1337 = tensor.empty() : tensor<1x32x256xf32>
    %1338 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1294, %1332 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1337 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb138(%1339: f32, %1340: f32, %1341: f32):
      %1342 = arith.mulf %1339, %1340 : f32
      linalg.yield %1342 : f32
    } -> tensor<1x32x256xf32>
    %1343 = tensor.empty() : tensor<1x32x256xf32>
    %1344 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1338 : tensor<1x32x256xf32>) outs(%1343 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_7", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb139(%1345: f32, %1346: f32):
      %1347 = math.roundeven %1345 : f32
      linalg.yield %1347 : f32
    } -> tensor<1x32x256xf32>
    %1348 = tensor.empty() : tensor<1x32x256xf32>
    %1349 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1344 : tensor<1x32x256xf32>) outs(%1348 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_8", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb140(%1350: f32, %1351: f32):
      %1352 = arith.constant -1.280000e+02 : f32
      %1353 = arith.maximumf %1350, %1352 : f32
      %1354 = arith.constant 1.270000e+02 : f32
      %1355 = arith.minimumf %1353, %1354 : f32
      linalg.yield %1355 : f32
    } -> tensor<1x32x256xf32>
    %1356 = tensor.empty() : tensor<1x32x256xf32>
    %1357 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1349, %1332 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1356 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_8", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb141(%1358: f32, %1359: f32, %1360: f32):
      %1361 = arith.divf %1358, %1359 : f32
      linalg.yield %1361 : f32
    } -> tensor<1x32x256xf32>
    %1362 = func.call @aten___and___Scalar(%95) {prov.region_id = "aten___and___Scalar_8", prov.dispatch_id = "aten___and___Scalar_8"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %1363 = func.call @aten___rshift___Scalar(%95) {prov.region_id = "aten___rshift___Scalar_6", prov.dispatch_id = "aten___rshift___Scalar_6"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %1364 = func.call @aten___and___Scalar(%1363) {prov.region_id = "aten___and___Scalar_9", prov.dispatch_id = "aten___and___Scalar_9"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %1365 = func.call @aten___rshift___Scalar(%95) {prov.region_id = "aten___rshift___Scalar_7", prov.dispatch_id = "aten___rshift___Scalar_7"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %1366 = func.call @aten___and___Scalar(%1365) {prov.region_id = "aten___and___Scalar_10", prov.dispatch_id = "aten___and___Scalar_10"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %1367 = func.call @aten___rshift___Scalar(%95) {prov.region_id = "aten___rshift___Scalar_8", prov.dispatch_id = "aten___rshift___Scalar_8"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %1368 = func.call @aten___and___Scalar(%1367) {prov.region_id = "aten___and___Scalar_11", prov.dispatch_id = "aten___and___Scalar_11"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %1369 = func.call @aten_stack_default(%1362, %1364, %1366, %1368) {prov.region_id = "aten_stack_default_2", prov.dispatch_id = "aten_stack_default_2"} : (tensor<16384xi8>, tensor<16384xi8>, tensor<16384xi8>, tensor<16384xi8>) -> tensor<16384x4xi8>
    %1370 = tensor.collapse_shape %1369 [[0 : i64, 1 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<16384x4xi8> into tensor<65536xi8>
    %1371 = "tensor.extract_slice"(%1370) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 65536>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_28", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : (tensor<65536xi8>) -> tensor<65536xi8>
    %1372 = tensor.empty() : tensor<65536xf32>
    %1373 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1371 : tensor<65536xi8>) outs(%1372 : tensor<65536xf32>) attrs =  {prov.region_id = "dtype_cast_10", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb142(%1374: i8, %1375: f32):
      %1376 = arith.sitofp %1374 : i8 to f32
      linalg.yield %1376 : f32
    } -> tensor<65536xf32>
    %1377 = arith.constant {prov.region_id = "sub_7", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} 1.000000e+00 : f32
    %1378 = tensor.splat %1377 {prov.region_id = "sub_7", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<65536xf32>
    %1379 = tensor.empty() : tensor<65536xf32>
    %1380 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1373, %1378 : tensor<65536xf32>, tensor<65536xf32>) outs(%1379 : tensor<65536xf32>) attrs =  {prov.region_id = "sub_7", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb143(%1381: f32, %1382: f32, %1383: f32):
      %1384 = arith.subf %1381, %1382 : f32
      linalg.yield %1384 : f32
    } -> tensor<65536xf32>
    %1385 = func.call @aten_mul_Tensor(%1380) {prov.region_id = "aten_mul_Tensor_2", prov.dispatch_id = "aten_mul_Tensor_2"} : (tensor<65536xf32>) -> tensor<65536xf32>
    %1386 = tensor.expand_shape %1385 [[0 : i64, 1 : i64]] output_shape [256, 256] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<65536xf32> into tensor<256x256xf32>
    %1387 = tensor.empty() : tensor<256x256xf32>
    %1388 = linalg.transpose ins(%1386:tensor<256x256xf32>) outs(%1387:tensor<256x256xf32>) permutation = [1, 0]
    %1389 = tensor.empty() : tensor<1x32x256xf32>
    %1390 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1391 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1390 : f32) outs(%1389 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %1392 = linalg.matmul {prov.region_id = "matmul_9", prov.dispatch_id = "matmul_9", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} ins(%1357, %1388 : tensor<1x32x256xf32>, tensor<256x256xf32>) outs(%1391 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %1393 = tensor.empty() : tensor<1x32x256xf32>
    %1394 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1294 : tensor<1x32x256xf32>) outs(%1393 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_8", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb144(%1395: f32, %1396: f32):
      %1397 = math.absf %1395 : f32
      linalg.yield %1397 : f32
    } -> tensor<1x32x256xf32>
    %1398 = arith.constant {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} 0xff800000 : f32
    %1399 = arith.constant {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} 0 : i64
    %1400 = tensor.splat %1398 {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<1x32xf32>
    %1401 = tensor.splat %1399 {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<1x32xi64>
    %1402, %1403 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1394 : tensor<1x32x256xf32>) outs(%1400, %1401 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb145(%1404: f32, %1405: f32, %1406: i64):
      %1407 = linalg.index 2 : index
      %1408 = arith.index_cast %1407 : index to i64
      %1409 = arith.cmpf ogt, %1404, %1405 : f32
      %1410 = arith.select %1409, %1404, %1405 : f32
      %1411 = arith.select %1409, %1408, %1406 : i64
      linalg.yield %1410, %1411 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1412 = tensor.collapse_shape %1402 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1413 = tensor.expand_shape %1412 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1414 = tensor.collapse_shape %1403 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1415 = tensor.expand_shape %1414 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1416 = func.call @aten_clamp__default(%1413) {prov.region_id = "aten_clamp__default_8", prov.dispatch_id = "aten_clamp__default_8"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1417 = tensor.empty() : tensor<1x32x1xf32>
    %1418 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1416 : tensor<1x32x1xf32>) outs(%1417 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_8", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb146(%1419: f32, %1420: f32):
      %1421 = arith.constant 1.000000e+00 : f32
      %1422 = arith.divf %1421, %1419 : f32
      linalg.yield %1422 : f32
    } -> tensor<1x32x1xf32>
    %1423 = arith.constant {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} 1.270000e+02 : f32
    %1424 = tensor.splat %1423 {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<1x32x1xf32>
    %1425 = tensor.empty() : tensor<1x32x1xf32>
    %1426 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1418, %1424 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1425 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb147(%1427: f32, %1428: f32, %1429: f32):
      %1430 = arith.mulf %1427, %1428 : f32
      linalg.yield %1430 : f32
    } -> tensor<1x32x1xf32>
    %1431 = tensor.empty() : tensor<1x32x256xf32>
    %1432 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1294, %1426 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1431 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb148(%1433: f32, %1434: f32, %1435: f32):
      %1436 = arith.mulf %1433, %1434 : f32
      linalg.yield %1436 : f32
    } -> tensor<1x32x256xf32>
    %1437 = tensor.empty() : tensor<1x32x256xf32>
    %1438 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1432 : tensor<1x32x256xf32>) outs(%1437 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_8", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb149(%1439: f32, %1440: f32):
      %1441 = math.roundeven %1439 : f32
      linalg.yield %1441 : f32
    } -> tensor<1x32x256xf32>
    %1442 = tensor.empty() : tensor<1x32x256xf32>
    %1443 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1438 : tensor<1x32x256xf32>) outs(%1442 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_9", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb150(%1444: f32, %1445: f32):
      %1446 = arith.constant -1.280000e+02 : f32
      %1447 = arith.maximumf %1444, %1446 : f32
      %1448 = arith.constant 1.270000e+02 : f32
      %1449 = arith.minimumf %1447, %1448 : f32
      linalg.yield %1449 : f32
    } -> tensor<1x32x256xf32>
    %1450 = tensor.empty() : tensor<1x32x256xf32>
    %1451 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1443, %1426 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1450 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_9", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb151(%1452: f32, %1453: f32, %1454: f32):
      %1455 = arith.divf %1452, %1453 : f32
      linalg.yield %1455 : f32
    } -> tensor<1x32x256xf32>
    %1456 = func.call @aten___and___Scalar_1(%97) {prov.region_id = "aten___and___Scalar_1_8", prov.dispatch_id = "aten___and___Scalar_1_8"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %1457 = func.call @aten___rshift___Scalar_1(%97) {prov.region_id = "aten___rshift___Scalar_1_6", prov.dispatch_id = "aten___rshift___Scalar_1_6"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %1458 = func.call @aten___and___Scalar_1(%1457) {prov.region_id = "aten___and___Scalar_1_9", prov.dispatch_id = "aten___and___Scalar_1_9"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %1459 = func.call @aten___rshift___Scalar_1(%97) {prov.region_id = "aten___rshift___Scalar_1_7", prov.dispatch_id = "aten___rshift___Scalar_1_7"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %1460 = func.call @aten___and___Scalar_1(%1459) {prov.region_id = "aten___and___Scalar_1_10", prov.dispatch_id = "aten___and___Scalar_1_10"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %1461 = func.call @aten___rshift___Scalar_1(%97) {prov.region_id = "aten___rshift___Scalar_1_8", prov.dispatch_id = "aten___rshift___Scalar_1_8"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %1462 = func.call @aten___and___Scalar_1(%1461) {prov.region_id = "aten___and___Scalar_1_11", prov.dispatch_id = "aten___and___Scalar_1_11"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %1463 = func.call @aten_stack_default_1(%1456, %1458, %1460, %1462) {prov.region_id = "aten_stack_default_1_2", prov.dispatch_id = "aten_stack_default_1_2"} : (tensor<8192xi8>, tensor<8192xi8>, tensor<8192xi8>, tensor<8192xi8>) -> tensor<8192x4xi8>
    %1464 = tensor.collapse_shape %1463 [[0 : i64, 1 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<8192x4xi8> into tensor<32768xi8>
    %1465 = "tensor.extract_slice"(%1464) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 32768>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_29", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1466 = tensor.empty() : tensor<32768xf32>
    %1467 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1465 : tensor<32768xi8>) outs(%1466 : tensor<32768xf32>) attrs =  {prov.region_id = "dtype_cast_11", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb152(%1468: i8, %1469: f32):
      %1470 = arith.sitofp %1468 : i8 to f32
      linalg.yield %1470 : f32
    } -> tensor<32768xf32>
    %1471 = arith.constant {prov.region_id = "sub_8", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} 1.000000e+00 : f32
    %1472 = tensor.splat %1471 {prov.region_id = "sub_8", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<32768xf32>
    %1473 = tensor.empty() : tensor<32768xf32>
    %1474 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1467, %1472 : tensor<32768xf32>, tensor<32768xf32>) outs(%1473 : tensor<32768xf32>) attrs =  {prov.region_id = "sub_8", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb153(%1475: f32, %1476: f32, %1477: f32):
      %1478 = arith.subf %1475, %1476 : f32
      linalg.yield %1478 : f32
    } -> tensor<32768xf32>
    %1479 = func.call @aten_mul_Tensor_1(%1474) {prov.region_id = "aten_mul_Tensor_1_2", prov.dispatch_id = "aten_mul_Tensor_1_2"} : (tensor<32768xf32>) -> tensor<32768xf32>
    %1480 = tensor.expand_shape %1479 [[0 : i64, 1 : i64]] output_shape [128, 256] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<32768xf32> into tensor<128x256xf32>
    %1481 = tensor.empty() : tensor<256x128xf32>
    %1482 = linalg.transpose ins(%1480:tensor<128x256xf32>) outs(%1481:tensor<256x128xf32>) permutation = [1, 0]
    %1483 = tensor.empty() : tensor<1x32x128xf32>
    %1484 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1485 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1484 : f32) outs(%1483 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %1486 = linalg.matmul {prov.region_id = "matmul_10", prov.dispatch_id = "matmul_10", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} ins(%1451, %1482 : tensor<1x32x256xf32>, tensor<256x128xf32>) outs(%1485 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %1487 = tensor.empty() : tensor<1x32x256xf32>
    %1488 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1294 : tensor<1x32x256xf32>) outs(%1487 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_9", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb154(%1489: f32, %1490: f32):
      %1491 = math.absf %1489 : f32
      linalg.yield %1491 : f32
    } -> tensor<1x32x256xf32>
    %1492 = arith.constant {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} 0xff800000 : f32
    %1493 = arith.constant {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} 0 : i64
    %1494 = tensor.splat %1492 {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<1x32xf32>
    %1495 = tensor.splat %1493 {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<1x32xi64>
    %1496, %1497 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1488 : tensor<1x32x256xf32>) outs(%1494, %1495 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb155(%1498: f32, %1499: f32, %1500: i64):
      %1501 = linalg.index 2 : index
      %1502 = arith.index_cast %1501 : index to i64
      %1503 = arith.cmpf ogt, %1498, %1499 : f32
      %1504 = arith.select %1503, %1498, %1499 : f32
      %1505 = arith.select %1503, %1502, %1500 : i64
      linalg.yield %1504, %1505 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1506 = tensor.collapse_shape %1496 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1507 = tensor.expand_shape %1506 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1508 = tensor.collapse_shape %1497 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1509 = tensor.expand_shape %1508 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1510 = func.call @aten_clamp__default(%1507) {prov.region_id = "aten_clamp__default_9", prov.dispatch_id = "aten_clamp__default_9"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1511 = tensor.empty() : tensor<1x32x1xf32>
    %1512 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1510 : tensor<1x32x1xf32>) outs(%1511 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_9", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb156(%1513: f32, %1514: f32):
      %1515 = arith.constant 1.000000e+00 : f32
      %1516 = arith.divf %1515, %1513 : f32
      linalg.yield %1516 : f32
    } -> tensor<1x32x1xf32>
    %1517 = arith.constant {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} 1.270000e+02 : f32
    %1518 = tensor.splat %1517 {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<1x32x1xf32>
    %1519 = tensor.empty() : tensor<1x32x1xf32>
    %1520 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1512, %1518 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1519 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb157(%1521: f32, %1522: f32, %1523: f32):
      %1524 = arith.mulf %1521, %1522 : f32
      linalg.yield %1524 : f32
    } -> tensor<1x32x1xf32>
    %1525 = tensor.empty() : tensor<1x32x256xf32>
    %1526 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1294, %1520 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1525 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_34", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb158(%1527: f32, %1528: f32, %1529: f32):
      %1530 = arith.mulf %1527, %1528 : f32
      linalg.yield %1530 : f32
    } -> tensor<1x32x256xf32>
    %1531 = tensor.empty() : tensor<1x32x256xf32>
    %1532 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1526 : tensor<1x32x256xf32>) outs(%1531 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_9", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb159(%1533: f32, %1534: f32):
      %1535 = math.roundeven %1533 : f32
      linalg.yield %1535 : f32
    } -> tensor<1x32x256xf32>
    %1536 = tensor.empty() : tensor<1x32x256xf32>
    %1537 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1532 : tensor<1x32x256xf32>) outs(%1536 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_10", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb160(%1538: f32, %1539: f32):
      %1540 = arith.constant -1.280000e+02 : f32
      %1541 = arith.maximumf %1538, %1540 : f32
      %1542 = arith.constant 1.270000e+02 : f32
      %1543 = arith.minimumf %1541, %1542 : f32
      linalg.yield %1543 : f32
    } -> tensor<1x32x256xf32>
    %1544 = tensor.empty() : tensor<1x32x256xf32>
    %1545 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1537, %1520 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1544 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_10", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb161(%1546: f32, %1547: f32, %1548: f32):
      %1549 = arith.divf %1546, %1547 : f32
      linalg.yield %1549 : f32
    } -> tensor<1x32x256xf32>
    %1550 = func.call @aten___and___Scalar_1(%99) {prov.region_id = "aten___and___Scalar_1_12", prov.dispatch_id = "aten___and___Scalar_1_12"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %1551 = func.call @aten___rshift___Scalar_1(%99) {prov.region_id = "aten___rshift___Scalar_1_9", prov.dispatch_id = "aten___rshift___Scalar_1_9"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %1552 = func.call @aten___and___Scalar_1(%1551) {prov.region_id = "aten___and___Scalar_1_13", prov.dispatch_id = "aten___and___Scalar_1_13"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %1553 = func.call @aten___rshift___Scalar_1(%99) {prov.region_id = "aten___rshift___Scalar_1_10", prov.dispatch_id = "aten___rshift___Scalar_1_10"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %1554 = func.call @aten___and___Scalar_1(%1553) {prov.region_id = "aten___and___Scalar_1_14", prov.dispatch_id = "aten___and___Scalar_1_14"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %1555 = func.call @aten___rshift___Scalar_1(%99) {prov.region_id = "aten___rshift___Scalar_1_11", prov.dispatch_id = "aten___rshift___Scalar_1_11"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %1556 = func.call @aten___and___Scalar_1(%1555) {prov.region_id = "aten___and___Scalar_1_15", prov.dispatch_id = "aten___and___Scalar_1_15"} : (tensor<8192xi8>) -> tensor<8192xi8>
    %1557 = func.call @aten_stack_default_1(%1550, %1552, %1554, %1556) {prov.region_id = "aten_stack_default_1_3", prov.dispatch_id = "aten_stack_default_1_3"} : (tensor<8192xi8>, tensor<8192xi8>, tensor<8192xi8>, tensor<8192xi8>) -> tensor<8192x4xi8>
    %1558 = tensor.collapse_shape %1557 [[0 : i64, 1 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<8192x4xi8> into tensor<32768xi8>
    %1559 = "tensor.extract_slice"(%1558) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 32768>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_30", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %1560 = tensor.empty() : tensor<32768xf32>
    %1561 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1559 : tensor<32768xi8>) outs(%1560 : tensor<32768xf32>) attrs =  {prov.region_id = "dtype_cast_12", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb162(%1562: i8, %1563: f32):
      %1564 = arith.sitofp %1562 : i8 to f32
      linalg.yield %1564 : f32
    } -> tensor<32768xf32>
    %1565 = arith.constant {prov.region_id = "sub_9", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} 1.000000e+00 : f32
    %1566 = tensor.splat %1565 {prov.region_id = "sub_9", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<32768xf32>
    %1567 = tensor.empty() : tensor<32768xf32>
    %1568 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1561, %1566 : tensor<32768xf32>, tensor<32768xf32>) outs(%1567 : tensor<32768xf32>) attrs =  {prov.region_id = "sub_9", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb163(%1569: f32, %1570: f32, %1571: f32):
      %1572 = arith.subf %1569, %1570 : f32
      linalg.yield %1572 : f32
    } -> tensor<32768xf32>
    %1573 = func.call @aten_mul_Tensor_1(%1568) {prov.region_id = "aten_mul_Tensor_1_3", prov.dispatch_id = "aten_mul_Tensor_1_3"} : (tensor<32768xf32>) -> tensor<32768xf32>
    %1574 = tensor.expand_shape %1573 [[0 : i64, 1 : i64]] output_shape [128, 256] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<32768xf32> into tensor<128x256xf32>
    %1575 = tensor.empty() : tensor<256x128xf32>
    %1576 = linalg.transpose ins(%1574:tensor<128x256xf32>) outs(%1575:tensor<256x128xf32>) permutation = [1, 0]
    %1577 = tensor.empty() : tensor<1x32x128xf32>
    %1578 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1579 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1578 : f32) outs(%1577 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %1580 = linalg.matmul {prov.region_id = "matmul_11", prov.dispatch_id = "matmul_11", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} ins(%1545, %1576 : tensor<1x32x256xf32>, tensor<256x128xf32>) outs(%1579 : tensor<1x32x128xf32>) -> tensor<1x32x128xf32>
    %1581 = tensor.collapse_shape %1392 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %1582 = tensor.expand_shape %1581 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 32] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x32x8x32xf32>
    %1583 = tensor.empty() : tensor<1x8x32x32xf32>
    %1584 = linalg.transpose ins(%1582:tensor<1x32x8x32xf32>) outs(%1583:tensor<1x8x32x32xf32>) permutation = [0, 2, 1, 3]
    %1585 = tensor.collapse_shape %1486 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %1586 = tensor.expand_shape %1585 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 32] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<4096xf32> into tensor<1x32x4x32xf32>
    %1587 = tensor.empty() : tensor<1x4x32x32xf32>
    %1588 = linalg.transpose ins(%1586:tensor<1x32x4x32xf32>) outs(%1587:tensor<1x4x32x32xf32>) permutation = [0, 2, 1, 3]
    %1589 = tensor.collapse_shape %1580 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %1590 = tensor.expand_shape %1589 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 32] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<4096xf32> into tensor<1x32x4x32xf32>
    %1591 = tensor.empty() : tensor<1x4x32x32xf32>
    %1592 = linalg.transpose ins(%1590:tensor<1x32x4x32xf32>) outs(%1591:tensor<1x4x32x32xf32>) permutation = [0, 2, 1, 3]
    %1593 = "tensor.extract_slice"(%104) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_31", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.rotary_emb"} : (tensor<2048x32xf32>) -> tensor<32x32xf32>
    %1594 = "tensor.extract_slice"(%105) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_32", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.rotary_emb"} : (tensor<2048x32xf32>) -> tensor<32x32xf32>
    %1595 = tensor.empty() : tensor<1x32x32xf32>
    %1596 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%123 : tensor<1x32xi64>) outs(%1595 : tensor<1x32x32xf32>) attrs =  {prov.region_id = "gather_2", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb164(%1597: i64, %1598: f32):
      %1599 = arith.index_cast %1597 : i64 to index
      %1600 = linalg.index 2 : index
      %1601 = tensor.extract %1593[%1599, %1600] : tensor<32x32xf32>
      linalg.yield %1601 : f32
    } -> tensor<1x32x32xf32>
    %1602 = tensor.collapse_shape %1596 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %1603 = tensor.expand_shape %1602 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %1604 = tensor.empty() : tensor<1x32x32xf32>
    %1605 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%123 : tensor<1x32xi64>) outs(%1604 : tensor<1x32x32xf32>) attrs =  {prov.region_id = "gather_3", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb165(%1606: i64, %1607: f32):
      %1608 = arith.index_cast %1606 : i64 to index
      %1609 = linalg.index 2 : index
      %1610 = tensor.extract %1594[%1608, %1609] : tensor<32x32xf32>
      linalg.yield %1610 : f32
    } -> tensor<1x32x32xf32>
    %1611 = tensor.collapse_shape %1605 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %1612 = tensor.expand_shape %1611 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %1613 = tensor.empty() : tensor<1x8x32x32xf32>
    %1614 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1584, %1603 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%1613 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "mul_35", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb166(%1615: f32, %1616: f32, %1617: f32):
      %1618 = arith.mulf %1615, %1616 : f32
      linalg.yield %1618 : f32
    } -> tensor<1x8x32x32xf32>
    %1619 = "tensor.extract_slice"(%1584) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_33", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x8x32x32xf32>) -> tensor<1x8x32x16xf32>
    %1620 = "tensor.extract_slice"(%1584) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 8, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_34", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x8x32x32xf32>) -> tensor<1x8x32x16xf32>
    %1621 = tensor.empty() : tensor<1x8x32x16xf32>
    %1622 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1620 : tensor<1x8x32x16xf32>) outs(%1621 : tensor<1x8x32x16xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb167(%1623: f32, %1624: f32):
      %1625 = arith.negf %1623 : f32
      linalg.yield %1625 : f32
    } -> tensor<1x8x32x16xf32>
    %1626 = tensor.concat dim(3) %1622, %1619 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x8x32x16xf32>, tensor<1x8x32x16xf32>) -> tensor<1x8x32x32xf32>
    %1627 = tensor.empty() : tensor<1x8x32x32xf32>
    %1628 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1626, %1612 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%1627 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "mul_36", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb168(%1629: f32, %1630: f32, %1631: f32):
      %1632 = arith.mulf %1629, %1630 : f32
      linalg.yield %1632 : f32
    } -> tensor<1x8x32x32xf32>
    %1633 = tensor.empty() : tensor<1x8x32x32xf32>
    %1634 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1614, %1628 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%1633 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb169(%1635: f32, %1636: f32, %1637: f32):
      %1638 = arith.addf %1635, %1636 : f32
      linalg.yield %1638 : f32
    } -> tensor<1x8x32x32xf32>
    %1639 = tensor.empty() : tensor<1x4x32x32xf32>
    %1640 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1588, %1603 : tensor<1x4x32x32xf32>, tensor<1x1x32x32xf32>) outs(%1639 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb170(%1641: f32, %1642: f32, %1643: f32):
      %1644 = arith.mulf %1641, %1642 : f32
      linalg.yield %1644 : f32
    } -> tensor<1x4x32x32xf32>
    %1645 = "tensor.extract_slice"(%1588) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_35", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x16xf32>
    %1646 = "tensor.extract_slice"(%1588) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_36", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x16xf32>
    %1647 = tensor.empty() : tensor<1x4x32x16xf32>
    %1648 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1646 : tensor<1x4x32x16xf32>) outs(%1647 : tensor<1x4x32x16xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb171(%1649: f32, %1650: f32):
      %1651 = arith.negf %1649 : f32
      linalg.yield %1651 : f32
    } -> tensor<1x4x32x16xf32>
    %1652 = tensor.concat dim(3) %1648, %1645 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x16xf32>, tensor<1x4x32x16xf32>) -> tensor<1x4x32x32xf32>
    %1653 = tensor.empty() : tensor<1x4x32x32xf32>
    %1654 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1652, %1612 : tensor<1x4x32x32xf32>, tensor<1x1x32x32xf32>) outs(%1653 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "mul_38", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb172(%1655: f32, %1656: f32, %1657: f32):
      %1658 = arith.mulf %1655, %1656 : f32
      linalg.yield %1658 : f32
    } -> tensor<1x4x32x32xf32>
    %1659 = tensor.empty() : tensor<1x4x32x32xf32>
    %1660 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1640, %1654 : tensor<1x4x32x32xf32>, tensor<1x4x32x32xf32>) outs(%1659 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb173(%1661: f32, %1662: f32, %1663: f32):
      %1664 = arith.addf %1661, %1662 : f32
      linalg.yield %1664 : f32
    } -> tensor<1x4x32x32xf32>
    %1665 = "tensor.extract_slice"(%1660) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_37", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %1666 = "tensor.extract_slice"(%1665) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_38", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %1667 = tensor.collapse_shape %1666 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x32x32xf32> into tensor<4096xf32>
    %1668 = tensor.expand_shape %1667 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 32, 32] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<4096xf32> into tensor<1x4x1x32x32xf32>
    %1669 = "tensor.extract_slice"(%1668) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_39", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %1670 = "tensor.extract_slice"(%1669) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_40", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %1671 = tensor.empty() : tensor<1x4x2x32x32xf32>
    %1672 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1670 : tensor<1x4x1x32x32xf32>) outs(%1671 : tensor<1x4x2x32x32xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb174(%1673: f32, %1674: f32):
      linalg.yield %1673 : f32
    } -> tensor<1x4x2x32x32xf32>
    %1675 = tensor.collapse_shape %1672 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x2x32x32xf32> into tensor<8192xf32>
    %1676 = tensor.expand_shape %1675 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 32] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x8x32x32xf32>
    %1677 = "tensor.extract_slice"(%1592) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_41", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %1678 = "tensor.extract_slice"(%1677) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_42", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %1679 = tensor.collapse_shape %1678 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x32x32xf32> into tensor<4096xf32>
    %1680 = tensor.expand_shape %1679 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 32, 32] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<4096xf32> into tensor<1x4x1x32x32xf32>
    %1681 = "tensor.extract_slice"(%1680) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_43", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %1682 = "tensor.extract_slice"(%1681) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_44", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %1683 = tensor.empty() : tensor<1x4x2x32x32xf32>
    %1684 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1682 : tensor<1x4x1x32x32xf32>) outs(%1683 : tensor<1x4x2x32x32xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb175(%1685: f32, %1686: f32):
      linalg.yield %1685 : f32
    } -> tensor<1x4x2x32x32xf32>
    %1687 = tensor.collapse_shape %1684 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x2x32x32xf32> into tensor<8192xf32>
    %1688 = tensor.expand_shape %1687 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 32] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x8x32x32xf32>
    %1689 = tensor.empty() : tensor<1x8x32x32xf32>
    %1690 = linalg.transpose ins(%1676:tensor<1x8x32x32xf32>) outs(%1689:tensor<1x8x32x32xf32>) permutation = [0, 1, 3, 2]
    %1691 = arith.constant {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 0.000000e+00 : f32
    %1692 = tensor.splat %1691 {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32x32xf32>
    %1693 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1634, %1690 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%1692 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb176(%1694: f32, %1695: f32, %1696: f32):
      %1697 = arith.mulf %1694, %1695 : f32
      %1698 = arith.addf %1696, %1697 : f32
      linalg.yield %1698 : f32
    } -> tensor<1x8x32x32xf32>
    %1699 = arith.constant {prov.region_id = "div_11", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 5.65685415 : f32
    %1700 = tensor.splat %1699 {prov.region_id = "div_11", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32x32xf32>
    %1701 = tensor.empty() : tensor<1x8x32x32xf32>
    %1702 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1693, %1700 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%1701 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "div_11", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb177(%1703: f32, %1704: f32, %1705: f32):
      %1706 = arith.divf %1703, %1704 : f32
      linalg.yield %1706 : f32
    } -> tensor<1x8x32x32xf32>
    %1707 = "tensor.extract_slice"(%188) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_45", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    %1708 = "tensor.extract_slice"(%1707) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_46", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    %1709 = "tensor.extract_slice"(%1708) <{static_offsets = array<i64: 0, 0, 31, 0>, static_sizes = array<i64: 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<32xf32>
    %1710 = tensor.expand_shape %1709 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "slice_47", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<32xf32> into tensor<1x1x32xf32>
    %1711 = tensor.collapse_shape %1710 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x1x32xf32> into tensor<32xf32>
    %1712 = tensor.expand_shape %1711 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<32xf32> into tensor<1x1x1x32xf32>
    %1713 = tensor.empty() : tensor<1x1x32x32xf32>
    %1714 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1712 : tensor<1x1x1x32xf32>) outs(%1713 : tensor<1x1x32x32xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb178(%1715: f32, %1716: f32):
      linalg.yield %1715 : f32
    } -> tensor<1x1x32x32xf32>
    %1717 = tensor.empty() : tensor<1x8x32x32xf32>
    %1718 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1702, %1714 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%1717 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb179(%1719: f32, %1720: f32, %1721: f32):
      %1722 = arith.addf %1719, %1720 : f32
      linalg.yield %1722 : f32
    } -> tensor<1x8x32x32xf32>
    %1723 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 0xff800000 : f32
    %1724 = tensor.splat %1723 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32xf32>
    %1725 = linalg.reduce ins(%1718:tensor<1x8x32x32xf32>) outs(%1724:tensor<1x8x32xf32>) dimensions = [3]
    (%1726: f32, %1727: f32) {
      %1728 = arith.maximumf %1726, %1727 : f32
      linalg.yield %1728 : f32
    }
    %1729 = tensor.collapse_shape %1725 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %1730 = tensor.expand_shape %1729 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %1731 = tensor.empty() : tensor<1x8x32x32xf32>
    %1732 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1718, %1730 : tensor<1x8x32x32xf32>, tensor<1x8x32x1xf32>) outs(%1731 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb180(%1733: f32, %1734: f32, %1735: f32):
      %1736 = arith.subf %1733, %1734 : f32
      linalg.yield %1736 : f32
    } -> tensor<1x8x32x32xf32>
    %1737 = tensor.empty() : tensor<1x8x32x32xf32>
    %1738 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1732 : tensor<1x8x32x32xf32>) outs(%1737 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb181(%1739: f32, %1740: f32):
      %1741 = math.exp %1739 : f32
      linalg.yield %1741 : f32
    } -> tensor<1x8x32x32xf32>
    %1742 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 0.000000e+00 : f32
    %1743 = tensor.splat %1742 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32xf32>
    %1744 = linalg.reduce ins(%1738:tensor<1x8x32x32xf32>) outs(%1743:tensor<1x8x32xf32>) dimensions = [3]
    (%1745: f32, %1746: f32) {
      %1747 = arith.addf %1745, %1746 : f32
      linalg.yield %1747 : f32
    }
    %1748 = tensor.collapse_shape %1744 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %1749 = tensor.expand_shape %1748 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %1750 = tensor.empty() : tensor<1x8x32x32xf32>
    %1751 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1738, %1749 : tensor<1x8x32x32xf32>, tensor<1x8x32x1xf32>) outs(%1750 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb182(%1752: f32, %1753: f32, %1754: f32):
      %1755 = arith.divf %1752, %1753 : f32
      linalg.yield %1755 : f32
    } -> tensor<1x8x32x32xf32>
    %1756 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 0.000000e+00 : f32
    %1757 = tensor.splat %1756 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32x32xf32>
    %1758 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1751, %1688 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%1757 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb183(%1759: f32, %1760: f32, %1761: f32):
      %1762 = arith.mulf %1759, %1760 : f32
      %1763 = arith.addf %1761, %1762 : f32
      linalg.yield %1763 : f32
    } -> tensor<1x8x32x32xf32>
    %1764 = tensor.empty() : tensor<1x32x8x32xf32>
    %1765 = linalg.transpose ins(%1758:tensor<1x8x32x32xf32>) outs(%1764:tensor<1x32x8x32xf32>) permutation = [0, 2, 1, 3]
    %1766 = tensor.collapse_shape %1765 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x8x32xf32> into tensor<8192xf32>
    %1767 = tensor.expand_shape %1766 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 256] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x32x256xf32>
    %1768 = tensor.empty() : tensor<1x32x256xf32>
    %1769 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1767 : tensor<1x32x256xf32>) outs(%1768 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb184(%1770: f32, %1771: f32):
      %1772 = arith.constant 2.000000e+00 : f32
      %1773 = math.powf %1770, %1772 : f32
      linalg.yield %1773 : f32
    } -> tensor<1x32x256xf32>
    %1774 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} 0.000000e+00 : f32
    %1775 = tensor.splat %1774 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} : tensor<1x32xf32>
    %1776 = linalg.reduce ins(%1769:tensor<1x32x256xf32>) outs(%1775:tensor<1x32xf32>) dimensions = [2]
    (%1777: f32, %1778: f32) {
      %1779 = arith.addf %1777, %1778 : f32
      linalg.yield %1779 : f32
    }
    %1780 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} 2.560000e+02 : f32
    %1781 = tensor.splat %1780 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} : tensor<1x32xf32>
    %1782 = tensor.empty() : tensor<1x32xf32>
    %1783 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1776, %1781 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1782 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb185(%1784: f32, %1785: f32, %1786: f32):
      %1787 = arith.divf %1784, %1785 : f32
      linalg.yield %1787 : f32
    } -> tensor<1x32xf32>
    %1788 = tensor.collapse_shape %1783 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} : tensor<1x32xf32> into tensor<32xf32>
    %1789 = tensor.expand_shape %1788 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1790 = arith.constant {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} 1.000000e-05 : f32
    %1791 = tensor.splat %1790 {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} : tensor<1x32x1xf32>
    %1792 = tensor.empty() : tensor<1x32x1xf32>
    %1793 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1789, %1791 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1792 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb186(%1794: f32, %1795: f32, %1796: f32):
      %1797 = arith.addf %1794, %1795 : f32
      linalg.yield %1797 : f32
    } -> tensor<1x32x1xf32>
    %1798 = tensor.empty() : tensor<1x32x1xf32>
    %1799 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1793 : tensor<1x32x1xf32>) outs(%1798 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_5", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb187(%1800: f32, %1801: f32):
      %1802 = math.rsqrt %1800 : f32
      linalg.yield %1802 : f32
    } -> tensor<1x32x1xf32>
    %1803 = tensor.empty() : tensor<1x32x256xf32>
    %1804 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1767, %1799 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1803 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_39", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb188(%1805: f32, %1806: f32, %1807: f32):
      %1808 = arith.mulf %1805, %1806 : f32
      linalg.yield %1808 : f32
    } -> tensor<1x32x256xf32>
    %1809 = tensor.empty() : tensor<1x32x256xf32>
    %1810 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43, %1804 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%1809 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_40", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb189(%1811: f32, %1812: f32, %1813: f32):
      %1814 = arith.mulf %1811, %1812 : f32
      linalg.yield %1814 : f32
    } -> tensor<1x32x256xf32>
    %1815 = tensor.empty() : tensor<1x32x256xf32>
    %1816 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1810 : tensor<1x32x256xf32>) outs(%1815 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_10", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb190(%1817: f32, %1818: f32):
      %1819 = math.absf %1817 : f32
      linalg.yield %1819 : f32
    } -> tensor<1x32x256xf32>
    %1820 = arith.constant {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} 0xff800000 : f32
    %1821 = arith.constant {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} 0 : i64
    %1822 = tensor.splat %1820 {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<1x32xf32>
    %1823 = tensor.splat %1821 {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<1x32xi64>
    %1824, %1825 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1816 : tensor<1x32x256xf32>) outs(%1822, %1823 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb191(%1826: f32, %1827: f32, %1828: i64):
      %1829 = linalg.index 2 : index
      %1830 = arith.index_cast %1829 : index to i64
      %1831 = arith.cmpf ogt, %1826, %1827 : f32
      %1832 = arith.select %1831, %1826, %1827 : f32
      %1833 = arith.select %1831, %1830, %1828 : i64
      linalg.yield %1832, %1833 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1834 = tensor.collapse_shape %1824 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1835 = tensor.expand_shape %1834 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1836 = tensor.collapse_shape %1825 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1837 = tensor.expand_shape %1836 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1838 = func.call @aten_clamp__default(%1835) {prov.region_id = "aten_clamp__default_10", prov.dispatch_id = "aten_clamp__default_10"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1839 = tensor.empty() : tensor<1x32x1xf32>
    %1840 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1838 : tensor<1x32x1xf32>) outs(%1839 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_10", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb192(%1841: f32, %1842: f32):
      %1843 = arith.constant 1.000000e+00 : f32
      %1844 = arith.divf %1843, %1841 : f32
      linalg.yield %1844 : f32
    } -> tensor<1x32x1xf32>
    %1845 = arith.constant {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} 1.270000e+02 : f32
    %1846 = tensor.splat %1845 {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<1x32x1xf32>
    %1847 = tensor.empty() : tensor<1x32x1xf32>
    %1848 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1840, %1846 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1847 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb193(%1849: f32, %1850: f32, %1851: f32):
      %1852 = arith.mulf %1849, %1850 : f32
      linalg.yield %1852 : f32
    } -> tensor<1x32x1xf32>
    %1853 = tensor.empty() : tensor<1x32x256xf32>
    %1854 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1810, %1848 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1853 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_42", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb194(%1855: f32, %1856: f32, %1857: f32):
      %1858 = arith.mulf %1855, %1856 : f32
      linalg.yield %1858 : f32
    } -> tensor<1x32x256xf32>
    %1859 = tensor.empty() : tensor<1x32x256xf32>
    %1860 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1854 : tensor<1x32x256xf32>) outs(%1859 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_10", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb195(%1861: f32, %1862: f32):
      %1863 = math.roundeven %1861 : f32
      linalg.yield %1863 : f32
    } -> tensor<1x32x256xf32>
    %1864 = tensor.empty() : tensor<1x32x256xf32>
    %1865 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1860 : tensor<1x32x256xf32>) outs(%1864 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_11", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb196(%1866: f32, %1867: f32):
      %1868 = arith.constant -1.280000e+02 : f32
      %1869 = arith.maximumf %1866, %1868 : f32
      %1870 = arith.constant 1.270000e+02 : f32
      %1871 = arith.minimumf %1869, %1870 : f32
      linalg.yield %1871 : f32
    } -> tensor<1x32x256xf32>
    %1872 = tensor.empty() : tensor<1x32x256xf32>
    %1873 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1865, %1848 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1872 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_12", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb197(%1874: f32, %1875: f32, %1876: f32):
      %1877 = arith.divf %1874, %1875 : f32
      linalg.yield %1877 : f32
    } -> tensor<1x32x256xf32>
    %1878 = func.call @aten___and___Scalar(%101) {prov.region_id = "aten___and___Scalar_12", prov.dispatch_id = "aten___and___Scalar_12"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %1879 = func.call @aten___rshift___Scalar(%101) {prov.region_id = "aten___rshift___Scalar_9", prov.dispatch_id = "aten___rshift___Scalar_9"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %1880 = func.call @aten___and___Scalar(%1879) {prov.region_id = "aten___and___Scalar_13", prov.dispatch_id = "aten___and___Scalar_13"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %1881 = func.call @aten___rshift___Scalar(%101) {prov.region_id = "aten___rshift___Scalar_10", prov.dispatch_id = "aten___rshift___Scalar_10"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %1882 = func.call @aten___and___Scalar(%1881) {prov.region_id = "aten___and___Scalar_14", prov.dispatch_id = "aten___and___Scalar_14"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %1883 = func.call @aten___rshift___Scalar(%101) {prov.region_id = "aten___rshift___Scalar_11", prov.dispatch_id = "aten___rshift___Scalar_11"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %1884 = func.call @aten___and___Scalar(%1883) {prov.region_id = "aten___and___Scalar_15", prov.dispatch_id = "aten___and___Scalar_15"} : (tensor<16384xi8>) -> tensor<16384xi8>
    %1885 = func.call @aten_stack_default(%1878, %1880, %1882, %1884) {prov.region_id = "aten_stack_default_3", prov.dispatch_id = "aten_stack_default_3"} : (tensor<16384xi8>, tensor<16384xi8>, tensor<16384xi8>, tensor<16384xi8>) -> tensor<16384x4xi8>
    %1886 = tensor.collapse_shape %1885 [[0 : i64, 1 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<16384x4xi8> into tensor<65536xi8>
    %1887 = "tensor.extract_slice"(%1886) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 65536>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_48", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : (tensor<65536xi8>) -> tensor<65536xi8>
    %1888 = tensor.empty() : tensor<65536xf32>
    %1889 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1887 : tensor<65536xi8>) outs(%1888 : tensor<65536xf32>) attrs =  {prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb198(%1890: i8, %1891: f32):
      %1892 = arith.sitofp %1890 : i8 to f32
      linalg.yield %1892 : f32
    } -> tensor<65536xf32>
    %1893 = arith.constant {prov.region_id = "sub_10", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} 1.000000e+00 : f32
    %1894 = tensor.splat %1893 {prov.region_id = "sub_10", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<65536xf32>
    %1895 = tensor.empty() : tensor<65536xf32>
    %1896 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1889, %1894 : tensor<65536xf32>, tensor<65536xf32>) outs(%1895 : tensor<65536xf32>) attrs =  {prov.region_id = "sub_10", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb199(%1897: f32, %1898: f32, %1899: f32):
      %1900 = arith.subf %1897, %1898 : f32
      linalg.yield %1900 : f32
    } -> tensor<65536xf32>
    %1901 = func.call @aten_mul_Tensor(%1896) {prov.region_id = "aten_mul_Tensor_3", prov.dispatch_id = "aten_mul_Tensor_3"} : (tensor<65536xf32>) -> tensor<65536xf32>
    %1902 = tensor.expand_shape %1901 [[0 : i64, 1 : i64]] output_shape [256, 256] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<65536xf32> into tensor<256x256xf32>
    %1903 = tensor.empty() : tensor<256x256xf32>
    %1904 = linalg.transpose ins(%1902:tensor<256x256xf32>) outs(%1903:tensor<256x256xf32>) permutation = [1, 0]
    %1905 = tensor.empty() : tensor<1x32x256xf32>
    %1906 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1907 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1906 : f32) outs(%1905 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %1908 = linalg.matmul {prov.region_id = "matmul_14", prov.dispatch_id = "matmul_14", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} ins(%1873, %1904 : tensor<1x32x256xf32>, tensor<256x256xf32>) outs(%1907 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %1909 = tensor.empty() : tensor<1x32x256xf32>
    %1910 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1247, %1908 : tensor<1x32x256xf32>, tensor<1x32x256xf32>) outs(%1909 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1"} {
    ^bb200(%1911: f32, %1912: f32, %1913: f32):
      %1914 = arith.addf %1911, %1912 : f32
      linalg.yield %1914 : f32
    } -> tensor<1x32x256xf32>
    %1915 = tensor.empty() : tensor<1x32x256xf32>
    %1916 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1910 : tensor<1x32x256xf32>) outs(%1915 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb201(%1917: f32, %1918: f32):
      %1919 = arith.constant 2.000000e+00 : f32
      %1920 = math.powf %1917, %1919 : f32
      linalg.yield %1920 : f32
    } -> tensor<1x32x256xf32>
    %1921 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} 0.000000e+00 : f32
    %1922 = tensor.splat %1921 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<1x32xf32>
    %1923 = linalg.reduce ins(%1916:tensor<1x32x256xf32>) outs(%1922:tensor<1x32xf32>) dimensions = [2]
    (%1924: f32, %1925: f32) {
      %1926 = arith.addf %1924, %1925 : f32
      linalg.yield %1926 : f32
    }
    %1927 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} 2.560000e+02 : f32
    %1928 = tensor.splat %1927 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<1x32xf32>
    %1929 = tensor.empty() : tensor<1x32xf32>
    %1930 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1923, %1928 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1929 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb202(%1931: f32, %1932: f32, %1933: f32):
      %1934 = arith.divf %1931, %1932 : f32
      linalg.yield %1934 : f32
    } -> tensor<1x32xf32>
    %1935 = tensor.collapse_shape %1930 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %1936 = tensor.expand_shape %1935 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1937 = arith.constant {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} 1.000000e-05 : f32
    %1938 = tensor.splat %1937 {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<1x32x1xf32>
    %1939 = tensor.empty() : tensor<1x32x1xf32>
    %1940 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1936, %1938 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1939 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb203(%1941: f32, %1942: f32, %1943: f32):
      %1944 = arith.addf %1941, %1942 : f32
      linalg.yield %1944 : f32
    } -> tensor<1x32x1xf32>
    %1945 = tensor.empty() : tensor<1x32x1xf32>
    %1946 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1940 : tensor<1x32x1xf32>) outs(%1945 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_6", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb204(%1947: f32, %1948: f32):
      %1949 = math.rsqrt %1947 : f32
      linalg.yield %1949 : f32
    } -> tensor<1x32x1xf32>
    %1950 = tensor.empty() : tensor<1x32x256xf32>
    %1951 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1910, %1946 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1950 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb205(%1952: f32, %1953: f32, %1954: f32):
      %1955 = arith.mulf %1952, %1953 : f32
      linalg.yield %1955 : f32
    } -> tensor<1x32x256xf32>
    %1956 = tensor.empty() : tensor<1x32x256xf32>
    %1957 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46, %1951 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%1956 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_44", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb206(%1958: f32, %1959: f32, %1960: f32):
      %1961 = arith.mulf %1958, %1959 : f32
      linalg.yield %1961 : f32
    } -> tensor<1x32x256xf32>
    %1962 = tensor.empty() : tensor<1x32x256xf32>
    %1963 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1957 : tensor<1x32x256xf32>) outs(%1962 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_11", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb207(%1964: f32, %1965: f32):
      %1966 = math.absf %1964 : f32
      linalg.yield %1966 : f32
    } -> tensor<1x32x256xf32>
    %1967 = arith.constant {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} 0xff800000 : f32
    %1968 = arith.constant {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} 0 : i64
    %1969 = tensor.splat %1967 {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<1x32xf32>
    %1970 = tensor.splat %1968 {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<1x32xi64>
    %1971, %1972 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1963 : tensor<1x32x256xf32>) outs(%1969, %1970 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb208(%1973: f32, %1974: f32, %1975: i64):
      %1976 = linalg.index 2 : index
      %1977 = arith.index_cast %1976 : index to i64
      %1978 = arith.cmpf ogt, %1973, %1974 : f32
      %1979 = arith.select %1978, %1973, %1974 : f32
      %1980 = arith.select %1978, %1977, %1975 : i64
      linalg.yield %1979, %1980 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1981 = tensor.collapse_shape %1971 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1982 = tensor.expand_shape %1981 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1983 = tensor.collapse_shape %1972 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1984 = tensor.expand_shape %1983 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1985 = func.call @aten_clamp__default(%1982) {prov.region_id = "aten_clamp__default_11", prov.dispatch_id = "aten_clamp__default_11"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %1986 = tensor.empty() : tensor<1x32x1xf32>
    %1987 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1985 : tensor<1x32x1xf32>) outs(%1986 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_11", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb209(%1988: f32, %1989: f32):
      %1990 = arith.constant 1.000000e+00 : f32
      %1991 = arith.divf %1990, %1988 : f32
      linalg.yield %1991 : f32
    } -> tensor<1x32x1xf32>
    %1992 = arith.constant {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} 1.270000e+02 : f32
    %1993 = tensor.splat %1992 {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<1x32x1xf32>
    %1994 = tensor.empty() : tensor<1x32x1xf32>
    %1995 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1987, %1993 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1994 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb210(%1996: f32, %1997: f32, %1998: f32):
      %1999 = arith.mulf %1996, %1997 : f32
      linalg.yield %1999 : f32
    } -> tensor<1x32x1xf32>
    %2000 = tensor.empty() : tensor<1x32x256xf32>
    %2001 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1957, %1995 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2000 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_46", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb211(%2002: f32, %2003: f32, %2004: f32):
      %2005 = arith.mulf %2002, %2003 : f32
      linalg.yield %2005 : f32
    } -> tensor<1x32x256xf32>
    %2006 = tensor.empty() : tensor<1x32x256xf32>
    %2007 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2001 : tensor<1x32x256xf32>) outs(%2006 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_11", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb212(%2008: f32, %2009: f32):
      %2010 = math.roundeven %2008 : f32
      linalg.yield %2010 : f32
    } -> tensor<1x32x256xf32>
    %2011 = tensor.empty() : tensor<1x32x256xf32>
    %2012 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2007 : tensor<1x32x256xf32>) outs(%2011 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_12", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb213(%2013: f32, %2014: f32):
      %2015 = arith.constant -1.280000e+02 : f32
      %2016 = arith.maximumf %2013, %2015 : f32
      %2017 = arith.constant 1.270000e+02 : f32
      %2018 = arith.minimumf %2016, %2017 : f32
      linalg.yield %2018 : f32
    } -> tensor<1x32x256xf32>
    %2019 = tensor.empty() : tensor<1x32x256xf32>
    %2020 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2012, %1995 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2019 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_13", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb214(%2021: f32, %2022: f32, %2023: f32):
      %2024 = arith.divf %2021, %2022 : f32
      linalg.yield %2024 : f32
    } -> tensor<1x32x256xf32>
    %2025 = func.call @aten___and___Scalar_2(%106) {prov.region_id = "aten___and___Scalar_2_12", prov.dispatch_id = "aten___and___Scalar_2_12"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2026 = func.call @aten___rshift___Scalar_2(%106) {prov.region_id = "aten___rshift___Scalar_2_9", prov.dispatch_id = "aten___rshift___Scalar_2_9"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2027 = func.call @aten___and___Scalar_2(%2026) {prov.region_id = "aten___and___Scalar_2_13", prov.dispatch_id = "aten___and___Scalar_2_13"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2028 = func.call @aten___rshift___Scalar_2(%106) {prov.region_id = "aten___rshift___Scalar_2_10", prov.dispatch_id = "aten___rshift___Scalar_2_10"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2029 = func.call @aten___and___Scalar_2(%2028) {prov.region_id = "aten___and___Scalar_2_14", prov.dispatch_id = "aten___and___Scalar_2_14"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2030 = func.call @aten___rshift___Scalar_2(%106) {prov.region_id = "aten___rshift___Scalar_2_11", prov.dispatch_id = "aten___rshift___Scalar_2_11"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2031 = func.call @aten___and___Scalar_2(%2030) {prov.region_id = "aten___and___Scalar_2_15", prov.dispatch_id = "aten___and___Scalar_2_15"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2032 = func.call @aten_stack_default_2(%2025, %2027, %2029, %2031) {prov.region_id = "aten_stack_default_2_3", prov.dispatch_id = "aten_stack_default_2_3"} : (tensor<32768xi8>, tensor<32768xi8>, tensor<32768xi8>, tensor<32768xi8>) -> tensor<32768x4xi8>
    %2033 = tensor.collapse_shape %2032 [[0 : i64, 1 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<32768x4xi8> into tensor<131072xi8>
    %2034 = "tensor.extract_slice"(%2033) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 131072>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_49", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : (tensor<131072xi8>) -> tensor<131072xi8>
    %2035 = tensor.empty() : tensor<131072xf32>
    %2036 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2034 : tensor<131072xi8>) outs(%2035 : tensor<131072xf32>) attrs =  {prov.region_id = "dtype_cast_14", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb215(%2037: i8, %2038: f32):
      %2039 = arith.sitofp %2037 : i8 to f32
      linalg.yield %2039 : f32
    } -> tensor<131072xf32>
    %2040 = arith.constant {prov.region_id = "sub_11", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} 1.000000e+00 : f32
    %2041 = tensor.splat %2040 {prov.region_id = "sub_11", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<131072xf32>
    %2042 = tensor.empty() : tensor<131072xf32>
    %2043 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2036, %2041 : tensor<131072xf32>, tensor<131072xf32>) outs(%2042 : tensor<131072xf32>) attrs =  {prov.region_id = "sub_11", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb216(%2044: f32, %2045: f32, %2046: f32):
      %2047 = arith.subf %2044, %2045 : f32
      linalg.yield %2047 : f32
    } -> tensor<131072xf32>
    %2048 = func.call @aten_mul_Tensor_2(%2043) {prov.region_id = "aten_mul_Tensor_2_3", prov.dispatch_id = "aten_mul_Tensor_2_3"} : (tensor<131072xf32>) -> tensor<131072xf32>
    %2049 = tensor.expand_shape %2048 [[0 : i64, 1 : i64]] output_shape [512, 256] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<131072xf32> into tensor<512x256xf32>
    %2050 = tensor.empty() : tensor<256x512xf32>
    %2051 = linalg.transpose ins(%2049:tensor<512x256xf32>) outs(%2050:tensor<256x512xf32>) permutation = [1, 0]
    %2052 = tensor.empty() : tensor<1x32x512xf32>
    %2053 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %2054 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%2053 : f32) outs(%2052 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %2055 = linalg.matmul {prov.region_id = "matmul_15", prov.dispatch_id = "matmul_15", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} ins(%2020, %2051 : tensor<1x32x256xf32>, tensor<256x512xf32>) outs(%2054 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %2056 = tensor.empty() : tensor<1x32x512xf32>
    %2057 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2055 : tensor<1x32x512xf32>) outs(%2056 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "minmax_13", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp"} {
    ^bb217(%2058: f32, %2059: f32):
      %2060 = arith.constant 0.000000e+00 : f32
      %2061 = arith.maximumf %2058, %2060 : f32
      linalg.yield %2061 : f32
    } -> tensor<1x32x512xf32>
    %2062 = tensor.empty() : tensor<1x32x512xf32>
    %2063 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2057 : tensor<1x32x512xf32>) outs(%2062 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp"} {
    ^bb218(%2064: f32, %2065: f32):
      %2066 = arith.constant 2.000000e+00 : f32
      %2067 = math.powf %2064, %2066 : f32
      linalg.yield %2067 : f32
    } -> tensor<1x32x512xf32>
    %2068 = tensor.empty() : tensor<1x32x256xf32>
    %2069 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1957 : tensor<1x32x256xf32>) outs(%2068 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_12", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb219(%2070: f32, %2071: f32):
      %2072 = math.absf %2070 : f32
      linalg.yield %2072 : f32
    } -> tensor<1x32x256xf32>
    %2073 = arith.constant {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} 0xff800000 : f32
    %2074 = arith.constant {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} 0 : i64
    %2075 = tensor.splat %2073 {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<1x32xf32>
    %2076 = tensor.splat %2074 {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<1x32xi64>
    %2077, %2078 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2069 : tensor<1x32x256xf32>) outs(%2075, %2076 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb220(%2079: f32, %2080: f32, %2081: i64):
      %2082 = linalg.index 2 : index
      %2083 = arith.index_cast %2082 : index to i64
      %2084 = arith.cmpf ogt, %2079, %2080 : f32
      %2085 = arith.select %2084, %2079, %2080 : f32
      %2086 = arith.select %2084, %2083, %2081 : i64
      linalg.yield %2085, %2086 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %2087 = tensor.collapse_shape %2077 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %2088 = tensor.expand_shape %2087 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2089 = tensor.collapse_shape %2078 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %2090 = tensor.expand_shape %2089 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %2091 = func.call @aten_clamp__default(%2088) {prov.region_id = "aten_clamp__default_12", prov.dispatch_id = "aten_clamp__default_12"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %2092 = tensor.empty() : tensor<1x32x1xf32>
    %2093 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2091 : tensor<1x32x1xf32>) outs(%2092 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_12", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb221(%2094: f32, %2095: f32):
      %2096 = arith.constant 1.000000e+00 : f32
      %2097 = arith.divf %2096, %2094 : f32
      linalg.yield %2097 : f32
    } -> tensor<1x32x1xf32>
    %2098 = arith.constant {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} 1.270000e+02 : f32
    %2099 = tensor.splat %2098 {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<1x32x1xf32>
    %2100 = tensor.empty() : tensor<1x32x1xf32>
    %2101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2093, %2099 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2100 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb222(%2102: f32, %2103: f32, %2104: f32):
      %2105 = arith.mulf %2102, %2103 : f32
      linalg.yield %2105 : f32
    } -> tensor<1x32x1xf32>
    %2106 = tensor.empty() : tensor<1x32x256xf32>
    %2107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1957, %2101 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2106 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_48", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb223(%2108: f32, %2109: f32, %2110: f32):
      %2111 = arith.mulf %2108, %2109 : f32
      linalg.yield %2111 : f32
    } -> tensor<1x32x256xf32>
    %2112 = tensor.empty() : tensor<1x32x256xf32>
    %2113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2107 : tensor<1x32x256xf32>) outs(%2112 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_12", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb224(%2114: f32, %2115: f32):
      %2116 = math.roundeven %2114 : f32
      linalg.yield %2116 : f32
    } -> tensor<1x32x256xf32>
    %2117 = tensor.empty() : tensor<1x32x256xf32>
    %2118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2113 : tensor<1x32x256xf32>) outs(%2117 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_14", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb225(%2119: f32, %2120: f32):
      %2121 = arith.constant -1.280000e+02 : f32
      %2122 = arith.maximumf %2119, %2121 : f32
      %2123 = arith.constant 1.270000e+02 : f32
      %2124 = arith.minimumf %2122, %2123 : f32
      linalg.yield %2124 : f32
    } -> tensor<1x32x256xf32>
    %2125 = tensor.empty() : tensor<1x32x256xf32>
    %2126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2118, %2101 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2125 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_14", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb226(%2127: f32, %2128: f32, %2129: f32):
      %2130 = arith.divf %2127, %2128 : f32
      linalg.yield %2130 : f32
    } -> tensor<1x32x256xf32>
    %2131 = func.call @aten___and___Scalar_2(%108) {prov.region_id = "aten___and___Scalar_2_16", prov.dispatch_id = "aten___and___Scalar_2_16"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2132 = func.call @aten___rshift___Scalar_2(%108) {prov.region_id = "aten___rshift___Scalar_2_12", prov.dispatch_id = "aten___rshift___Scalar_2_12"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2133 = func.call @aten___and___Scalar_2(%2132) {prov.region_id = "aten___and___Scalar_2_17", prov.dispatch_id = "aten___and___Scalar_2_17"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2134 = func.call @aten___rshift___Scalar_2(%108) {prov.region_id = "aten___rshift___Scalar_2_13", prov.dispatch_id = "aten___rshift___Scalar_2_13"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2135 = func.call @aten___and___Scalar_2(%2134) {prov.region_id = "aten___and___Scalar_2_18", prov.dispatch_id = "aten___and___Scalar_2_18"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2136 = func.call @aten___rshift___Scalar_2(%108) {prov.region_id = "aten___rshift___Scalar_2_14", prov.dispatch_id = "aten___rshift___Scalar_2_14"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2137 = func.call @aten___and___Scalar_2(%2136) {prov.region_id = "aten___and___Scalar_2_19", prov.dispatch_id = "aten___and___Scalar_2_19"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2138 = func.call @aten_stack_default_2(%2131, %2133, %2135, %2137) {prov.region_id = "aten_stack_default_2_4", prov.dispatch_id = "aten_stack_default_2_4"} : (tensor<32768xi8>, tensor<32768xi8>, tensor<32768xi8>, tensor<32768xi8>) -> tensor<32768x4xi8>
    %2139 = tensor.collapse_shape %2138 [[0 : i64, 1 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<32768x4xi8> into tensor<131072xi8>
    %2140 = "tensor.extract_slice"(%2139) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 131072>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_50", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : (tensor<131072xi8>) -> tensor<131072xi8>
    %2141 = tensor.empty() : tensor<131072xf32>
    %2142 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2140 : tensor<131072xi8>) outs(%2141 : tensor<131072xf32>) attrs =  {prov.region_id = "dtype_cast_15", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb227(%2143: i8, %2144: f32):
      %2145 = arith.sitofp %2143 : i8 to f32
      linalg.yield %2145 : f32
    } -> tensor<131072xf32>
    %2146 = arith.constant {prov.region_id = "sub_12", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} 1.000000e+00 : f32
    %2147 = tensor.splat %2146 {prov.region_id = "sub_12", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<131072xf32>
    %2148 = tensor.empty() : tensor<131072xf32>
    %2149 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2142, %2147 : tensor<131072xf32>, tensor<131072xf32>) outs(%2148 : tensor<131072xf32>) attrs =  {prov.region_id = "sub_12", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb228(%2150: f32, %2151: f32, %2152: f32):
      %2153 = arith.subf %2150, %2151 : f32
      linalg.yield %2153 : f32
    } -> tensor<131072xf32>
    %2154 = func.call @aten_mul_Tensor_2(%2149) {prov.region_id = "aten_mul_Tensor_2_4", prov.dispatch_id = "aten_mul_Tensor_2_4"} : (tensor<131072xf32>) -> tensor<131072xf32>
    %2155 = tensor.expand_shape %2154 [[0 : i64, 1 : i64]] output_shape [512, 256] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<131072xf32> into tensor<512x256xf32>
    %2156 = tensor.empty() : tensor<256x512xf32>
    %2157 = linalg.transpose ins(%2155:tensor<512x256xf32>) outs(%2156:tensor<256x512xf32>) permutation = [1, 0]
    %2158 = tensor.empty() : tensor<1x32x512xf32>
    %2159 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %2160 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%2159 : f32) outs(%2158 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %2161 = linalg.matmul {prov.region_id = "matmul_16", prov.dispatch_id = "matmul_16", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} ins(%2126, %2157 : tensor<1x32x256xf32>, tensor<256x512xf32>) outs(%2160 : tensor<1x32x512xf32>) -> tensor<1x32x512xf32>
    %2162 = tensor.empty() : tensor<1x32x512xf32>
    %2163 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2063, %2161 : tensor<1x32x512xf32>, tensor<1x32x512xf32>) outs(%2162 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_49", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp"} {
    ^bb229(%2164: f32, %2165: f32, %2166: f32):
      %2167 = arith.mulf %2164, %2165 : f32
      linalg.yield %2167 : f32
    } -> tensor<1x32x512xf32>
    %2168 = tensor.empty() : tensor<1x32x512xf32>
    %2169 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2163 : tensor<1x32x512xf32>) outs(%2168 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "pow_9", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb230(%2170: f32, %2171: f32):
      %2172 = arith.constant 2.000000e+00 : f32
      %2173 = math.powf %2170, %2172 : f32
      linalg.yield %2173 : f32
    } -> tensor<1x32x512xf32>
    %2174 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} 0.000000e+00 : f32
    %2175 = tensor.splat %2174 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} : tensor<1x32xf32>
    %2176 = linalg.reduce ins(%2169:tensor<1x32x512xf32>) outs(%2175:tensor<1x32xf32>) dimensions = [2]
    (%2177: f32, %2178: f32) {
      %2179 = arith.addf %2177, %2178 : f32
      linalg.yield %2179 : f32
    }
    %2180 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} 5.120000e+02 : f32
    %2181 = tensor.splat %2180 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} : tensor<1x32xf32>
    %2182 = tensor.empty() : tensor<1x32xf32>
    %2183 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2176, %2181 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%2182 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb231(%2184: f32, %2185: f32, %2186: f32):
      %2187 = arith.divf %2184, %2185 : f32
      linalg.yield %2187 : f32
    } -> tensor<1x32xf32>
    %2188 = tensor.collapse_shape %2183 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} : tensor<1x32xf32> into tensor<32xf32>
    %2189 = tensor.expand_shape %2188 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2190 = arith.constant {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} 1.000000e-05 : f32
    %2191 = tensor.splat %2190 {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} : tensor<1x32x1xf32>
    %2192 = tensor.empty() : tensor<1x32x1xf32>
    %2193 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2189, %2191 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2192 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb232(%2194: f32, %2195: f32, %2196: f32):
      %2197 = arith.addf %2194, %2195 : f32
      linalg.yield %2197 : f32
    } -> tensor<1x32x1xf32>
    %2198 = tensor.empty() : tensor<1x32x1xf32>
    %2199 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2193 : tensor<1x32x1xf32>) outs(%2198 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_7", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb233(%2200: f32, %2201: f32):
      %2202 = math.rsqrt %2200 : f32
      linalg.yield %2202 : f32
    } -> tensor<1x32x1xf32>
    %2203 = tensor.empty() : tensor<1x32x512xf32>
    %2204 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2163, %2199 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%2203 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_50", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb234(%2205: f32, %2206: f32, %2207: f32):
      %2208 = arith.mulf %2205, %2206 : f32
      linalg.yield %2208 : f32
    } -> tensor<1x32x512xf32>
    %2209 = tensor.empty() : tensor<1x32x512xf32>
    %2210 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %2204 : tensor<512xf32>, tensor<1x32x512xf32>) outs(%2209 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_51", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb235(%2211: f32, %2212: f32, %2213: f32):
      %2214 = arith.mulf %2211, %2212 : f32
      linalg.yield %2214 : f32
    } -> tensor<1x32x512xf32>
    %2215 = tensor.empty() : tensor<1x32x512xf32>
    %2216 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2210 : tensor<1x32x512xf32>) outs(%2215 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "abs_13", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb236(%2217: f32, %2218: f32):
      %2219 = math.absf %2217 : f32
      linalg.yield %2219 : f32
    } -> tensor<1x32x512xf32>
    %2220 = arith.constant {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} 0xff800000 : f32
    %2221 = arith.constant {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} 0 : i64
    %2222 = tensor.splat %2220 {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<1x32xf32>
    %2223 = tensor.splat %2221 {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<1x32xi64>
    %2224, %2225 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2216 : tensor<1x32x512xf32>) outs(%2222, %2223 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb237(%2226: f32, %2227: f32, %2228: i64):
      %2229 = linalg.index 2 : index
      %2230 = arith.index_cast %2229 : index to i64
      %2231 = arith.cmpf ogt, %2226, %2227 : f32
      %2232 = arith.select %2231, %2226, %2227 : f32
      %2233 = arith.select %2231, %2230, %2228 : i64
      linalg.yield %2232, %2233 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %2234 = tensor.collapse_shape %2224 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %2235 = tensor.expand_shape %2234 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2236 = tensor.collapse_shape %2225 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %2237 = tensor.expand_shape %2236 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %2238 = func.call @aten_clamp__default(%2235) {prov.region_id = "aten_clamp__default_13", prov.dispatch_id = "aten_clamp__default_13"} : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
    %2239 = tensor.empty() : tensor<1x32x1xf32>
    %2240 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2238 : tensor<1x32x1xf32>) outs(%2239 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_13", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb238(%2241: f32, %2242: f32):
      %2243 = arith.constant 1.000000e+00 : f32
      %2244 = arith.divf %2243, %2241 : f32
      linalg.yield %2244 : f32
    } -> tensor<1x32x1xf32>
    %2245 = arith.constant {prov.region_id = "mul_52", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} 1.270000e+02 : f32
    %2246 = tensor.splat %2245 {prov.region_id = "mul_52", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<1x32x1xf32>
    %2247 = tensor.empty() : tensor<1x32x1xf32>
    %2248 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2240, %2246 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2247 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_52", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb239(%2249: f32, %2250: f32, %2251: f32):
      %2252 = arith.mulf %2249, %2250 : f32
      linalg.yield %2252 : f32
    } -> tensor<1x32x1xf32>
    %2253 = tensor.empty() : tensor<1x32x512xf32>
    %2254 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2210, %2248 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%2253 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_53", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb240(%2255: f32, %2256: f32, %2257: f32):
      %2258 = arith.mulf %2255, %2256 : f32
      linalg.yield %2258 : f32
    } -> tensor<1x32x512xf32>
    %2259 = tensor.empty() : tensor<1x32x512xf32>
    %2260 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2254 : tensor<1x32x512xf32>) outs(%2259 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "round_13", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb241(%2261: f32, %2262: f32):
      %2263 = math.roundeven %2261 : f32
      linalg.yield %2263 : f32
    } -> tensor<1x32x512xf32>
    %2264 = tensor.empty() : tensor<1x32x512xf32>
    %2265 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2260 : tensor<1x32x512xf32>) outs(%2264 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "minmax_15", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb242(%2266: f32, %2267: f32):
      %2268 = arith.constant -1.280000e+02 : f32
      %2269 = arith.maximumf %2266, %2268 : f32
      %2270 = arith.constant 1.270000e+02 : f32
      %2271 = arith.minimumf %2269, %2270 : f32
      linalg.yield %2271 : f32
    } -> tensor<1x32x512xf32>
    %2272 = tensor.empty() : tensor<1x32x512xf32>
    %2273 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2265, %2248 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%2272 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "div_15", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb243(%2274: f32, %2275: f32, %2276: f32):
      %2277 = arith.divf %2274, %2275 : f32
      linalg.yield %2277 : f32
    } -> tensor<1x32x512xf32>
    %2278 = func.call @aten___and___Scalar_2(%110) {prov.region_id = "aten___and___Scalar_2_20", prov.dispatch_id = "aten___and___Scalar_2_20"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2279 = func.call @aten___rshift___Scalar_2(%110) {prov.region_id = "aten___rshift___Scalar_2_15", prov.dispatch_id = "aten___rshift___Scalar_2_15"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2280 = func.call @aten___and___Scalar_2(%2279) {prov.region_id = "aten___and___Scalar_2_21", prov.dispatch_id = "aten___and___Scalar_2_21"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2281 = func.call @aten___rshift___Scalar_2(%110) {prov.region_id = "aten___rshift___Scalar_2_16", prov.dispatch_id = "aten___rshift___Scalar_2_16"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2282 = func.call @aten___and___Scalar_2(%2281) {prov.region_id = "aten___and___Scalar_2_22", prov.dispatch_id = "aten___and___Scalar_2_22"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2283 = func.call @aten___rshift___Scalar_2(%110) {prov.region_id = "aten___rshift___Scalar_2_17", prov.dispatch_id = "aten___rshift___Scalar_2_17"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2284 = func.call @aten___and___Scalar_2(%2283) {prov.region_id = "aten___and___Scalar_2_23", prov.dispatch_id = "aten___and___Scalar_2_23"} : (tensor<32768xi8>) -> tensor<32768xi8>
    %2285 = func.call @aten_stack_default_2(%2278, %2280, %2282, %2284) {prov.region_id = "aten_stack_default_2_5", prov.dispatch_id = "aten_stack_default_2_5"} : (tensor<32768xi8>, tensor<32768xi8>, tensor<32768xi8>, tensor<32768xi8>) -> tensor<32768x4xi8>
    %2286 = tensor.collapse_shape %2285 [[0 : i64, 1 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<32768x4xi8> into tensor<131072xi8>
    %2287 = "tensor.extract_slice"(%2286) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 131072>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_51", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "uint8", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : (tensor<131072xi8>) -> tensor<131072xi8>
    %2288 = tensor.empty() : tensor<131072xf32>
    %2289 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2287 : tensor<131072xi8>) outs(%2288 : tensor<131072xf32>) attrs =  {prov.region_id = "dtype_cast_16", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb244(%2290: i8, %2291: f32):
      %2292 = arith.sitofp %2290 : i8 to f32
      linalg.yield %2292 : f32
    } -> tensor<131072xf32>
    %2293 = arith.constant {prov.region_id = "sub_13", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} 1.000000e+00 : f32
    %2294 = tensor.splat %2293 {prov.region_id = "sub_13", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<131072xf32>
    %2295 = tensor.empty() : tensor<131072xf32>
    %2296 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2289, %2294 : tensor<131072xf32>, tensor<131072xf32>) outs(%2295 : tensor<131072xf32>) attrs =  {prov.region_id = "sub_13", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb245(%2297: f32, %2298: f32, %2299: f32):
      %2300 = arith.subf %2297, %2298 : f32
      linalg.yield %2300 : f32
    } -> tensor<131072xf32>
    %2301 = func.call @aten_mul_Tensor_2(%2296) {prov.region_id = "aten_mul_Tensor_2_5", prov.dispatch_id = "aten_mul_Tensor_2_5"} : (tensor<131072xf32>) -> tensor<131072xf32>
    %2302 = tensor.expand_shape %2301 [[0 : i64, 1 : i64]] output_shape [256, 512] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<131072xf32> into tensor<256x512xf32>
    %2303 = tensor.empty() : tensor<512x256xf32>
    %2304 = linalg.transpose ins(%2302:tensor<256x512xf32>) outs(%2303:tensor<512x256xf32>) permutation = [1, 0]
    %2305 = tensor.empty() : tensor<1x32x256xf32>
    %2306 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %2307 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%2306 : f32) outs(%2305 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %2308 = linalg.matmul {prov.region_id = "matmul_17", prov.dispatch_id = "matmul_17", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} ins(%2273, %2304 : tensor<1x32x512xf32>, tensor<512x256xf32>) outs(%2307 : tensor<1x32x256xf32>) -> tensor<1x32x256xf32>
    %2309 = tensor.empty() : tensor<1x32x256xf32>
    %2310 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1910, %2308 : tensor<1x32x256xf32>, tensor<1x32x256xf32>) outs(%2309 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1"} {
    ^bb246(%2311: f32, %2312: f32, %2313: f32):
      %2314 = arith.addf %2311, %2312 : f32
      linalg.yield %2314 : f32
    } -> tensor<1x32x256xf32>
    %2315 = tensor.empty() : tensor<1x32x256xf32>
    %2316 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2310 : tensor<1x32x256xf32>) outs(%2315 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_10", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb247(%2317: f32, %2318: f32):
      %2319 = arith.constant 2.000000e+00 : f32
      %2320 = math.powf %2317, %2319 : f32
      linalg.yield %2320 : f32
    } -> tensor<1x32x256xf32>
    %2321 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} 0.000000e+00 : f32
    %2322 = tensor.splat %2321 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<1x32xf32>
    %2323 = linalg.reduce ins(%2316:tensor<1x32x256xf32>) outs(%2322:tensor<1x32xf32>) dimensions = [2]
    (%2324: f32, %2325: f32) {
      %2326 = arith.addf %2324, %2325 : f32
      linalg.yield %2326 : f32
    }
    %2327 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} 2.560000e+02 : f32
    %2328 = tensor.splat %2327 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<1x32xf32>
    %2329 = tensor.empty() : tensor<1x32xf32>
    %2330 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2323, %2328 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%2329 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb248(%2331: f32, %2332: f32, %2333: f32):
      %2334 = arith.divf %2331, %2332 : f32
      linalg.yield %2334 : f32
    } -> tensor<1x32xf32>
    %2335 = tensor.collapse_shape %2330 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<1x32xf32> into tensor<32xf32>
    %2336 = tensor.expand_shape %2335 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2337 = arith.constant {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} 1.000000e-05 : f32
    %2338 = tensor.splat %2337 {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<1x32x1xf32>
    %2339 = tensor.empty() : tensor<1x32x1xf32>
    %2340 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2336, %2338 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2339 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb249(%2341: f32, %2342: f32, %2343: f32):
      %2344 = arith.addf %2341, %2342 : f32
      linalg.yield %2344 : f32
    } -> tensor<1x32x1xf32>
    %2345 = tensor.empty() : tensor<1x32x1xf32>
    %2346 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2340 : tensor<1x32x1xf32>) outs(%2345 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_8", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb250(%2347: f32, %2348: f32):
      %2349 = math.rsqrt %2347 : f32
      linalg.yield %2349 : f32
    } -> tensor<1x32x1xf32>
    %2350 = tensor.empty() : tensor<1x32x256xf32>
    %2351 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2310, %2346 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2350 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_54", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb251(%2352: f32, %2353: f32, %2354: f32):
      %2355 = arith.mulf %2352, %2353 : f32
      linalg.yield %2355 : f32
    } -> tensor<1x32x256xf32>
    %2356 = tensor.empty() : tensor<1x32x256xf32>
    %2357 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %2351 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%2356 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_55", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb252(%2358: f32, %2359: f32, %2360: f32):
      %2361 = arith.mulf %2358, %2359 : f32
      linalg.yield %2361 : f32
    } -> tensor<1x32x256xf32>
    %2362 = tensor.empty() : tensor<256x1024xf32>
    %2363 = linalg.transpose ins(%48:tensor<1024x256xf32>) outs(%2362:tensor<256x1024xf32>) permutation = [1, 0]
    %2364 = tensor.empty() : tensor<1x32x1024xf32>
    %2365 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %2366 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%2365 : f32) outs(%2364 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
    %2367 = linalg.matmul {prov.region_id = "matmul_18", prov.dispatch_id = "matmul_18", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.lm_head"} ins(%2357, %2363 : tensor<1x32x256xf32>, tensor<256x1024xf32>) outs(%2366 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
    func.return %2367 : tensor<1x32x1024xf32>
  }
}
