builtin.module attributes {prov.weights_file = "/path/to/model2MLIR/workloads/bitvla/bitvla.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<128x3x16x16xf32>, %1: tensor<128xf32>, %2: tensor<196x128xf32>, %3: tensor<128x128xf32>, %4: tensor<128xf32>, %5: tensor<128x128xf32>, %6: tensor<128xf32>, %7: tensor<128x128xf32>, %8: tensor<128xf32>, %9: tensor<128x128xf32>, %10: tensor<128xf32>, %11: tensor<128xf32>, %12: tensor<128xf32>, %13: tensor<256x128xf32>, %14: tensor<256xf32>, %15: tensor<128x256xf32>, %16: tensor<128xf32>, %17: tensor<128xf32>, %18: tensor<128xf32>, %19: tensor<128x128xf32>, %20: tensor<128xf32>, %21: tensor<128x128xf32>, %22: tensor<128xf32>, %23: tensor<128x128xf32>, %24: tensor<128xf32>, %25: tensor<128x128xf32>, %26: tensor<128xf32>, %27: tensor<128xf32>, %28: tensor<128xf32>, %29: tensor<256x128xf32>, %30: tensor<256xf32>, %31: tensor<128x256xf32>, %32: tensor<128xf32>, %33: tensor<128xf32>, %34: tensor<128xf32>, %35: tensor<128xf32>, %36: tensor<128xf32>, %37: tensor<1x1x128xf32>, %38: tensor<384x128xf32>, %39: tensor<384xf32>, %40: tensor<128x128xf32>, %41: tensor<128xf32>, %42: tensor<128xf32>, %43: tensor<128xf32>, %44: tensor<256x128xf32>, %45: tensor<256xf32>, %46: tensor<128x256xf32>, %47: tensor<128xf32>, %48: tensor<256x128xf32>, %49: tensor<256xf32>, %50: tensor<256x256xf32>, %51: tensor<256xf32>, %52: tensor<1024x256xf32>, %53: tensor<256x256xf32>, %54: tensor<128x256xf32>, %55: tensor<128x256xf32>, %56: tensor<256x256xf32>, %57: tensor<256xf32>, %58: tensor<512x256xf32>, %59: tensor<512x256xf32>, %60: tensor<256x512xf32>, %61: tensor<512xf32>, %62: tensor<256xf32>, %63: tensor<256xf32>, %64: tensor<256x256xf32>, %65: tensor<128x256xf32>, %66: tensor<128x256xf32>, %67: tensor<256x256xf32>, %68: tensor<256xf32>, %69: tensor<512x256xf32>, %70: tensor<512x256xf32>, %71: tensor<256x512xf32>, %72: tensor<512xf32>, %73: tensor<256xf32>, %74: tensor<256xf32>, %75: tensor<256xf32>, %76: tensor<1024x256xf32>, %77: tensor<1x196xi64>, %78: tensor<16xf32>, %79: tensor<2048x32xf32>, %80: tensor<2048x32xf32>, %81: tensor<16xf32>, %82: tensor<2048x32xf32>, %83: tensor<2048x32xf32>, %84: tensor<1x32x256xf32>, %85: tensor<1x32xi64>) -> tensor<1x32x1024xf32> {
    %86 = tensor.empty() : tensor<32xi64>
    %87 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%86 : tensor<32xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb0(%88: i64):
      %89 = linalg.index 0 : index
      %90 = arith.index_cast %89 : index to i64
      %91 = arith.constant 1 : i64
      %92 = arith.muli %90, %91 : i64
      %93 = arith.constant 0 : i64
      %94 = arith.addi %93, %92 : i64
      linalg.yield %94 : i64
    } -> tensor<32xi64>
    %95 = tensor.expand_shape %87 [[0 : i64, 1 : i64]] output_shape [1, 32] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<32xi64> into tensor<1x32xi64>
    %96 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} -3.40282347e+38 : f32
    %97 = tensor.splat %96 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<32x32xf32>
    %98 = tensor.empty() : tensor<32xi64>
    %99 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%98 : tensor<32xi64>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb1(%100: i64):
      %101 = linalg.index 0 : index
      %102 = arith.index_cast %101 : index to i64
      %103 = arith.constant 1 : i64
      %104 = arith.muli %102, %103 : i64
      %105 = arith.constant 0 : i64
      %106 = arith.addi %105, %104 : i64
      linalg.yield %106 : i64
    } -> tensor<32xi64>
    %107 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} 1 : i64
    %108 = tensor.splat %107 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<32xi64>
    %109 = tensor.empty() : tensor<32xi64>
    %110 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%99, %108 : tensor<32xi64>, tensor<32xi64>) outs(%109 : tensor<32xi64>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb2(%111: i64, %112: i64, %113: i64):
      %114 = arith.addi %111, %112 : i64
      linalg.yield %114 : i64
    } -> tensor<32xi64>
    %115 = tensor.expand_shape %110 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<32xi64> into tensor<32x1xi64>
    %116 = tensor.empty() : tensor<32x32xi1>
    %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%99, %115 : tensor<32xi64>, tensor<32x1xi64>) outs(%116 : tensor<32x32xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.lt.Tensor", prov.orig_dtype = "bool", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb3(%118: i64, %119: i64, %120: i1):
      %121 = arith.cmpi slt, %118, %119 : i64
      linalg.yield %121 : i1
    } -> tensor<32x32xi1>
    %122 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} 0.000000e+00 : f32
    %123 = tensor.splat %122 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<f32>
    %124 = tensor.empty() : tensor<32x32xf32>
    %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%117, %123, %97 : tensor<32x32xi1>, tensor<f32>, tensor<32x32xf32>) outs(%124 : tensor<32x32xf32>) attrs =  {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb4(%126: i1, %127: f32, %128: f32, %129: f32):
      %130 = arith.select %126, %127, %128 : f32
      linalg.yield %130 : f32
    } -> tensor<32x32xf32>
    %131 = "tensor.extract_slice"(%85) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : (tensor<1x32xi64>) -> tensor<1x32xi64>
    %132 = tensor.collapse_shape %131 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1x32xi64> into tensor<32xi64>
    %133 = tensor.expand_shape %132 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<32xi64> into tensor<1x1x32xi64>
    %134 = tensor.collapse_shape %133 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1x1x32xi64> into tensor<32xi64>
    %135 = tensor.expand_shape %134 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<32xi64> into tensor<1x1x1x32xi64>
    %136 = "tensor.extract_slice"(%135) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 1, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} : (tensor<1x1x1x32xi64>) -> tensor<1x1x1x32xi64>
    %137 = tensor.empty() : tensor<1x1x32x32xi64>
    %138 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%136 : tensor<1x1x1x32xi64>) outs(%137 : tensor<1x1x32x32xi64>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "int64", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb5(%139: i64, %140: i64):
      linalg.yield %139 : i64
    } -> tensor<1x1x32x32xi64>
    %141 = tensor.empty() : tensor<1x1x32x32xf32>
    %142 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%138 : tensor<1x1x32x32xi64>) outs(%141 : tensor<1x1x32x32xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb6(%143: i64, %144: f32):
      %145 = arith.sitofp %143 : i64 to f32
      linalg.yield %145 : f32
    } -> tensor<1x1x32x32xf32>
    %146 = arith.constant {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} 1.000000e+00 : f32
    %147 = tensor.splat %146 {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1x1x32x32xf32>
    %148 = tensor.empty() : tensor<1x1x32x32xf32>
    %149 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%147, %142 : tensor<1x1x32x32xf32>, tensor<1x1x32x32xf32>) outs(%148 : tensor<1x1x32x32xf32>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb7(%150: f32, %151: f32, %152: f32):
      %153 = arith.subf %150, %151 : f32
      linalg.yield %153 : f32
    } -> tensor<1x1x32x32xf32>
    %154 = tensor.empty() : tensor<1x1x32x32xi1>
    %155 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%149 : tensor<1x1x32x32xf32>) outs(%154 : tensor<1x1x32x32xi1>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bool", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb8(%156: f32, %157: i1):
      %158 = arith.fptosi %156 : f32 to i1
      linalg.yield %158 : i1
    } -> tensor<1x1x32x32xi1>
    %159 = arith.constant {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} -3.40282347e+38 : f32
    %160 = tensor.splat %159 {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<f32>
    %161 = tensor.empty() : tensor<1x1x32x32xf32>
    %162 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%155, %160, %149 : tensor<1x1x32x32xi1>, tensor<f32>, tensor<1x1x32x32xf32>) outs(%161 : tensor<1x1x32x32xf32>) attrs =  {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb9(%163: i1, %164: f32, %165: f32, %166: f32):
      %167 = arith.select %163, %164, %165 : f32
      linalg.yield %167 : f32
    } -> tensor<1x1x32x32xf32>
    %168 = tensor.empty() : tensor<1x1x32x32xi1>
    %169 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%162 : tensor<1x1x32x32xf32>) outs(%168 : tensor<1x1x32x32xi1>) attrs =  {prov.region_id = "dtype_cast_2", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "bool", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb10(%170: f32, %171: i1):
      %172 = arith.fptosi %170 : f32 to i1
      linalg.yield %172 : i1
    } -> tensor<1x1x32x32xi1>
    %173 = tensor.collapse_shape %125 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<32x32xf32> into tensor<1024xf32>
    %174 = tensor.expand_shape %173 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 32] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1024xf32> into tensor<1x32x32xf32>
    %175 = tensor.collapse_shape %174 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %176 = tensor.expand_shape %175 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %177 = "tensor.extract_slice"(%176) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    %178 = "tensor.extract_slice"(%177) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    %179 = tensor.empty() : tensor<1x1x32x32xf32>
    %180 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%178 : tensor<1x1x32x32xf32>) outs(%179 : tensor<1x1x32x32xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb11(%181: f32, %182: f32):
      linalg.yield %181 : f32
    } -> tensor<1x1x32x32xf32>
    %183 = arith.constant {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} -3.40282347e+38 : f32
    %184 = tensor.splat %183 {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} : tensor<f32>
    %185 = tensor.empty() : tensor<1x1x32x32xf32>
    %186 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%169, %184, %180 : tensor<1x1x32x32xi1>, tensor<f32>, tensor<1x1x32x32xf32>) outs(%185 : tensor<1x1x32x32xf32>) attrs =  {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model"} {
    ^bb12(%187: i1, %188: f32, %189: f32, %190: f32):
      %191 = arith.select %187, %188, %189 : f32
      linalg.yield %191 : f32
    } -> tensor<1x1x32x32xf32>
    %192 = tensor.empty() : tensor<1x32x256xf32>
    %193 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84 : tensor<1x32x256xf32>) outs(%192 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb13(%194: f32, %195: f32):
      %196 = arith.constant 2.000000e+00 : f32
      %197 = math.powf %194, %196 : f32
      linalg.yield %197 : f32
    } -> tensor<1x32x256xf32>
    %198 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} 0.000000e+00 : f32
    %199 = tensor.splat %198 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} : tensor<1x32xf32>
    %200 = linalg.reduce ins(%193:tensor<1x32x256xf32>) outs(%199:tensor<1x32xf32>) dimensions = [2]
    (%201: f32, %202: f32) {
      %203 = arith.addf %201, %202 : f32
      linalg.yield %203 : f32
    }
    %204 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} 2.560000e+02 : f32
    %205 = tensor.splat %204 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} : tensor<1x32xf32>
    %206 = tensor.empty() : tensor<1x32xf32>
    %207 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%200, %205 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%206 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb14(%208: f32, %209: f32, %210: f32):
      %211 = arith.divf %208, %209 : f32
      linalg.yield %211 : f32
    } -> tensor<1x32xf32>
    %212 = tensor.collapse_shape %207 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %213 = tensor.expand_shape %212 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %214 = arith.constant {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} 1.000000e-05 : f32
    %215 = tensor.splat %214 {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} : tensor<1x32x1xf32>
    %216 = tensor.empty() : tensor<1x32x1xf32>
    %217 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%213, %215 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%216 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb15(%218: f32, %219: f32, %220: f32):
      %221 = arith.addf %218, %219 : f32
      linalg.yield %221 : f32
    } -> tensor<1x32x1xf32>
    %222 = tensor.empty() : tensor<1x32x1xf32>
    %223 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%217 : tensor<1x32x1xf32>) outs(%222 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb16(%224: f32, %225: f32):
      %226 = math.rsqrt %224 : f32
      linalg.yield %226 : f32
    } -> tensor<1x32x1xf32>
    %227 = tensor.empty() : tensor<1x32x256xf32>
    %228 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84, %223 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%227 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb17(%229: f32, %230: f32, %231: f32):
      %232 = arith.mulf %229, %230 : f32
      linalg.yield %232 : f32
    } -> tensor<1x32x256xf32>
    %233 = tensor.empty() : tensor<1x32x256xf32>
    %234 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62, %228 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%233 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.input_layernorm"} {
    ^bb18(%235: f32, %236: f32, %237: f32):
      %238 = arith.mulf %235, %236 : f32
      linalg.yield %238 : f32
    } -> tensor<1x32x256xf32>
    %239 = tensor.empty() : tensor<1x32x256xf32>
    %240 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%234 : tensor<1x32x256xf32>) outs(%239 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_0", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb19(%241: f32, %242: f32):
      %243 = math.absf %241 : f32
      linalg.yield %243 : f32
    } -> tensor<1x32x256xf32>
    %244 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} 0xff800000 : f32
    %245 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} 0 : i64
    %246 = tensor.splat %244 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<1x32xf32>
    %247 = tensor.splat %245 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<1x32xi64>
    %248, %249 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%240 : tensor<1x32x256xf32>) outs(%246, %247 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb20(%250: f32, %251: f32, %252: i64):
      %253 = linalg.index 2 : index
      %254 = arith.index_cast %253 : index to i64
      %255 = arith.cmpf ogt, %250, %251 : f32
      %256 = arith.select %255, %250, %251 : f32
      %257 = arith.select %255, %254, %252 : i64
      linalg.yield %256, %257 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %258 = tensor.collapse_shape %248 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %259 = tensor.expand_shape %258 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %260 = tensor.collapse_shape %249 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %261 = tensor.expand_shape %260 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %262 = tensor.empty() : tensor<1x32x1xf32>
    %263 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%259 : tensor<1x32x1xf32>) outs(%262 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "minmax_0", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb21(%264: f32, %265: f32):
      %266 = arith.constant 1.000000e-05 : f32
      %267 = arith.maximumf %264, %266 : f32
      linalg.yield %267 : f32
    } -> tensor<1x32x1xf32>
    %268 = tensor.empty() : tensor<1x32x1xf32>
    %269 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%263 : tensor<1x32x1xf32>) outs(%268 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_0", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb22(%270: f32, %271: f32):
      %272 = arith.constant 1.000000e+00 : f32
      %273 = arith.divf %272, %270 : f32
      linalg.yield %273 : f32
    } -> tensor<1x32x1xf32>
    %274 = arith.constant {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} 1.270000e+02 : f32
    %275 = tensor.splat %274 {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<1x32x1xf32>
    %276 = tensor.empty() : tensor<1x32x1xf32>
    %277 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%269, %275 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%276 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb23(%278: f32, %279: f32, %280: f32):
      %281 = arith.mulf %278, %279 : f32
      linalg.yield %281 : f32
    } -> tensor<1x32x1xf32>
    %282 = tensor.empty() : tensor<1x32x256xf32>
    %283 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%234, %277 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%282 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb24(%284: f32, %285: f32, %286: f32):
      %287 = arith.mulf %284, %285 : f32
      linalg.yield %287 : f32
    } -> tensor<1x32x256xf32>
    %288 = tensor.empty() : tensor<1x32x256xf32>
    %289 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%283 : tensor<1x32x256xf32>) outs(%288 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_0", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb25(%290: f32, %291: f32):
      %292 = math.roundeven %290 : f32
      linalg.yield %292 : f32
    } -> tensor<1x32x256xf32>
    %293 = tensor.empty() : tensor<1x32x256xf32>
    %294 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%289 : tensor<1x32x256xf32>) outs(%293 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_1", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb26(%295: f32, %296: f32):
      %297 = arith.constant -1.280000e+02 : f32
      %298 = arith.maximumf %295, %297 : f32
      %299 = arith.constant 1.270000e+02 : f32
      %300 = arith.minimumf %298, %299 : f32
      linalg.yield %300 : f32
    } -> tensor<1x32x256xf32>
    %301 = tensor.empty() : tensor<1x32x256xf32>
    %302 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%294, %277 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%301 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb27(%303: f32, %304: f32, %305: f32):
      %306 = arith.divf %303, %304 : f32
      linalg.yield %306 : f32
    } -> tensor<1x32x256xf32>
    %307 = tensor.empty() : tensor<256x256xf32>
    %308 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%53 : tensor<256x256xf32>) outs(%307 : tensor<256x256xf32>) attrs =  {prov.region_id = "abs_1", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb28(%309: f32, %310: f32):
      %311 = math.absf %309 : f32
      linalg.yield %311 : f32
    } -> tensor<256x256xf32>
    %312 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} 0.000000e+00 : f32
    %313 = tensor.splat %312 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<f32>
    %314 = linalg.reduce ins(%308:tensor<256x256xf32>) outs(%313:tensor<f32>) dimensions = [0, 1]
    (%315: f32, %316: f32) {
      %317 = arith.addf %315, %316 : f32
      linalg.yield %317 : f32
    }
    %318 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} 6.553600e+04 : f32
    %319 = tensor.splat %318 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<f32>
    %320 = tensor.empty() : tensor<f32>
    %321 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%314, %319 : tensor<f32>, tensor<f32>) outs(%320 : tensor<f32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb29(%322: f32, %323: f32, %324: f32):
      %325 = arith.divf %322, %323 : f32
      linalg.yield %325 : f32
    } -> tensor<f32>
    %326 = tensor.empty() : tensor<f32>
    %327 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%321 : tensor<f32>) outs(%326 : tensor<f32>) attrs =  {prov.region_id = "minmax_2", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb30(%328: f32, %329: f32):
      %330 = arith.constant 1.000000e-05 : f32
      %331 = arith.maximumf %328, %330 : f32
      linalg.yield %331 : f32
    } -> tensor<f32>
    %332 = tensor.empty() : tensor<f32>
    %333 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%327 : tensor<f32>) outs(%332 : tensor<f32>) attrs =  {prov.region_id = "elementwise_1", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb31(%334: f32, %335: f32):
      %336 = arith.constant 1.000000e+00 : f32
      %337 = arith.divf %336, %334 : f32
      linalg.yield %337 : f32
    } -> tensor<f32>
    %338 = arith.constant {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} 1.000000e+00 : f32
    %339 = tensor.splat %338 {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<f32>
    %340 = tensor.empty() : tensor<f32>
    %341 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%333, %339 : tensor<f32>, tensor<f32>) outs(%340 : tensor<f32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb32(%342: f32, %343: f32, %344: f32):
      %345 = arith.mulf %342, %343 : f32
      linalg.yield %345 : f32
    } -> tensor<f32>
    %346 = tensor.empty() : tensor<256x256xf32>
    %347 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%53, %341 : tensor<256x256xf32>, tensor<f32>) outs(%346 : tensor<256x256xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb33(%348: f32, %349: f32, %350: f32):
      %351 = arith.mulf %348, %349 : f32
      linalg.yield %351 : f32
    } -> tensor<256x256xf32>
    %352 = tensor.empty() : tensor<256x256xf32>
    %353 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%347 : tensor<256x256xf32>) outs(%352 : tensor<256x256xf32>) attrs =  {prov.region_id = "round_1", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb34(%354: f32, %355: f32):
      %356 = math.roundeven %354 : f32
      linalg.yield %356 : f32
    } -> tensor<256x256xf32>
    %357 = tensor.empty() : tensor<256x256xf32>
    %358 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%353 : tensor<256x256xf32>) outs(%357 : tensor<256x256xf32>) attrs =  {prov.region_id = "minmax_3", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb35(%359: f32, %360: f32):
      %361 = arith.constant -1.000000e+00 : f32
      %362 = arith.maximumf %359, %361 : f32
      %363 = arith.constant 1.000000e+00 : f32
      %364 = arith.minimumf %362, %363 : f32
      linalg.yield %364 : f32
    } -> tensor<256x256xf32>
    %365 = tensor.empty() : tensor<256x256xf32>
    %366 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%358, %341 : tensor<256x256xf32>, tensor<f32>) outs(%365 : tensor<256x256xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} {
    ^bb36(%367: f32, %368: f32, %369: f32):
      %370 = arith.divf %367, %368 : f32
      linalg.yield %370 : f32
    } -> tensor<256x256xf32>
    %371 = tensor.empty() : tensor<256x256xf32>
    %372 = linalg.transpose ins(%366:tensor<256x256xf32>) outs(%371:tensor<256x256xf32>) permutation = [1, 0]
    %373 = tensor.collapse_shape %302 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %374 = tensor.expand_shape %373 [[0 : i64, 1 : i64]] output_shape [32, 256] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<8192xf32> into tensor<32x256xf32>
    %375 = tensor.empty() : tensor<32x256xf32>
    %376 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %377 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%376 : f32) outs(%375 : tensor<32x256xf32>) -> tensor<32x256xf32>
    %378 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj", prov.transposed_b = "true"} ins(%374, %372 : tensor<32x256xf32>, tensor<256x256xf32>) outs(%377 : tensor<32x256xf32>) -> tensor<32x256xf32>
    %379 = tensor.collapse_shape %378 [[0 : i64, 1 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<32x256xf32> into tensor<8192xf32>
    %380 = tensor.expand_shape %379 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 256] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.q_proj"} : tensor<8192xf32> into tensor<1x32x256xf32>
    %381 = tensor.empty() : tensor<1x32x256xf32>
    %382 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%234 : tensor<1x32x256xf32>) outs(%381 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_2", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb37(%383: f32, %384: f32):
      %385 = math.absf %383 : f32
      linalg.yield %385 : f32
    } -> tensor<1x32x256xf32>
    %386 = arith.constant {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} 0xff800000 : f32
    %387 = arith.constant {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} 0 : i64
    %388 = tensor.splat %386 {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<1x32xf32>
    %389 = tensor.splat %387 {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<1x32xi64>
    %390, %391 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%382 : tensor<1x32x256xf32>) outs(%388, %389 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb38(%392: f32, %393: f32, %394: i64):
      %395 = linalg.index 2 : index
      %396 = arith.index_cast %395 : index to i64
      %397 = arith.cmpf ogt, %392, %393 : f32
      %398 = arith.select %397, %392, %393 : f32
      %399 = arith.select %397, %396, %394 : i64
      linalg.yield %398, %399 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %400 = tensor.collapse_shape %390 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %401 = tensor.expand_shape %400 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %402 = tensor.collapse_shape %391 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %403 = tensor.expand_shape %402 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_1", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %404 = tensor.empty() : tensor<1x32x1xf32>
    %405 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%401 : tensor<1x32x1xf32>) outs(%404 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "minmax_4", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb39(%406: f32, %407: f32):
      %408 = arith.constant 1.000000e-05 : f32
      %409 = arith.maximumf %406, %408 : f32
      linalg.yield %409 : f32
    } -> tensor<1x32x1xf32>
    %410 = tensor.empty() : tensor<1x32x1xf32>
    %411 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%405 : tensor<1x32x1xf32>) outs(%410 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_2", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb40(%412: f32, %413: f32):
      %414 = arith.constant 1.000000e+00 : f32
      %415 = arith.divf %414, %412 : f32
      linalg.yield %415 : f32
    } -> tensor<1x32x1xf32>
    %416 = arith.constant {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} 1.270000e+02 : f32
    %417 = tensor.splat %416 {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<1x32x1xf32>
    %418 = tensor.empty() : tensor<1x32x1xf32>
    %419 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%411, %417 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%418 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb41(%420: f32, %421: f32, %422: f32):
      %423 = arith.mulf %420, %421 : f32
      linalg.yield %423 : f32
    } -> tensor<1x32x1xf32>
    %424 = tensor.empty() : tensor<1x32x256xf32>
    %425 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%234, %419 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%424 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb42(%426: f32, %427: f32, %428: f32):
      %429 = arith.mulf %426, %427 : f32
      linalg.yield %429 : f32
    } -> tensor<1x32x256xf32>
    %430 = tensor.empty() : tensor<1x32x256xf32>
    %431 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%425 : tensor<1x32x256xf32>) outs(%430 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_2", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb43(%432: f32, %433: f32):
      %434 = math.roundeven %432 : f32
      linalg.yield %434 : f32
    } -> tensor<1x32x256xf32>
    %435 = tensor.empty() : tensor<1x32x256xf32>
    %436 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%431 : tensor<1x32x256xf32>) outs(%435 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_5", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb44(%437: f32, %438: f32):
      %439 = arith.constant -1.280000e+02 : f32
      %440 = arith.maximumf %437, %439 : f32
      %441 = arith.constant 1.270000e+02 : f32
      %442 = arith.minimumf %440, %441 : f32
      linalg.yield %442 : f32
    } -> tensor<1x32x256xf32>
    %443 = tensor.empty() : tensor<1x32x256xf32>
    %444 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%436, %419 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%443 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb45(%445: f32, %446: f32, %447: f32):
      %448 = arith.divf %445, %446 : f32
      linalg.yield %448 : f32
    } -> tensor<1x32x256xf32>
    %449 = tensor.empty() : tensor<128x256xf32>
    %450 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%54 : tensor<128x256xf32>) outs(%449 : tensor<128x256xf32>) attrs =  {prov.region_id = "abs_3", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb46(%451: f32, %452: f32):
      %453 = math.absf %451 : f32
      linalg.yield %453 : f32
    } -> tensor<128x256xf32>
    %454 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} 0.000000e+00 : f32
    %455 = tensor.splat %454 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<f32>
    %456 = linalg.reduce ins(%450:tensor<128x256xf32>) outs(%455:tensor<f32>) dimensions = [0, 1]
    (%457: f32, %458: f32) {
      %459 = arith.addf %457, %458 : f32
      linalg.yield %459 : f32
    }
    %460 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} 3.276800e+04 : f32
    %461 = tensor.splat %460 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<f32>
    %462 = tensor.empty() : tensor<f32>
    %463 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%456, %461 : tensor<f32>, tensor<f32>) outs(%462 : tensor<f32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb47(%464: f32, %465: f32, %466: f32):
      %467 = arith.divf %464, %465 : f32
      linalg.yield %467 : f32
    } -> tensor<f32>
    %468 = tensor.empty() : tensor<f32>
    %469 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%463 : tensor<f32>) outs(%468 : tensor<f32>) attrs =  {prov.region_id = "minmax_6", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb48(%470: f32, %471: f32):
      %472 = arith.constant 1.000000e-05 : f32
      %473 = arith.maximumf %470, %472 : f32
      linalg.yield %473 : f32
    } -> tensor<f32>
    %474 = tensor.empty() : tensor<f32>
    %475 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%469 : tensor<f32>) outs(%474 : tensor<f32>) attrs =  {prov.region_id = "elementwise_3", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb49(%476: f32, %477: f32):
      %478 = arith.constant 1.000000e+00 : f32
      %479 = arith.divf %478, %476 : f32
      linalg.yield %479 : f32
    } -> tensor<f32>
    %480 = arith.constant {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} 1.000000e+00 : f32
    %481 = tensor.splat %480 {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<f32>
    %482 = tensor.empty() : tensor<f32>
    %483 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%475, %481 : tensor<f32>, tensor<f32>) outs(%482 : tensor<f32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb50(%484: f32, %485: f32, %486: f32):
      %487 = arith.mulf %484, %485 : f32
      linalg.yield %487 : f32
    } -> tensor<f32>
    %488 = tensor.empty() : tensor<128x256xf32>
    %489 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%54, %483 : tensor<128x256xf32>, tensor<f32>) outs(%488 : tensor<128x256xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb51(%490: f32, %491: f32, %492: f32):
      %493 = arith.mulf %490, %491 : f32
      linalg.yield %493 : f32
    } -> tensor<128x256xf32>
    %494 = tensor.empty() : tensor<128x256xf32>
    %495 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%489 : tensor<128x256xf32>) outs(%494 : tensor<128x256xf32>) attrs =  {prov.region_id = "round_3", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb52(%496: f32, %497: f32):
      %498 = math.roundeven %496 : f32
      linalg.yield %498 : f32
    } -> tensor<128x256xf32>
    %499 = tensor.empty() : tensor<128x256xf32>
    %500 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%495 : tensor<128x256xf32>) outs(%499 : tensor<128x256xf32>) attrs =  {prov.region_id = "minmax_7", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb53(%501: f32, %502: f32):
      %503 = arith.constant -1.000000e+00 : f32
      %504 = arith.maximumf %501, %503 : f32
      %505 = arith.constant 1.000000e+00 : f32
      %506 = arith.minimumf %504, %505 : f32
      linalg.yield %506 : f32
    } -> tensor<128x256xf32>
    %507 = tensor.empty() : tensor<128x256xf32>
    %508 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%500, %483 : tensor<128x256xf32>, tensor<f32>) outs(%507 : tensor<128x256xf32>) attrs =  {prov.region_id = "div_3", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} {
    ^bb54(%509: f32, %510: f32, %511: f32):
      %512 = arith.divf %509, %510 : f32
      linalg.yield %512 : f32
    } -> tensor<128x256xf32>
    %513 = tensor.empty() : tensor<256x128xf32>
    %514 = linalg.transpose ins(%508:tensor<128x256xf32>) outs(%513:tensor<256x128xf32>) permutation = [1, 0]
    %515 = tensor.collapse_shape %444 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %516 = tensor.expand_shape %515 [[0 : i64, 1 : i64]] output_shape [32, 256] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<8192xf32> into tensor<32x256xf32>
    %517 = tensor.empty() : tensor<32x128xf32>
    %518 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %519 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%518 : f32) outs(%517 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %520 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj", prov.transposed_b = "true"} ins(%516, %514 : tensor<32x256xf32>, tensor<256x128xf32>) outs(%519 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %521 = tensor.collapse_shape %520 [[0 : i64, 1 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<32x128xf32> into tensor<4096xf32>
    %522 = tensor.expand_shape %521 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 128] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.k_proj"} : tensor<4096xf32> into tensor<1x32x128xf32>
    %523 = tensor.empty() : tensor<1x32x256xf32>
    %524 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%234 : tensor<1x32x256xf32>) outs(%523 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_4", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb55(%525: f32, %526: f32):
      %527 = math.absf %525 : f32
      linalg.yield %527 : f32
    } -> tensor<1x32x256xf32>
    %528 = arith.constant {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} 0xff800000 : f32
    %529 = arith.constant {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} 0 : i64
    %530 = tensor.splat %528 {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<1x32xf32>
    %531 = tensor.splat %529 {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<1x32xi64>
    %532, %533 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%524 : tensor<1x32x256xf32>) outs(%530, %531 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb56(%534: f32, %535: f32, %536: i64):
      %537 = linalg.index 2 : index
      %538 = arith.index_cast %537 : index to i64
      %539 = arith.cmpf ogt, %534, %535 : f32
      %540 = arith.select %539, %534, %535 : f32
      %541 = arith.select %539, %538, %536 : i64
      linalg.yield %540, %541 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %542 = tensor.collapse_shape %532 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %543 = tensor.expand_shape %542 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %544 = tensor.collapse_shape %533 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %545 = tensor.expand_shape %544 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_2", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %546 = tensor.empty() : tensor<1x32x1xf32>
    %547 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%543 : tensor<1x32x1xf32>) outs(%546 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "minmax_8", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb57(%548: f32, %549: f32):
      %550 = arith.constant 1.000000e-05 : f32
      %551 = arith.maximumf %548, %550 : f32
      linalg.yield %551 : f32
    } -> tensor<1x32x1xf32>
    %552 = tensor.empty() : tensor<1x32x1xf32>
    %553 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%547 : tensor<1x32x1xf32>) outs(%552 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_4", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb58(%554: f32, %555: f32):
      %556 = arith.constant 1.000000e+00 : f32
      %557 = arith.divf %556, %554 : f32
      linalg.yield %557 : f32
    } -> tensor<1x32x1xf32>
    %558 = arith.constant {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} 1.270000e+02 : f32
    %559 = tensor.splat %558 {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<1x32x1xf32>
    %560 = tensor.empty() : tensor<1x32x1xf32>
    %561 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%553, %559 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%560 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb59(%562: f32, %563: f32, %564: f32):
      %565 = arith.mulf %562, %563 : f32
      linalg.yield %565 : f32
    } -> tensor<1x32x1xf32>
    %566 = tensor.empty() : tensor<1x32x256xf32>
    %567 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%234, %561 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%566 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb60(%568: f32, %569: f32, %570: f32):
      %571 = arith.mulf %568, %569 : f32
      linalg.yield %571 : f32
    } -> tensor<1x32x256xf32>
    %572 = tensor.empty() : tensor<1x32x256xf32>
    %573 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%567 : tensor<1x32x256xf32>) outs(%572 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_4", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb61(%574: f32, %575: f32):
      %576 = math.roundeven %574 : f32
      linalg.yield %576 : f32
    } -> tensor<1x32x256xf32>
    %577 = tensor.empty() : tensor<1x32x256xf32>
    %578 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%573 : tensor<1x32x256xf32>) outs(%577 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_9", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb62(%579: f32, %580: f32):
      %581 = arith.constant -1.280000e+02 : f32
      %582 = arith.maximumf %579, %581 : f32
      %583 = arith.constant 1.270000e+02 : f32
      %584 = arith.minimumf %582, %583 : f32
      linalg.yield %584 : f32
    } -> tensor<1x32x256xf32>
    %585 = tensor.empty() : tensor<1x32x256xf32>
    %586 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%578, %561 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%585 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_4", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb63(%587: f32, %588: f32, %589: f32):
      %590 = arith.divf %587, %588 : f32
      linalg.yield %590 : f32
    } -> tensor<1x32x256xf32>
    %591 = tensor.empty() : tensor<128x256xf32>
    %592 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%55 : tensor<128x256xf32>) outs(%591 : tensor<128x256xf32>) attrs =  {prov.region_id = "abs_5", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb64(%593: f32, %594: f32):
      %595 = math.absf %593 : f32
      linalg.yield %595 : f32
    } -> tensor<128x256xf32>
    %596 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} 0.000000e+00 : f32
    %597 = tensor.splat %596 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<f32>
    %598 = linalg.reduce ins(%592:tensor<128x256xf32>) outs(%597:tensor<f32>) dimensions = [0, 1]
    (%599: f32, %600: f32) {
      %601 = arith.addf %599, %600 : f32
      linalg.yield %601 : f32
    }
    %602 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} 3.276800e+04 : f32
    %603 = tensor.splat %602 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<f32>
    %604 = tensor.empty() : tensor<f32>
    %605 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%598, %603 : tensor<f32>, tensor<f32>) outs(%604 : tensor<f32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb65(%606: f32, %607: f32, %608: f32):
      %609 = arith.divf %606, %607 : f32
      linalg.yield %609 : f32
    } -> tensor<f32>
    %610 = tensor.empty() : tensor<f32>
    %611 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%605 : tensor<f32>) outs(%610 : tensor<f32>) attrs =  {prov.region_id = "minmax_10", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb66(%612: f32, %613: f32):
      %614 = arith.constant 1.000000e-05 : f32
      %615 = arith.maximumf %612, %614 : f32
      linalg.yield %615 : f32
    } -> tensor<f32>
    %616 = tensor.empty() : tensor<f32>
    %617 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%611 : tensor<f32>) outs(%616 : tensor<f32>) attrs =  {prov.region_id = "elementwise_5", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb67(%618: f32, %619: f32):
      %620 = arith.constant 1.000000e+00 : f32
      %621 = arith.divf %620, %618 : f32
      linalg.yield %621 : f32
    } -> tensor<f32>
    %622 = arith.constant {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} 1.000000e+00 : f32
    %623 = tensor.splat %622 {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<f32>
    %624 = tensor.empty() : tensor<f32>
    %625 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%617, %623 : tensor<f32>, tensor<f32>) outs(%624 : tensor<f32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb68(%626: f32, %627: f32, %628: f32):
      %629 = arith.mulf %626, %627 : f32
      linalg.yield %629 : f32
    } -> tensor<f32>
    %630 = tensor.empty() : tensor<128x256xf32>
    %631 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%55, %625 : tensor<128x256xf32>, tensor<f32>) outs(%630 : tensor<128x256xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb69(%632: f32, %633: f32, %634: f32):
      %635 = arith.mulf %632, %633 : f32
      linalg.yield %635 : f32
    } -> tensor<128x256xf32>
    %636 = tensor.empty() : tensor<128x256xf32>
    %637 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%631 : tensor<128x256xf32>) outs(%636 : tensor<128x256xf32>) attrs =  {prov.region_id = "round_5", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb70(%638: f32, %639: f32):
      %640 = math.roundeven %638 : f32
      linalg.yield %640 : f32
    } -> tensor<128x256xf32>
    %641 = tensor.empty() : tensor<128x256xf32>
    %642 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%637 : tensor<128x256xf32>) outs(%641 : tensor<128x256xf32>) attrs =  {prov.region_id = "minmax_11", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb71(%643: f32, %644: f32):
      %645 = arith.constant -1.000000e+00 : f32
      %646 = arith.maximumf %643, %645 : f32
      %647 = arith.constant 1.000000e+00 : f32
      %648 = arith.minimumf %646, %647 : f32
      linalg.yield %648 : f32
    } -> tensor<128x256xf32>
    %649 = tensor.empty() : tensor<128x256xf32>
    %650 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%642, %625 : tensor<128x256xf32>, tensor<f32>) outs(%649 : tensor<128x256xf32>) attrs =  {prov.region_id = "div_5", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} {
    ^bb72(%651: f32, %652: f32, %653: f32):
      %654 = arith.divf %651, %652 : f32
      linalg.yield %654 : f32
    } -> tensor<128x256xf32>
    %655 = tensor.empty() : tensor<256x128xf32>
    %656 = linalg.transpose ins(%650:tensor<128x256xf32>) outs(%655:tensor<256x128xf32>) permutation = [1, 0]
    %657 = tensor.collapse_shape %586 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %658 = tensor.expand_shape %657 [[0 : i64, 1 : i64]] output_shape [32, 256] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<8192xf32> into tensor<32x256xf32>
    %659 = tensor.empty() : tensor<32x128xf32>
    %660 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %661 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%660 : f32) outs(%659 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %662 = linalg.matmul {prov.region_id = "matmul_2", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj", prov.transposed_b = "true"} ins(%658, %656 : tensor<32x256xf32>, tensor<256x128xf32>) outs(%661 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %663 = tensor.collapse_shape %662 [[0 : i64, 1 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<32x128xf32> into tensor<4096xf32>
    %664 = tensor.expand_shape %663 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 128] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.v_proj"} : tensor<4096xf32> into tensor<1x32x128xf32>
    %665 = tensor.collapse_shape %380 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %666 = tensor.expand_shape %665 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 32] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x32x8x32xf32>
    %667 = tensor.empty() : tensor<1x8x32x32xf32>
    %668 = linalg.transpose ins(%666:tensor<1x32x8x32xf32>) outs(%667:tensor<1x8x32x32xf32>) permutation = [0, 2, 1, 3]
    %669 = tensor.collapse_shape %522 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %670 = tensor.expand_shape %669 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 32] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<4096xf32> into tensor<1x32x4x32xf32>
    %671 = tensor.empty() : tensor<1x4x32x32xf32>
    %672 = linalg.transpose ins(%670:tensor<1x32x4x32xf32>) outs(%671:tensor<1x4x32x32xf32>) permutation = [0, 2, 1, 3]
    %673 = tensor.collapse_shape %664 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %674 = tensor.expand_shape %673 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 32] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<4096xf32> into tensor<1x32x4x32xf32>
    %675 = tensor.empty() : tensor<1x4x32x32xf32>
    %676 = linalg.transpose ins(%674:tensor<1x32x4x32xf32>) outs(%675:tensor<1x4x32x32xf32>) permutation = [0, 2, 1, 3]
    %677 = "tensor.extract_slice"(%79) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.rotary_emb"} : (tensor<2048x32xf32>) -> tensor<32x32xf32>
    %678 = "tensor.extract_slice"(%80) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.rotary_emb"} : (tensor<2048x32xf32>) -> tensor<32x32xf32>
    %679 = tensor.empty() : tensor<1x32x32xf32>
    %680 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95 : tensor<1x32xi64>) outs(%679 : tensor<1x32x32xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb73(%681: i64, %682: f32):
      %683 = arith.index_cast %681 : i64 to index
      %684 = linalg.index 2 : index
      %685 = tensor.extract %677[%683, %684] : tensor<32x32xf32>
      linalg.yield %685 : f32
    } -> tensor<1x32x32xf32>
    %686 = tensor.collapse_shape %680 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %687 = tensor.expand_shape %686 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %688 = tensor.empty() : tensor<1x32x32xf32>
    %689 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95 : tensor<1x32xi64>) outs(%688 : tensor<1x32x32xf32>) attrs =  {prov.region_id = "gather_1", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb74(%690: i64, %691: f32):
      %692 = arith.index_cast %690 : i64 to index
      %693 = linalg.index 2 : index
      %694 = tensor.extract %678[%692, %693] : tensor<32x32xf32>
      linalg.yield %694 : f32
    } -> tensor<1x32x32xf32>
    %695 = tensor.collapse_shape %689 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %696 = tensor.expand_shape %695 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %697 = tensor.empty() : tensor<1x8x32x32xf32>
    %698 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%668, %687 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%697 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb75(%699: f32, %700: f32, %701: f32):
      %702 = arith.mulf %699, %700 : f32
      linalg.yield %702 : f32
    } -> tensor<1x8x32x32xf32>
    %703 = "tensor.extract_slice"(%668) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x8x32x32xf32>) -> tensor<1x8x32x16xf32>
    %704 = "tensor.extract_slice"(%668) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 8, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x8x32x32xf32>) -> tensor<1x8x32x16xf32>
    %705 = tensor.empty() : tensor<1x8x32x16xf32>
    %706 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%704 : tensor<1x8x32x16xf32>) outs(%705 : tensor<1x8x32x16xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb76(%707: f32, %708: f32):
      %709 = arith.negf %707 : f32
      linalg.yield %709 : f32
    } -> tensor<1x8x32x16xf32>
    %710 = tensor.concat dim(3) %706, %703 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x8x32x16xf32>, tensor<1x8x32x16xf32>) -> tensor<1x8x32x32xf32>
    %711 = tensor.empty() : tensor<1x8x32x32xf32>
    %712 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%710, %696 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%711 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb77(%713: f32, %714: f32, %715: f32):
      %716 = arith.mulf %713, %714 : f32
      linalg.yield %716 : f32
    } -> tensor<1x8x32x32xf32>
    %717 = tensor.empty() : tensor<1x8x32x32xf32>
    %718 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%698, %712 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%717 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb78(%719: f32, %720: f32, %721: f32):
      %722 = arith.addf %719, %720 : f32
      linalg.yield %722 : f32
    } -> tensor<1x8x32x32xf32>
    %723 = tensor.empty() : tensor<1x4x32x32xf32>
    %724 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%672, %687 : tensor<1x4x32x32xf32>, tensor<1x1x32x32xf32>) outs(%723 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb79(%725: f32, %726: f32, %727: f32):
      %728 = arith.mulf %725, %726 : f32
      linalg.yield %728 : f32
    } -> tensor<1x4x32x32xf32>
    %729 = "tensor.extract_slice"(%672) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x16xf32>
    %730 = "tensor.extract_slice"(%672) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_9", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x16xf32>
    %731 = tensor.empty() : tensor<1x4x32x16xf32>
    %732 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%730 : tensor<1x4x32x16xf32>) outs(%731 : tensor<1x4x32x16xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb80(%733: f32, %734: f32):
      %735 = arith.negf %733 : f32
      linalg.yield %735 : f32
    } -> tensor<1x4x32x16xf32>
    %736 = tensor.concat dim(3) %732, %729 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x16xf32>, tensor<1x4x32x16xf32>) -> tensor<1x4x32x32xf32>
    %737 = tensor.empty() : tensor<1x4x32x32xf32>
    %738 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%736, %696 : tensor<1x4x32x32xf32>, tensor<1x1x32x32xf32>) outs(%737 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb81(%739: f32, %740: f32, %741: f32):
      %742 = arith.mulf %739, %740 : f32
      linalg.yield %742 : f32
    } -> tensor<1x4x32x32xf32>
    %743 = tensor.empty() : tensor<1x4x32x32xf32>
    %744 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%724, %738 : tensor<1x4x32x32xf32>, tensor<1x4x32x32xf32>) outs(%743 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb82(%745: f32, %746: f32, %747: f32):
      %748 = arith.addf %745, %746 : f32
      linalg.yield %748 : f32
    } -> tensor<1x4x32x32xf32>
    %749 = "tensor.extract_slice"(%744) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_10", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %750 = "tensor.extract_slice"(%749) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_11", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %751 = tensor.collapse_shape %750 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x32x32xf32> into tensor<4096xf32>
    %752 = tensor.expand_shape %751 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 32, 32] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<4096xf32> into tensor<1x4x1x32x32xf32>
    %753 = "tensor.extract_slice"(%752) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_12", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %754 = "tensor.extract_slice"(%753) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_13", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %755 = tensor.empty() : tensor<1x4x2x32x32xf32>
    %756 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%754 : tensor<1x4x1x32x32xf32>) outs(%755 : tensor<1x4x2x32x32xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb83(%757: f32, %758: f32):
      linalg.yield %757 : f32
    } -> tensor<1x4x2x32x32xf32>
    %759 = tensor.collapse_shape %756 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x2x32x32xf32> into tensor<8192xf32>
    %760 = tensor.expand_shape %759 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 32] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x8x32x32xf32>
    %761 = "tensor.extract_slice"(%676) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_14", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %762 = "tensor.extract_slice"(%761) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_15", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %763 = tensor.collapse_shape %762 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x32x32xf32> into tensor<4096xf32>
    %764 = tensor.expand_shape %763 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 32, 32] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<4096xf32> into tensor<1x4x1x32x32xf32>
    %765 = "tensor.extract_slice"(%764) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_16", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %766 = "tensor.extract_slice"(%765) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_17", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %767 = tensor.empty() : tensor<1x4x2x32x32xf32>
    %768 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%766 : tensor<1x4x1x32x32xf32>) outs(%767 : tensor<1x4x2x32x32xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb84(%769: f32, %770: f32):
      linalg.yield %769 : f32
    } -> tensor<1x4x2x32x32xf32>
    %771 = tensor.collapse_shape %768 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x4x2x32x32xf32> into tensor<8192xf32>
    %772 = tensor.expand_shape %771 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 32] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x8x32x32xf32>
    %773 = tensor.empty() : tensor<1x8x32x32xf32>
    %774 = linalg.transpose ins(%760:tensor<1x8x32x32xf32>) outs(%773:tensor<1x8x32x32xf32>) permutation = [0, 1, 3, 2]
    %775 = tensor.empty() : tensor<1x8x32x32xf32>
    %776 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%718 : tensor<1x8x32x32xf32>) outs(%775 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb85(%777: f32, %778: f32):
      linalg.yield %777 : f32
    } -> tensor<1x8x32x32xf32>
    %779 = tensor.collapse_shape %776 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32x32xf32> into tensor<8192xf32>
    %780 = tensor.expand_shape %779 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 32, 32] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<8x32x32xf32>
    %781 = tensor.empty() : tensor<1x8x32x32xf32>
    %782 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%774 : tensor<1x8x32x32xf32>) outs(%781 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb86(%783: f32, %784: f32):
      linalg.yield %783 : f32
    } -> tensor<1x8x32x32xf32>
    %785 = tensor.collapse_shape %782 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32x32xf32> into tensor<8192xf32>
    %786 = tensor.expand_shape %785 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 32, 32] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<8x32x32xf32>
    %787 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 0.000000e+00 : f32
    %788 = tensor.splat %787 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8x32x32xf32>
    %789 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%780, %786 : tensor<8x32x32xf32>, tensor<8x32x32xf32>) outs(%788 : tensor<8x32x32xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb87(%790: f32, %791: f32, %792: f32):
      %793 = arith.mulf %790, %791 : f32
      %794 = arith.addf %792, %793 : f32
      linalg.yield %794 : f32
    } -> tensor<8x32x32xf32>
    %795 = tensor.collapse_shape %789 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8x32x32xf32> into tensor<8192xf32>
    %796 = tensor.expand_shape %795 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 32] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x8x32x32xf32>
    %797 = arith.constant {prov.region_id = "div_6", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 5.65685415 : f32
    %798 = tensor.splat %797 {prov.region_id = "div_6", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32x32xf32>
    %799 = tensor.empty() : tensor<1x8x32x32xf32>
    %800 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%796, %798 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%799 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "div_6", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb88(%801: f32, %802: f32, %803: f32):
      %804 = arith.divf %801, %802 : f32
      linalg.yield %804 : f32
    } -> tensor<1x8x32x32xf32>
    %805 = "tensor.extract_slice"(%186) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_18", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    %806 = "tensor.extract_slice"(%805) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_19", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    %807 = "tensor.extract_slice"(%806) <{static_offsets = array<i64: 0, 0, 31, 0>, static_sizes = array<i64: 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<32xf32>
    %808 = tensor.expand_shape %807 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "slice_20", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<32xf32> into tensor<1x1x32xf32>
    %809 = tensor.collapse_shape %808 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x1x32xf32> into tensor<32xf32>
    %810 = tensor.expand_shape %809 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<32xf32> into tensor<1x1x1x32xf32>
    %811 = tensor.empty() : tensor<1x1x32x32xf32>
    %812 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%810 : tensor<1x1x1x32xf32>) outs(%811 : tensor<1x1x32x32xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb89(%813: f32, %814: f32):
      linalg.yield %813 : f32
    } -> tensor<1x1x32x32xf32>
    %815 = tensor.empty() : tensor<1x8x32x32xf32>
    %816 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%800, %812 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%815 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb90(%817: f32, %818: f32, %819: f32):
      %820 = arith.addf %817, %818 : f32
      linalg.yield %820 : f32
    } -> tensor<1x8x32x32xf32>
    %821 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 0xff800000 : f32
    %822 = tensor.splat %821 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32xf32>
    %823 = linalg.reduce ins(%816:tensor<1x8x32x32xf32>) outs(%822:tensor<1x8x32xf32>) dimensions = [3]
    (%824: f32, %825: f32) {
      %826 = arith.maximumf %824, %825 : f32
      linalg.yield %826 : f32
    }
    %827 = tensor.collapse_shape %823 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %828 = tensor.expand_shape %827 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %829 = tensor.empty() : tensor<1x8x32x32xf32>
    %830 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%816, %828 : tensor<1x8x32x32xf32>, tensor<1x8x32x1xf32>) outs(%829 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb91(%831: f32, %832: f32, %833: f32):
      %834 = arith.subf %831, %832 : f32
      linalg.yield %834 : f32
    } -> tensor<1x8x32x32xf32>
    %835 = tensor.empty() : tensor<1x8x32x32xf32>
    %836 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%830 : tensor<1x8x32x32xf32>) outs(%835 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb92(%837: f32, %838: f32):
      %839 = math.exp %837 : f32
      linalg.yield %839 : f32
    } -> tensor<1x8x32x32xf32>
    %840 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 0.000000e+00 : f32
    %841 = tensor.splat %840 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32xf32>
    %842 = linalg.reduce ins(%836:tensor<1x8x32x32xf32>) outs(%841:tensor<1x8x32xf32>) dimensions = [3]
    (%843: f32, %844: f32) {
      %845 = arith.addf %843, %844 : f32
      linalg.yield %845 : f32
    }
    %846 = tensor.collapse_shape %842 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %847 = tensor.expand_shape %846 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %848 = tensor.empty() : tensor<1x8x32x32xf32>
    %849 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%836, %847 : tensor<1x8x32x32xf32>, tensor<1x8x32x1xf32>) outs(%848 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb93(%850: f32, %851: f32, %852: f32):
      %853 = arith.divf %850, %851 : f32
      linalg.yield %853 : f32
    } -> tensor<1x8x32x32xf32>
    %854 = tensor.empty() : tensor<1x8x32x32xf32>
    %855 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%849 : tensor<1x8x32x32xf32>) outs(%854 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb94(%856: f32, %857: f32):
      linalg.yield %856 : f32
    } -> tensor<1x8x32x32xf32>
    %858 = tensor.collapse_shape %855 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32x32xf32> into tensor<8192xf32>
    %859 = tensor.expand_shape %858 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 32, 32] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<8x32x32xf32>
    %860 = tensor.empty() : tensor<1x8x32x32xf32>
    %861 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%772 : tensor<1x8x32x32xf32>) outs(%860 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb95(%862: f32, %863: f32):
      linalg.yield %862 : f32
    } -> tensor<1x8x32x32xf32>
    %864 = tensor.collapse_shape %861 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x8x32x32xf32> into tensor<8192xf32>
    %865 = tensor.expand_shape %864 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 32, 32] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<8x32x32xf32>
    %866 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} 0.000000e+00 : f32
    %867 = tensor.splat %866 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8x32x32xf32>
    %868 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%859, %865 : tensor<8x32x32xf32>, tensor<8x32x32xf32>) outs(%867 : tensor<8x32x32xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} {
    ^bb96(%869: f32, %870: f32, %871: f32):
      %872 = arith.mulf %869, %870 : f32
      %873 = arith.addf %871, %872 : f32
      linalg.yield %873 : f32
    } -> tensor<8x32x32xf32>
    %874 = tensor.collapse_shape %868 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8x32x32xf32> into tensor<8192xf32>
    %875 = tensor.expand_shape %874 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 32] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x8x32x32xf32>
    %876 = tensor.empty() : tensor<1x32x8x32xf32>
    %877 = linalg.transpose ins(%875:tensor<1x8x32x32xf32>) outs(%876:tensor<1x32x8x32xf32>) permutation = [0, 2, 1, 3]
    %878 = tensor.collapse_shape %877 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<1x32x8x32xf32> into tensor<8192xf32>
    %879 = tensor.expand_shape %878 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 256] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x32x256xf32>
    %880 = tensor.empty() : tensor<1x32x256xf32>
    %881 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%879 : tensor<1x32x256xf32>) outs(%880 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb97(%882: f32, %883: f32):
      %884 = arith.constant 2.000000e+00 : f32
      %885 = math.powf %882, %884 : f32
      linalg.yield %885 : f32
    } -> tensor<1x32x256xf32>
    %886 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} 0.000000e+00 : f32
    %887 = tensor.splat %886 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} : tensor<1x32xf32>
    %888 = linalg.reduce ins(%881:tensor<1x32x256xf32>) outs(%887:tensor<1x32xf32>) dimensions = [2]
    (%889: f32, %890: f32) {
      %891 = arith.addf %889, %890 : f32
      linalg.yield %891 : f32
    }
    %892 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} 2.560000e+02 : f32
    %893 = tensor.splat %892 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} : tensor<1x32xf32>
    %894 = tensor.empty() : tensor<1x32xf32>
    %895 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%888, %893 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%894 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb98(%896: f32, %897: f32, %898: f32):
      %899 = arith.divf %896, %897 : f32
      linalg.yield %899 : f32
    } -> tensor<1x32xf32>
    %900 = tensor.collapse_shape %895 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} : tensor<1x32xf32> into tensor<32xf32>
    %901 = tensor.expand_shape %900 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %902 = arith.constant {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} 1.000000e-05 : f32
    %903 = tensor.splat %902 {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} : tensor<1x32x1xf32>
    %904 = tensor.empty() : tensor<1x32x1xf32>
    %905 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%901, %903 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%904 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb99(%906: f32, %907: f32, %908: f32):
      %909 = arith.addf %906, %907 : f32
      linalg.yield %909 : f32
    } -> tensor<1x32x1xf32>
    %910 = tensor.empty() : tensor<1x32x1xf32>
    %911 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%905 : tensor<1x32x1xf32>) outs(%910 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb100(%912: f32, %913: f32):
      %914 = math.rsqrt %912 : f32
      linalg.yield %914 : f32
    } -> tensor<1x32x1xf32>
    %915 = tensor.empty() : tensor<1x32x256xf32>
    %916 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%879, %911 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%915 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb101(%917: f32, %918: f32, %919: f32):
      %920 = arith.mulf %917, %918 : f32
      linalg.yield %920 : f32
    } -> tensor<1x32x256xf32>
    %921 = tensor.empty() : tensor<1x32x256xf32>
    %922 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%57, %916 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%921 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.attn_sub_norm"} {
    ^bb102(%923: f32, %924: f32, %925: f32):
      %926 = arith.mulf %923, %924 : f32
      linalg.yield %926 : f32
    } -> tensor<1x32x256xf32>
    %927 = tensor.empty() : tensor<1x32x256xf32>
    %928 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%922 : tensor<1x32x256xf32>) outs(%927 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_6", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb103(%929: f32, %930: f32):
      %931 = math.absf %929 : f32
      linalg.yield %931 : f32
    } -> tensor<1x32x256xf32>
    %932 = arith.constant {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} 0xff800000 : f32
    %933 = arith.constant {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} 0 : i64
    %934 = tensor.splat %932 {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<1x32xf32>
    %935 = tensor.splat %933 {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<1x32xi64>
    %936, %937 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%928 : tensor<1x32x256xf32>) outs(%934, %935 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb104(%938: f32, %939: f32, %940: i64):
      %941 = linalg.index 2 : index
      %942 = arith.index_cast %941 : index to i64
      %943 = arith.cmpf ogt, %938, %939 : f32
      %944 = arith.select %943, %938, %939 : f32
      %945 = arith.select %943, %942, %940 : i64
      linalg.yield %944, %945 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %946 = tensor.collapse_shape %936 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %947 = tensor.expand_shape %946 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %948 = tensor.collapse_shape %937 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %949 = tensor.expand_shape %948 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_3", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %950 = tensor.empty() : tensor<1x32x1xf32>
    %951 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%947 : tensor<1x32x1xf32>) outs(%950 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "minmax_12", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb105(%952: f32, %953: f32):
      %954 = arith.constant 1.000000e-05 : f32
      %955 = arith.maximumf %952, %954 : f32
      linalg.yield %955 : f32
    } -> tensor<1x32x1xf32>
    %956 = tensor.empty() : tensor<1x32x1xf32>
    %957 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%951 : tensor<1x32x1xf32>) outs(%956 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_6", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb106(%958: f32, %959: f32):
      %960 = arith.constant 1.000000e+00 : f32
      %961 = arith.divf %960, %958 : f32
      linalg.yield %961 : f32
    } -> tensor<1x32x1xf32>
    %962 = arith.constant {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} 1.270000e+02 : f32
    %963 = tensor.splat %962 {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<1x32x1xf32>
    %964 = tensor.empty() : tensor<1x32x1xf32>
    %965 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%957, %963 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%964 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb107(%966: f32, %967: f32, %968: f32):
      %969 = arith.mulf %966, %967 : f32
      linalg.yield %969 : f32
    } -> tensor<1x32x1xf32>
    %970 = tensor.empty() : tensor<1x32x256xf32>
    %971 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%922, %965 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%970 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb108(%972: f32, %973: f32, %974: f32):
      %975 = arith.mulf %972, %973 : f32
      linalg.yield %975 : f32
    } -> tensor<1x32x256xf32>
    %976 = tensor.empty() : tensor<1x32x256xf32>
    %977 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%971 : tensor<1x32x256xf32>) outs(%976 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_6", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb109(%978: f32, %979: f32):
      %980 = math.roundeven %978 : f32
      linalg.yield %980 : f32
    } -> tensor<1x32x256xf32>
    %981 = tensor.empty() : tensor<1x32x256xf32>
    %982 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%977 : tensor<1x32x256xf32>) outs(%981 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_13", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb110(%983: f32, %984: f32):
      %985 = arith.constant -1.280000e+02 : f32
      %986 = arith.maximumf %983, %985 : f32
      %987 = arith.constant 1.270000e+02 : f32
      %988 = arith.minimumf %986, %987 : f32
      linalg.yield %988 : f32
    } -> tensor<1x32x256xf32>
    %989 = tensor.empty() : tensor<1x32x256xf32>
    %990 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%982, %965 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%989 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_7", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb111(%991: f32, %992: f32, %993: f32):
      %994 = arith.divf %991, %992 : f32
      linalg.yield %994 : f32
    } -> tensor<1x32x256xf32>
    %995 = tensor.empty() : tensor<256x256xf32>
    %996 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%56 : tensor<256x256xf32>) outs(%995 : tensor<256x256xf32>) attrs =  {prov.region_id = "abs_7", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb112(%997: f32, %998: f32):
      %999 = math.absf %997 : f32
      linalg.yield %999 : f32
    } -> tensor<256x256xf32>
    %1000 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} 0.000000e+00 : f32
    %1001 = tensor.splat %1000 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<f32>
    %1002 = linalg.reduce ins(%996:tensor<256x256xf32>) outs(%1001:tensor<f32>) dimensions = [0, 1]
    (%1003: f32, %1004: f32) {
      %1005 = arith.addf %1003, %1004 : f32
      linalg.yield %1005 : f32
    }
    %1006 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} 6.553600e+04 : f32
    %1007 = tensor.splat %1006 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<f32>
    %1008 = tensor.empty() : tensor<f32>
    %1009 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1002, %1007 : tensor<f32>, tensor<f32>) outs(%1008 : tensor<f32>) attrs =  {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb113(%1010: f32, %1011: f32, %1012: f32):
      %1013 = arith.divf %1010, %1011 : f32
      linalg.yield %1013 : f32
    } -> tensor<f32>
    %1014 = tensor.empty() : tensor<f32>
    %1015 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1009 : tensor<f32>) outs(%1014 : tensor<f32>) attrs =  {prov.region_id = "minmax_14", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb114(%1016: f32, %1017: f32):
      %1018 = arith.constant 1.000000e-05 : f32
      %1019 = arith.maximumf %1016, %1018 : f32
      linalg.yield %1019 : f32
    } -> tensor<f32>
    %1020 = tensor.empty() : tensor<f32>
    %1021 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1015 : tensor<f32>) outs(%1020 : tensor<f32>) attrs =  {prov.region_id = "elementwise_7", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb115(%1022: f32, %1023: f32):
      %1024 = arith.constant 1.000000e+00 : f32
      %1025 = arith.divf %1024, %1022 : f32
      linalg.yield %1025 : f32
    } -> tensor<f32>
    %1026 = arith.constant {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} 1.000000e+00 : f32
    %1027 = tensor.splat %1026 {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<f32>
    %1028 = tensor.empty() : tensor<f32>
    %1029 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1021, %1027 : tensor<f32>, tensor<f32>) outs(%1028 : tensor<f32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb116(%1030: f32, %1031: f32, %1032: f32):
      %1033 = arith.mulf %1030, %1031 : f32
      linalg.yield %1033 : f32
    } -> tensor<f32>
    %1034 = tensor.empty() : tensor<256x256xf32>
    %1035 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%56, %1029 : tensor<256x256xf32>, tensor<f32>) outs(%1034 : tensor<256x256xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb117(%1036: f32, %1037: f32, %1038: f32):
      %1039 = arith.mulf %1036, %1037 : f32
      linalg.yield %1039 : f32
    } -> tensor<256x256xf32>
    %1040 = tensor.empty() : tensor<256x256xf32>
    %1041 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1035 : tensor<256x256xf32>) outs(%1040 : tensor<256x256xf32>) attrs =  {prov.region_id = "round_7", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb118(%1042: f32, %1043: f32):
      %1044 = math.roundeven %1042 : f32
      linalg.yield %1044 : f32
    } -> tensor<256x256xf32>
    %1045 = tensor.empty() : tensor<256x256xf32>
    %1046 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1041 : tensor<256x256xf32>) outs(%1045 : tensor<256x256xf32>) attrs =  {prov.region_id = "minmax_15", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb119(%1047: f32, %1048: f32):
      %1049 = arith.constant -1.000000e+00 : f32
      %1050 = arith.maximumf %1047, %1049 : f32
      %1051 = arith.constant 1.000000e+00 : f32
      %1052 = arith.minimumf %1050, %1051 : f32
      linalg.yield %1052 : f32
    } -> tensor<256x256xf32>
    %1053 = tensor.empty() : tensor<256x256xf32>
    %1054 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1046, %1029 : tensor<256x256xf32>, tensor<f32>) outs(%1053 : tensor<256x256xf32>) attrs =  {prov.region_id = "div_8", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} {
    ^bb120(%1055: f32, %1056: f32, %1057: f32):
      %1058 = arith.divf %1055, %1056 : f32
      linalg.yield %1058 : f32
    } -> tensor<256x256xf32>
    %1059 = tensor.empty() : tensor<256x256xf32>
    %1060 = linalg.transpose ins(%1054:tensor<256x256xf32>) outs(%1059:tensor<256x256xf32>) permutation = [1, 0]
    %1061 = tensor.collapse_shape %990 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %1062 = tensor.expand_shape %1061 [[0 : i64, 1 : i64]] output_shape [32, 256] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<8192xf32> into tensor<32x256xf32>
    %1063 = tensor.empty() : tensor<32x256xf32>
    %1064 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1065 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1064 : f32) outs(%1063 : tensor<32x256xf32>) -> tensor<32x256xf32>
    %1066 = linalg.matmul {prov.region_id = "matmul_5", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj", prov.transposed_b = "true"} ins(%1062, %1060 : tensor<32x256xf32>, tensor<256x256xf32>) outs(%1065 : tensor<32x256xf32>) -> tensor<32x256xf32>
    %1067 = tensor.collapse_shape %1066 [[0 : i64, 1 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<32x256xf32> into tensor<8192xf32>
    %1068 = tensor.expand_shape %1067 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 256] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.self_attn.o_proj"} : tensor<8192xf32> into tensor<1x32x256xf32>
    %1069 = tensor.empty() : tensor<1x32x256xf32>
    %1070 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84, %1068 : tensor<1x32x256xf32>, tensor<1x32x256xf32>) outs(%1069 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0"} {
    ^bb121(%1071: f32, %1072: f32, %1073: f32):
      %1074 = arith.addf %1071, %1072 : f32
      linalg.yield %1074 : f32
    } -> tensor<1x32x256xf32>
    %1075 = tensor.empty() : tensor<1x32x256xf32>
    %1076 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1070 : tensor<1x32x256xf32>) outs(%1075 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb122(%1077: f32, %1078: f32):
      %1079 = arith.constant 2.000000e+00 : f32
      %1080 = math.powf %1077, %1079 : f32
      linalg.yield %1080 : f32
    } -> tensor<1x32x256xf32>
    %1081 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} 0.000000e+00 : f32
    %1082 = tensor.splat %1081 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<1x32xf32>
    %1083 = linalg.reduce ins(%1076:tensor<1x32x256xf32>) outs(%1082:tensor<1x32xf32>) dimensions = [2]
    (%1084: f32, %1085: f32) {
      %1086 = arith.addf %1084, %1085 : f32
      linalg.yield %1086 : f32
    }
    %1087 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} 2.560000e+02 : f32
    %1088 = tensor.splat %1087 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<1x32xf32>
    %1089 = tensor.empty() : tensor<1x32xf32>
    %1090 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1083, %1088 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1089 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb123(%1091: f32, %1092: f32, %1093: f32):
      %1094 = arith.divf %1091, %1092 : f32
      linalg.yield %1094 : f32
    } -> tensor<1x32xf32>
    %1095 = tensor.collapse_shape %1090 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %1096 = tensor.expand_shape %1095 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1097 = arith.constant {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} 1.000000e-05 : f32
    %1098 = tensor.splat %1097 {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} : tensor<1x32x1xf32>
    %1099 = tensor.empty() : tensor<1x32x1xf32>
    %1100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1096, %1098 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1099 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb124(%1101: f32, %1102: f32, %1103: f32):
      %1104 = arith.addf %1101, %1102 : f32
      linalg.yield %1104 : f32
    } -> tensor<1x32x1xf32>
    %1105 = tensor.empty() : tensor<1x32x1xf32>
    %1106 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1100 : tensor<1x32x1xf32>) outs(%1105 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb125(%1107: f32, %1108: f32):
      %1109 = math.rsqrt %1107 : f32
      linalg.yield %1109 : f32
    } -> tensor<1x32x1xf32>
    %1110 = tensor.empty() : tensor<1x32x256xf32>
    %1111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1070, %1106 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1110 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb126(%1112: f32, %1113: f32, %1114: f32):
      %1115 = arith.mulf %1112, %1113 : f32
      linalg.yield %1115 : f32
    } -> tensor<1x32x256xf32>
    %1116 = tensor.empty() : tensor<1x32x256xf32>
    %1117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63, %1111 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%1116 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.post_attention_layernorm"} {
    ^bb127(%1118: f32, %1119: f32, %1120: f32):
      %1121 = arith.mulf %1118, %1119 : f32
      linalg.yield %1121 : f32
    } -> tensor<1x32x256xf32>
    %1122 = tensor.empty() : tensor<1x32x256xf32>
    %1123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1117 : tensor<1x32x256xf32>) outs(%1122 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_8", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb128(%1124: f32, %1125: f32):
      %1126 = math.absf %1124 : f32
      linalg.yield %1126 : f32
    } -> tensor<1x32x256xf32>
    %1127 = arith.constant {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} 0xff800000 : f32
    %1128 = arith.constant {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} 0 : i64
    %1129 = tensor.splat %1127 {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<1x32xf32>
    %1130 = tensor.splat %1128 {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<1x32xi64>
    %1131, %1132 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1123 : tensor<1x32x256xf32>) outs(%1129, %1130 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb129(%1133: f32, %1134: f32, %1135: i64):
      %1136 = linalg.index 2 : index
      %1137 = arith.index_cast %1136 : index to i64
      %1138 = arith.cmpf ogt, %1133, %1134 : f32
      %1139 = arith.select %1138, %1133, %1134 : f32
      %1140 = arith.select %1138, %1137, %1135 : i64
      linalg.yield %1139, %1140 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1141 = tensor.collapse_shape %1131 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1142 = tensor.expand_shape %1141 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1143 = tensor.collapse_shape %1132 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1144 = tensor.expand_shape %1143 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_4", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1145 = tensor.empty() : tensor<1x32x1xf32>
    %1146 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1142 : tensor<1x32x1xf32>) outs(%1145 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "minmax_16", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb130(%1147: f32, %1148: f32):
      %1149 = arith.constant 1.000000e-05 : f32
      %1150 = arith.maximumf %1147, %1149 : f32
      linalg.yield %1150 : f32
    } -> tensor<1x32x1xf32>
    %1151 = tensor.empty() : tensor<1x32x1xf32>
    %1152 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1146 : tensor<1x32x1xf32>) outs(%1151 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_8", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb131(%1153: f32, %1154: f32):
      %1155 = arith.constant 1.000000e+00 : f32
      %1156 = arith.divf %1155, %1153 : f32
      linalg.yield %1156 : f32
    } -> tensor<1x32x1xf32>
    %1157 = arith.constant {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} 1.270000e+02 : f32
    %1158 = tensor.splat %1157 {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<1x32x1xf32>
    %1159 = tensor.empty() : tensor<1x32x1xf32>
    %1160 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1152, %1158 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1159 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb132(%1161: f32, %1162: f32, %1163: f32):
      %1164 = arith.mulf %1161, %1162 : f32
      linalg.yield %1164 : f32
    } -> tensor<1x32x1xf32>
    %1165 = tensor.empty() : tensor<1x32x256xf32>
    %1166 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1117, %1160 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1165 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb133(%1167: f32, %1168: f32, %1169: f32):
      %1170 = arith.mulf %1167, %1168 : f32
      linalg.yield %1170 : f32
    } -> tensor<1x32x256xf32>
    %1171 = tensor.empty() : tensor<1x32x256xf32>
    %1172 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1166 : tensor<1x32x256xf32>) outs(%1171 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_8", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb134(%1173: f32, %1174: f32):
      %1175 = math.roundeven %1173 : f32
      linalg.yield %1175 : f32
    } -> tensor<1x32x256xf32>
    %1176 = tensor.empty() : tensor<1x32x256xf32>
    %1177 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1172 : tensor<1x32x256xf32>) outs(%1176 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_17", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb135(%1178: f32, %1179: f32):
      %1180 = arith.constant -1.280000e+02 : f32
      %1181 = arith.maximumf %1178, %1180 : f32
      %1182 = arith.constant 1.270000e+02 : f32
      %1183 = arith.minimumf %1181, %1182 : f32
      linalg.yield %1183 : f32
    } -> tensor<1x32x256xf32>
    %1184 = tensor.empty() : tensor<1x32x256xf32>
    %1185 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1177, %1160 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1184 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_9", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb136(%1186: f32, %1187: f32, %1188: f32):
      %1189 = arith.divf %1186, %1187 : f32
      linalg.yield %1189 : f32
    } -> tensor<1x32x256xf32>
    %1190 = tensor.empty() : tensor<512x256xf32>
    %1191 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%58 : tensor<512x256xf32>) outs(%1190 : tensor<512x256xf32>) attrs =  {prov.region_id = "abs_9", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb137(%1192: f32, %1193: f32):
      %1194 = math.absf %1192 : f32
      linalg.yield %1194 : f32
    } -> tensor<512x256xf32>
    %1195 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} 0.000000e+00 : f32
    %1196 = tensor.splat %1195 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<f32>
    %1197 = linalg.reduce ins(%1191:tensor<512x256xf32>) outs(%1196:tensor<f32>) dimensions = [0, 1]
    (%1198: f32, %1199: f32) {
      %1200 = arith.addf %1198, %1199 : f32
      linalg.yield %1200 : f32
    }
    %1201 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} 1.310720e+05 : f32
    %1202 = tensor.splat %1201 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<f32>
    %1203 = tensor.empty() : tensor<f32>
    %1204 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1197, %1202 : tensor<f32>, tensor<f32>) outs(%1203 : tensor<f32>) attrs =  {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb138(%1205: f32, %1206: f32, %1207: f32):
      %1208 = arith.divf %1205, %1206 : f32
      linalg.yield %1208 : f32
    } -> tensor<f32>
    %1209 = tensor.empty() : tensor<f32>
    %1210 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1204 : tensor<f32>) outs(%1209 : tensor<f32>) attrs =  {prov.region_id = "minmax_18", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb139(%1211: f32, %1212: f32):
      %1213 = arith.constant 1.000000e-05 : f32
      %1214 = arith.maximumf %1211, %1213 : f32
      linalg.yield %1214 : f32
    } -> tensor<f32>
    %1215 = tensor.empty() : tensor<f32>
    %1216 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1210 : tensor<f32>) outs(%1215 : tensor<f32>) attrs =  {prov.region_id = "elementwise_9", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb140(%1217: f32, %1218: f32):
      %1219 = arith.constant 1.000000e+00 : f32
      %1220 = arith.divf %1219, %1217 : f32
      linalg.yield %1220 : f32
    } -> tensor<f32>
    %1221 = arith.constant {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} 1.000000e+00 : f32
    %1222 = tensor.splat %1221 {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<f32>
    %1223 = tensor.empty() : tensor<f32>
    %1224 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1216, %1222 : tensor<f32>, tensor<f32>) outs(%1223 : tensor<f32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb141(%1225: f32, %1226: f32, %1227: f32):
      %1228 = arith.mulf %1225, %1226 : f32
      linalg.yield %1228 : f32
    } -> tensor<f32>
    %1229 = tensor.empty() : tensor<512x256xf32>
    %1230 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%58, %1224 : tensor<512x256xf32>, tensor<f32>) outs(%1229 : tensor<512x256xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb142(%1231: f32, %1232: f32, %1233: f32):
      %1234 = arith.mulf %1231, %1232 : f32
      linalg.yield %1234 : f32
    } -> tensor<512x256xf32>
    %1235 = tensor.empty() : tensor<512x256xf32>
    %1236 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1230 : tensor<512x256xf32>) outs(%1235 : tensor<512x256xf32>) attrs =  {prov.region_id = "round_9", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb143(%1237: f32, %1238: f32):
      %1239 = math.roundeven %1237 : f32
      linalg.yield %1239 : f32
    } -> tensor<512x256xf32>
    %1240 = tensor.empty() : tensor<512x256xf32>
    %1241 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1236 : tensor<512x256xf32>) outs(%1240 : tensor<512x256xf32>) attrs =  {prov.region_id = "minmax_19", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb144(%1242: f32, %1243: f32):
      %1244 = arith.constant -1.000000e+00 : f32
      %1245 = arith.maximumf %1242, %1244 : f32
      %1246 = arith.constant 1.000000e+00 : f32
      %1247 = arith.minimumf %1245, %1246 : f32
      linalg.yield %1247 : f32
    } -> tensor<512x256xf32>
    %1248 = tensor.empty() : tensor<512x256xf32>
    %1249 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1241, %1224 : tensor<512x256xf32>, tensor<f32>) outs(%1248 : tensor<512x256xf32>) attrs =  {prov.region_id = "div_10", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} {
    ^bb145(%1250: f32, %1251: f32, %1252: f32):
      %1253 = arith.divf %1250, %1251 : f32
      linalg.yield %1253 : f32
    } -> tensor<512x256xf32>
    %1254 = tensor.empty() : tensor<256x512xf32>
    %1255 = linalg.transpose ins(%1249:tensor<512x256xf32>) outs(%1254:tensor<256x512xf32>) permutation = [1, 0]
    %1256 = tensor.collapse_shape %1185 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %1257 = tensor.expand_shape %1256 [[0 : i64, 1 : i64]] output_shape [32, 256] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<8192xf32> into tensor<32x256xf32>
    %1258 = tensor.empty() : tensor<32x512xf32>
    %1259 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1260 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1259 : f32) outs(%1258 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %1261 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj", prov.transposed_b = "true"} ins(%1257, %1255 : tensor<32x256xf32>, tensor<256x512xf32>) outs(%1260 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %1262 = tensor.collapse_shape %1261 [[0 : i64, 1 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<32x512xf32> into tensor<16384xf32>
    %1263 = tensor.expand_shape %1262 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 512] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.gate_proj"} : tensor<16384xf32> into tensor<1x32x512xf32>
    %1264 = tensor.empty() : tensor<1x32x512xf32>
    %1265 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1263 : tensor<1x32x512xf32>) outs(%1264 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "minmax_20", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp"} {
    ^bb146(%1266: f32, %1267: f32):
      %1268 = arith.constant 0.000000e+00 : f32
      %1269 = arith.maximumf %1266, %1268 : f32
      linalg.yield %1269 : f32
    } -> tensor<1x32x512xf32>
    %1270 = tensor.empty() : tensor<1x32x512xf32>
    %1271 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1265 : tensor<1x32x512xf32>) outs(%1270 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp"} {
    ^bb147(%1272: f32, %1273: f32):
      %1274 = arith.constant 2.000000e+00 : f32
      %1275 = math.powf %1272, %1274 : f32
      linalg.yield %1275 : f32
    } -> tensor<1x32x512xf32>
    %1276 = tensor.empty() : tensor<1x32x256xf32>
    %1277 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1117 : tensor<1x32x256xf32>) outs(%1276 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_10", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb148(%1278: f32, %1279: f32):
      %1280 = math.absf %1278 : f32
      linalg.yield %1280 : f32
    } -> tensor<1x32x256xf32>
    %1281 = arith.constant {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} 0xff800000 : f32
    %1282 = arith.constant {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} 0 : i64
    %1283 = tensor.splat %1281 {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<1x32xf32>
    %1284 = tensor.splat %1282 {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<1x32xi64>
    %1285, %1286 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1277 : tensor<1x32x256xf32>) outs(%1283, %1284 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb149(%1287: f32, %1288: f32, %1289: i64):
      %1290 = linalg.index 2 : index
      %1291 = arith.index_cast %1290 : index to i64
      %1292 = arith.cmpf ogt, %1287, %1288 : f32
      %1293 = arith.select %1292, %1287, %1288 : f32
      %1294 = arith.select %1292, %1291, %1289 : i64
      linalg.yield %1293, %1294 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1295 = tensor.collapse_shape %1285 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1296 = tensor.expand_shape %1295 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1297 = tensor.collapse_shape %1286 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1298 = tensor.expand_shape %1297 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_5", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1299 = tensor.empty() : tensor<1x32x1xf32>
    %1300 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1296 : tensor<1x32x1xf32>) outs(%1299 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "minmax_21", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb150(%1301: f32, %1302: f32):
      %1303 = arith.constant 1.000000e-05 : f32
      %1304 = arith.maximumf %1301, %1303 : f32
      linalg.yield %1304 : f32
    } -> tensor<1x32x1xf32>
    %1305 = tensor.empty() : tensor<1x32x1xf32>
    %1306 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1300 : tensor<1x32x1xf32>) outs(%1305 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_10", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb151(%1307: f32, %1308: f32):
      %1309 = arith.constant 1.000000e+00 : f32
      %1310 = arith.divf %1309, %1307 : f32
      linalg.yield %1310 : f32
    } -> tensor<1x32x1xf32>
    %1311 = arith.constant {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} 1.270000e+02 : f32
    %1312 = tensor.splat %1311 {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<1x32x1xf32>
    %1313 = tensor.empty() : tensor<1x32x1xf32>
    %1314 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1306, %1312 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1313 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb152(%1315: f32, %1316: f32, %1317: f32):
      %1318 = arith.mulf %1315, %1316 : f32
      linalg.yield %1318 : f32
    } -> tensor<1x32x1xf32>
    %1319 = tensor.empty() : tensor<1x32x256xf32>
    %1320 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1117, %1314 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1319 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb153(%1321: f32, %1322: f32, %1323: f32):
      %1324 = arith.mulf %1321, %1322 : f32
      linalg.yield %1324 : f32
    } -> tensor<1x32x256xf32>
    %1325 = tensor.empty() : tensor<1x32x256xf32>
    %1326 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1320 : tensor<1x32x256xf32>) outs(%1325 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_10", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb154(%1327: f32, %1328: f32):
      %1329 = math.roundeven %1327 : f32
      linalg.yield %1329 : f32
    } -> tensor<1x32x256xf32>
    %1330 = tensor.empty() : tensor<1x32x256xf32>
    %1331 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1326 : tensor<1x32x256xf32>) outs(%1330 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_22", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb155(%1332: f32, %1333: f32):
      %1334 = arith.constant -1.280000e+02 : f32
      %1335 = arith.maximumf %1332, %1334 : f32
      %1336 = arith.constant 1.270000e+02 : f32
      %1337 = arith.minimumf %1335, %1336 : f32
      linalg.yield %1337 : f32
    } -> tensor<1x32x256xf32>
    %1338 = tensor.empty() : tensor<1x32x256xf32>
    %1339 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1331, %1314 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1338 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_11", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb156(%1340: f32, %1341: f32, %1342: f32):
      %1343 = arith.divf %1340, %1341 : f32
      linalg.yield %1343 : f32
    } -> tensor<1x32x256xf32>
    %1344 = tensor.empty() : tensor<512x256xf32>
    %1345 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%59 : tensor<512x256xf32>) outs(%1344 : tensor<512x256xf32>) attrs =  {prov.region_id = "abs_11", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb157(%1346: f32, %1347: f32):
      %1348 = math.absf %1346 : f32
      linalg.yield %1348 : f32
    } -> tensor<512x256xf32>
    %1349 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} 0.000000e+00 : f32
    %1350 = tensor.splat %1349 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<f32>
    %1351 = linalg.reduce ins(%1345:tensor<512x256xf32>) outs(%1350:tensor<f32>) dimensions = [0, 1]
    (%1352: f32, %1353: f32) {
      %1354 = arith.addf %1352, %1353 : f32
      linalg.yield %1354 : f32
    }
    %1355 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} 1.310720e+05 : f32
    %1356 = tensor.splat %1355 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<f32>
    %1357 = tensor.empty() : tensor<f32>
    %1358 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1351, %1356 : tensor<f32>, tensor<f32>) outs(%1357 : tensor<f32>) attrs =  {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb158(%1359: f32, %1360: f32, %1361: f32):
      %1362 = arith.divf %1359, %1360 : f32
      linalg.yield %1362 : f32
    } -> tensor<f32>
    %1363 = tensor.empty() : tensor<f32>
    %1364 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1358 : tensor<f32>) outs(%1363 : tensor<f32>) attrs =  {prov.region_id = "minmax_23", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb159(%1365: f32, %1366: f32):
      %1367 = arith.constant 1.000000e-05 : f32
      %1368 = arith.maximumf %1365, %1367 : f32
      linalg.yield %1368 : f32
    } -> tensor<f32>
    %1369 = tensor.empty() : tensor<f32>
    %1370 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1364 : tensor<f32>) outs(%1369 : tensor<f32>) attrs =  {prov.region_id = "elementwise_11", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb160(%1371: f32, %1372: f32):
      %1373 = arith.constant 1.000000e+00 : f32
      %1374 = arith.divf %1373, %1371 : f32
      linalg.yield %1374 : f32
    } -> tensor<f32>
    %1375 = arith.constant {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} 1.000000e+00 : f32
    %1376 = tensor.splat %1375 {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<f32>
    %1377 = tensor.empty() : tensor<f32>
    %1378 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1370, %1376 : tensor<f32>, tensor<f32>) outs(%1377 : tensor<f32>) attrs =  {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb161(%1379: f32, %1380: f32, %1381: f32):
      %1382 = arith.mulf %1379, %1380 : f32
      linalg.yield %1382 : f32
    } -> tensor<f32>
    %1383 = tensor.empty() : tensor<512x256xf32>
    %1384 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%59, %1378 : tensor<512x256xf32>, tensor<f32>) outs(%1383 : tensor<512x256xf32>) attrs =  {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb162(%1385: f32, %1386: f32, %1387: f32):
      %1388 = arith.mulf %1385, %1386 : f32
      linalg.yield %1388 : f32
    } -> tensor<512x256xf32>
    %1389 = tensor.empty() : tensor<512x256xf32>
    %1390 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1384 : tensor<512x256xf32>) outs(%1389 : tensor<512x256xf32>) attrs =  {prov.region_id = "round_11", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb163(%1391: f32, %1392: f32):
      %1393 = math.roundeven %1391 : f32
      linalg.yield %1393 : f32
    } -> tensor<512x256xf32>
    %1394 = tensor.empty() : tensor<512x256xf32>
    %1395 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1390 : tensor<512x256xf32>) outs(%1394 : tensor<512x256xf32>) attrs =  {prov.region_id = "minmax_24", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb164(%1396: f32, %1397: f32):
      %1398 = arith.constant -1.000000e+00 : f32
      %1399 = arith.maximumf %1396, %1398 : f32
      %1400 = arith.constant 1.000000e+00 : f32
      %1401 = arith.minimumf %1399, %1400 : f32
      linalg.yield %1401 : f32
    } -> tensor<512x256xf32>
    %1402 = tensor.empty() : tensor<512x256xf32>
    %1403 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1395, %1378 : tensor<512x256xf32>, tensor<f32>) outs(%1402 : tensor<512x256xf32>) attrs =  {prov.region_id = "div_12", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} {
    ^bb165(%1404: f32, %1405: f32, %1406: f32):
      %1407 = arith.divf %1404, %1405 : f32
      linalg.yield %1407 : f32
    } -> tensor<512x256xf32>
    %1408 = tensor.empty() : tensor<256x512xf32>
    %1409 = linalg.transpose ins(%1403:tensor<512x256xf32>) outs(%1408:tensor<256x512xf32>) permutation = [1, 0]
    %1410 = tensor.collapse_shape %1339 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %1411 = tensor.expand_shape %1410 [[0 : i64, 1 : i64]] output_shape [32, 256] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<8192xf32> into tensor<32x256xf32>
    %1412 = tensor.empty() : tensor<32x512xf32>
    %1413 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1414 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1413 : f32) outs(%1412 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %1415 = linalg.matmul {prov.region_id = "matmul_7", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj", prov.transposed_b = "true"} ins(%1411, %1409 : tensor<32x256xf32>, tensor<256x512xf32>) outs(%1414 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %1416 = tensor.collapse_shape %1415 [[0 : i64, 1 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<32x512xf32> into tensor<16384xf32>
    %1417 = tensor.expand_shape %1416 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 512] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.up_proj"} : tensor<16384xf32> into tensor<1x32x512xf32>
    %1418 = tensor.empty() : tensor<1x32x512xf32>
    %1419 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1271, %1417 : tensor<1x32x512xf32>, tensor<1x32x512xf32>) outs(%1418 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_34", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp"} {
    ^bb166(%1420: f32, %1421: f32, %1422: f32):
      %1423 = arith.mulf %1420, %1421 : f32
      linalg.yield %1423 : f32
    } -> tensor<1x32x512xf32>
    %1424 = tensor.empty() : tensor<1x32x512xf32>
    %1425 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1419 : tensor<1x32x512xf32>) outs(%1424 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb167(%1426: f32, %1427: f32):
      %1428 = arith.constant 2.000000e+00 : f32
      %1429 = math.powf %1426, %1428 : f32
      linalg.yield %1429 : f32
    } -> tensor<1x32x512xf32>
    %1430 = arith.constant {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} 0.000000e+00 : f32
    %1431 = tensor.splat %1430 {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} : tensor<1x32xf32>
    %1432 = linalg.reduce ins(%1425:tensor<1x32x512xf32>) outs(%1431:tensor<1x32xf32>) dimensions = [2]
    (%1433: f32, %1434: f32) {
      %1435 = arith.addf %1433, %1434 : f32
      linalg.yield %1435 : f32
    }
    %1436 = arith.constant {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} 5.120000e+02 : f32
    %1437 = tensor.splat %1436 {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} : tensor<1x32xf32>
    %1438 = tensor.empty() : tensor<1x32xf32>
    %1439 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1432, %1437 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1438 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb168(%1440: f32, %1441: f32, %1442: f32):
      %1443 = arith.divf %1440, %1441 : f32
      linalg.yield %1443 : f32
    } -> tensor<1x32xf32>
    %1444 = tensor.collapse_shape %1439 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} : tensor<1x32xf32> into tensor<32xf32>
    %1445 = tensor.expand_shape %1444 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1446 = arith.constant {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} 1.000000e-05 : f32
    %1447 = tensor.splat %1446 {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} : tensor<1x32x1xf32>
    %1448 = tensor.empty() : tensor<1x32x1xf32>
    %1449 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1445, %1447 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1448 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb169(%1450: f32, %1451: f32, %1452: f32):
      %1453 = arith.addf %1450, %1451 : f32
      linalg.yield %1453 : f32
    } -> tensor<1x32x1xf32>
    %1454 = tensor.empty() : tensor<1x32x1xf32>
    %1455 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1449 : tensor<1x32x1xf32>) outs(%1454 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb170(%1456: f32, %1457: f32):
      %1458 = math.rsqrt %1456 : f32
      linalg.yield %1458 : f32
    } -> tensor<1x32x1xf32>
    %1459 = tensor.empty() : tensor<1x32x512xf32>
    %1460 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1419, %1455 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%1459 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_35", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb171(%1461: f32, %1462: f32, %1463: f32):
      %1464 = arith.mulf %1461, %1462 : f32
      linalg.yield %1464 : f32
    } -> tensor<1x32x512xf32>
    %1465 = tensor.empty() : tensor<1x32x512xf32>
    %1466 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %1460 : tensor<512xf32>, tensor<1x32x512xf32>) outs(%1465 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_36", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.ffn_sub_norm"} {
    ^bb172(%1467: f32, %1468: f32, %1469: f32):
      %1470 = arith.mulf %1467, %1468 : f32
      linalg.yield %1470 : f32
    } -> tensor<1x32x512xf32>
    %1471 = tensor.empty() : tensor<1x32x512xf32>
    %1472 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1466 : tensor<1x32x512xf32>) outs(%1471 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "abs_12", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb173(%1473: f32, %1474: f32):
      %1475 = math.absf %1473 : f32
      linalg.yield %1475 : f32
    } -> tensor<1x32x512xf32>
    %1476 = arith.constant {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} 0xff800000 : f32
    %1477 = arith.constant {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} 0 : i64
    %1478 = tensor.splat %1476 {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<1x32xf32>
    %1479 = tensor.splat %1477 {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<1x32xi64>
    %1480, %1481 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1472 : tensor<1x32x512xf32>) outs(%1478, %1479 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb174(%1482: f32, %1483: f32, %1484: i64):
      %1485 = linalg.index 2 : index
      %1486 = arith.index_cast %1485 : index to i64
      %1487 = arith.cmpf ogt, %1482, %1483 : f32
      %1488 = arith.select %1487, %1482, %1483 : f32
      %1489 = arith.select %1487, %1486, %1484 : i64
      linalg.yield %1488, %1489 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1490 = tensor.collapse_shape %1480 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1491 = tensor.expand_shape %1490 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1492 = tensor.collapse_shape %1481 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1493 = tensor.expand_shape %1492 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_6", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1494 = tensor.empty() : tensor<1x32x1xf32>
    %1495 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1491 : tensor<1x32x1xf32>) outs(%1494 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "minmax_25", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb175(%1496: f32, %1497: f32):
      %1498 = arith.constant 1.000000e-05 : f32
      %1499 = arith.maximumf %1496, %1498 : f32
      linalg.yield %1499 : f32
    } -> tensor<1x32x1xf32>
    %1500 = tensor.empty() : tensor<1x32x1xf32>
    %1501 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1495 : tensor<1x32x1xf32>) outs(%1500 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_12", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb176(%1502: f32, %1503: f32):
      %1504 = arith.constant 1.000000e+00 : f32
      %1505 = arith.divf %1504, %1502 : f32
      linalg.yield %1505 : f32
    } -> tensor<1x32x1xf32>
    %1506 = arith.constant {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} 1.270000e+02 : f32
    %1507 = tensor.splat %1506 {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<1x32x1xf32>
    %1508 = tensor.empty() : tensor<1x32x1xf32>
    %1509 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1501, %1507 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1508 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb177(%1510: f32, %1511: f32, %1512: f32):
      %1513 = arith.mulf %1510, %1511 : f32
      linalg.yield %1513 : f32
    } -> tensor<1x32x1xf32>
    %1514 = tensor.empty() : tensor<1x32x512xf32>
    %1515 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1466, %1509 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%1514 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_38", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb178(%1516: f32, %1517: f32, %1518: f32):
      %1519 = arith.mulf %1516, %1517 : f32
      linalg.yield %1519 : f32
    } -> tensor<1x32x512xf32>
    %1520 = tensor.empty() : tensor<1x32x512xf32>
    %1521 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1515 : tensor<1x32x512xf32>) outs(%1520 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "round_12", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb179(%1522: f32, %1523: f32):
      %1524 = math.roundeven %1522 : f32
      linalg.yield %1524 : f32
    } -> tensor<1x32x512xf32>
    %1525 = tensor.empty() : tensor<1x32x512xf32>
    %1526 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1521 : tensor<1x32x512xf32>) outs(%1525 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "minmax_26", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb180(%1527: f32, %1528: f32):
      %1529 = arith.constant -1.280000e+02 : f32
      %1530 = arith.maximumf %1527, %1529 : f32
      %1531 = arith.constant 1.270000e+02 : f32
      %1532 = arith.minimumf %1530, %1531 : f32
      linalg.yield %1532 : f32
    } -> tensor<1x32x512xf32>
    %1533 = tensor.empty() : tensor<1x32x512xf32>
    %1534 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1526, %1509 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%1533 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "div_13", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb181(%1535: f32, %1536: f32, %1537: f32):
      %1538 = arith.divf %1535, %1536 : f32
      linalg.yield %1538 : f32
    } -> tensor<1x32x512xf32>
    %1539 = tensor.empty() : tensor<256x512xf32>
    %1540 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%60 : tensor<256x512xf32>) outs(%1539 : tensor<256x512xf32>) attrs =  {prov.region_id = "abs_13", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb182(%1541: f32, %1542: f32):
      %1543 = math.absf %1541 : f32
      linalg.yield %1543 : f32
    } -> tensor<256x512xf32>
    %1544 = arith.constant {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} 0.000000e+00 : f32
    %1545 = tensor.splat %1544 {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<f32>
    %1546 = linalg.reduce ins(%1540:tensor<256x512xf32>) outs(%1545:tensor<f32>) dimensions = [0, 1]
    (%1547: f32, %1548: f32) {
      %1549 = arith.addf %1547, %1548 : f32
      linalg.yield %1549 : f32
    }
    %1550 = arith.constant {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} 1.310720e+05 : f32
    %1551 = tensor.splat %1550 {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<f32>
    %1552 = tensor.empty() : tensor<f32>
    %1553 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1546, %1551 : tensor<f32>, tensor<f32>) outs(%1552 : tensor<f32>) attrs =  {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb183(%1554: f32, %1555: f32, %1556: f32):
      %1557 = arith.divf %1554, %1555 : f32
      linalg.yield %1557 : f32
    } -> tensor<f32>
    %1558 = tensor.empty() : tensor<f32>
    %1559 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1553 : tensor<f32>) outs(%1558 : tensor<f32>) attrs =  {prov.region_id = "minmax_27", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb184(%1560: f32, %1561: f32):
      %1562 = arith.constant 1.000000e-05 : f32
      %1563 = arith.maximumf %1560, %1562 : f32
      linalg.yield %1563 : f32
    } -> tensor<f32>
    %1564 = tensor.empty() : tensor<f32>
    %1565 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1559 : tensor<f32>) outs(%1564 : tensor<f32>) attrs =  {prov.region_id = "elementwise_13", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb185(%1566: f32, %1567: f32):
      %1568 = arith.constant 1.000000e+00 : f32
      %1569 = arith.divf %1568, %1566 : f32
      linalg.yield %1569 : f32
    } -> tensor<f32>
    %1570 = arith.constant {prov.region_id = "mul_39", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} 1.000000e+00 : f32
    %1571 = tensor.splat %1570 {prov.region_id = "mul_39", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<f32>
    %1572 = tensor.empty() : tensor<f32>
    %1573 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1565, %1571 : tensor<f32>, tensor<f32>) outs(%1572 : tensor<f32>) attrs =  {prov.region_id = "mul_39", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb186(%1574: f32, %1575: f32, %1576: f32):
      %1577 = arith.mulf %1574, %1575 : f32
      linalg.yield %1577 : f32
    } -> tensor<f32>
    %1578 = tensor.empty() : tensor<256x512xf32>
    %1579 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%60, %1573 : tensor<256x512xf32>, tensor<f32>) outs(%1578 : tensor<256x512xf32>) attrs =  {prov.region_id = "mul_40", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb187(%1580: f32, %1581: f32, %1582: f32):
      %1583 = arith.mulf %1580, %1581 : f32
      linalg.yield %1583 : f32
    } -> tensor<256x512xf32>
    %1584 = tensor.empty() : tensor<256x512xf32>
    %1585 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1579 : tensor<256x512xf32>) outs(%1584 : tensor<256x512xf32>) attrs =  {prov.region_id = "round_13", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb188(%1586: f32, %1587: f32):
      %1588 = math.roundeven %1586 : f32
      linalg.yield %1588 : f32
    } -> tensor<256x512xf32>
    %1589 = tensor.empty() : tensor<256x512xf32>
    %1590 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1585 : tensor<256x512xf32>) outs(%1589 : tensor<256x512xf32>) attrs =  {prov.region_id = "minmax_28", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb189(%1591: f32, %1592: f32):
      %1593 = arith.constant -1.000000e+00 : f32
      %1594 = arith.maximumf %1591, %1593 : f32
      %1595 = arith.constant 1.000000e+00 : f32
      %1596 = arith.minimumf %1594, %1595 : f32
      linalg.yield %1596 : f32
    } -> tensor<256x512xf32>
    %1597 = tensor.empty() : tensor<256x512xf32>
    %1598 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1590, %1573 : tensor<256x512xf32>, tensor<f32>) outs(%1597 : tensor<256x512xf32>) attrs =  {prov.region_id = "div_14", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} {
    ^bb190(%1599: f32, %1600: f32, %1601: f32):
      %1602 = arith.divf %1599, %1600 : f32
      linalg.yield %1602 : f32
    } -> tensor<256x512xf32>
    %1603 = tensor.empty() : tensor<512x256xf32>
    %1604 = linalg.transpose ins(%1598:tensor<256x512xf32>) outs(%1603:tensor<512x256xf32>) permutation = [1, 0]
    %1605 = tensor.collapse_shape %1534 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<1x32x512xf32> into tensor<16384xf32>
    %1606 = tensor.expand_shape %1605 [[0 : i64, 1 : i64]] output_shape [32, 512] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<16384xf32> into tensor<32x512xf32>
    %1607 = tensor.empty() : tensor<32x256xf32>
    %1608 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1609 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1608 : f32) outs(%1607 : tensor<32x256xf32>) -> tensor<32x256xf32>
    %1610 = linalg.matmul {prov.region_id = "matmul_8", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj", prov.transposed_b = "true"} ins(%1606, %1604 : tensor<32x512xf32>, tensor<512x256xf32>) outs(%1609 : tensor<32x256xf32>) -> tensor<32x256xf32>
    %1611 = tensor.collapse_shape %1610 [[0 : i64, 1 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<32x256xf32> into tensor<8192xf32>
    %1612 = tensor.expand_shape %1611 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 256] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0.mlp.down_proj"} : tensor<8192xf32> into tensor<1x32x256xf32>
    %1613 = tensor.empty() : tensor<1x32x256xf32>
    %1614 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1070, %1612 : tensor<1x32x256xf32>, tensor<1x32x256xf32>) outs(%1613 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.0"} {
    ^bb191(%1615: f32, %1616: f32, %1617: f32):
      %1618 = arith.addf %1615, %1616 : f32
      linalg.yield %1618 : f32
    } -> tensor<1x32x256xf32>
    %1619 = tensor.empty() : tensor<1x32x256xf32>
    %1620 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1614 : tensor<1x32x256xf32>) outs(%1619 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb192(%1621: f32, %1622: f32):
      %1623 = arith.constant 2.000000e+00 : f32
      %1624 = math.powf %1621, %1623 : f32
      linalg.yield %1624 : f32
    } -> tensor<1x32x256xf32>
    %1625 = arith.constant {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} 0.000000e+00 : f32
    %1626 = tensor.splat %1625 {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<1x32xf32>
    %1627 = linalg.reduce ins(%1620:tensor<1x32x256xf32>) outs(%1626:tensor<1x32xf32>) dimensions = [2]
    (%1628: f32, %1629: f32) {
      %1630 = arith.addf %1628, %1629 : f32
      linalg.yield %1630 : f32
    }
    %1631 = arith.constant {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} 2.560000e+02 : f32
    %1632 = tensor.splat %1631 {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<1x32xf32>
    %1633 = tensor.empty() : tensor<1x32xf32>
    %1634 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1627, %1632 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1633 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb193(%1635: f32, %1636: f32, %1637: f32):
      %1638 = arith.divf %1635, %1636 : f32
      linalg.yield %1638 : f32
    } -> tensor<1x32xf32>
    %1639 = tensor.collapse_shape %1634 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %1640 = tensor.expand_shape %1639 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1641 = arith.constant {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} 1.000000e-05 : f32
    %1642 = tensor.splat %1641 {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} : tensor<1x32x1xf32>
    %1643 = tensor.empty() : tensor<1x32x1xf32>
    %1644 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1640, %1642 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1643 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb194(%1645: f32, %1646: f32, %1647: f32):
      %1648 = arith.addf %1645, %1646 : f32
      linalg.yield %1648 : f32
    } -> tensor<1x32x1xf32>
    %1649 = tensor.empty() : tensor<1x32x1xf32>
    %1650 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1644 : tensor<1x32x1xf32>) outs(%1649 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb195(%1651: f32, %1652: f32):
      %1653 = math.rsqrt %1651 : f32
      linalg.yield %1653 : f32
    } -> tensor<1x32x1xf32>
    %1654 = tensor.empty() : tensor<1x32x256xf32>
    %1655 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1614, %1650 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1654 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb196(%1656: f32, %1657: f32, %1658: f32):
      %1659 = arith.mulf %1656, %1657 : f32
      linalg.yield %1659 : f32
    } -> tensor<1x32x256xf32>
    %1660 = tensor.empty() : tensor<1x32x256xf32>
    %1661 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%73, %1655 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%1660 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_42", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.input_layernorm"} {
    ^bb197(%1662: f32, %1663: f32, %1664: f32):
      %1665 = arith.mulf %1662, %1663 : f32
      linalg.yield %1665 : f32
    } -> tensor<1x32x256xf32>
    %1666 = tensor.empty() : tensor<1x32x256xf32>
    %1667 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1661 : tensor<1x32x256xf32>) outs(%1666 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_14", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb198(%1668: f32, %1669: f32):
      %1670 = math.absf %1668 : f32
      linalg.yield %1670 : f32
    } -> tensor<1x32x256xf32>
    %1671 = arith.constant {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} 0xff800000 : f32
    %1672 = arith.constant {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} 0 : i64
    %1673 = tensor.splat %1671 {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<1x32xf32>
    %1674 = tensor.splat %1672 {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<1x32xi64>
    %1675, %1676 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1667 : tensor<1x32x256xf32>) outs(%1673, %1674 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb199(%1677: f32, %1678: f32, %1679: i64):
      %1680 = linalg.index 2 : index
      %1681 = arith.index_cast %1680 : index to i64
      %1682 = arith.cmpf ogt, %1677, %1678 : f32
      %1683 = arith.select %1682, %1677, %1678 : f32
      %1684 = arith.select %1682, %1681, %1679 : i64
      linalg.yield %1683, %1684 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1685 = tensor.collapse_shape %1675 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1686 = tensor.expand_shape %1685 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1687 = tensor.collapse_shape %1676 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1688 = tensor.expand_shape %1687 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_7", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1689 = tensor.empty() : tensor<1x32x1xf32>
    %1690 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1686 : tensor<1x32x1xf32>) outs(%1689 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "minmax_29", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb200(%1691: f32, %1692: f32):
      %1693 = arith.constant 1.000000e-05 : f32
      %1694 = arith.maximumf %1691, %1693 : f32
      linalg.yield %1694 : f32
    } -> tensor<1x32x1xf32>
    %1695 = tensor.empty() : tensor<1x32x1xf32>
    %1696 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1690 : tensor<1x32x1xf32>) outs(%1695 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_14", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb201(%1697: f32, %1698: f32):
      %1699 = arith.constant 1.000000e+00 : f32
      %1700 = arith.divf %1699, %1697 : f32
      linalg.yield %1700 : f32
    } -> tensor<1x32x1xf32>
    %1701 = arith.constant {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} 1.270000e+02 : f32
    %1702 = tensor.splat %1701 {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<1x32x1xf32>
    %1703 = tensor.empty() : tensor<1x32x1xf32>
    %1704 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1696, %1702 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1703 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb202(%1705: f32, %1706: f32, %1707: f32):
      %1708 = arith.mulf %1705, %1706 : f32
      linalg.yield %1708 : f32
    } -> tensor<1x32x1xf32>
    %1709 = tensor.empty() : tensor<1x32x256xf32>
    %1710 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1661, %1704 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1709 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_44", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb203(%1711: f32, %1712: f32, %1713: f32):
      %1714 = arith.mulf %1711, %1712 : f32
      linalg.yield %1714 : f32
    } -> tensor<1x32x256xf32>
    %1715 = tensor.empty() : tensor<1x32x256xf32>
    %1716 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1710 : tensor<1x32x256xf32>) outs(%1715 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_14", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb204(%1717: f32, %1718: f32):
      %1719 = math.roundeven %1717 : f32
      linalg.yield %1719 : f32
    } -> tensor<1x32x256xf32>
    %1720 = tensor.empty() : tensor<1x32x256xf32>
    %1721 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1716 : tensor<1x32x256xf32>) outs(%1720 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_30", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb205(%1722: f32, %1723: f32):
      %1724 = arith.constant -1.280000e+02 : f32
      %1725 = arith.maximumf %1722, %1724 : f32
      %1726 = arith.constant 1.270000e+02 : f32
      %1727 = arith.minimumf %1725, %1726 : f32
      linalg.yield %1727 : f32
    } -> tensor<1x32x256xf32>
    %1728 = tensor.empty() : tensor<1x32x256xf32>
    %1729 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1721, %1704 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1728 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_15", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb206(%1730: f32, %1731: f32, %1732: f32):
      %1733 = arith.divf %1730, %1731 : f32
      linalg.yield %1733 : f32
    } -> tensor<1x32x256xf32>
    %1734 = tensor.empty() : tensor<256x256xf32>
    %1735 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%64 : tensor<256x256xf32>) outs(%1734 : tensor<256x256xf32>) attrs =  {prov.region_id = "abs_15", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb207(%1736: f32, %1737: f32):
      %1738 = math.absf %1736 : f32
      linalg.yield %1738 : f32
    } -> tensor<256x256xf32>
    %1739 = arith.constant {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} 0.000000e+00 : f32
    %1740 = tensor.splat %1739 {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<f32>
    %1741 = linalg.reduce ins(%1735:tensor<256x256xf32>) outs(%1740:tensor<f32>) dimensions = [0, 1]
    (%1742: f32, %1743: f32) {
      %1744 = arith.addf %1742, %1743 : f32
      linalg.yield %1744 : f32
    }
    %1745 = arith.constant {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} 6.553600e+04 : f32
    %1746 = tensor.splat %1745 {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<f32>
    %1747 = tensor.empty() : tensor<f32>
    %1748 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1741, %1746 : tensor<f32>, tensor<f32>) outs(%1747 : tensor<f32>) attrs =  {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb208(%1749: f32, %1750: f32, %1751: f32):
      %1752 = arith.divf %1749, %1750 : f32
      linalg.yield %1752 : f32
    } -> tensor<f32>
    %1753 = tensor.empty() : tensor<f32>
    %1754 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1748 : tensor<f32>) outs(%1753 : tensor<f32>) attrs =  {prov.region_id = "minmax_31", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb209(%1755: f32, %1756: f32):
      %1757 = arith.constant 1.000000e-05 : f32
      %1758 = arith.maximumf %1755, %1757 : f32
      linalg.yield %1758 : f32
    } -> tensor<f32>
    %1759 = tensor.empty() : tensor<f32>
    %1760 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1754 : tensor<f32>) outs(%1759 : tensor<f32>) attrs =  {prov.region_id = "elementwise_15", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb210(%1761: f32, %1762: f32):
      %1763 = arith.constant 1.000000e+00 : f32
      %1764 = arith.divf %1763, %1761 : f32
      linalg.yield %1764 : f32
    } -> tensor<f32>
    %1765 = arith.constant {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} 1.000000e+00 : f32
    %1766 = tensor.splat %1765 {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<f32>
    %1767 = tensor.empty() : tensor<f32>
    %1768 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1760, %1766 : tensor<f32>, tensor<f32>) outs(%1767 : tensor<f32>) attrs =  {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb211(%1769: f32, %1770: f32, %1771: f32):
      %1772 = arith.mulf %1769, %1770 : f32
      linalg.yield %1772 : f32
    } -> tensor<f32>
    %1773 = tensor.empty() : tensor<256x256xf32>
    %1774 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%64, %1768 : tensor<256x256xf32>, tensor<f32>) outs(%1773 : tensor<256x256xf32>) attrs =  {prov.region_id = "mul_46", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb212(%1775: f32, %1776: f32, %1777: f32):
      %1778 = arith.mulf %1775, %1776 : f32
      linalg.yield %1778 : f32
    } -> tensor<256x256xf32>
    %1779 = tensor.empty() : tensor<256x256xf32>
    %1780 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1774 : tensor<256x256xf32>) outs(%1779 : tensor<256x256xf32>) attrs =  {prov.region_id = "round_15", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb213(%1781: f32, %1782: f32):
      %1783 = math.roundeven %1781 : f32
      linalg.yield %1783 : f32
    } -> tensor<256x256xf32>
    %1784 = tensor.empty() : tensor<256x256xf32>
    %1785 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1780 : tensor<256x256xf32>) outs(%1784 : tensor<256x256xf32>) attrs =  {prov.region_id = "minmax_32", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb214(%1786: f32, %1787: f32):
      %1788 = arith.constant -1.000000e+00 : f32
      %1789 = arith.maximumf %1786, %1788 : f32
      %1790 = arith.constant 1.000000e+00 : f32
      %1791 = arith.minimumf %1789, %1790 : f32
      linalg.yield %1791 : f32
    } -> tensor<256x256xf32>
    %1792 = tensor.empty() : tensor<256x256xf32>
    %1793 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1785, %1768 : tensor<256x256xf32>, tensor<f32>) outs(%1792 : tensor<256x256xf32>) attrs =  {prov.region_id = "div_16", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} {
    ^bb215(%1794: f32, %1795: f32, %1796: f32):
      %1797 = arith.divf %1794, %1795 : f32
      linalg.yield %1797 : f32
    } -> tensor<256x256xf32>
    %1798 = tensor.empty() : tensor<256x256xf32>
    %1799 = linalg.transpose ins(%1793:tensor<256x256xf32>) outs(%1798:tensor<256x256xf32>) permutation = [1, 0]
    %1800 = tensor.collapse_shape %1729 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %1801 = tensor.expand_shape %1800 [[0 : i64, 1 : i64]] output_shape [32, 256] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<8192xf32> into tensor<32x256xf32>
    %1802 = tensor.empty() : tensor<32x256xf32>
    %1803 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1804 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1803 : f32) outs(%1802 : tensor<32x256xf32>) -> tensor<32x256xf32>
    %1805 = linalg.matmul {prov.region_id = "matmul_9", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj", prov.transposed_b = "true"} ins(%1801, %1799 : tensor<32x256xf32>, tensor<256x256xf32>) outs(%1804 : tensor<32x256xf32>) -> tensor<32x256xf32>
    %1806 = tensor.collapse_shape %1805 [[0 : i64, 1 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<32x256xf32> into tensor<8192xf32>
    %1807 = tensor.expand_shape %1806 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 256] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.q_proj"} : tensor<8192xf32> into tensor<1x32x256xf32>
    %1808 = tensor.empty() : tensor<1x32x256xf32>
    %1809 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1661 : tensor<1x32x256xf32>) outs(%1808 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_16", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb216(%1810: f32, %1811: f32):
      %1812 = math.absf %1810 : f32
      linalg.yield %1812 : f32
    } -> tensor<1x32x256xf32>
    %1813 = arith.constant {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} 0xff800000 : f32
    %1814 = arith.constant {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} 0 : i64
    %1815 = tensor.splat %1813 {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<1x32xf32>
    %1816 = tensor.splat %1814 {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<1x32xi64>
    %1817, %1818 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1809 : tensor<1x32x256xf32>) outs(%1815, %1816 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb217(%1819: f32, %1820: f32, %1821: i64):
      %1822 = linalg.index 2 : index
      %1823 = arith.index_cast %1822 : index to i64
      %1824 = arith.cmpf ogt, %1819, %1820 : f32
      %1825 = arith.select %1824, %1819, %1820 : f32
      %1826 = arith.select %1824, %1823, %1821 : i64
      linalg.yield %1825, %1826 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1827 = tensor.collapse_shape %1817 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1828 = tensor.expand_shape %1827 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1829 = tensor.collapse_shape %1818 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1830 = tensor.expand_shape %1829 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_8", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1831 = tensor.empty() : tensor<1x32x1xf32>
    %1832 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1828 : tensor<1x32x1xf32>) outs(%1831 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "minmax_33", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb218(%1833: f32, %1834: f32):
      %1835 = arith.constant 1.000000e-05 : f32
      %1836 = arith.maximumf %1833, %1835 : f32
      linalg.yield %1836 : f32
    } -> tensor<1x32x1xf32>
    %1837 = tensor.empty() : tensor<1x32x1xf32>
    %1838 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1832 : tensor<1x32x1xf32>) outs(%1837 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_16", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb219(%1839: f32, %1840: f32):
      %1841 = arith.constant 1.000000e+00 : f32
      %1842 = arith.divf %1841, %1839 : f32
      linalg.yield %1842 : f32
    } -> tensor<1x32x1xf32>
    %1843 = arith.constant {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} 1.270000e+02 : f32
    %1844 = tensor.splat %1843 {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<1x32x1xf32>
    %1845 = tensor.empty() : tensor<1x32x1xf32>
    %1846 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1838, %1844 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1845 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb220(%1847: f32, %1848: f32, %1849: f32):
      %1850 = arith.mulf %1847, %1848 : f32
      linalg.yield %1850 : f32
    } -> tensor<1x32x1xf32>
    %1851 = tensor.empty() : tensor<1x32x256xf32>
    %1852 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1661, %1846 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1851 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_48", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb221(%1853: f32, %1854: f32, %1855: f32):
      %1856 = arith.mulf %1853, %1854 : f32
      linalg.yield %1856 : f32
    } -> tensor<1x32x256xf32>
    %1857 = tensor.empty() : tensor<1x32x256xf32>
    %1858 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1852 : tensor<1x32x256xf32>) outs(%1857 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_16", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb222(%1859: f32, %1860: f32):
      %1861 = math.roundeven %1859 : f32
      linalg.yield %1861 : f32
    } -> tensor<1x32x256xf32>
    %1862 = tensor.empty() : tensor<1x32x256xf32>
    %1863 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1858 : tensor<1x32x256xf32>) outs(%1862 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_34", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb223(%1864: f32, %1865: f32):
      %1866 = arith.constant -1.280000e+02 : f32
      %1867 = arith.maximumf %1864, %1866 : f32
      %1868 = arith.constant 1.270000e+02 : f32
      %1869 = arith.minimumf %1867, %1868 : f32
      linalg.yield %1869 : f32
    } -> tensor<1x32x256xf32>
    %1870 = tensor.empty() : tensor<1x32x256xf32>
    %1871 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1863, %1846 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1870 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_17", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb224(%1872: f32, %1873: f32, %1874: f32):
      %1875 = arith.divf %1872, %1873 : f32
      linalg.yield %1875 : f32
    } -> tensor<1x32x256xf32>
    %1876 = tensor.empty() : tensor<128x256xf32>
    %1877 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65 : tensor<128x256xf32>) outs(%1876 : tensor<128x256xf32>) attrs =  {prov.region_id = "abs_17", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb225(%1878: f32, %1879: f32):
      %1880 = math.absf %1878 : f32
      linalg.yield %1880 : f32
    } -> tensor<128x256xf32>
    %1881 = arith.constant {prov.region_id = "reduce_13", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} 0.000000e+00 : f32
    %1882 = tensor.splat %1881 {prov.region_id = "reduce_13", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<f32>
    %1883 = linalg.reduce ins(%1877:tensor<128x256xf32>) outs(%1882:tensor<f32>) dimensions = [0, 1]
    (%1884: f32, %1885: f32) {
      %1886 = arith.addf %1884, %1885 : f32
      linalg.yield %1886 : f32
    }
    %1887 = arith.constant {prov.region_id = "reduce_13", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} 3.276800e+04 : f32
    %1888 = tensor.splat %1887 {prov.region_id = "reduce_13", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<f32>
    %1889 = tensor.empty() : tensor<f32>
    %1890 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1883, %1888 : tensor<f32>, tensor<f32>) outs(%1889 : tensor<f32>) attrs =  {prov.region_id = "reduce_13", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb226(%1891: f32, %1892: f32, %1893: f32):
      %1894 = arith.divf %1891, %1892 : f32
      linalg.yield %1894 : f32
    } -> tensor<f32>
    %1895 = tensor.empty() : tensor<f32>
    %1896 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1890 : tensor<f32>) outs(%1895 : tensor<f32>) attrs =  {prov.region_id = "minmax_35", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb227(%1897: f32, %1898: f32):
      %1899 = arith.constant 1.000000e-05 : f32
      %1900 = arith.maximumf %1897, %1899 : f32
      linalg.yield %1900 : f32
    } -> tensor<f32>
    %1901 = tensor.empty() : tensor<f32>
    %1902 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1896 : tensor<f32>) outs(%1901 : tensor<f32>) attrs =  {prov.region_id = "elementwise_17", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb228(%1903: f32, %1904: f32):
      %1905 = arith.constant 1.000000e+00 : f32
      %1906 = arith.divf %1905, %1903 : f32
      linalg.yield %1906 : f32
    } -> tensor<f32>
    %1907 = arith.constant {prov.region_id = "mul_49", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} 1.000000e+00 : f32
    %1908 = tensor.splat %1907 {prov.region_id = "mul_49", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<f32>
    %1909 = tensor.empty() : tensor<f32>
    %1910 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1902, %1908 : tensor<f32>, tensor<f32>) outs(%1909 : tensor<f32>) attrs =  {prov.region_id = "mul_49", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb229(%1911: f32, %1912: f32, %1913: f32):
      %1914 = arith.mulf %1911, %1912 : f32
      linalg.yield %1914 : f32
    } -> tensor<f32>
    %1915 = tensor.empty() : tensor<128x256xf32>
    %1916 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %1910 : tensor<128x256xf32>, tensor<f32>) outs(%1915 : tensor<128x256xf32>) attrs =  {prov.region_id = "mul_50", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb230(%1917: f32, %1918: f32, %1919: f32):
      %1920 = arith.mulf %1917, %1918 : f32
      linalg.yield %1920 : f32
    } -> tensor<128x256xf32>
    %1921 = tensor.empty() : tensor<128x256xf32>
    %1922 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1916 : tensor<128x256xf32>) outs(%1921 : tensor<128x256xf32>) attrs =  {prov.region_id = "round_17", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb231(%1923: f32, %1924: f32):
      %1925 = math.roundeven %1923 : f32
      linalg.yield %1925 : f32
    } -> tensor<128x256xf32>
    %1926 = tensor.empty() : tensor<128x256xf32>
    %1927 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1922 : tensor<128x256xf32>) outs(%1926 : tensor<128x256xf32>) attrs =  {prov.region_id = "minmax_36", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb232(%1928: f32, %1929: f32):
      %1930 = arith.constant -1.000000e+00 : f32
      %1931 = arith.maximumf %1928, %1930 : f32
      %1932 = arith.constant 1.000000e+00 : f32
      %1933 = arith.minimumf %1931, %1932 : f32
      linalg.yield %1933 : f32
    } -> tensor<128x256xf32>
    %1934 = tensor.empty() : tensor<128x256xf32>
    %1935 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1927, %1910 : tensor<128x256xf32>, tensor<f32>) outs(%1934 : tensor<128x256xf32>) attrs =  {prov.region_id = "div_18", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} {
    ^bb233(%1936: f32, %1937: f32, %1938: f32):
      %1939 = arith.divf %1936, %1937 : f32
      linalg.yield %1939 : f32
    } -> tensor<128x256xf32>
    %1940 = tensor.empty() : tensor<256x128xf32>
    %1941 = linalg.transpose ins(%1935:tensor<128x256xf32>) outs(%1940:tensor<256x128xf32>) permutation = [1, 0]
    %1942 = tensor.collapse_shape %1871 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %1943 = tensor.expand_shape %1942 [[0 : i64, 1 : i64]] output_shape [32, 256] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<8192xf32> into tensor<32x256xf32>
    %1944 = tensor.empty() : tensor<32x128xf32>
    %1945 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %1946 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%1945 : f32) outs(%1944 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %1947 = linalg.matmul {prov.region_id = "matmul_10", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj", prov.transposed_b = "true"} ins(%1943, %1941 : tensor<32x256xf32>, tensor<256x128xf32>) outs(%1946 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %1948 = tensor.collapse_shape %1947 [[0 : i64, 1 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<32x128xf32> into tensor<4096xf32>
    %1949 = tensor.expand_shape %1948 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 128] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.k_proj"} : tensor<4096xf32> into tensor<1x32x128xf32>
    %1950 = tensor.empty() : tensor<1x32x256xf32>
    %1951 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1661 : tensor<1x32x256xf32>) outs(%1950 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_18", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb234(%1952: f32, %1953: f32):
      %1954 = math.absf %1952 : f32
      linalg.yield %1954 : f32
    } -> tensor<1x32x256xf32>
    %1955 = arith.constant {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} 0xff800000 : f32
    %1956 = arith.constant {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} 0 : i64
    %1957 = tensor.splat %1955 {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<1x32xf32>
    %1958 = tensor.splat %1956 {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<1x32xi64>
    %1959, %1960 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1951 : tensor<1x32x256xf32>) outs(%1957, %1958 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb235(%1961: f32, %1962: f32, %1963: i64):
      %1964 = linalg.index 2 : index
      %1965 = arith.index_cast %1964 : index to i64
      %1966 = arith.cmpf ogt, %1961, %1962 : f32
      %1967 = arith.select %1966, %1961, %1962 : f32
      %1968 = arith.select %1966, %1965, %1963 : i64
      linalg.yield %1967, %1968 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %1969 = tensor.collapse_shape %1959 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %1970 = tensor.expand_shape %1969 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %1971 = tensor.collapse_shape %1960 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %1972 = tensor.expand_shape %1971 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_9", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %1973 = tensor.empty() : tensor<1x32x1xf32>
    %1974 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1970 : tensor<1x32x1xf32>) outs(%1973 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "minmax_37", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb236(%1975: f32, %1976: f32):
      %1977 = arith.constant 1.000000e-05 : f32
      %1978 = arith.maximumf %1975, %1977 : f32
      linalg.yield %1978 : f32
    } -> tensor<1x32x1xf32>
    %1979 = tensor.empty() : tensor<1x32x1xf32>
    %1980 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1974 : tensor<1x32x1xf32>) outs(%1979 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_18", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb237(%1981: f32, %1982: f32):
      %1983 = arith.constant 1.000000e+00 : f32
      %1984 = arith.divf %1983, %1981 : f32
      linalg.yield %1984 : f32
    } -> tensor<1x32x1xf32>
    %1985 = arith.constant {prov.region_id = "mul_51", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} 1.270000e+02 : f32
    %1986 = tensor.splat %1985 {prov.region_id = "mul_51", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<1x32x1xf32>
    %1987 = tensor.empty() : tensor<1x32x1xf32>
    %1988 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1980, %1986 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1987 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_51", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb238(%1989: f32, %1990: f32, %1991: f32):
      %1992 = arith.mulf %1989, %1990 : f32
      linalg.yield %1992 : f32
    } -> tensor<1x32x1xf32>
    %1993 = tensor.empty() : tensor<1x32x256xf32>
    %1994 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1661, %1988 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%1993 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_52", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb239(%1995: f32, %1996: f32, %1997: f32):
      %1998 = arith.mulf %1995, %1996 : f32
      linalg.yield %1998 : f32
    } -> tensor<1x32x256xf32>
    %1999 = tensor.empty() : tensor<1x32x256xf32>
    %2000 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1994 : tensor<1x32x256xf32>) outs(%1999 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_18", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb240(%2001: f32, %2002: f32):
      %2003 = math.roundeven %2001 : f32
      linalg.yield %2003 : f32
    } -> tensor<1x32x256xf32>
    %2004 = tensor.empty() : tensor<1x32x256xf32>
    %2005 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2000 : tensor<1x32x256xf32>) outs(%2004 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_38", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb241(%2006: f32, %2007: f32):
      %2008 = arith.constant -1.280000e+02 : f32
      %2009 = arith.maximumf %2006, %2008 : f32
      %2010 = arith.constant 1.270000e+02 : f32
      %2011 = arith.minimumf %2009, %2010 : f32
      linalg.yield %2011 : f32
    } -> tensor<1x32x256xf32>
    %2012 = tensor.empty() : tensor<1x32x256xf32>
    %2013 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2005, %1988 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2012 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_19", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb242(%2014: f32, %2015: f32, %2016: f32):
      %2017 = arith.divf %2014, %2015 : f32
      linalg.yield %2017 : f32
    } -> tensor<1x32x256xf32>
    %2018 = tensor.empty() : tensor<128x256xf32>
    %2019 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66 : tensor<128x256xf32>) outs(%2018 : tensor<128x256xf32>) attrs =  {prov.region_id = "abs_19", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb243(%2020: f32, %2021: f32):
      %2022 = math.absf %2020 : f32
      linalg.yield %2022 : f32
    } -> tensor<128x256xf32>
    %2023 = arith.constant {prov.region_id = "reduce_14", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} 0.000000e+00 : f32
    %2024 = tensor.splat %2023 {prov.region_id = "reduce_14", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<f32>
    %2025 = linalg.reduce ins(%2019:tensor<128x256xf32>) outs(%2024:tensor<f32>) dimensions = [0, 1]
    (%2026: f32, %2027: f32) {
      %2028 = arith.addf %2026, %2027 : f32
      linalg.yield %2028 : f32
    }
    %2029 = arith.constant {prov.region_id = "reduce_14", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} 3.276800e+04 : f32
    %2030 = tensor.splat %2029 {prov.region_id = "reduce_14", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<f32>
    %2031 = tensor.empty() : tensor<f32>
    %2032 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2025, %2030 : tensor<f32>, tensor<f32>) outs(%2031 : tensor<f32>) attrs =  {prov.region_id = "reduce_14", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb244(%2033: f32, %2034: f32, %2035: f32):
      %2036 = arith.divf %2033, %2034 : f32
      linalg.yield %2036 : f32
    } -> tensor<f32>
    %2037 = tensor.empty() : tensor<f32>
    %2038 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2032 : tensor<f32>) outs(%2037 : tensor<f32>) attrs =  {prov.region_id = "minmax_39", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb245(%2039: f32, %2040: f32):
      %2041 = arith.constant 1.000000e-05 : f32
      %2042 = arith.maximumf %2039, %2041 : f32
      linalg.yield %2042 : f32
    } -> tensor<f32>
    %2043 = tensor.empty() : tensor<f32>
    %2044 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2038 : tensor<f32>) outs(%2043 : tensor<f32>) attrs =  {prov.region_id = "elementwise_19", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb246(%2045: f32, %2046: f32):
      %2047 = arith.constant 1.000000e+00 : f32
      %2048 = arith.divf %2047, %2045 : f32
      linalg.yield %2048 : f32
    } -> tensor<f32>
    %2049 = arith.constant {prov.region_id = "mul_53", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} 1.000000e+00 : f32
    %2050 = tensor.splat %2049 {prov.region_id = "mul_53", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<f32>
    %2051 = tensor.empty() : tensor<f32>
    %2052 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2044, %2050 : tensor<f32>, tensor<f32>) outs(%2051 : tensor<f32>) attrs =  {prov.region_id = "mul_53", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb247(%2053: f32, %2054: f32, %2055: f32):
      %2056 = arith.mulf %2053, %2054 : f32
      linalg.yield %2056 : f32
    } -> tensor<f32>
    %2057 = tensor.empty() : tensor<128x256xf32>
    %2058 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %2052 : tensor<128x256xf32>, tensor<f32>) outs(%2057 : tensor<128x256xf32>) attrs =  {prov.region_id = "mul_54", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb248(%2059: f32, %2060: f32, %2061: f32):
      %2062 = arith.mulf %2059, %2060 : f32
      linalg.yield %2062 : f32
    } -> tensor<128x256xf32>
    %2063 = tensor.empty() : tensor<128x256xf32>
    %2064 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2058 : tensor<128x256xf32>) outs(%2063 : tensor<128x256xf32>) attrs =  {prov.region_id = "round_19", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb249(%2065: f32, %2066: f32):
      %2067 = math.roundeven %2065 : f32
      linalg.yield %2067 : f32
    } -> tensor<128x256xf32>
    %2068 = tensor.empty() : tensor<128x256xf32>
    %2069 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2064 : tensor<128x256xf32>) outs(%2068 : tensor<128x256xf32>) attrs =  {prov.region_id = "minmax_40", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb250(%2070: f32, %2071: f32):
      %2072 = arith.constant -1.000000e+00 : f32
      %2073 = arith.maximumf %2070, %2072 : f32
      %2074 = arith.constant 1.000000e+00 : f32
      %2075 = arith.minimumf %2073, %2074 : f32
      linalg.yield %2075 : f32
    } -> tensor<128x256xf32>
    %2076 = tensor.empty() : tensor<128x256xf32>
    %2077 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2069, %2052 : tensor<128x256xf32>, tensor<f32>) outs(%2076 : tensor<128x256xf32>) attrs =  {prov.region_id = "div_20", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} {
    ^bb251(%2078: f32, %2079: f32, %2080: f32):
      %2081 = arith.divf %2078, %2079 : f32
      linalg.yield %2081 : f32
    } -> tensor<128x256xf32>
    %2082 = tensor.empty() : tensor<256x128xf32>
    %2083 = linalg.transpose ins(%2077:tensor<128x256xf32>) outs(%2082:tensor<256x128xf32>) permutation = [1, 0]
    %2084 = tensor.collapse_shape %2013 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %2085 = tensor.expand_shape %2084 [[0 : i64, 1 : i64]] output_shape [32, 256] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<8192xf32> into tensor<32x256xf32>
    %2086 = tensor.empty() : tensor<32x128xf32>
    %2087 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %2088 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%2087 : f32) outs(%2086 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %2089 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj", prov.transposed_b = "true"} ins(%2085, %2083 : tensor<32x256xf32>, tensor<256x128xf32>) outs(%2088 : tensor<32x128xf32>) -> tensor<32x128xf32>
    %2090 = tensor.collapse_shape %2089 [[0 : i64, 1 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<32x128xf32> into tensor<4096xf32>
    %2091 = tensor.expand_shape %2090 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 128] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.v_proj"} : tensor<4096xf32> into tensor<1x32x128xf32>
    %2092 = tensor.collapse_shape %1807 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %2093 = tensor.expand_shape %2092 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 32] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x32x8x32xf32>
    %2094 = tensor.empty() : tensor<1x8x32x32xf32>
    %2095 = linalg.transpose ins(%2093:tensor<1x32x8x32xf32>) outs(%2094:tensor<1x8x32x32xf32>) permutation = [0, 2, 1, 3]
    %2096 = tensor.collapse_shape %1949 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %2097 = tensor.expand_shape %2096 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 32] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<4096xf32> into tensor<1x32x4x32xf32>
    %2098 = tensor.empty() : tensor<1x4x32x32xf32>
    %2099 = linalg.transpose ins(%2097:tensor<1x32x4x32xf32>) outs(%2098:tensor<1x4x32x32xf32>) permutation = [0, 2, 1, 3]
    %2100 = tensor.collapse_shape %2091 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x128xf32> into tensor<4096xf32>
    %2101 = tensor.expand_shape %2100 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 32] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<4096xf32> into tensor<1x32x4x32xf32>
    %2102 = tensor.empty() : tensor<1x4x32x32xf32>
    %2103 = linalg.transpose ins(%2101:tensor<1x32x4x32xf32>) outs(%2102:tensor<1x4x32x32xf32>) permutation = [0, 2, 1, 3]
    %2104 = "tensor.extract_slice"(%82) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_21", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.rotary_emb"} : (tensor<2048x32xf32>) -> tensor<32x32xf32>
    %2105 = "tensor.extract_slice"(%83) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 32, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_22", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.rotary_emb"} : (tensor<2048x32xf32>) -> tensor<32x32xf32>
    %2106 = tensor.empty() : tensor<1x32x32xf32>
    %2107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95 : tensor<1x32xi64>) outs(%2106 : tensor<1x32x32xf32>) attrs =  {prov.region_id = "gather_2", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb252(%2108: i64, %2109: f32):
      %2110 = arith.index_cast %2108 : i64 to index
      %2111 = linalg.index 2 : index
      %2112 = tensor.extract %2104[%2110, %2111] : tensor<32x32xf32>
      linalg.yield %2112 : f32
    } -> tensor<1x32x32xf32>
    %2113 = tensor.collapse_shape %2107 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %2114 = tensor.expand_shape %2113 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %2115 = tensor.empty() : tensor<1x32x32xf32>
    %2116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95 : tensor<1x32xi64>) outs(%2115 : tensor<1x32x32xf32>) attrs =  {prov.region_id = "gather_3", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb253(%2117: i64, %2118: f32):
      %2119 = arith.index_cast %2117 : i64 to index
      %2120 = linalg.index 2 : index
      %2121 = tensor.extract %2105[%2119, %2120] : tensor<32x32xf32>
      linalg.yield %2121 : f32
    } -> tensor<1x32x32xf32>
    %2122 = tensor.collapse_shape %2116 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %2123 = tensor.expand_shape %2122 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 32] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1024xf32> into tensor<1x1x32x32xf32>
    %2124 = tensor.empty() : tensor<1x8x32x32xf32>
    %2125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2095, %2114 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%2124 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "mul_55", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb254(%2126: f32, %2127: f32, %2128: f32):
      %2129 = arith.mulf %2126, %2127 : f32
      linalg.yield %2129 : f32
    } -> tensor<1x8x32x32xf32>
    %2130 = "tensor.extract_slice"(%2095) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_23", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x8x32x32xf32>) -> tensor<1x8x32x16xf32>
    %2131 = "tensor.extract_slice"(%2095) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 8, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_24", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x8x32x32xf32>) -> tensor<1x8x32x16xf32>
    %2132 = tensor.empty() : tensor<1x8x32x16xf32>
    %2133 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2131 : tensor<1x8x32x16xf32>) outs(%2132 : tensor<1x8x32x16xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb255(%2134: f32, %2135: f32):
      %2136 = arith.negf %2134 : f32
      linalg.yield %2136 : f32
    } -> tensor<1x8x32x16xf32>
    %2137 = tensor.concat dim(3) %2133, %2130 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x8x32x16xf32>, tensor<1x8x32x16xf32>) -> tensor<1x8x32x32xf32>
    %2138 = tensor.empty() : tensor<1x8x32x32xf32>
    %2139 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2137, %2123 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%2138 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "mul_56", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb256(%2140: f32, %2141: f32, %2142: f32):
      %2143 = arith.mulf %2140, %2141 : f32
      linalg.yield %2143 : f32
    } -> tensor<1x8x32x32xf32>
    %2144 = tensor.empty() : tensor<1x8x32x32xf32>
    %2145 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2125, %2139 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%2144 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb257(%2146: f32, %2147: f32, %2148: f32):
      %2149 = arith.addf %2146, %2147 : f32
      linalg.yield %2149 : f32
    } -> tensor<1x8x32x32xf32>
    %2150 = tensor.empty() : tensor<1x4x32x32xf32>
    %2151 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2099, %2114 : tensor<1x4x32x32xf32>, tensor<1x1x32x32xf32>) outs(%2150 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "mul_57", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb258(%2152: f32, %2153: f32, %2154: f32):
      %2155 = arith.mulf %2152, %2153 : f32
      linalg.yield %2155 : f32
    } -> tensor<1x4x32x32xf32>
    %2156 = "tensor.extract_slice"(%2099) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_25", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x16xf32>
    %2157 = "tensor.extract_slice"(%2099) <{static_offsets = array<i64: 0, 0, 0, 16>, static_sizes = array<i64: 1, 4, 32, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_26", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x16xf32>
    %2158 = tensor.empty() : tensor<1x4x32x16xf32>
    %2159 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2157 : tensor<1x4x32x16xf32>) outs(%2158 : tensor<1x4x32x16xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb259(%2160: f32, %2161: f32):
      %2162 = arith.negf %2160 : f32
      linalg.yield %2162 : f32
    } -> tensor<1x4x32x16xf32>
    %2163 = tensor.concat dim(3) %2159, %2156 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x16xf32>, tensor<1x4x32x16xf32>) -> tensor<1x4x32x32xf32>
    %2164 = tensor.empty() : tensor<1x4x32x32xf32>
    %2165 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2163, %2123 : tensor<1x4x32x32xf32>, tensor<1x1x32x32xf32>) outs(%2164 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "mul_58", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb260(%2166: f32, %2167: f32, %2168: f32):
      %2169 = arith.mulf %2166, %2167 : f32
      linalg.yield %2169 : f32
    } -> tensor<1x4x32x32xf32>
    %2170 = tensor.empty() : tensor<1x4x32x32xf32>
    %2171 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2151, %2165 : tensor<1x4x32x32xf32>, tensor<1x4x32x32xf32>) outs(%2170 : tensor<1x4x32x32xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb261(%2172: f32, %2173: f32, %2174: f32):
      %2175 = arith.addf %2172, %2173 : f32
      linalg.yield %2175 : f32
    } -> tensor<1x4x32x32xf32>
    %2176 = "tensor.extract_slice"(%2171) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_27", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %2177 = "tensor.extract_slice"(%2176) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_28", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %2178 = tensor.collapse_shape %2177 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x32x32xf32> into tensor<4096xf32>
    %2179 = tensor.expand_shape %2178 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 32, 32] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<4096xf32> into tensor<1x4x1x32x32xf32>
    %2180 = "tensor.extract_slice"(%2179) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_29", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %2181 = "tensor.extract_slice"(%2180) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_30", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %2182 = tensor.empty() : tensor<1x4x2x32x32xf32>
    %2183 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2181 : tensor<1x4x1x32x32xf32>) outs(%2182 : tensor<1x4x2x32x32xf32>) attrs =  {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb262(%2184: f32, %2185: f32):
      linalg.yield %2184 : f32
    } -> tensor<1x4x2x32x32xf32>
    %2186 = tensor.collapse_shape %2183 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x2x32x32xf32> into tensor<8192xf32>
    %2187 = tensor.expand_shape %2186 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 32] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x8x32x32xf32>
    %2188 = "tensor.extract_slice"(%2103) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_31", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %2189 = "tensor.extract_slice"(%2188) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_32", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x32x32xf32>) -> tensor<1x4x32x32xf32>
    %2190 = tensor.collapse_shape %2189 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x32x32xf32> into tensor<4096xf32>
    %2191 = tensor.expand_shape %2190 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 32, 32] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<4096xf32> into tensor<1x4x1x32x32xf32>
    %2192 = "tensor.extract_slice"(%2191) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_33", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %2193 = "tensor.extract_slice"(%2192) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_34", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x4x1x32x32xf32>) -> tensor<1x4x1x32x32xf32>
    %2194 = tensor.empty() : tensor<1x4x2x32x32xf32>
    %2195 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2193 : tensor<1x4x1x32x32xf32>) outs(%2194 : tensor<1x4x2x32x32xf32>) attrs =  {prov.region_id = "expand_10", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb263(%2196: f32, %2197: f32):
      linalg.yield %2196 : f32
    } -> tensor<1x4x2x32x32xf32>
    %2198 = tensor.collapse_shape %2195 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x4x2x32x32xf32> into tensor<8192xf32>
    %2199 = tensor.expand_shape %2198 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 32] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x8x32x32xf32>
    %2200 = tensor.empty() : tensor<1x8x32x32xf32>
    %2201 = linalg.transpose ins(%2187:tensor<1x8x32x32xf32>) outs(%2200:tensor<1x8x32x32xf32>) permutation = [0, 1, 3, 2]
    %2202 = tensor.empty() : tensor<1x8x32x32xf32>
    %2203 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2145 : tensor<1x8x32x32xf32>) outs(%2202 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "expand_11", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb264(%2204: f32, %2205: f32):
      linalg.yield %2204 : f32
    } -> tensor<1x8x32x32xf32>
    %2206 = tensor.collapse_shape %2203 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32x32xf32> into tensor<8192xf32>
    %2207 = tensor.expand_shape %2206 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 32, 32] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<8x32x32xf32>
    %2208 = tensor.empty() : tensor<1x8x32x32xf32>
    %2209 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2201 : tensor<1x8x32x32xf32>) outs(%2208 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "expand_12", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb265(%2210: f32, %2211: f32):
      linalg.yield %2210 : f32
    } -> tensor<1x8x32x32xf32>
    %2212 = tensor.collapse_shape %2209 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32x32xf32> into tensor<8192xf32>
    %2213 = tensor.expand_shape %2212 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 32, 32] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<8x32x32xf32>
    %2214 = arith.constant {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 0.000000e+00 : f32
    %2215 = tensor.splat %2214 {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8x32x32xf32>
    %2216 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2207, %2213 : tensor<8x32x32xf32>, tensor<8x32x32xf32>) outs(%2215 : tensor<8x32x32xf32>) attrs =  {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb266(%2217: f32, %2218: f32, %2219: f32):
      %2220 = arith.mulf %2217, %2218 : f32
      %2221 = arith.addf %2219, %2220 : f32
      linalg.yield %2221 : f32
    } -> tensor<8x32x32xf32>
    %2222 = tensor.collapse_shape %2216 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8x32x32xf32> into tensor<8192xf32>
    %2223 = tensor.expand_shape %2222 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 32] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x8x32x32xf32>
    %2224 = arith.constant {prov.region_id = "div_21", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 5.65685415 : f32
    %2225 = tensor.splat %2224 {prov.region_id = "div_21", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32x32xf32>
    %2226 = tensor.empty() : tensor<1x8x32x32xf32>
    %2227 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2223, %2225 : tensor<1x8x32x32xf32>, tensor<1x8x32x32xf32>) outs(%2226 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "div_21", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb267(%2228: f32, %2229: f32, %2230: f32):
      %2231 = arith.divf %2228, %2229 : f32
      linalg.yield %2231 : f32
    } -> tensor<1x8x32x32xf32>
    %2232 = "tensor.extract_slice"(%186) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_35", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    %2233 = "tensor.extract_slice"(%2232) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_36", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    %2234 = "tensor.extract_slice"(%2233) <{static_offsets = array<i64: 0, 0, 31, 0>, static_sizes = array<i64: 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : (tensor<1x1x32x32xf32>) -> tensor<32xf32>
    %2235 = tensor.expand_shape %2234 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 32] {prov.region_id = "slice_37", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<32xf32> into tensor<1x1x32xf32>
    %2236 = tensor.collapse_shape %2235 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x1x32xf32> into tensor<32xf32>
    %2237 = tensor.expand_shape %2236 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<32xf32> into tensor<1x1x1x32xf32>
    %2238 = tensor.empty() : tensor<1x1x32x32xf32>
    %2239 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2237 : tensor<1x1x1x32xf32>) outs(%2238 : tensor<1x1x32x32xf32>) attrs =  {prov.region_id = "expand_13", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb268(%2240: f32, %2241: f32):
      linalg.yield %2240 : f32
    } -> tensor<1x1x32x32xf32>
    %2242 = tensor.empty() : tensor<1x8x32x32xf32>
    %2243 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2227, %2239 : tensor<1x8x32x32xf32>, tensor<1x1x32x32xf32>) outs(%2242 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb269(%2244: f32, %2245: f32, %2246: f32):
      %2247 = arith.addf %2244, %2245 : f32
      linalg.yield %2247 : f32
    } -> tensor<1x8x32x32xf32>
    %2248 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 0xff800000 : f32
    %2249 = tensor.splat %2248 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32xf32>
    %2250 = linalg.reduce ins(%2243:tensor<1x8x32x32xf32>) outs(%2249:tensor<1x8x32xf32>) dimensions = [3]
    (%2251: f32, %2252: f32) {
      %2253 = arith.maximumf %2251, %2252 : f32
      linalg.yield %2253 : f32
    }
    %2254 = tensor.collapse_shape %2250 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %2255 = tensor.expand_shape %2254 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %2256 = tensor.empty() : tensor<1x8x32x32xf32>
    %2257 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2243, %2255 : tensor<1x8x32x32xf32>, tensor<1x8x32x1xf32>) outs(%2256 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb270(%2258: f32, %2259: f32, %2260: f32):
      %2261 = arith.subf %2258, %2259 : f32
      linalg.yield %2261 : f32
    } -> tensor<1x8x32x32xf32>
    %2262 = tensor.empty() : tensor<1x8x32x32xf32>
    %2263 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2257 : tensor<1x8x32x32xf32>) outs(%2262 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb271(%2264: f32, %2265: f32):
      %2266 = math.exp %2264 : f32
      linalg.yield %2266 : f32
    } -> tensor<1x8x32x32xf32>
    %2267 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 0.000000e+00 : f32
    %2268 = tensor.splat %2267 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32xf32>
    %2269 = linalg.reduce ins(%2263:tensor<1x8x32x32xf32>) outs(%2268:tensor<1x8x32xf32>) dimensions = [3]
    (%2270: f32, %2271: f32) {
      %2272 = arith.addf %2270, %2271 : f32
      linalg.yield %2272 : f32
    }
    %2273 = tensor.collapse_shape %2269 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32xf32> into tensor<256xf32>
    %2274 = tensor.expand_shape %2273 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<256xf32> into tensor<1x8x32x1xf32>
    %2275 = tensor.empty() : tensor<1x8x32x32xf32>
    %2276 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2263, %2274 : tensor<1x8x32x32xf32>, tensor<1x8x32x1xf32>) outs(%2275 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb272(%2277: f32, %2278: f32, %2279: f32):
      %2280 = arith.divf %2277, %2278 : f32
      linalg.yield %2280 : f32
    } -> tensor<1x8x32x32xf32>
    %2281 = tensor.empty() : tensor<1x8x32x32xf32>
    %2282 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2276 : tensor<1x8x32x32xf32>) outs(%2281 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "expand_14", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb273(%2283: f32, %2284: f32):
      linalg.yield %2283 : f32
    } -> tensor<1x8x32x32xf32>
    %2285 = tensor.collapse_shape %2282 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32x32xf32> into tensor<8192xf32>
    %2286 = tensor.expand_shape %2285 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 32, 32] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<8x32x32xf32>
    %2287 = tensor.empty() : tensor<1x8x32x32xf32>
    %2288 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2199 : tensor<1x8x32x32xf32>) outs(%2287 : tensor<1x8x32x32xf32>) attrs =  {prov.region_id = "expand_15", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb274(%2289: f32, %2290: f32):
      linalg.yield %2289 : f32
    } -> tensor<1x8x32x32xf32>
    %2291 = tensor.collapse_shape %2288 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x8x32x32xf32> into tensor<8192xf32>
    %2292 = tensor.expand_shape %2291 [[0 : i64, 1 : i64, 2 : i64]] output_shape [8, 32, 32] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<8x32x32xf32>
    %2293 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} 0.000000e+00 : f32
    %2294 = tensor.splat %2293 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8x32x32xf32>
    %2295 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2286, %2292 : tensor<8x32x32xf32>, tensor<8x32x32xf32>) outs(%2294 : tensor<8x32x32xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} {
    ^bb275(%2296: f32, %2297: f32, %2298: f32):
      %2299 = arith.mulf %2296, %2297 : f32
      %2300 = arith.addf %2298, %2299 : f32
      linalg.yield %2300 : f32
    } -> tensor<8x32x32xf32>
    %2301 = tensor.collapse_shape %2295 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8x32x32xf32> into tensor<8192xf32>
    %2302 = tensor.expand_shape %2301 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 32] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x8x32x32xf32>
    %2303 = tensor.empty() : tensor<1x32x8x32xf32>
    %2304 = linalg.transpose ins(%2302:tensor<1x8x32x32xf32>) outs(%2303:tensor<1x32x8x32xf32>) permutation = [0, 2, 1, 3]
    %2305 = tensor.collapse_shape %2304 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<1x32x8x32xf32> into tensor<8192xf32>
    %2306 = tensor.expand_shape %2305 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 256] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x32x256xf32>
    %2307 = tensor.empty() : tensor<1x32x256xf32>
    %2308 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2306 : tensor<1x32x256xf32>) outs(%2307 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb276(%2309: f32, %2310: f32):
      %2311 = arith.constant 2.000000e+00 : f32
      %2312 = math.powf %2309, %2311 : f32
      linalg.yield %2312 : f32
    } -> tensor<1x32x256xf32>
    %2313 = arith.constant {prov.region_id = "reduce_15", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} 0.000000e+00 : f32
    %2314 = tensor.splat %2313 {prov.region_id = "reduce_15", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} : tensor<1x32xf32>
    %2315 = linalg.reduce ins(%2308:tensor<1x32x256xf32>) outs(%2314:tensor<1x32xf32>) dimensions = [2]
    (%2316: f32, %2317: f32) {
      %2318 = arith.addf %2316, %2317 : f32
      linalg.yield %2318 : f32
    }
    %2319 = arith.constant {prov.region_id = "reduce_15", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} 2.560000e+02 : f32
    %2320 = tensor.splat %2319 {prov.region_id = "reduce_15", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} : tensor<1x32xf32>
    %2321 = tensor.empty() : tensor<1x32xf32>
    %2322 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2315, %2320 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%2321 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_15", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb277(%2323: f32, %2324: f32, %2325: f32):
      %2326 = arith.divf %2323, %2324 : f32
      linalg.yield %2326 : f32
    } -> tensor<1x32xf32>
    %2327 = tensor.collapse_shape %2322 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_15", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} : tensor<1x32xf32> into tensor<32xf32>
    %2328 = tensor.expand_shape %2327 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_15", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2329 = arith.constant {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} 1.000000e-05 : f32
    %2330 = tensor.splat %2329 {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} : tensor<1x32x1xf32>
    %2331 = tensor.empty() : tensor<1x32x1xf32>
    %2332 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2328, %2330 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2331 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb278(%2333: f32, %2334: f32, %2335: f32):
      %2336 = arith.addf %2333, %2334 : f32
      linalg.yield %2336 : f32
    } -> tensor<1x32x1xf32>
    %2337 = tensor.empty() : tensor<1x32x1xf32>
    %2338 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2332 : tensor<1x32x1xf32>) outs(%2337 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_5", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb279(%2339: f32, %2340: f32):
      %2341 = math.rsqrt %2339 : f32
      linalg.yield %2341 : f32
    } -> tensor<1x32x1xf32>
    %2342 = tensor.empty() : tensor<1x32x256xf32>
    %2343 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2306, %2338 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2342 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_59", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb280(%2344: f32, %2345: f32, %2346: f32):
      %2347 = arith.mulf %2344, %2345 : f32
      linalg.yield %2347 : f32
    } -> tensor<1x32x256xf32>
    %2348 = tensor.empty() : tensor<1x32x256xf32>
    %2349 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%68, %2343 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%2348 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_60", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.attn_sub_norm"} {
    ^bb281(%2350: f32, %2351: f32, %2352: f32):
      %2353 = arith.mulf %2350, %2351 : f32
      linalg.yield %2353 : f32
    } -> tensor<1x32x256xf32>
    %2354 = tensor.empty() : tensor<1x32x256xf32>
    %2355 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2349 : tensor<1x32x256xf32>) outs(%2354 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_20", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb282(%2356: f32, %2357: f32):
      %2358 = math.absf %2356 : f32
      linalg.yield %2358 : f32
    } -> tensor<1x32x256xf32>
    %2359 = arith.constant {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} 0xff800000 : f32
    %2360 = arith.constant {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} 0 : i64
    %2361 = tensor.splat %2359 {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<1x32xf32>
    %2362 = tensor.splat %2360 {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<1x32xi64>
    %2363, %2364 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2355 : tensor<1x32x256xf32>) outs(%2361, %2362 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb283(%2365: f32, %2366: f32, %2367: i64):
      %2368 = linalg.index 2 : index
      %2369 = arith.index_cast %2368 : index to i64
      %2370 = arith.cmpf ogt, %2365, %2366 : f32
      %2371 = arith.select %2370, %2365, %2366 : f32
      %2372 = arith.select %2370, %2369, %2367 : i64
      linalg.yield %2371, %2372 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %2373 = tensor.collapse_shape %2363 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %2374 = tensor.expand_shape %2373 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2375 = tensor.collapse_shape %2364 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %2376 = tensor.expand_shape %2375 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_10", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %2377 = tensor.empty() : tensor<1x32x1xf32>
    %2378 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2374 : tensor<1x32x1xf32>) outs(%2377 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "minmax_41", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb284(%2379: f32, %2380: f32):
      %2381 = arith.constant 1.000000e-05 : f32
      %2382 = arith.maximumf %2379, %2381 : f32
      linalg.yield %2382 : f32
    } -> tensor<1x32x1xf32>
    %2383 = tensor.empty() : tensor<1x32x1xf32>
    %2384 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2378 : tensor<1x32x1xf32>) outs(%2383 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_20", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb285(%2385: f32, %2386: f32):
      %2387 = arith.constant 1.000000e+00 : f32
      %2388 = arith.divf %2387, %2385 : f32
      linalg.yield %2388 : f32
    } -> tensor<1x32x1xf32>
    %2389 = arith.constant {prov.region_id = "mul_61", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} 1.270000e+02 : f32
    %2390 = tensor.splat %2389 {prov.region_id = "mul_61", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<1x32x1xf32>
    %2391 = tensor.empty() : tensor<1x32x1xf32>
    %2392 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2384, %2390 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2391 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_61", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb286(%2393: f32, %2394: f32, %2395: f32):
      %2396 = arith.mulf %2393, %2394 : f32
      linalg.yield %2396 : f32
    } -> tensor<1x32x1xf32>
    %2397 = tensor.empty() : tensor<1x32x256xf32>
    %2398 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2349, %2392 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2397 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_62", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb287(%2399: f32, %2400: f32, %2401: f32):
      %2402 = arith.mulf %2399, %2400 : f32
      linalg.yield %2402 : f32
    } -> tensor<1x32x256xf32>
    %2403 = tensor.empty() : tensor<1x32x256xf32>
    %2404 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2398 : tensor<1x32x256xf32>) outs(%2403 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_20", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb288(%2405: f32, %2406: f32):
      %2407 = math.roundeven %2405 : f32
      linalg.yield %2407 : f32
    } -> tensor<1x32x256xf32>
    %2408 = tensor.empty() : tensor<1x32x256xf32>
    %2409 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2404 : tensor<1x32x256xf32>) outs(%2408 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_42", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb289(%2410: f32, %2411: f32):
      %2412 = arith.constant -1.280000e+02 : f32
      %2413 = arith.maximumf %2410, %2412 : f32
      %2414 = arith.constant 1.270000e+02 : f32
      %2415 = arith.minimumf %2413, %2414 : f32
      linalg.yield %2415 : f32
    } -> tensor<1x32x256xf32>
    %2416 = tensor.empty() : tensor<1x32x256xf32>
    %2417 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2409, %2392 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2416 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_22", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb290(%2418: f32, %2419: f32, %2420: f32):
      %2421 = arith.divf %2418, %2419 : f32
      linalg.yield %2421 : f32
    } -> tensor<1x32x256xf32>
    %2422 = tensor.empty() : tensor<256x256xf32>
    %2423 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67 : tensor<256x256xf32>) outs(%2422 : tensor<256x256xf32>) attrs =  {prov.region_id = "abs_21", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb291(%2424: f32, %2425: f32):
      %2426 = math.absf %2424 : f32
      linalg.yield %2426 : f32
    } -> tensor<256x256xf32>
    %2427 = arith.constant {prov.region_id = "reduce_16", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} 0.000000e+00 : f32
    %2428 = tensor.splat %2427 {prov.region_id = "reduce_16", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<f32>
    %2429 = linalg.reduce ins(%2423:tensor<256x256xf32>) outs(%2428:tensor<f32>) dimensions = [0, 1]
    (%2430: f32, %2431: f32) {
      %2432 = arith.addf %2430, %2431 : f32
      linalg.yield %2432 : f32
    }
    %2433 = arith.constant {prov.region_id = "reduce_16", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} 6.553600e+04 : f32
    %2434 = tensor.splat %2433 {prov.region_id = "reduce_16", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<f32>
    %2435 = tensor.empty() : tensor<f32>
    %2436 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2429, %2434 : tensor<f32>, tensor<f32>) outs(%2435 : tensor<f32>) attrs =  {prov.region_id = "reduce_16", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb292(%2437: f32, %2438: f32, %2439: f32):
      %2440 = arith.divf %2437, %2438 : f32
      linalg.yield %2440 : f32
    } -> tensor<f32>
    %2441 = tensor.empty() : tensor<f32>
    %2442 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2436 : tensor<f32>) outs(%2441 : tensor<f32>) attrs =  {prov.region_id = "minmax_43", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb293(%2443: f32, %2444: f32):
      %2445 = arith.constant 1.000000e-05 : f32
      %2446 = arith.maximumf %2443, %2445 : f32
      linalg.yield %2446 : f32
    } -> tensor<f32>
    %2447 = tensor.empty() : tensor<f32>
    %2448 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2442 : tensor<f32>) outs(%2447 : tensor<f32>) attrs =  {prov.region_id = "elementwise_21", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb294(%2449: f32, %2450: f32):
      %2451 = arith.constant 1.000000e+00 : f32
      %2452 = arith.divf %2451, %2449 : f32
      linalg.yield %2452 : f32
    } -> tensor<f32>
    %2453 = arith.constant {prov.region_id = "mul_63", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} 1.000000e+00 : f32
    %2454 = tensor.splat %2453 {prov.region_id = "mul_63", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<f32>
    %2455 = tensor.empty() : tensor<f32>
    %2456 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2448, %2454 : tensor<f32>, tensor<f32>) outs(%2455 : tensor<f32>) attrs =  {prov.region_id = "mul_63", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb295(%2457: f32, %2458: f32, %2459: f32):
      %2460 = arith.mulf %2457, %2458 : f32
      linalg.yield %2460 : f32
    } -> tensor<f32>
    %2461 = tensor.empty() : tensor<256x256xf32>
    %2462 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67, %2456 : tensor<256x256xf32>, tensor<f32>) outs(%2461 : tensor<256x256xf32>) attrs =  {prov.region_id = "mul_64", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb296(%2463: f32, %2464: f32, %2465: f32):
      %2466 = arith.mulf %2463, %2464 : f32
      linalg.yield %2466 : f32
    } -> tensor<256x256xf32>
    %2467 = tensor.empty() : tensor<256x256xf32>
    %2468 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2462 : tensor<256x256xf32>) outs(%2467 : tensor<256x256xf32>) attrs =  {prov.region_id = "round_21", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb297(%2469: f32, %2470: f32):
      %2471 = math.roundeven %2469 : f32
      linalg.yield %2471 : f32
    } -> tensor<256x256xf32>
    %2472 = tensor.empty() : tensor<256x256xf32>
    %2473 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2468 : tensor<256x256xf32>) outs(%2472 : tensor<256x256xf32>) attrs =  {prov.region_id = "minmax_44", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb298(%2474: f32, %2475: f32):
      %2476 = arith.constant -1.000000e+00 : f32
      %2477 = arith.maximumf %2474, %2476 : f32
      %2478 = arith.constant 1.000000e+00 : f32
      %2479 = arith.minimumf %2477, %2478 : f32
      linalg.yield %2479 : f32
    } -> tensor<256x256xf32>
    %2480 = tensor.empty() : tensor<256x256xf32>
    %2481 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2473, %2456 : tensor<256x256xf32>, tensor<f32>) outs(%2480 : tensor<256x256xf32>) attrs =  {prov.region_id = "div_23", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} {
    ^bb299(%2482: f32, %2483: f32, %2484: f32):
      %2485 = arith.divf %2482, %2483 : f32
      linalg.yield %2485 : f32
    } -> tensor<256x256xf32>
    %2486 = tensor.empty() : tensor<256x256xf32>
    %2487 = linalg.transpose ins(%2481:tensor<256x256xf32>) outs(%2486:tensor<256x256xf32>) permutation = [1, 0]
    %2488 = tensor.collapse_shape %2417 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %2489 = tensor.expand_shape %2488 [[0 : i64, 1 : i64]] output_shape [32, 256] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<8192xf32> into tensor<32x256xf32>
    %2490 = tensor.empty() : tensor<32x256xf32>
    %2491 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %2492 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%2491 : f32) outs(%2490 : tensor<32x256xf32>) -> tensor<32x256xf32>
    %2493 = linalg.matmul {prov.region_id = "matmul_14", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj", prov.transposed_b = "true"} ins(%2489, %2487 : tensor<32x256xf32>, tensor<256x256xf32>) outs(%2492 : tensor<32x256xf32>) -> tensor<32x256xf32>
    %2494 = tensor.collapse_shape %2493 [[0 : i64, 1 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<32x256xf32> into tensor<8192xf32>
    %2495 = tensor.expand_shape %2494 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 256] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.self_attn.o_proj"} : tensor<8192xf32> into tensor<1x32x256xf32>
    %2496 = tensor.empty() : tensor<1x32x256xf32>
    %2497 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1614, %2495 : tensor<1x32x256xf32>, tensor<1x32x256xf32>) outs(%2496 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1"} {
    ^bb300(%2498: f32, %2499: f32, %2500: f32):
      %2501 = arith.addf %2498, %2499 : f32
      linalg.yield %2501 : f32
    } -> tensor<1x32x256xf32>
    %2502 = tensor.empty() : tensor<1x32x256xf32>
    %2503 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2497 : tensor<1x32x256xf32>) outs(%2502 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb301(%2504: f32, %2505: f32):
      %2506 = arith.constant 2.000000e+00 : f32
      %2507 = math.powf %2504, %2506 : f32
      linalg.yield %2507 : f32
    } -> tensor<1x32x256xf32>
    %2508 = arith.constant {prov.region_id = "reduce_17", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} 0.000000e+00 : f32
    %2509 = tensor.splat %2508 {prov.region_id = "reduce_17", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<1x32xf32>
    %2510 = linalg.reduce ins(%2503:tensor<1x32x256xf32>) outs(%2509:tensor<1x32xf32>) dimensions = [2]
    (%2511: f32, %2512: f32) {
      %2513 = arith.addf %2511, %2512 : f32
      linalg.yield %2513 : f32
    }
    %2514 = arith.constant {prov.region_id = "reduce_17", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} 2.560000e+02 : f32
    %2515 = tensor.splat %2514 {prov.region_id = "reduce_17", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<1x32xf32>
    %2516 = tensor.empty() : tensor<1x32xf32>
    %2517 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2510, %2515 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%2516 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_17", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb302(%2518: f32, %2519: f32, %2520: f32):
      %2521 = arith.divf %2518, %2519 : f32
      linalg.yield %2521 : f32
    } -> tensor<1x32xf32>
    %2522 = tensor.collapse_shape %2517 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_17", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<1x32xf32> into tensor<32xf32>
    %2523 = tensor.expand_shape %2522 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_17", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2524 = arith.constant {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} 1.000000e-05 : f32
    %2525 = tensor.splat %2524 {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} : tensor<1x32x1xf32>
    %2526 = tensor.empty() : tensor<1x32x1xf32>
    %2527 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2523, %2525 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2526 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb303(%2528: f32, %2529: f32, %2530: f32):
      %2531 = arith.addf %2528, %2529 : f32
      linalg.yield %2531 : f32
    } -> tensor<1x32x1xf32>
    %2532 = tensor.empty() : tensor<1x32x1xf32>
    %2533 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2527 : tensor<1x32x1xf32>) outs(%2532 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_6", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb304(%2534: f32, %2535: f32):
      %2536 = math.rsqrt %2534 : f32
      linalg.yield %2536 : f32
    } -> tensor<1x32x1xf32>
    %2537 = tensor.empty() : tensor<1x32x256xf32>
    %2538 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2497, %2533 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2537 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_65", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb305(%2539: f32, %2540: f32, %2541: f32):
      %2542 = arith.mulf %2539, %2540 : f32
      linalg.yield %2542 : f32
    } -> tensor<1x32x256xf32>
    %2543 = tensor.empty() : tensor<1x32x256xf32>
    %2544 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %2538 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%2543 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_66", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.post_attention_layernorm"} {
    ^bb306(%2545: f32, %2546: f32, %2547: f32):
      %2548 = arith.mulf %2545, %2546 : f32
      linalg.yield %2548 : f32
    } -> tensor<1x32x256xf32>
    %2549 = tensor.empty() : tensor<1x32x256xf32>
    %2550 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2544 : tensor<1x32x256xf32>) outs(%2549 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_22", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb307(%2551: f32, %2552: f32):
      %2553 = math.absf %2551 : f32
      linalg.yield %2553 : f32
    } -> tensor<1x32x256xf32>
    %2554 = arith.constant {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} 0xff800000 : f32
    %2555 = arith.constant {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} 0 : i64
    %2556 = tensor.splat %2554 {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<1x32xf32>
    %2557 = tensor.splat %2555 {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<1x32xi64>
    %2558, %2559 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2550 : tensor<1x32x256xf32>) outs(%2556, %2557 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb308(%2560: f32, %2561: f32, %2562: i64):
      %2563 = linalg.index 2 : index
      %2564 = arith.index_cast %2563 : index to i64
      %2565 = arith.cmpf ogt, %2560, %2561 : f32
      %2566 = arith.select %2565, %2560, %2561 : f32
      %2567 = arith.select %2565, %2564, %2562 : i64
      linalg.yield %2566, %2567 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %2568 = tensor.collapse_shape %2558 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %2569 = tensor.expand_shape %2568 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2570 = tensor.collapse_shape %2559 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %2571 = tensor.expand_shape %2570 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_11", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %2572 = tensor.empty() : tensor<1x32x1xf32>
    %2573 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2569 : tensor<1x32x1xf32>) outs(%2572 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "minmax_45", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb309(%2574: f32, %2575: f32):
      %2576 = arith.constant 1.000000e-05 : f32
      %2577 = arith.maximumf %2574, %2576 : f32
      linalg.yield %2577 : f32
    } -> tensor<1x32x1xf32>
    %2578 = tensor.empty() : tensor<1x32x1xf32>
    %2579 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2573 : tensor<1x32x1xf32>) outs(%2578 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_22", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb310(%2580: f32, %2581: f32):
      %2582 = arith.constant 1.000000e+00 : f32
      %2583 = arith.divf %2582, %2580 : f32
      linalg.yield %2583 : f32
    } -> tensor<1x32x1xf32>
    %2584 = arith.constant {prov.region_id = "mul_67", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} 1.270000e+02 : f32
    %2585 = tensor.splat %2584 {prov.region_id = "mul_67", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<1x32x1xf32>
    %2586 = tensor.empty() : tensor<1x32x1xf32>
    %2587 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2579, %2585 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2586 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_67", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb311(%2588: f32, %2589: f32, %2590: f32):
      %2591 = arith.mulf %2588, %2589 : f32
      linalg.yield %2591 : f32
    } -> tensor<1x32x1xf32>
    %2592 = tensor.empty() : tensor<1x32x256xf32>
    %2593 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2544, %2587 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2592 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_68", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb312(%2594: f32, %2595: f32, %2596: f32):
      %2597 = arith.mulf %2594, %2595 : f32
      linalg.yield %2597 : f32
    } -> tensor<1x32x256xf32>
    %2598 = tensor.empty() : tensor<1x32x256xf32>
    %2599 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2593 : tensor<1x32x256xf32>) outs(%2598 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_22", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb313(%2600: f32, %2601: f32):
      %2602 = math.roundeven %2600 : f32
      linalg.yield %2602 : f32
    } -> tensor<1x32x256xf32>
    %2603 = tensor.empty() : tensor<1x32x256xf32>
    %2604 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2599 : tensor<1x32x256xf32>) outs(%2603 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_46", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb314(%2605: f32, %2606: f32):
      %2607 = arith.constant -1.280000e+02 : f32
      %2608 = arith.maximumf %2605, %2607 : f32
      %2609 = arith.constant 1.270000e+02 : f32
      %2610 = arith.minimumf %2608, %2609 : f32
      linalg.yield %2610 : f32
    } -> tensor<1x32x256xf32>
    %2611 = tensor.empty() : tensor<1x32x256xf32>
    %2612 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2604, %2587 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2611 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_24", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb315(%2613: f32, %2614: f32, %2615: f32):
      %2616 = arith.divf %2613, %2614 : f32
      linalg.yield %2616 : f32
    } -> tensor<1x32x256xf32>
    %2617 = tensor.empty() : tensor<512x256xf32>
    %2618 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%69 : tensor<512x256xf32>) outs(%2617 : tensor<512x256xf32>) attrs =  {prov.region_id = "abs_23", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb316(%2619: f32, %2620: f32):
      %2621 = math.absf %2619 : f32
      linalg.yield %2621 : f32
    } -> tensor<512x256xf32>
    %2622 = arith.constant {prov.region_id = "reduce_18", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} 0.000000e+00 : f32
    %2623 = tensor.splat %2622 {prov.region_id = "reduce_18", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<f32>
    %2624 = linalg.reduce ins(%2618:tensor<512x256xf32>) outs(%2623:tensor<f32>) dimensions = [0, 1]
    (%2625: f32, %2626: f32) {
      %2627 = arith.addf %2625, %2626 : f32
      linalg.yield %2627 : f32
    }
    %2628 = arith.constant {prov.region_id = "reduce_18", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} 1.310720e+05 : f32
    %2629 = tensor.splat %2628 {prov.region_id = "reduce_18", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<f32>
    %2630 = tensor.empty() : tensor<f32>
    %2631 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2624, %2629 : tensor<f32>, tensor<f32>) outs(%2630 : tensor<f32>) attrs =  {prov.region_id = "reduce_18", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb317(%2632: f32, %2633: f32, %2634: f32):
      %2635 = arith.divf %2632, %2633 : f32
      linalg.yield %2635 : f32
    } -> tensor<f32>
    %2636 = tensor.empty() : tensor<f32>
    %2637 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2631 : tensor<f32>) outs(%2636 : tensor<f32>) attrs =  {prov.region_id = "minmax_47", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb318(%2638: f32, %2639: f32):
      %2640 = arith.constant 1.000000e-05 : f32
      %2641 = arith.maximumf %2638, %2640 : f32
      linalg.yield %2641 : f32
    } -> tensor<f32>
    %2642 = tensor.empty() : tensor<f32>
    %2643 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2637 : tensor<f32>) outs(%2642 : tensor<f32>) attrs =  {prov.region_id = "elementwise_23", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb319(%2644: f32, %2645: f32):
      %2646 = arith.constant 1.000000e+00 : f32
      %2647 = arith.divf %2646, %2644 : f32
      linalg.yield %2647 : f32
    } -> tensor<f32>
    %2648 = arith.constant {prov.region_id = "mul_69", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} 1.000000e+00 : f32
    %2649 = tensor.splat %2648 {prov.region_id = "mul_69", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<f32>
    %2650 = tensor.empty() : tensor<f32>
    %2651 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2643, %2649 : tensor<f32>, tensor<f32>) outs(%2650 : tensor<f32>) attrs =  {prov.region_id = "mul_69", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb320(%2652: f32, %2653: f32, %2654: f32):
      %2655 = arith.mulf %2652, %2653 : f32
      linalg.yield %2655 : f32
    } -> tensor<f32>
    %2656 = tensor.empty() : tensor<512x256xf32>
    %2657 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%69, %2651 : tensor<512x256xf32>, tensor<f32>) outs(%2656 : tensor<512x256xf32>) attrs =  {prov.region_id = "mul_70", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb321(%2658: f32, %2659: f32, %2660: f32):
      %2661 = arith.mulf %2658, %2659 : f32
      linalg.yield %2661 : f32
    } -> tensor<512x256xf32>
    %2662 = tensor.empty() : tensor<512x256xf32>
    %2663 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2657 : tensor<512x256xf32>) outs(%2662 : tensor<512x256xf32>) attrs =  {prov.region_id = "round_23", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb322(%2664: f32, %2665: f32):
      %2666 = math.roundeven %2664 : f32
      linalg.yield %2666 : f32
    } -> tensor<512x256xf32>
    %2667 = tensor.empty() : tensor<512x256xf32>
    %2668 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2663 : tensor<512x256xf32>) outs(%2667 : tensor<512x256xf32>) attrs =  {prov.region_id = "minmax_48", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb323(%2669: f32, %2670: f32):
      %2671 = arith.constant -1.000000e+00 : f32
      %2672 = arith.maximumf %2669, %2671 : f32
      %2673 = arith.constant 1.000000e+00 : f32
      %2674 = arith.minimumf %2672, %2673 : f32
      linalg.yield %2674 : f32
    } -> tensor<512x256xf32>
    %2675 = tensor.empty() : tensor<512x256xf32>
    %2676 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2668, %2651 : tensor<512x256xf32>, tensor<f32>) outs(%2675 : tensor<512x256xf32>) attrs =  {prov.region_id = "div_25", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} {
    ^bb324(%2677: f32, %2678: f32, %2679: f32):
      %2680 = arith.divf %2677, %2678 : f32
      linalg.yield %2680 : f32
    } -> tensor<512x256xf32>
    %2681 = tensor.empty() : tensor<256x512xf32>
    %2682 = linalg.transpose ins(%2676:tensor<512x256xf32>) outs(%2681:tensor<256x512xf32>) permutation = [1, 0]
    %2683 = tensor.collapse_shape %2612 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %2684 = tensor.expand_shape %2683 [[0 : i64, 1 : i64]] output_shape [32, 256] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<8192xf32> into tensor<32x256xf32>
    %2685 = tensor.empty() : tensor<32x512xf32>
    %2686 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %2687 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%2686 : f32) outs(%2685 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %2688 = linalg.matmul {prov.region_id = "matmul_15", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj", prov.transposed_b = "true"} ins(%2684, %2682 : tensor<32x256xf32>, tensor<256x512xf32>) outs(%2687 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %2689 = tensor.collapse_shape %2688 [[0 : i64, 1 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<32x512xf32> into tensor<16384xf32>
    %2690 = tensor.expand_shape %2689 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 512] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.gate_proj"} : tensor<16384xf32> into tensor<1x32x512xf32>
    %2691 = tensor.empty() : tensor<1x32x512xf32>
    %2692 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2690 : tensor<1x32x512xf32>) outs(%2691 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "minmax_49", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp"} {
    ^bb325(%2693: f32, %2694: f32):
      %2695 = arith.constant 0.000000e+00 : f32
      %2696 = arith.maximumf %2693, %2695 : f32
      linalg.yield %2696 : f32
    } -> tensor<1x32x512xf32>
    %2697 = tensor.empty() : tensor<1x32x512xf32>
    %2698 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2692 : tensor<1x32x512xf32>) outs(%2697 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp"} {
    ^bb326(%2699: f32, %2700: f32):
      %2701 = arith.constant 2.000000e+00 : f32
      %2702 = math.powf %2699, %2701 : f32
      linalg.yield %2702 : f32
    } -> tensor<1x32x512xf32>
    %2703 = tensor.empty() : tensor<1x32x256xf32>
    %2704 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2544 : tensor<1x32x256xf32>) outs(%2703 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "abs_24", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb327(%2705: f32, %2706: f32):
      %2707 = math.absf %2705 : f32
      linalg.yield %2707 : f32
    } -> tensor<1x32x256xf32>
    %2708 = arith.constant {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} 0xff800000 : f32
    %2709 = arith.constant {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} 0 : i64
    %2710 = tensor.splat %2708 {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<1x32xf32>
    %2711 = tensor.splat %2709 {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<1x32xi64>
    %2712, %2713 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2704 : tensor<1x32x256xf32>) outs(%2710, %2711 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb328(%2714: f32, %2715: f32, %2716: i64):
      %2717 = linalg.index 2 : index
      %2718 = arith.index_cast %2717 : index to i64
      %2719 = arith.cmpf ogt, %2714, %2715 : f32
      %2720 = arith.select %2719, %2714, %2715 : f32
      %2721 = arith.select %2719, %2718, %2716 : i64
      linalg.yield %2720, %2721 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %2722 = tensor.collapse_shape %2712 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %2723 = tensor.expand_shape %2722 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2724 = tensor.collapse_shape %2713 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %2725 = tensor.expand_shape %2724 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_12", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %2726 = tensor.empty() : tensor<1x32x1xf32>
    %2727 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2723 : tensor<1x32x1xf32>) outs(%2726 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "minmax_50", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb329(%2728: f32, %2729: f32):
      %2730 = arith.constant 1.000000e-05 : f32
      %2731 = arith.maximumf %2728, %2730 : f32
      linalg.yield %2731 : f32
    } -> tensor<1x32x1xf32>
    %2732 = tensor.empty() : tensor<1x32x1xf32>
    %2733 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2727 : tensor<1x32x1xf32>) outs(%2732 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_24", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb330(%2734: f32, %2735: f32):
      %2736 = arith.constant 1.000000e+00 : f32
      %2737 = arith.divf %2736, %2734 : f32
      linalg.yield %2737 : f32
    } -> tensor<1x32x1xf32>
    %2738 = arith.constant {prov.region_id = "mul_71", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} 1.270000e+02 : f32
    %2739 = tensor.splat %2738 {prov.region_id = "mul_71", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<1x32x1xf32>
    %2740 = tensor.empty() : tensor<1x32x1xf32>
    %2741 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2733, %2739 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2740 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_71", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb331(%2742: f32, %2743: f32, %2744: f32):
      %2745 = arith.mulf %2742, %2743 : f32
      linalg.yield %2745 : f32
    } -> tensor<1x32x1xf32>
    %2746 = tensor.empty() : tensor<1x32x256xf32>
    %2747 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2544, %2741 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2746 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_72", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb332(%2748: f32, %2749: f32, %2750: f32):
      %2751 = arith.mulf %2748, %2749 : f32
      linalg.yield %2751 : f32
    } -> tensor<1x32x256xf32>
    %2752 = tensor.empty() : tensor<1x32x256xf32>
    %2753 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2747 : tensor<1x32x256xf32>) outs(%2752 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "round_24", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb333(%2754: f32, %2755: f32):
      %2756 = math.roundeven %2754 : f32
      linalg.yield %2756 : f32
    } -> tensor<1x32x256xf32>
    %2757 = tensor.empty() : tensor<1x32x256xf32>
    %2758 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2753 : tensor<1x32x256xf32>) outs(%2757 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "minmax_51", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb334(%2759: f32, %2760: f32):
      %2761 = arith.constant -1.280000e+02 : f32
      %2762 = arith.maximumf %2759, %2761 : f32
      %2763 = arith.constant 1.270000e+02 : f32
      %2764 = arith.minimumf %2762, %2763 : f32
      linalg.yield %2764 : f32
    } -> tensor<1x32x256xf32>
    %2765 = tensor.empty() : tensor<1x32x256xf32>
    %2766 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2758, %2741 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%2765 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "div_26", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb335(%2767: f32, %2768: f32, %2769: f32):
      %2770 = arith.divf %2767, %2768 : f32
      linalg.yield %2770 : f32
    } -> tensor<1x32x256xf32>
    %2771 = tensor.empty() : tensor<512x256xf32>
    %2772 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70 : tensor<512x256xf32>) outs(%2771 : tensor<512x256xf32>) attrs =  {prov.region_id = "abs_25", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb336(%2773: f32, %2774: f32):
      %2775 = math.absf %2773 : f32
      linalg.yield %2775 : f32
    } -> tensor<512x256xf32>
    %2776 = arith.constant {prov.region_id = "reduce_19", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} 0.000000e+00 : f32
    %2777 = tensor.splat %2776 {prov.region_id = "reduce_19", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<f32>
    %2778 = linalg.reduce ins(%2772:tensor<512x256xf32>) outs(%2777:tensor<f32>) dimensions = [0, 1]
    (%2779: f32, %2780: f32) {
      %2781 = arith.addf %2779, %2780 : f32
      linalg.yield %2781 : f32
    }
    %2782 = arith.constant {prov.region_id = "reduce_19", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} 1.310720e+05 : f32
    %2783 = tensor.splat %2782 {prov.region_id = "reduce_19", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<f32>
    %2784 = tensor.empty() : tensor<f32>
    %2785 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2778, %2783 : tensor<f32>, tensor<f32>) outs(%2784 : tensor<f32>) attrs =  {prov.region_id = "reduce_19", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb337(%2786: f32, %2787: f32, %2788: f32):
      %2789 = arith.divf %2786, %2787 : f32
      linalg.yield %2789 : f32
    } -> tensor<f32>
    %2790 = tensor.empty() : tensor<f32>
    %2791 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2785 : tensor<f32>) outs(%2790 : tensor<f32>) attrs =  {prov.region_id = "minmax_52", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb338(%2792: f32, %2793: f32):
      %2794 = arith.constant 1.000000e-05 : f32
      %2795 = arith.maximumf %2792, %2794 : f32
      linalg.yield %2795 : f32
    } -> tensor<f32>
    %2796 = tensor.empty() : tensor<f32>
    %2797 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2791 : tensor<f32>) outs(%2796 : tensor<f32>) attrs =  {prov.region_id = "elementwise_25", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb339(%2798: f32, %2799: f32):
      %2800 = arith.constant 1.000000e+00 : f32
      %2801 = arith.divf %2800, %2798 : f32
      linalg.yield %2801 : f32
    } -> tensor<f32>
    %2802 = arith.constant {prov.region_id = "mul_73", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} 1.000000e+00 : f32
    %2803 = tensor.splat %2802 {prov.region_id = "mul_73", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<f32>
    %2804 = tensor.empty() : tensor<f32>
    %2805 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2797, %2803 : tensor<f32>, tensor<f32>) outs(%2804 : tensor<f32>) attrs =  {prov.region_id = "mul_73", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb340(%2806: f32, %2807: f32, %2808: f32):
      %2809 = arith.mulf %2806, %2807 : f32
      linalg.yield %2809 : f32
    } -> tensor<f32>
    %2810 = tensor.empty() : tensor<512x256xf32>
    %2811 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %2805 : tensor<512x256xf32>, tensor<f32>) outs(%2810 : tensor<512x256xf32>) attrs =  {prov.region_id = "mul_74", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb341(%2812: f32, %2813: f32, %2814: f32):
      %2815 = arith.mulf %2812, %2813 : f32
      linalg.yield %2815 : f32
    } -> tensor<512x256xf32>
    %2816 = tensor.empty() : tensor<512x256xf32>
    %2817 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2811 : tensor<512x256xf32>) outs(%2816 : tensor<512x256xf32>) attrs =  {prov.region_id = "round_25", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb342(%2818: f32, %2819: f32):
      %2820 = math.roundeven %2818 : f32
      linalg.yield %2820 : f32
    } -> tensor<512x256xf32>
    %2821 = tensor.empty() : tensor<512x256xf32>
    %2822 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2817 : tensor<512x256xf32>) outs(%2821 : tensor<512x256xf32>) attrs =  {prov.region_id = "minmax_53", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb343(%2823: f32, %2824: f32):
      %2825 = arith.constant -1.000000e+00 : f32
      %2826 = arith.maximumf %2823, %2825 : f32
      %2827 = arith.constant 1.000000e+00 : f32
      %2828 = arith.minimumf %2826, %2827 : f32
      linalg.yield %2828 : f32
    } -> tensor<512x256xf32>
    %2829 = tensor.empty() : tensor<512x256xf32>
    %2830 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2822, %2805 : tensor<512x256xf32>, tensor<f32>) outs(%2829 : tensor<512x256xf32>) attrs =  {prov.region_id = "div_27", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} {
    ^bb344(%2831: f32, %2832: f32, %2833: f32):
      %2834 = arith.divf %2831, %2832 : f32
      linalg.yield %2834 : f32
    } -> tensor<512x256xf32>
    %2835 = tensor.empty() : tensor<256x512xf32>
    %2836 = linalg.transpose ins(%2830:tensor<512x256xf32>) outs(%2835:tensor<256x512xf32>) permutation = [1, 0]
    %2837 = tensor.collapse_shape %2766 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %2838 = tensor.expand_shape %2837 [[0 : i64, 1 : i64]] output_shape [32, 256] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<8192xf32> into tensor<32x256xf32>
    %2839 = tensor.empty() : tensor<32x512xf32>
    %2840 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %2841 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%2840 : f32) outs(%2839 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %2842 = linalg.matmul {prov.region_id = "matmul_16", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj", prov.transposed_b = "true"} ins(%2838, %2836 : tensor<32x256xf32>, tensor<256x512xf32>) outs(%2841 : tensor<32x512xf32>) -> tensor<32x512xf32>
    %2843 = tensor.collapse_shape %2842 [[0 : i64, 1 : i64]] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<32x512xf32> into tensor<16384xf32>
    %2844 = tensor.expand_shape %2843 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 512] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.up_proj"} : tensor<16384xf32> into tensor<1x32x512xf32>
    %2845 = tensor.empty() : tensor<1x32x512xf32>
    %2846 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2698, %2844 : tensor<1x32x512xf32>, tensor<1x32x512xf32>) outs(%2845 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_75", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp"} {
    ^bb345(%2847: f32, %2848: f32, %2849: f32):
      %2850 = arith.mulf %2847, %2848 : f32
      linalg.yield %2850 : f32
    } -> tensor<1x32x512xf32>
    %2851 = tensor.empty() : tensor<1x32x512xf32>
    %2852 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2846 : tensor<1x32x512xf32>) outs(%2851 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "pow_9", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb346(%2853: f32, %2854: f32):
      %2855 = arith.constant 2.000000e+00 : f32
      %2856 = math.powf %2853, %2855 : f32
      linalg.yield %2856 : f32
    } -> tensor<1x32x512xf32>
    %2857 = arith.constant {prov.region_id = "reduce_20", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} 0.000000e+00 : f32
    %2858 = tensor.splat %2857 {prov.region_id = "reduce_20", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} : tensor<1x32xf32>
    %2859 = linalg.reduce ins(%2852:tensor<1x32x512xf32>) outs(%2858:tensor<1x32xf32>) dimensions = [2]
    (%2860: f32, %2861: f32) {
      %2862 = arith.addf %2860, %2861 : f32
      linalg.yield %2862 : f32
    }
    %2863 = arith.constant {prov.region_id = "reduce_20", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} 5.120000e+02 : f32
    %2864 = tensor.splat %2863 {prov.region_id = "reduce_20", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} : tensor<1x32xf32>
    %2865 = tensor.empty() : tensor<1x32xf32>
    %2866 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2859, %2864 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%2865 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_20", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb347(%2867: f32, %2868: f32, %2869: f32):
      %2870 = arith.divf %2867, %2868 : f32
      linalg.yield %2870 : f32
    } -> tensor<1x32xf32>
    %2871 = tensor.collapse_shape %2866 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_20", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} : tensor<1x32xf32> into tensor<32xf32>
    %2872 = tensor.expand_shape %2871 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_20", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2873 = arith.constant {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} 1.000000e-05 : f32
    %2874 = tensor.splat %2873 {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} : tensor<1x32x1xf32>
    %2875 = tensor.empty() : tensor<1x32x1xf32>
    %2876 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2872, %2874 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2875 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb348(%2877: f32, %2878: f32, %2879: f32):
      %2880 = arith.addf %2877, %2878 : f32
      linalg.yield %2880 : f32
    } -> tensor<1x32x1xf32>
    %2881 = tensor.empty() : tensor<1x32x1xf32>
    %2882 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2876 : tensor<1x32x1xf32>) outs(%2881 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_7", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb349(%2883: f32, %2884: f32):
      %2885 = math.rsqrt %2883 : f32
      linalg.yield %2885 : f32
    } -> tensor<1x32x1xf32>
    %2886 = tensor.empty() : tensor<1x32x512xf32>
    %2887 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2846, %2882 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%2886 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_76", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb350(%2888: f32, %2889: f32, %2890: f32):
      %2891 = arith.mulf %2888, %2889 : f32
      linalg.yield %2891 : f32
    } -> tensor<1x32x512xf32>
    %2892 = tensor.empty() : tensor<1x32x512xf32>
    %2893 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%72, %2887 : tensor<512xf32>, tensor<1x32x512xf32>) outs(%2892 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_77", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.ffn_sub_norm"} {
    ^bb351(%2894: f32, %2895: f32, %2896: f32):
      %2897 = arith.mulf %2894, %2895 : f32
      linalg.yield %2897 : f32
    } -> tensor<1x32x512xf32>
    %2898 = tensor.empty() : tensor<1x32x512xf32>
    %2899 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2893 : tensor<1x32x512xf32>) outs(%2898 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "abs_26", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb352(%2900: f32, %2901: f32):
      %2902 = math.absf %2900 : f32
      linalg.yield %2902 : f32
    } -> tensor<1x32x512xf32>
    %2903 = arith.constant {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} 0xff800000 : f32
    %2904 = arith.constant {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} 0 : i64
    %2905 = tensor.splat %2903 {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<1x32xf32>
    %2906 = tensor.splat %2904 {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<1x32xi64>
    %2907, %2908 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2899 : tensor<1x32x512xf32>) outs(%2905, %2906 : tensor<1x32xf32>, tensor<1x32xi64>) attrs =  {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb353(%2909: f32, %2910: f32, %2911: i64):
      %2912 = linalg.index 2 : index
      %2913 = arith.index_cast %2912 : index to i64
      %2914 = arith.cmpf ogt, %2909, %2910 : f32
      %2915 = arith.select %2914, %2909, %2910 : f32
      %2916 = arith.select %2914, %2913, %2911 : i64
      linalg.yield %2915, %2916 : f32, i64
    } -> (tensor<1x32xf32>, tensor<1x32xi64>)
    %2917 = tensor.collapse_shape %2907 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<1x32xf32> into tensor<32xf32>
    %2918 = tensor.expand_shape %2917 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<32xf32> into tensor<1x32x1xf32>
    %2919 = tensor.collapse_shape %2908 [[0 : i64, 1 : i64]] {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<1x32xi64> into tensor<32xi64>
    %2920 = tensor.expand_shape %2919 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "arg_reduce_13", prov.family = "arg_reduce", prov._pattern_hint = "aten_max_dim", prov.op = "aten_max_dim", prov.aten = "aten.max.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<32xi64> into tensor<1x32x1xi64>
    %2921 = tensor.empty() : tensor<1x32x1xf32>
    %2922 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2918 : tensor<1x32x1xf32>) outs(%2921 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "minmax_54", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb354(%2923: f32, %2924: f32):
      %2925 = arith.constant 1.000000e-05 : f32
      %2926 = arith.maximumf %2923, %2925 : f32
      linalg.yield %2926 : f32
    } -> tensor<1x32x1xf32>
    %2927 = tensor.empty() : tensor<1x32x1xf32>
    %2928 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2922 : tensor<1x32x1xf32>) outs(%2927 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "elementwise_26", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb355(%2929: f32, %2930: f32):
      %2931 = arith.constant 1.000000e+00 : f32
      %2932 = arith.divf %2931, %2929 : f32
      linalg.yield %2932 : f32
    } -> tensor<1x32x1xf32>
    %2933 = arith.constant {prov.region_id = "mul_78", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} 1.270000e+02 : f32
    %2934 = tensor.splat %2933 {prov.region_id = "mul_78", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<1x32x1xf32>
    %2935 = tensor.empty() : tensor<1x32x1xf32>
    %2936 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2928, %2934 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%2935 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "mul_78", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb356(%2937: f32, %2938: f32, %2939: f32):
      %2940 = arith.mulf %2937, %2938 : f32
      linalg.yield %2940 : f32
    } -> tensor<1x32x1xf32>
    %2941 = tensor.empty() : tensor<1x32x512xf32>
    %2942 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2893, %2936 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%2941 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "mul_79", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb357(%2943: f32, %2944: f32, %2945: f32):
      %2946 = arith.mulf %2943, %2944 : f32
      linalg.yield %2946 : f32
    } -> tensor<1x32x512xf32>
    %2947 = tensor.empty() : tensor<1x32x512xf32>
    %2948 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2942 : tensor<1x32x512xf32>) outs(%2947 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "round_26", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb358(%2949: f32, %2950: f32):
      %2951 = math.roundeven %2949 : f32
      linalg.yield %2951 : f32
    } -> tensor<1x32x512xf32>
    %2952 = tensor.empty() : tensor<1x32x512xf32>
    %2953 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2948 : tensor<1x32x512xf32>) outs(%2952 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "minmax_55", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb359(%2954: f32, %2955: f32):
      %2956 = arith.constant -1.280000e+02 : f32
      %2957 = arith.maximumf %2954, %2956 : f32
      %2958 = arith.constant 1.270000e+02 : f32
      %2959 = arith.minimumf %2957, %2958 : f32
      linalg.yield %2959 : f32
    } -> tensor<1x32x512xf32>
    %2960 = tensor.empty() : tensor<1x32x512xf32>
    %2961 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2953, %2936 : tensor<1x32x512xf32>, tensor<1x32x1xf32>) outs(%2960 : tensor<1x32x512xf32>) attrs =  {prov.region_id = "div_28", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb360(%2962: f32, %2963: f32, %2964: f32):
      %2965 = arith.divf %2962, %2963 : f32
      linalg.yield %2965 : f32
    } -> tensor<1x32x512xf32>
    %2966 = tensor.empty() : tensor<256x512xf32>
    %2967 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%71 : tensor<256x512xf32>) outs(%2966 : tensor<256x512xf32>) attrs =  {prov.region_id = "abs_27", prov._pattern_hint = "abs", prov.op = "abs", prov.family = "elementwise", prov.aten = "aten.abs.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb361(%2968: f32, %2969: f32):
      %2970 = math.absf %2968 : f32
      linalg.yield %2970 : f32
    } -> tensor<256x512xf32>
    %2971 = arith.constant {prov.region_id = "reduce_21", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} 0.000000e+00 : f32
    %2972 = tensor.splat %2971 {prov.region_id = "reduce_21", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<f32>
    %2973 = linalg.reduce ins(%2967:tensor<256x512xf32>) outs(%2972:tensor<f32>) dimensions = [0, 1]
    (%2974: f32, %2975: f32) {
      %2976 = arith.addf %2974, %2975 : f32
      linalg.yield %2976 : f32
    }
    %2977 = arith.constant {prov.region_id = "reduce_21", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} 1.310720e+05 : f32
    %2978 = tensor.splat %2977 {prov.region_id = "reduce_21", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<f32>
    %2979 = tensor.empty() : tensor<f32>
    %2980 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2973, %2978 : tensor<f32>, tensor<f32>) outs(%2979 : tensor<f32>) attrs =  {prov.region_id = "reduce_21", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb362(%2981: f32, %2982: f32, %2983: f32):
      %2984 = arith.divf %2981, %2982 : f32
      linalg.yield %2984 : f32
    } -> tensor<f32>
    %2985 = tensor.empty() : tensor<f32>
    %2986 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2980 : tensor<f32>) outs(%2985 : tensor<f32>) attrs =  {prov.region_id = "minmax_56", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb363(%2987: f32, %2988: f32):
      %2989 = arith.constant 1.000000e-05 : f32
      %2990 = arith.maximumf %2987, %2989 : f32
      linalg.yield %2990 : f32
    } -> tensor<f32>
    %2991 = tensor.empty() : tensor<f32>
    %2992 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2986 : tensor<f32>) outs(%2991 : tensor<f32>) attrs =  {prov.region_id = "elementwise_27", prov.family = "elementwise", prov._pattern_hint = "elementwise", prov.op = "elementwise", prov.aten = "aten.reciprocal.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb364(%2993: f32, %2994: f32):
      %2995 = arith.constant 1.000000e+00 : f32
      %2996 = arith.divf %2995, %2993 : f32
      linalg.yield %2996 : f32
    } -> tensor<f32>
    %2997 = arith.constant {prov.region_id = "mul_80", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} 1.000000e+00 : f32
    %2998 = tensor.splat %2997 {prov.region_id = "mul_80", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<f32>
    %2999 = tensor.empty() : tensor<f32>
    %3000 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2992, %2998 : tensor<f32>, tensor<f32>) outs(%2999 : tensor<f32>) attrs =  {prov.region_id = "mul_80", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb365(%3001: f32, %3002: f32, %3003: f32):
      %3004 = arith.mulf %3001, %3002 : f32
      linalg.yield %3004 : f32
    } -> tensor<f32>
    %3005 = tensor.empty() : tensor<256x512xf32>
    %3006 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%71, %3000 : tensor<256x512xf32>, tensor<f32>) outs(%3005 : tensor<256x512xf32>) attrs =  {prov.region_id = "mul_81", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb366(%3007: f32, %3008: f32, %3009: f32):
      %3010 = arith.mulf %3007, %3008 : f32
      linalg.yield %3010 : f32
    } -> tensor<256x512xf32>
    %3011 = tensor.empty() : tensor<256x512xf32>
    %3012 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3006 : tensor<256x512xf32>) outs(%3011 : tensor<256x512xf32>) attrs =  {prov.region_id = "round_27", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb367(%3013: f32, %3014: f32):
      %3015 = math.roundeven %3013 : f32
      linalg.yield %3015 : f32
    } -> tensor<256x512xf32>
    %3016 = tensor.empty() : tensor<256x512xf32>
    %3017 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3012 : tensor<256x512xf32>) outs(%3016 : tensor<256x512xf32>) attrs =  {prov.region_id = "minmax_57", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb368(%3018: f32, %3019: f32):
      %3020 = arith.constant -1.000000e+00 : f32
      %3021 = arith.maximumf %3018, %3020 : f32
      %3022 = arith.constant 1.000000e+00 : f32
      %3023 = arith.minimumf %3021, %3022 : f32
      linalg.yield %3023 : f32
    } -> tensor<256x512xf32>
    %3024 = tensor.empty() : tensor<256x512xf32>
    %3025 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3017, %3000 : tensor<256x512xf32>, tensor<f32>) outs(%3024 : tensor<256x512xf32>) attrs =  {prov.region_id = "div_29", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} {
    ^bb369(%3026: f32, %3027: f32, %3028: f32):
      %3029 = arith.divf %3026, %3027 : f32
      linalg.yield %3029 : f32
    } -> tensor<256x512xf32>
    %3030 = tensor.empty() : tensor<512x256xf32>
    %3031 = linalg.transpose ins(%3025:tensor<256x512xf32>) outs(%3030:tensor<512x256xf32>) permutation = [1, 0]
    %3032 = tensor.collapse_shape %2961 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<1x32x512xf32> into tensor<16384xf32>
    %3033 = tensor.expand_shape %3032 [[0 : i64, 1 : i64]] output_shape [32, 512] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<16384xf32> into tensor<32x512xf32>
    %3034 = tensor.empty() : tensor<32x256xf32>
    %3035 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %3036 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%3035 : f32) outs(%3034 : tensor<32x256xf32>) -> tensor<32x256xf32>
    %3037 = linalg.matmul {prov.region_id = "matmul_17", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj", prov.transposed_b = "true"} ins(%3033, %3031 : tensor<32x512xf32>, tensor<512x256xf32>) outs(%3036 : tensor<32x256xf32>) -> tensor<32x256xf32>
    %3038 = tensor.collapse_shape %3037 [[0 : i64, 1 : i64]] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<32x256xf32> into tensor<8192xf32>
    %3039 = tensor.expand_shape %3038 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 256] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1.mlp.down_proj"} : tensor<8192xf32> into tensor<1x32x256xf32>
    %3040 = tensor.empty() : tensor<1x32x256xf32>
    %3041 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2497, %3039 : tensor<1x32x256xf32>, tensor<1x32x256xf32>) outs(%3040 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.layers.1"} {
    ^bb370(%3042: f32, %3043: f32, %3044: f32):
      %3045 = arith.addf %3042, %3043 : f32
      linalg.yield %3045 : f32
    } -> tensor<1x32x256xf32>
    %3046 = tensor.empty() : tensor<1x32x256xf32>
    %3047 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3041 : tensor<1x32x256xf32>) outs(%3046 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "pow_10", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb371(%3048: f32, %3049: f32):
      %3050 = arith.constant 2.000000e+00 : f32
      %3051 = math.powf %3048, %3050 : f32
      linalg.yield %3051 : f32
    } -> tensor<1x32x256xf32>
    %3052 = arith.constant {prov.region_id = "reduce_22", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} 0.000000e+00 : f32
    %3053 = tensor.splat %3052 {prov.region_id = "reduce_22", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<1x32xf32>
    %3054 = linalg.reduce ins(%3047:tensor<1x32x256xf32>) outs(%3053:tensor<1x32xf32>) dimensions = [2]
    (%3055: f32, %3056: f32) {
      %3057 = arith.addf %3055, %3056 : f32
      linalg.yield %3057 : f32
    }
    %3058 = arith.constant {prov.region_id = "reduce_22", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} 2.560000e+02 : f32
    %3059 = tensor.splat %3058 {prov.region_id = "reduce_22", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<1x32xf32>
    %3060 = tensor.empty() : tensor<1x32xf32>
    %3061 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3054, %3059 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%3060 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_22", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb372(%3062: f32, %3063: f32, %3064: f32):
      %3065 = arith.divf %3062, %3063 : f32
      linalg.yield %3065 : f32
    } -> tensor<1x32xf32>
    %3066 = tensor.collapse_shape %3061 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_22", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<1x32xf32> into tensor<32xf32>
    %3067 = tensor.expand_shape %3066 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_22", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<32xf32> into tensor<1x32x1xf32>
    %3068 = arith.constant {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} 1.000000e-05 : f32
    %3069 = tensor.splat %3068 {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} : tensor<1x32x1xf32>
    %3070 = tensor.empty() : tensor<1x32x1xf32>
    %3071 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3067, %3069 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%3070 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb373(%3072: f32, %3073: f32, %3074: f32):
      %3075 = arith.addf %3072, %3073 : f32
      linalg.yield %3075 : f32
    } -> tensor<1x32x1xf32>
    %3076 = tensor.empty() : tensor<1x32x1xf32>
    %3077 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3071 : tensor<1x32x1xf32>) outs(%3076 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_8", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb374(%3078: f32, %3079: f32):
      %3080 = math.rsqrt %3078 : f32
      linalg.yield %3080 : f32
    } -> tensor<1x32x1xf32>
    %3081 = tensor.empty() : tensor<1x32x256xf32>
    %3082 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3041, %3077 : tensor<1x32x256xf32>, tensor<1x32x1xf32>) outs(%3081 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_82", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb375(%3083: f32, %3084: f32, %3085: f32):
      %3086 = arith.mulf %3083, %3084 : f32
      linalg.yield %3086 : f32
    } -> tensor<1x32x256xf32>
    %3087 = tensor.empty() : tensor<1x32x256xf32>
    %3088 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%75, %3082 : tensor<256xf32>, tensor<1x32x256xf32>) outs(%3087 : tensor<1x32x256xf32>) attrs =  {prov.region_id = "mul_83", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.model.norm"} {
    ^bb376(%3089: f32, %3090: f32, %3091: f32):
      %3092 = arith.mulf %3089, %3090 : f32
      linalg.yield %3092 : f32
    } -> tensor<1x32x256xf32>
    %3093 = tensor.empty() : tensor<256x1024xf32>
    %3094 = linalg.transpose ins(%76:tensor<1024x256xf32>) outs(%3093:tensor<256x1024xf32>) permutation = [1, 0]
    %3095 = tensor.collapse_shape %3088 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.lm_head"} : tensor<1x32x256xf32> into tensor<8192xf32>
    %3096 = tensor.expand_shape %3095 [[0 : i64, 1 : i64]] output_shape [32, 256] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.lm_head"} : tensor<8192xf32> into tensor<32x256xf32>
    %3097 = tensor.empty() : tensor<32x1024xf32>
    %3098 = arith.constant {prov.module = "vla"} 0.000000e+00 : f32
    %3099 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "vla"} ins(%3098 : f32) outs(%3097 : tensor<32x1024xf32>) -> tensor<32x1024xf32>
    %3100 = linalg.matmul {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.lm_head", prov.transposed_b = "true"} ins(%3096, %3094 : tensor<32x256xf32>, tensor<256x1024xf32>) outs(%3099 : tensor<32x1024xf32>) -> tensor<32x1024xf32>
    %3101 = tensor.collapse_shape %3100 [[0 : i64, 1 : i64]] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.lm_head"} : tensor<32x1024xf32> into tensor<32768xf32>
    %3102 = tensor.expand_shape %3101 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1024] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "vla", prov.fqn = "vla.language_model.lm_head"} : tensor<32768xf32> into tensor<1x32x1024xf32>
    func.return %3102 : tensor<1x32x1024xf32>
  }
}
