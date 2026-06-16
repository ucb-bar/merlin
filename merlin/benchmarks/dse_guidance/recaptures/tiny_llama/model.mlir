builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<32000x2048xf32>, %1: tensor<2048x2048xf32>, %2: tensor<256x2048xf32>, %3: tensor<256x2048xf32>, %4: tensor<2048x2048xf32>, %5: tensor<5632x2048xf32>, %6: tensor<5632x2048xf32>, %7: tensor<2048x5632xf32>, %8: tensor<2048xf32>, %9: tensor<2048xf32>, %10: tensor<2048x2048xf32>, %11: tensor<256x2048xf32>, %12: tensor<256x2048xf32>, %13: tensor<2048x2048xf32>, %14: tensor<5632x2048xf32>, %15: tensor<5632x2048xf32>, %16: tensor<2048x5632xf32>, %17: tensor<2048xf32>, %18: tensor<2048xf32>, %19: tensor<2048xf32>, %20: tensor<32000x2048xf32>, %21: tensor<32xf32>, %22: tensor<32xf32>, %23: tensor<1x4xi64>) -> tensor<1x4x32000xf32> {
    %24 = tensor.empty() : tensor<1x4x2048xf32>
    %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23 : tensor<1x4xi64>) outs(%24 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.embed_tokens"} {
    ^bb0(%26: i64, %27: f32):
      %28 = arith.index_cast %26 : i64 to index
      %29 = linalg.index 2 : index
      %30 = tensor.extract %0[%28, %29] : tensor<32000x2048xf32>
      linalg.yield %30 : f32
    } -> tensor<1x4x2048xf32>
    %31 = tensor.empty() : tensor<4xi64>
    %32 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%31 : tensor<4xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb1(%33: i64):
      %34 = linalg.index 0 : index
      %35 = arith.index_cast %34 : index to i64
      %36 = arith.constant 1 : i64
      %37 = arith.muli %35, %36 : i64
      %38 = arith.constant 0 : i64
      %39 = arith.addi %38, %37 : i64
      linalg.yield %39 : i64
    } -> tensor<4xi64>
    %40 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} 0 : i64
    %41 = tensor.splat %40 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<4xi64>
    %42 = tensor.empty() : tensor<4xi64>
    %43 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%32, %41 : tensor<4xi64>, tensor<4xi64>) outs(%42 : tensor<4xi64>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb2(%44: i64, %45: i64, %46: i64):
      %47 = arith.addi %44, %45 : i64
      linalg.yield %47 : i64
    } -> tensor<4xi64>
    %48 = tensor.expand_shape %43 [[0 : i64, 1 : i64]] output_shape [1, 4] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<4xi64> into tensor<1x4xi64>
    %49 = "tensor.extract_slice"(%48) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : (tensor<1x4xi64>) -> tensor<1x1xi64>
    %50 = arith.constant {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} 1 : i64
    %51 = tensor.splat %50 {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<1x1xi64>
    %52 = tensor.empty() : tensor<1x1xi64>
    %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%49, %51 : tensor<1x1xi64>, tensor<1x1xi64>) outs(%52 : tensor<1x1xi64>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb3(%54: i64, %55: i64, %56: i64):
      %57 = arith.subi %54, %55 : i64
      linalg.yield %57 : i64
    } -> tensor<1x1xi64>
    %58 = tensor.concat dim(1) %53, %48 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : (tensor<1x1xi64>, tensor<1x4xi64>) -> tensor<1x5xi64>
    %59 = "tensor.extract_slice"(%58) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 4>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : (tensor<1x5xi64>) -> tensor<1x4xi64>
    %60 = "tensor.extract_slice"(%58) <{static_offsets = array<i64: 0, 1>, static_sizes = array<i64: 1, 4>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : (tensor<1x5xi64>) -> tensor<1x4xi64>
    %61 = tensor.empty() : tensor<1x4xi64>
    %62 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%60, %59 : tensor<1x4xi64>, tensor<1x4xi64>) outs(%61 : tensor<1x4xi64>) attrs =  {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb4(%63: i64, %64: i64, %65: i64):
      %66 = arith.subi %63, %64 : i64
      linalg.yield %66 : i64
    } -> tensor<1x4xi64>
    %67 = arith.constant {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ne.Scalar", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model"} 1 : i64
    %68 = tensor.splat %67 {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ne.Scalar", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model"} : tensor<1x4xi64>
    %69 = tensor.empty() : tensor<1x4xi1>
    %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%62, %68 : tensor<1x4xi64>, tensor<1x4xi64>) outs(%69 : tensor<1x4xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.ne.Scalar", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb5(%71: i64, %72: i64, %73: i1):
      %74 = arith.cmpi ne, %71, %72 : i64
      linalg.yield %74 : i1
    } -> tensor<1x4xi1>
    %75 = arith.constant {prov.region_id = "scan_0", prov.family = "scan", prov._pattern_hint = "cumsum", prov.op = "cumsum", prov.aten = "aten.cumsum.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} 0 : i64
    %76 = tensor.splat %75 {prov.region_id = "scan_0", prov.family = "scan", prov._pattern_hint = "cumsum", prov.op = "cumsum", prov.aten = "aten.cumsum.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<1x4xi64>
    %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%70 : tensor<1x4xi1>) outs(%76 : tensor<1x4xi64>) attrs =  {prov.region_id = "scan_0", prov.family = "scan", prov._pattern_hint = "cumsum", prov.op = "cumsum", prov.aten = "aten.cumsum.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb6(%78: i1, %79: i64):
      %80 = linalg.index 1 : index
      %81 = linalg.index 2 : index
      %82 = arith.cmpi ule, %81, %80 : index
      %83 = arith.extui %78 : i1 to i64
      %84 = arith.select %82, %83, %75 : i64
      %85 = arith.addi %79, %84 : i64
      linalg.yield %85 : i64
    } -> tensor<1x4xi64>
    %86 = tensor.empty() : tensor<1xi64>
    %87 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%86 : tensor<1xi64>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb7(%88: i64):
      %89 = linalg.index 0 : index
      %90 = arith.index_cast %89 : index to i64
      %91 = arith.constant 1 : i64
      %92 = arith.muli %90, %91 : i64
      %93 = arith.constant 0 : i64
      %94 = arith.addi %93, %92 : i64
      linalg.yield %94 : i64
    } -> tensor<1xi64>
    %95 = tensor.empty() : tensor<4xi64>
    %96 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%95 : tensor<4xi64>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb8(%97: i64):
      %98 = linalg.index 0 : index
      %99 = arith.index_cast %98 : index to i64
      %100 = arith.constant 1 : i64
      %101 = arith.muli %99, %100 : i64
      %102 = arith.constant 0 : i64
      %103 = arith.addi %102, %101 : i64
      linalg.yield %103 : i64
    } -> tensor<4xi64>
    %104 = arith.constant {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} 0 : i64
    %105 = tensor.splat %104 {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<4xi64>
    %106 = tensor.empty() : tensor<4xi64>
    %107 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%96, %105 : tensor<4xi64>, tensor<4xi64>) outs(%106 : tensor<4xi64>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb9(%108: i64, %109: i64, %110: i64):
      %111 = arith.addi %108, %109 : i64
      linalg.yield %111 : i64
    } -> tensor<4xi64>
    %112 = tensor.empty() : tensor<4xi64>
    %113 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%112 : tensor<4xi64>) attrs =  {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb10(%114: i64):
      %115 = linalg.index 0 : index
      %116 = arith.index_cast %115 : index to i64
      %117 = arith.constant 1 : i64
      %118 = arith.muli %116, %117 : i64
      %119 = arith.constant 0 : i64
      %120 = arith.addi %119, %118 : i64
      linalg.yield %120 : i64
    } -> tensor<4xi64>
    %121 = arith.constant {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} 0 : i64
    %122 = tensor.splat %121 {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<4xi64>
    %123 = tensor.empty() : tensor<4xi64>
    %124 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%113, %122 : tensor<4xi64>, tensor<4xi64>) outs(%123 : tensor<4xi64>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb11(%125: i64, %126: i64, %127: i64):
      %128 = arith.addi %125, %126 : i64
      linalg.yield %128 : i64
    } -> tensor<4xi64>
    %129 = tensor.expand_shape %87 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<1xi64> into tensor<1x1xi64>
    %130 = tensor.collapse_shape %129 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<1x1xi64> into tensor<1xi64>
    %131 = tensor.expand_shape %130 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<1xi64> into tensor<1x1x1xi64>
    %132 = tensor.collapse_shape %131 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<1x1x1xi64> into tensor<1xi64>
    %133 = tensor.expand_shape %132 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 1] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<1xi64> into tensor<1x1x1x1xi64>
    %134 = tensor.expand_shape %107 [[0 : i64, 1 : i64]] output_shape [1, 4] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<4xi64> into tensor<1x4xi64>
    %135 = tensor.collapse_shape %134 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<1x4xi64> into tensor<4xi64>
    %136 = tensor.expand_shape %135 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 4] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<4xi64> into tensor<1x1x4xi64>
    %137 = tensor.collapse_shape %136 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<1x1x4xi64> into tensor<4xi64>
    %138 = tensor.expand_shape %137 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 1] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<4xi64> into tensor<1x1x4x1xi64>
    %139 = tensor.expand_shape %124 [[0 : i64, 1 : i64]] output_shape [1, 4] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<4xi64> into tensor<1x4xi64>
    %140 = tensor.collapse_shape %139 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<1x4xi64> into tensor<4xi64>
    %141 = tensor.expand_shape %140 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 4] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<4xi64> into tensor<1x1x4xi64>
    %142 = tensor.collapse_shape %141 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<1x1x4xi64> into tensor<4xi64>
    %143 = tensor.expand_shape %142 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 4] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<4xi64> into tensor<1x1x1x4xi64>
    %144 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model"} true
    %145 = tensor.splat %144 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model"} : tensor<i1>
    %146 = tensor.empty() : tensor<1x1x4x4xi1>
    %147 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%143, %138 : tensor<1x1x1x4xi64>, tensor<1x1x4x1xi64>) outs(%146 : tensor<1x1x4x4xi1>) attrs =  {prov.region_id = "compare_1", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb12(%148: i64, %149: i64, %150: i1):
      %151 = arith.cmpi sle, %148, %149 : i64
      linalg.yield %151 : i1
    } -> tensor<1x1x4x4xi1>
    %152 = tensor.empty() : tensor<1x1x4x4xi1>
    %153 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%145, %147 : tensor<i1>, tensor<1x1x4x4xi1>) outs(%152 : tensor<1x1x4x4xi1>) attrs =  {prov.region_id = "bitwise_0", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_and.Tensor", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb13(%154: i1, %155: i1, %156: i1):
      %157 = arith.andi %154, %155 : i1
      linalg.yield %157 : i1
    } -> tensor<1x1x4x4xi1>
    %158 = tensor.empty() : tensor<1x1x4x1xi64>
    %159 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%133, %138 : tensor<1x1x1x1xi64>, tensor<1x1x4x1xi64>) outs(%158 : tensor<1x1x4x1xi64>) attrs =  {prov.region_id = "gather_1", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb14(%160: i64, %161: i64, %162: i64):
      %163 = arith.index_cast %160 : i64 to index
      %164 = arith.index_cast %161 : i64 to index
      %165 = tensor.extract %77[%163, %164] : tensor<1x4xi64>
      linalg.yield %165 : i64
    } -> tensor<1x1x4x1xi64>
    %166 = tensor.empty() : tensor<1x1x1x4xi64>
    %167 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%133, %143 : tensor<1x1x1x1xi64>, tensor<1x1x1x4xi64>) outs(%166 : tensor<1x1x1x4xi64>) attrs =  {prov.region_id = "gather_2", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb15(%168: i64, %169: i64, %170: i64):
      %171 = arith.index_cast %168 : i64 to index
      %172 = arith.index_cast %169 : i64 to index
      %173 = tensor.extract %77[%171, %172] : tensor<1x4xi64>
      linalg.yield %173 : i64
    } -> tensor<1x1x1x4xi64>
    %174 = tensor.empty() : tensor<1x1x4x4xi1>
    %175 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%159, %167 : tensor<1x1x4x1xi64>, tensor<1x1x1x4xi64>) outs(%174 : tensor<1x1x4x4xi1>) attrs =  {prov.region_id = "compare_2", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.eq.Tensor", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb16(%176: i64, %177: i64, %178: i1):
      %179 = arith.cmpi eq, %176, %177 : i64
      linalg.yield %179 : i1
    } -> tensor<1x1x4x4xi1>
    %180 = tensor.empty() : tensor<1x1x4x4xi1>
    %181 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%153, %175 : tensor<1x1x4x4xi1>, tensor<1x1x4x4xi1>) outs(%180 : tensor<1x1x4x4xi1>) attrs =  {prov.region_id = "bitwise_1", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_and.Tensor", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb17(%182: i1, %183: i1, %184: i1):
      %185 = arith.andi %182, %183 : i1
      linalg.yield %185 : i1
    } -> tensor<1x1x4x4xi1>
    %186 = tensor.empty() : tensor<1x1x4x4xi1>
    %187 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%181 : tensor<1x1x4x4xi1>) outs(%186 : tensor<1x1x4x4xi1>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb18(%188: i1, %189: i1):
      linalg.yield %188 : i1
    } -> tensor<1x1x4x4xi1>
    %190 = tensor.expand_shape %21 [[0 : i64, 1 : i64]] output_shape [1, 32] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} : tensor<32xf32> into tensor<1x32xf32>
    %191 = tensor.collapse_shape %190 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} : tensor<1x32xf32> into tensor<32xf32>
    %192 = tensor.expand_shape %191 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} : tensor<32xf32> into tensor<1x32x1xf32>
    %193 = tensor.empty() : tensor<1x32x1xf32>
    %194 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%192 : tensor<1x32x1xf32>) outs(%193 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} {
    ^bb19(%195: f32, %196: f32):
      linalg.yield %195 : f32
    } -> tensor<1x32x1xf32>
    %197 = tensor.collapse_shape %48 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} : tensor<1x4xi64> into tensor<4xi64>
    %198 = tensor.expand_shape %197 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 4] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} : tensor<4xi64> into tensor<1x1x4xi64>
    %199 = tensor.empty() : tensor<1x1x4xf32>
    %200 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%198 : tensor<1x1x4xi64>) outs(%199 : tensor<1x1x4xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} {
    ^bb20(%201: i64, %202: f32):
      %203 = arith.sitofp %201 : i64 to f32
      linalg.yield %203 : f32
    } -> tensor<1x1x4xf32>
    %204 = tensor.empty() : tensor<1x32x1xf32>
    %205 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%194 : tensor<1x32x1xf32>) outs(%204 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} {
    ^bb21(%206: f32, %207: f32):
      linalg.yield %206 : f32
    } -> tensor<1x32x1xf32>
    %208 = tensor.empty() : tensor<1x1x4xf32>
    %209 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%200 : tensor<1x1x4xf32>) outs(%208 : tensor<1x1x4xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} {
    ^bb22(%210: f32, %211: f32):
      linalg.yield %210 : f32
    } -> tensor<1x1x4xf32>
    %212 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} 0.000000e+00 : f32
    %213 = tensor.splat %212 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} : tensor<1x32x4xf32>
    %214 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%205, %209 : tensor<1x32x1xf32>, tensor<1x1x4xf32>) outs(%213 : tensor<1x32x4xf32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} {
    ^bb23(%215: f32, %216: f32, %217: f32):
      %218 = arith.mulf %215, %216 : f32
      %219 = arith.addf %217, %218 : f32
      linalg.yield %219 : f32
    } -> tensor<1x32x4xf32>
    %220 = tensor.empty() : tensor<1x4x32xf32>
    %221 = linalg.transpose ins(%214:tensor<1x32x4xf32>) outs(%220:tensor<1x4x32xf32>) permutation = [0, 2, 1]
    %222 = tensor.concat dim(2) %221, %221 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} : (tensor<1x4x32xf32>, tensor<1x4x32xf32>) -> tensor<1x4x64xf32>
    %223 = tensor.empty() : tensor<1x4x64xf32>
    %224 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%222 : tensor<1x4x64xf32>) outs(%223 : tensor<1x4x64xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} {
    ^bb24(%225: f32, %226: f32):
      %227 = math.cos %225 : f32
      linalg.yield %227 : f32
    } -> tensor<1x4x64xf32>
    %228 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} 1.000000e+00 : f32
    %229 = tensor.splat %228 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} : tensor<1x4x64xf32>
    %230 = tensor.empty() : tensor<1x4x64xf32>
    %231 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%224, %229 : tensor<1x4x64xf32>, tensor<1x4x64xf32>) outs(%230 : tensor<1x4x64xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} {
    ^bb25(%232: f32, %233: f32, %234: f32):
      %235 = arith.mulf %232, %233 : f32
      linalg.yield %235 : f32
    } -> tensor<1x4x64xf32>
    %236 = tensor.empty() : tensor<1x4x64xf32>
    %237 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%222 : tensor<1x4x64xf32>) outs(%236 : tensor<1x4x64xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} {
    ^bb26(%238: f32, %239: f32):
      %240 = math.sin %238 : f32
      linalg.yield %240 : f32
    } -> tensor<1x4x64xf32>
    %241 = arith.constant {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} 1.000000e+00 : f32
    %242 = tensor.splat %241 {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} : tensor<1x4x64xf32>
    %243 = tensor.empty() : tensor<1x4x64xf32>
    %244 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%237, %242 : tensor<1x4x64xf32>, tensor<1x4x64xf32>) outs(%243 : tensor<1x4x64xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.rotary_emb"} {
    ^bb27(%245: f32, %246: f32, %247: f32):
      %248 = arith.mulf %245, %246 : f32
      linalg.yield %248 : f32
    } -> tensor<1x4x64xf32>
    %249 = tensor.empty() : tensor<1x4x2048xf32>
    %250 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25 : tensor<1x4x2048xf32>) outs(%249 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.input_layernorm"} {
    ^bb28(%251: f32, %252: f32):
      %253 = arith.constant 2.000000e+00 : f32
      %254 = math.powf %251, %253 : f32
      linalg.yield %254 : f32
    } -> tensor<1x4x2048xf32>
    %255 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.input_layernorm"} 0.000000e+00 : f32
    %256 = tensor.splat %255 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.input_layernorm"} : tensor<1x4xf32>
    %257 = linalg.reduce ins(%250:tensor<1x4x2048xf32>) outs(%256:tensor<1x4xf32>) dimensions = [2]
    (%258: f32, %259: f32) {
      %260 = arith.addf %258, %259 : f32
      linalg.yield %260 : f32
    }
    %261 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.input_layernorm"} 2.048000e+03 : f32
    %262 = tensor.splat %261 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.input_layernorm"} : tensor<1x4xf32>
    %263 = tensor.empty() : tensor<1x4xf32>
    %264 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%257, %262 : tensor<1x4xf32>, tensor<1x4xf32>) outs(%263 : tensor<1x4xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.input_layernorm"} {
    ^bb29(%265: f32, %266: f32, %267: f32):
      %268 = arith.divf %265, %266 : f32
      linalg.yield %268 : f32
    } -> tensor<1x4xf32>
    %269 = tensor.collapse_shape %264 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.input_layernorm"} : tensor<1x4xf32> into tensor<4xf32>
    %270 = tensor.expand_shape %269 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.input_layernorm"} : tensor<4xf32> into tensor<1x4x1xf32>
    %271 = arith.constant {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.input_layernorm"} 1.000000e-05 : f32
    %272 = tensor.splat %271 {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.input_layernorm"} : tensor<1x4x1xf32>
    %273 = tensor.empty() : tensor<1x4x1xf32>
    %274 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%270, %272 : tensor<1x4x1xf32>, tensor<1x4x1xf32>) outs(%273 : tensor<1x4x1xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.input_layernorm"} {
    ^bb30(%275: f32, %276: f32, %277: f32):
      %278 = arith.addf %275, %276 : f32
      linalg.yield %278 : f32
    } -> tensor<1x4x1xf32>
    %279 = tensor.empty() : tensor<1x4x1xf32>
    %280 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%274 : tensor<1x4x1xf32>) outs(%279 : tensor<1x4x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.input_layernorm"} {
    ^bb31(%281: f32, %282: f32):
      %283 = math.rsqrt %281 : f32
      linalg.yield %283 : f32
    } -> tensor<1x4x1xf32>
    %284 = tensor.empty() : tensor<1x4x2048xf32>
    %285 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %280 : tensor<1x4x2048xf32>, tensor<1x4x1xf32>) outs(%284 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.input_layernorm"} {
    ^bb32(%286: f32, %287: f32, %288: f32):
      %289 = arith.mulf %286, %287 : f32
      linalg.yield %289 : f32
    } -> tensor<1x4x2048xf32>
    %290 = tensor.empty() : tensor<1x4x2048xf32>
    %291 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %285 : tensor<2048xf32>, tensor<1x4x2048xf32>) outs(%290 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.input_layernorm"} {
    ^bb33(%292: f32, %293: f32, %294: f32):
      %295 = arith.mulf %292, %293 : f32
      linalg.yield %295 : f32
    } -> tensor<1x4x2048xf32>
    %296 = tensor.empty() : tensor<2048x2048xf32>
    %297 = linalg.transpose ins(%1:tensor<2048x2048xf32>) outs(%296:tensor<2048x2048xf32>) permutation = [1, 0]
    %298 = tensor.collapse_shape %291 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.q_proj"} : tensor<1x4x2048xf32> into tensor<8192xf32>
    %299 = tensor.expand_shape %298 [[0 : i64, 1 : i64]] output_shape [4, 2048] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.q_proj"} : tensor<8192xf32> into tensor<4x2048xf32>
    %300 = tensor.empty() : tensor<4x2048xf32>
    %301 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %302 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%301 : f32) outs(%300 : tensor<4x2048xf32>) -> tensor<4x2048xf32>
    %303 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.q_proj", prov.transposed_b = "true"} ins(%299, %297 : tensor<4x2048xf32>, tensor<2048x2048xf32>) outs(%302 : tensor<4x2048xf32>) -> tensor<4x2048xf32>
    %304 = tensor.collapse_shape %303 [[0 : i64, 1 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.q_proj"} : tensor<4x2048xf32> into tensor<8192xf32>
    %305 = tensor.expand_shape %304 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 2048] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.q_proj"} : tensor<8192xf32> into tensor<1x4x2048xf32>
    %306 = tensor.collapse_shape %305 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x4x2048xf32> into tensor<8192xf32>
    %307 = tensor.expand_shape %306 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 32, 64] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x4x32x64xf32>
    %308 = tensor.empty() : tensor<1x32x4x64xf32>
    %309 = linalg.transpose ins(%307:tensor<1x4x32x64xf32>) outs(%308:tensor<1x32x4x64xf32>) permutation = [0, 2, 1, 3]
    %310 = tensor.empty() : tensor<2048x256xf32>
    %311 = linalg.transpose ins(%2:tensor<256x2048xf32>) outs(%310:tensor<2048x256xf32>) permutation = [1, 0]
    %312 = tensor.collapse_shape %291 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.k_proj"} : tensor<1x4x2048xf32> into tensor<8192xf32>
    %313 = tensor.expand_shape %312 [[0 : i64, 1 : i64]] output_shape [4, 2048] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.k_proj"} : tensor<8192xf32> into tensor<4x2048xf32>
    %314 = tensor.empty() : tensor<4x256xf32>
    %315 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %316 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%315 : f32) outs(%314 : tensor<4x256xf32>) -> tensor<4x256xf32>
    %317 = linalg.matmul {prov.region_id = "matmul_2", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.k_proj", prov.transposed_b = "true"} ins(%313, %311 : tensor<4x2048xf32>, tensor<2048x256xf32>) outs(%316 : tensor<4x256xf32>) -> tensor<4x256xf32>
    %318 = tensor.collapse_shape %317 [[0 : i64, 1 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.k_proj"} : tensor<4x256xf32> into tensor<1024xf32>
    %319 = tensor.expand_shape %318 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 256] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.k_proj"} : tensor<1024xf32> into tensor<1x4x256xf32>
    %320 = tensor.collapse_shape %319 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x4x256xf32> into tensor<1024xf32>
    %321 = tensor.expand_shape %320 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 4, 64] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1024xf32> into tensor<1x4x4x64xf32>
    %322 = tensor.empty() : tensor<1x4x4x64xf32>
    %323 = linalg.transpose ins(%321:tensor<1x4x4x64xf32>) outs(%322:tensor<1x4x4x64xf32>) permutation = [0, 2, 1, 3]
    %324 = tensor.empty() : tensor<2048x256xf32>
    %325 = linalg.transpose ins(%3:tensor<256x2048xf32>) outs(%324:tensor<2048x256xf32>) permutation = [1, 0]
    %326 = tensor.collapse_shape %291 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.v_proj"} : tensor<1x4x2048xf32> into tensor<8192xf32>
    %327 = tensor.expand_shape %326 [[0 : i64, 1 : i64]] output_shape [4, 2048] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.v_proj"} : tensor<8192xf32> into tensor<4x2048xf32>
    %328 = tensor.empty() : tensor<4x256xf32>
    %329 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %330 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%329 : f32) outs(%328 : tensor<4x256xf32>) -> tensor<4x256xf32>
    %331 = linalg.matmul {prov.region_id = "matmul_3", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.v_proj", prov.transposed_b = "true"} ins(%327, %325 : tensor<4x2048xf32>, tensor<2048x256xf32>) outs(%330 : tensor<4x256xf32>) -> tensor<4x256xf32>
    %332 = tensor.collapse_shape %331 [[0 : i64, 1 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.v_proj"} : tensor<4x256xf32> into tensor<1024xf32>
    %333 = tensor.expand_shape %332 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 256] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.v_proj"} : tensor<1024xf32> into tensor<1x4x256xf32>
    %334 = tensor.collapse_shape %333 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x4x256xf32> into tensor<1024xf32>
    %335 = tensor.expand_shape %334 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 4, 64] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1024xf32> into tensor<1x4x4x64xf32>
    %336 = tensor.empty() : tensor<1x4x4x64xf32>
    %337 = linalg.transpose ins(%335:tensor<1x4x4x64xf32>) outs(%336:tensor<1x4x4x64xf32>) permutation = [0, 2, 1, 3]
    %338 = tensor.collapse_shape %231 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x4x64xf32> into tensor<256xf32>
    %339 = tensor.expand_shape %338 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 64] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<256xf32> into tensor<1x1x4x64xf32>
    %340 = tensor.collapse_shape %244 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x4x64xf32> into tensor<256xf32>
    %341 = tensor.expand_shape %340 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 64] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<256xf32> into tensor<1x1x4x64xf32>
    %342 = tensor.empty() : tensor<1x32x4x64xf32>
    %343 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%309, %339 : tensor<1x32x4x64xf32>, tensor<1x1x4x64xf32>) outs(%342 : tensor<1x32x4x64xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb34(%344: f32, %345: f32, %346: f32):
      %347 = arith.mulf %344, %345 : f32
      linalg.yield %347 : f32
    } -> tensor<1x32x4x64xf32>
    %348 = "tensor.extract_slice"(%309) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 32, 4, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : (tensor<1x32x4x64xf32>) -> tensor<1x32x4x32xf32>
    %349 = "tensor.extract_slice"(%309) <{static_offsets = array<i64: 0, 0, 0, 32>, static_sizes = array<i64: 1, 32, 4, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : (tensor<1x32x4x64xf32>) -> tensor<1x32x4x32xf32>
    %350 = tensor.empty() : tensor<1x32x4x32xf32>
    %351 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%349 : tensor<1x32x4x32xf32>) outs(%350 : tensor<1x32x4x32xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb35(%352: f32, %353: f32):
      %354 = arith.negf %352 : f32
      linalg.yield %354 : f32
    } -> tensor<1x32x4x32xf32>
    %355 = tensor.concat dim(3) %351, %348 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : (tensor<1x32x4x32xf32>, tensor<1x32x4x32xf32>) -> tensor<1x32x4x64xf32>
    %356 = tensor.empty() : tensor<1x32x4x64xf32>
    %357 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%355, %341 : tensor<1x32x4x64xf32>, tensor<1x1x4x64xf32>) outs(%356 : tensor<1x32x4x64xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb36(%358: f32, %359: f32, %360: f32):
      %361 = arith.mulf %358, %359 : f32
      linalg.yield %361 : f32
    } -> tensor<1x32x4x64xf32>
    %362 = tensor.empty() : tensor<1x32x4x64xf32>
    %363 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%343, %357 : tensor<1x32x4x64xf32>, tensor<1x32x4x64xf32>) outs(%362 : tensor<1x32x4x64xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb37(%364: f32, %365: f32, %366: f32):
      %367 = arith.addf %364, %365 : f32
      linalg.yield %367 : f32
    } -> tensor<1x32x4x64xf32>
    %368 = tensor.empty() : tensor<1x4x4x64xf32>
    %369 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%323, %339 : tensor<1x4x4x64xf32>, tensor<1x1x4x64xf32>) outs(%368 : tensor<1x4x4x64xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb38(%370: f32, %371: f32, %372: f32):
      %373 = arith.mulf %370, %371 : f32
      linalg.yield %373 : f32
    } -> tensor<1x4x4x64xf32>
    %374 = "tensor.extract_slice"(%323) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 4, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : (tensor<1x4x4x64xf32>) -> tensor<1x4x4x32xf32>
    %375 = "tensor.extract_slice"(%323) <{static_offsets = array<i64: 0, 0, 0, 32>, static_sizes = array<i64: 1, 4, 4, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : (tensor<1x4x4x64xf32>) -> tensor<1x4x4x32xf32>
    %376 = tensor.empty() : tensor<1x4x4x32xf32>
    %377 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%375 : tensor<1x4x4x32xf32>) outs(%376 : tensor<1x4x4x32xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb39(%378: f32, %379: f32):
      %380 = arith.negf %378 : f32
      linalg.yield %380 : f32
    } -> tensor<1x4x4x32xf32>
    %381 = tensor.concat dim(3) %377, %374 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : (tensor<1x4x4x32xf32>, tensor<1x4x4x32xf32>) -> tensor<1x4x4x64xf32>
    %382 = tensor.empty() : tensor<1x4x4x64xf32>
    %383 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%381, %341 : tensor<1x4x4x64xf32>, tensor<1x1x4x64xf32>) outs(%382 : tensor<1x4x4x64xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb40(%384: f32, %385: f32, %386: f32):
      %387 = arith.mulf %384, %385 : f32
      linalg.yield %387 : f32
    } -> tensor<1x4x4x64xf32>
    %388 = tensor.empty() : tensor<1x4x4x64xf32>
    %389 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%369, %383 : tensor<1x4x4x64xf32>, tensor<1x4x4x64xf32>) outs(%388 : tensor<1x4x4x64xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb41(%390: f32, %391: f32, %392: f32):
      %393 = arith.addf %390, %391 : f32
      linalg.yield %393 : f32
    } -> tensor<1x4x4x64xf32>
    %394 = tensor.collapse_shape %389 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x4x4x64xf32> into tensor<1024xf32>
    %395 = tensor.expand_shape %394 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 4, 64] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1024xf32> into tensor<1x4x1x4x64xf32>
    %396 = tensor.empty() : tensor<1x4x8x4x64xf32>
    %397 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%395 : tensor<1x4x1x4x64xf32>) outs(%396 : tensor<1x4x8x4x64xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb42(%398: f32, %399: f32):
      linalg.yield %398 : f32
    } -> tensor<1x4x8x4x64xf32>
    %400 = tensor.collapse_shape %397 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x4x8x4x64xf32> into tensor<8192xf32>
    %401 = tensor.expand_shape %400 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 64] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x32x4x64xf32>
    %402 = tensor.collapse_shape %337 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x4x4x64xf32> into tensor<1024xf32>
    %403 = tensor.expand_shape %402 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 4, 64] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1024xf32> into tensor<1x4x1x4x64xf32>
    %404 = tensor.empty() : tensor<1x4x8x4x64xf32>
    %405 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%403 : tensor<1x4x1x4x64xf32>) outs(%404 : tensor<1x4x8x4x64xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb43(%406: f32, %407: f32):
      linalg.yield %406 : f32
    } -> tensor<1x4x8x4x64xf32>
    %408 = tensor.collapse_shape %405 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x4x8x4x64xf32> into tensor<8192xf32>
    %409 = tensor.expand_shape %408 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 64] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x32x4x64xf32>
    %410 = tensor.empty() : tensor<1x32x64x4xf32>
    %411 = linalg.transpose ins(%401:tensor<1x32x4x64xf32>) outs(%410:tensor<1x32x64x4xf32>) permutation = [0, 1, 3, 2]
    %412 = tensor.empty() : tensor<1x32x4x64xf32>
    %413 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%363 : tensor<1x32x4x64xf32>) outs(%412 : tensor<1x32x4x64xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb44(%414: f32, %415: f32):
      linalg.yield %414 : f32
    } -> tensor<1x32x4x64xf32>
    %416 = tensor.collapse_shape %413 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x32x4x64xf32> into tensor<8192xf32>
    %417 = tensor.expand_shape %416 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 4, 64] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<32x4x64xf32>
    %418 = tensor.empty() : tensor<1x32x64x4xf32>
    %419 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%411 : tensor<1x32x64x4xf32>) outs(%418 : tensor<1x32x64x4xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb45(%420: f32, %421: f32):
      linalg.yield %420 : f32
    } -> tensor<1x32x64x4xf32>
    %422 = tensor.collapse_shape %419 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x32x64x4xf32> into tensor<8192xf32>
    %423 = tensor.expand_shape %422 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 64, 4] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<32x64x4xf32>
    %424 = arith.constant {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} 0.000000e+00 : f32
    %425 = tensor.splat %424 {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<32x4x4xf32>
    %426 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%417, %423 : tensor<32x4x64xf32>, tensor<32x64x4xf32>) outs(%425 : tensor<32x4x4xf32>) attrs =  {prov.region_id = "matmul_4", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb46(%427: f32, %428: f32, %429: f32):
      %430 = arith.mulf %427, %428 : f32
      %431 = arith.addf %429, %430 : f32
      linalg.yield %431 : f32
    } -> tensor<32x4x4xf32>
    %432 = tensor.collapse_shape %426 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<32x4x4xf32> into tensor<512xf32>
    %433 = tensor.expand_shape %432 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 4] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<512xf32> into tensor<1x32x4x4xf32>
    %434 = arith.constant {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} 1.250000e-01 : f32
    %435 = tensor.splat %434 {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x32x4x4xf32>
    %436 = tensor.empty() : tensor<1x32x4x4xf32>
    %437 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%433, %435 : tensor<1x32x4x4xf32>, tensor<1x32x4x4xf32>) outs(%436 : tensor<1x32x4x4xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb47(%438: f32, %439: f32, %440: f32):
      %441 = arith.mulf %438, %439 : f32
      linalg.yield %441 : f32
    } -> tensor<1x32x4x4xf32>
    %442 = tensor.empty() : tensor<1x1x4x4xi1>
    %443 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%187 : tensor<1x1x4x4xi1>) outs(%442 : tensor<1x1x4x4xi1>) attrs =  {prov.region_id = "bitwise_2", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb48(%444: i1, %445: i1):
      %446 = arith.constant true
      %447 = arith.xori %444, %446 : i1
      linalg.yield %447 : i1
    } -> tensor<1x1x4x4xi1>
    %448 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} 0xff800000 : f32
    %449 = tensor.splat %448 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<f32>
    %450 = tensor.empty() : tensor<1x32x4x4xf32>
    %451 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%443, %449, %437 : tensor<1x1x4x4xi1>, tensor<f32>, tensor<1x32x4x4xf32>) outs(%450 : tensor<1x32x4x4xf32>) attrs =  {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb49(%452: i1, %453: f32, %454: f32, %455: f32):
      %456 = arith.select %452, %453, %454 : f32
      linalg.yield %456 : f32
    } -> tensor<1x32x4x4xf32>
    %457 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} 0xff800000 : f32
    %458 = tensor.splat %457 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x32x4xf32>
    %459 = linalg.reduce ins(%451:tensor<1x32x4x4xf32>) outs(%458:tensor<1x32x4xf32>) dimensions = [3]
    (%460: f32, %461: f32) {
      %462 = arith.maximumf %460, %461 : f32
      linalg.yield %462 : f32
    }
    %463 = tensor.collapse_shape %459 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x32x4xf32> into tensor<128xf32>
    %464 = tensor.expand_shape %463 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<128xf32> into tensor<1x32x4x1xf32>
    %465 = tensor.empty() : tensor<1x32x4x4xf32>
    %466 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%451, %464 : tensor<1x32x4x4xf32>, tensor<1x32x4x1xf32>) outs(%465 : tensor<1x32x4x4xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb50(%467: f32, %468: f32, %469: f32):
      %470 = arith.subf %467, %468 : f32
      linalg.yield %470 : f32
    } -> tensor<1x32x4x4xf32>
    %471 = tensor.empty() : tensor<1x32x4x4xf32>
    %472 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%466 : tensor<1x32x4x4xf32>) outs(%471 : tensor<1x32x4x4xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb51(%473: f32, %474: f32):
      %475 = math.exp %473 : f32
      linalg.yield %475 : f32
    } -> tensor<1x32x4x4xf32>
    %476 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} 0.000000e+00 : f32
    %477 = tensor.splat %476 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x32x4xf32>
    %478 = linalg.reduce ins(%472:tensor<1x32x4x4xf32>) outs(%477:tensor<1x32x4xf32>) dimensions = [3]
    (%479: f32, %480: f32) {
      %481 = arith.addf %479, %480 : f32
      linalg.yield %481 : f32
    }
    %482 = tensor.collapse_shape %478 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x32x4xf32> into tensor<128xf32>
    %483 = tensor.expand_shape %482 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<128xf32> into tensor<1x32x4x1xf32>
    %484 = tensor.empty() : tensor<1x32x4x4xf32>
    %485 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%472, %483 : tensor<1x32x4x4xf32>, tensor<1x32x4x1xf32>) outs(%484 : tensor<1x32x4x4xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb52(%486: f32, %487: f32, %488: f32):
      %489 = arith.divf %486, %487 : f32
      linalg.yield %489 : f32
    } -> tensor<1x32x4x4xf32>
    %490 = tensor.empty() : tensor<1x32x4x4xf32>
    %491 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%485 : tensor<1x32x4x4xf32>) outs(%490 : tensor<1x32x4x4xf32>) attrs =  {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb53(%492: f32, %493: f32):
      linalg.yield %492 : f32
    } -> tensor<1x32x4x4xf32>
    %494 = tensor.collapse_shape %491 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x32x4x4xf32> into tensor<512xf32>
    %495 = tensor.expand_shape %494 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 4, 4] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<512xf32> into tensor<32x4x4xf32>
    %496 = tensor.empty() : tensor<1x32x4x64xf32>
    %497 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%409 : tensor<1x32x4x64xf32>) outs(%496 : tensor<1x32x4x64xf32>) attrs =  {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb54(%498: f32, %499: f32):
      linalg.yield %498 : f32
    } -> tensor<1x32x4x64xf32>
    %500 = tensor.collapse_shape %497 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x32x4x64xf32> into tensor<8192xf32>
    %501 = tensor.expand_shape %500 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 4, 64] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<32x4x64xf32>
    %502 = arith.constant {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} 0.000000e+00 : f32
    %503 = tensor.splat %502 {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<32x4x64xf32>
    %504 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%495, %501 : tensor<32x4x4xf32>, tensor<32x4x64xf32>) outs(%503 : tensor<32x4x64xf32>) attrs =  {prov.region_id = "matmul_5", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} {
    ^bb55(%505: f32, %506: f32, %507: f32):
      %508 = arith.mulf %505, %506 : f32
      %509 = arith.addf %507, %508 : f32
      linalg.yield %509 : f32
    } -> tensor<32x4x64xf32>
    %510 = tensor.collapse_shape %504 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<32x4x64xf32> into tensor<8192xf32>
    %511 = tensor.expand_shape %510 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 64] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x32x4x64xf32>
    %512 = tensor.empty() : tensor<1x4x32x64xf32>
    %513 = linalg.transpose ins(%511:tensor<1x32x4x64xf32>) outs(%512:tensor<1x4x32x64xf32>) permutation = [0, 2, 1, 3]
    %514 = tensor.collapse_shape %513 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<1x4x32x64xf32> into tensor<8192xf32>
    %515 = tensor.expand_shape %514 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 2048] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn"} : tensor<8192xf32> into tensor<1x4x2048xf32>
    %516 = tensor.empty() : tensor<2048x2048xf32>
    %517 = linalg.transpose ins(%4:tensor<2048x2048xf32>) outs(%516:tensor<2048x2048xf32>) permutation = [1, 0]
    %518 = tensor.collapse_shape %515 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.o_proj"} : tensor<1x4x2048xf32> into tensor<8192xf32>
    %519 = tensor.expand_shape %518 [[0 : i64, 1 : i64]] output_shape [4, 2048] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.o_proj"} : tensor<8192xf32> into tensor<4x2048xf32>
    %520 = tensor.empty() : tensor<4x2048xf32>
    %521 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %522 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%521 : f32) outs(%520 : tensor<4x2048xf32>) -> tensor<4x2048xf32>
    %523 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.o_proj", prov.transposed_b = "true"} ins(%519, %517 : tensor<4x2048xf32>, tensor<2048x2048xf32>) outs(%522 : tensor<4x2048xf32>) -> tensor<4x2048xf32>
    %524 = tensor.collapse_shape %523 [[0 : i64, 1 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.o_proj"} : tensor<4x2048xf32> into tensor<8192xf32>
    %525 = tensor.expand_shape %524 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 2048] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.self_attn.o_proj"} : tensor<8192xf32> into tensor<1x4x2048xf32>
    %526 = tensor.empty() : tensor<1x4x2048xf32>
    %527 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %525 : tensor<1x4x2048xf32>, tensor<1x4x2048xf32>) outs(%526 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0"} {
    ^bb56(%528: f32, %529: f32, %530: f32):
      %531 = arith.addf %528, %529 : f32
      linalg.yield %531 : f32
    } -> tensor<1x4x2048xf32>
    %532 = tensor.empty() : tensor<1x4x2048xf32>
    %533 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%527 : tensor<1x4x2048xf32>) outs(%532 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.post_attention_layernorm"} {
    ^bb57(%534: f32, %535: f32):
      %536 = arith.constant 2.000000e+00 : f32
      %537 = math.powf %534, %536 : f32
      linalg.yield %537 : f32
    } -> tensor<1x4x2048xf32>
    %538 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.post_attention_layernorm"} 0.000000e+00 : f32
    %539 = tensor.splat %538 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.post_attention_layernorm"} : tensor<1x4xf32>
    %540 = linalg.reduce ins(%533:tensor<1x4x2048xf32>) outs(%539:tensor<1x4xf32>) dimensions = [2]
    (%541: f32, %542: f32) {
      %543 = arith.addf %541, %542 : f32
      linalg.yield %543 : f32
    }
    %544 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.post_attention_layernorm"} 2.048000e+03 : f32
    %545 = tensor.splat %544 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.post_attention_layernorm"} : tensor<1x4xf32>
    %546 = tensor.empty() : tensor<1x4xf32>
    %547 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%540, %545 : tensor<1x4xf32>, tensor<1x4xf32>) outs(%546 : tensor<1x4xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.post_attention_layernorm"} {
    ^bb58(%548: f32, %549: f32, %550: f32):
      %551 = arith.divf %548, %549 : f32
      linalg.yield %551 : f32
    } -> tensor<1x4xf32>
    %552 = tensor.collapse_shape %547 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.post_attention_layernorm"} : tensor<1x4xf32> into tensor<4xf32>
    %553 = tensor.expand_shape %552 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.post_attention_layernorm"} : tensor<4xf32> into tensor<1x4x1xf32>
    %554 = arith.constant {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.post_attention_layernorm"} 1.000000e-05 : f32
    %555 = tensor.splat %554 {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.post_attention_layernorm"} : tensor<1x4x1xf32>
    %556 = tensor.empty() : tensor<1x4x1xf32>
    %557 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%553, %555 : tensor<1x4x1xf32>, tensor<1x4x1xf32>) outs(%556 : tensor<1x4x1xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.post_attention_layernorm"} {
    ^bb59(%558: f32, %559: f32, %560: f32):
      %561 = arith.addf %558, %559 : f32
      linalg.yield %561 : f32
    } -> tensor<1x4x1xf32>
    %562 = tensor.empty() : tensor<1x4x1xf32>
    %563 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%557 : tensor<1x4x1xf32>) outs(%562 : tensor<1x4x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.post_attention_layernorm"} {
    ^bb60(%564: f32, %565: f32):
      %566 = math.rsqrt %564 : f32
      linalg.yield %566 : f32
    } -> tensor<1x4x1xf32>
    %567 = tensor.empty() : tensor<1x4x2048xf32>
    %568 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%527, %563 : tensor<1x4x2048xf32>, tensor<1x4x1xf32>) outs(%567 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.post_attention_layernorm"} {
    ^bb61(%569: f32, %570: f32, %571: f32):
      %572 = arith.mulf %569, %570 : f32
      linalg.yield %572 : f32
    } -> tensor<1x4x2048xf32>
    %573 = tensor.empty() : tensor<1x4x2048xf32>
    %574 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %568 : tensor<2048xf32>, tensor<1x4x2048xf32>) outs(%573 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.post_attention_layernorm"} {
    ^bb62(%575: f32, %576: f32, %577: f32):
      %578 = arith.mulf %575, %576 : f32
      linalg.yield %578 : f32
    } -> tensor<1x4x2048xf32>
    %579 = tensor.empty() : tensor<2048x5632xf32>
    %580 = linalg.transpose ins(%5:tensor<5632x2048xf32>) outs(%579:tensor<2048x5632xf32>) permutation = [1, 0]
    %581 = tensor.collapse_shape %574 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp.gate_proj"} : tensor<1x4x2048xf32> into tensor<8192xf32>
    %582 = tensor.expand_shape %581 [[0 : i64, 1 : i64]] output_shape [4, 2048] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp.gate_proj"} : tensor<8192xf32> into tensor<4x2048xf32>
    %583 = tensor.empty() : tensor<4x5632xf32>
    %584 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %585 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%584 : f32) outs(%583 : tensor<4x5632xf32>) -> tensor<4x5632xf32>
    %586 = linalg.matmul {prov.region_id = "matmul_7", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp.gate_proj", prov.transposed_b = "true"} ins(%582, %580 : tensor<4x2048xf32>, tensor<2048x5632xf32>) outs(%585 : tensor<4x5632xf32>) -> tensor<4x5632xf32>
    %587 = tensor.collapse_shape %586 [[0 : i64, 1 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp.gate_proj"} : tensor<4x5632xf32> into tensor<22528xf32>
    %588 = tensor.expand_shape %587 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 5632] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp.gate_proj"} : tensor<22528xf32> into tensor<1x4x5632xf32>
    %589 = tensor.empty() : tensor<1x4x5632xf32>
    %590 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%588 : tensor<1x4x5632xf32>) outs(%589 : tensor<1x4x5632xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp.act_fn"} {
    ^bb63(%591: f32, %592: f32):
      %593 = arith.constant 1.000000e+00 : f32
      %594 = arith.negf %591 : f32
      %595 = math.exp %594 : f32
      %596 = arith.addf %593, %595 : f32
      %597 = arith.divf %593, %596 : f32
      linalg.yield %597 : f32
    } -> tensor<1x4x5632xf32>
    %598 = tensor.empty() : tensor<1x4x5632xf32>
    %599 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%588, %590 : tensor<1x4x5632xf32>, tensor<1x4x5632xf32>) outs(%598 : tensor<1x4x5632xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp.act_fn"} {
    ^bb64(%600: f32, %601: f32, %602: f32):
      %603 = arith.mulf %600, %601 : f32
      linalg.yield %603 : f32
    } -> tensor<1x4x5632xf32>
    %604 = tensor.empty() : tensor<2048x5632xf32>
    %605 = linalg.transpose ins(%6:tensor<5632x2048xf32>) outs(%604:tensor<2048x5632xf32>) permutation = [1, 0]
    %606 = tensor.collapse_shape %574 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp.up_proj"} : tensor<1x4x2048xf32> into tensor<8192xf32>
    %607 = tensor.expand_shape %606 [[0 : i64, 1 : i64]] output_shape [4, 2048] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp.up_proj"} : tensor<8192xf32> into tensor<4x2048xf32>
    %608 = tensor.empty() : tensor<4x5632xf32>
    %609 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %610 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%609 : f32) outs(%608 : tensor<4x5632xf32>) -> tensor<4x5632xf32>
    %611 = linalg.matmul {prov.region_id = "matmul_8", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp.up_proj", prov.transposed_b = "true"} ins(%607, %605 : tensor<4x2048xf32>, tensor<2048x5632xf32>) outs(%610 : tensor<4x5632xf32>) -> tensor<4x5632xf32>
    %612 = tensor.collapse_shape %611 [[0 : i64, 1 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp.up_proj"} : tensor<4x5632xf32> into tensor<22528xf32>
    %613 = tensor.expand_shape %612 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 5632] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp.up_proj"} : tensor<22528xf32> into tensor<1x4x5632xf32>
    %614 = tensor.empty() : tensor<1x4x5632xf32>
    %615 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%599, %613 : tensor<1x4x5632xf32>, tensor<1x4x5632xf32>) outs(%614 : tensor<1x4x5632xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp"} {
    ^bb65(%616: f32, %617: f32, %618: f32):
      %619 = arith.mulf %616, %617 : f32
      linalg.yield %619 : f32
    } -> tensor<1x4x5632xf32>
    %620 = tensor.empty() : tensor<5632x2048xf32>
    %621 = linalg.transpose ins(%7:tensor<2048x5632xf32>) outs(%620:tensor<5632x2048xf32>) permutation = [1, 0]
    %622 = tensor.collapse_shape %615 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp.down_proj"} : tensor<1x4x5632xf32> into tensor<22528xf32>
    %623 = tensor.expand_shape %622 [[0 : i64, 1 : i64]] output_shape [4, 5632] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp.down_proj"} : tensor<22528xf32> into tensor<4x5632xf32>
    %624 = tensor.empty() : tensor<4x2048xf32>
    %625 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %626 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%625 : f32) outs(%624 : tensor<4x2048xf32>) -> tensor<4x2048xf32>
    %627 = linalg.matmul {prov.region_id = "matmul_9", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp.down_proj", prov.transposed_b = "true"} ins(%623, %621 : tensor<4x5632xf32>, tensor<5632x2048xf32>) outs(%626 : tensor<4x2048xf32>) -> tensor<4x2048xf32>
    %628 = tensor.collapse_shape %627 [[0 : i64, 1 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp.down_proj"} : tensor<4x2048xf32> into tensor<8192xf32>
    %629 = tensor.expand_shape %628 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 2048] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0.mlp.down_proj"} : tensor<8192xf32> into tensor<1x4x2048xf32>
    %630 = tensor.empty() : tensor<1x4x2048xf32>
    %631 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%527, %629 : tensor<1x4x2048xf32>, tensor<1x4x2048xf32>) outs(%630 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.0"} {
    ^bb66(%632: f32, %633: f32, %634: f32):
      %635 = arith.addf %632, %633 : f32
      linalg.yield %635 : f32
    } -> tensor<1x4x2048xf32>
    %636 = tensor.empty() : tensor<1x4x2048xf32>
    %637 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%631 : tensor<1x4x2048xf32>) outs(%636 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.input_layernorm"} {
    ^bb67(%638: f32, %639: f32):
      %640 = arith.constant 2.000000e+00 : f32
      %641 = math.powf %638, %640 : f32
      linalg.yield %641 : f32
    } -> tensor<1x4x2048xf32>
    %642 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.input_layernorm"} 0.000000e+00 : f32
    %643 = tensor.splat %642 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.input_layernorm"} : tensor<1x4xf32>
    %644 = linalg.reduce ins(%637:tensor<1x4x2048xf32>) outs(%643:tensor<1x4xf32>) dimensions = [2]
    (%645: f32, %646: f32) {
      %647 = arith.addf %645, %646 : f32
      linalg.yield %647 : f32
    }
    %648 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.input_layernorm"} 2.048000e+03 : f32
    %649 = tensor.splat %648 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.input_layernorm"} : tensor<1x4xf32>
    %650 = tensor.empty() : tensor<1x4xf32>
    %651 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%644, %649 : tensor<1x4xf32>, tensor<1x4xf32>) outs(%650 : tensor<1x4xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.input_layernorm"} {
    ^bb68(%652: f32, %653: f32, %654: f32):
      %655 = arith.divf %652, %653 : f32
      linalg.yield %655 : f32
    } -> tensor<1x4xf32>
    %656 = tensor.collapse_shape %651 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.input_layernorm"} : tensor<1x4xf32> into tensor<4xf32>
    %657 = tensor.expand_shape %656 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.input_layernorm"} : tensor<4xf32> into tensor<1x4x1xf32>
    %658 = arith.constant {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.input_layernorm"} 1.000000e-05 : f32
    %659 = tensor.splat %658 {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.input_layernorm"} : tensor<1x4x1xf32>
    %660 = tensor.empty() : tensor<1x4x1xf32>
    %661 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%657, %659 : tensor<1x4x1xf32>, tensor<1x4x1xf32>) outs(%660 : tensor<1x4x1xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.input_layernorm"} {
    ^bb69(%662: f32, %663: f32, %664: f32):
      %665 = arith.addf %662, %663 : f32
      linalg.yield %665 : f32
    } -> tensor<1x4x1xf32>
    %666 = tensor.empty() : tensor<1x4x1xf32>
    %667 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%661 : tensor<1x4x1xf32>) outs(%666 : tensor<1x4x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.input_layernorm"} {
    ^bb70(%668: f32, %669: f32):
      %670 = math.rsqrt %668 : f32
      linalg.yield %670 : f32
    } -> tensor<1x4x1xf32>
    %671 = tensor.empty() : tensor<1x4x2048xf32>
    %672 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%631, %667 : tensor<1x4x2048xf32>, tensor<1x4x1xf32>) outs(%671 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.input_layernorm"} {
    ^bb71(%673: f32, %674: f32, %675: f32):
      %676 = arith.mulf %673, %674 : f32
      linalg.yield %676 : f32
    } -> tensor<1x4x2048xf32>
    %677 = tensor.empty() : tensor<1x4x2048xf32>
    %678 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %672 : tensor<2048xf32>, tensor<1x4x2048xf32>) outs(%677 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.input_layernorm"} {
    ^bb72(%679: f32, %680: f32, %681: f32):
      %682 = arith.mulf %679, %680 : f32
      linalg.yield %682 : f32
    } -> tensor<1x4x2048xf32>
    %683 = tensor.empty() : tensor<2048x2048xf32>
    %684 = linalg.transpose ins(%10:tensor<2048x2048xf32>) outs(%683:tensor<2048x2048xf32>) permutation = [1, 0]
    %685 = tensor.collapse_shape %678 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.q_proj"} : tensor<1x4x2048xf32> into tensor<8192xf32>
    %686 = tensor.expand_shape %685 [[0 : i64, 1 : i64]] output_shape [4, 2048] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.q_proj"} : tensor<8192xf32> into tensor<4x2048xf32>
    %687 = tensor.empty() : tensor<4x2048xf32>
    %688 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %689 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%688 : f32) outs(%687 : tensor<4x2048xf32>) -> tensor<4x2048xf32>
    %690 = linalg.matmul {prov.region_id = "matmul_10", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.q_proj", prov.transposed_b = "true"} ins(%686, %684 : tensor<4x2048xf32>, tensor<2048x2048xf32>) outs(%689 : tensor<4x2048xf32>) -> tensor<4x2048xf32>
    %691 = tensor.collapse_shape %690 [[0 : i64, 1 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.q_proj"} : tensor<4x2048xf32> into tensor<8192xf32>
    %692 = tensor.expand_shape %691 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 2048] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.q_proj"} : tensor<8192xf32> into tensor<1x4x2048xf32>
    %693 = tensor.collapse_shape %692 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x4x2048xf32> into tensor<8192xf32>
    %694 = tensor.expand_shape %693 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 32, 64] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x4x32x64xf32>
    %695 = tensor.empty() : tensor<1x32x4x64xf32>
    %696 = linalg.transpose ins(%694:tensor<1x4x32x64xf32>) outs(%695:tensor<1x32x4x64xf32>) permutation = [0, 2, 1, 3]
    %697 = tensor.empty() : tensor<2048x256xf32>
    %698 = linalg.transpose ins(%11:tensor<256x2048xf32>) outs(%697:tensor<2048x256xf32>) permutation = [1, 0]
    %699 = tensor.collapse_shape %678 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.k_proj"} : tensor<1x4x2048xf32> into tensor<8192xf32>
    %700 = tensor.expand_shape %699 [[0 : i64, 1 : i64]] output_shape [4, 2048] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.k_proj"} : tensor<8192xf32> into tensor<4x2048xf32>
    %701 = tensor.empty() : tensor<4x256xf32>
    %702 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %703 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%702 : f32) outs(%701 : tensor<4x256xf32>) -> tensor<4x256xf32>
    %704 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.k_proj", prov.transposed_b = "true"} ins(%700, %698 : tensor<4x2048xf32>, tensor<2048x256xf32>) outs(%703 : tensor<4x256xf32>) -> tensor<4x256xf32>
    %705 = tensor.collapse_shape %704 [[0 : i64, 1 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.k_proj"} : tensor<4x256xf32> into tensor<1024xf32>
    %706 = tensor.expand_shape %705 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 256] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.k_proj"} : tensor<1024xf32> into tensor<1x4x256xf32>
    %707 = tensor.collapse_shape %706 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x4x256xf32> into tensor<1024xf32>
    %708 = tensor.expand_shape %707 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 4, 64] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1024xf32> into tensor<1x4x4x64xf32>
    %709 = tensor.empty() : tensor<1x4x4x64xf32>
    %710 = linalg.transpose ins(%708:tensor<1x4x4x64xf32>) outs(%709:tensor<1x4x4x64xf32>) permutation = [0, 2, 1, 3]
    %711 = tensor.empty() : tensor<2048x256xf32>
    %712 = linalg.transpose ins(%12:tensor<256x2048xf32>) outs(%711:tensor<2048x256xf32>) permutation = [1, 0]
    %713 = tensor.collapse_shape %678 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.v_proj"} : tensor<1x4x2048xf32> into tensor<8192xf32>
    %714 = tensor.expand_shape %713 [[0 : i64, 1 : i64]] output_shape [4, 2048] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.v_proj"} : tensor<8192xf32> into tensor<4x2048xf32>
    %715 = tensor.empty() : tensor<4x256xf32>
    %716 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %717 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%716 : f32) outs(%715 : tensor<4x256xf32>) -> tensor<4x256xf32>
    %718 = linalg.matmul {prov.region_id = "matmul_12", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.v_proj", prov.transposed_b = "true"} ins(%714, %712 : tensor<4x2048xf32>, tensor<2048x256xf32>) outs(%717 : tensor<4x256xf32>) -> tensor<4x256xf32>
    %719 = tensor.collapse_shape %718 [[0 : i64, 1 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.v_proj"} : tensor<4x256xf32> into tensor<1024xf32>
    %720 = tensor.expand_shape %719 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 256] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.v_proj"} : tensor<1024xf32> into tensor<1x4x256xf32>
    %721 = tensor.collapse_shape %720 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x4x256xf32> into tensor<1024xf32>
    %722 = tensor.expand_shape %721 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 4, 64] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1024xf32> into tensor<1x4x4x64xf32>
    %723 = tensor.empty() : tensor<1x4x4x64xf32>
    %724 = linalg.transpose ins(%722:tensor<1x4x4x64xf32>) outs(%723:tensor<1x4x4x64xf32>) permutation = [0, 2, 1, 3]
    %725 = tensor.collapse_shape %231 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x4x64xf32> into tensor<256xf32>
    %726 = tensor.expand_shape %725 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 64] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<256xf32> into tensor<1x1x4x64xf32>
    %727 = tensor.collapse_shape %244 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x4x64xf32> into tensor<256xf32>
    %728 = tensor.expand_shape %727 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 64] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<256xf32> into tensor<1x1x4x64xf32>
    %729 = tensor.empty() : tensor<1x32x4x64xf32>
    %730 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%696, %726 : tensor<1x32x4x64xf32>, tensor<1x1x4x64xf32>) outs(%729 : tensor<1x32x4x64xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb73(%731: f32, %732: f32, %733: f32):
      %734 = arith.mulf %731, %732 : f32
      linalg.yield %734 : f32
    } -> tensor<1x32x4x64xf32>
    %735 = "tensor.extract_slice"(%696) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 32, 4, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : (tensor<1x32x4x64xf32>) -> tensor<1x32x4x32xf32>
    %736 = "tensor.extract_slice"(%696) <{static_offsets = array<i64: 0, 0, 0, 32>, static_sizes = array<i64: 1, 32, 4, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : (tensor<1x32x4x64xf32>) -> tensor<1x32x4x32xf32>
    %737 = tensor.empty() : tensor<1x32x4x32xf32>
    %738 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%736 : tensor<1x32x4x32xf32>) outs(%737 : tensor<1x32x4x32xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb74(%739: f32, %740: f32):
      %741 = arith.negf %739 : f32
      linalg.yield %741 : f32
    } -> tensor<1x32x4x32xf32>
    %742 = tensor.concat dim(3) %738, %735 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : (tensor<1x32x4x32xf32>, tensor<1x32x4x32xf32>) -> tensor<1x32x4x64xf32>
    %743 = tensor.empty() : tensor<1x32x4x64xf32>
    %744 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%742, %728 : tensor<1x32x4x64xf32>, tensor<1x1x4x64xf32>) outs(%743 : tensor<1x32x4x64xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb75(%745: f32, %746: f32, %747: f32):
      %748 = arith.mulf %745, %746 : f32
      linalg.yield %748 : f32
    } -> tensor<1x32x4x64xf32>
    %749 = tensor.empty() : tensor<1x32x4x64xf32>
    %750 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%730, %744 : tensor<1x32x4x64xf32>, tensor<1x32x4x64xf32>) outs(%749 : tensor<1x32x4x64xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb76(%751: f32, %752: f32, %753: f32):
      %754 = arith.addf %751, %752 : f32
      linalg.yield %754 : f32
    } -> tensor<1x32x4x64xf32>
    %755 = tensor.empty() : tensor<1x4x4x64xf32>
    %756 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%710, %726 : tensor<1x4x4x64xf32>, tensor<1x1x4x64xf32>) outs(%755 : tensor<1x4x4x64xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb77(%757: f32, %758: f32, %759: f32):
      %760 = arith.mulf %757, %758 : f32
      linalg.yield %760 : f32
    } -> tensor<1x4x4x64xf32>
    %761 = "tensor.extract_slice"(%710) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 4, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_9", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : (tensor<1x4x4x64xf32>) -> tensor<1x4x4x32xf32>
    %762 = "tensor.extract_slice"(%710) <{static_offsets = array<i64: 0, 0, 0, 32>, static_sizes = array<i64: 1, 4, 4, 32>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_10", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : (tensor<1x4x4x64xf32>) -> tensor<1x4x4x32xf32>
    %763 = tensor.empty() : tensor<1x4x4x32xf32>
    %764 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%762 : tensor<1x4x4x32xf32>) outs(%763 : tensor<1x4x4x32xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb78(%765: f32, %766: f32):
      %767 = arith.negf %765 : f32
      linalg.yield %767 : f32
    } -> tensor<1x4x4x32xf32>
    %768 = tensor.concat dim(3) %764, %761 {prov.region_id = "cat_5", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : (tensor<1x4x4x32xf32>, tensor<1x4x4x32xf32>) -> tensor<1x4x4x64xf32>
    %769 = tensor.empty() : tensor<1x4x4x64xf32>
    %770 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%768, %728 : tensor<1x4x4x64xf32>, tensor<1x1x4x64xf32>) outs(%769 : tensor<1x4x4x64xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb79(%771: f32, %772: f32, %773: f32):
      %774 = arith.mulf %771, %772 : f32
      linalg.yield %774 : f32
    } -> tensor<1x4x4x64xf32>
    %775 = tensor.empty() : tensor<1x4x4x64xf32>
    %776 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%756, %770 : tensor<1x4x4x64xf32>, tensor<1x4x4x64xf32>) outs(%775 : tensor<1x4x4x64xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb80(%777: f32, %778: f32, %779: f32):
      %780 = arith.addf %777, %778 : f32
      linalg.yield %780 : f32
    } -> tensor<1x4x4x64xf32>
    %781 = tensor.collapse_shape %776 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x4x4x64xf32> into tensor<1024xf32>
    %782 = tensor.expand_shape %781 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 4, 64] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1024xf32> into tensor<1x4x1x4x64xf32>
    %783 = tensor.empty() : tensor<1x4x8x4x64xf32>
    %784 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%782 : tensor<1x4x1x4x64xf32>) outs(%783 : tensor<1x4x8x4x64xf32>) attrs =  {prov.region_id = "expand_10", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb81(%785: f32, %786: f32):
      linalg.yield %785 : f32
    } -> tensor<1x4x8x4x64xf32>
    %787 = tensor.collapse_shape %784 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x4x8x4x64xf32> into tensor<8192xf32>
    %788 = tensor.expand_shape %787 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 64] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x32x4x64xf32>
    %789 = tensor.collapse_shape %724 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x4x4x64xf32> into tensor<1024xf32>
    %790 = tensor.expand_shape %789 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 4, 64] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1024xf32> into tensor<1x4x1x4x64xf32>
    %791 = tensor.empty() : tensor<1x4x8x4x64xf32>
    %792 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%790 : tensor<1x4x1x4x64xf32>) outs(%791 : tensor<1x4x8x4x64xf32>) attrs =  {prov.region_id = "expand_11", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb82(%793: f32, %794: f32):
      linalg.yield %793 : f32
    } -> tensor<1x4x8x4x64xf32>
    %795 = tensor.collapse_shape %792 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x4x8x4x64xf32> into tensor<8192xf32>
    %796 = tensor.expand_shape %795 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 64] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x32x4x64xf32>
    %797 = tensor.empty() : tensor<1x32x64x4xf32>
    %798 = linalg.transpose ins(%788:tensor<1x32x4x64xf32>) outs(%797:tensor<1x32x64x4xf32>) permutation = [0, 1, 3, 2]
    %799 = tensor.empty() : tensor<1x32x4x64xf32>
    %800 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%750 : tensor<1x32x4x64xf32>) outs(%799 : tensor<1x32x4x64xf32>) attrs =  {prov.region_id = "expand_12", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb83(%801: f32, %802: f32):
      linalg.yield %801 : f32
    } -> tensor<1x32x4x64xf32>
    %803 = tensor.collapse_shape %800 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x32x4x64xf32> into tensor<8192xf32>
    %804 = tensor.expand_shape %803 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 4, 64] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<32x4x64xf32>
    %805 = tensor.empty() : tensor<1x32x64x4xf32>
    %806 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%798 : tensor<1x32x64x4xf32>) outs(%805 : tensor<1x32x64x4xf32>) attrs =  {prov.region_id = "expand_13", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb84(%807: f32, %808: f32):
      linalg.yield %807 : f32
    } -> tensor<1x32x64x4xf32>
    %809 = tensor.collapse_shape %806 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x32x64x4xf32> into tensor<8192xf32>
    %810 = tensor.expand_shape %809 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 64, 4] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<32x64x4xf32>
    %811 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} 0.000000e+00 : f32
    %812 = tensor.splat %811 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<32x4x4xf32>
    %813 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%804, %810 : tensor<32x4x64xf32>, tensor<32x64x4xf32>) outs(%812 : tensor<32x4x4xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb85(%814: f32, %815: f32, %816: f32):
      %817 = arith.mulf %814, %815 : f32
      %818 = arith.addf %816, %817 : f32
      linalg.yield %818 : f32
    } -> tensor<32x4x4xf32>
    %819 = tensor.collapse_shape %813 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<32x4x4xf32> into tensor<512xf32>
    %820 = tensor.expand_shape %819 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 4] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<512xf32> into tensor<1x32x4x4xf32>
    %821 = arith.constant {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} 1.250000e-01 : f32
    %822 = tensor.splat %821 {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x32x4x4xf32>
    %823 = tensor.empty() : tensor<1x32x4x4xf32>
    %824 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%820, %822 : tensor<1x32x4x4xf32>, tensor<1x32x4x4xf32>) outs(%823 : tensor<1x32x4x4xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb86(%825: f32, %826: f32, %827: f32):
      %828 = arith.mulf %825, %826 : f32
      linalg.yield %828 : f32
    } -> tensor<1x32x4x4xf32>
    %829 = tensor.empty() : tensor<1x1x4x4xi1>
    %830 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%187 : tensor<1x1x4x4xi1>) outs(%829 : tensor<1x1x4x4xi1>) attrs =  {prov.region_id = "bitwise_3", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb87(%831: i1, %832: i1):
      %833 = arith.constant true
      %834 = arith.xori %831, %833 : i1
      linalg.yield %834 : i1
    } -> tensor<1x1x4x4xi1>
    %835 = arith.constant {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} 0xff800000 : f32
    %836 = tensor.splat %835 {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<f32>
    %837 = tensor.empty() : tensor<1x32x4x4xf32>
    %838 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%830, %836, %824 : tensor<1x1x4x4xi1>, tensor<f32>, tensor<1x32x4x4xf32>) outs(%837 : tensor<1x32x4x4xf32>) attrs =  {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb88(%839: i1, %840: f32, %841: f32, %842: f32):
      %843 = arith.select %839, %840, %841 : f32
      linalg.yield %843 : f32
    } -> tensor<1x32x4x4xf32>
    %844 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} 0xff800000 : f32
    %845 = tensor.splat %844 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x32x4xf32>
    %846 = linalg.reduce ins(%838:tensor<1x32x4x4xf32>) outs(%845:tensor<1x32x4xf32>) dimensions = [3]
    (%847: f32, %848: f32) {
      %849 = arith.maximumf %847, %848 : f32
      linalg.yield %849 : f32
    }
    %850 = tensor.collapse_shape %846 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x32x4xf32> into tensor<128xf32>
    %851 = tensor.expand_shape %850 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<128xf32> into tensor<1x32x4x1xf32>
    %852 = tensor.empty() : tensor<1x32x4x4xf32>
    %853 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%838, %851 : tensor<1x32x4x4xf32>, tensor<1x32x4x1xf32>) outs(%852 : tensor<1x32x4x4xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb89(%854: f32, %855: f32, %856: f32):
      %857 = arith.subf %854, %855 : f32
      linalg.yield %857 : f32
    } -> tensor<1x32x4x4xf32>
    %858 = tensor.empty() : tensor<1x32x4x4xf32>
    %859 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%853 : tensor<1x32x4x4xf32>) outs(%858 : tensor<1x32x4x4xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb90(%860: f32, %861: f32):
      %862 = math.exp %860 : f32
      linalg.yield %862 : f32
    } -> tensor<1x32x4x4xf32>
    %863 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} 0.000000e+00 : f32
    %864 = tensor.splat %863 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x32x4xf32>
    %865 = linalg.reduce ins(%859:tensor<1x32x4x4xf32>) outs(%864:tensor<1x32x4xf32>) dimensions = [3]
    (%866: f32, %867: f32) {
      %868 = arith.addf %866, %867 : f32
      linalg.yield %868 : f32
    }
    %869 = tensor.collapse_shape %865 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x32x4xf32> into tensor<128xf32>
    %870 = tensor.expand_shape %869 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<128xf32> into tensor<1x32x4x1xf32>
    %871 = tensor.empty() : tensor<1x32x4x4xf32>
    %872 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%859, %870 : tensor<1x32x4x4xf32>, tensor<1x32x4x1xf32>) outs(%871 : tensor<1x32x4x4xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb91(%873: f32, %874: f32, %875: f32):
      %876 = arith.divf %873, %874 : f32
      linalg.yield %876 : f32
    } -> tensor<1x32x4x4xf32>
    %877 = tensor.empty() : tensor<1x32x4x4xf32>
    %878 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%872 : tensor<1x32x4x4xf32>) outs(%877 : tensor<1x32x4x4xf32>) attrs =  {prov.region_id = "expand_14", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb92(%879: f32, %880: f32):
      linalg.yield %879 : f32
    } -> tensor<1x32x4x4xf32>
    %881 = tensor.collapse_shape %878 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x32x4x4xf32> into tensor<512xf32>
    %882 = tensor.expand_shape %881 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 4, 4] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<512xf32> into tensor<32x4x4xf32>
    %883 = tensor.empty() : tensor<1x32x4x64xf32>
    %884 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%796 : tensor<1x32x4x64xf32>) outs(%883 : tensor<1x32x4x64xf32>) attrs =  {prov.region_id = "expand_15", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb93(%885: f32, %886: f32):
      linalg.yield %885 : f32
    } -> tensor<1x32x4x64xf32>
    %887 = tensor.collapse_shape %884 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x32x4x64xf32> into tensor<8192xf32>
    %888 = tensor.expand_shape %887 [[0 : i64, 1 : i64, 2 : i64]] output_shape [32, 4, 64] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<32x4x64xf32>
    %889 = arith.constant {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} 0.000000e+00 : f32
    %890 = tensor.splat %889 {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<32x4x64xf32>
    %891 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%882, %888 : tensor<32x4x4xf32>, tensor<32x4x64xf32>) outs(%890 : tensor<32x4x64xf32>) attrs =  {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} {
    ^bb94(%892: f32, %893: f32, %894: f32):
      %895 = arith.mulf %892, %893 : f32
      %896 = arith.addf %894, %895 : f32
      linalg.yield %896 : f32
    } -> tensor<32x4x64xf32>
    %897 = tensor.collapse_shape %891 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<32x4x64xf32> into tensor<8192xf32>
    %898 = tensor.expand_shape %897 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4, 64] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x32x4x64xf32>
    %899 = tensor.empty() : tensor<1x4x32x64xf32>
    %900 = linalg.transpose ins(%898:tensor<1x32x4x64xf32>) outs(%899:tensor<1x4x32x64xf32>) permutation = [0, 2, 1, 3]
    %901 = tensor.collapse_shape %900 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<1x4x32x64xf32> into tensor<8192xf32>
    %902 = tensor.expand_shape %901 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 2048] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn"} : tensor<8192xf32> into tensor<1x4x2048xf32>
    %903 = tensor.empty() : tensor<2048x2048xf32>
    %904 = linalg.transpose ins(%13:tensor<2048x2048xf32>) outs(%903:tensor<2048x2048xf32>) permutation = [1, 0]
    %905 = tensor.collapse_shape %902 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.o_proj"} : tensor<1x4x2048xf32> into tensor<8192xf32>
    %906 = tensor.expand_shape %905 [[0 : i64, 1 : i64]] output_shape [4, 2048] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.o_proj"} : tensor<8192xf32> into tensor<4x2048xf32>
    %907 = tensor.empty() : tensor<4x2048xf32>
    %908 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %909 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%908 : f32) outs(%907 : tensor<4x2048xf32>) -> tensor<4x2048xf32>
    %910 = linalg.matmul {prov.region_id = "matmul_15", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.o_proj", prov.transposed_b = "true"} ins(%906, %904 : tensor<4x2048xf32>, tensor<2048x2048xf32>) outs(%909 : tensor<4x2048xf32>) -> tensor<4x2048xf32>
    %911 = tensor.collapse_shape %910 [[0 : i64, 1 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.o_proj"} : tensor<4x2048xf32> into tensor<8192xf32>
    %912 = tensor.expand_shape %911 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 2048] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.self_attn.o_proj"} : tensor<8192xf32> into tensor<1x4x2048xf32>
    %913 = tensor.empty() : tensor<1x4x2048xf32>
    %914 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%631, %912 : tensor<1x4x2048xf32>, tensor<1x4x2048xf32>) outs(%913 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1"} {
    ^bb95(%915: f32, %916: f32, %917: f32):
      %918 = arith.addf %915, %916 : f32
      linalg.yield %918 : f32
    } -> tensor<1x4x2048xf32>
    %919 = tensor.empty() : tensor<1x4x2048xf32>
    %920 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%914 : tensor<1x4x2048xf32>) outs(%919 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.post_attention_layernorm"} {
    ^bb96(%921: f32, %922: f32):
      %923 = arith.constant 2.000000e+00 : f32
      %924 = math.powf %921, %923 : f32
      linalg.yield %924 : f32
    } -> tensor<1x4x2048xf32>
    %925 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.post_attention_layernorm"} 0.000000e+00 : f32
    %926 = tensor.splat %925 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.post_attention_layernorm"} : tensor<1x4xf32>
    %927 = linalg.reduce ins(%920:tensor<1x4x2048xf32>) outs(%926:tensor<1x4xf32>) dimensions = [2]
    (%928: f32, %929: f32) {
      %930 = arith.addf %928, %929 : f32
      linalg.yield %930 : f32
    }
    %931 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.post_attention_layernorm"} 2.048000e+03 : f32
    %932 = tensor.splat %931 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.post_attention_layernorm"} : tensor<1x4xf32>
    %933 = tensor.empty() : tensor<1x4xf32>
    %934 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%927, %932 : tensor<1x4xf32>, tensor<1x4xf32>) outs(%933 : tensor<1x4xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.post_attention_layernorm"} {
    ^bb97(%935: f32, %936: f32, %937: f32):
      %938 = arith.divf %935, %936 : f32
      linalg.yield %938 : f32
    } -> tensor<1x4xf32>
    %939 = tensor.collapse_shape %934 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.post_attention_layernorm"} : tensor<1x4xf32> into tensor<4xf32>
    %940 = tensor.expand_shape %939 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.post_attention_layernorm"} : tensor<4xf32> into tensor<1x4x1xf32>
    %941 = arith.constant {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.post_attention_layernorm"} 1.000000e-05 : f32
    %942 = tensor.splat %941 {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.post_attention_layernorm"} : tensor<1x4x1xf32>
    %943 = tensor.empty() : tensor<1x4x1xf32>
    %944 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%940, %942 : tensor<1x4x1xf32>, tensor<1x4x1xf32>) outs(%943 : tensor<1x4x1xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.post_attention_layernorm"} {
    ^bb98(%945: f32, %946: f32, %947: f32):
      %948 = arith.addf %945, %946 : f32
      linalg.yield %948 : f32
    } -> tensor<1x4x1xf32>
    %949 = tensor.empty() : tensor<1x4x1xf32>
    %950 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%944 : tensor<1x4x1xf32>) outs(%949 : tensor<1x4x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.post_attention_layernorm"} {
    ^bb99(%951: f32, %952: f32):
      %953 = math.rsqrt %951 : f32
      linalg.yield %953 : f32
    } -> tensor<1x4x1xf32>
    %954 = tensor.empty() : tensor<1x4x2048xf32>
    %955 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%914, %950 : tensor<1x4x2048xf32>, tensor<1x4x1xf32>) outs(%954 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.post_attention_layernorm"} {
    ^bb100(%956: f32, %957: f32, %958: f32):
      %959 = arith.mulf %956, %957 : f32
      linalg.yield %959 : f32
    } -> tensor<1x4x2048xf32>
    %960 = tensor.empty() : tensor<1x4x2048xf32>
    %961 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18, %955 : tensor<2048xf32>, tensor<1x4x2048xf32>) outs(%960 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.post_attention_layernorm"} {
    ^bb101(%962: f32, %963: f32, %964: f32):
      %965 = arith.mulf %962, %963 : f32
      linalg.yield %965 : f32
    } -> tensor<1x4x2048xf32>
    %966 = tensor.empty() : tensor<2048x5632xf32>
    %967 = linalg.transpose ins(%14:tensor<5632x2048xf32>) outs(%966:tensor<2048x5632xf32>) permutation = [1, 0]
    %968 = tensor.collapse_shape %961 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp.gate_proj"} : tensor<1x4x2048xf32> into tensor<8192xf32>
    %969 = tensor.expand_shape %968 [[0 : i64, 1 : i64]] output_shape [4, 2048] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp.gate_proj"} : tensor<8192xf32> into tensor<4x2048xf32>
    %970 = tensor.empty() : tensor<4x5632xf32>
    %971 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %972 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%971 : f32) outs(%970 : tensor<4x5632xf32>) -> tensor<4x5632xf32>
    %973 = linalg.matmul {prov.region_id = "matmul_16", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp.gate_proj", prov.transposed_b = "true"} ins(%969, %967 : tensor<4x2048xf32>, tensor<2048x5632xf32>) outs(%972 : tensor<4x5632xf32>) -> tensor<4x5632xf32>
    %974 = tensor.collapse_shape %973 [[0 : i64, 1 : i64]] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp.gate_proj"} : tensor<4x5632xf32> into tensor<22528xf32>
    %975 = tensor.expand_shape %974 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 5632] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp.gate_proj"} : tensor<22528xf32> into tensor<1x4x5632xf32>
    %976 = tensor.empty() : tensor<1x4x5632xf32>
    %977 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%975 : tensor<1x4x5632xf32>) outs(%976 : tensor<1x4x5632xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp.act_fn"} {
    ^bb102(%978: f32, %979: f32):
      %980 = arith.constant 1.000000e+00 : f32
      %981 = arith.negf %978 : f32
      %982 = math.exp %981 : f32
      %983 = arith.addf %980, %982 : f32
      %984 = arith.divf %980, %983 : f32
      linalg.yield %984 : f32
    } -> tensor<1x4x5632xf32>
    %985 = tensor.empty() : tensor<1x4x5632xf32>
    %986 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%975, %977 : tensor<1x4x5632xf32>, tensor<1x4x5632xf32>) outs(%985 : tensor<1x4x5632xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp.act_fn"} {
    ^bb103(%987: f32, %988: f32, %989: f32):
      %990 = arith.mulf %987, %988 : f32
      linalg.yield %990 : f32
    } -> tensor<1x4x5632xf32>
    %991 = tensor.empty() : tensor<2048x5632xf32>
    %992 = linalg.transpose ins(%15:tensor<5632x2048xf32>) outs(%991:tensor<2048x5632xf32>) permutation = [1, 0]
    %993 = tensor.collapse_shape %961 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp.up_proj"} : tensor<1x4x2048xf32> into tensor<8192xf32>
    %994 = tensor.expand_shape %993 [[0 : i64, 1 : i64]] output_shape [4, 2048] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp.up_proj"} : tensor<8192xf32> into tensor<4x2048xf32>
    %995 = tensor.empty() : tensor<4x5632xf32>
    %996 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %997 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%996 : f32) outs(%995 : tensor<4x5632xf32>) -> tensor<4x5632xf32>
    %998 = linalg.matmul {prov.region_id = "matmul_17", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp.up_proj", prov.transposed_b = "true"} ins(%994, %992 : tensor<4x2048xf32>, tensor<2048x5632xf32>) outs(%997 : tensor<4x5632xf32>) -> tensor<4x5632xf32>
    %999 = tensor.collapse_shape %998 [[0 : i64, 1 : i64]] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp.up_proj"} : tensor<4x5632xf32> into tensor<22528xf32>
    %1000 = tensor.expand_shape %999 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 5632] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp.up_proj"} : tensor<22528xf32> into tensor<1x4x5632xf32>
    %1001 = tensor.empty() : tensor<1x4x5632xf32>
    %1002 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%986, %1000 : tensor<1x4x5632xf32>, tensor<1x4x5632xf32>) outs(%1001 : tensor<1x4x5632xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp"} {
    ^bb104(%1003: f32, %1004: f32, %1005: f32):
      %1006 = arith.mulf %1003, %1004 : f32
      linalg.yield %1006 : f32
    } -> tensor<1x4x5632xf32>
    %1007 = tensor.empty() : tensor<5632x2048xf32>
    %1008 = linalg.transpose ins(%16:tensor<2048x5632xf32>) outs(%1007:tensor<5632x2048xf32>) permutation = [1, 0]
    %1009 = tensor.collapse_shape %1002 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp.down_proj"} : tensor<1x4x5632xf32> into tensor<22528xf32>
    %1010 = tensor.expand_shape %1009 [[0 : i64, 1 : i64]] output_shape [4, 5632] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp.down_proj"} : tensor<22528xf32> into tensor<4x5632xf32>
    %1011 = tensor.empty() : tensor<4x2048xf32>
    %1012 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %1013 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%1012 : f32) outs(%1011 : tensor<4x2048xf32>) -> tensor<4x2048xf32>
    %1014 = linalg.matmul {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp.down_proj", prov.transposed_b = "true"} ins(%1010, %1008 : tensor<4x5632xf32>, tensor<5632x2048xf32>) outs(%1013 : tensor<4x2048xf32>) -> tensor<4x2048xf32>
    %1015 = tensor.collapse_shape %1014 [[0 : i64, 1 : i64]] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp.down_proj"} : tensor<4x2048xf32> into tensor<8192xf32>
    %1016 = tensor.expand_shape %1015 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 2048] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1.mlp.down_proj"} : tensor<8192xf32> into tensor<1x4x2048xf32>
    %1017 = tensor.empty() : tensor<1x4x2048xf32>
    %1018 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%914, %1016 : tensor<1x4x2048xf32>, tensor<1x4x2048xf32>) outs(%1017 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.layers.1"} {
    ^bb105(%1019: f32, %1020: f32, %1021: f32):
      %1022 = arith.addf %1019, %1020 : f32
      linalg.yield %1022 : f32
    } -> tensor<1x4x2048xf32>
    %1023 = tensor.empty() : tensor<1x4x2048xf32>
    %1024 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1018 : tensor<1x4x2048xf32>) outs(%1023 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.norm"} {
    ^bb106(%1025: f32, %1026: f32):
      %1027 = arith.constant 2.000000e+00 : f32
      %1028 = math.powf %1025, %1027 : f32
      linalg.yield %1028 : f32
    } -> tensor<1x4x2048xf32>
    %1029 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.norm"} 0.000000e+00 : f32
    %1030 = tensor.splat %1029 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.norm"} : tensor<1x4xf32>
    %1031 = linalg.reduce ins(%1024:tensor<1x4x2048xf32>) outs(%1030:tensor<1x4xf32>) dimensions = [2]
    (%1032: f32, %1033: f32) {
      %1034 = arith.addf %1032, %1033 : f32
      linalg.yield %1034 : f32
    }
    %1035 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.norm"} 2.048000e+03 : f32
    %1036 = tensor.splat %1035 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.norm"} : tensor<1x4xf32>
    %1037 = tensor.empty() : tensor<1x4xf32>
    %1038 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1031, %1036 : tensor<1x4xf32>, tensor<1x4xf32>) outs(%1037 : tensor<1x4xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.norm"} {
    ^bb107(%1039: f32, %1040: f32, %1041: f32):
      %1042 = arith.divf %1039, %1040 : f32
      linalg.yield %1042 : f32
    } -> tensor<1x4xf32>
    %1043 = tensor.collapse_shape %1038 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.norm"} : tensor<1x4xf32> into tensor<4xf32>
    %1044 = tensor.expand_shape %1043 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.norm"} : tensor<4xf32> into tensor<1x4x1xf32>
    %1045 = arith.constant {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.norm"} 1.000000e-05 : f32
    %1046 = tensor.splat %1045 {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.norm"} : tensor<1x4x1xf32>
    %1047 = tensor.empty() : tensor<1x4x1xf32>
    %1048 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1044, %1046 : tensor<1x4x1xf32>, tensor<1x4x1xf32>) outs(%1047 : tensor<1x4x1xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.norm"} {
    ^bb108(%1049: f32, %1050: f32, %1051: f32):
      %1052 = arith.addf %1049, %1050 : f32
      linalg.yield %1052 : f32
    } -> tensor<1x4x1xf32>
    %1053 = tensor.empty() : tensor<1x4x1xf32>
    %1054 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1048 : tensor<1x4x1xf32>) outs(%1053 : tensor<1x4x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.norm"} {
    ^bb109(%1055: f32, %1056: f32):
      %1057 = math.rsqrt %1055 : f32
      linalg.yield %1057 : f32
    } -> tensor<1x4x1xf32>
    %1058 = tensor.empty() : tensor<1x4x2048xf32>
    %1059 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1018, %1054 : tensor<1x4x2048xf32>, tensor<1x4x1xf32>) outs(%1058 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.norm"} {
    ^bb110(%1060: f32, %1061: f32, %1062: f32):
      %1063 = arith.mulf %1060, %1061 : f32
      linalg.yield %1063 : f32
    } -> tensor<1x4x2048xf32>
    %1064 = tensor.empty() : tensor<1x4x2048xf32>
    %1065 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %1059 : tensor<2048xf32>, tensor<1x4x2048xf32>) outs(%1064 : tensor<1x4x2048xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.norm"} {
    ^bb111(%1066: f32, %1067: f32, %1068: f32):
      %1069 = arith.mulf %1066, %1067 : f32
      linalg.yield %1069 : f32
    } -> tensor<1x4x2048xf32>
    %1070 = tensor.empty() : tensor<2048x32000xf32>
    %1071 = linalg.transpose ins(%20:tensor<32000x2048xf32>) outs(%1070:tensor<2048x32000xf32>) permutation = [1, 0]
    %1072 = tensor.collapse_shape %1065 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.lm_head"} : tensor<1x4x2048xf32> into tensor<8192xf32>
    %1073 = tensor.expand_shape %1072 [[0 : i64, 1 : i64]] output_shape [4, 2048] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.lm_head"} : tensor<8192xf32> into tensor<4x2048xf32>
    %1074 = tensor.empty() : tensor<4x32000xf32>
    %1075 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %1076 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%1075 : f32) outs(%1074 : tensor<4x32000xf32>) -> tensor<4x32000xf32>
    %1077 = linalg.matmul {prov.region_id = "matmul_19", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.lm_head", prov.transposed_b = "true"} ins(%1073, %1071 : tensor<4x2048xf32>, tensor<2048x32000xf32>) outs(%1076 : tensor<4x32000xf32>) -> tensor<4x32000xf32>
    %1078 = tensor.collapse_shape %1077 [[0 : i64, 1 : i64]] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.lm_head"} : tensor<4x32000xf32> into tensor<128000xf32>
    %1079 = tensor.expand_shape %1078 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 4, 32000] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.lm_head"} : tensor<128000xf32> into tensor<1x4x32000xf32>
    func.return %1079 : tensor<1x4x32000xf32>
  }
}
