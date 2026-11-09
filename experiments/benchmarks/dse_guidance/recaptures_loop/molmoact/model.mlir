builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func private @aten_index_put_default(tensor<4x16x128xf32>, tensor<8xi64>, tensor<1x4x8x128xf32>) -> tensor<1x4x16x128xf32>
  func.func private @aten_index_put_default_wl0(tensor<1x8xi64>, tensor<1xi64>, tensor<1x1xi64>) -> tensor<1x8xi64>
  func.func private @aten_index_put_default_1_wl1(tensor<4x16x128xf32>, tensor<1xi64>, tensor<1x4x1x128xf32>) -> tensor<1x4x16x128xf32>
  func.func @forward(%0: tensor<4096x3584xf32>, %1: tensor<128x3584xf32>, %2: tensor<3584xf32>, %3: tensor<3584xf32>, %4: tensor<4608x3584xf32>, %5: tensor<4608xf32>, %6: tensor<3584x3584xf32>, %7: tensor<37888x3584xf32>, %8: tensor<3584x18944xf32>, %9: tensor<3584xf32>, %10: tensor<3584xf32>, %11: tensor<4608x3584xf32>, %12: tensor<4608xf32>, %13: tensor<3584x3584xf32>, %14: tensor<37888x3584xf32>, %15: tensor<3584x18944xf32>, %16: tensor<3584xf32>, %17: tensor<3584xf32>, %18: tensor<4608x3584xf32>, %19: tensor<4608xf32>, %20: tensor<3584x3584xf32>, %21: tensor<37888x3584xf32>, %22: tensor<3584x18944xf32>, %23: tensor<3584xf32>, %24: tensor<3584xf32>, %25: tensor<4608x3584xf32>, %26: tensor<4608xf32>, %27: tensor<3584x3584xf32>, %28: tensor<37888x3584xf32>, %29: tensor<3584x18944xf32>, %30: tensor<3584xf32>, %31: tensor<4096x3584xf32>, %32: tensor<i64>, %33: tensor<64xf32>, %34: tensor<1x8xi64>) -> tensor<1x8xi64> {
    %35 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %36 = tensor.splat %35 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32"} : tensor<4x1x4x16x128xf32>
    %37 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %38 = tensor.splat %37 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32"} : tensor<4x1x4x16x128xf32>
    %39 = arith.constant {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ne.Scalar", prov.orig_dtype = "bool"} -1 : i64
    %40 = tensor.splat %39 {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ne.Scalar", prov.orig_dtype = "bool"} : tensor<1x8xi64>
    %41 = tensor.empty() : tensor<1x8xi1>
    %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%34, %40 : tensor<1x8xi64>, tensor<1x8xi64>) outs(%41 : tensor<1x8xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.ne.Scalar", prov.orig_dtype = "bool"} {
    ^bb0(%43: i64, %44: i64, %45: i1):
      %46 = arith.cmpi ne, %43, %44 : i64
      linalg.yield %46 : i1
    } -> tensor<1x8xi1>
    %47 = tensor.empty() : tensor<1x8xi64>
    %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%42 : tensor<1x8xi1>) outs(%47 : tensor<1x8xi64>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int64"} {
    ^bb1(%49: i1, %50: i64):
      %51 = arith.extui %49 : i1 to i64
      linalg.yield %51 : i64
    } -> tensor<1x8xi64>
    %52 = tensor.empty() : tensor<1x8xi64>
    %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%34, %48 : tensor<1x8xi64>, tensor<1x8xi64>) outs(%52 : tensor<1x8xi64>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} {
    ^bb2(%54: i64, %55: i64, %56: i64):
      %57 = arith.muli %54, %55 : i64
      linalg.yield %57 : i64
    } -> tensor<1x8xi64>
    %58 = tensor.concat dim(0) %0, %1 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "embed", prov.fqn = "embed"} : (tensor<4096x3584xf32>, tensor<128x3584xf32>) -> tensor<4224x3584xf32>
    %59 = tensor.empty() : tensor<1x8x3584xf32>
    %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%53 : tensor<1x8xi64>) outs(%59 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32", prov.module = "embed", prov.fqn = "embed"} {
    ^bb3(%61: i64, %62: f32):
      %63 = arith.index_cast %61 : i64 to index
      %64 = linalg.index 2 : index
      %65 = tensor.extract %58[%63, %64] : tensor<4224x3584xf32>
      linalg.yield %65 : f32
    } -> tensor<1x8x3584xf32>
    %66 = arith.constant {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "int64"} 0 : i64
    %67 = tensor.splat %66 {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "int64"} : tensor<i64>
    %68 = tensor.empty() : tensor<8xi64>
    %69 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%68 : tensor<8xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb4(%70: i64):
      %71 = linalg.index 0 : index
      %72 = arith.index_cast %71 : index to i64
      %73 = arith.constant 1 : i64
      %74 = arith.muli %72, %73 : i64
      %75 = arith.constant 0 : i64
      %76 = arith.addi %75, %74 : i64
      linalg.yield %76 : i64
    } -> tensor<8xi64>
    %77 = tensor.expand_shape %69 [[0 : i64, 1 : i64]] output_shape [1, 8] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<8xi64> into tensor<1x8xi64>
    %78 = tensor.expand_shape %33 [[0 : i64, 1 : i64]] output_shape [1, 64] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<64xf32> into tensor<1x64xf32>
    %79 = "tensor.extract_slice"(%78) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 64>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x64xf32>) -> tensor<1x64xf32>
    %80 = tensor.collapse_shape %79 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x64xf32> into tensor<64xf32>
    %81 = tensor.expand_shape %80 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 1] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<64xf32> into tensor<1x64x1xf32>
    %82 = tensor.empty() : tensor<1x64x1xf32>
    %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81 : tensor<1x64x1xf32>) outs(%82 : tensor<1x64x1xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb5(%84: f32, %85: f32):
      linalg.yield %84 : f32
    } -> tensor<1x64x1xf32>
    %86 = "tensor.extract_slice"(%77) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 8>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int64"} : (tensor<1x8xi64>) -> tensor<1x8xi64>
    %87 = tensor.collapse_shape %86 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1x8xi64> into tensor<8xi64>
    %88 = tensor.expand_shape %87 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 8] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<8xi64> into tensor<1x1x8xi64>
    %89 = "tensor.extract_slice"(%88) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 8>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int64"} : (tensor<1x1x8xi64>) -> tensor<1x1x8xi64>
    %90 = tensor.empty() : tensor<1x1x8xf32>
    %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89 : tensor<1x1x8xi64>) outs(%90 : tensor<1x1x8xf32>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb6(%92: i64, %93: f32):
      %94 = arith.sitofp %92 : i64 to f32
      linalg.yield %94 : f32
    } -> tensor<1x1x8xf32>
    %95 = tensor.empty() : tensor<1x64x1xf32>
    %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83 : tensor<1x64x1xf32>) outs(%95 : tensor<1x64x1xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb7(%97: f32, %98: f32):
      linalg.yield %97 : f32
    } -> tensor<1x64x1xf32>
    %99 = tensor.empty() : tensor<1x1x8xf32>
    %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%91 : tensor<1x1x8xf32>) outs(%99 : tensor<1x1x8xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb8(%101: f32, %102: f32):
      linalg.yield %101 : f32
    } -> tensor<1x1x8xf32>
    %103 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %104 = tensor.splat %103 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<1x64x8xf32>
    %105 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%96, %100 : tensor<1x64x1xf32>, tensor<1x1x8xf32>) outs(%104 : tensor<1x64x8xf32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb9(%106: f32, %107: f32, %108: f32):
      %109 = arith.mulf %106, %107 : f32
      %110 = arith.addf %108, %109 : f32
      linalg.yield %110 : f32
    } -> tensor<1x64x8xf32>
    %111 = tensor.empty() : tensor<1x8x64xf32>
    %112 = linalg.transpose ins(%105:tensor<1x64x8xf32>) outs(%111:tensor<1x8x64xf32>) permutation = [0, 2, 1]
    %113 = tensor.concat dim(2) %112, %112 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x8x128xf32>
    %114 = tensor.empty() : tensor<1x8x128xf32>
    %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%113 : tensor<1x8x128xf32>) outs(%114 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32"} {
    ^bb10(%116: f32, %117: f32):
      %118 = math.cos %116 : f32
      linalg.yield %118 : f32
    } -> tensor<1x8x128xf32>
    %119 = arith.constant {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
    %120 = tensor.splat %119 {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x128xf32>
    %121 = tensor.empty() : tensor<1x8x128xf32>
    %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%115, %120 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%121 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb11(%123: f32, %124: f32, %125: f32):
      %126 = arith.mulf %123, %124 : f32
      linalg.yield %126 : f32
    } -> tensor<1x8x128xf32>
    %127 = tensor.empty() : tensor<1x8x128xf32>
    %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%113 : tensor<1x8x128xf32>) outs(%127 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32"} {
    ^bb12(%129: f32, %130: f32):
      %131 = math.sin %129 : f32
      linalg.yield %131 : f32
    } -> tensor<1x8x128xf32>
    %132 = arith.constant {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
    %133 = tensor.splat %132 {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x128xf32>
    %134 = tensor.empty() : tensor<1x8x128xf32>
    %135 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%128, %133 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%134 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb13(%136: f32, %137: f32, %138: f32):
      %139 = arith.mulf %136, %137 : f32
      linalg.yield %139 : f32
    } -> tensor<1x8x128xf32>
    %140 = tensor.empty() : tensor<16xi64>
    %141 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%140 : tensor<16xi64>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb14(%142: i64):
      %143 = linalg.index 0 : index
      %144 = arith.index_cast %143 : index to i64
      %145 = arith.constant 1 : i64
      %146 = arith.muli %144, %145 : i64
      %147 = arith.constant 0 : i64
      %148 = arith.addi %147, %146 : i64
      linalg.yield %148 : i64
    } -> tensor<16xi64>
    %149 = tensor.empty() : tensor<8xi64>
    %150 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%149 : tensor<8xi64>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb15(%151: i64):
      %152 = linalg.index 0 : index
      %153 = arith.index_cast %152 : index to i64
      %154 = arith.constant 1 : i64
      %155 = arith.muli %153, %154 : i64
      %156 = arith.constant 0 : i64
      %157 = arith.addi %156, %155 : i64
      linalg.yield %157 : i64
    } -> tensor<8xi64>
    %158 = tensor.empty() : tensor<8xi64>
    %159 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%67, %150 : tensor<i64>, tensor<8xi64>) outs(%158 : tensor<8xi64>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb16(%160: i64, %161: i64, %162: i64):
      %163 = arith.addi %160, %161 : i64
      linalg.yield %163 : i64
    } -> tensor<8xi64>
    %164 = tensor.expand_shape %159 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<8xi64> into tensor<8x1xi64>
    %165 = tensor.expand_shape %141 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<16xi64> into tensor<1x16xi64>
    %166 = tensor.empty() : tensor<8x16xi1>
    %167 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%165, %164 : tensor<1x16xi64>, tensor<8x1xi64>) outs(%166 : tensor<8x16xi1>) attrs =  {prov.region_id = "compare_1", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
    ^bb17(%168: i64, %169: i64, %170: i1):
      %171 = arith.cmpi sle, %168, %169 : i64
      linalg.yield %171 : i1
    } -> tensor<8x16xi1>
    %172 = tensor.collapse_shape %167 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<8x16xi1> into tensor<128xi1>
    %173 = tensor.expand_shape %172 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 16] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<128xi1> into tensor<1x8x16xi1>
    %174 = tensor.collapse_shape %173 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x8x16xi1> into tensor<128xi1>
    %175 = tensor.expand_shape %174 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 16] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<128xi1> into tensor<1x1x8x16xi1>
    %176 = tensor.empty() : tensor<1x8x3584xf32>
    %177 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60 : tensor<1x8x3584xf32>) outs(%176 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb18(%178: f32, %179: f32):
      %180 = arith.constant 2.000000e+00 : f32
      %181 = math.powf %178, %180 : f32
      linalg.yield %181 : f32
    } -> tensor<1x8x3584xf32>
    %182 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %183 = tensor.splat %182 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %184 = linalg.reduce ins(%177:tensor<1x8x3584xf32>) outs(%183:tensor<1x8xf32>) dimensions = [2]
    (%185: f32, %186: f32) {
      %187 = arith.addf %185, %186 : f32
      linalg.yield %187 : f32
    }
    %188 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %189 = tensor.splat %188 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %190 = tensor.empty() : tensor<1x8xf32>
    %191 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%184, %189 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%190 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb19(%192: f32, %193: f32, %194: f32):
      %195 = arith.divf %192, %193 : f32
      linalg.yield %195 : f32
    } -> tensor<1x8xf32>
    %196 = tensor.collapse_shape %191 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %197 = tensor.expand_shape %196 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %198 = arith.constant {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %199 = tensor.splat %198 {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %200 = tensor.empty() : tensor<1x8x1xf32>
    %201 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%197, %199 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%200 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb20(%202: f32, %203: f32, %204: f32):
      %205 = arith.addf %202, %203 : f32
      linalg.yield %205 : f32
    } -> tensor<1x8x1xf32>
    %206 = tensor.empty() : tensor<1x8x1xf32>
    %207 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%201 : tensor<1x8x1xf32>) outs(%206 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb21(%208: f32, %209: f32):
      %210 = math.rsqrt %208 : f32
      linalg.yield %210 : f32
    } -> tensor<1x8x1xf32>
    %211 = tensor.empty() : tensor<1x8x3584xf32>
    %212 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %207 : tensor<1x8x3584xf32>, tensor<1x8x1xf32>) outs(%211 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb22(%213: f32, %214: f32, %215: f32):
      %216 = arith.mulf %213, %214 : f32
      linalg.yield %216 : f32
    } -> tensor<1x8x3584xf32>
    %217 = tensor.empty() : tensor<1x8x3584xf32>
    %218 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2, %212 : tensor<3584xf32>, tensor<1x8x3584xf32>) outs(%217 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb23(%219: f32, %220: f32, %221: f32):
      %222 = arith.mulf %219, %220 : f32
      linalg.yield %222 : f32
    } -> tensor<1x8x3584xf32>
    %223 = tensor.collapse_shape %218 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.att_proj"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %224 = tensor.expand_shape %223 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.att_proj"} : tensor<28672xf32> into tensor<8x3584xf32>
    %225 = tensor.empty() : tensor<3584x4608xf32>
    %226 = linalg.transpose ins(%4:tensor<4608x3584xf32>) outs(%225:tensor<3584x4608xf32>) permutation = [1, 0]
    %227 = tensor.empty() : tensor<8x4608xf32>
    %228 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %229 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%228 : f32) outs(%227 : tensor<8x4608xf32>) -> tensor<8x4608xf32>
    %230 = linalg.matmul {prov.region_id = "matmul_1", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.att_proj", prov.transposed_b = "true"} ins(%224, %226 : tensor<8x3584xf32>, tensor<3584x4608xf32>) outs(%229 : tensor<8x4608xf32>) -> tensor<8x4608xf32>
    %231 = tensor.empty() : tensor<8x4608xf32>
    %232 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%230, %5 : tensor<8x4608xf32>, tensor<4608xf32>) outs(%231 : tensor<8x4608xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.att_proj"} {
    ^bb24(%233: f32, %234: f32, %235: f32):
      %236 = arith.addf %233, %234 : f32
      linalg.yield %236 : f32
    } -> tensor<8x4608xf32>
    %237 = tensor.collapse_shape %232 [[0 : i64, 1 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.att_proj"} : tensor<8x4608xf32> into tensor<36864xf32>
    %238 = tensor.expand_shape %237 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 4608] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.att_proj"} : tensor<36864xf32> into tensor<1x8x4608xf32>
    %239 = "tensor.extract_slice"(%238) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x4608xf32>) -> tensor<1x8x3584xf32>
    %240 = "tensor.extract_slice"(%238) <{static_offsets = array<i64: 0, 0, 3584>, static_sizes = array<i64: 1, 8, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x4608xf32>) -> tensor<1x8x512xf32>
    %241 = "tensor.extract_slice"(%238) <{static_offsets = array<i64: 0, 0, 4096>, static_sizes = array<i64: 1, 8, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x4608xf32>) -> tensor<1x8x512xf32>
    %242 = tensor.collapse_shape %239 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %243 = tensor.expand_shape %242 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 128] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x8x28x128xf32>
    %244 = tensor.empty() : tensor<1x28x8x128xf32>
    %245 = linalg.transpose ins(%243:tensor<1x8x28x128xf32>) outs(%244:tensor<1x28x8x128xf32>) permutation = [0, 2, 1, 3]
    %246 = tensor.collapse_shape %240 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x512xf32> into tensor<4096xf32>
    %247 = tensor.expand_shape %246 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 128] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x8x4x128xf32>
    %248 = tensor.empty() : tensor<1x4x8x128xf32>
    %249 = linalg.transpose ins(%247:tensor<1x8x4x128xf32>) outs(%248:tensor<1x4x8x128xf32>) permutation = [0, 2, 1, 3]
    %250 = tensor.collapse_shape %241 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x512xf32> into tensor<4096xf32>
    %251 = tensor.expand_shape %250 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 128] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x8x4x128xf32>
    %252 = tensor.empty() : tensor<1x4x8x128xf32>
    %253 = linalg.transpose ins(%251:tensor<1x8x4x128xf32>) outs(%252:tensor<1x4x8x128xf32>) permutation = [0, 2, 1, 3]
    %254 = tensor.collapse_shape %122 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %255 = tensor.expand_shape %254 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 128] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x8x128xf32>
    %256 = tensor.collapse_shape %135 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %257 = tensor.expand_shape %256 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 128] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x8x128xf32>
    %258 = tensor.empty() : tensor<1x28x8x128xf32>
    %259 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%245, %255 : tensor<1x28x8x128xf32>, tensor<1x1x8x128xf32>) outs(%258 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb25(%260: f32, %261: f32, %262: f32):
      %263 = arith.mulf %260, %261 : f32
      linalg.yield %263 : f32
    } -> tensor<1x28x8x128xf32>
    %264 = "tensor.extract_slice"(%245) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 28, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x28x8x128xf32>) -> tensor<1x28x8x64xf32>
    %265 = "tensor.extract_slice"(%245) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 28, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x28x8x128xf32>) -> tensor<1x28x8x64xf32>
    %266 = tensor.empty() : tensor<1x28x8x64xf32>
    %267 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%265 : tensor<1x28x8x64xf32>) outs(%266 : tensor<1x28x8x64xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb26(%268: f32, %269: f32):
      %270 = arith.negf %268 : f32
      linalg.yield %270 : f32
    } -> tensor<1x28x8x64xf32>
    %271 = tensor.concat dim(3) %267, %264 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x28x8x64xf32>, tensor<1x28x8x64xf32>) -> tensor<1x28x8x128xf32>
    %272 = tensor.empty() : tensor<1x28x8x128xf32>
    %273 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%271, %257 : tensor<1x28x8x128xf32>, tensor<1x1x8x128xf32>) outs(%272 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb27(%274: f32, %275: f32, %276: f32):
      %277 = arith.mulf %274, %275 : f32
      linalg.yield %277 : f32
    } -> tensor<1x28x8x128xf32>
    %278 = tensor.empty() : tensor<1x28x8x128xf32>
    %279 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%259, %273 : tensor<1x28x8x128xf32>, tensor<1x28x8x128xf32>) outs(%278 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb28(%280: f32, %281: f32, %282: f32):
      %283 = arith.addf %280, %281 : f32
      linalg.yield %283 : f32
    } -> tensor<1x28x8x128xf32>
    %284 = tensor.empty() : tensor<1x4x8x128xf32>
    %285 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%249, %255 : tensor<1x4x8x128xf32>, tensor<1x1x8x128xf32>) outs(%284 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb29(%286: f32, %287: f32, %288: f32):
      %289 = arith.mulf %286, %287 : f32
      linalg.yield %289 : f32
    } -> tensor<1x4x8x128xf32>
    %290 = "tensor.extract_slice"(%249) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x64xf32>
    %291 = "tensor.extract_slice"(%249) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x64xf32>
    %292 = tensor.empty() : tensor<1x4x8x64xf32>
    %293 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%291 : tensor<1x4x8x64xf32>) outs(%292 : tensor<1x4x8x64xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb30(%294: f32, %295: f32):
      %296 = arith.negf %294 : f32
      linalg.yield %296 : f32
    } -> tensor<1x4x8x64xf32>
    %297 = tensor.concat dim(3) %293, %290 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x8x64xf32>, tensor<1x4x8x64xf32>) -> tensor<1x4x8x128xf32>
    %298 = tensor.empty() : tensor<1x4x8x128xf32>
    %299 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%297, %257 : tensor<1x4x8x128xf32>, tensor<1x1x8x128xf32>) outs(%298 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb31(%300: f32, %301: f32, %302: f32):
      %303 = arith.mulf %300, %301 : f32
      linalg.yield %303 : f32
    } -> tensor<1x4x8x128xf32>
    %304 = tensor.empty() : tensor<1x4x8x128xf32>
    %305 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%285, %299 : tensor<1x4x8x128xf32>, tensor<1x4x8x128xf32>) outs(%304 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb32(%306: f32, %307: f32, %308: f32):
      %309 = arith.addf %306, %307 : f32
      linalg.yield %309 : f32
    } -> tensor<1x4x8x128xf32>
    %310 = tensor.empty() : tensor<8xi64>
    %311 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%310 : tensor<8xi64>) attrs =  {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb33(%312: i64):
      %313 = linalg.index 0 : index
      %314 = arith.index_cast %313 : index to i64
      %315 = arith.constant 1 : i64
      %316 = arith.muli %314, %315 : i64
      %317 = arith.constant 0 : i64
      %318 = arith.addi %317, %316 : i64
      linalg.yield %318 : i64
    } -> tensor<8xi64>
    %319 = tensor.empty() : tensor<8xi64>
    %320 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%67, %311 : tensor<i64>, tensor<8xi64>) outs(%319 : tensor<8xi64>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb34(%321: i64, %322: i64, %323: i64):
      %324 = arith.addi %321, %322 : i64
      linalg.yield %324 : i64
    } -> tensor<8xi64>
    %325 = "tensor.extract_slice"(%36) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<4x1x4x16x128xf32>) -> tensor<4x16x128xf32>
    %326 = func.call @aten_index_put_default(%325, %320, %305) {prov.region_id = "aten_index_put_default_0", prov.dispatch_id = "aten_index_put_default_0"} : (tensor<4x16x128xf32>, tensor<8xi64>, tensor<1x4x8x128xf32>) -> tensor<1x4x16x128xf32>
    %327 = "tensor.extract_slice"(%38) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<4x1x4x16x128xf32>) -> tensor<4x16x128xf32>
    %328 = func.call @aten_index_put_default(%327, %320, %253) {prov.region_id = "aten_index_put_default_1", prov.dispatch_id = "aten_index_put_default_1"} : (tensor<4x16x128xf32>, tensor<8xi64>, tensor<1x4x8x128xf32>) -> tensor<1x4x16x128xf32>
    %329 = "tensor.extract_slice"(%326) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
    %330 = "tensor.extract_slice"(%329) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
    %331 = tensor.collapse_shape %330 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x16x128xf32> into tensor<8192xf32>
    %332 = tensor.expand_shape %331 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 16, 128] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x4x1x16x128xf32>
    %333 = "tensor.extract_slice"(%332) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_9", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
    %334 = "tensor.extract_slice"(%333) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_10", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
    %335 = tensor.empty() : tensor<1x4x7x16x128xf32>
    %336 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%334 : tensor<1x4x1x16x128xf32>) outs(%335 : tensor<1x4x7x16x128xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb35(%337: f32, %338: f32):
      linalg.yield %337 : f32
    } -> tensor<1x4x7x16x128xf32>
    %339 = tensor.collapse_shape %336 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x7x16x128xf32> into tensor<57344xf32>
    %340 = tensor.expand_shape %339 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 16, 128] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<1x28x16x128xf32>
    %341 = "tensor.extract_slice"(%328) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_11", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
    %342 = "tensor.extract_slice"(%341) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_12", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
    %343 = tensor.collapse_shape %342 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x16x128xf32> into tensor<8192xf32>
    %344 = tensor.expand_shape %343 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 16, 128] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x4x1x16x128xf32>
    %345 = "tensor.extract_slice"(%344) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_13", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
    %346 = "tensor.extract_slice"(%345) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_14", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
    %347 = tensor.empty() : tensor<1x4x7x16x128xf32>
    %348 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%346 : tensor<1x4x1x16x128xf32>) outs(%347 : tensor<1x4x7x16x128xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb36(%349: f32, %350: f32):
      linalg.yield %349 : f32
    } -> tensor<1x4x7x16x128xf32>
    %351 = tensor.collapse_shape %348 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x7x16x128xf32> into tensor<57344xf32>
    %352 = tensor.expand_shape %351 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 16, 128] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<1x28x16x128xf32>
    %353 = tensor.empty() : tensor<1x28x128x16xf32>
    %354 = linalg.transpose ins(%340:tensor<1x28x16x128xf32>) outs(%353:tensor<1x28x128x16xf32>) permutation = [0, 1, 3, 2]
    %355 = tensor.empty() : tensor<1x28x8x128xf32>
    %356 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%279 : tensor<1x28x8x128xf32>) outs(%355 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb37(%357: f32, %358: f32):
      linalg.yield %357 : f32
    } -> tensor<1x28x8x128xf32>
    %359 = tensor.collapse_shape %356 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
    %360 = tensor.expand_shape %359 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 128] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<28x8x128xf32>
    %361 = tensor.empty() : tensor<1x28x128x16xf32>
    %362 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%354 : tensor<1x28x128x16xf32>) outs(%361 : tensor<1x28x128x16xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb38(%363: f32, %364: f32):
      linalg.yield %363 : f32
    } -> tensor<1x28x128x16xf32>
    %365 = tensor.collapse_shape %362 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x128x16xf32> into tensor<57344xf32>
    %366 = tensor.expand_shape %365 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 128, 16] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<28x128x16xf32>
    %367 = arith.constant {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %368 = tensor.splat %367 {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<28x8x16xf32>
    %369 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%360, %366 : tensor<28x8x128xf32>, tensor<28x128x16xf32>) outs(%368 : tensor<28x8x16xf32>) attrs =  {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb39(%370: f32, %371: f32, %372: f32):
      %373 = arith.mulf %370, %371 : f32
      %374 = arith.addf %372, %373 : f32
      linalg.yield %374 : f32
    } -> tensor<28x8x16xf32>
    %375 = tensor.collapse_shape %369 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28x8x16xf32> into tensor<3584xf32>
    %376 = tensor.expand_shape %375 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 16] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<1x28x8x16xf32>
    %377 = arith.constant {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.0883883461 : f32
    %378 = tensor.splat %377 {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x8x16xf32>
    %379 = tensor.empty() : tensor<1x28x8x16xf32>
    %380 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%376, %378 : tensor<1x28x8x16xf32>, tensor<1x28x8x16xf32>) outs(%379 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb40(%381: f32, %382: f32, %383: f32):
      %384 = arith.mulf %381, %382 : f32
      linalg.yield %384 : f32
    } -> tensor<1x28x8x16xf32>
    %385 = tensor.empty() : tensor<1x1x8x16xi1>
    %386 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%175 : tensor<1x1x8x16xi1>) outs(%385 : tensor<1x1x8x16xi1>) attrs =  {prov.region_id = "bitwise_0", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
    ^bb41(%387: i1, %388: i1):
      %389 = arith.constant true
      %390 = arith.xori %387, %389 : i1
      linalg.yield %390 : i1
    } -> tensor<1x1x8x16xi1>
    %391 = arith.constant {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %392 = tensor.splat %391 {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %393 = tensor.empty() : tensor<1x28x8x16xf32>
    %394 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%386, %392, %380 : tensor<1x1x8x16xi1>, tensor<f32>, tensor<1x28x8x16xf32>) outs(%393 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32"} {
    ^bb42(%395: i1, %396: f32, %397: f32, %398: f32):
      %399 = arith.select %395, %396, %397 : f32
      linalg.yield %399 : f32
    } -> tensor<1x28x8x16xf32>
    %400 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %401 = tensor.splat %400 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x8xf32>
    %402 = linalg.reduce ins(%394:tensor<1x28x8x16xf32>) outs(%401:tensor<1x28x8xf32>) dimensions = [3]
    (%403: f32, %404: f32) {
      %405 = arith.maximumf %403, %404 : f32
      linalg.yield %405 : f32
    }
    %406 = tensor.collapse_shape %402 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x8xf32> into tensor<224xf32>
    %407 = tensor.expand_shape %406 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %408 = tensor.empty() : tensor<1x28x8x16xf32>
    %409 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%394, %407 : tensor<1x28x8x16xf32>, tensor<1x28x8x1xf32>) outs(%408 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb43(%410: f32, %411: f32, %412: f32):
      %413 = arith.subf %410, %411 : f32
      linalg.yield %413 : f32
    } -> tensor<1x28x8x16xf32>
    %414 = tensor.empty() : tensor<1x28x8x16xf32>
    %415 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%409 : tensor<1x28x8x16xf32>) outs(%414 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb44(%416: f32, %417: f32):
      %418 = math.exp %416 : f32
      linalg.yield %418 : f32
    } -> tensor<1x28x8x16xf32>
    %419 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %420 = tensor.splat %419 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x8xf32>
    %421 = linalg.reduce ins(%415:tensor<1x28x8x16xf32>) outs(%420:tensor<1x28x8xf32>) dimensions = [3]
    (%422: f32, %423: f32) {
      %424 = arith.addf %422, %423 : f32
      linalg.yield %424 : f32
    }
    %425 = tensor.collapse_shape %421 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x8xf32> into tensor<224xf32>
    %426 = tensor.expand_shape %425 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %427 = tensor.empty() : tensor<1x28x8x16xf32>
    %428 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%415, %426 : tensor<1x28x8x16xf32>, tensor<1x28x8x1xf32>) outs(%427 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb45(%429: f32, %430: f32, %431: f32):
      %432 = arith.divf %429, %430 : f32
      linalg.yield %432 : f32
    } -> tensor<1x28x8x16xf32>
    %433 = tensor.empty() : tensor<1x28x8x16xf32>
    %434 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%428 : tensor<1x28x8x16xf32>) outs(%433 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb46(%435: f32, %436: f32):
      linalg.yield %435 : f32
    } -> tensor<1x28x8x16xf32>
    %437 = tensor.collapse_shape %434 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x8x16xf32> into tensor<3584xf32>
    %438 = tensor.expand_shape %437 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 16] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<28x8x16xf32>
    %439 = tensor.empty() : tensor<1x28x16x128xf32>
    %440 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%352 : tensor<1x28x16x128xf32>) outs(%439 : tensor<1x28x16x128xf32>) attrs =  {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb47(%441: f32, %442: f32):
      linalg.yield %441 : f32
    } -> tensor<1x28x16x128xf32>
    %443 = tensor.collapse_shape %440 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x16x128xf32> into tensor<57344xf32>
    %444 = tensor.expand_shape %443 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 16, 128] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<28x16x128xf32>
    %445 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %446 = tensor.splat %445 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<28x8x128xf32>
    %447 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%438, %444 : tensor<28x8x16xf32>, tensor<28x16x128xf32>) outs(%446 : tensor<28x8x128xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb48(%448: f32, %449: f32, %450: f32):
      %451 = arith.mulf %448, %449 : f32
      %452 = arith.addf %450, %451 : f32
      linalg.yield %452 : f32
    } -> tensor<28x8x128xf32>
    %453 = tensor.collapse_shape %447 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28x8x128xf32> into tensor<28672xf32>
    %454 = tensor.expand_shape %453 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %455 = tensor.empty() : tensor<1x8x28x128xf32>
    %456 = linalg.transpose ins(%454:tensor<1x28x8x128xf32>) outs(%455:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
    %457 = tensor.collapse_shape %456 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x28x128xf32> into tensor<28672xf32>
    %458 = tensor.expand_shape %457 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %459 = tensor.empty() : tensor<3584x3584xf32>
    %460 = linalg.transpose ins(%6:tensor<3584x3584xf32>) outs(%459:tensor<3584x3584xf32>) permutation = [1, 0]
    %461 = tensor.collapse_shape %458 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.attn_out"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %462 = tensor.expand_shape %461 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.attn_out"} : tensor<28672xf32> into tensor<8x3584xf32>
    %463 = tensor.empty() : tensor<8x3584xf32>
    %464 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %465 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%464 : f32) outs(%463 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %466 = linalg.matmul {prov.region_id = "matmul_4", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.attn_out", prov.transposed_b = "true"} ins(%462, %460 : tensor<8x3584xf32>, tensor<3584x3584xf32>) outs(%465 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %467 = tensor.collapse_shape %466 [[0 : i64, 1 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.attn_out"} : tensor<8x3584xf32> into tensor<28672xf32>
    %468 = tensor.expand_shape %467 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.attn_out"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %469 = tensor.empty() : tensor<1x8x3584xf32>
    %470 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %468 : tensor<1x8x3584xf32>, tensor<1x8x3584xf32>) outs(%469 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb49(%471: f32, %472: f32, %473: f32):
      %474 = arith.addf %471, %472 : f32
      linalg.yield %474 : f32
    } -> tensor<1x8x3584xf32>
    %475 = tensor.empty() : tensor<1x8x3584xf32>
    %476 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%470 : tensor<1x8x3584xf32>) outs(%475 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb50(%477: f32, %478: f32):
      %479 = arith.constant 2.000000e+00 : f32
      %480 = math.powf %477, %479 : f32
      linalg.yield %480 : f32
    } -> tensor<1x8x3584xf32>
    %481 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %482 = tensor.splat %481 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %483 = linalg.reduce ins(%476:tensor<1x8x3584xf32>) outs(%482:tensor<1x8xf32>) dimensions = [2]
    (%484: f32, %485: f32) {
      %486 = arith.addf %484, %485 : f32
      linalg.yield %486 : f32
    }
    %487 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %488 = tensor.splat %487 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %489 = tensor.empty() : tensor<1x8xf32>
    %490 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%483, %488 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%489 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb51(%491: f32, %492: f32, %493: f32):
      %494 = arith.divf %491, %492 : f32
      linalg.yield %494 : f32
    } -> tensor<1x8xf32>
    %495 = tensor.collapse_shape %490 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %496 = tensor.expand_shape %495 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %497 = arith.constant {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %498 = tensor.splat %497 {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %499 = tensor.empty() : tensor<1x8x1xf32>
    %500 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%496, %498 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%499 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb52(%501: f32, %502: f32, %503: f32):
      %504 = arith.addf %501, %502 : f32
      linalg.yield %504 : f32
    } -> tensor<1x8x1xf32>
    %505 = tensor.empty() : tensor<1x8x1xf32>
    %506 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%500 : tensor<1x8x1xf32>) outs(%505 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb53(%507: f32, %508: f32):
      %509 = math.rsqrt %507 : f32
      linalg.yield %509 : f32
    } -> tensor<1x8x1xf32>
    %510 = tensor.empty() : tensor<1x8x3584xf32>
    %511 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%470, %506 : tensor<1x8x3584xf32>, tensor<1x8x1xf32>) outs(%510 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb54(%512: f32, %513: f32, %514: f32):
      %515 = arith.mulf %512, %513 : f32
      linalg.yield %515 : f32
    } -> tensor<1x8x3584xf32>
    %516 = tensor.empty() : tensor<1x8x3584xf32>
    %517 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %511 : tensor<3584xf32>, tensor<1x8x3584xf32>) outs(%516 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb55(%518: f32, %519: f32, %520: f32):
      %521 = arith.mulf %518, %519 : f32
      linalg.yield %521 : f32
    } -> tensor<1x8x3584xf32>
    %522 = tensor.empty() : tensor<3584x37888xf32>
    %523 = linalg.transpose ins(%7:tensor<37888x3584xf32>) outs(%522:tensor<3584x37888xf32>) permutation = [1, 0]
    %524 = tensor.collapse_shape %517 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_proj"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %525 = tensor.expand_shape %524 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_proj"} : tensor<28672xf32> into tensor<8x3584xf32>
    %526 = tensor.empty() : tensor<8x37888xf32>
    %527 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %528 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%527 : f32) outs(%526 : tensor<8x37888xf32>) -> tensor<8x37888xf32>
    %529 = linalg.matmul {prov.region_id = "matmul_5", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_proj", prov.transposed_b = "true"} ins(%525, %523 : tensor<8x3584xf32>, tensor<3584x37888xf32>) outs(%528 : tensor<8x37888xf32>) -> tensor<8x37888xf32>
    %530 = tensor.collapse_shape %529 [[0 : i64, 1 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_proj"} : tensor<8x37888xf32> into tensor<303104xf32>
    %531 = tensor.expand_shape %530 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 37888] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_proj"} : tensor<303104xf32> into tensor<1x8x37888xf32>
    %532 = "tensor.extract_slice"(%531) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x37888xf32>) -> tensor<1x8x18944xf32>
    %533 = "tensor.extract_slice"(%531) <{static_offsets = array<i64: 0, 0, 18944>, static_sizes = array<i64: 1, 8, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x37888xf32>) -> tensor<1x8x18944xf32>
    %534 = tensor.empty() : tensor<1x8x18944xf32>
    %535 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%533 : tensor<1x8x18944xf32>) outs(%534 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.act"} {
    ^bb56(%536: f32, %537: f32):
      %538 = arith.constant 1.000000e+00 : f32
      %539 = arith.negf %536 : f32
      %540 = math.exp %539 : f32
      %541 = arith.addf %538, %540 : f32
      %542 = arith.divf %538, %541 : f32
      linalg.yield %542 : f32
    } -> tensor<1x8x18944xf32>
    %543 = tensor.empty() : tensor<1x8x18944xf32>
    %544 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%533, %535 : tensor<1x8x18944xf32>, tensor<1x8x18944xf32>) outs(%543 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.act"} {
    ^bb57(%545: f32, %546: f32, %547: f32):
      %548 = arith.mulf %545, %546 : f32
      linalg.yield %548 : f32
    } -> tensor<1x8x18944xf32>
    %549 = tensor.empty() : tensor<1x8x18944xf32>
    %550 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%544, %532 : tensor<1x8x18944xf32>, tensor<1x8x18944xf32>) outs(%549 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb58(%551: f32, %552: f32, %553: f32):
      %554 = arith.mulf %551, %552 : f32
      linalg.yield %554 : f32
    } -> tensor<1x8x18944xf32>
    %555 = tensor.empty() : tensor<18944x3584xf32>
    %556 = linalg.transpose ins(%8:tensor<3584x18944xf32>) outs(%555:tensor<18944x3584xf32>) permutation = [1, 0]
    %557 = tensor.collapse_shape %550 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_out"} : tensor<1x8x18944xf32> into tensor<151552xf32>
    %558 = tensor.expand_shape %557 [[0 : i64, 1 : i64]] output_shape [8, 18944] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_out"} : tensor<151552xf32> into tensor<8x18944xf32>
    %559 = tensor.empty() : tensor<8x3584xf32>
    %560 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %561 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%560 : f32) outs(%559 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %562 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_out", prov.transposed_b = "true"} ins(%558, %556 : tensor<8x18944xf32>, tensor<18944x3584xf32>) outs(%561 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %563 = tensor.collapse_shape %562 [[0 : i64, 1 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_out"} : tensor<8x3584xf32> into tensor<28672xf32>
    %564 = tensor.expand_shape %563 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_out"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %565 = tensor.empty() : tensor<1x8x3584xf32>
    %566 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%470, %564 : tensor<1x8x3584xf32>, tensor<1x8x3584xf32>) outs(%565 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb59(%567: f32, %568: f32, %569: f32):
      %570 = arith.addf %567, %568 : f32
      linalg.yield %570 : f32
    } -> tensor<1x8x3584xf32>
    %571 = tensor.empty() : tensor<1x8x3584xf32>
    %572 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%566 : tensor<1x8x3584xf32>) outs(%571 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb60(%573: f32, %574: f32):
      %575 = arith.constant 2.000000e+00 : f32
      %576 = math.powf %573, %575 : f32
      linalg.yield %576 : f32
    } -> tensor<1x8x3584xf32>
    %577 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %578 = tensor.splat %577 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %579 = linalg.reduce ins(%572:tensor<1x8x3584xf32>) outs(%578:tensor<1x8xf32>) dimensions = [2]
    (%580: f32, %581: f32) {
      %582 = arith.addf %580, %581 : f32
      linalg.yield %582 : f32
    }
    %583 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %584 = tensor.splat %583 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %585 = tensor.empty() : tensor<1x8xf32>
    %586 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%579, %584 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%585 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb61(%587: f32, %588: f32, %589: f32):
      %590 = arith.divf %587, %588 : f32
      linalg.yield %590 : f32
    } -> tensor<1x8xf32>
    %591 = tensor.collapse_shape %586 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %592 = tensor.expand_shape %591 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %593 = arith.constant {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %594 = tensor.splat %593 {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %595 = tensor.empty() : tensor<1x8x1xf32>
    %596 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%592, %594 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%595 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb62(%597: f32, %598: f32, %599: f32):
      %600 = arith.addf %597, %598 : f32
      linalg.yield %600 : f32
    } -> tensor<1x8x1xf32>
    %601 = tensor.empty() : tensor<1x8x1xf32>
    %602 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%596 : tensor<1x8x1xf32>) outs(%601 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb63(%603: f32, %604: f32):
      %605 = math.rsqrt %603 : f32
      linalg.yield %605 : f32
    } -> tensor<1x8x1xf32>
    %606 = tensor.empty() : tensor<1x8x3584xf32>
    %607 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%566, %602 : tensor<1x8x3584xf32>, tensor<1x8x1xf32>) outs(%606 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb64(%608: f32, %609: f32, %610: f32):
      %611 = arith.mulf %608, %609 : f32
      linalg.yield %611 : f32
    } -> tensor<1x8x3584xf32>
    %612 = tensor.empty() : tensor<1x8x3584xf32>
    %613 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %607 : tensor<3584xf32>, tensor<1x8x3584xf32>) outs(%612 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb65(%614: f32, %615: f32, %616: f32):
      %617 = arith.mulf %614, %615 : f32
      linalg.yield %617 : f32
    } -> tensor<1x8x3584xf32>
    %618 = tensor.collapse_shape %613 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.att_proj"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %619 = tensor.expand_shape %618 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.att_proj"} : tensor<28672xf32> into tensor<8x3584xf32>
    %620 = tensor.empty() : tensor<3584x4608xf32>
    %621 = linalg.transpose ins(%11:tensor<4608x3584xf32>) outs(%620:tensor<3584x4608xf32>) permutation = [1, 0]
    %622 = tensor.empty() : tensor<8x4608xf32>
    %623 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %624 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%623 : f32) outs(%622 : tensor<8x4608xf32>) -> tensor<8x4608xf32>
    %625 = linalg.matmul {prov.region_id = "matmul_7", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.att_proj", prov.transposed_b = "true"} ins(%619, %621 : tensor<8x3584xf32>, tensor<3584x4608xf32>) outs(%624 : tensor<8x4608xf32>) -> tensor<8x4608xf32>
    %626 = tensor.empty() : tensor<8x4608xf32>
    %627 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%625, %12 : tensor<8x4608xf32>, tensor<4608xf32>) outs(%626 : tensor<8x4608xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.att_proj"} {
    ^bb66(%628: f32, %629: f32, %630: f32):
      %631 = arith.addf %628, %629 : f32
      linalg.yield %631 : f32
    } -> tensor<8x4608xf32>
    %632 = tensor.collapse_shape %627 [[0 : i64, 1 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.att_proj"} : tensor<8x4608xf32> into tensor<36864xf32>
    %633 = tensor.expand_shape %632 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 4608] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.att_proj"} : tensor<36864xf32> into tensor<1x8x4608xf32>
    %634 = "tensor.extract_slice"(%633) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x4608xf32>) -> tensor<1x8x3584xf32>
    %635 = "tensor.extract_slice"(%633) <{static_offsets = array<i64: 0, 0, 3584>, static_sizes = array<i64: 1, 8, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x4608xf32>) -> tensor<1x8x512xf32>
    %636 = "tensor.extract_slice"(%633) <{static_offsets = array<i64: 0, 0, 4096>, static_sizes = array<i64: 1, 8, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x4608xf32>) -> tensor<1x8x512xf32>
    %637 = tensor.collapse_shape %634 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %638 = tensor.expand_shape %637 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 128] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x8x28x128xf32>
    %639 = tensor.empty() : tensor<1x28x8x128xf32>
    %640 = linalg.transpose ins(%638:tensor<1x8x28x128xf32>) outs(%639:tensor<1x28x8x128xf32>) permutation = [0, 2, 1, 3]
    %641 = tensor.collapse_shape %635 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x512xf32> into tensor<4096xf32>
    %642 = tensor.expand_shape %641 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 128] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x8x4x128xf32>
    %643 = tensor.empty() : tensor<1x4x8x128xf32>
    %644 = linalg.transpose ins(%642:tensor<1x8x4x128xf32>) outs(%643:tensor<1x4x8x128xf32>) permutation = [0, 2, 1, 3]
    %645 = tensor.collapse_shape %636 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x512xf32> into tensor<4096xf32>
    %646 = tensor.expand_shape %645 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 128] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x8x4x128xf32>
    %647 = tensor.empty() : tensor<1x4x8x128xf32>
    %648 = linalg.transpose ins(%646:tensor<1x8x4x128xf32>) outs(%647:tensor<1x4x8x128xf32>) permutation = [0, 2, 1, 3]
    %649 = tensor.collapse_shape %122 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %650 = tensor.expand_shape %649 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 128] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x8x128xf32>
    %651 = tensor.collapse_shape %135 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %652 = tensor.expand_shape %651 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 128] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x8x128xf32>
    %653 = tensor.empty() : tensor<1x28x8x128xf32>
    %654 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%640, %650 : tensor<1x28x8x128xf32>, tensor<1x1x8x128xf32>) outs(%653 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb67(%655: f32, %656: f32, %657: f32):
      %658 = arith.mulf %655, %656 : f32
      linalg.yield %658 : f32
    } -> tensor<1x28x8x128xf32>
    %659 = "tensor.extract_slice"(%640) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 28, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_15", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x28x8x128xf32>) -> tensor<1x28x8x64xf32>
    %660 = "tensor.extract_slice"(%640) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 28, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_16", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x28x8x128xf32>) -> tensor<1x28x8x64xf32>
    %661 = tensor.empty() : tensor<1x28x8x64xf32>
    %662 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%660 : tensor<1x28x8x64xf32>) outs(%661 : tensor<1x28x8x64xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb68(%663: f32, %664: f32):
      %665 = arith.negf %663 : f32
      linalg.yield %665 : f32
    } -> tensor<1x28x8x64xf32>
    %666 = tensor.concat dim(3) %662, %659 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x28x8x64xf32>, tensor<1x28x8x64xf32>) -> tensor<1x28x8x128xf32>
    %667 = tensor.empty() : tensor<1x28x8x128xf32>
    %668 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%666, %652 : tensor<1x28x8x128xf32>, tensor<1x1x8x128xf32>) outs(%667 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb69(%669: f32, %670: f32, %671: f32):
      %672 = arith.mulf %669, %670 : f32
      linalg.yield %672 : f32
    } -> tensor<1x28x8x128xf32>
    %673 = tensor.empty() : tensor<1x28x8x128xf32>
    %674 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%654, %668 : tensor<1x28x8x128xf32>, tensor<1x28x8x128xf32>) outs(%673 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb70(%675: f32, %676: f32, %677: f32):
      %678 = arith.addf %675, %676 : f32
      linalg.yield %678 : f32
    } -> tensor<1x28x8x128xf32>
    %679 = tensor.empty() : tensor<1x4x8x128xf32>
    %680 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%644, %650 : tensor<1x4x8x128xf32>, tensor<1x1x8x128xf32>) outs(%679 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb71(%681: f32, %682: f32, %683: f32):
      %684 = arith.mulf %681, %682 : f32
      linalg.yield %684 : f32
    } -> tensor<1x4x8x128xf32>
    %685 = "tensor.extract_slice"(%644) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_17", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x64xf32>
    %686 = "tensor.extract_slice"(%644) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_18", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x64xf32>
    %687 = tensor.empty() : tensor<1x4x8x64xf32>
    %688 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%686 : tensor<1x4x8x64xf32>) outs(%687 : tensor<1x4x8x64xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb72(%689: f32, %690: f32):
      %691 = arith.negf %689 : f32
      linalg.yield %691 : f32
    } -> tensor<1x4x8x64xf32>
    %692 = tensor.concat dim(3) %688, %685 {prov.region_id = "cat_5", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x8x64xf32>, tensor<1x4x8x64xf32>) -> tensor<1x4x8x128xf32>
    %693 = tensor.empty() : tensor<1x4x8x128xf32>
    %694 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%692, %652 : tensor<1x4x8x128xf32>, tensor<1x1x8x128xf32>) outs(%693 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb73(%695: f32, %696: f32, %697: f32):
      %698 = arith.mulf %695, %696 : f32
      linalg.yield %698 : f32
    } -> tensor<1x4x8x128xf32>
    %699 = tensor.empty() : tensor<1x4x8x128xf32>
    %700 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%680, %694 : tensor<1x4x8x128xf32>, tensor<1x4x8x128xf32>) outs(%699 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb74(%701: f32, %702: f32, %703: f32):
      %704 = arith.addf %701, %702 : f32
      linalg.yield %704 : f32
    } -> tensor<1x4x8x128xf32>
    %705 = tensor.empty() : tensor<8xi64>
    %706 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%705 : tensor<8xi64>) attrs =  {prov.region_id = "iota_4", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb75(%707: i64):
      %708 = linalg.index 0 : index
      %709 = arith.index_cast %708 : index to i64
      %710 = arith.constant 1 : i64
      %711 = arith.muli %709, %710 : i64
      %712 = arith.constant 0 : i64
      %713 = arith.addi %712, %711 : i64
      linalg.yield %713 : i64
    } -> tensor<8xi64>
    %714 = tensor.empty() : tensor<8xi64>
    %715 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%67, %706 : tensor<i64>, tensor<8xi64>) outs(%714 : tensor<8xi64>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb76(%716: i64, %717: i64, %718: i64):
      %719 = arith.addi %716, %717 : i64
      linalg.yield %719 : i64
    } -> tensor<8xi64>
    %720 = "tensor.extract_slice"(%36) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<4x1x4x16x128xf32>) -> tensor<4x16x128xf32>
    %721 = func.call @aten_index_put_default(%720, %715, %700) {prov.region_id = "aten_index_put_default_2", prov.dispatch_id = "aten_index_put_default_2"} : (tensor<4x16x128xf32>, tensor<8xi64>, tensor<1x4x8x128xf32>) -> tensor<1x4x16x128xf32>
    %722 = "tensor.extract_slice"(%38) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<4x1x4x16x128xf32>) -> tensor<4x16x128xf32>
    %723 = func.call @aten_index_put_default(%722, %715, %648) {prov.region_id = "aten_index_put_default_3", prov.dispatch_id = "aten_index_put_default_3"} : (tensor<4x16x128xf32>, tensor<8xi64>, tensor<1x4x8x128xf32>) -> tensor<1x4x16x128xf32>
    %724 = "tensor.extract_slice"(%721) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_19", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
    %725 = "tensor.extract_slice"(%724) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_20", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
    %726 = tensor.collapse_shape %725 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x16x128xf32> into tensor<8192xf32>
    %727 = tensor.expand_shape %726 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 16, 128] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x4x1x16x128xf32>
    %728 = "tensor.extract_slice"(%727) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_21", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
    %729 = "tensor.extract_slice"(%728) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_22", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
    %730 = tensor.empty() : tensor<1x4x7x16x128xf32>
    %731 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%729 : tensor<1x4x1x16x128xf32>) outs(%730 : tensor<1x4x7x16x128xf32>) attrs =  {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb77(%732: f32, %733: f32):
      linalg.yield %732 : f32
    } -> tensor<1x4x7x16x128xf32>
    %734 = tensor.collapse_shape %731 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x7x16x128xf32> into tensor<57344xf32>
    %735 = tensor.expand_shape %734 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 16, 128] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<1x28x16x128xf32>
    %736 = "tensor.extract_slice"(%723) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_23", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
    %737 = "tensor.extract_slice"(%736) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_24", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
    %738 = tensor.collapse_shape %737 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x16x128xf32> into tensor<8192xf32>
    %739 = tensor.expand_shape %738 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 16, 128] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x4x1x16x128xf32>
    %740 = "tensor.extract_slice"(%739) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_25", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
    %741 = "tensor.extract_slice"(%740) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_26", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
    %742 = tensor.empty() : tensor<1x4x7x16x128xf32>
    %743 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%741 : tensor<1x4x1x16x128xf32>) outs(%742 : tensor<1x4x7x16x128xf32>) attrs =  {prov.region_id = "expand_10", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb78(%744: f32, %745: f32):
      linalg.yield %744 : f32
    } -> tensor<1x4x7x16x128xf32>
    %746 = tensor.collapse_shape %743 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x7x16x128xf32> into tensor<57344xf32>
    %747 = tensor.expand_shape %746 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 16, 128] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<1x28x16x128xf32>
    %748 = tensor.empty() : tensor<1x28x128x16xf32>
    %749 = linalg.transpose ins(%735:tensor<1x28x16x128xf32>) outs(%748:tensor<1x28x128x16xf32>) permutation = [0, 1, 3, 2]
    %750 = tensor.empty() : tensor<1x28x8x128xf32>
    %751 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%674 : tensor<1x28x8x128xf32>) outs(%750 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "expand_11", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb79(%752: f32, %753: f32):
      linalg.yield %752 : f32
    } -> tensor<1x28x8x128xf32>
    %754 = tensor.collapse_shape %751 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
    %755 = tensor.expand_shape %754 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 128] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<28x8x128xf32>
    %756 = tensor.empty() : tensor<1x28x128x16xf32>
    %757 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%749 : tensor<1x28x128x16xf32>) outs(%756 : tensor<1x28x128x16xf32>) attrs =  {prov.region_id = "expand_12", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb80(%758: f32, %759: f32):
      linalg.yield %758 : f32
    } -> tensor<1x28x128x16xf32>
    %760 = tensor.collapse_shape %757 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x128x16xf32> into tensor<57344xf32>
    %761 = tensor.expand_shape %760 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 128, 16] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<28x128x16xf32>
    %762 = arith.constant {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %763 = tensor.splat %762 {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<28x8x16xf32>
    %764 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%755, %761 : tensor<28x8x128xf32>, tensor<28x128x16xf32>) outs(%763 : tensor<28x8x16xf32>) attrs =  {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb81(%765: f32, %766: f32, %767: f32):
      %768 = arith.mulf %765, %766 : f32
      %769 = arith.addf %767, %768 : f32
      linalg.yield %769 : f32
    } -> tensor<28x8x16xf32>
    %770 = tensor.collapse_shape %764 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28x8x16xf32> into tensor<3584xf32>
    %771 = tensor.expand_shape %770 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 16] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<1x28x8x16xf32>
    %772 = arith.constant {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.0883883461 : f32
    %773 = tensor.splat %772 {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x8x16xf32>
    %774 = tensor.empty() : tensor<1x28x8x16xf32>
    %775 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%771, %773 : tensor<1x28x8x16xf32>, tensor<1x28x8x16xf32>) outs(%774 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb82(%776: f32, %777: f32, %778: f32):
      %779 = arith.mulf %776, %777 : f32
      linalg.yield %779 : f32
    } -> tensor<1x28x8x16xf32>
    %780 = tensor.empty() : tensor<1x1x8x16xi1>
    %781 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%175 : tensor<1x1x8x16xi1>) outs(%780 : tensor<1x1x8x16xi1>) attrs =  {prov.region_id = "bitwise_1", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
    ^bb83(%782: i1, %783: i1):
      %784 = arith.constant true
      %785 = arith.xori %782, %784 : i1
      linalg.yield %785 : i1
    } -> tensor<1x1x8x16xi1>
    %786 = arith.constant {prov.region_id = "fill_4", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %787 = tensor.splat %786 {prov.region_id = "fill_4", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %788 = tensor.empty() : tensor<1x28x8x16xf32>
    %789 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%781, %787, %775 : tensor<1x1x8x16xi1>, tensor<f32>, tensor<1x28x8x16xf32>) outs(%788 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32"} {
    ^bb84(%790: i1, %791: f32, %792: f32, %793: f32):
      %794 = arith.select %790, %791, %792 : f32
      linalg.yield %794 : f32
    } -> tensor<1x28x8x16xf32>
    %795 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %796 = tensor.splat %795 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x8xf32>
    %797 = linalg.reduce ins(%789:tensor<1x28x8x16xf32>) outs(%796:tensor<1x28x8xf32>) dimensions = [3]
    (%798: f32, %799: f32) {
      %800 = arith.maximumf %798, %799 : f32
      linalg.yield %800 : f32
    }
    %801 = tensor.collapse_shape %797 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x8xf32> into tensor<224xf32>
    %802 = tensor.expand_shape %801 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %803 = tensor.empty() : tensor<1x28x8x16xf32>
    %804 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%789, %802 : tensor<1x28x8x16xf32>, tensor<1x28x8x1xf32>) outs(%803 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb85(%805: f32, %806: f32, %807: f32):
      %808 = arith.subf %805, %806 : f32
      linalg.yield %808 : f32
    } -> tensor<1x28x8x16xf32>
    %809 = tensor.empty() : tensor<1x28x8x16xf32>
    %810 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%804 : tensor<1x28x8x16xf32>) outs(%809 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb86(%811: f32, %812: f32):
      %813 = math.exp %811 : f32
      linalg.yield %813 : f32
    } -> tensor<1x28x8x16xf32>
    %814 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %815 = tensor.splat %814 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x8xf32>
    %816 = linalg.reduce ins(%810:tensor<1x28x8x16xf32>) outs(%815:tensor<1x28x8xf32>) dimensions = [3]
    (%817: f32, %818: f32) {
      %819 = arith.addf %817, %818 : f32
      linalg.yield %819 : f32
    }
    %820 = tensor.collapse_shape %816 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x8xf32> into tensor<224xf32>
    %821 = tensor.expand_shape %820 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %822 = tensor.empty() : tensor<1x28x8x16xf32>
    %823 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%810, %821 : tensor<1x28x8x16xf32>, tensor<1x28x8x1xf32>) outs(%822 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb87(%824: f32, %825: f32, %826: f32):
      %827 = arith.divf %824, %825 : f32
      linalg.yield %827 : f32
    } -> tensor<1x28x8x16xf32>
    %828 = tensor.empty() : tensor<1x28x8x16xf32>
    %829 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%823 : tensor<1x28x8x16xf32>) outs(%828 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "expand_13", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb88(%830: f32, %831: f32):
      linalg.yield %830 : f32
    } -> tensor<1x28x8x16xf32>
    %832 = tensor.collapse_shape %829 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x8x16xf32> into tensor<3584xf32>
    %833 = tensor.expand_shape %832 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 16] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<28x8x16xf32>
    %834 = tensor.empty() : tensor<1x28x16x128xf32>
    %835 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%747 : tensor<1x28x16x128xf32>) outs(%834 : tensor<1x28x16x128xf32>) attrs =  {prov.region_id = "expand_14", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb89(%836: f32, %837: f32):
      linalg.yield %836 : f32
    } -> tensor<1x28x16x128xf32>
    %838 = tensor.collapse_shape %835 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x16x128xf32> into tensor<57344xf32>
    %839 = tensor.expand_shape %838 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 16, 128] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<28x16x128xf32>
    %840 = arith.constant {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %841 = tensor.splat %840 {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<28x8x128xf32>
    %842 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%833, %839 : tensor<28x8x16xf32>, tensor<28x16x128xf32>) outs(%841 : tensor<28x8x128xf32>) attrs =  {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb90(%843: f32, %844: f32, %845: f32):
      %846 = arith.mulf %843, %844 : f32
      %847 = arith.addf %845, %846 : f32
      linalg.yield %847 : f32
    } -> tensor<28x8x128xf32>
    %848 = tensor.collapse_shape %842 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28x8x128xf32> into tensor<28672xf32>
    %849 = tensor.expand_shape %848 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %850 = tensor.empty() : tensor<1x8x28x128xf32>
    %851 = linalg.transpose ins(%849:tensor<1x28x8x128xf32>) outs(%850:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
    %852 = tensor.collapse_shape %851 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x28x128xf32> into tensor<28672xf32>
    %853 = tensor.expand_shape %852 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %854 = tensor.empty() : tensor<3584x3584xf32>
    %855 = linalg.transpose ins(%13:tensor<3584x3584xf32>) outs(%854:tensor<3584x3584xf32>) permutation = [1, 0]
    %856 = tensor.collapse_shape %853 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.attn_out"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %857 = tensor.expand_shape %856 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.attn_out"} : tensor<28672xf32> into tensor<8x3584xf32>
    %858 = tensor.empty() : tensor<8x3584xf32>
    %859 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %860 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%859 : f32) outs(%858 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %861 = linalg.matmul {prov.region_id = "matmul_10", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.attn_out", prov.transposed_b = "true"} ins(%857, %855 : tensor<8x3584xf32>, tensor<3584x3584xf32>) outs(%860 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %862 = tensor.collapse_shape %861 [[0 : i64, 1 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.attn_out"} : tensor<8x3584xf32> into tensor<28672xf32>
    %863 = tensor.expand_shape %862 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.attn_out"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %864 = tensor.empty() : tensor<1x8x3584xf32>
    %865 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%566, %863 : tensor<1x8x3584xf32>, tensor<1x8x3584xf32>) outs(%864 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb91(%866: f32, %867: f32, %868: f32):
      %869 = arith.addf %866, %867 : f32
      linalg.yield %869 : f32
    } -> tensor<1x8x3584xf32>
    %870 = tensor.empty() : tensor<1x8x3584xf32>
    %871 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%865 : tensor<1x8x3584xf32>) outs(%870 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb92(%872: f32, %873: f32):
      %874 = arith.constant 2.000000e+00 : f32
      %875 = math.powf %872, %874 : f32
      linalg.yield %875 : f32
    } -> tensor<1x8x3584xf32>
    %876 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %877 = tensor.splat %876 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %878 = linalg.reduce ins(%871:tensor<1x8x3584xf32>) outs(%877:tensor<1x8xf32>) dimensions = [2]
    (%879: f32, %880: f32) {
      %881 = arith.addf %879, %880 : f32
      linalg.yield %881 : f32
    }
    %882 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %883 = tensor.splat %882 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %884 = tensor.empty() : tensor<1x8xf32>
    %885 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%878, %883 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%884 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb93(%886: f32, %887: f32, %888: f32):
      %889 = arith.divf %886, %887 : f32
      linalg.yield %889 : f32
    } -> tensor<1x8xf32>
    %890 = tensor.collapse_shape %885 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %891 = tensor.expand_shape %890 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %892 = arith.constant {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %893 = tensor.splat %892 {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %894 = tensor.empty() : tensor<1x8x1xf32>
    %895 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%891, %893 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%894 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb94(%896: f32, %897: f32, %898: f32):
      %899 = arith.addf %896, %897 : f32
      linalg.yield %899 : f32
    } -> tensor<1x8x1xf32>
    %900 = tensor.empty() : tensor<1x8x1xf32>
    %901 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%895 : tensor<1x8x1xf32>) outs(%900 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb95(%902: f32, %903: f32):
      %904 = math.rsqrt %902 : f32
      linalg.yield %904 : f32
    } -> tensor<1x8x1xf32>
    %905 = tensor.empty() : tensor<1x8x3584xf32>
    %906 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%865, %901 : tensor<1x8x3584xf32>, tensor<1x8x1xf32>) outs(%905 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb96(%907: f32, %908: f32, %909: f32):
      %910 = arith.mulf %907, %908 : f32
      linalg.yield %910 : f32
    } -> tensor<1x8x3584xf32>
    %911 = tensor.empty() : tensor<1x8x3584xf32>
    %912 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %906 : tensor<3584xf32>, tensor<1x8x3584xf32>) outs(%911 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb97(%913: f32, %914: f32, %915: f32):
      %916 = arith.mulf %913, %914 : f32
      linalg.yield %916 : f32
    } -> tensor<1x8x3584xf32>
    %917 = tensor.empty() : tensor<3584x37888xf32>
    %918 = linalg.transpose ins(%14:tensor<37888x3584xf32>) outs(%917:tensor<3584x37888xf32>) permutation = [1, 0]
    %919 = tensor.collapse_shape %912 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_proj"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %920 = tensor.expand_shape %919 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_proj"} : tensor<28672xf32> into tensor<8x3584xf32>
    %921 = tensor.empty() : tensor<8x37888xf32>
    %922 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %923 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%922 : f32) outs(%921 : tensor<8x37888xf32>) -> tensor<8x37888xf32>
    %924 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_proj", prov.transposed_b = "true"} ins(%920, %918 : tensor<8x3584xf32>, tensor<3584x37888xf32>) outs(%923 : tensor<8x37888xf32>) -> tensor<8x37888xf32>
    %925 = tensor.collapse_shape %924 [[0 : i64, 1 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_proj"} : tensor<8x37888xf32> into tensor<303104xf32>
    %926 = tensor.expand_shape %925 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 37888] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_proj"} : tensor<303104xf32> into tensor<1x8x37888xf32>
    %927 = "tensor.extract_slice"(%926) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x37888xf32>) -> tensor<1x8x18944xf32>
    %928 = "tensor.extract_slice"(%926) <{static_offsets = array<i64: 0, 0, 18944>, static_sizes = array<i64: 1, 8, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x37888xf32>) -> tensor<1x8x18944xf32>
    %929 = tensor.empty() : tensor<1x8x18944xf32>
    %930 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%928 : tensor<1x8x18944xf32>) outs(%929 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.act"} {
    ^bb98(%931: f32, %932: f32):
      %933 = arith.constant 1.000000e+00 : f32
      %934 = arith.negf %931 : f32
      %935 = math.exp %934 : f32
      %936 = arith.addf %933, %935 : f32
      %937 = arith.divf %933, %936 : f32
      linalg.yield %937 : f32
    } -> tensor<1x8x18944xf32>
    %938 = tensor.empty() : tensor<1x8x18944xf32>
    %939 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%928, %930 : tensor<1x8x18944xf32>, tensor<1x8x18944xf32>) outs(%938 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.act"} {
    ^bb99(%940: f32, %941: f32, %942: f32):
      %943 = arith.mulf %940, %941 : f32
      linalg.yield %943 : f32
    } -> tensor<1x8x18944xf32>
    %944 = tensor.empty() : tensor<1x8x18944xf32>
    %945 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%939, %927 : tensor<1x8x18944xf32>, tensor<1x8x18944xf32>) outs(%944 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb100(%946: f32, %947: f32, %948: f32):
      %949 = arith.mulf %946, %947 : f32
      linalg.yield %949 : f32
    } -> tensor<1x8x18944xf32>
    %950 = tensor.empty() : tensor<18944x3584xf32>
    %951 = linalg.transpose ins(%15:tensor<3584x18944xf32>) outs(%950:tensor<18944x3584xf32>) permutation = [1, 0]
    %952 = tensor.collapse_shape %945 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_out"} : tensor<1x8x18944xf32> into tensor<151552xf32>
    %953 = tensor.expand_shape %952 [[0 : i64, 1 : i64]] output_shape [8, 18944] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_out"} : tensor<151552xf32> into tensor<8x18944xf32>
    %954 = tensor.empty() : tensor<8x3584xf32>
    %955 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %956 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%955 : f32) outs(%954 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %957 = linalg.matmul {prov.region_id = "matmul_12", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_out", prov.transposed_b = "true"} ins(%953, %951 : tensor<8x18944xf32>, tensor<18944x3584xf32>) outs(%956 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %958 = tensor.collapse_shape %957 [[0 : i64, 1 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_out"} : tensor<8x3584xf32> into tensor<28672xf32>
    %959 = tensor.expand_shape %958 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_out"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %960 = tensor.empty() : tensor<1x8x3584xf32>
    %961 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%865, %959 : tensor<1x8x3584xf32>, tensor<1x8x3584xf32>) outs(%960 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb101(%962: f32, %963: f32, %964: f32):
      %965 = arith.addf %962, %963 : f32
      linalg.yield %965 : f32
    } -> tensor<1x8x3584xf32>
    %966 = tensor.empty() : tensor<1x8x3584xf32>
    %967 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%961 : tensor<1x8x3584xf32>) outs(%966 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb102(%968: f32, %969: f32):
      %970 = arith.constant 2.000000e+00 : f32
      %971 = math.powf %968, %970 : f32
      linalg.yield %971 : f32
    } -> tensor<1x8x3584xf32>
    %972 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %973 = tensor.splat %972 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %974 = linalg.reduce ins(%967:tensor<1x8x3584xf32>) outs(%973:tensor<1x8xf32>) dimensions = [2]
    (%975: f32, %976: f32) {
      %977 = arith.addf %975, %976 : f32
      linalg.yield %977 : f32
    }
    %978 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %979 = tensor.splat %978 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %980 = tensor.empty() : tensor<1x8xf32>
    %981 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%974, %979 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%980 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb103(%982: f32, %983: f32, %984: f32):
      %985 = arith.divf %982, %983 : f32
      linalg.yield %985 : f32
    } -> tensor<1x8xf32>
    %986 = tensor.collapse_shape %981 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %987 = tensor.expand_shape %986 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %988 = arith.constant {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %989 = tensor.splat %988 {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %990 = tensor.empty() : tensor<1x8x1xf32>
    %991 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%987, %989 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%990 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb104(%992: f32, %993: f32, %994: f32):
      %995 = arith.addf %992, %993 : f32
      linalg.yield %995 : f32
    } -> tensor<1x8x1xf32>
    %996 = tensor.empty() : tensor<1x8x1xf32>
    %997 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%991 : tensor<1x8x1xf32>) outs(%996 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb105(%998: f32, %999: f32):
      %1000 = math.rsqrt %998 : f32
      linalg.yield %1000 : f32
    } -> tensor<1x8x1xf32>
    %1001 = tensor.empty() : tensor<1x8x3584xf32>
    %1002 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%961, %997 : tensor<1x8x3584xf32>, tensor<1x8x1xf32>) outs(%1001 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb106(%1003: f32, %1004: f32, %1005: f32):
      %1006 = arith.mulf %1003, %1004 : f32
      linalg.yield %1006 : f32
    } -> tensor<1x8x3584xf32>
    %1007 = tensor.empty() : tensor<1x8x3584xf32>
    %1008 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16, %1002 : tensor<3584xf32>, tensor<1x8x3584xf32>) outs(%1007 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb107(%1009: f32, %1010: f32, %1011: f32):
      %1012 = arith.mulf %1009, %1010 : f32
      linalg.yield %1012 : f32
    } -> tensor<1x8x3584xf32>
    %1013 = tensor.collapse_shape %1008 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.att_proj"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %1014 = tensor.expand_shape %1013 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.att_proj"} : tensor<28672xf32> into tensor<8x3584xf32>
    %1015 = tensor.empty() : tensor<3584x4608xf32>
    %1016 = linalg.transpose ins(%18:tensor<4608x3584xf32>) outs(%1015:tensor<3584x4608xf32>) permutation = [1, 0]
    %1017 = tensor.empty() : tensor<8x4608xf32>
    %1018 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1019 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1018 : f32) outs(%1017 : tensor<8x4608xf32>) -> tensor<8x4608xf32>
    %1020 = linalg.matmul {prov.region_id = "matmul_13", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.att_proj", prov.transposed_b = "true"} ins(%1014, %1016 : tensor<8x3584xf32>, tensor<3584x4608xf32>) outs(%1019 : tensor<8x4608xf32>) -> tensor<8x4608xf32>
    %1021 = tensor.empty() : tensor<8x4608xf32>
    %1022 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1020, %19 : tensor<8x4608xf32>, tensor<4608xf32>) outs(%1021 : tensor<8x4608xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.att_proj"} {
    ^bb108(%1023: f32, %1024: f32, %1025: f32):
      %1026 = arith.addf %1023, %1024 : f32
      linalg.yield %1026 : f32
    } -> tensor<8x4608xf32>
    %1027 = tensor.collapse_shape %1022 [[0 : i64, 1 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.att_proj"} : tensor<8x4608xf32> into tensor<36864xf32>
    %1028 = tensor.expand_shape %1027 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 4608] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.att_proj"} : tensor<36864xf32> into tensor<1x8x4608xf32>
    %1029 = "tensor.extract_slice"(%1028) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_4", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x4608xf32>) -> tensor<1x8x3584xf32>
    %1030 = "tensor.extract_slice"(%1028) <{static_offsets = array<i64: 0, 0, 3584>, static_sizes = array<i64: 1, 8, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_4", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x4608xf32>) -> tensor<1x8x512xf32>
    %1031 = "tensor.extract_slice"(%1028) <{static_offsets = array<i64: 0, 0, 4096>, static_sizes = array<i64: 1, 8, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_4", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x4608xf32>) -> tensor<1x8x512xf32>
    %1032 = tensor.collapse_shape %1029 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %1033 = tensor.expand_shape %1032 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 128] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x8x28x128xf32>
    %1034 = tensor.empty() : tensor<1x28x8x128xf32>
    %1035 = linalg.transpose ins(%1033:tensor<1x8x28x128xf32>) outs(%1034:tensor<1x28x8x128xf32>) permutation = [0, 2, 1, 3]
    %1036 = tensor.collapse_shape %1030 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x512xf32> into tensor<4096xf32>
    %1037 = tensor.expand_shape %1036 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 128] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x8x4x128xf32>
    %1038 = tensor.empty() : tensor<1x4x8x128xf32>
    %1039 = linalg.transpose ins(%1037:tensor<1x8x4x128xf32>) outs(%1038:tensor<1x4x8x128xf32>) permutation = [0, 2, 1, 3]
    %1040 = tensor.collapse_shape %1031 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x512xf32> into tensor<4096xf32>
    %1041 = tensor.expand_shape %1040 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 128] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x8x4x128xf32>
    %1042 = tensor.empty() : tensor<1x4x8x128xf32>
    %1043 = linalg.transpose ins(%1041:tensor<1x8x4x128xf32>) outs(%1042:tensor<1x4x8x128xf32>) permutation = [0, 2, 1, 3]
    %1044 = tensor.collapse_shape %122 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1045 = tensor.expand_shape %1044 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 128] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x8x128xf32>
    %1046 = tensor.collapse_shape %135 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1047 = tensor.expand_shape %1046 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 128] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x8x128xf32>
    %1048 = tensor.empty() : tensor<1x28x8x128xf32>
    %1049 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1035, %1045 : tensor<1x28x8x128xf32>, tensor<1x1x8x128xf32>) outs(%1048 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb109(%1050: f32, %1051: f32, %1052: f32):
      %1053 = arith.mulf %1050, %1051 : f32
      linalg.yield %1053 : f32
    } -> tensor<1x28x8x128xf32>
    %1054 = "tensor.extract_slice"(%1035) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 28, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_27", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x28x8x128xf32>) -> tensor<1x28x8x64xf32>
    %1055 = "tensor.extract_slice"(%1035) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 28, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_28", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x28x8x128xf32>) -> tensor<1x28x8x64xf32>
    %1056 = tensor.empty() : tensor<1x28x8x64xf32>
    %1057 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1055 : tensor<1x28x8x64xf32>) outs(%1056 : tensor<1x28x8x64xf32>) attrs =  {prov.region_id = "neg_4", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb110(%1058: f32, %1059: f32):
      %1060 = arith.negf %1058 : f32
      linalg.yield %1060 : f32
    } -> tensor<1x28x8x64xf32>
    %1061 = tensor.concat dim(3) %1057, %1054 {prov.region_id = "cat_6", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x28x8x64xf32>, tensor<1x28x8x64xf32>) -> tensor<1x28x8x128xf32>
    %1062 = tensor.empty() : tensor<1x28x8x128xf32>
    %1063 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1061, %1047 : tensor<1x28x8x128xf32>, tensor<1x1x8x128xf32>) outs(%1062 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb111(%1064: f32, %1065: f32, %1066: f32):
      %1067 = arith.mulf %1064, %1065 : f32
      linalg.yield %1067 : f32
    } -> tensor<1x28x8x128xf32>
    %1068 = tensor.empty() : tensor<1x28x8x128xf32>
    %1069 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1049, %1063 : tensor<1x28x8x128xf32>, tensor<1x28x8x128xf32>) outs(%1068 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb112(%1070: f32, %1071: f32, %1072: f32):
      %1073 = arith.addf %1070, %1071 : f32
      linalg.yield %1073 : f32
    } -> tensor<1x28x8x128xf32>
    %1074 = tensor.empty() : tensor<1x4x8x128xf32>
    %1075 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1039, %1045 : tensor<1x4x8x128xf32>, tensor<1x1x8x128xf32>) outs(%1074 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb113(%1076: f32, %1077: f32, %1078: f32):
      %1079 = arith.mulf %1076, %1077 : f32
      linalg.yield %1079 : f32
    } -> tensor<1x4x8x128xf32>
    %1080 = "tensor.extract_slice"(%1039) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_29", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x64xf32>
    %1081 = "tensor.extract_slice"(%1039) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_30", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x64xf32>
    %1082 = tensor.empty() : tensor<1x4x8x64xf32>
    %1083 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1081 : tensor<1x4x8x64xf32>) outs(%1082 : tensor<1x4x8x64xf32>) attrs =  {prov.region_id = "neg_5", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb114(%1084: f32, %1085: f32):
      %1086 = arith.negf %1084 : f32
      linalg.yield %1086 : f32
    } -> tensor<1x4x8x64xf32>
    %1087 = tensor.concat dim(3) %1083, %1080 {prov.region_id = "cat_7", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x8x64xf32>, tensor<1x4x8x64xf32>) -> tensor<1x4x8x128xf32>
    %1088 = tensor.empty() : tensor<1x4x8x128xf32>
    %1089 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1087, %1047 : tensor<1x4x8x128xf32>, tensor<1x1x8x128xf32>) outs(%1088 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb115(%1090: f32, %1091: f32, %1092: f32):
      %1093 = arith.mulf %1090, %1091 : f32
      linalg.yield %1093 : f32
    } -> tensor<1x4x8x128xf32>
    %1094 = tensor.empty() : tensor<1x4x8x128xf32>
    %1095 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1075, %1089 : tensor<1x4x8x128xf32>, tensor<1x4x8x128xf32>) outs(%1094 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb116(%1096: f32, %1097: f32, %1098: f32):
      %1099 = arith.addf %1096, %1097 : f32
      linalg.yield %1099 : f32
    } -> tensor<1x4x8x128xf32>
    %1100 = tensor.empty() : tensor<8xi64>
    %1101 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1100 : tensor<8xi64>) attrs =  {prov.region_id = "iota_5", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb117(%1102: i64):
      %1103 = linalg.index 0 : index
      %1104 = arith.index_cast %1103 : index to i64
      %1105 = arith.constant 1 : i64
      %1106 = arith.muli %1104, %1105 : i64
      %1107 = arith.constant 0 : i64
      %1108 = arith.addi %1107, %1106 : i64
      linalg.yield %1108 : i64
    } -> tensor<8xi64>
    %1109 = tensor.empty() : tensor<8xi64>
    %1110 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%67, %1101 : tensor<i64>, tensor<8xi64>) outs(%1109 : tensor<8xi64>) attrs =  {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb118(%1111: i64, %1112: i64, %1113: i64):
      %1114 = arith.addi %1111, %1112 : i64
      linalg.yield %1114 : i64
    } -> tensor<8xi64>
    %1115 = "tensor.extract_slice"(%36) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<4x1x4x16x128xf32>) -> tensor<4x16x128xf32>
    %1116 = func.call @aten_index_put_default(%1115, %1110, %1095) {prov.region_id = "aten_index_put_default_4", prov.dispatch_id = "aten_index_put_default_4"} : (tensor<4x16x128xf32>, tensor<8xi64>, tensor<1x4x8x128xf32>) -> tensor<1x4x16x128xf32>
    %1117 = "tensor.extract_slice"(%38) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<4x1x4x16x128xf32>) -> tensor<4x16x128xf32>
    %1118 = func.call @aten_index_put_default(%1117, %1110, %1043) {prov.region_id = "aten_index_put_default_5", prov.dispatch_id = "aten_index_put_default_5"} : (tensor<4x16x128xf32>, tensor<8xi64>, tensor<1x4x8x128xf32>) -> tensor<1x4x16x128xf32>
    %1119 = "tensor.extract_slice"(%1116) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_31", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
    %1120 = "tensor.extract_slice"(%1119) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_32", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
    %1121 = tensor.collapse_shape %1120 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x16x128xf32> into tensor<8192xf32>
    %1122 = tensor.expand_shape %1121 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 16, 128] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x4x1x16x128xf32>
    %1123 = "tensor.extract_slice"(%1122) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_33", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
    %1124 = "tensor.extract_slice"(%1123) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_34", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
    %1125 = tensor.empty() : tensor<1x4x7x16x128xf32>
    %1126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1124 : tensor<1x4x1x16x128xf32>) outs(%1125 : tensor<1x4x7x16x128xf32>) attrs =  {prov.region_id = "expand_15", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb119(%1127: f32, %1128: f32):
      linalg.yield %1127 : f32
    } -> tensor<1x4x7x16x128xf32>
    %1129 = tensor.collapse_shape %1126 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x7x16x128xf32> into tensor<57344xf32>
    %1130 = tensor.expand_shape %1129 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 16, 128] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<1x28x16x128xf32>
    %1131 = "tensor.extract_slice"(%1118) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_35", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
    %1132 = "tensor.extract_slice"(%1131) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_36", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
    %1133 = tensor.collapse_shape %1132 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x16x128xf32> into tensor<8192xf32>
    %1134 = tensor.expand_shape %1133 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 16, 128] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x4x1x16x128xf32>
    %1135 = "tensor.extract_slice"(%1134) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_37", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
    %1136 = "tensor.extract_slice"(%1135) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_38", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
    %1137 = tensor.empty() : tensor<1x4x7x16x128xf32>
    %1138 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1136 : tensor<1x4x1x16x128xf32>) outs(%1137 : tensor<1x4x7x16x128xf32>) attrs =  {prov.region_id = "expand_16", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb120(%1139: f32, %1140: f32):
      linalg.yield %1139 : f32
    } -> tensor<1x4x7x16x128xf32>
    %1141 = tensor.collapse_shape %1138 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x7x16x128xf32> into tensor<57344xf32>
    %1142 = tensor.expand_shape %1141 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 16, 128] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<1x28x16x128xf32>
    %1143 = tensor.empty() : tensor<1x28x128x16xf32>
    %1144 = linalg.transpose ins(%1130:tensor<1x28x16x128xf32>) outs(%1143:tensor<1x28x128x16xf32>) permutation = [0, 1, 3, 2]
    %1145 = tensor.empty() : tensor<1x28x8x128xf32>
    %1146 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1069 : tensor<1x28x8x128xf32>) outs(%1145 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "expand_17", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb121(%1147: f32, %1148: f32):
      linalg.yield %1147 : f32
    } -> tensor<1x28x8x128xf32>
    %1149 = tensor.collapse_shape %1146 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
    %1150 = tensor.expand_shape %1149 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 128] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<28x8x128xf32>
    %1151 = tensor.empty() : tensor<1x28x128x16xf32>
    %1152 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1144 : tensor<1x28x128x16xf32>) outs(%1151 : tensor<1x28x128x16xf32>) attrs =  {prov.region_id = "expand_18", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb122(%1153: f32, %1154: f32):
      linalg.yield %1153 : f32
    } -> tensor<1x28x128x16xf32>
    %1155 = tensor.collapse_shape %1152 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x128x16xf32> into tensor<57344xf32>
    %1156 = tensor.expand_shape %1155 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 128, 16] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<28x128x16xf32>
    %1157 = arith.constant {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %1158 = tensor.splat %1157 {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<28x8x16xf32>
    %1159 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1150, %1156 : tensor<28x8x128xf32>, tensor<28x128x16xf32>) outs(%1158 : tensor<28x8x16xf32>) attrs =  {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb123(%1160: f32, %1161: f32, %1162: f32):
      %1163 = arith.mulf %1160, %1161 : f32
      %1164 = arith.addf %1162, %1163 : f32
      linalg.yield %1164 : f32
    } -> tensor<28x8x16xf32>
    %1165 = tensor.collapse_shape %1159 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28x8x16xf32> into tensor<3584xf32>
    %1166 = tensor.expand_shape %1165 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 16] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<1x28x8x16xf32>
    %1167 = arith.constant {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.0883883461 : f32
    %1168 = tensor.splat %1167 {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x8x16xf32>
    %1169 = tensor.empty() : tensor<1x28x8x16xf32>
    %1170 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1166, %1168 : tensor<1x28x8x16xf32>, tensor<1x28x8x16xf32>) outs(%1169 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb124(%1171: f32, %1172: f32, %1173: f32):
      %1174 = arith.mulf %1171, %1172 : f32
      linalg.yield %1174 : f32
    } -> tensor<1x28x8x16xf32>
    %1175 = tensor.empty() : tensor<1x1x8x16xi1>
    %1176 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%175 : tensor<1x1x8x16xi1>) outs(%1175 : tensor<1x1x8x16xi1>) attrs =  {prov.region_id = "bitwise_2", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
    ^bb125(%1177: i1, %1178: i1):
      %1179 = arith.constant true
      %1180 = arith.xori %1177, %1179 : i1
      linalg.yield %1180 : i1
    } -> tensor<1x1x8x16xi1>
    %1181 = arith.constant {prov.region_id = "fill_5", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %1182 = tensor.splat %1181 {prov.region_id = "fill_5", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1183 = tensor.empty() : tensor<1x28x8x16xf32>
    %1184 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1176, %1182, %1170 : tensor<1x1x8x16xi1>, tensor<f32>, tensor<1x28x8x16xf32>) outs(%1183 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "select_8", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32"} {
    ^bb126(%1185: i1, %1186: f32, %1187: f32, %1188: f32):
      %1189 = arith.select %1185, %1186, %1187 : f32
      linalg.yield %1189 : f32
    } -> tensor<1x28x8x16xf32>
    %1190 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %1191 = tensor.splat %1190 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x8xf32>
    %1192 = linalg.reduce ins(%1184:tensor<1x28x8x16xf32>) outs(%1191:tensor<1x28x8xf32>) dimensions = [3]
    (%1193: f32, %1194: f32) {
      %1195 = arith.maximumf %1193, %1194 : f32
      linalg.yield %1195 : f32
    }
    %1196 = tensor.collapse_shape %1192 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x8xf32> into tensor<224xf32>
    %1197 = tensor.expand_shape %1196 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %1198 = tensor.empty() : tensor<1x28x8x16xf32>
    %1199 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1184, %1197 : tensor<1x28x8x16xf32>, tensor<1x28x8x1xf32>) outs(%1198 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb127(%1200: f32, %1201: f32, %1202: f32):
      %1203 = arith.subf %1200, %1201 : f32
      linalg.yield %1203 : f32
    } -> tensor<1x28x8x16xf32>
    %1204 = tensor.empty() : tensor<1x28x8x16xf32>
    %1205 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1199 : tensor<1x28x8x16xf32>) outs(%1204 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb128(%1206: f32, %1207: f32):
      %1208 = math.exp %1206 : f32
      linalg.yield %1208 : f32
    } -> tensor<1x28x8x16xf32>
    %1209 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %1210 = tensor.splat %1209 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x8xf32>
    %1211 = linalg.reduce ins(%1205:tensor<1x28x8x16xf32>) outs(%1210:tensor<1x28x8xf32>) dimensions = [3]
    (%1212: f32, %1213: f32) {
      %1214 = arith.addf %1212, %1213 : f32
      linalg.yield %1214 : f32
    }
    %1215 = tensor.collapse_shape %1211 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x8xf32> into tensor<224xf32>
    %1216 = tensor.expand_shape %1215 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %1217 = tensor.empty() : tensor<1x28x8x16xf32>
    %1218 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1205, %1216 : tensor<1x28x8x16xf32>, tensor<1x28x8x1xf32>) outs(%1217 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb129(%1219: f32, %1220: f32, %1221: f32):
      %1222 = arith.divf %1219, %1220 : f32
      linalg.yield %1222 : f32
    } -> tensor<1x28x8x16xf32>
    %1223 = tensor.empty() : tensor<1x28x8x16xf32>
    %1224 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1218 : tensor<1x28x8x16xf32>) outs(%1223 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "expand_19", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb130(%1225: f32, %1226: f32):
      linalg.yield %1225 : f32
    } -> tensor<1x28x8x16xf32>
    %1227 = tensor.collapse_shape %1224 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x8x16xf32> into tensor<3584xf32>
    %1228 = tensor.expand_shape %1227 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 16] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<28x8x16xf32>
    %1229 = tensor.empty() : tensor<1x28x16x128xf32>
    %1230 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1142 : tensor<1x28x16x128xf32>) outs(%1229 : tensor<1x28x16x128xf32>) attrs =  {prov.region_id = "expand_20", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb131(%1231: f32, %1232: f32):
      linalg.yield %1231 : f32
    } -> tensor<1x28x16x128xf32>
    %1233 = tensor.collapse_shape %1230 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x16x128xf32> into tensor<57344xf32>
    %1234 = tensor.expand_shape %1233 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 16, 128] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<28x16x128xf32>
    %1235 = arith.constant {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %1236 = tensor.splat %1235 {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<28x8x128xf32>
    %1237 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1228, %1234 : tensor<28x8x16xf32>, tensor<28x16x128xf32>) outs(%1236 : tensor<28x8x128xf32>) attrs =  {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb132(%1238: f32, %1239: f32, %1240: f32):
      %1241 = arith.mulf %1238, %1239 : f32
      %1242 = arith.addf %1240, %1241 : f32
      linalg.yield %1242 : f32
    } -> tensor<28x8x128xf32>
    %1243 = tensor.collapse_shape %1237 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28x8x128xf32> into tensor<28672xf32>
    %1244 = tensor.expand_shape %1243 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %1245 = tensor.empty() : tensor<1x8x28x128xf32>
    %1246 = linalg.transpose ins(%1244:tensor<1x28x8x128xf32>) outs(%1245:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
    %1247 = tensor.collapse_shape %1246 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x28x128xf32> into tensor<28672xf32>
    %1248 = tensor.expand_shape %1247 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %1249 = tensor.empty() : tensor<3584x3584xf32>
    %1250 = linalg.transpose ins(%20:tensor<3584x3584xf32>) outs(%1249:tensor<3584x3584xf32>) permutation = [1, 0]
    %1251 = tensor.collapse_shape %1248 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.attn_out"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %1252 = tensor.expand_shape %1251 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.attn_out"} : tensor<28672xf32> into tensor<8x3584xf32>
    %1253 = tensor.empty() : tensor<8x3584xf32>
    %1254 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1255 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1254 : f32) outs(%1253 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %1256 = linalg.matmul {prov.region_id = "matmul_16", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.attn_out", prov.transposed_b = "true"} ins(%1252, %1250 : tensor<8x3584xf32>, tensor<3584x3584xf32>) outs(%1255 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %1257 = tensor.collapse_shape %1256 [[0 : i64, 1 : i64]] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.attn_out"} : tensor<8x3584xf32> into tensor<28672xf32>
    %1258 = tensor.expand_shape %1257 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.attn_out"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %1259 = tensor.empty() : tensor<1x8x3584xf32>
    %1260 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%961, %1258 : tensor<1x8x3584xf32>, tensor<1x8x3584xf32>) outs(%1259 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb133(%1261: f32, %1262: f32, %1263: f32):
      %1264 = arith.addf %1261, %1262 : f32
      linalg.yield %1264 : f32
    } -> tensor<1x8x3584xf32>
    %1265 = tensor.empty() : tensor<1x8x3584xf32>
    %1266 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1260 : tensor<1x8x3584xf32>) outs(%1265 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb134(%1267: f32, %1268: f32):
      %1269 = arith.constant 2.000000e+00 : f32
      %1270 = math.powf %1267, %1269 : f32
      linalg.yield %1270 : f32
    } -> tensor<1x8x3584xf32>
    %1271 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %1272 = tensor.splat %1271 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %1273 = linalg.reduce ins(%1266:tensor<1x8x3584xf32>) outs(%1272:tensor<1x8xf32>) dimensions = [2]
    (%1274: f32, %1275: f32) {
      %1276 = arith.addf %1274, %1275 : f32
      linalg.yield %1276 : f32
    }
    %1277 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %1278 = tensor.splat %1277 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %1279 = tensor.empty() : tensor<1x8xf32>
    %1280 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1273, %1278 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%1279 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb135(%1281: f32, %1282: f32, %1283: f32):
      %1284 = arith.divf %1281, %1282 : f32
      linalg.yield %1284 : f32
    } -> tensor<1x8xf32>
    %1285 = tensor.collapse_shape %1280 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %1286 = tensor.expand_shape %1285 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %1287 = arith.constant {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %1288 = tensor.splat %1287 {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %1289 = tensor.empty() : tensor<1x8x1xf32>
    %1290 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1286, %1288 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%1289 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb136(%1291: f32, %1292: f32, %1293: f32):
      %1294 = arith.addf %1291, %1292 : f32
      linalg.yield %1294 : f32
    } -> tensor<1x8x1xf32>
    %1295 = tensor.empty() : tensor<1x8x1xf32>
    %1296 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1290 : tensor<1x8x1xf32>) outs(%1295 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_5", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb137(%1297: f32, %1298: f32):
      %1299 = math.rsqrt %1297 : f32
      linalg.yield %1299 : f32
    } -> tensor<1x8x1xf32>
    %1300 = tensor.empty() : tensor<1x8x3584xf32>
    %1301 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1260, %1296 : tensor<1x8x3584xf32>, tensor<1x8x1xf32>) outs(%1300 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb138(%1302: f32, %1303: f32, %1304: f32):
      %1305 = arith.mulf %1302, %1303 : f32
      linalg.yield %1305 : f32
    } -> tensor<1x8x3584xf32>
    %1306 = tensor.empty() : tensor<1x8x3584xf32>
    %1307 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %1301 : tensor<3584xf32>, tensor<1x8x3584xf32>) outs(%1306 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb139(%1308: f32, %1309: f32, %1310: f32):
      %1311 = arith.mulf %1308, %1309 : f32
      linalg.yield %1311 : f32
    } -> tensor<1x8x3584xf32>
    %1312 = tensor.empty() : tensor<3584x37888xf32>
    %1313 = linalg.transpose ins(%21:tensor<37888x3584xf32>) outs(%1312:tensor<3584x37888xf32>) permutation = [1, 0]
    %1314 = tensor.collapse_shape %1307 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_proj"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %1315 = tensor.expand_shape %1314 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_proj"} : tensor<28672xf32> into tensor<8x3584xf32>
    %1316 = tensor.empty() : tensor<8x37888xf32>
    %1317 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1318 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1317 : f32) outs(%1316 : tensor<8x37888xf32>) -> tensor<8x37888xf32>
    %1319 = linalg.matmul {prov.region_id = "matmul_17", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_proj", prov.transposed_b = "true"} ins(%1315, %1313 : tensor<8x3584xf32>, tensor<3584x37888xf32>) outs(%1318 : tensor<8x37888xf32>) -> tensor<8x37888xf32>
    %1320 = tensor.collapse_shape %1319 [[0 : i64, 1 : i64]] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_proj"} : tensor<8x37888xf32> into tensor<303104xf32>
    %1321 = tensor.expand_shape %1320 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 37888] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_proj"} : tensor<303104xf32> into tensor<1x8x37888xf32>
    %1322 = "tensor.extract_slice"(%1321) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_5", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x37888xf32>) -> tensor<1x8x18944xf32>
    %1323 = "tensor.extract_slice"(%1321) <{static_offsets = array<i64: 0, 0, 18944>, static_sizes = array<i64: 1, 8, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_5", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x37888xf32>) -> tensor<1x8x18944xf32>
    %1324 = tensor.empty() : tensor<1x8x18944xf32>
    %1325 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1323 : tensor<1x8x18944xf32>) outs(%1324 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "sigmoid_2", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.act"} {
    ^bb140(%1326: f32, %1327: f32):
      %1328 = arith.constant 1.000000e+00 : f32
      %1329 = arith.negf %1326 : f32
      %1330 = math.exp %1329 : f32
      %1331 = arith.addf %1328, %1330 : f32
      %1332 = arith.divf %1328, %1331 : f32
      linalg.yield %1332 : f32
    } -> tensor<1x8x18944xf32>
    %1333 = tensor.empty() : tensor<1x8x18944xf32>
    %1334 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1323, %1325 : tensor<1x8x18944xf32>, tensor<1x8x18944xf32>) outs(%1333 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "mul_34", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.act"} {
    ^bb141(%1335: f32, %1336: f32, %1337: f32):
      %1338 = arith.mulf %1335, %1336 : f32
      linalg.yield %1338 : f32
    } -> tensor<1x8x18944xf32>
    %1339 = tensor.empty() : tensor<1x8x18944xf32>
    %1340 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1334, %1322 : tensor<1x8x18944xf32>, tensor<1x8x18944xf32>) outs(%1339 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "mul_35", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb142(%1341: f32, %1342: f32, %1343: f32):
      %1344 = arith.mulf %1341, %1342 : f32
      linalg.yield %1344 : f32
    } -> tensor<1x8x18944xf32>
    %1345 = tensor.empty() : tensor<18944x3584xf32>
    %1346 = linalg.transpose ins(%22:tensor<3584x18944xf32>) outs(%1345:tensor<18944x3584xf32>) permutation = [1, 0]
    %1347 = tensor.collapse_shape %1340 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_out"} : tensor<1x8x18944xf32> into tensor<151552xf32>
    %1348 = tensor.expand_shape %1347 [[0 : i64, 1 : i64]] output_shape [8, 18944] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_out"} : tensor<151552xf32> into tensor<8x18944xf32>
    %1349 = tensor.empty() : tensor<8x3584xf32>
    %1350 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1351 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1350 : f32) outs(%1349 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %1352 = linalg.matmul {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_out", prov.transposed_b = "true"} ins(%1348, %1346 : tensor<8x18944xf32>, tensor<18944x3584xf32>) outs(%1351 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %1353 = tensor.collapse_shape %1352 [[0 : i64, 1 : i64]] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_out"} : tensor<8x3584xf32> into tensor<28672xf32>
    %1354 = tensor.expand_shape %1353 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_out"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %1355 = tensor.empty() : tensor<1x8x3584xf32>
    %1356 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1260, %1354 : tensor<1x8x3584xf32>, tensor<1x8x3584xf32>) outs(%1355 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb143(%1357: f32, %1358: f32, %1359: f32):
      %1360 = arith.addf %1357, %1358 : f32
      linalg.yield %1360 : f32
    } -> tensor<1x8x3584xf32>
    %1361 = tensor.empty() : tensor<1x8x3584xf32>
    %1362 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1356 : tensor<1x8x3584xf32>) outs(%1361 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb144(%1363: f32, %1364: f32):
      %1365 = arith.constant 2.000000e+00 : f32
      %1366 = math.powf %1363, %1365 : f32
      linalg.yield %1366 : f32
    } -> tensor<1x8x3584xf32>
    %1367 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %1368 = tensor.splat %1367 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %1369 = linalg.reduce ins(%1362:tensor<1x8x3584xf32>) outs(%1368:tensor<1x8xf32>) dimensions = [2]
    (%1370: f32, %1371: f32) {
      %1372 = arith.addf %1370, %1371 : f32
      linalg.yield %1372 : f32
    }
    %1373 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %1374 = tensor.splat %1373 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %1375 = tensor.empty() : tensor<1x8xf32>
    %1376 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1369, %1374 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%1375 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb145(%1377: f32, %1378: f32, %1379: f32):
      %1380 = arith.divf %1377, %1378 : f32
      linalg.yield %1380 : f32
    } -> tensor<1x8xf32>
    %1381 = tensor.collapse_shape %1376 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %1382 = tensor.expand_shape %1381 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %1383 = arith.constant {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %1384 = tensor.splat %1383 {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %1385 = tensor.empty() : tensor<1x8x1xf32>
    %1386 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1382, %1384 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%1385 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb146(%1387: f32, %1388: f32, %1389: f32):
      %1390 = arith.addf %1387, %1388 : f32
      linalg.yield %1390 : f32
    } -> tensor<1x8x1xf32>
    %1391 = tensor.empty() : tensor<1x8x1xf32>
    %1392 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1386 : tensor<1x8x1xf32>) outs(%1391 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_6", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb147(%1393: f32, %1394: f32):
      %1395 = math.rsqrt %1393 : f32
      linalg.yield %1395 : f32
    } -> tensor<1x8x1xf32>
    %1396 = tensor.empty() : tensor<1x8x3584xf32>
    %1397 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1356, %1392 : tensor<1x8x3584xf32>, tensor<1x8x1xf32>) outs(%1396 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_36", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb148(%1398: f32, %1399: f32, %1400: f32):
      %1401 = arith.mulf %1398, %1399 : f32
      linalg.yield %1401 : f32
    } -> tensor<1x8x3584xf32>
    %1402 = tensor.empty() : tensor<1x8x3584xf32>
    %1403 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %1397 : tensor<3584xf32>, tensor<1x8x3584xf32>) outs(%1402 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb149(%1404: f32, %1405: f32, %1406: f32):
      %1407 = arith.mulf %1404, %1405 : f32
      linalg.yield %1407 : f32
    } -> tensor<1x8x3584xf32>
    %1408 = tensor.collapse_shape %1403 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.att_proj"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %1409 = tensor.expand_shape %1408 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.att_proj"} : tensor<28672xf32> into tensor<8x3584xf32>
    %1410 = tensor.empty() : tensor<3584x4608xf32>
    %1411 = linalg.transpose ins(%25:tensor<4608x3584xf32>) outs(%1410:tensor<3584x4608xf32>) permutation = [1, 0]
    %1412 = tensor.empty() : tensor<8x4608xf32>
    %1413 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1414 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1413 : f32) outs(%1412 : tensor<8x4608xf32>) -> tensor<8x4608xf32>
    %1415 = linalg.matmul {prov.region_id = "matmul_19", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.att_proj", prov.transposed_b = "true"} ins(%1409, %1411 : tensor<8x3584xf32>, tensor<3584x4608xf32>) outs(%1414 : tensor<8x4608xf32>) -> tensor<8x4608xf32>
    %1416 = tensor.empty() : tensor<8x4608xf32>
    %1417 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1415, %26 : tensor<8x4608xf32>, tensor<4608xf32>) outs(%1416 : tensor<8x4608xf32>) attrs =  {prov.region_id = "add_26", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.att_proj"} {
    ^bb150(%1418: f32, %1419: f32, %1420: f32):
      %1421 = arith.addf %1418, %1419 : f32
      linalg.yield %1421 : f32
    } -> tensor<8x4608xf32>
    %1422 = tensor.collapse_shape %1417 [[0 : i64, 1 : i64]] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.att_proj"} : tensor<8x4608xf32> into tensor<36864xf32>
    %1423 = tensor.expand_shape %1422 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 4608] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.att_proj"} : tensor<36864xf32> into tensor<1x8x4608xf32>
    %1424 = "tensor.extract_slice"(%1423) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_6", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x4608xf32>) -> tensor<1x8x3584xf32>
    %1425 = "tensor.extract_slice"(%1423) <{static_offsets = array<i64: 0, 0, 3584>, static_sizes = array<i64: 1, 8, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_6", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x4608xf32>) -> tensor<1x8x512xf32>
    %1426 = "tensor.extract_slice"(%1423) <{static_offsets = array<i64: 0, 0, 4096>, static_sizes = array<i64: 1, 8, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_6", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x4608xf32>) -> tensor<1x8x512xf32>
    %1427 = tensor.collapse_shape %1424 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %1428 = tensor.expand_shape %1427 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 128] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x8x28x128xf32>
    %1429 = tensor.empty() : tensor<1x28x8x128xf32>
    %1430 = linalg.transpose ins(%1428:tensor<1x8x28x128xf32>) outs(%1429:tensor<1x28x8x128xf32>) permutation = [0, 2, 1, 3]
    %1431 = tensor.collapse_shape %1425 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x512xf32> into tensor<4096xf32>
    %1432 = tensor.expand_shape %1431 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 128] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x8x4x128xf32>
    %1433 = tensor.empty() : tensor<1x4x8x128xf32>
    %1434 = linalg.transpose ins(%1432:tensor<1x8x4x128xf32>) outs(%1433:tensor<1x4x8x128xf32>) permutation = [0, 2, 1, 3]
    %1435 = tensor.collapse_shape %1426 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x512xf32> into tensor<4096xf32>
    %1436 = tensor.expand_shape %1435 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 128] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x8x4x128xf32>
    %1437 = tensor.empty() : tensor<1x4x8x128xf32>
    %1438 = linalg.transpose ins(%1436:tensor<1x8x4x128xf32>) outs(%1437:tensor<1x4x8x128xf32>) permutation = [0, 2, 1, 3]
    %1439 = tensor.collapse_shape %122 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1440 = tensor.expand_shape %1439 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 128] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x8x128xf32>
    %1441 = tensor.collapse_shape %135 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1442 = tensor.expand_shape %1441 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 128] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x8x128xf32>
    %1443 = tensor.empty() : tensor<1x28x8x128xf32>
    %1444 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1430, %1440 : tensor<1x28x8x128xf32>, tensor<1x1x8x128xf32>) outs(%1443 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_38", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb151(%1445: f32, %1446: f32, %1447: f32):
      %1448 = arith.mulf %1445, %1446 : f32
      linalg.yield %1448 : f32
    } -> tensor<1x28x8x128xf32>
    %1449 = "tensor.extract_slice"(%1430) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 28, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_39", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x28x8x128xf32>) -> tensor<1x28x8x64xf32>
    %1450 = "tensor.extract_slice"(%1430) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 28, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_40", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x28x8x128xf32>) -> tensor<1x28x8x64xf32>
    %1451 = tensor.empty() : tensor<1x28x8x64xf32>
    %1452 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1450 : tensor<1x28x8x64xf32>) outs(%1451 : tensor<1x28x8x64xf32>) attrs =  {prov.region_id = "neg_6", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb152(%1453: f32, %1454: f32):
      %1455 = arith.negf %1453 : f32
      linalg.yield %1455 : f32
    } -> tensor<1x28x8x64xf32>
    %1456 = tensor.concat dim(3) %1452, %1449 {prov.region_id = "cat_8", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x28x8x64xf32>, tensor<1x28x8x64xf32>) -> tensor<1x28x8x128xf32>
    %1457 = tensor.empty() : tensor<1x28x8x128xf32>
    %1458 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1456, %1442 : tensor<1x28x8x128xf32>, tensor<1x1x8x128xf32>) outs(%1457 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_39", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb153(%1459: f32, %1460: f32, %1461: f32):
      %1462 = arith.mulf %1459, %1460 : f32
      linalg.yield %1462 : f32
    } -> tensor<1x28x8x128xf32>
    %1463 = tensor.empty() : tensor<1x28x8x128xf32>
    %1464 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1444, %1458 : tensor<1x28x8x128xf32>, tensor<1x28x8x128xf32>) outs(%1463 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb154(%1465: f32, %1466: f32, %1467: f32):
      %1468 = arith.addf %1465, %1466 : f32
      linalg.yield %1468 : f32
    } -> tensor<1x28x8x128xf32>
    %1469 = tensor.empty() : tensor<1x4x8x128xf32>
    %1470 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1434, %1440 : tensor<1x4x8x128xf32>, tensor<1x1x8x128xf32>) outs(%1469 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "mul_40", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb155(%1471: f32, %1472: f32, %1473: f32):
      %1474 = arith.mulf %1471, %1472 : f32
      linalg.yield %1474 : f32
    } -> tensor<1x4x8x128xf32>
    %1475 = "tensor.extract_slice"(%1434) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_41", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x64xf32>
    %1476 = "tensor.extract_slice"(%1434) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_42", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x64xf32>
    %1477 = tensor.empty() : tensor<1x4x8x64xf32>
    %1478 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1476 : tensor<1x4x8x64xf32>) outs(%1477 : tensor<1x4x8x64xf32>) attrs =  {prov.region_id = "neg_7", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
    ^bb156(%1479: f32, %1480: f32):
      %1481 = arith.negf %1479 : f32
      linalg.yield %1481 : f32
    } -> tensor<1x4x8x64xf32>
    %1482 = tensor.concat dim(3) %1478, %1475 {prov.region_id = "cat_9", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x8x64xf32>, tensor<1x4x8x64xf32>) -> tensor<1x4x8x128xf32>
    %1483 = tensor.empty() : tensor<1x4x8x128xf32>
    %1484 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1482, %1442 : tensor<1x4x8x128xf32>, tensor<1x1x8x128xf32>) outs(%1483 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb157(%1485: f32, %1486: f32, %1487: f32):
      %1488 = arith.mulf %1485, %1486 : f32
      linalg.yield %1488 : f32
    } -> tensor<1x4x8x128xf32>
    %1489 = tensor.empty() : tensor<1x4x8x128xf32>
    %1490 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1470, %1484 : tensor<1x4x8x128xf32>, tensor<1x4x8x128xf32>) outs(%1489 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb158(%1491: f32, %1492: f32, %1493: f32):
      %1494 = arith.addf %1491, %1492 : f32
      linalg.yield %1494 : f32
    } -> tensor<1x4x8x128xf32>
    %1495 = tensor.empty() : tensor<8xi64>
    %1496 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1495 : tensor<8xi64>) attrs =  {prov.region_id = "iota_6", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb159(%1497: i64):
      %1498 = linalg.index 0 : index
      %1499 = arith.index_cast %1498 : index to i64
      %1500 = arith.constant 1 : i64
      %1501 = arith.muli %1499, %1500 : i64
      %1502 = arith.constant 0 : i64
      %1503 = arith.addi %1502, %1501 : i64
      linalg.yield %1503 : i64
    } -> tensor<8xi64>
    %1504 = tensor.empty() : tensor<8xi64>
    %1505 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%67, %1496 : tensor<i64>, tensor<8xi64>) outs(%1504 : tensor<8xi64>) attrs =  {prov.region_id = "add_29", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb160(%1506: i64, %1507: i64, %1508: i64):
      %1509 = arith.addi %1506, %1507 : i64
      linalg.yield %1509 : i64
    } -> tensor<8xi64>
    %1510 = "tensor.extract_slice"(%36) <{static_offsets = array<i64: 3, 0, 0, 0, 0>, static_sizes = array<i64: 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_9", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<4x1x4x16x128xf32>) -> tensor<4x16x128xf32>
    %1511 = func.call @aten_index_put_default(%1510, %1505, %1490) {prov.region_id = "aten_index_put_default_6", prov.dispatch_id = "aten_index_put_default_6"} : (tensor<4x16x128xf32>, tensor<8xi64>, tensor<1x4x8x128xf32>) -> tensor<1x4x16x128xf32>
    %1512 = "tensor.extract_slice"(%38) <{static_offsets = array<i64: 3, 0, 0, 0, 0>, static_sizes = array<i64: 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_10", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<4x1x4x16x128xf32>) -> tensor<4x16x128xf32>
    %1513 = func.call @aten_index_put_default(%1512, %1505, %1438) {prov.region_id = "aten_index_put_default_7", prov.dispatch_id = "aten_index_put_default_7"} : (tensor<4x16x128xf32>, tensor<8xi64>, tensor<1x4x8x128xf32>) -> tensor<1x4x16x128xf32>
    %1514 = "tensor.extract_slice"(%1511) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_43", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
    %1515 = "tensor.extract_slice"(%1514) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_44", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
    %1516 = tensor.collapse_shape %1515 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x16x128xf32> into tensor<8192xf32>
    %1517 = tensor.expand_shape %1516 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 16, 128] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x4x1x16x128xf32>
    %1518 = "tensor.extract_slice"(%1517) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_45", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
    %1519 = "tensor.extract_slice"(%1518) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_46", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
    %1520 = tensor.empty() : tensor<1x4x7x16x128xf32>
    %1521 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1519 : tensor<1x4x1x16x128xf32>) outs(%1520 : tensor<1x4x7x16x128xf32>) attrs =  {prov.region_id = "expand_21", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb161(%1522: f32, %1523: f32):
      linalg.yield %1522 : f32
    } -> tensor<1x4x7x16x128xf32>
    %1524 = tensor.collapse_shape %1521 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x7x16x128xf32> into tensor<57344xf32>
    %1525 = tensor.expand_shape %1524 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 16, 128] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<1x28x16x128xf32>
    %1526 = "tensor.extract_slice"(%1513) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_47", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
    %1527 = "tensor.extract_slice"(%1526) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_48", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
    %1528 = tensor.collapse_shape %1527 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x16x128xf32> into tensor<8192xf32>
    %1529 = tensor.expand_shape %1528 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 16, 128] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x4x1x16x128xf32>
    %1530 = "tensor.extract_slice"(%1529) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_49", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
    %1531 = "tensor.extract_slice"(%1530) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_50", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
    %1532 = tensor.empty() : tensor<1x4x7x16x128xf32>
    %1533 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1531 : tensor<1x4x1x16x128xf32>) outs(%1532 : tensor<1x4x7x16x128xf32>) attrs =  {prov.region_id = "expand_22", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb162(%1534: f32, %1535: f32):
      linalg.yield %1534 : f32
    } -> tensor<1x4x7x16x128xf32>
    %1536 = tensor.collapse_shape %1533 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x7x16x128xf32> into tensor<57344xf32>
    %1537 = tensor.expand_shape %1536 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 16, 128] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<1x28x16x128xf32>
    %1538 = tensor.empty() : tensor<1x28x128x16xf32>
    %1539 = linalg.transpose ins(%1525:tensor<1x28x16x128xf32>) outs(%1538:tensor<1x28x128x16xf32>) permutation = [0, 1, 3, 2]
    %1540 = tensor.empty() : tensor<1x28x8x128xf32>
    %1541 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1464 : tensor<1x28x8x128xf32>) outs(%1540 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "expand_23", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb163(%1542: f32, %1543: f32):
      linalg.yield %1542 : f32
    } -> tensor<1x28x8x128xf32>
    %1544 = tensor.collapse_shape %1541 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
    %1545 = tensor.expand_shape %1544 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 128] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<28x8x128xf32>
    %1546 = tensor.empty() : tensor<1x28x128x16xf32>
    %1547 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1539 : tensor<1x28x128x16xf32>) outs(%1546 : tensor<1x28x128x16xf32>) attrs =  {prov.region_id = "expand_24", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb164(%1548: f32, %1549: f32):
      linalg.yield %1548 : f32
    } -> tensor<1x28x128x16xf32>
    %1550 = tensor.collapse_shape %1547 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x128x16xf32> into tensor<57344xf32>
    %1551 = tensor.expand_shape %1550 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 128, 16] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<28x128x16xf32>
    %1552 = arith.constant {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %1553 = tensor.splat %1552 {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<28x8x16xf32>
    %1554 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1545, %1551 : tensor<28x8x128xf32>, tensor<28x128x16xf32>) outs(%1553 : tensor<28x8x16xf32>) attrs =  {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb165(%1555: f32, %1556: f32, %1557: f32):
      %1558 = arith.mulf %1555, %1556 : f32
      %1559 = arith.addf %1557, %1558 : f32
      linalg.yield %1559 : f32
    } -> tensor<28x8x16xf32>
    %1560 = tensor.collapse_shape %1554 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28x8x16xf32> into tensor<3584xf32>
    %1561 = tensor.expand_shape %1560 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 16] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<1x28x8x16xf32>
    %1562 = arith.constant {prov.region_id = "mul_42", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.0883883461 : f32
    %1563 = tensor.splat %1562 {prov.region_id = "mul_42", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x8x16xf32>
    %1564 = tensor.empty() : tensor<1x28x8x16xf32>
    %1565 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1561, %1563 : tensor<1x28x8x16xf32>, tensor<1x28x8x16xf32>) outs(%1564 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "mul_42", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb166(%1566: f32, %1567: f32, %1568: f32):
      %1569 = arith.mulf %1566, %1567 : f32
      linalg.yield %1569 : f32
    } -> tensor<1x28x8x16xf32>
    %1570 = tensor.empty() : tensor<1x1x8x16xi1>
    %1571 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%175 : tensor<1x1x8x16xi1>) outs(%1570 : tensor<1x1x8x16xi1>) attrs =  {prov.region_id = "bitwise_3", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
    ^bb167(%1572: i1, %1573: i1):
      %1574 = arith.constant true
      %1575 = arith.xori %1572, %1574 : i1
      linalg.yield %1575 : i1
    } -> tensor<1x1x8x16xi1>
    %1576 = arith.constant {prov.region_id = "fill_6", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %1577 = tensor.splat %1576 {prov.region_id = "fill_6", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %1578 = tensor.empty() : tensor<1x28x8x16xf32>
    %1579 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1571, %1577, %1565 : tensor<1x1x8x16xi1>, tensor<f32>, tensor<1x28x8x16xf32>) outs(%1578 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "select_11", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32"} {
    ^bb168(%1580: i1, %1581: f32, %1582: f32, %1583: f32):
      %1584 = arith.select %1580, %1581, %1582 : f32
      linalg.yield %1584 : f32
    } -> tensor<1x28x8x16xf32>
    %1585 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %1586 = tensor.splat %1585 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x8xf32>
    %1587 = linalg.reduce ins(%1579:tensor<1x28x8x16xf32>) outs(%1586:tensor<1x28x8xf32>) dimensions = [3]
    (%1588: f32, %1589: f32) {
      %1590 = arith.maximumf %1588, %1589 : f32
      linalg.yield %1590 : f32
    }
    %1591 = tensor.collapse_shape %1587 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x8xf32> into tensor<224xf32>
    %1592 = tensor.expand_shape %1591 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %1593 = tensor.empty() : tensor<1x28x8x16xf32>
    %1594 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1579, %1592 : tensor<1x28x8x16xf32>, tensor<1x28x8x1xf32>) outs(%1593 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb169(%1595: f32, %1596: f32, %1597: f32):
      %1598 = arith.subf %1595, %1596 : f32
      linalg.yield %1598 : f32
    } -> tensor<1x28x8x16xf32>
    %1599 = tensor.empty() : tensor<1x28x8x16xf32>
    %1600 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1594 : tensor<1x28x8x16xf32>) outs(%1599 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb170(%1601: f32, %1602: f32):
      %1603 = math.exp %1601 : f32
      linalg.yield %1603 : f32
    } -> tensor<1x28x8x16xf32>
    %1604 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %1605 = tensor.splat %1604 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x8xf32>
    %1606 = linalg.reduce ins(%1600:tensor<1x28x8x16xf32>) outs(%1605:tensor<1x28x8xf32>) dimensions = [3]
    (%1607: f32, %1608: f32) {
      %1609 = arith.addf %1607, %1608 : f32
      linalg.yield %1609 : f32
    }
    %1610 = tensor.collapse_shape %1606 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x8xf32> into tensor<224xf32>
    %1611 = tensor.expand_shape %1610 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %1612 = tensor.empty() : tensor<1x28x8x16xf32>
    %1613 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1600, %1611 : tensor<1x28x8x16xf32>, tensor<1x28x8x1xf32>) outs(%1612 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb171(%1614: f32, %1615: f32, %1616: f32):
      %1617 = arith.divf %1614, %1615 : f32
      linalg.yield %1617 : f32
    } -> tensor<1x28x8x16xf32>
    %1618 = tensor.empty() : tensor<1x28x8x16xf32>
    %1619 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1613 : tensor<1x28x8x16xf32>) outs(%1618 : tensor<1x28x8x16xf32>) attrs =  {prov.region_id = "expand_25", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb172(%1620: f32, %1621: f32):
      linalg.yield %1620 : f32
    } -> tensor<1x28x8x16xf32>
    %1622 = tensor.collapse_shape %1619 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x8x16xf32> into tensor<3584xf32>
    %1623 = tensor.expand_shape %1622 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 16] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<28x8x16xf32>
    %1624 = tensor.empty() : tensor<1x28x16x128xf32>
    %1625 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1537 : tensor<1x28x16x128xf32>) outs(%1624 : tensor<1x28x16x128xf32>) attrs =  {prov.region_id = "expand_26", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb173(%1626: f32, %1627: f32):
      linalg.yield %1626 : f32
    } -> tensor<1x28x16x128xf32>
    %1628 = tensor.collapse_shape %1625 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x16x128xf32> into tensor<57344xf32>
    %1629 = tensor.expand_shape %1628 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 16, 128] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<28x16x128xf32>
    %1630 = arith.constant {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %1631 = tensor.splat %1630 {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<28x8x128xf32>
    %1632 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1623, %1629 : tensor<28x8x16xf32>, tensor<28x16x128xf32>) outs(%1631 : tensor<28x8x128xf32>) attrs =  {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb174(%1633: f32, %1634: f32, %1635: f32):
      %1636 = arith.mulf %1633, %1634 : f32
      %1637 = arith.addf %1635, %1636 : f32
      linalg.yield %1637 : f32
    } -> tensor<28x8x128xf32>
    %1638 = tensor.collapse_shape %1632 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28x8x128xf32> into tensor<28672xf32>
    %1639 = tensor.expand_shape %1638 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %1640 = tensor.empty() : tensor<1x8x28x128xf32>
    %1641 = linalg.transpose ins(%1639:tensor<1x28x8x128xf32>) outs(%1640:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
    %1642 = tensor.collapse_shape %1641 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x28x128xf32> into tensor<28672xf32>
    %1643 = tensor.expand_shape %1642 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %1644 = tensor.empty() : tensor<3584x3584xf32>
    %1645 = linalg.transpose ins(%27:tensor<3584x3584xf32>) outs(%1644:tensor<3584x3584xf32>) permutation = [1, 0]
    %1646 = tensor.collapse_shape %1643 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.attn_out"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %1647 = tensor.expand_shape %1646 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.attn_out"} : tensor<28672xf32> into tensor<8x3584xf32>
    %1648 = tensor.empty() : tensor<8x3584xf32>
    %1649 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1650 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1649 : f32) outs(%1648 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %1651 = linalg.matmul {prov.region_id = "matmul_22", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.attn_out", prov.transposed_b = "true"} ins(%1647, %1645 : tensor<8x3584xf32>, tensor<3584x3584xf32>) outs(%1650 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %1652 = tensor.collapse_shape %1651 [[0 : i64, 1 : i64]] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.attn_out"} : tensor<8x3584xf32> into tensor<28672xf32>
    %1653 = tensor.expand_shape %1652 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.attn_out"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %1654 = tensor.empty() : tensor<1x8x3584xf32>
    %1655 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1356, %1653 : tensor<1x8x3584xf32>, tensor<1x8x3584xf32>) outs(%1654 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb175(%1656: f32, %1657: f32, %1658: f32):
      %1659 = arith.addf %1656, %1657 : f32
      linalg.yield %1659 : f32
    } -> tensor<1x8x3584xf32>
    %1660 = tensor.empty() : tensor<1x8x3584xf32>
    %1661 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1655 : tensor<1x8x3584xf32>) outs(%1660 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb176(%1662: f32, %1663: f32):
      %1664 = arith.constant 2.000000e+00 : f32
      %1665 = math.powf %1662, %1664 : f32
      linalg.yield %1665 : f32
    } -> tensor<1x8x3584xf32>
    %1666 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %1667 = tensor.splat %1666 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %1668 = linalg.reduce ins(%1661:tensor<1x8x3584xf32>) outs(%1667:tensor<1x8xf32>) dimensions = [2]
    (%1669: f32, %1670: f32) {
      %1671 = arith.addf %1669, %1670 : f32
      linalg.yield %1671 : f32
    }
    %1672 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %1673 = tensor.splat %1672 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %1674 = tensor.empty() : tensor<1x8xf32>
    %1675 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1668, %1673 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%1674 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb177(%1676: f32, %1677: f32, %1678: f32):
      %1679 = arith.divf %1676, %1677 : f32
      linalg.yield %1679 : f32
    } -> tensor<1x8xf32>
    %1680 = tensor.collapse_shape %1675 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %1681 = tensor.expand_shape %1680 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %1682 = arith.constant {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %1683 = tensor.splat %1682 {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %1684 = tensor.empty() : tensor<1x8x1xf32>
    %1685 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1681, %1683 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%1684 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb178(%1686: f32, %1687: f32, %1688: f32):
      %1689 = arith.addf %1686, %1687 : f32
      linalg.yield %1689 : f32
    } -> tensor<1x8x1xf32>
    %1690 = tensor.empty() : tensor<1x8x1xf32>
    %1691 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1685 : tensor<1x8x1xf32>) outs(%1690 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_7", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb179(%1692: f32, %1693: f32):
      %1694 = math.rsqrt %1692 : f32
      linalg.yield %1694 : f32
    } -> tensor<1x8x1xf32>
    %1695 = tensor.empty() : tensor<1x8x3584xf32>
    %1696 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1655, %1691 : tensor<1x8x3584xf32>, tensor<1x8x1xf32>) outs(%1695 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb180(%1697: f32, %1698: f32, %1699: f32):
      %1700 = arith.mulf %1697, %1698 : f32
      linalg.yield %1700 : f32
    } -> tensor<1x8x3584xf32>
    %1701 = tensor.empty() : tensor<1x8x3584xf32>
    %1702 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %1696 : tensor<3584xf32>, tensor<1x8x3584xf32>) outs(%1701 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_44", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb181(%1703: f32, %1704: f32, %1705: f32):
      %1706 = arith.mulf %1703, %1704 : f32
      linalg.yield %1706 : f32
    } -> tensor<1x8x3584xf32>
    %1707 = tensor.empty() : tensor<3584x37888xf32>
    %1708 = linalg.transpose ins(%28:tensor<37888x3584xf32>) outs(%1707:tensor<3584x37888xf32>) permutation = [1, 0]
    %1709 = tensor.collapse_shape %1702 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_79", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_proj"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %1710 = tensor.expand_shape %1709 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_79", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_proj"} : tensor<28672xf32> into tensor<8x3584xf32>
    %1711 = tensor.empty() : tensor<8x37888xf32>
    %1712 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1713 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1712 : f32) outs(%1711 : tensor<8x37888xf32>) -> tensor<8x37888xf32>
    %1714 = linalg.matmul {prov.region_id = "matmul_23", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_proj", prov.transposed_b = "true"} ins(%1710, %1708 : tensor<8x3584xf32>, tensor<3584x37888xf32>) outs(%1713 : tensor<8x37888xf32>) -> tensor<8x37888xf32>
    %1715 = tensor.collapse_shape %1714 [[0 : i64, 1 : i64]] {prov.region_id = "view_80", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_proj"} : tensor<8x37888xf32> into tensor<303104xf32>
    %1716 = tensor.expand_shape %1715 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 37888] {prov.region_id = "view_80", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_proj"} : tensor<303104xf32> into tensor<1x8x37888xf32>
    %1717 = "tensor.extract_slice"(%1716) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_7", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x37888xf32>) -> tensor<1x8x18944xf32>
    %1718 = "tensor.extract_slice"(%1716) <{static_offsets = array<i64: 0, 0, 18944>, static_sizes = array<i64: 1, 8, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_7", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x8x37888xf32>) -> tensor<1x8x18944xf32>
    %1719 = tensor.empty() : tensor<1x8x18944xf32>
    %1720 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1718 : tensor<1x8x18944xf32>) outs(%1719 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "sigmoid_3", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.act"} {
    ^bb182(%1721: f32, %1722: f32):
      %1723 = arith.constant 1.000000e+00 : f32
      %1724 = arith.negf %1721 : f32
      %1725 = math.exp %1724 : f32
      %1726 = arith.addf %1723, %1725 : f32
      %1727 = arith.divf %1723, %1726 : f32
      linalg.yield %1727 : f32
    } -> tensor<1x8x18944xf32>
    %1728 = tensor.empty() : tensor<1x8x18944xf32>
    %1729 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1718, %1720 : tensor<1x8x18944xf32>, tensor<1x8x18944xf32>) outs(%1728 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.act"} {
    ^bb183(%1730: f32, %1731: f32, %1732: f32):
      %1733 = arith.mulf %1730, %1731 : f32
      linalg.yield %1733 : f32
    } -> tensor<1x8x18944xf32>
    %1734 = tensor.empty() : tensor<1x8x18944xf32>
    %1735 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1729, %1717 : tensor<1x8x18944xf32>, tensor<1x8x18944xf32>) outs(%1734 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "mul_46", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb184(%1736: f32, %1737: f32, %1738: f32):
      %1739 = arith.mulf %1736, %1737 : f32
      linalg.yield %1739 : f32
    } -> tensor<1x8x18944xf32>
    %1740 = tensor.empty() : tensor<18944x3584xf32>
    %1741 = linalg.transpose ins(%29:tensor<3584x18944xf32>) outs(%1740:tensor<18944x3584xf32>) permutation = [1, 0]
    %1742 = tensor.collapse_shape %1735 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_81", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_out"} : tensor<1x8x18944xf32> into tensor<151552xf32>
    %1743 = tensor.expand_shape %1742 [[0 : i64, 1 : i64]] output_shape [8, 18944] {prov.region_id = "view_81", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_out"} : tensor<151552xf32> into tensor<8x18944xf32>
    %1744 = tensor.empty() : tensor<8x3584xf32>
    %1745 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
    %1746 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%1745 : f32) outs(%1744 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %1747 = linalg.matmul {prov.region_id = "matmul_24", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_out", prov.transposed_b = "true"} ins(%1743, %1741 : tensor<8x18944xf32>, tensor<18944x3584xf32>) outs(%1746 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %1748 = tensor.collapse_shape %1747 [[0 : i64, 1 : i64]] {prov.region_id = "view_82", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_out"} : tensor<8x3584xf32> into tensor<28672xf32>
    %1749 = tensor.expand_shape %1748 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_82", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_out"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %1750 = tensor.empty() : tensor<1x8x3584xf32>
    %1751 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1655, %1749 : tensor<1x8x3584xf32>, tensor<1x8x3584xf32>) outs(%1750 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb185(%1752: f32, %1753: f32, %1754: f32):
      %1755 = arith.addf %1752, %1753 : f32
      linalg.yield %1755 : f32
    } -> tensor<1x8x3584xf32>
    %1756 = tensor.concat dim(0) %326, %721, %1116, %1511 {prov.region_id = "cat_10", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>, tensor<1x4x16x128xf32>, tensor<1x4x16x128xf32>, tensor<1x4x16x128xf32>) -> tensor<4x4x16x128xf32>
    %1757 = tensor.collapse_shape %1756 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_83", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<4x4x16x128xf32> into tensor<32768xf32>
    %1758 = tensor.expand_shape %1757 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [4, 1, 4, 16, 128] {prov.region_id = "view_83", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<4x1x4x16x128xf32>
    %1759 = tensor.concat dim(0) %328, %723, %1118, %1513 {prov.region_id = "cat_11", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>, tensor<1x4x16x128xf32>, tensor<1x4x16x128xf32>, tensor<1x4x16x128xf32>) -> tensor<4x4x16x128xf32>
    %1760 = tensor.collapse_shape %1759 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_84", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<4x4x16x128xf32> into tensor<32768xf32>
    %1761 = tensor.expand_shape %1760 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [4, 1, 4, 16, 128] {prov.region_id = "view_84", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<4x1x4x16x128xf32>
    %1762 = "tensor.extract_slice"(%1751) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_51", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x3584xf32>) -> tensor<1x8x3584xf32>
    %1763 = "tensor.extract_slice"(%1762) <{static_offsets = array<i64: 0, 7, 0>, static_sizes = array<i64: 1, 1, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_52", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x3584xf32>) -> tensor<1x1x3584xf32>
    %1764 = "tensor.extract_slice"(%1763) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_53", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x1x3584xf32>) -> tensor<1x1x3584xf32>
    %1765 = tensor.empty() : tensor<1x1x3584xf32>
    %1766 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1764 : tensor<1x1x3584xf32>) outs(%1765 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb186(%1767: f32, %1768: f32):
      %1769 = arith.constant 2.000000e+00 : f32
      %1770 = math.powf %1767, %1769 : f32
      linalg.yield %1770 : f32
    } -> tensor<1x1x3584xf32>
    %1771 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %1772 = tensor.splat %1771 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
    %1773 = linalg.reduce ins(%1766:tensor<1x1x3584xf32>) outs(%1772:tensor<1x1xf32>) dimensions = [2]
    (%1774: f32, %1775: f32) {
      %1776 = arith.addf %1774, %1775 : f32
      linalg.yield %1776 : f32
    }
    %1777 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %1778 = tensor.splat %1777 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
    %1779 = tensor.empty() : tensor<1x1xf32>
    %1780 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1773, %1778 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%1779 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb187(%1781: f32, %1782: f32, %1783: f32):
      %1784 = arith.divf %1781, %1782 : f32
      linalg.yield %1784 : f32
    } -> tensor<1x1xf32>
    %1785 = tensor.collapse_shape %1780 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
    %1786 = tensor.expand_shape %1785 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
    %1787 = arith.constant {prov.region_id = "add_33", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %1788 = tensor.splat %1787 {prov.region_id = "add_33", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
    %1789 = tensor.empty() : tensor<1x1x1xf32>
    %1790 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1786, %1788 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%1789 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_33", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb188(%1791: f32, %1792: f32, %1793: f32):
      %1794 = arith.addf %1791, %1792 : f32
      linalg.yield %1794 : f32
    } -> tensor<1x1x1xf32>
    %1795 = tensor.empty() : tensor<1x1x1xf32>
    %1796 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1790 : tensor<1x1x1xf32>) outs(%1795 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_8", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb189(%1797: f32, %1798: f32):
      %1799 = math.rsqrt %1797 : f32
      linalg.yield %1799 : f32
    } -> tensor<1x1x1xf32>
    %1800 = tensor.empty() : tensor<1x1x3584xf32>
    %1801 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1764, %1796 : tensor<1x1x3584xf32>, tensor<1x1x1xf32>) outs(%1800 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb190(%1802: f32, %1803: f32, %1804: f32):
      %1805 = arith.mulf %1802, %1803 : f32
      linalg.yield %1805 : f32
    } -> tensor<1x1x3584xf32>
    %1806 = tensor.empty() : tensor<1x1x3584xf32>
    %1807 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %1801 : tensor<3584xf32>, tensor<1x1x3584xf32>) outs(%1806 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_48", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} {
    ^bb191(%1808: f32, %1809: f32, %1810: f32):
      %1811 = arith.mulf %1808, %1809 : f32
      linalg.yield %1811 : f32
    } -> tensor<1x1x3584xf32>
    %1812 = tensor.empty() : tensor<3584x4096xf32>
    %1813 = linalg.transpose ins(%31:tensor<4096x3584xf32>) outs(%1812:tensor<3584x4096xf32>) permutation = [1, 0]
    %1814 = tensor.collapse_shape %1807 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_85", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm_head", prov.fqn = "lm_head"} : tensor<1x1x3584xf32> into tensor<3584xf32>
    %1815 = tensor.expand_shape %1814 [[0 : i64, 1 : i64]] output_shape [1, 3584] {prov.region_id = "view_85", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm_head", prov.fqn = "lm_head"} : tensor<3584xf32> into tensor<1x3584xf32>
    %1816 = tensor.empty() : tensor<1x4096xf32>
    %1817 = arith.constant {prov.module = "lm_head"} 0.000000e+00 : f32
    %1818 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm_head"} ins(%1817 : f32) outs(%1816 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %1819 = linalg.matmul {prov.region_id = "matmul_25", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm_head", prov.fqn = "lm_head", prov.transposed_b = "true"} ins(%1815, %1813 : tensor<1x3584xf32>, tensor<3584x4096xf32>) outs(%1818 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
    %1820 = tensor.collapse_shape %1819 [[0 : i64, 1 : i64]] {prov.region_id = "view_86", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm_head", prov.fqn = "lm_head"} : tensor<1x4096xf32> into tensor<4096xf32>
    %1821 = tensor.expand_shape %1820 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 4096] {prov.region_id = "view_86", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm_head", prov.fqn = "lm_head"} : tensor<4096xf32> into tensor<1x1x4096xf32>
    %1822 = "tensor.extract_slice"(%1821) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 4096>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_54", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
    %1823 = "tensor.extract_slice"(%1822) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 4096>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_12", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<1x1x4096xf32>) -> tensor<4096xf32>
    %1824 = tensor.expand_shape %1823 [[0 : i64, 1 : i64]] output_shape [1, 4096] {prov.region_id = "slice_55", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x4096xf32>
    %1825 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0xff800000 : f32
    %1826 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0 : i64
    %1827 = tensor.splat %1825 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xf32>
    %1828 = tensor.splat %1826 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xi64>
    %1829, %1830 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%1824 : tensor<1x4096xf32>) outs(%1827, %1828 : tensor<1xf32>, tensor<1xi64>) attrs =  {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} {
    ^bb192(%1831: f32, %1832: f32, %1833: i64):
      %1834 = linalg.index 1 : index
      %1835 = arith.index_cast %1834 : index to i64
      %1836 = arith.cmpf ogt, %1831, %1832 : f32
      %1837 = arith.select %1836, %1831, %1832 : f32
      %1838 = arith.select %1836, %1835, %1833 : i64
      linalg.yield %1837, %1838 : f32, i64
    } -> (tensor<1xf32>, tensor<1xi64>)
    %1839 = tensor.expand_shape %1829 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xf32> into tensor<1x1xf32>
    %1840 = tensor.expand_shape %1830 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xi64> into tensor<1x1xi64>
    %1841 = arith.constant {prov.region_id = "fill_7", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "int64"} 0 : i64
    %1842 = tensor.splat %1841 {prov.region_id = "fill_7", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "int64"} : tensor<i64>
    %1843 = arith.constant {prov.region_id = "fill_8", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "int64"} 0 : i64
    %1844 = tensor.splat %1843 {prov.region_id = "fill_8", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "int64"} : tensor<1x8xi64>
    %1845 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 0 : index
    %1846 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 8 : index
    %1847 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 1 : index
    %1848, %1849, %1850, %1851, %1852 = scf.for %1853 = %1845 to %1846 step %1847 iter_args(%1854 = %1842, %1855 = %1840, %1856 = %1844, %1857 = %1758, %1858 = %1761) -> (tensor<i64>, tensor<1x1xi64>, tensor<1x8xi64>, tensor<4x1x4x16x128xf32>, tensor<4x1x4x16x128xf32>) {
      %1859 = tensor.extract %1854[] : tensor<i64>
      %1860 = tensor.from_elements %1859 {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1xi64>
      %1861 = func.call @aten_index_put_default_wl0(%1856, %1860, %1855) {prov.region_id = "aten_index_put_default_0", prov.dispatch_id = "aten_index_put_default_0"} : (tensor<1x8xi64>, tensor<1xi64>, tensor<1x1xi64>) -> tensor<1x8xi64>
      %1862 = tensor.empty() : tensor<i64>
      %1863 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%32, %1854 : tensor<i64>, tensor<i64>) outs(%1862 : tensor<i64>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb193(%1864: i64, %1865: i64, %1866: i64):
        %1867 = arith.addi %1864, %1865 : i64
        linalg.yield %1867 : i64
      } -> tensor<i64>
      %1868 = tensor.concat dim(0) %0, %1 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "embed", prov.fqn = "embed"} : (tensor<4096x3584xf32>, tensor<128x3584xf32>) -> tensor<4224x3584xf32>
      %1869 = tensor.empty() : tensor<1x1x3584xf32>
      %1870 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1855 : tensor<1x1xi64>) outs(%1869 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32", prov.module = "embed", prov.fqn = "embed"} {
      ^bb194(%1871: i64, %1872: f32):
        %1873 = arith.index_cast %1871 : i64 to index
        %1874 = linalg.index 2 : index
        %1875 = tensor.extract %1868[%1873, %1874] : tensor<4224x3584xf32>
        linalg.yield %1875 : f32
      } -> tensor<1x1x3584xf32>
      %1876 = tensor.extract %1863[] : tensor<i64>
      %1877 = tensor.from_elements %1876 {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int64"} : tensor<1x1xi64>
      %1878 = tensor.expand_shape %33 [[0 : i64, 1 : i64]] output_shape [1, 64] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<64xf32> into tensor<1x64xf32>
      %1879 = "tensor.extract_slice"(%1878) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 64>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x64xf32>) -> tensor<1x64xf32>
      %1880 = tensor.collapse_shape %1879 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x64xf32> into tensor<64xf32>
      %1881 = tensor.expand_shape %1880 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 1] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<64xf32> into tensor<1x64x1xf32>
      %1882 = tensor.empty() : tensor<1x64x1xf32>
      %1883 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1881 : tensor<1x64x1xf32>) outs(%1882 : tensor<1x64x1xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb195(%1884: f32, %1885: f32):
        linalg.yield %1884 : f32
      } -> tensor<1x64x1xf32>
      %1886 = "tensor.extract_slice"(%1877) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int64"} : (tensor<1x1xi64>) -> tensor<1x1xi64>
      %1887 = tensor.collapse_shape %1886 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1x1xi64> into tensor<1xi64>
      %1888 = tensor.expand_shape %1887 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1xi64> into tensor<1x1x1xi64>
      %1889 = "tensor.extract_slice"(%1888) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 1>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int64"} : (tensor<1x1x1xi64>) -> tensor<1x1x1xi64>
      %1890 = tensor.empty() : tensor<1x1x1xf32>
      %1891 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1889 : tensor<1x1x1xi64>) outs(%1890 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
      ^bb196(%1892: i64, %1893: f32):
        %1894 = arith.sitofp %1892 : i64 to f32
        linalg.yield %1894 : f32
      } -> tensor<1x1x1xf32>
      %1895 = tensor.empty() : tensor<1x64x1xf32>
      %1896 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1883 : tensor<1x64x1xf32>) outs(%1895 : tensor<1x64x1xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb197(%1897: f32, %1898: f32):
        linalg.yield %1897 : f32
      } -> tensor<1x64x1xf32>
      %1899 = tensor.empty() : tensor<1x1x1xf32>
      %1900 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1891 : tensor<1x1x1xf32>) outs(%1899 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb198(%1901: f32, %1902: f32):
        linalg.yield %1901 : f32
      } -> tensor<1x1x1xf32>
      %1903 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1904 = tensor.splat %1903 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<1x64x1xf32>
      %1905 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1896, %1900 : tensor<1x64x1xf32>, tensor<1x1x1xf32>) outs(%1904 : tensor<1x64x1xf32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
      ^bb199(%1906: f32, %1907: f32, %1908: f32):
        %1909 = arith.mulf %1906, %1907 : f32
        %1910 = arith.addf %1908, %1909 : f32
        linalg.yield %1910 : f32
      } -> tensor<1x64x1xf32>
      %1911 = tensor.empty() : tensor<1x1x64xf32>
      %1912 = linalg.transpose ins(%1905:tensor<1x64x1xf32>) outs(%1911:tensor<1x1x64xf32>) permutation = [0, 2, 1]
      %1913 = tensor.concat dim(2) %1912, %1912 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x1x64xf32>, tensor<1x1x64xf32>) -> tensor<1x1x128xf32>
      %1914 = tensor.empty() : tensor<1x1x128xf32>
      %1915 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1913 : tensor<1x1x128xf32>) outs(%1914 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32"} {
      ^bb200(%1916: f32, %1917: f32):
        %1918 = math.cos %1916 : f32
        linalg.yield %1918 : f32
      } -> tensor<1x1x128xf32>
      %1919 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
      %1920 = tensor.splat %1919 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x128xf32>
      %1921 = tensor.empty() : tensor<1x1x128xf32>
      %1922 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1915, %1920 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%1921 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb201(%1923: f32, %1924: f32, %1925: f32):
        %1926 = arith.mulf %1923, %1924 : f32
        linalg.yield %1926 : f32
      } -> tensor<1x1x128xf32>
      %1927 = tensor.empty() : tensor<1x1x128xf32>
      %1928 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1913 : tensor<1x1x128xf32>) outs(%1927 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32"} {
      ^bb202(%1929: f32, %1930: f32):
        %1931 = math.sin %1929 : f32
        linalg.yield %1931 : f32
      } -> tensor<1x1x128xf32>
      %1932 = arith.constant {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
      %1933 = tensor.splat %1932 {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x128xf32>
      %1934 = tensor.empty() : tensor<1x1x128xf32>
      %1935 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1928, %1933 : tensor<1x1x128xf32>, tensor<1x1x128xf32>) outs(%1934 : tensor<1x1x128xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb203(%1936: f32, %1937: f32, %1938: f32):
        %1939 = arith.mulf %1936, %1937 : f32
        linalg.yield %1939 : f32
      } -> tensor<1x1x128xf32>
      %1940 = tensor.empty() : tensor<16xi64>
      %1941 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1940 : tensor<16xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
      ^bb204(%1942: i64):
        %1943 = linalg.index 0 : index
        %1944 = arith.index_cast %1943 : index to i64
        %1945 = arith.constant 1 : i64
        %1946 = arith.muli %1944, %1945 : i64
        %1947 = arith.constant 0 : i64
        %1948 = arith.addi %1947, %1946 : i64
        linalg.yield %1948 : i64
      } -> tensor<16xi64>
      %1949 = tensor.empty() : tensor<1xi64>
      %1950 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%1949 : tensor<1xi64>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
      ^bb205(%1951: i64):
        %1952 = linalg.index 0 : index
        %1953 = arith.index_cast %1952 : index to i64
        %1954 = arith.constant 1 : i64
        %1955 = arith.muli %1953, %1954 : i64
        %1956 = arith.constant 0 : i64
        %1957 = arith.addi %1956, %1955 : i64
        linalg.yield %1957 : i64
      } -> tensor<1xi64>
      %1958 = tensor.empty() : tensor<1xi64>
      %1959 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1863, %1950 : tensor<i64>, tensor<1xi64>) outs(%1958 : tensor<1xi64>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb206(%1960: i64, %1961: i64, %1962: i64):
        %1963 = arith.addi %1960, %1961 : i64
        linalg.yield %1963 : i64
      } -> tensor<1xi64>
      %1964 = tensor.expand_shape %1959 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1xi64> into tensor<1x1xi64>
      %1965 = tensor.expand_shape %1941 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<16xi64> into tensor<1x16xi64>
      %1966 = tensor.empty() : tensor<1x16xi1>
      %1967 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1965, %1964 : tensor<1x16xi64>, tensor<1x1xi64>) outs(%1966 : tensor<1x16xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.le.Tensor", prov.orig_dtype = "bool"} {
      ^bb207(%1968: i64, %1969: i64, %1970: i1):
        %1971 = arith.cmpi sle, %1968, %1969 : i64
        linalg.yield %1971 : i1
      } -> tensor<1x16xi1>
      %1972 = tensor.collapse_shape %1967 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x16xi1> into tensor<16xi1>
      %1973 = tensor.expand_shape %1972 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 16] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<16xi1> into tensor<1x1x16xi1>
      %1974 = tensor.collapse_shape %1973 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<1x1x16xi1> into tensor<16xi1>
      %1975 = tensor.expand_shape %1974 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 16] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "bool"} : tensor<16xi1> into tensor<1x1x1x16xi1>
      %1976 = tensor.empty() : tensor<1x1x3584xf32>
      %1977 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1870 : tensor<1x1x3584xf32>) outs(%1976 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb208(%1978: f32, %1979: f32):
        %1980 = arith.constant 2.000000e+00 : f32
        %1981 = math.powf %1978, %1980 : f32
        linalg.yield %1981 : f32
      } -> tensor<1x1x3584xf32>
      %1982 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1983 = tensor.splat %1982 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1984 = linalg.reduce ins(%1977:tensor<1x1x3584xf32>) outs(%1983:tensor<1x1xf32>) dimensions = [2]
      (%1985: f32, %1986: f32) {
        %1987 = arith.addf %1985, %1986 : f32
        linalg.yield %1987 : f32
      }
      %1988 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
      %1989 = tensor.splat %1988 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %1990 = tensor.empty() : tensor<1x1xf32>
      %1991 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1984, %1989 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%1990 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb209(%1992: f32, %1993: f32, %1994: f32):
        %1995 = arith.divf %1992, %1993 : f32
        linalg.yield %1995 : f32
      } -> tensor<1x1xf32>
      %1996 = tensor.collapse_shape %1991 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %1997 = tensor.expand_shape %1996 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %1998 = arith.constant {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %1999 = tensor.splat %1998 {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %2000 = tensor.empty() : tensor<1x1x1xf32>
      %2001 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1997, %1999 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%2000 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb210(%2002: f32, %2003: f32, %2004: f32):
        %2005 = arith.addf %2002, %2003 : f32
        linalg.yield %2005 : f32
      } -> tensor<1x1x1xf32>
      %2006 = tensor.empty() : tensor<1x1x1xf32>
      %2007 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2001 : tensor<1x1x1xf32>) outs(%2006 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb211(%2008: f32, %2009: f32):
        %2010 = math.rsqrt %2008 : f32
        linalg.yield %2010 : f32
      } -> tensor<1x1x1xf32>
      %2011 = tensor.empty() : tensor<1x1x3584xf32>
      %2012 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1870, %2007 : tensor<1x1x3584xf32>, tensor<1x1x1xf32>) outs(%2011 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb212(%2013: f32, %2014: f32, %2015: f32):
        %2016 = arith.mulf %2013, %2014 : f32
        linalg.yield %2016 : f32
      } -> tensor<1x1x3584xf32>
      %2017 = tensor.empty() : tensor<1x1x3584xf32>
      %2018 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2, %2012 : tensor<3584xf32>, tensor<1x1x3584xf32>) outs(%2017 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb213(%2019: f32, %2020: f32, %2021: f32):
        %2022 = arith.mulf %2019, %2020 : f32
        linalg.yield %2022 : f32
      } -> tensor<1x1x3584xf32>
      %2023 = tensor.collapse_shape %2018 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.att_proj"} : tensor<1x1x3584xf32> into tensor<3584xf32>
      %2024 = tensor.expand_shape %2023 [[0 : i64, 1 : i64]] output_shape [1, 3584] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.att_proj"} : tensor<3584xf32> into tensor<1x3584xf32>
      %2025 = tensor.empty() : tensor<3584x4608xf32>
      %2026 = linalg.transpose ins(%4:tensor<4608x3584xf32>) outs(%2025:tensor<3584x4608xf32>) permutation = [1, 0]
      %2027 = tensor.empty() : tensor<1x4608xf32>
      %2028 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
      %2029 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2028 : f32) outs(%2027 : tensor<1x4608xf32>) -> tensor<1x4608xf32>
      %2030 = linalg.matmul {prov.region_id = "matmul_1", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.att_proj", prov.transposed_b = "true"} ins(%2024, %2026 : tensor<1x3584xf32>, tensor<3584x4608xf32>) outs(%2029 : tensor<1x4608xf32>) -> tensor<1x4608xf32>
      %2031 = tensor.empty() : tensor<1x4608xf32>
      %2032 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2030, %5 : tensor<1x4608xf32>, tensor<4608xf32>) outs(%2031 : tensor<1x4608xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.att_proj"} {
      ^bb214(%2033: f32, %2034: f32, %2035: f32):
        %2036 = arith.addf %2033, %2034 : f32
        linalg.yield %2036 : f32
      } -> tensor<1x4608xf32>
      %2037 = tensor.collapse_shape %2032 [[0 : i64, 1 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.att_proj"} : tensor<1x4608xf32> into tensor<4608xf32>
      %2038 = tensor.expand_shape %2037 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 4608] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.att_proj"} : tensor<4608xf32> into tensor<1x1x4608xf32>
      %2039 = "tensor.extract_slice"(%2038) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x4608xf32>) -> tensor<1x1x3584xf32>
      %2040 = "tensor.extract_slice"(%2038) <{static_offsets = array<i64: 0, 0, 3584>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x4608xf32>) -> tensor<1x1x512xf32>
      %2041 = "tensor.extract_slice"(%2038) <{static_offsets = array<i64: 0, 0, 4096>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x4608xf32>) -> tensor<1x1x512xf32>
      %2042 = tensor.collapse_shape %2039 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x3584xf32> into tensor<3584xf32>
      %2043 = tensor.expand_shape %2042 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 28, 128] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<1x1x28x128xf32>
      %2044 = tensor.empty() : tensor<1x28x1x128xf32>
      %2045 = linalg.transpose ins(%2043:tensor<1x1x28x128xf32>) outs(%2044:tensor<1x28x1x128xf32>) permutation = [0, 2, 1, 3]
      %2046 = tensor.collapse_shape %2040 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x512xf32> into tensor<512xf32>
      %2047 = tensor.expand_shape %2046 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 128] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x4x128xf32>
      %2048 = tensor.empty() : tensor<1x4x1x128xf32>
      %2049 = linalg.transpose ins(%2047:tensor<1x1x4x128xf32>) outs(%2048:tensor<1x4x1x128xf32>) permutation = [0, 2, 1, 3]
      %2050 = tensor.collapse_shape %2041 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x512xf32> into tensor<512xf32>
      %2051 = tensor.expand_shape %2050 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 128] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x4x128xf32>
      %2052 = tensor.empty() : tensor<1x4x1x128xf32>
      %2053 = linalg.transpose ins(%2051:tensor<1x1x4x128xf32>) outs(%2052:tensor<1x4x1x128xf32>) permutation = [0, 2, 1, 3]
      %2054 = tensor.collapse_shape %1922 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %2055 = tensor.expand_shape %2054 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 128] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x1x128xf32>
      %2056 = tensor.collapse_shape %1935 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %2057 = tensor.expand_shape %2056 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 128] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x1x128xf32>
      %2058 = tensor.empty() : tensor<1x28x1x128xf32>
      %2059 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2045, %2055 : tensor<1x28x1x128xf32>, tensor<1x1x1x128xf32>) outs(%2058 : tensor<1x28x1x128xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb215(%2060: f32, %2061: f32, %2062: f32):
        %2063 = arith.mulf %2060, %2061 : f32
        linalg.yield %2063 : f32
      } -> tensor<1x28x1x128xf32>
      %2064 = "tensor.extract_slice"(%2045) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 28, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x28x1x128xf32>) -> tensor<1x28x1x64xf32>
      %2065 = "tensor.extract_slice"(%2045) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 28, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x28x1x128xf32>) -> tensor<1x28x1x64xf32>
      %2066 = tensor.empty() : tensor<1x28x1x64xf32>
      %2067 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2065 : tensor<1x28x1x64xf32>) outs(%2066 : tensor<1x28x1x64xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb216(%2068: f32, %2069: f32):
        %2070 = arith.negf %2068 : f32
        linalg.yield %2070 : f32
      } -> tensor<1x28x1x64xf32>
      %2071 = tensor.concat dim(3) %2067, %2064 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x28x1x64xf32>, tensor<1x28x1x64xf32>) -> tensor<1x28x1x128xf32>
      %2072 = tensor.empty() : tensor<1x28x1x128xf32>
      %2073 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2071, %2057 : tensor<1x28x1x128xf32>, tensor<1x1x1x128xf32>) outs(%2072 : tensor<1x28x1x128xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb217(%2074: f32, %2075: f32, %2076: f32):
        %2077 = arith.mulf %2074, %2075 : f32
        linalg.yield %2077 : f32
      } -> tensor<1x28x1x128xf32>
      %2078 = tensor.empty() : tensor<1x28x1x128xf32>
      %2079 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2059, %2073 : tensor<1x28x1x128xf32>, tensor<1x28x1x128xf32>) outs(%2078 : tensor<1x28x1x128xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb218(%2080: f32, %2081: f32, %2082: f32):
        %2083 = arith.addf %2080, %2081 : f32
        linalg.yield %2083 : f32
      } -> tensor<1x28x1x128xf32>
      %2084 = tensor.empty() : tensor<1x4x1x128xf32>
      %2085 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2049, %2055 : tensor<1x4x1x128xf32>, tensor<1x1x1x128xf32>) outs(%2084 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb219(%2086: f32, %2087: f32, %2088: f32):
        %2089 = arith.mulf %2086, %2087 : f32
        linalg.yield %2089 : f32
      } -> tensor<1x4x1x128xf32>
      %2090 = "tensor.extract_slice"(%2049) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x128xf32>) -> tensor<1x4x1x64xf32>
      %2091 = "tensor.extract_slice"(%2049) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x128xf32>) -> tensor<1x4x1x64xf32>
      %2092 = tensor.empty() : tensor<1x4x1x64xf32>
      %2093 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2091 : tensor<1x4x1x64xf32>) outs(%2092 : tensor<1x4x1x64xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb220(%2094: f32, %2095: f32):
        %2096 = arith.negf %2094 : f32
        linalg.yield %2096 : f32
      } -> tensor<1x4x1x64xf32>
      %2097 = tensor.concat dim(3) %2093, %2090 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>) -> tensor<1x4x1x128xf32>
      %2098 = tensor.empty() : tensor<1x4x1x128xf32>
      %2099 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2097, %2057 : tensor<1x4x1x128xf32>, tensor<1x1x1x128xf32>) outs(%2098 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb221(%2100: f32, %2101: f32, %2102: f32):
        %2103 = arith.mulf %2100, %2101 : f32
        linalg.yield %2103 : f32
      } -> tensor<1x4x1x128xf32>
      %2104 = tensor.empty() : tensor<1x4x1x128xf32>
      %2105 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2085, %2099 : tensor<1x4x1x128xf32>, tensor<1x4x1x128xf32>) outs(%2104 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb222(%2106: f32, %2107: f32, %2108: f32):
        %2109 = arith.addf %2106, %2107 : f32
        linalg.yield %2109 : f32
      } -> tensor<1x4x1x128xf32>
      %2110 = tensor.empty() : tensor<1xi64>
      %2111 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%2110 : tensor<1xi64>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
      ^bb223(%2112: i64):
        %2113 = linalg.index 0 : index
        %2114 = arith.index_cast %2113 : index to i64
        %2115 = arith.constant 1 : i64
        %2116 = arith.muli %2114, %2115 : i64
        %2117 = arith.constant 0 : i64
        %2118 = arith.addi %2117, %2116 : i64
        linalg.yield %2118 : i64
      } -> tensor<1xi64>
      %2119 = tensor.empty() : tensor<1xi64>
      %2120 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1863, %2111 : tensor<i64>, tensor<1xi64>) outs(%2119 : tensor<1xi64>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb224(%2121: i64, %2122: i64, %2123: i64):
        %2124 = arith.addi %2121, %2122 : i64
        linalg.yield %2124 : i64
      } -> tensor<1xi64>
      %2125 = "tensor.extract_slice"(%1857) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<4x1x4x16x128xf32>) -> tensor<4x16x128xf32>
      %2126 = func.call @aten_index_put_default_1_wl1(%2125, %2120, %2105) {prov.region_id = "aten_index_put_default_1_0", prov.dispatch_id = "aten_index_put_default_1_0"} : (tensor<4x16x128xf32>, tensor<1xi64>, tensor<1x4x1x128xf32>) -> tensor<1x4x16x128xf32>
      %2127 = "tensor.extract_slice"(%1858) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<4x1x4x16x128xf32>) -> tensor<4x16x128xf32>
      %2128 = func.call @aten_index_put_default_1_wl1(%2127, %2120, %2053) {prov.region_id = "aten_index_put_default_1_1", prov.dispatch_id = "aten_index_put_default_1_1"} : (tensor<4x16x128xf32>, tensor<1xi64>, tensor<1x4x1x128xf32>) -> tensor<1x4x16x128xf32>
      %2129 = "tensor.extract_slice"(%2126) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
      %2130 = "tensor.extract_slice"(%2129) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
      %2131 = tensor.collapse_shape %2130 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x16x128xf32> into tensor<8192xf32>
      %2132 = tensor.expand_shape %2131 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 16, 128] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x4x1x16x128xf32>
      %2133 = "tensor.extract_slice"(%2132) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_9", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
      %2134 = "tensor.extract_slice"(%2133) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_10", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
      %2135 = tensor.empty() : tensor<1x4x7x16x128xf32>
      %2136 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2134 : tensor<1x4x1x16x128xf32>) outs(%2135 : tensor<1x4x7x16x128xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb225(%2137: f32, %2138: f32):
        linalg.yield %2137 : f32
      } -> tensor<1x4x7x16x128xf32>
      %2139 = tensor.collapse_shape %2136 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x7x16x128xf32> into tensor<57344xf32>
      %2140 = tensor.expand_shape %2139 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 16, 128] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<1x28x16x128xf32>
      %2141 = "tensor.extract_slice"(%2128) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_11", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
      %2142 = "tensor.extract_slice"(%2141) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_12", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
      %2143 = tensor.collapse_shape %2142 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x16x128xf32> into tensor<8192xf32>
      %2144 = tensor.expand_shape %2143 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 16, 128] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x4x1x16x128xf32>
      %2145 = "tensor.extract_slice"(%2144) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_13", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
      %2146 = "tensor.extract_slice"(%2145) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_14", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
      %2147 = tensor.empty() : tensor<1x4x7x16x128xf32>
      %2148 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2146 : tensor<1x4x1x16x128xf32>) outs(%2147 : tensor<1x4x7x16x128xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb226(%2149: f32, %2150: f32):
        linalg.yield %2149 : f32
      } -> tensor<1x4x7x16x128xf32>
      %2151 = tensor.collapse_shape %2148 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x7x16x128xf32> into tensor<57344xf32>
      %2152 = tensor.expand_shape %2151 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 16, 128] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<1x28x16x128xf32>
      %2153 = tensor.empty() : tensor<1x28x128x16xf32>
      %2154 = linalg.transpose ins(%2140:tensor<1x28x16x128xf32>) outs(%2153:tensor<1x28x128x16xf32>) permutation = [0, 1, 3, 2]
      %2155 = tensor.empty() : tensor<1x28x1x128xf32>
      %2156 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2079 : tensor<1x28x1x128xf32>) outs(%2155 : tensor<1x28x1x128xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb227(%2157: f32, %2158: f32):
        linalg.yield %2157 : f32
      } -> tensor<1x28x1x128xf32>
      %2159 = tensor.collapse_shape %2156 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x1x128xf32> into tensor<3584xf32>
      %2160 = tensor.expand_shape %2159 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 1, 128] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<28x1x128xf32>
      %2161 = tensor.empty() : tensor<1x28x128x16xf32>
      %2162 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2154 : tensor<1x28x128x16xf32>) outs(%2161 : tensor<1x28x128x16xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb228(%2163: f32, %2164: f32):
        linalg.yield %2163 : f32
      } -> tensor<1x28x128x16xf32>
      %2165 = tensor.collapse_shape %2162 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x128x16xf32> into tensor<57344xf32>
      %2166 = tensor.expand_shape %2165 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 128, 16] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<28x128x16xf32>
      %2167 = arith.constant {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %2168 = tensor.splat %2167 {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<28x1x16xf32>
      %2169 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2160, %2166 : tensor<28x1x128xf32>, tensor<28x128x16xf32>) outs(%2168 : tensor<28x1x16xf32>) attrs =  {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
      ^bb229(%2170: f32, %2171: f32, %2172: f32):
        %2173 = arith.mulf %2170, %2171 : f32
        %2174 = arith.addf %2172, %2173 : f32
        linalg.yield %2174 : f32
      } -> tensor<28x1x16xf32>
      %2175 = tensor.collapse_shape %2169 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28x1x16xf32> into tensor<448xf32>
      %2176 = tensor.expand_shape %2175 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 1, 16] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<448xf32> into tensor<1x28x1x16xf32>
      %2177 = arith.constant {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.0883883461 : f32
      %2178 = tensor.splat %2177 {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x1x16xf32>
      %2179 = tensor.empty() : tensor<1x28x1x16xf32>
      %2180 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2176, %2178 : tensor<1x28x1x16xf32>, tensor<1x28x1x16xf32>) outs(%2179 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb230(%2181: f32, %2182: f32, %2183: f32):
        %2184 = arith.mulf %2181, %2182 : f32
        linalg.yield %2184 : f32
      } -> tensor<1x28x1x16xf32>
      %2185 = tensor.empty() : tensor<1x1x1x16xi1>
      %2186 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1975 : tensor<1x1x1x16xi1>) outs(%2185 : tensor<1x1x1x16xi1>) attrs =  {prov.region_id = "bitwise_0", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
      ^bb231(%2187: i1, %2188: i1):
        %2189 = arith.constant true
        %2190 = arith.xori %2187, %2189 : i1
        linalg.yield %2190 : i1
      } -> tensor<1x1x1x16xi1>
      %2191 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %2192 = tensor.splat %2191 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
      %2193 = tensor.empty() : tensor<1x28x1x16xf32>
      %2194 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2186, %2192, %2180 : tensor<1x1x1x16xi1>, tensor<f32>, tensor<1x28x1x16xf32>) outs(%2193 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32"} {
      ^bb232(%2195: i1, %2196: f32, %2197: f32, %2198: f32):
        %2199 = arith.select %2195, %2196, %2197 : f32
        linalg.yield %2199 : f32
      } -> tensor<1x28x1x16xf32>
      %2200 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %2201 = tensor.splat %2200 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x1xf32>
      %2202 = linalg.reduce ins(%2194:tensor<1x28x1x16xf32>) outs(%2201:tensor<1x28x1xf32>) dimensions = [3]
      (%2203: f32, %2204: f32) {
        %2205 = arith.maximumf %2203, %2204 : f32
        linalg.yield %2205 : f32
      }
      %2206 = tensor.collapse_shape %2202 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x1xf32> into tensor<28xf32>
      %2207 = tensor.expand_shape %2206 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 1, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<28xf32> into tensor<1x28x1x1xf32>
      %2208 = tensor.empty() : tensor<1x28x1x16xf32>
      %2209 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2194, %2207 : tensor<1x28x1x16xf32>, tensor<1x28x1x1xf32>) outs(%2208 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb233(%2210: f32, %2211: f32, %2212: f32):
        %2213 = arith.subf %2210, %2211 : f32
        linalg.yield %2213 : f32
      } -> tensor<1x28x1x16xf32>
      %2214 = tensor.empty() : tensor<1x28x1x16xf32>
      %2215 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2209 : tensor<1x28x1x16xf32>) outs(%2214 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb234(%2216: f32, %2217: f32):
        %2218 = math.exp %2216 : f32
        linalg.yield %2218 : f32
      } -> tensor<1x28x1x16xf32>
      %2219 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %2220 = tensor.splat %2219 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x1xf32>
      %2221 = linalg.reduce ins(%2215:tensor<1x28x1x16xf32>) outs(%2220:tensor<1x28x1xf32>) dimensions = [3]
      (%2222: f32, %2223: f32) {
        %2224 = arith.addf %2222, %2223 : f32
        linalg.yield %2224 : f32
      }
      %2225 = tensor.collapse_shape %2221 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x1xf32> into tensor<28xf32>
      %2226 = tensor.expand_shape %2225 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 1, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<28xf32> into tensor<1x28x1x1xf32>
      %2227 = tensor.empty() : tensor<1x28x1x16xf32>
      %2228 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2215, %2226 : tensor<1x28x1x16xf32>, tensor<1x28x1x1xf32>) outs(%2227 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb235(%2229: f32, %2230: f32, %2231: f32):
        %2232 = arith.divf %2229, %2230 : f32
        linalg.yield %2232 : f32
      } -> tensor<1x28x1x16xf32>
      %2233 = tensor.empty() : tensor<1x28x1x16xf32>
      %2234 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2228 : tensor<1x28x1x16xf32>) outs(%2233 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb236(%2235: f32, %2236: f32):
        linalg.yield %2235 : f32
      } -> tensor<1x28x1x16xf32>
      %2237 = tensor.collapse_shape %2234 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x1x16xf32> into tensor<448xf32>
      %2238 = tensor.expand_shape %2237 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 1, 16] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<448xf32> into tensor<28x1x16xf32>
      %2239 = tensor.empty() : tensor<1x28x16x128xf32>
      %2240 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2152 : tensor<1x28x16x128xf32>) outs(%2239 : tensor<1x28x16x128xf32>) attrs =  {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb237(%2241: f32, %2242: f32):
        linalg.yield %2241 : f32
      } -> tensor<1x28x16x128xf32>
      %2243 = tensor.collapse_shape %2240 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x16x128xf32> into tensor<57344xf32>
      %2244 = tensor.expand_shape %2243 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 16, 128] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<28x16x128xf32>
      %2245 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %2246 = tensor.splat %2245 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<28x1x128xf32>
      %2247 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2238, %2244 : tensor<28x1x16xf32>, tensor<28x16x128xf32>) outs(%2246 : tensor<28x1x128xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
      ^bb238(%2248: f32, %2249: f32, %2250: f32):
        %2251 = arith.mulf %2248, %2249 : f32
        %2252 = arith.addf %2250, %2251 : f32
        linalg.yield %2252 : f32
      } -> tensor<28x1x128xf32>
      %2253 = tensor.collapse_shape %2247 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28x1x128xf32> into tensor<3584xf32>
      %2254 = tensor.expand_shape %2253 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 1, 128] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<1x28x1x128xf32>
      %2255 = tensor.empty() : tensor<1x1x28x128xf32>
      %2256 = linalg.transpose ins(%2254:tensor<1x28x1x128xf32>) outs(%2255:tensor<1x1x28x128xf32>) permutation = [0, 2, 1, 3]
      %2257 = tensor.collapse_shape %2256 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x28x128xf32> into tensor<3584xf32>
      %2258 = tensor.expand_shape %2257 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 3584] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<1x1x3584xf32>
      %2259 = tensor.empty() : tensor<3584x3584xf32>
      %2260 = linalg.transpose ins(%6:tensor<3584x3584xf32>) outs(%2259:tensor<3584x3584xf32>) permutation = [1, 0]
      %2261 = tensor.collapse_shape %2258 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.attn_out"} : tensor<1x1x3584xf32> into tensor<3584xf32>
      %2262 = tensor.expand_shape %2261 [[0 : i64, 1 : i64]] output_shape [1, 3584] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.attn_out"} : tensor<3584xf32> into tensor<1x3584xf32>
      %2263 = tensor.empty() : tensor<1x3584xf32>
      %2264 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
      %2265 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2264 : f32) outs(%2263 : tensor<1x3584xf32>) -> tensor<1x3584xf32>
      %2266 = linalg.matmul {prov.region_id = "matmul_4", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.attn_out", prov.transposed_b = "true"} ins(%2262, %2260 : tensor<1x3584xf32>, tensor<3584x3584xf32>) outs(%2265 : tensor<1x3584xf32>) -> tensor<1x3584xf32>
      %2267 = tensor.collapse_shape %2266 [[0 : i64, 1 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.attn_out"} : tensor<1x3584xf32> into tensor<3584xf32>
      %2268 = tensor.expand_shape %2267 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 3584] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.self_attn.attn_out"} : tensor<3584xf32> into tensor<1x1x3584xf32>
      %2269 = tensor.empty() : tensor<1x1x3584xf32>
      %2270 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1870, %2268 : tensor<1x1x3584xf32>, tensor<1x1x3584xf32>) outs(%2269 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb239(%2271: f32, %2272: f32, %2273: f32):
        %2274 = arith.addf %2271, %2272 : f32
        linalg.yield %2274 : f32
      } -> tensor<1x1x3584xf32>
      %2275 = tensor.empty() : tensor<1x1x3584xf32>
      %2276 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2270 : tensor<1x1x3584xf32>) outs(%2275 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb240(%2277: f32, %2278: f32):
        %2279 = arith.constant 2.000000e+00 : f32
        %2280 = math.powf %2277, %2279 : f32
        linalg.yield %2280 : f32
      } -> tensor<1x1x3584xf32>
      %2281 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %2282 = tensor.splat %2281 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %2283 = linalg.reduce ins(%2276:tensor<1x1x3584xf32>) outs(%2282:tensor<1x1xf32>) dimensions = [2]
      (%2284: f32, %2285: f32) {
        %2286 = arith.addf %2284, %2285 : f32
        linalg.yield %2286 : f32
      }
      %2287 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
      %2288 = tensor.splat %2287 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %2289 = tensor.empty() : tensor<1x1xf32>
      %2290 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2283, %2288 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%2289 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb241(%2291: f32, %2292: f32, %2293: f32):
        %2294 = arith.divf %2291, %2292 : f32
        linalg.yield %2294 : f32
      } -> tensor<1x1xf32>
      %2295 = tensor.collapse_shape %2290 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %2296 = tensor.expand_shape %2295 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %2297 = arith.constant {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %2298 = tensor.splat %2297 {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %2299 = tensor.empty() : tensor<1x1x1xf32>
      %2300 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2296, %2298 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%2299 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb242(%2301: f32, %2302: f32, %2303: f32):
        %2304 = arith.addf %2301, %2302 : f32
        linalg.yield %2304 : f32
      } -> tensor<1x1x1xf32>
      %2305 = tensor.empty() : tensor<1x1x1xf32>
      %2306 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2300 : tensor<1x1x1xf32>) outs(%2305 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb243(%2307: f32, %2308: f32):
        %2309 = math.rsqrt %2307 : f32
        linalg.yield %2309 : f32
      } -> tensor<1x1x1xf32>
      %2310 = tensor.empty() : tensor<1x1x3584xf32>
      %2311 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2270, %2306 : tensor<1x1x3584xf32>, tensor<1x1x1xf32>) outs(%2310 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb244(%2312: f32, %2313: f32, %2314: f32):
        %2315 = arith.mulf %2312, %2313 : f32
        linalg.yield %2315 : f32
      } -> tensor<1x1x3584xf32>
      %2316 = tensor.empty() : tensor<1x1x3584xf32>
      %2317 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %2311 : tensor<3584xf32>, tensor<1x1x3584xf32>) outs(%2316 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb245(%2318: f32, %2319: f32, %2320: f32):
        %2321 = arith.mulf %2318, %2319 : f32
        linalg.yield %2321 : f32
      } -> tensor<1x1x3584xf32>
      %2322 = tensor.empty() : tensor<3584x37888xf32>
      %2323 = linalg.transpose ins(%7:tensor<37888x3584xf32>) outs(%2322:tensor<3584x37888xf32>) permutation = [1, 0]
      %2324 = tensor.collapse_shape %2317 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_proj"} : tensor<1x1x3584xf32> into tensor<3584xf32>
      %2325 = tensor.expand_shape %2324 [[0 : i64, 1 : i64]] output_shape [1, 3584] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_proj"} : tensor<3584xf32> into tensor<1x3584xf32>
      %2326 = tensor.empty() : tensor<1x37888xf32>
      %2327 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
      %2328 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2327 : f32) outs(%2326 : tensor<1x37888xf32>) -> tensor<1x37888xf32>
      %2329 = linalg.matmul {prov.region_id = "matmul_5", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_proj", prov.transposed_b = "true"} ins(%2325, %2323 : tensor<1x3584xf32>, tensor<3584x37888xf32>) outs(%2328 : tensor<1x37888xf32>) -> tensor<1x37888xf32>
      %2330 = tensor.collapse_shape %2329 [[0 : i64, 1 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_proj"} : tensor<1x37888xf32> into tensor<37888xf32>
      %2331 = tensor.expand_shape %2330 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 37888] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_proj"} : tensor<37888xf32> into tensor<1x1x37888xf32>
      %2332 = "tensor.extract_slice"(%2331) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x37888xf32>) -> tensor<1x1x18944xf32>
      %2333 = "tensor.extract_slice"(%2331) <{static_offsets = array<i64: 0, 0, 18944>, static_sizes = array<i64: 1, 1, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x37888xf32>) -> tensor<1x1x18944xf32>
      %2334 = tensor.empty() : tensor<1x1x18944xf32>
      %2335 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2333 : tensor<1x1x18944xf32>) outs(%2334 : tensor<1x1x18944xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.act"} {
      ^bb246(%2336: f32, %2337: f32):
        %2338 = arith.constant 1.000000e+00 : f32
        %2339 = arith.negf %2336 : f32
        %2340 = math.exp %2339 : f32
        %2341 = arith.addf %2338, %2340 : f32
        %2342 = arith.divf %2338, %2341 : f32
        linalg.yield %2342 : f32
      } -> tensor<1x1x18944xf32>
      %2343 = tensor.empty() : tensor<1x1x18944xf32>
      %2344 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2333, %2335 : tensor<1x1x18944xf32>, tensor<1x1x18944xf32>) outs(%2343 : tensor<1x1x18944xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.act"} {
      ^bb247(%2345: f32, %2346: f32, %2347: f32):
        %2348 = arith.mulf %2345, %2346 : f32
        linalg.yield %2348 : f32
      } -> tensor<1x1x18944xf32>
      %2349 = tensor.empty() : tensor<1x1x18944xf32>
      %2350 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2344, %2332 : tensor<1x1x18944xf32>, tensor<1x1x18944xf32>) outs(%2349 : tensor<1x1x18944xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb248(%2351: f32, %2352: f32, %2353: f32):
        %2354 = arith.mulf %2351, %2352 : f32
        linalg.yield %2354 : f32
      } -> tensor<1x1x18944xf32>
      %2355 = tensor.empty() : tensor<18944x3584xf32>
      %2356 = linalg.transpose ins(%8:tensor<3584x18944xf32>) outs(%2355:tensor<18944x3584xf32>) permutation = [1, 0]
      %2357 = tensor.collapse_shape %2350 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_out"} : tensor<1x1x18944xf32> into tensor<18944xf32>
      %2358 = tensor.expand_shape %2357 [[0 : i64, 1 : i64]] output_shape [1, 18944] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_out"} : tensor<18944xf32> into tensor<1x18944xf32>
      %2359 = tensor.empty() : tensor<1x3584xf32>
      %2360 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
      %2361 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2360 : f32) outs(%2359 : tensor<1x3584xf32>) -> tensor<1x3584xf32>
      %2362 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_out", prov.transposed_b = "true"} ins(%2358, %2356 : tensor<1x18944xf32>, tensor<18944x3584xf32>) outs(%2361 : tensor<1x3584xf32>) -> tensor<1x3584xf32>
      %2363 = tensor.collapse_shape %2362 [[0 : i64, 1 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_out"} : tensor<1x3584xf32> into tensor<3584xf32>
      %2364 = tensor.expand_shape %2363 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 3584] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.0.mlp.ff_out"} : tensor<3584xf32> into tensor<1x1x3584xf32>
      %2365 = tensor.empty() : tensor<1x1x3584xf32>
      %2366 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2270, %2364 : tensor<1x1x3584xf32>, tensor<1x1x3584xf32>) outs(%2365 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb249(%2367: f32, %2368: f32, %2369: f32):
        %2370 = arith.addf %2367, %2368 : f32
        linalg.yield %2370 : f32
      } -> tensor<1x1x3584xf32>
      %2371 = tensor.empty() : tensor<1x1x3584xf32>
      %2372 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2366 : tensor<1x1x3584xf32>) outs(%2371 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb250(%2373: f32, %2374: f32):
        %2375 = arith.constant 2.000000e+00 : f32
        %2376 = math.powf %2373, %2375 : f32
        linalg.yield %2376 : f32
      } -> tensor<1x1x3584xf32>
      %2377 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %2378 = tensor.splat %2377 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %2379 = linalg.reduce ins(%2372:tensor<1x1x3584xf32>) outs(%2378:tensor<1x1xf32>) dimensions = [2]
      (%2380: f32, %2381: f32) {
        %2382 = arith.addf %2380, %2381 : f32
        linalg.yield %2382 : f32
      }
      %2383 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
      %2384 = tensor.splat %2383 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %2385 = tensor.empty() : tensor<1x1xf32>
      %2386 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2379, %2384 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%2385 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb251(%2387: f32, %2388: f32, %2389: f32):
        %2390 = arith.divf %2387, %2388 : f32
        linalg.yield %2390 : f32
      } -> tensor<1x1xf32>
      %2391 = tensor.collapse_shape %2386 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %2392 = tensor.expand_shape %2391 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %2393 = arith.constant {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %2394 = tensor.splat %2393 {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %2395 = tensor.empty() : tensor<1x1x1xf32>
      %2396 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2392, %2394 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%2395 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb252(%2397: f32, %2398: f32, %2399: f32):
        %2400 = arith.addf %2397, %2398 : f32
        linalg.yield %2400 : f32
      } -> tensor<1x1x1xf32>
      %2401 = tensor.empty() : tensor<1x1x1xf32>
      %2402 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2396 : tensor<1x1x1xf32>) outs(%2401 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb253(%2403: f32, %2404: f32):
        %2405 = math.rsqrt %2403 : f32
        linalg.yield %2405 : f32
      } -> tensor<1x1x1xf32>
      %2406 = tensor.empty() : tensor<1x1x3584xf32>
      %2407 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2366, %2402 : tensor<1x1x3584xf32>, tensor<1x1x1xf32>) outs(%2406 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb254(%2408: f32, %2409: f32, %2410: f32):
        %2411 = arith.mulf %2408, %2409 : f32
        linalg.yield %2411 : f32
      } -> tensor<1x1x3584xf32>
      %2412 = tensor.empty() : tensor<1x1x3584xf32>
      %2413 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %2407 : tensor<3584xf32>, tensor<1x1x3584xf32>) outs(%2412 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb255(%2414: f32, %2415: f32, %2416: f32):
        %2417 = arith.mulf %2414, %2415 : f32
        linalg.yield %2417 : f32
      } -> tensor<1x1x3584xf32>
      %2418 = tensor.collapse_shape %2413 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.att_proj"} : tensor<1x1x3584xf32> into tensor<3584xf32>
      %2419 = tensor.expand_shape %2418 [[0 : i64, 1 : i64]] output_shape [1, 3584] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.att_proj"} : tensor<3584xf32> into tensor<1x3584xf32>
      %2420 = tensor.empty() : tensor<3584x4608xf32>
      %2421 = linalg.transpose ins(%11:tensor<4608x3584xf32>) outs(%2420:tensor<3584x4608xf32>) permutation = [1, 0]
      %2422 = tensor.empty() : tensor<1x4608xf32>
      %2423 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
      %2424 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2423 : f32) outs(%2422 : tensor<1x4608xf32>) -> tensor<1x4608xf32>
      %2425 = linalg.matmul {prov.region_id = "matmul_7", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.att_proj", prov.transposed_b = "true"} ins(%2419, %2421 : tensor<1x3584xf32>, tensor<3584x4608xf32>) outs(%2424 : tensor<1x4608xf32>) -> tensor<1x4608xf32>
      %2426 = tensor.empty() : tensor<1x4608xf32>
      %2427 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2425, %12 : tensor<1x4608xf32>, tensor<4608xf32>) outs(%2426 : tensor<1x4608xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.att_proj"} {
      ^bb256(%2428: f32, %2429: f32, %2430: f32):
        %2431 = arith.addf %2428, %2429 : f32
        linalg.yield %2431 : f32
      } -> tensor<1x4608xf32>
      %2432 = tensor.collapse_shape %2427 [[0 : i64, 1 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.att_proj"} : tensor<1x4608xf32> into tensor<4608xf32>
      %2433 = tensor.expand_shape %2432 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 4608] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.att_proj"} : tensor<4608xf32> into tensor<1x1x4608xf32>
      %2434 = "tensor.extract_slice"(%2433) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x4608xf32>) -> tensor<1x1x3584xf32>
      %2435 = "tensor.extract_slice"(%2433) <{static_offsets = array<i64: 0, 0, 3584>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x4608xf32>) -> tensor<1x1x512xf32>
      %2436 = "tensor.extract_slice"(%2433) <{static_offsets = array<i64: 0, 0, 4096>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x4608xf32>) -> tensor<1x1x512xf32>
      %2437 = tensor.collapse_shape %2434 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x3584xf32> into tensor<3584xf32>
      %2438 = tensor.expand_shape %2437 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 28, 128] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<1x1x28x128xf32>
      %2439 = tensor.empty() : tensor<1x28x1x128xf32>
      %2440 = linalg.transpose ins(%2438:tensor<1x1x28x128xf32>) outs(%2439:tensor<1x28x1x128xf32>) permutation = [0, 2, 1, 3]
      %2441 = tensor.collapse_shape %2435 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x512xf32> into tensor<512xf32>
      %2442 = tensor.expand_shape %2441 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 128] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x4x128xf32>
      %2443 = tensor.empty() : tensor<1x4x1x128xf32>
      %2444 = linalg.transpose ins(%2442:tensor<1x1x4x128xf32>) outs(%2443:tensor<1x4x1x128xf32>) permutation = [0, 2, 1, 3]
      %2445 = tensor.collapse_shape %2436 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x512xf32> into tensor<512xf32>
      %2446 = tensor.expand_shape %2445 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 128] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x4x128xf32>
      %2447 = tensor.empty() : tensor<1x4x1x128xf32>
      %2448 = linalg.transpose ins(%2446:tensor<1x1x4x128xf32>) outs(%2447:tensor<1x4x1x128xf32>) permutation = [0, 2, 1, 3]
      %2449 = tensor.collapse_shape %1922 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %2450 = tensor.expand_shape %2449 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 128] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x1x128xf32>
      %2451 = tensor.collapse_shape %1935 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %2452 = tensor.expand_shape %2451 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 128] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x1x128xf32>
      %2453 = tensor.empty() : tensor<1x28x1x128xf32>
      %2454 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2440, %2450 : tensor<1x28x1x128xf32>, tensor<1x1x1x128xf32>) outs(%2453 : tensor<1x28x1x128xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb257(%2455: f32, %2456: f32, %2457: f32):
        %2458 = arith.mulf %2455, %2456 : f32
        linalg.yield %2458 : f32
      } -> tensor<1x28x1x128xf32>
      %2459 = "tensor.extract_slice"(%2440) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 28, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_15", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x28x1x128xf32>) -> tensor<1x28x1x64xf32>
      %2460 = "tensor.extract_slice"(%2440) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 28, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_16", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x28x1x128xf32>) -> tensor<1x28x1x64xf32>
      %2461 = tensor.empty() : tensor<1x28x1x64xf32>
      %2462 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2460 : tensor<1x28x1x64xf32>) outs(%2461 : tensor<1x28x1x64xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb258(%2463: f32, %2464: f32):
        %2465 = arith.negf %2463 : f32
        linalg.yield %2465 : f32
      } -> tensor<1x28x1x64xf32>
      %2466 = tensor.concat dim(3) %2462, %2459 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x28x1x64xf32>, tensor<1x28x1x64xf32>) -> tensor<1x28x1x128xf32>
      %2467 = tensor.empty() : tensor<1x28x1x128xf32>
      %2468 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2466, %2452 : tensor<1x28x1x128xf32>, tensor<1x1x1x128xf32>) outs(%2467 : tensor<1x28x1x128xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb259(%2469: f32, %2470: f32, %2471: f32):
        %2472 = arith.mulf %2469, %2470 : f32
        linalg.yield %2472 : f32
      } -> tensor<1x28x1x128xf32>
      %2473 = tensor.empty() : tensor<1x28x1x128xf32>
      %2474 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2454, %2468 : tensor<1x28x1x128xf32>, tensor<1x28x1x128xf32>) outs(%2473 : tensor<1x28x1x128xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb260(%2475: f32, %2476: f32, %2477: f32):
        %2478 = arith.addf %2475, %2476 : f32
        linalg.yield %2478 : f32
      } -> tensor<1x28x1x128xf32>
      %2479 = tensor.empty() : tensor<1x4x1x128xf32>
      %2480 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2444, %2450 : tensor<1x4x1x128xf32>, tensor<1x1x1x128xf32>) outs(%2479 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb261(%2481: f32, %2482: f32, %2483: f32):
        %2484 = arith.mulf %2481, %2482 : f32
        linalg.yield %2484 : f32
      } -> tensor<1x4x1x128xf32>
      %2485 = "tensor.extract_slice"(%2444) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_17", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x128xf32>) -> tensor<1x4x1x64xf32>
      %2486 = "tensor.extract_slice"(%2444) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_18", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x128xf32>) -> tensor<1x4x1x64xf32>
      %2487 = tensor.empty() : tensor<1x4x1x64xf32>
      %2488 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2486 : tensor<1x4x1x64xf32>) outs(%2487 : tensor<1x4x1x64xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb262(%2489: f32, %2490: f32):
        %2491 = arith.negf %2489 : f32
        linalg.yield %2491 : f32
      } -> tensor<1x4x1x64xf32>
      %2492 = tensor.concat dim(3) %2488, %2485 {prov.region_id = "cat_5", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>) -> tensor<1x4x1x128xf32>
      %2493 = tensor.empty() : tensor<1x4x1x128xf32>
      %2494 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2492, %2452 : tensor<1x4x1x128xf32>, tensor<1x1x1x128xf32>) outs(%2493 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb263(%2495: f32, %2496: f32, %2497: f32):
        %2498 = arith.mulf %2495, %2496 : f32
        linalg.yield %2498 : f32
      } -> tensor<1x4x1x128xf32>
      %2499 = tensor.empty() : tensor<1x4x1x128xf32>
      %2500 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2480, %2494 : tensor<1x4x1x128xf32>, tensor<1x4x1x128xf32>) outs(%2499 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb264(%2501: f32, %2502: f32, %2503: f32):
        %2504 = arith.addf %2501, %2502 : f32
        linalg.yield %2504 : f32
      } -> tensor<1x4x1x128xf32>
      %2505 = tensor.empty() : tensor<1xi64>
      %2506 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%2505 : tensor<1xi64>) attrs =  {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
      ^bb265(%2507: i64):
        %2508 = linalg.index 0 : index
        %2509 = arith.index_cast %2508 : index to i64
        %2510 = arith.constant 1 : i64
        %2511 = arith.muli %2509, %2510 : i64
        %2512 = arith.constant 0 : i64
        %2513 = arith.addi %2512, %2511 : i64
        linalg.yield %2513 : i64
      } -> tensor<1xi64>
      %2514 = tensor.empty() : tensor<1xi64>
      %2515 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1863, %2506 : tensor<i64>, tensor<1xi64>) outs(%2514 : tensor<1xi64>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb266(%2516: i64, %2517: i64, %2518: i64):
        %2519 = arith.addi %2516, %2517 : i64
        linalg.yield %2519 : i64
      } -> tensor<1xi64>
      %2520 = "tensor.extract_slice"(%1857) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<4x1x4x16x128xf32>) -> tensor<4x16x128xf32>
      %2521 = func.call @aten_index_put_default_1_wl1(%2520, %2515, %2500) {prov.region_id = "aten_index_put_default_1_2", prov.dispatch_id = "aten_index_put_default_1_2"} : (tensor<4x16x128xf32>, tensor<1xi64>, tensor<1x4x1x128xf32>) -> tensor<1x4x16x128xf32>
      %2522 = "tensor.extract_slice"(%1858) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<4x1x4x16x128xf32>) -> tensor<4x16x128xf32>
      %2523 = func.call @aten_index_put_default_1_wl1(%2522, %2515, %2448) {prov.region_id = "aten_index_put_default_1_3", prov.dispatch_id = "aten_index_put_default_1_3"} : (tensor<4x16x128xf32>, tensor<1xi64>, tensor<1x4x1x128xf32>) -> tensor<1x4x16x128xf32>
      %2524 = "tensor.extract_slice"(%2521) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_19", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
      %2525 = "tensor.extract_slice"(%2524) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_20", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
      %2526 = tensor.collapse_shape %2525 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x16x128xf32> into tensor<8192xf32>
      %2527 = tensor.expand_shape %2526 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 16, 128] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x4x1x16x128xf32>
      %2528 = "tensor.extract_slice"(%2527) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_21", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
      %2529 = "tensor.extract_slice"(%2528) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_22", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
      %2530 = tensor.empty() : tensor<1x4x7x16x128xf32>
      %2531 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2529 : tensor<1x4x1x16x128xf32>) outs(%2530 : tensor<1x4x7x16x128xf32>) attrs =  {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb267(%2532: f32, %2533: f32):
        linalg.yield %2532 : f32
      } -> tensor<1x4x7x16x128xf32>
      %2534 = tensor.collapse_shape %2531 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x7x16x128xf32> into tensor<57344xf32>
      %2535 = tensor.expand_shape %2534 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 16, 128] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<1x28x16x128xf32>
      %2536 = "tensor.extract_slice"(%2523) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_23", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
      %2537 = "tensor.extract_slice"(%2536) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_24", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
      %2538 = tensor.collapse_shape %2537 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x16x128xf32> into tensor<8192xf32>
      %2539 = tensor.expand_shape %2538 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 16, 128] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x4x1x16x128xf32>
      %2540 = "tensor.extract_slice"(%2539) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_25", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
      %2541 = "tensor.extract_slice"(%2540) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_26", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
      %2542 = tensor.empty() : tensor<1x4x7x16x128xf32>
      %2543 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2541 : tensor<1x4x1x16x128xf32>) outs(%2542 : tensor<1x4x7x16x128xf32>) attrs =  {prov.region_id = "expand_10", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb268(%2544: f32, %2545: f32):
        linalg.yield %2544 : f32
      } -> tensor<1x4x7x16x128xf32>
      %2546 = tensor.collapse_shape %2543 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x7x16x128xf32> into tensor<57344xf32>
      %2547 = tensor.expand_shape %2546 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 16, 128] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<1x28x16x128xf32>
      %2548 = tensor.empty() : tensor<1x28x128x16xf32>
      %2549 = linalg.transpose ins(%2535:tensor<1x28x16x128xf32>) outs(%2548:tensor<1x28x128x16xf32>) permutation = [0, 1, 3, 2]
      %2550 = tensor.empty() : tensor<1x28x1x128xf32>
      %2551 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2474 : tensor<1x28x1x128xf32>) outs(%2550 : tensor<1x28x1x128xf32>) attrs =  {prov.region_id = "expand_11", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb269(%2552: f32, %2553: f32):
        linalg.yield %2552 : f32
      } -> tensor<1x28x1x128xf32>
      %2554 = tensor.collapse_shape %2551 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x1x128xf32> into tensor<3584xf32>
      %2555 = tensor.expand_shape %2554 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 1, 128] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<28x1x128xf32>
      %2556 = tensor.empty() : tensor<1x28x128x16xf32>
      %2557 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2549 : tensor<1x28x128x16xf32>) outs(%2556 : tensor<1x28x128x16xf32>) attrs =  {prov.region_id = "expand_12", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb270(%2558: f32, %2559: f32):
        linalg.yield %2558 : f32
      } -> tensor<1x28x128x16xf32>
      %2560 = tensor.collapse_shape %2557 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x128x16xf32> into tensor<57344xf32>
      %2561 = tensor.expand_shape %2560 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 128, 16] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<28x128x16xf32>
      %2562 = arith.constant {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %2563 = tensor.splat %2562 {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<28x1x16xf32>
      %2564 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2555, %2561 : tensor<28x1x128xf32>, tensor<28x128x16xf32>) outs(%2563 : tensor<28x1x16xf32>) attrs =  {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
      ^bb271(%2565: f32, %2566: f32, %2567: f32):
        %2568 = arith.mulf %2565, %2566 : f32
        %2569 = arith.addf %2567, %2568 : f32
        linalg.yield %2569 : f32
      } -> tensor<28x1x16xf32>
      %2570 = tensor.collapse_shape %2564 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28x1x16xf32> into tensor<448xf32>
      %2571 = tensor.expand_shape %2570 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 1, 16] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<448xf32> into tensor<1x28x1x16xf32>
      %2572 = arith.constant {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.0883883461 : f32
      %2573 = tensor.splat %2572 {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x1x16xf32>
      %2574 = tensor.empty() : tensor<1x28x1x16xf32>
      %2575 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2571, %2573 : tensor<1x28x1x16xf32>, tensor<1x28x1x16xf32>) outs(%2574 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb272(%2576: f32, %2577: f32, %2578: f32):
        %2579 = arith.mulf %2576, %2577 : f32
        linalg.yield %2579 : f32
      } -> tensor<1x28x1x16xf32>
      %2580 = tensor.empty() : tensor<1x1x1x16xi1>
      %2581 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1975 : tensor<1x1x1x16xi1>) outs(%2580 : tensor<1x1x1x16xi1>) attrs =  {prov.region_id = "bitwise_1", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
      ^bb273(%2582: i1, %2583: i1):
        %2584 = arith.constant true
        %2585 = arith.xori %2582, %2584 : i1
        linalg.yield %2585 : i1
      } -> tensor<1x1x1x16xi1>
      %2586 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %2587 = tensor.splat %2586 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
      %2588 = tensor.empty() : tensor<1x28x1x16xf32>
      %2589 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2581, %2587, %2575 : tensor<1x1x1x16xi1>, tensor<f32>, tensor<1x28x1x16xf32>) outs(%2588 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32"} {
      ^bb274(%2590: i1, %2591: f32, %2592: f32, %2593: f32):
        %2594 = arith.select %2590, %2591, %2592 : f32
        linalg.yield %2594 : f32
      } -> tensor<1x28x1x16xf32>
      %2595 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %2596 = tensor.splat %2595 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x1xf32>
      %2597 = linalg.reduce ins(%2589:tensor<1x28x1x16xf32>) outs(%2596:tensor<1x28x1xf32>) dimensions = [3]
      (%2598: f32, %2599: f32) {
        %2600 = arith.maximumf %2598, %2599 : f32
        linalg.yield %2600 : f32
      }
      %2601 = tensor.collapse_shape %2597 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x1xf32> into tensor<28xf32>
      %2602 = tensor.expand_shape %2601 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 1, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<28xf32> into tensor<1x28x1x1xf32>
      %2603 = tensor.empty() : tensor<1x28x1x16xf32>
      %2604 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2589, %2602 : tensor<1x28x1x16xf32>, tensor<1x28x1x1xf32>) outs(%2603 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb275(%2605: f32, %2606: f32, %2607: f32):
        %2608 = arith.subf %2605, %2606 : f32
        linalg.yield %2608 : f32
      } -> tensor<1x28x1x16xf32>
      %2609 = tensor.empty() : tensor<1x28x1x16xf32>
      %2610 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2604 : tensor<1x28x1x16xf32>) outs(%2609 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb276(%2611: f32, %2612: f32):
        %2613 = math.exp %2611 : f32
        linalg.yield %2613 : f32
      } -> tensor<1x28x1x16xf32>
      %2614 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %2615 = tensor.splat %2614 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x1xf32>
      %2616 = linalg.reduce ins(%2610:tensor<1x28x1x16xf32>) outs(%2615:tensor<1x28x1xf32>) dimensions = [3]
      (%2617: f32, %2618: f32) {
        %2619 = arith.addf %2617, %2618 : f32
        linalg.yield %2619 : f32
      }
      %2620 = tensor.collapse_shape %2616 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x1xf32> into tensor<28xf32>
      %2621 = tensor.expand_shape %2620 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 1, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<28xf32> into tensor<1x28x1x1xf32>
      %2622 = tensor.empty() : tensor<1x28x1x16xf32>
      %2623 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2610, %2621 : tensor<1x28x1x16xf32>, tensor<1x28x1x1xf32>) outs(%2622 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb277(%2624: f32, %2625: f32, %2626: f32):
        %2627 = arith.divf %2624, %2625 : f32
        linalg.yield %2627 : f32
      } -> tensor<1x28x1x16xf32>
      %2628 = tensor.empty() : tensor<1x28x1x16xf32>
      %2629 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2623 : tensor<1x28x1x16xf32>) outs(%2628 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "expand_13", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb278(%2630: f32, %2631: f32):
        linalg.yield %2630 : f32
      } -> tensor<1x28x1x16xf32>
      %2632 = tensor.collapse_shape %2629 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x1x16xf32> into tensor<448xf32>
      %2633 = tensor.expand_shape %2632 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 1, 16] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<448xf32> into tensor<28x1x16xf32>
      %2634 = tensor.empty() : tensor<1x28x16x128xf32>
      %2635 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2547 : tensor<1x28x16x128xf32>) outs(%2634 : tensor<1x28x16x128xf32>) attrs =  {prov.region_id = "expand_14", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb279(%2636: f32, %2637: f32):
        linalg.yield %2636 : f32
      } -> tensor<1x28x16x128xf32>
      %2638 = tensor.collapse_shape %2635 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x16x128xf32> into tensor<57344xf32>
      %2639 = tensor.expand_shape %2638 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 16, 128] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<28x16x128xf32>
      %2640 = arith.constant {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %2641 = tensor.splat %2640 {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<28x1x128xf32>
      %2642 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2633, %2639 : tensor<28x1x16xf32>, tensor<28x16x128xf32>) outs(%2641 : tensor<28x1x128xf32>) attrs =  {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
      ^bb280(%2643: f32, %2644: f32, %2645: f32):
        %2646 = arith.mulf %2643, %2644 : f32
        %2647 = arith.addf %2645, %2646 : f32
        linalg.yield %2647 : f32
      } -> tensor<28x1x128xf32>
      %2648 = tensor.collapse_shape %2642 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28x1x128xf32> into tensor<3584xf32>
      %2649 = tensor.expand_shape %2648 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 1, 128] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<1x28x1x128xf32>
      %2650 = tensor.empty() : tensor<1x1x28x128xf32>
      %2651 = linalg.transpose ins(%2649:tensor<1x28x1x128xf32>) outs(%2650:tensor<1x1x28x128xf32>) permutation = [0, 2, 1, 3]
      %2652 = tensor.collapse_shape %2651 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x28x128xf32> into tensor<3584xf32>
      %2653 = tensor.expand_shape %2652 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 3584] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<1x1x3584xf32>
      %2654 = tensor.empty() : tensor<3584x3584xf32>
      %2655 = linalg.transpose ins(%13:tensor<3584x3584xf32>) outs(%2654:tensor<3584x3584xf32>) permutation = [1, 0]
      %2656 = tensor.collapse_shape %2653 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.attn_out"} : tensor<1x1x3584xf32> into tensor<3584xf32>
      %2657 = tensor.expand_shape %2656 [[0 : i64, 1 : i64]] output_shape [1, 3584] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.attn_out"} : tensor<3584xf32> into tensor<1x3584xf32>
      %2658 = tensor.empty() : tensor<1x3584xf32>
      %2659 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
      %2660 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2659 : f32) outs(%2658 : tensor<1x3584xf32>) -> tensor<1x3584xf32>
      %2661 = linalg.matmul {prov.region_id = "matmul_10", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.attn_out", prov.transposed_b = "true"} ins(%2657, %2655 : tensor<1x3584xf32>, tensor<3584x3584xf32>) outs(%2660 : tensor<1x3584xf32>) -> tensor<1x3584xf32>
      %2662 = tensor.collapse_shape %2661 [[0 : i64, 1 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.attn_out"} : tensor<1x3584xf32> into tensor<3584xf32>
      %2663 = tensor.expand_shape %2662 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 3584] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.self_attn.attn_out"} : tensor<3584xf32> into tensor<1x1x3584xf32>
      %2664 = tensor.empty() : tensor<1x1x3584xf32>
      %2665 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2366, %2663 : tensor<1x1x3584xf32>, tensor<1x1x3584xf32>) outs(%2664 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb281(%2666: f32, %2667: f32, %2668: f32):
        %2669 = arith.addf %2666, %2667 : f32
        linalg.yield %2669 : f32
      } -> tensor<1x1x3584xf32>
      %2670 = tensor.empty() : tensor<1x1x3584xf32>
      %2671 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2665 : tensor<1x1x3584xf32>) outs(%2670 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb282(%2672: f32, %2673: f32):
        %2674 = arith.constant 2.000000e+00 : f32
        %2675 = math.powf %2672, %2674 : f32
        linalg.yield %2675 : f32
      } -> tensor<1x1x3584xf32>
      %2676 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %2677 = tensor.splat %2676 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %2678 = linalg.reduce ins(%2671:tensor<1x1x3584xf32>) outs(%2677:tensor<1x1xf32>) dimensions = [2]
      (%2679: f32, %2680: f32) {
        %2681 = arith.addf %2679, %2680 : f32
        linalg.yield %2681 : f32
      }
      %2682 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
      %2683 = tensor.splat %2682 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %2684 = tensor.empty() : tensor<1x1xf32>
      %2685 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2678, %2683 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%2684 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb283(%2686: f32, %2687: f32, %2688: f32):
        %2689 = arith.divf %2686, %2687 : f32
        linalg.yield %2689 : f32
      } -> tensor<1x1xf32>
      %2690 = tensor.collapse_shape %2685 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %2691 = tensor.expand_shape %2690 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %2692 = arith.constant {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %2693 = tensor.splat %2692 {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %2694 = tensor.empty() : tensor<1x1x1xf32>
      %2695 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2691, %2693 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%2694 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb284(%2696: f32, %2697: f32, %2698: f32):
        %2699 = arith.addf %2696, %2697 : f32
        linalg.yield %2699 : f32
      } -> tensor<1x1x1xf32>
      %2700 = tensor.empty() : tensor<1x1x1xf32>
      %2701 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2695 : tensor<1x1x1xf32>) outs(%2700 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb285(%2702: f32, %2703: f32):
        %2704 = math.rsqrt %2702 : f32
        linalg.yield %2704 : f32
      } -> tensor<1x1x1xf32>
      %2705 = tensor.empty() : tensor<1x1x3584xf32>
      %2706 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2665, %2701 : tensor<1x1x3584xf32>, tensor<1x1x1xf32>) outs(%2705 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb286(%2707: f32, %2708: f32, %2709: f32):
        %2710 = arith.mulf %2707, %2708 : f32
        linalg.yield %2710 : f32
      } -> tensor<1x1x3584xf32>
      %2711 = tensor.empty() : tensor<1x1x3584xf32>
      %2712 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %2706 : tensor<3584xf32>, tensor<1x1x3584xf32>) outs(%2711 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb287(%2713: f32, %2714: f32, %2715: f32):
        %2716 = arith.mulf %2713, %2714 : f32
        linalg.yield %2716 : f32
      } -> tensor<1x1x3584xf32>
      %2717 = tensor.empty() : tensor<3584x37888xf32>
      %2718 = linalg.transpose ins(%14:tensor<37888x3584xf32>) outs(%2717:tensor<3584x37888xf32>) permutation = [1, 0]
      %2719 = tensor.collapse_shape %2712 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_proj"} : tensor<1x1x3584xf32> into tensor<3584xf32>
      %2720 = tensor.expand_shape %2719 [[0 : i64, 1 : i64]] output_shape [1, 3584] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_proj"} : tensor<3584xf32> into tensor<1x3584xf32>
      %2721 = tensor.empty() : tensor<1x37888xf32>
      %2722 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
      %2723 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2722 : f32) outs(%2721 : tensor<1x37888xf32>) -> tensor<1x37888xf32>
      %2724 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_proj", prov.transposed_b = "true"} ins(%2720, %2718 : tensor<1x3584xf32>, tensor<3584x37888xf32>) outs(%2723 : tensor<1x37888xf32>) -> tensor<1x37888xf32>
      %2725 = tensor.collapse_shape %2724 [[0 : i64, 1 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_proj"} : tensor<1x37888xf32> into tensor<37888xf32>
      %2726 = tensor.expand_shape %2725 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 37888] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_proj"} : tensor<37888xf32> into tensor<1x1x37888xf32>
      %2727 = "tensor.extract_slice"(%2726) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x37888xf32>) -> tensor<1x1x18944xf32>
      %2728 = "tensor.extract_slice"(%2726) <{static_offsets = array<i64: 0, 0, 18944>, static_sizes = array<i64: 1, 1, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x37888xf32>) -> tensor<1x1x18944xf32>
      %2729 = tensor.empty() : tensor<1x1x18944xf32>
      %2730 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2728 : tensor<1x1x18944xf32>) outs(%2729 : tensor<1x1x18944xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.act"} {
      ^bb288(%2731: f32, %2732: f32):
        %2733 = arith.constant 1.000000e+00 : f32
        %2734 = arith.negf %2731 : f32
        %2735 = math.exp %2734 : f32
        %2736 = arith.addf %2733, %2735 : f32
        %2737 = arith.divf %2733, %2736 : f32
        linalg.yield %2737 : f32
      } -> tensor<1x1x18944xf32>
      %2738 = tensor.empty() : tensor<1x1x18944xf32>
      %2739 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2728, %2730 : tensor<1x1x18944xf32>, tensor<1x1x18944xf32>) outs(%2738 : tensor<1x1x18944xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.act"} {
      ^bb289(%2740: f32, %2741: f32, %2742: f32):
        %2743 = arith.mulf %2740, %2741 : f32
        linalg.yield %2743 : f32
      } -> tensor<1x1x18944xf32>
      %2744 = tensor.empty() : tensor<1x1x18944xf32>
      %2745 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2739, %2727 : tensor<1x1x18944xf32>, tensor<1x1x18944xf32>) outs(%2744 : tensor<1x1x18944xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb290(%2746: f32, %2747: f32, %2748: f32):
        %2749 = arith.mulf %2746, %2747 : f32
        linalg.yield %2749 : f32
      } -> tensor<1x1x18944xf32>
      %2750 = tensor.empty() : tensor<18944x3584xf32>
      %2751 = linalg.transpose ins(%15:tensor<3584x18944xf32>) outs(%2750:tensor<18944x3584xf32>) permutation = [1, 0]
      %2752 = tensor.collapse_shape %2745 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_out"} : tensor<1x1x18944xf32> into tensor<18944xf32>
      %2753 = tensor.expand_shape %2752 [[0 : i64, 1 : i64]] output_shape [1, 18944] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_out"} : tensor<18944xf32> into tensor<1x18944xf32>
      %2754 = tensor.empty() : tensor<1x3584xf32>
      %2755 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
      %2756 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2755 : f32) outs(%2754 : tensor<1x3584xf32>) -> tensor<1x3584xf32>
      %2757 = linalg.matmul {prov.region_id = "matmul_12", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_out", prov.transposed_b = "true"} ins(%2753, %2751 : tensor<1x18944xf32>, tensor<18944x3584xf32>) outs(%2756 : tensor<1x3584xf32>) -> tensor<1x3584xf32>
      %2758 = tensor.collapse_shape %2757 [[0 : i64, 1 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_out"} : tensor<1x3584xf32> into tensor<3584xf32>
      %2759 = tensor.expand_shape %2758 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 3584] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.1.mlp.ff_out"} : tensor<3584xf32> into tensor<1x1x3584xf32>
      %2760 = tensor.empty() : tensor<1x1x3584xf32>
      %2761 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2665, %2759 : tensor<1x1x3584xf32>, tensor<1x1x3584xf32>) outs(%2760 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb291(%2762: f32, %2763: f32, %2764: f32):
        %2765 = arith.addf %2762, %2763 : f32
        linalg.yield %2765 : f32
      } -> tensor<1x1x3584xf32>
      %2766 = tensor.empty() : tensor<1x1x3584xf32>
      %2767 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2761 : tensor<1x1x3584xf32>) outs(%2766 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb292(%2768: f32, %2769: f32):
        %2770 = arith.constant 2.000000e+00 : f32
        %2771 = math.powf %2768, %2770 : f32
        linalg.yield %2771 : f32
      } -> tensor<1x1x3584xf32>
      %2772 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %2773 = tensor.splat %2772 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %2774 = linalg.reduce ins(%2767:tensor<1x1x3584xf32>) outs(%2773:tensor<1x1xf32>) dimensions = [2]
      (%2775: f32, %2776: f32) {
        %2777 = arith.addf %2775, %2776 : f32
        linalg.yield %2777 : f32
      }
      %2778 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
      %2779 = tensor.splat %2778 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %2780 = tensor.empty() : tensor<1x1xf32>
      %2781 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2774, %2779 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%2780 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb293(%2782: f32, %2783: f32, %2784: f32):
        %2785 = arith.divf %2782, %2783 : f32
        linalg.yield %2785 : f32
      } -> tensor<1x1xf32>
      %2786 = tensor.collapse_shape %2781 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %2787 = tensor.expand_shape %2786 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %2788 = arith.constant {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %2789 = tensor.splat %2788 {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %2790 = tensor.empty() : tensor<1x1x1xf32>
      %2791 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2787, %2789 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%2790 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb294(%2792: f32, %2793: f32, %2794: f32):
        %2795 = arith.addf %2792, %2793 : f32
        linalg.yield %2795 : f32
      } -> tensor<1x1x1xf32>
      %2796 = tensor.empty() : tensor<1x1x1xf32>
      %2797 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2791 : tensor<1x1x1xf32>) outs(%2796 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb295(%2798: f32, %2799: f32):
        %2800 = math.rsqrt %2798 : f32
        linalg.yield %2800 : f32
      } -> tensor<1x1x1xf32>
      %2801 = tensor.empty() : tensor<1x1x3584xf32>
      %2802 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2761, %2797 : tensor<1x1x3584xf32>, tensor<1x1x1xf32>) outs(%2801 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb296(%2803: f32, %2804: f32, %2805: f32):
        %2806 = arith.mulf %2803, %2804 : f32
        linalg.yield %2806 : f32
      } -> tensor<1x1x3584xf32>
      %2807 = tensor.empty() : tensor<1x1x3584xf32>
      %2808 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16, %2802 : tensor<3584xf32>, tensor<1x1x3584xf32>) outs(%2807 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb297(%2809: f32, %2810: f32, %2811: f32):
        %2812 = arith.mulf %2809, %2810 : f32
        linalg.yield %2812 : f32
      } -> tensor<1x1x3584xf32>
      %2813 = tensor.collapse_shape %2808 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.att_proj"} : tensor<1x1x3584xf32> into tensor<3584xf32>
      %2814 = tensor.expand_shape %2813 [[0 : i64, 1 : i64]] output_shape [1, 3584] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.att_proj"} : tensor<3584xf32> into tensor<1x3584xf32>
      %2815 = tensor.empty() : tensor<3584x4608xf32>
      %2816 = linalg.transpose ins(%18:tensor<4608x3584xf32>) outs(%2815:tensor<3584x4608xf32>) permutation = [1, 0]
      %2817 = tensor.empty() : tensor<1x4608xf32>
      %2818 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
      %2819 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%2818 : f32) outs(%2817 : tensor<1x4608xf32>) -> tensor<1x4608xf32>
      %2820 = linalg.matmul {prov.region_id = "matmul_13", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.att_proj", prov.transposed_b = "true"} ins(%2814, %2816 : tensor<1x3584xf32>, tensor<3584x4608xf32>) outs(%2819 : tensor<1x4608xf32>) -> tensor<1x4608xf32>
      %2821 = tensor.empty() : tensor<1x4608xf32>
      %2822 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2820, %19 : tensor<1x4608xf32>, tensor<4608xf32>) outs(%2821 : tensor<1x4608xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.att_proj"} {
      ^bb298(%2823: f32, %2824: f32, %2825: f32):
        %2826 = arith.addf %2823, %2824 : f32
        linalg.yield %2826 : f32
      } -> tensor<1x4608xf32>
      %2827 = tensor.collapse_shape %2822 [[0 : i64, 1 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.att_proj"} : tensor<1x4608xf32> into tensor<4608xf32>
      %2828 = tensor.expand_shape %2827 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 4608] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.att_proj"} : tensor<4608xf32> into tensor<1x1x4608xf32>
      %2829 = "tensor.extract_slice"(%2828) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_4", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x4608xf32>) -> tensor<1x1x3584xf32>
      %2830 = "tensor.extract_slice"(%2828) <{static_offsets = array<i64: 0, 0, 3584>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_4", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x4608xf32>) -> tensor<1x1x512xf32>
      %2831 = "tensor.extract_slice"(%2828) <{static_offsets = array<i64: 0, 0, 4096>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_4", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x4608xf32>) -> tensor<1x1x512xf32>
      %2832 = tensor.collapse_shape %2829 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x3584xf32> into tensor<3584xf32>
      %2833 = tensor.expand_shape %2832 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 28, 128] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<1x1x28x128xf32>
      %2834 = tensor.empty() : tensor<1x28x1x128xf32>
      %2835 = linalg.transpose ins(%2833:tensor<1x1x28x128xf32>) outs(%2834:tensor<1x28x1x128xf32>) permutation = [0, 2, 1, 3]
      %2836 = tensor.collapse_shape %2830 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x512xf32> into tensor<512xf32>
      %2837 = tensor.expand_shape %2836 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 128] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x4x128xf32>
      %2838 = tensor.empty() : tensor<1x4x1x128xf32>
      %2839 = linalg.transpose ins(%2837:tensor<1x1x4x128xf32>) outs(%2838:tensor<1x4x1x128xf32>) permutation = [0, 2, 1, 3]
      %2840 = tensor.collapse_shape %2831 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x512xf32> into tensor<512xf32>
      %2841 = tensor.expand_shape %2840 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 128] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x4x128xf32>
      %2842 = tensor.empty() : tensor<1x4x1x128xf32>
      %2843 = linalg.transpose ins(%2841:tensor<1x1x4x128xf32>) outs(%2842:tensor<1x4x1x128xf32>) permutation = [0, 2, 1, 3]
      %2844 = tensor.collapse_shape %1922 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %2845 = tensor.expand_shape %2844 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 128] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x1x128xf32>
      %2846 = tensor.collapse_shape %1935 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %2847 = tensor.expand_shape %2846 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 128] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x1x128xf32>
      %2848 = tensor.empty() : tensor<1x28x1x128xf32>
      %2849 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2835, %2845 : tensor<1x28x1x128xf32>, tensor<1x1x1x128xf32>) outs(%2848 : tensor<1x28x1x128xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb299(%2850: f32, %2851: f32, %2852: f32):
        %2853 = arith.mulf %2850, %2851 : f32
        linalg.yield %2853 : f32
      } -> tensor<1x28x1x128xf32>
      %2854 = "tensor.extract_slice"(%2835) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 28, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_27", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x28x1x128xf32>) -> tensor<1x28x1x64xf32>
      %2855 = "tensor.extract_slice"(%2835) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 28, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_28", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x28x1x128xf32>) -> tensor<1x28x1x64xf32>
      %2856 = tensor.empty() : tensor<1x28x1x64xf32>
      %2857 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2855 : tensor<1x28x1x64xf32>) outs(%2856 : tensor<1x28x1x64xf32>) attrs =  {prov.region_id = "neg_4", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb300(%2858: f32, %2859: f32):
        %2860 = arith.negf %2858 : f32
        linalg.yield %2860 : f32
      } -> tensor<1x28x1x64xf32>
      %2861 = tensor.concat dim(3) %2857, %2854 {prov.region_id = "cat_6", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x28x1x64xf32>, tensor<1x28x1x64xf32>) -> tensor<1x28x1x128xf32>
      %2862 = tensor.empty() : tensor<1x28x1x128xf32>
      %2863 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2861, %2847 : tensor<1x28x1x128xf32>, tensor<1x1x1x128xf32>) outs(%2862 : tensor<1x28x1x128xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb301(%2864: f32, %2865: f32, %2866: f32):
        %2867 = arith.mulf %2864, %2865 : f32
        linalg.yield %2867 : f32
      } -> tensor<1x28x1x128xf32>
      %2868 = tensor.empty() : tensor<1x28x1x128xf32>
      %2869 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2849, %2863 : tensor<1x28x1x128xf32>, tensor<1x28x1x128xf32>) outs(%2868 : tensor<1x28x1x128xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb302(%2870: f32, %2871: f32, %2872: f32):
        %2873 = arith.addf %2870, %2871 : f32
        linalg.yield %2873 : f32
      } -> tensor<1x28x1x128xf32>
      %2874 = tensor.empty() : tensor<1x4x1x128xf32>
      %2875 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2839, %2845 : tensor<1x4x1x128xf32>, tensor<1x1x1x128xf32>) outs(%2874 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb303(%2876: f32, %2877: f32, %2878: f32):
        %2879 = arith.mulf %2876, %2877 : f32
        linalg.yield %2879 : f32
      } -> tensor<1x4x1x128xf32>
      %2880 = "tensor.extract_slice"(%2839) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_29", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x128xf32>) -> tensor<1x4x1x64xf32>
      %2881 = "tensor.extract_slice"(%2839) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_30", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x128xf32>) -> tensor<1x4x1x64xf32>
      %2882 = tensor.empty() : tensor<1x4x1x64xf32>
      %2883 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2881 : tensor<1x4x1x64xf32>) outs(%2882 : tensor<1x4x1x64xf32>) attrs =  {prov.region_id = "neg_5", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb304(%2884: f32, %2885: f32):
        %2886 = arith.negf %2884 : f32
        linalg.yield %2886 : f32
      } -> tensor<1x4x1x64xf32>
      %2887 = tensor.concat dim(3) %2883, %2880 {prov.region_id = "cat_7", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>) -> tensor<1x4x1x128xf32>
      %2888 = tensor.empty() : tensor<1x4x1x128xf32>
      %2889 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2887, %2847 : tensor<1x4x1x128xf32>, tensor<1x1x1x128xf32>) outs(%2888 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb305(%2890: f32, %2891: f32, %2892: f32):
        %2893 = arith.mulf %2890, %2891 : f32
        linalg.yield %2893 : f32
      } -> tensor<1x4x1x128xf32>
      %2894 = tensor.empty() : tensor<1x4x1x128xf32>
      %2895 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2875, %2889 : tensor<1x4x1x128xf32>, tensor<1x4x1x128xf32>) outs(%2894 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb306(%2896: f32, %2897: f32, %2898: f32):
        %2899 = arith.addf %2896, %2897 : f32
        linalg.yield %2899 : f32
      } -> tensor<1x4x1x128xf32>
      %2900 = tensor.empty() : tensor<1xi64>
      %2901 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%2900 : tensor<1xi64>) attrs =  {prov.region_id = "iota_4", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
      ^bb307(%2902: i64):
        %2903 = linalg.index 0 : index
        %2904 = arith.index_cast %2903 : index to i64
        %2905 = arith.constant 1 : i64
        %2906 = arith.muli %2904, %2905 : i64
        %2907 = arith.constant 0 : i64
        %2908 = arith.addi %2907, %2906 : i64
        linalg.yield %2908 : i64
      } -> tensor<1xi64>
      %2909 = tensor.empty() : tensor<1xi64>
      %2910 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1863, %2901 : tensor<i64>, tensor<1xi64>) outs(%2909 : tensor<1xi64>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb308(%2911: i64, %2912: i64, %2913: i64):
        %2914 = arith.addi %2911, %2912 : i64
        linalg.yield %2914 : i64
      } -> tensor<1xi64>
      %2915 = "tensor.extract_slice"(%1857) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<4x1x4x16x128xf32>) -> tensor<4x16x128xf32>
      %2916 = func.call @aten_index_put_default_1_wl1(%2915, %2910, %2895) {prov.region_id = "aten_index_put_default_1_4", prov.dispatch_id = "aten_index_put_default_1_4"} : (tensor<4x16x128xf32>, tensor<1xi64>, tensor<1x4x1x128xf32>) -> tensor<1x4x16x128xf32>
      %2917 = "tensor.extract_slice"(%1858) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<4x1x4x16x128xf32>) -> tensor<4x16x128xf32>
      %2918 = func.call @aten_index_put_default_1_wl1(%2917, %2910, %2843) {prov.region_id = "aten_index_put_default_1_5", prov.dispatch_id = "aten_index_put_default_1_5"} : (tensor<4x16x128xf32>, tensor<1xi64>, tensor<1x4x1x128xf32>) -> tensor<1x4x16x128xf32>
      %2919 = "tensor.extract_slice"(%2916) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_31", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
      %2920 = "tensor.extract_slice"(%2919) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_32", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
      %2921 = tensor.collapse_shape %2920 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x16x128xf32> into tensor<8192xf32>
      %2922 = tensor.expand_shape %2921 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 16, 128] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x4x1x16x128xf32>
      %2923 = "tensor.extract_slice"(%2922) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_33", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
      %2924 = "tensor.extract_slice"(%2923) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_34", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
      %2925 = tensor.empty() : tensor<1x4x7x16x128xf32>
      %2926 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2924 : tensor<1x4x1x16x128xf32>) outs(%2925 : tensor<1x4x7x16x128xf32>) attrs =  {prov.region_id = "expand_15", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb309(%2927: f32, %2928: f32):
        linalg.yield %2927 : f32
      } -> tensor<1x4x7x16x128xf32>
      %2929 = tensor.collapse_shape %2926 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x7x16x128xf32> into tensor<57344xf32>
      %2930 = tensor.expand_shape %2929 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 16, 128] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<1x28x16x128xf32>
      %2931 = "tensor.extract_slice"(%2918) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_35", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
      %2932 = "tensor.extract_slice"(%2931) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_36", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
      %2933 = tensor.collapse_shape %2932 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x16x128xf32> into tensor<8192xf32>
      %2934 = tensor.expand_shape %2933 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 16, 128] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x4x1x16x128xf32>
      %2935 = "tensor.extract_slice"(%2934) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_37", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
      %2936 = "tensor.extract_slice"(%2935) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_38", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
      %2937 = tensor.empty() : tensor<1x4x7x16x128xf32>
      %2938 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2936 : tensor<1x4x1x16x128xf32>) outs(%2937 : tensor<1x4x7x16x128xf32>) attrs =  {prov.region_id = "expand_16", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb310(%2939: f32, %2940: f32):
        linalg.yield %2939 : f32
      } -> tensor<1x4x7x16x128xf32>
      %2941 = tensor.collapse_shape %2938 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x7x16x128xf32> into tensor<57344xf32>
      %2942 = tensor.expand_shape %2941 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 16, 128] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<1x28x16x128xf32>
      %2943 = tensor.empty() : tensor<1x28x128x16xf32>
      %2944 = linalg.transpose ins(%2930:tensor<1x28x16x128xf32>) outs(%2943:tensor<1x28x128x16xf32>) permutation = [0, 1, 3, 2]
      %2945 = tensor.empty() : tensor<1x28x1x128xf32>
      %2946 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2869 : tensor<1x28x1x128xf32>) outs(%2945 : tensor<1x28x1x128xf32>) attrs =  {prov.region_id = "expand_17", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb311(%2947: f32, %2948: f32):
        linalg.yield %2947 : f32
      } -> tensor<1x28x1x128xf32>
      %2949 = tensor.collapse_shape %2946 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x1x128xf32> into tensor<3584xf32>
      %2950 = tensor.expand_shape %2949 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 1, 128] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<28x1x128xf32>
      %2951 = tensor.empty() : tensor<1x28x128x16xf32>
      %2952 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2944 : tensor<1x28x128x16xf32>) outs(%2951 : tensor<1x28x128x16xf32>) attrs =  {prov.region_id = "expand_18", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb312(%2953: f32, %2954: f32):
        linalg.yield %2953 : f32
      } -> tensor<1x28x128x16xf32>
      %2955 = tensor.collapse_shape %2952 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x128x16xf32> into tensor<57344xf32>
      %2956 = tensor.expand_shape %2955 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 128, 16] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<28x128x16xf32>
      %2957 = arith.constant {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %2958 = tensor.splat %2957 {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<28x1x16xf32>
      %2959 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2950, %2956 : tensor<28x1x128xf32>, tensor<28x128x16xf32>) outs(%2958 : tensor<28x1x16xf32>) attrs =  {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
      ^bb313(%2960: f32, %2961: f32, %2962: f32):
        %2963 = arith.mulf %2960, %2961 : f32
        %2964 = arith.addf %2962, %2963 : f32
        linalg.yield %2964 : f32
      } -> tensor<28x1x16xf32>
      %2965 = tensor.collapse_shape %2959 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28x1x16xf32> into tensor<448xf32>
      %2966 = tensor.expand_shape %2965 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 1, 16] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<448xf32> into tensor<1x28x1x16xf32>
      %2967 = arith.constant {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.0883883461 : f32
      %2968 = tensor.splat %2967 {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x1x16xf32>
      %2969 = tensor.empty() : tensor<1x28x1x16xf32>
      %2970 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2966, %2968 : tensor<1x28x1x16xf32>, tensor<1x28x1x16xf32>) outs(%2969 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb314(%2971: f32, %2972: f32, %2973: f32):
        %2974 = arith.mulf %2971, %2972 : f32
        linalg.yield %2974 : f32
      } -> tensor<1x28x1x16xf32>
      %2975 = tensor.empty() : tensor<1x1x1x16xi1>
      %2976 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1975 : tensor<1x1x1x16xi1>) outs(%2975 : tensor<1x1x1x16xi1>) attrs =  {prov.region_id = "bitwise_2", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
      ^bb315(%2977: i1, %2978: i1):
        %2979 = arith.constant true
        %2980 = arith.xori %2977, %2979 : i1
        linalg.yield %2980 : i1
      } -> tensor<1x1x1x16xi1>
      %2981 = arith.constant {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %2982 = tensor.splat %2981 {prov.region_id = "fill_2", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
      %2983 = tensor.empty() : tensor<1x28x1x16xf32>
      %2984 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2976, %2982, %2970 : tensor<1x1x1x16xi1>, tensor<f32>, tensor<1x28x1x16xf32>) outs(%2983 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "select_8", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32"} {
      ^bb316(%2985: i1, %2986: f32, %2987: f32, %2988: f32):
        %2989 = arith.select %2985, %2986, %2987 : f32
        linalg.yield %2989 : f32
      } -> tensor<1x28x1x16xf32>
      %2990 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %2991 = tensor.splat %2990 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x1xf32>
      %2992 = linalg.reduce ins(%2984:tensor<1x28x1x16xf32>) outs(%2991:tensor<1x28x1xf32>) dimensions = [3]
      (%2993: f32, %2994: f32) {
        %2995 = arith.maximumf %2993, %2994 : f32
        linalg.yield %2995 : f32
      }
      %2996 = tensor.collapse_shape %2992 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x1xf32> into tensor<28xf32>
      %2997 = tensor.expand_shape %2996 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 1, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<28xf32> into tensor<1x28x1x1xf32>
      %2998 = tensor.empty() : tensor<1x28x1x16xf32>
      %2999 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2984, %2997 : tensor<1x28x1x16xf32>, tensor<1x28x1x1xf32>) outs(%2998 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb317(%3000: f32, %3001: f32, %3002: f32):
        %3003 = arith.subf %3000, %3001 : f32
        linalg.yield %3003 : f32
      } -> tensor<1x28x1x16xf32>
      %3004 = tensor.empty() : tensor<1x28x1x16xf32>
      %3005 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2999 : tensor<1x28x1x16xf32>) outs(%3004 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb318(%3006: f32, %3007: f32):
        %3008 = math.exp %3006 : f32
        linalg.yield %3008 : f32
      } -> tensor<1x28x1x16xf32>
      %3009 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3010 = tensor.splat %3009 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x1xf32>
      %3011 = linalg.reduce ins(%3005:tensor<1x28x1x16xf32>) outs(%3010:tensor<1x28x1xf32>) dimensions = [3]
      (%3012: f32, %3013: f32) {
        %3014 = arith.addf %3012, %3013 : f32
        linalg.yield %3014 : f32
      }
      %3015 = tensor.collapse_shape %3011 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x1xf32> into tensor<28xf32>
      %3016 = tensor.expand_shape %3015 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 1, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<28xf32> into tensor<1x28x1x1xf32>
      %3017 = tensor.empty() : tensor<1x28x1x16xf32>
      %3018 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3005, %3016 : tensor<1x28x1x16xf32>, tensor<1x28x1x1xf32>) outs(%3017 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb319(%3019: f32, %3020: f32, %3021: f32):
        %3022 = arith.divf %3019, %3020 : f32
        linalg.yield %3022 : f32
      } -> tensor<1x28x1x16xf32>
      %3023 = tensor.empty() : tensor<1x28x1x16xf32>
      %3024 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3018 : tensor<1x28x1x16xf32>) outs(%3023 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "expand_19", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb320(%3025: f32, %3026: f32):
        linalg.yield %3025 : f32
      } -> tensor<1x28x1x16xf32>
      %3027 = tensor.collapse_shape %3024 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x1x16xf32> into tensor<448xf32>
      %3028 = tensor.expand_shape %3027 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 1, 16] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<448xf32> into tensor<28x1x16xf32>
      %3029 = tensor.empty() : tensor<1x28x16x128xf32>
      %3030 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2942 : tensor<1x28x16x128xf32>) outs(%3029 : tensor<1x28x16x128xf32>) attrs =  {prov.region_id = "expand_20", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb321(%3031: f32, %3032: f32):
        linalg.yield %3031 : f32
      } -> tensor<1x28x16x128xf32>
      %3033 = tensor.collapse_shape %3030 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x16x128xf32> into tensor<57344xf32>
      %3034 = tensor.expand_shape %3033 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 16, 128] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<28x16x128xf32>
      %3035 = arith.constant {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3036 = tensor.splat %3035 {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<28x1x128xf32>
      %3037 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3028, %3034 : tensor<28x1x16xf32>, tensor<28x16x128xf32>) outs(%3036 : tensor<28x1x128xf32>) attrs =  {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
      ^bb322(%3038: f32, %3039: f32, %3040: f32):
        %3041 = arith.mulf %3038, %3039 : f32
        %3042 = arith.addf %3040, %3041 : f32
        linalg.yield %3042 : f32
      } -> tensor<28x1x128xf32>
      %3043 = tensor.collapse_shape %3037 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28x1x128xf32> into tensor<3584xf32>
      %3044 = tensor.expand_shape %3043 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 1, 128] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<1x28x1x128xf32>
      %3045 = tensor.empty() : tensor<1x1x28x128xf32>
      %3046 = linalg.transpose ins(%3044:tensor<1x28x1x128xf32>) outs(%3045:tensor<1x1x28x128xf32>) permutation = [0, 2, 1, 3]
      %3047 = tensor.collapse_shape %3046 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x28x128xf32> into tensor<3584xf32>
      %3048 = tensor.expand_shape %3047 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 3584] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<1x1x3584xf32>
      %3049 = tensor.empty() : tensor<3584x3584xf32>
      %3050 = linalg.transpose ins(%20:tensor<3584x3584xf32>) outs(%3049:tensor<3584x3584xf32>) permutation = [1, 0]
      %3051 = tensor.collapse_shape %3048 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.attn_out"} : tensor<1x1x3584xf32> into tensor<3584xf32>
      %3052 = tensor.expand_shape %3051 [[0 : i64, 1 : i64]] output_shape [1, 3584] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.attn_out"} : tensor<3584xf32> into tensor<1x3584xf32>
      %3053 = tensor.empty() : tensor<1x3584xf32>
      %3054 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
      %3055 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3054 : f32) outs(%3053 : tensor<1x3584xf32>) -> tensor<1x3584xf32>
      %3056 = linalg.matmul {prov.region_id = "matmul_16", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.attn_out", prov.transposed_b = "true"} ins(%3052, %3050 : tensor<1x3584xf32>, tensor<3584x3584xf32>) outs(%3055 : tensor<1x3584xf32>) -> tensor<1x3584xf32>
      %3057 = tensor.collapse_shape %3056 [[0 : i64, 1 : i64]] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.attn_out"} : tensor<1x3584xf32> into tensor<3584xf32>
      %3058 = tensor.expand_shape %3057 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 3584] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.self_attn.attn_out"} : tensor<3584xf32> into tensor<1x1x3584xf32>
      %3059 = tensor.empty() : tensor<1x1x3584xf32>
      %3060 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2761, %3058 : tensor<1x1x3584xf32>, tensor<1x1x3584xf32>) outs(%3059 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb323(%3061: f32, %3062: f32, %3063: f32):
        %3064 = arith.addf %3061, %3062 : f32
        linalg.yield %3064 : f32
      } -> tensor<1x1x3584xf32>
      %3065 = tensor.empty() : tensor<1x1x3584xf32>
      %3066 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3060 : tensor<1x1x3584xf32>) outs(%3065 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb324(%3067: f32, %3068: f32):
        %3069 = arith.constant 2.000000e+00 : f32
        %3070 = math.powf %3067, %3069 : f32
        linalg.yield %3070 : f32
      } -> tensor<1x1x3584xf32>
      %3071 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3072 = tensor.splat %3071 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3073 = linalg.reduce ins(%3066:tensor<1x1x3584xf32>) outs(%3072:tensor<1x1xf32>) dimensions = [2]
      (%3074: f32, %3075: f32) {
        %3076 = arith.addf %3074, %3075 : f32
        linalg.yield %3076 : f32
      }
      %3077 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
      %3078 = tensor.splat %3077 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3079 = tensor.empty() : tensor<1x1xf32>
      %3080 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3073, %3078 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%3079 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb325(%3081: f32, %3082: f32, %3083: f32):
        %3084 = arith.divf %3081, %3082 : f32
        linalg.yield %3084 : f32
      } -> tensor<1x1xf32>
      %3085 = tensor.collapse_shape %3080 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %3086 = tensor.expand_shape %3085 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %3087 = arith.constant {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %3088 = tensor.splat %3087 {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %3089 = tensor.empty() : tensor<1x1x1xf32>
      %3090 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3086, %3088 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%3089 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb326(%3091: f32, %3092: f32, %3093: f32):
        %3094 = arith.addf %3091, %3092 : f32
        linalg.yield %3094 : f32
      } -> tensor<1x1x1xf32>
      %3095 = tensor.empty() : tensor<1x1x1xf32>
      %3096 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3090 : tensor<1x1x1xf32>) outs(%3095 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_5", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb327(%3097: f32, %3098: f32):
        %3099 = math.rsqrt %3097 : f32
        linalg.yield %3099 : f32
      } -> tensor<1x1x1xf32>
      %3100 = tensor.empty() : tensor<1x1x3584xf32>
      %3101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3060, %3096 : tensor<1x1x3584xf32>, tensor<1x1x1xf32>) outs(%3100 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb328(%3102: f32, %3103: f32, %3104: f32):
        %3105 = arith.mulf %3102, %3103 : f32
        linalg.yield %3105 : f32
      } -> tensor<1x1x3584xf32>
      %3106 = tensor.empty() : tensor<1x1x3584xf32>
      %3107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %3101 : tensor<3584xf32>, tensor<1x1x3584xf32>) outs(%3106 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb329(%3108: f32, %3109: f32, %3110: f32):
        %3111 = arith.mulf %3108, %3109 : f32
        linalg.yield %3111 : f32
      } -> tensor<1x1x3584xf32>
      %3112 = tensor.empty() : tensor<3584x37888xf32>
      %3113 = linalg.transpose ins(%21:tensor<37888x3584xf32>) outs(%3112:tensor<3584x37888xf32>) permutation = [1, 0]
      %3114 = tensor.collapse_shape %3107 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_proj"} : tensor<1x1x3584xf32> into tensor<3584xf32>
      %3115 = tensor.expand_shape %3114 [[0 : i64, 1 : i64]] output_shape [1, 3584] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_proj"} : tensor<3584xf32> into tensor<1x3584xf32>
      %3116 = tensor.empty() : tensor<1x37888xf32>
      %3117 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
      %3118 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3117 : f32) outs(%3116 : tensor<1x37888xf32>) -> tensor<1x37888xf32>
      %3119 = linalg.matmul {prov.region_id = "matmul_17", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_proj", prov.transposed_b = "true"} ins(%3115, %3113 : tensor<1x3584xf32>, tensor<3584x37888xf32>) outs(%3118 : tensor<1x37888xf32>) -> tensor<1x37888xf32>
      %3120 = tensor.collapse_shape %3119 [[0 : i64, 1 : i64]] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_proj"} : tensor<1x37888xf32> into tensor<37888xf32>
      %3121 = tensor.expand_shape %3120 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 37888] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_proj"} : tensor<37888xf32> into tensor<1x1x37888xf32>
      %3122 = "tensor.extract_slice"(%3121) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_5", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x37888xf32>) -> tensor<1x1x18944xf32>
      %3123 = "tensor.extract_slice"(%3121) <{static_offsets = array<i64: 0, 0, 18944>, static_sizes = array<i64: 1, 1, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_5", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x37888xf32>) -> tensor<1x1x18944xf32>
      %3124 = tensor.empty() : tensor<1x1x18944xf32>
      %3125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3123 : tensor<1x1x18944xf32>) outs(%3124 : tensor<1x1x18944xf32>) attrs =  {prov.region_id = "sigmoid_2", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.act"} {
      ^bb330(%3126: f32, %3127: f32):
        %3128 = arith.constant 1.000000e+00 : f32
        %3129 = arith.negf %3126 : f32
        %3130 = math.exp %3129 : f32
        %3131 = arith.addf %3128, %3130 : f32
        %3132 = arith.divf %3128, %3131 : f32
        linalg.yield %3132 : f32
      } -> tensor<1x1x18944xf32>
      %3133 = tensor.empty() : tensor<1x1x18944xf32>
      %3134 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3123, %3125 : tensor<1x1x18944xf32>, tensor<1x1x18944xf32>) outs(%3133 : tensor<1x1x18944xf32>) attrs =  {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.act"} {
      ^bb331(%3135: f32, %3136: f32, %3137: f32):
        %3138 = arith.mulf %3135, %3136 : f32
        linalg.yield %3138 : f32
      } -> tensor<1x1x18944xf32>
      %3139 = tensor.empty() : tensor<1x1x18944xf32>
      %3140 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3134, %3122 : tensor<1x1x18944xf32>, tensor<1x1x18944xf32>) outs(%3139 : tensor<1x1x18944xf32>) attrs =  {prov.region_id = "mul_34", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb332(%3141: f32, %3142: f32, %3143: f32):
        %3144 = arith.mulf %3141, %3142 : f32
        linalg.yield %3144 : f32
      } -> tensor<1x1x18944xf32>
      %3145 = tensor.empty() : tensor<18944x3584xf32>
      %3146 = linalg.transpose ins(%22:tensor<3584x18944xf32>) outs(%3145:tensor<18944x3584xf32>) permutation = [1, 0]
      %3147 = tensor.collapse_shape %3140 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_out"} : tensor<1x1x18944xf32> into tensor<18944xf32>
      %3148 = tensor.expand_shape %3147 [[0 : i64, 1 : i64]] output_shape [1, 18944] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_out"} : tensor<18944xf32> into tensor<1x18944xf32>
      %3149 = tensor.empty() : tensor<1x3584xf32>
      %3150 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
      %3151 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3150 : f32) outs(%3149 : tensor<1x3584xf32>) -> tensor<1x3584xf32>
      %3152 = linalg.matmul {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_out", prov.transposed_b = "true"} ins(%3148, %3146 : tensor<1x18944xf32>, tensor<18944x3584xf32>) outs(%3151 : tensor<1x3584xf32>) -> tensor<1x3584xf32>
      %3153 = tensor.collapse_shape %3152 [[0 : i64, 1 : i64]] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_out"} : tensor<1x3584xf32> into tensor<3584xf32>
      %3154 = tensor.expand_shape %3153 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 3584] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.2.mlp.ff_out"} : tensor<3584xf32> into tensor<1x1x3584xf32>
      %3155 = tensor.empty() : tensor<1x1x3584xf32>
      %3156 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3060, %3154 : tensor<1x1x3584xf32>, tensor<1x1x3584xf32>) outs(%3155 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb333(%3157: f32, %3158: f32, %3159: f32):
        %3160 = arith.addf %3157, %3158 : f32
        linalg.yield %3160 : f32
      } -> tensor<1x1x3584xf32>
      %3161 = tensor.empty() : tensor<1x1x3584xf32>
      %3162 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3156 : tensor<1x1x3584xf32>) outs(%3161 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb334(%3163: f32, %3164: f32):
        %3165 = arith.constant 2.000000e+00 : f32
        %3166 = math.powf %3163, %3165 : f32
        linalg.yield %3166 : f32
      } -> tensor<1x1x3584xf32>
      %3167 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3168 = tensor.splat %3167 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3169 = linalg.reduce ins(%3162:tensor<1x1x3584xf32>) outs(%3168:tensor<1x1xf32>) dimensions = [2]
      (%3170: f32, %3171: f32) {
        %3172 = arith.addf %3170, %3171 : f32
        linalg.yield %3172 : f32
      }
      %3173 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
      %3174 = tensor.splat %3173 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3175 = tensor.empty() : tensor<1x1xf32>
      %3176 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3169, %3174 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%3175 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb335(%3177: f32, %3178: f32, %3179: f32):
        %3180 = arith.divf %3177, %3178 : f32
        linalg.yield %3180 : f32
      } -> tensor<1x1xf32>
      %3181 = tensor.collapse_shape %3176 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %3182 = tensor.expand_shape %3181 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %3183 = arith.constant {prov.region_id = "add_26", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %3184 = tensor.splat %3183 {prov.region_id = "add_26", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %3185 = tensor.empty() : tensor<1x1x1xf32>
      %3186 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3182, %3184 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%3185 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_26", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb336(%3187: f32, %3188: f32, %3189: f32):
        %3190 = arith.addf %3187, %3188 : f32
        linalg.yield %3190 : f32
      } -> tensor<1x1x1xf32>
      %3191 = tensor.empty() : tensor<1x1x1xf32>
      %3192 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3186 : tensor<1x1x1xf32>) outs(%3191 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_6", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb337(%3193: f32, %3194: f32):
        %3195 = math.rsqrt %3193 : f32
        linalg.yield %3195 : f32
      } -> tensor<1x1x1xf32>
      %3196 = tensor.empty() : tensor<1x1x3584xf32>
      %3197 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3156, %3192 : tensor<1x1x3584xf32>, tensor<1x1x1xf32>) outs(%3196 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_35", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb338(%3198: f32, %3199: f32, %3200: f32):
        %3201 = arith.mulf %3198, %3199 : f32
        linalg.yield %3201 : f32
      } -> tensor<1x1x3584xf32>
      %3202 = tensor.empty() : tensor<1x1x3584xf32>
      %3203 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %3197 : tensor<3584xf32>, tensor<1x1x3584xf32>) outs(%3202 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_36", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb339(%3204: f32, %3205: f32, %3206: f32):
        %3207 = arith.mulf %3204, %3205 : f32
        linalg.yield %3207 : f32
      } -> tensor<1x1x3584xf32>
      %3208 = tensor.collapse_shape %3203 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.att_proj"} : tensor<1x1x3584xf32> into tensor<3584xf32>
      %3209 = tensor.expand_shape %3208 [[0 : i64, 1 : i64]] output_shape [1, 3584] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.att_proj"} : tensor<3584xf32> into tensor<1x3584xf32>
      %3210 = tensor.empty() : tensor<3584x4608xf32>
      %3211 = linalg.transpose ins(%25:tensor<4608x3584xf32>) outs(%3210:tensor<3584x4608xf32>) permutation = [1, 0]
      %3212 = tensor.empty() : tensor<1x4608xf32>
      %3213 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
      %3214 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3213 : f32) outs(%3212 : tensor<1x4608xf32>) -> tensor<1x4608xf32>
      %3215 = linalg.matmul {prov.region_id = "matmul_19", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.att_proj", prov.transposed_b = "true"} ins(%3209, %3211 : tensor<1x3584xf32>, tensor<3584x4608xf32>) outs(%3214 : tensor<1x4608xf32>) -> tensor<1x4608xf32>
      %3216 = tensor.empty() : tensor<1x4608xf32>
      %3217 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3215, %26 : tensor<1x4608xf32>, tensor<4608xf32>) outs(%3216 : tensor<1x4608xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.att_proj"} {
      ^bb340(%3218: f32, %3219: f32, %3220: f32):
        %3221 = arith.addf %3218, %3219 : f32
        linalg.yield %3221 : f32
      } -> tensor<1x4608xf32>
      %3222 = tensor.collapse_shape %3217 [[0 : i64, 1 : i64]] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.att_proj"} : tensor<1x4608xf32> into tensor<4608xf32>
      %3223 = tensor.expand_shape %3222 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 4608] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.att_proj"} : tensor<4608xf32> into tensor<1x1x4608xf32>
      %3224 = "tensor.extract_slice"(%3223) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_6", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x4608xf32>) -> tensor<1x1x3584xf32>
      %3225 = "tensor.extract_slice"(%3223) <{static_offsets = array<i64: 0, 0, 3584>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_6", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x4608xf32>) -> tensor<1x1x512xf32>
      %3226 = "tensor.extract_slice"(%3223) <{static_offsets = array<i64: 0, 0, 4096>, static_sizes = array<i64: 1, 1, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_6", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x4608xf32>) -> tensor<1x1x512xf32>
      %3227 = tensor.collapse_shape %3224 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x3584xf32> into tensor<3584xf32>
      %3228 = tensor.expand_shape %3227 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 28, 128] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<1x1x28x128xf32>
      %3229 = tensor.empty() : tensor<1x28x1x128xf32>
      %3230 = linalg.transpose ins(%3228:tensor<1x1x28x128xf32>) outs(%3229:tensor<1x28x1x128xf32>) permutation = [0, 2, 1, 3]
      %3231 = tensor.collapse_shape %3225 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x512xf32> into tensor<512xf32>
      %3232 = tensor.expand_shape %3231 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 128] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x4x128xf32>
      %3233 = tensor.empty() : tensor<1x4x1x128xf32>
      %3234 = linalg.transpose ins(%3232:tensor<1x1x4x128xf32>) outs(%3233:tensor<1x4x1x128xf32>) permutation = [0, 2, 1, 3]
      %3235 = tensor.collapse_shape %3226 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x512xf32> into tensor<512xf32>
      %3236 = tensor.expand_shape %3235 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 4, 128] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x1x4x128xf32>
      %3237 = tensor.empty() : tensor<1x4x1x128xf32>
      %3238 = linalg.transpose ins(%3236:tensor<1x1x4x128xf32>) outs(%3237:tensor<1x4x1x128xf32>) permutation = [0, 2, 1, 3]
      %3239 = tensor.collapse_shape %1922 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %3240 = tensor.expand_shape %3239 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 128] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x1x128xf32>
      %3241 = tensor.collapse_shape %1935 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1x128xf32> into tensor<128xf32>
      %3242 = tensor.expand_shape %3241 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 128] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x1x1x128xf32>
      %3243 = tensor.empty() : tensor<1x28x1x128xf32>
      %3244 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3230, %3240 : tensor<1x28x1x128xf32>, tensor<1x1x1x128xf32>) outs(%3243 : tensor<1x28x1x128xf32>) attrs =  {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb341(%3245: f32, %3246: f32, %3247: f32):
        %3248 = arith.mulf %3245, %3246 : f32
        linalg.yield %3248 : f32
      } -> tensor<1x28x1x128xf32>
      %3249 = "tensor.extract_slice"(%3230) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 28, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_39", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x28x1x128xf32>) -> tensor<1x28x1x64xf32>
      %3250 = "tensor.extract_slice"(%3230) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 28, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_40", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x28x1x128xf32>) -> tensor<1x28x1x64xf32>
      %3251 = tensor.empty() : tensor<1x28x1x64xf32>
      %3252 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3250 : tensor<1x28x1x64xf32>) outs(%3251 : tensor<1x28x1x64xf32>) attrs =  {prov.region_id = "neg_6", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb342(%3253: f32, %3254: f32):
        %3255 = arith.negf %3253 : f32
        linalg.yield %3255 : f32
      } -> tensor<1x28x1x64xf32>
      %3256 = tensor.concat dim(3) %3252, %3249 {prov.region_id = "cat_8", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x28x1x64xf32>, tensor<1x28x1x64xf32>) -> tensor<1x28x1x128xf32>
      %3257 = tensor.empty() : tensor<1x28x1x128xf32>
      %3258 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3256, %3242 : tensor<1x28x1x128xf32>, tensor<1x1x1x128xf32>) outs(%3257 : tensor<1x28x1x128xf32>) attrs =  {prov.region_id = "mul_38", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb343(%3259: f32, %3260: f32, %3261: f32):
        %3262 = arith.mulf %3259, %3260 : f32
        linalg.yield %3262 : f32
      } -> tensor<1x28x1x128xf32>
      %3263 = tensor.empty() : tensor<1x28x1x128xf32>
      %3264 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3244, %3258 : tensor<1x28x1x128xf32>, tensor<1x28x1x128xf32>) outs(%3263 : tensor<1x28x1x128xf32>) attrs =  {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb344(%3265: f32, %3266: f32, %3267: f32):
        %3268 = arith.addf %3265, %3266 : f32
        linalg.yield %3268 : f32
      } -> tensor<1x28x1x128xf32>
      %3269 = tensor.empty() : tensor<1x4x1x128xf32>
      %3270 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3234, %3240 : tensor<1x4x1x128xf32>, tensor<1x1x1x128xf32>) outs(%3269 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "mul_39", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb345(%3271: f32, %3272: f32, %3273: f32):
        %3274 = arith.mulf %3271, %3272 : f32
        linalg.yield %3274 : f32
      } -> tensor<1x4x1x128xf32>
      %3275 = "tensor.extract_slice"(%3234) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_41", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x128xf32>) -> tensor<1x4x1x64xf32>
      %3276 = "tensor.extract_slice"(%3234) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 1, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_42", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x128xf32>) -> tensor<1x4x1x64xf32>
      %3277 = tensor.empty() : tensor<1x4x1x64xf32>
      %3278 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3276 : tensor<1x4x1x64xf32>) outs(%3277 : tensor<1x4x1x64xf32>) attrs =  {prov.region_id = "neg_7", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb346(%3279: f32, %3280: f32):
        %3281 = arith.negf %3279 : f32
        linalg.yield %3281 : f32
      } -> tensor<1x4x1x64xf32>
      %3282 = tensor.concat dim(3) %3278, %3275 {prov.region_id = "cat_9", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x1x64xf32>, tensor<1x4x1x64xf32>) -> tensor<1x4x1x128xf32>
      %3283 = tensor.empty() : tensor<1x4x1x128xf32>
      %3284 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3282, %3242 : tensor<1x4x1x128xf32>, tensor<1x1x1x128xf32>) outs(%3283 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "mul_40", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb347(%3285: f32, %3286: f32, %3287: f32):
        %3288 = arith.mulf %3285, %3286 : f32
        linalg.yield %3288 : f32
      } -> tensor<1x4x1x128xf32>
      %3289 = tensor.empty() : tensor<1x4x1x128xf32>
      %3290 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3270, %3284 : tensor<1x4x1x128xf32>, tensor<1x4x1x128xf32>) outs(%3289 : tensor<1x4x1x128xf32>) attrs =  {prov.region_id = "add_29", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb348(%3291: f32, %3292: f32, %3293: f32):
        %3294 = arith.addf %3291, %3292 : f32
        linalg.yield %3294 : f32
      } -> tensor<1x4x1x128xf32>
      %3295 = tensor.empty() : tensor<1xi64>
      %3296 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%3295 : tensor<1xi64>) attrs =  {prov.region_id = "iota_5", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
      ^bb349(%3297: i64):
        %3298 = linalg.index 0 : index
        %3299 = arith.index_cast %3298 : index to i64
        %3300 = arith.constant 1 : i64
        %3301 = arith.muli %3299, %3300 : i64
        %3302 = arith.constant 0 : i64
        %3303 = arith.addi %3302, %3301 : i64
        linalg.yield %3303 : i64
      } -> tensor<1xi64>
      %3304 = tensor.empty() : tensor<1xi64>
      %3305 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1863, %3296 : tensor<i64>, tensor<1xi64>) outs(%3304 : tensor<1xi64>) attrs =  {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb350(%3306: i64, %3307: i64, %3308: i64):
        %3309 = arith.addi %3306, %3307 : i64
        linalg.yield %3309 : i64
      } -> tensor<1xi64>
      %3310 = "tensor.extract_slice"(%1857) <{static_offsets = array<i64: 3, 0, 0, 0, 0>, static_sizes = array<i64: 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_9", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<4x1x4x16x128xf32>) -> tensor<4x16x128xf32>
      %3311 = func.call @aten_index_put_default_1_wl1(%3310, %3305, %3290) {prov.region_id = "aten_index_put_default_1_6", prov.dispatch_id = "aten_index_put_default_1_6"} : (tensor<4x16x128xf32>, tensor<1xi64>, tensor<1x4x1x128xf32>) -> tensor<1x4x16x128xf32>
      %3312 = "tensor.extract_slice"(%1858) <{static_offsets = array<i64: 3, 0, 0, 0, 0>, static_sizes = array<i64: 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_10", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<4x1x4x16x128xf32>) -> tensor<4x16x128xf32>
      %3313 = func.call @aten_index_put_default_1_wl1(%3312, %3305, %3238) {prov.region_id = "aten_index_put_default_1_7", prov.dispatch_id = "aten_index_put_default_1_7"} : (tensor<4x16x128xf32>, tensor<1xi64>, tensor<1x4x1x128xf32>) -> tensor<1x4x16x128xf32>
      %3314 = "tensor.extract_slice"(%3311) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_43", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
      %3315 = "tensor.extract_slice"(%3314) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_44", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
      %3316 = tensor.collapse_shape %3315 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x16x128xf32> into tensor<8192xf32>
      %3317 = tensor.expand_shape %3316 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 16, 128] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x4x1x16x128xf32>
      %3318 = "tensor.extract_slice"(%3317) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_45", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
      %3319 = "tensor.extract_slice"(%3318) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_46", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
      %3320 = tensor.empty() : tensor<1x4x7x16x128xf32>
      %3321 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3319 : tensor<1x4x1x16x128xf32>) outs(%3320 : tensor<1x4x7x16x128xf32>) attrs =  {prov.region_id = "expand_21", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb351(%3322: f32, %3323: f32):
        linalg.yield %3322 : f32
      } -> tensor<1x4x7x16x128xf32>
      %3324 = tensor.collapse_shape %3321 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x7x16x128xf32> into tensor<57344xf32>
      %3325 = tensor.expand_shape %3324 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 16, 128] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<1x28x16x128xf32>
      %3326 = "tensor.extract_slice"(%3313) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_47", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
      %3327 = "tensor.extract_slice"(%3326) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 16, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_48", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>) -> tensor<1x4x16x128xf32>
      %3328 = tensor.collapse_shape %3327 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x4x16x128xf32> into tensor<8192xf32>
      %3329 = tensor.expand_shape %3328 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 16, 128] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<8192xf32> into tensor<1x4x1x16x128xf32>
      %3330 = "tensor.extract_slice"(%3329) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_49", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
      %3331 = "tensor.extract_slice"(%3330) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 16, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_50", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x4x1x16x128xf32>) -> tensor<1x4x1x16x128xf32>
      %3332 = tensor.empty() : tensor<1x4x7x16x128xf32>
      %3333 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3331 : tensor<1x4x1x16x128xf32>) outs(%3332 : tensor<1x4x7x16x128xf32>) attrs =  {prov.region_id = "expand_22", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb352(%3334: f32, %3335: f32):
        linalg.yield %3334 : f32
      } -> tensor<1x4x7x16x128xf32>
      %3336 = tensor.collapse_shape %3333 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x7x16x128xf32> into tensor<57344xf32>
      %3337 = tensor.expand_shape %3336 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 16, 128] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<1x28x16x128xf32>
      %3338 = tensor.empty() : tensor<1x28x128x16xf32>
      %3339 = linalg.transpose ins(%3325:tensor<1x28x16x128xf32>) outs(%3338:tensor<1x28x128x16xf32>) permutation = [0, 1, 3, 2]
      %3340 = tensor.empty() : tensor<1x28x1x128xf32>
      %3341 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3264 : tensor<1x28x1x128xf32>) outs(%3340 : tensor<1x28x1x128xf32>) attrs =  {prov.region_id = "expand_23", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb353(%3342: f32, %3343: f32):
        linalg.yield %3342 : f32
      } -> tensor<1x28x1x128xf32>
      %3344 = tensor.collapse_shape %3341 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x1x128xf32> into tensor<3584xf32>
      %3345 = tensor.expand_shape %3344 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 1, 128] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<28x1x128xf32>
      %3346 = tensor.empty() : tensor<1x28x128x16xf32>
      %3347 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3339 : tensor<1x28x128x16xf32>) outs(%3346 : tensor<1x28x128x16xf32>) attrs =  {prov.region_id = "expand_24", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb354(%3348: f32, %3349: f32):
        linalg.yield %3348 : f32
      } -> tensor<1x28x128x16xf32>
      %3350 = tensor.collapse_shape %3347 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x128x16xf32> into tensor<57344xf32>
      %3351 = tensor.expand_shape %3350 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 128, 16] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<28x128x16xf32>
      %3352 = arith.constant {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3353 = tensor.splat %3352 {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<28x1x16xf32>
      %3354 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3345, %3351 : tensor<28x1x128xf32>, tensor<28x128x16xf32>) outs(%3353 : tensor<28x1x16xf32>) attrs =  {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
      ^bb355(%3355: f32, %3356: f32, %3357: f32):
        %3358 = arith.mulf %3355, %3356 : f32
        %3359 = arith.addf %3357, %3358 : f32
        linalg.yield %3359 : f32
      } -> tensor<28x1x16xf32>
      %3360 = tensor.collapse_shape %3354 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28x1x16xf32> into tensor<448xf32>
      %3361 = tensor.expand_shape %3360 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 1, 16] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<448xf32> into tensor<1x28x1x16xf32>
      %3362 = arith.constant {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.0883883461 : f32
      %3363 = tensor.splat %3362 {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x1x16xf32>
      %3364 = tensor.empty() : tensor<1x28x1x16xf32>
      %3365 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3361, %3363 : tensor<1x28x1x16xf32>, tensor<1x28x1x16xf32>) outs(%3364 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb356(%3366: f32, %3367: f32, %3368: f32):
        %3369 = arith.mulf %3366, %3367 : f32
        linalg.yield %3369 : f32
      } -> tensor<1x28x1x16xf32>
      %3370 = tensor.empty() : tensor<1x1x1x16xi1>
      %3371 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1975 : tensor<1x1x1x16xi1>) outs(%3370 : tensor<1x1x1x16xi1>) attrs =  {prov.region_id = "bitwise_3", prov.family = "bitwise", prov._pattern_hint = "bitwise", prov.op = "bitwise", prov.aten = "aten.bitwise_not.default", prov.orig_dtype = "bool"} {
      ^bb357(%3372: i1, %3373: i1):
        %3374 = arith.constant true
        %3375 = arith.xori %3372, %3374 : i1
        linalg.yield %3375 : i1
      } -> tensor<1x1x1x16xi1>
      %3376 = arith.constant {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %3377 = tensor.splat %3376 {prov.region_id = "fill_3", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
      %3378 = tensor.empty() : tensor<1x28x1x16xf32>
      %3379 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> ()>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3371, %3377, %3365 : tensor<1x1x1x16xi1>, tensor<f32>, tensor<1x28x1x16xf32>) outs(%3378 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "select_11", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32"} {
      ^bb358(%3380: i1, %3381: f32, %3382: f32, %3383: f32):
        %3384 = arith.select %3380, %3381, %3382 : f32
        linalg.yield %3384 : f32
      } -> tensor<1x28x1x16xf32>
      %3385 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %3386 = tensor.splat %3385 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x1xf32>
      %3387 = linalg.reduce ins(%3379:tensor<1x28x1x16xf32>) outs(%3386:tensor<1x28x1xf32>) dimensions = [3]
      (%3388: f32, %3389: f32) {
        %3390 = arith.maximumf %3388, %3389 : f32
        linalg.yield %3390 : f32
      }
      %3391 = tensor.collapse_shape %3387 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x1xf32> into tensor<28xf32>
      %3392 = tensor.expand_shape %3391 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 1, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<28xf32> into tensor<1x28x1x1xf32>
      %3393 = tensor.empty() : tensor<1x28x1x16xf32>
      %3394 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3379, %3392 : tensor<1x28x1x16xf32>, tensor<1x28x1x1xf32>) outs(%3393 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb359(%3395: f32, %3396: f32, %3397: f32):
        %3398 = arith.subf %3395, %3396 : f32
        linalg.yield %3398 : f32
      } -> tensor<1x28x1x16xf32>
      %3399 = tensor.empty() : tensor<1x28x1x16xf32>
      %3400 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3394 : tensor<1x28x1x16xf32>) outs(%3399 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb360(%3401: f32, %3402: f32):
        %3403 = math.exp %3401 : f32
        linalg.yield %3403 : f32
      } -> tensor<1x28x1x16xf32>
      %3404 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3405 = tensor.splat %3404 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x1xf32>
      %3406 = linalg.reduce ins(%3400:tensor<1x28x1x16xf32>) outs(%3405:tensor<1x28x1xf32>) dimensions = [3]
      (%3407: f32, %3408: f32) {
        %3409 = arith.addf %3407, %3408 : f32
        linalg.yield %3409 : f32
      }
      %3410 = tensor.collapse_shape %3406 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<1x28x1xf32> into tensor<28xf32>
      %3411 = tensor.expand_shape %3410 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 1, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<28xf32> into tensor<1x28x1x1xf32>
      %3412 = tensor.empty() : tensor<1x28x1x16xf32>
      %3413 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3400, %3411 : tensor<1x28x1x16xf32>, tensor<1x28x1x1xf32>) outs(%3412 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
      ^bb361(%3414: f32, %3415: f32, %3416: f32):
        %3417 = arith.divf %3414, %3415 : f32
        linalg.yield %3417 : f32
      } -> tensor<1x28x1x16xf32>
      %3418 = tensor.empty() : tensor<1x28x1x16xf32>
      %3419 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3413 : tensor<1x28x1x16xf32>) outs(%3418 : tensor<1x28x1x16xf32>) attrs =  {prov.region_id = "expand_25", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb362(%3420: f32, %3421: f32):
        linalg.yield %3420 : f32
      } -> tensor<1x28x1x16xf32>
      %3422 = tensor.collapse_shape %3419 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x1x16xf32> into tensor<448xf32>
      %3423 = tensor.expand_shape %3422 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 1, 16] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<448xf32> into tensor<28x1x16xf32>
      %3424 = tensor.empty() : tensor<1x28x16x128xf32>
      %3425 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3337 : tensor<1x28x16x128xf32>) outs(%3424 : tensor<1x28x16x128xf32>) attrs =  {prov.region_id = "expand_26", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb363(%3426: f32, %3427: f32):
        linalg.yield %3426 : f32
      } -> tensor<1x28x16x128xf32>
      %3428 = tensor.collapse_shape %3425 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x16x128xf32> into tensor<57344xf32>
      %3429 = tensor.expand_shape %3428 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 16, 128] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<57344xf32> into tensor<28x16x128xf32>
      %3430 = arith.constant {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3431 = tensor.splat %3430 {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<28x1x128xf32>
      %3432 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%3423, %3429 : tensor<28x1x16xf32>, tensor<28x16x128xf32>) outs(%3431 : tensor<28x1x128xf32>) attrs =  {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
      ^bb364(%3433: f32, %3434: f32, %3435: f32):
        %3436 = arith.mulf %3433, %3434 : f32
        %3437 = arith.addf %3435, %3436 : f32
        linalg.yield %3437 : f32
      } -> tensor<28x1x128xf32>
      %3438 = tensor.collapse_shape %3432 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28x1x128xf32> into tensor<3584xf32>
      %3439 = tensor.expand_shape %3438 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 1, 128] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<1x28x1x128xf32>
      %3440 = tensor.empty() : tensor<1x1x28x128xf32>
      %3441 = linalg.transpose ins(%3439:tensor<1x28x1x128xf32>) outs(%3440:tensor<1x1x28x128xf32>) permutation = [0, 2, 1, 3]
      %3442 = tensor.collapse_shape %3441 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x28x128xf32> into tensor<3584xf32>
      %3443 = tensor.expand_shape %3442 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 3584] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<3584xf32> into tensor<1x1x3584xf32>
      %3444 = tensor.empty() : tensor<3584x3584xf32>
      %3445 = linalg.transpose ins(%27:tensor<3584x3584xf32>) outs(%3444:tensor<3584x3584xf32>) permutation = [1, 0]
      %3446 = tensor.collapse_shape %3443 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.attn_out"} : tensor<1x1x3584xf32> into tensor<3584xf32>
      %3447 = tensor.expand_shape %3446 [[0 : i64, 1 : i64]] output_shape [1, 3584] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.attn_out"} : tensor<3584xf32> into tensor<1x3584xf32>
      %3448 = tensor.empty() : tensor<1x3584xf32>
      %3449 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
      %3450 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3449 : f32) outs(%3448 : tensor<1x3584xf32>) -> tensor<1x3584xf32>
      %3451 = linalg.matmul {prov.region_id = "matmul_22", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.attn_out", prov.transposed_b = "true"} ins(%3447, %3445 : tensor<1x3584xf32>, tensor<3584x3584xf32>) outs(%3450 : tensor<1x3584xf32>) -> tensor<1x3584xf32>
      %3452 = tensor.collapse_shape %3451 [[0 : i64, 1 : i64]] {prov.region_id = "view_79", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.attn_out"} : tensor<1x3584xf32> into tensor<3584xf32>
      %3453 = tensor.expand_shape %3452 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 3584] {prov.region_id = "view_79", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.self_attn.attn_out"} : tensor<3584xf32> into tensor<1x1x3584xf32>
      %3454 = tensor.empty() : tensor<1x1x3584xf32>
      %3455 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3156, %3453 : tensor<1x1x3584xf32>, tensor<1x1x3584xf32>) outs(%3454 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb365(%3456: f32, %3457: f32, %3458: f32):
        %3459 = arith.addf %3456, %3457 : f32
        linalg.yield %3459 : f32
      } -> tensor<1x1x3584xf32>
      %3460 = tensor.empty() : tensor<1x1x3584xf32>
      %3461 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3455 : tensor<1x1x3584xf32>) outs(%3460 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb366(%3462: f32, %3463: f32):
        %3464 = arith.constant 2.000000e+00 : f32
        %3465 = math.powf %3462, %3464 : f32
        linalg.yield %3465 : f32
      } -> tensor<1x1x3584xf32>
      %3466 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3467 = tensor.splat %3466 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3468 = linalg.reduce ins(%3461:tensor<1x1x3584xf32>) outs(%3467:tensor<1x1xf32>) dimensions = [2]
      (%3469: f32, %3470: f32) {
        %3471 = arith.addf %3469, %3470 : f32
        linalg.yield %3471 : f32
      }
      %3472 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
      %3473 = tensor.splat %3472 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3474 = tensor.empty() : tensor<1x1xf32>
      %3475 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3468, %3473 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%3474 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb367(%3476: f32, %3477: f32, %3478: f32):
        %3479 = arith.divf %3476, %3477 : f32
        linalg.yield %3479 : f32
      } -> tensor<1x1xf32>
      %3480 = tensor.collapse_shape %3475 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %3481 = tensor.expand_shape %3480 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %3482 = arith.constant {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %3483 = tensor.splat %3482 {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %3484 = tensor.empty() : tensor<1x1x1xf32>
      %3485 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3481, %3483 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%3484 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb368(%3486: f32, %3487: f32, %3488: f32):
        %3489 = arith.addf %3486, %3487 : f32
        linalg.yield %3489 : f32
      } -> tensor<1x1x1xf32>
      %3490 = tensor.empty() : tensor<1x1x1xf32>
      %3491 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3485 : tensor<1x1x1xf32>) outs(%3490 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_7", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb369(%3492: f32, %3493: f32):
        %3494 = math.rsqrt %3492 : f32
        linalg.yield %3494 : f32
      } -> tensor<1x1x1xf32>
      %3495 = tensor.empty() : tensor<1x1x3584xf32>
      %3496 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3455, %3491 : tensor<1x1x3584xf32>, tensor<1x1x1xf32>) outs(%3495 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_42", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb370(%3497: f32, %3498: f32, %3499: f32):
        %3500 = arith.mulf %3497, %3498 : f32
        linalg.yield %3500 : f32
      } -> tensor<1x1x3584xf32>
      %3501 = tensor.empty() : tensor<1x1x3584xf32>
      %3502 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %3496 : tensor<3584xf32>, tensor<1x1x3584xf32>) outs(%3501 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb371(%3503: f32, %3504: f32, %3505: f32):
        %3506 = arith.mulf %3503, %3504 : f32
        linalg.yield %3506 : f32
      } -> tensor<1x1x3584xf32>
      %3507 = tensor.empty() : tensor<3584x37888xf32>
      %3508 = linalg.transpose ins(%28:tensor<37888x3584xf32>) outs(%3507:tensor<3584x37888xf32>) permutation = [1, 0]
      %3509 = tensor.collapse_shape %3502 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_80", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_proj"} : tensor<1x1x3584xf32> into tensor<3584xf32>
      %3510 = tensor.expand_shape %3509 [[0 : i64, 1 : i64]] output_shape [1, 3584] {prov.region_id = "view_80", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_proj"} : tensor<3584xf32> into tensor<1x3584xf32>
      %3511 = tensor.empty() : tensor<1x37888xf32>
      %3512 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
      %3513 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3512 : f32) outs(%3511 : tensor<1x37888xf32>) -> tensor<1x37888xf32>
      %3514 = linalg.matmul {prov.region_id = "matmul_23", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_proj", prov.transposed_b = "true"} ins(%3510, %3508 : tensor<1x3584xf32>, tensor<3584x37888xf32>) outs(%3513 : tensor<1x37888xf32>) -> tensor<1x37888xf32>
      %3515 = tensor.collapse_shape %3514 [[0 : i64, 1 : i64]] {prov.region_id = "view_81", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_proj"} : tensor<1x37888xf32> into tensor<37888xf32>
      %3516 = tensor.expand_shape %3515 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 37888] {prov.region_id = "view_81", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_proj"} : tensor<37888xf32> into tensor<1x1x37888xf32>
      %3517 = "tensor.extract_slice"(%3516) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_7", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x37888xf32>) -> tensor<1x1x18944xf32>
      %3518 = "tensor.extract_slice"(%3516) <{static_offsets = array<i64: 0, 0, 18944>, static_sizes = array<i64: 1, 1, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_7", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x1x37888xf32>) -> tensor<1x1x18944xf32>
      %3519 = tensor.empty() : tensor<1x1x18944xf32>
      %3520 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3518 : tensor<1x1x18944xf32>) outs(%3519 : tensor<1x1x18944xf32>) attrs =  {prov.region_id = "sigmoid_3", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.act"} {
      ^bb372(%3521: f32, %3522: f32):
        %3523 = arith.constant 1.000000e+00 : f32
        %3524 = arith.negf %3521 : f32
        %3525 = math.exp %3524 : f32
        %3526 = arith.addf %3523, %3525 : f32
        %3527 = arith.divf %3523, %3526 : f32
        linalg.yield %3527 : f32
      } -> tensor<1x1x18944xf32>
      %3528 = tensor.empty() : tensor<1x1x18944xf32>
      %3529 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3518, %3520 : tensor<1x1x18944xf32>, tensor<1x1x18944xf32>) outs(%3528 : tensor<1x1x18944xf32>) attrs =  {prov.region_id = "mul_44", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.act"} {
      ^bb373(%3530: f32, %3531: f32, %3532: f32):
        %3533 = arith.mulf %3530, %3531 : f32
        linalg.yield %3533 : f32
      } -> tensor<1x1x18944xf32>
      %3534 = tensor.empty() : tensor<1x1x18944xf32>
      %3535 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3529, %3517 : tensor<1x1x18944xf32>, tensor<1x1x18944xf32>) outs(%3534 : tensor<1x1x18944xf32>) attrs =  {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb374(%3536: f32, %3537: f32, %3538: f32):
        %3539 = arith.mulf %3536, %3537 : f32
        linalg.yield %3539 : f32
      } -> tensor<1x1x18944xf32>
      %3540 = tensor.empty() : tensor<18944x3584xf32>
      %3541 = linalg.transpose ins(%29:tensor<3584x18944xf32>) outs(%3540:tensor<18944x3584xf32>) permutation = [1, 0]
      %3542 = tensor.collapse_shape %3535 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_82", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_out"} : tensor<1x1x18944xf32> into tensor<18944xf32>
      %3543 = tensor.expand_shape %3542 [[0 : i64, 1 : i64]] output_shape [1, 18944] {prov.region_id = "view_82", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_out"} : tensor<18944xf32> into tensor<1x18944xf32>
      %3544 = tensor.empty() : tensor<1x3584xf32>
      %3545 = arith.constant {prov.module = "blocks"} 0.000000e+00 : f32
      %3546 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "blocks"} ins(%3545 : f32) outs(%3544 : tensor<1x3584xf32>) -> tensor<1x3584xf32>
      %3547 = linalg.matmul {prov.region_id = "matmul_24", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_out", prov.transposed_b = "true"} ins(%3543, %3541 : tensor<1x18944xf32>, tensor<18944x3584xf32>) outs(%3546 : tensor<1x3584xf32>) -> tensor<1x3584xf32>
      %3548 = tensor.collapse_shape %3547 [[0 : i64, 1 : i64]] {prov.region_id = "view_83", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_out"} : tensor<1x3584xf32> into tensor<3584xf32>
      %3549 = tensor.expand_shape %3548 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 3584] {prov.region_id = "view_83", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "blocks", prov.fqn = "blocks.3.mlp.ff_out"} : tensor<3584xf32> into tensor<1x1x3584xf32>
      %3550 = tensor.empty() : tensor<1x1x3584xf32>
      %3551 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3455, %3549 : tensor<1x1x3584xf32>, tensor<1x1x3584xf32>) outs(%3550 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "add_33", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb375(%3552: f32, %3553: f32, %3554: f32):
        %3555 = arith.addf %3552, %3553 : f32
        linalg.yield %3555 : f32
      } -> tensor<1x1x3584xf32>
      %3556 = tensor.concat dim(0) %2126, %2521, %2916, %3311 {prov.region_id = "cat_10", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>, tensor<1x4x16x128xf32>, tensor<1x4x16x128xf32>, tensor<1x4x16x128xf32>) -> tensor<4x4x16x128xf32>
      %3557 = tensor.collapse_shape %3556 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_84", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<4x4x16x128xf32> into tensor<32768xf32>
      %3558 = tensor.expand_shape %3557 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [4, 1, 4, 16, 128] {prov.region_id = "view_84", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<4x1x4x16x128xf32>
      %3559 = tensor.concat dim(0) %2128, %2523, %2918, %3313 {prov.region_id = "cat_11", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x4x16x128xf32>, tensor<1x4x16x128xf32>, tensor<1x4x16x128xf32>, tensor<1x4x16x128xf32>) -> tensor<4x4x16x128xf32>
      %3560 = tensor.collapse_shape %3559 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_85", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<4x4x16x128xf32> into tensor<32768xf32>
      %3561 = tensor.expand_shape %3560 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [4, 1, 4, 16, 128] {prov.region_id = "view_85", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<4x1x4x16x128xf32>
      %3562 = tensor.empty() : tensor<1x1x3584xf32>
      %3563 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3551 : tensor<1x1x3584xf32>) outs(%3562 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb376(%3564: f32, %3565: f32):
        %3566 = arith.constant 2.000000e+00 : f32
        %3567 = math.powf %3564, %3566 : f32
        linalg.yield %3567 : f32
      } -> tensor<1x1x3584xf32>
      %3568 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %3569 = tensor.splat %3568 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3570 = linalg.reduce ins(%3563:tensor<1x1x3584xf32>) outs(%3569:tensor<1x1xf32>) dimensions = [2]
      (%3571: f32, %3572: f32) {
        %3573 = arith.addf %3571, %3572 : f32
        linalg.yield %3573 : f32
      }
      %3574 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
      %3575 = tensor.splat %3574 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32>
      %3576 = tensor.empty() : tensor<1x1xf32>
      %3577 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3570, %3575 : tensor<1x1xf32>, tensor<1x1xf32>) outs(%3576 : tensor<1x1xf32>) attrs =  {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb377(%3578: f32, %3579: f32, %3580: f32):
        %3581 = arith.divf %3578, %3579 : f32
        linalg.yield %3581 : f32
      } -> tensor<1x1xf32>
      %3582 = tensor.collapse_shape %3577 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x1xf32> into tensor<1xf32>
      %3583 = tensor.expand_shape %3582 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1x1xf32>
      %3584 = arith.constant {prov.region_id = "add_34", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %3585 = tensor.splat %3584 {prov.region_id = "add_34", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %3586 = tensor.empty() : tensor<1x1x1xf32>
      %3587 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3583, %3585 : tensor<1x1x1xf32>, tensor<1x1x1xf32>) outs(%3586 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "add_34", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb378(%3588: f32, %3589: f32, %3590: f32):
        %3591 = arith.addf %3588, %3589 : f32
        linalg.yield %3591 : f32
      } -> tensor<1x1x1xf32>
      %3592 = tensor.empty() : tensor<1x1x1xf32>
      %3593 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3587 : tensor<1x1x1xf32>) outs(%3592 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "rsqrt_8", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb379(%3594: f32, %3595: f32):
        %3596 = math.rsqrt %3594 : f32
        linalg.yield %3596 : f32
      } -> tensor<1x1x1xf32>
      %3597 = tensor.empty() : tensor<1x1x3584xf32>
      %3598 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3551, %3593 : tensor<1x1x3584xf32>, tensor<1x1x1xf32>) outs(%3597 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_46", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb380(%3599: f32, %3600: f32, %3601: f32):
        %3602 = arith.mulf %3599, %3600 : f32
        linalg.yield %3602 : f32
      } -> tensor<1x1x3584xf32>
      %3603 = tensor.empty() : tensor<1x1x3584xf32>
      %3604 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %3598 : tensor<3584xf32>, tensor<1x1x3584xf32>) outs(%3603 : tensor<1x1x3584xf32>) attrs =  {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "final_norm", prov.fqn = "final_norm"} {
      ^bb381(%3605: f32, %3606: f32, %3607: f32):
        %3608 = arith.mulf %3605, %3606 : f32
        linalg.yield %3608 : f32
      } -> tensor<1x1x3584xf32>
      %3609 = tensor.empty() : tensor<3584x4096xf32>
      %3610 = linalg.transpose ins(%31:tensor<4096x3584xf32>) outs(%3609:tensor<3584x4096xf32>) permutation = [1, 0]
      %3611 = tensor.collapse_shape %3604 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_86", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm_head", prov.fqn = "lm_head"} : tensor<1x1x3584xf32> into tensor<3584xf32>
      %3612 = tensor.expand_shape %3611 [[0 : i64, 1 : i64]] output_shape [1, 3584] {prov.region_id = "view_86", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm_head", prov.fqn = "lm_head"} : tensor<3584xf32> into tensor<1x3584xf32>
      %3613 = tensor.empty() : tensor<1x4096xf32>
      %3614 = arith.constant {prov.module = "lm_head"} 0.000000e+00 : f32
      %3615 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm_head"} ins(%3614 : f32) outs(%3613 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
      %3616 = linalg.matmul {prov.region_id = "matmul_25", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm_head", prov.fqn = "lm_head", prov.transposed_b = "true"} ins(%3612, %3610 : tensor<1x3584xf32>, tensor<3584x4096xf32>) outs(%3615 : tensor<1x4096xf32>) -> tensor<1x4096xf32>
      %3617 = tensor.collapse_shape %3616 [[0 : i64, 1 : i64]] {prov.region_id = "view_87", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm_head", prov.fqn = "lm_head"} : tensor<1x4096xf32> into tensor<4096xf32>
      %3618 = tensor.expand_shape %3617 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 4096] {prov.region_id = "view_87", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm_head", prov.fqn = "lm_head"} : tensor<4096xf32> into tensor<1x1x4096xf32>
      %3619 = "tensor.extract_slice"(%3618) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 4096>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_51", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
      %3620 = "tensor.extract_slice"(%3619) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 4096>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_12", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<1x1x4096xf32>) -> tensor<4096xf32>
      %3621 = tensor.expand_shape %3620 [[0 : i64, 1 : i64]] output_shape [1, 4096] {prov.region_id = "slice_52", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x4096xf32>
      %3622 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0xff800000 : f32
      %3623 = arith.constant {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} 0 : i64
      %3624 = tensor.splat %3622 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xf32>
      %3625 = tensor.splat %3623 {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xi64>
      %3626, %3627 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%3621 : tensor<1x4096xf32>) outs(%3624, %3625 : tensor<1xf32>, tensor<1xi64>) attrs =  {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} {
      ^bb382(%3628: f32, %3629: f32, %3630: i64):
        %3631 = linalg.index 1 : index
        %3632 = arith.index_cast %3631 : index to i64
        %3633 = arith.cmpf ogt, %3628, %3629 : f32
        %3634 = arith.select %3633, %3628, %3629 : f32
        %3635 = arith.select %3633, %3632, %3630 : i64
        linalg.yield %3634, %3635 : f32, i64
      } -> (tensor<1xf32>, tensor<1xi64>)
      %3636 = tensor.expand_shape %3626 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xf32> into tensor<1x1xf32>
      %3637 = tensor.expand_shape %3627 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "arg_reduce_0", prov.family = "arg_reduce", prov._pattern_hint = "aten_argmax", prov.op = "aten_argmax", prov.aten = "aten.argmax.default", prov.orig_dtype = "int64"} : tensor<1xi64> into tensor<1x1xi64>
      %3638 = arith.constant {prov.region_id = "add_35", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} 1 : i64
      %3639 = tensor.splat %3638 {prov.region_id = "add_35", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} : tensor<i64>
      %3640 = tensor.empty() : tensor<i64>
      %3641 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%1854, %3639 : tensor<i64>, tensor<i64>) outs(%3640 : tensor<i64>) attrs =  {prov.region_id = "add_35", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb383(%3642: i64, %3643: i64, %3644: i64):
        %3645 = arith.addi %3642, %3643 : i64
        linalg.yield %3645 : i64
      } -> tensor<i64>
      scf.yield %3641, %3637, %1861, %3558, %3561 : tensor<i64>, tensor<1x1xi64>, tensor<1x8xi64>, tensor<4x1x4x16x128xf32>, tensor<4x1x4x16x128xf32>
    }
    func.return %1850 : tensor<1x8xi64>
  }
}
