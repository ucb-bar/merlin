builtin.module attributes {prov.weights_file = "/scratch/agustin/projects/model2MLIR/workloads/molmoact/molmoact.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<4096x3584xf32>, %1: tensor<128x3584xf32>, %2: tensor<3584xf32>, %3: tensor<3584xf32>, %4: tensor<4608x3584xf32>, %5: tensor<4608xf32>, %6: tensor<3584x3584xf32>, %7: tensor<37888x3584xf32>, %8: tensor<3584x18944xf32>, %9: tensor<3584xf32>, %10: tensor<3584xf32>, %11: tensor<4608x3584xf32>, %12: tensor<4608xf32>, %13: tensor<3584x3584xf32>, %14: tensor<37888x3584xf32>, %15: tensor<3584x18944xf32>, %16: tensor<3584xf32>, %17: tensor<3584xf32>, %18: tensor<4608x3584xf32>, %19: tensor<4608xf32>, %20: tensor<3584x3584xf32>, %21: tensor<37888x3584xf32>, %22: tensor<3584x18944xf32>, %23: tensor<3584xf32>, %24: tensor<3584xf32>, %25: tensor<4608x3584xf32>, %26: tensor<4608xf32>, %27: tensor<3584x3584xf32>, %28: tensor<37888x3584xf32>, %29: tensor<3584x18944xf32>, %30: tensor<3584xf32>, %31: tensor<4096x3584xf32>, %32: tensor<64xf32>, %33: tensor<1x8xi64>) -> tensor<1x8x4096xf32> {
    %34 = arith.constant {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ne.Scalar", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model"} -1 : i64
    %35 = tensor.splat %34 {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ne.Scalar", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model"} : tensor<1x8xi64>
    %36 = tensor.empty() : tensor<1x8xi1>
    %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %35 : tensor<1x8xi64>, tensor<1x8xi64>) outs(%36 : tensor<1x8xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.ne.Scalar", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb0(%38: i64, %39: i64, %40: i1):
      %41 = arith.cmpi ne, %38, %39 : i64
      linalg.yield %41 : i1
    } -> tensor<1x8xi1>
    %42 = tensor.empty() : tensor<1x8xi64>
    %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%37 : tensor<1x8xi1>) outs(%42 : tensor<1x8xi64>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb1(%44: i1, %45: i64):
      %46 = arith.extui %44 : i1 to i64
      linalg.yield %46 : i64
    } -> tensor<1x8xi64>
    %47 = tensor.empty() : tensor<1x8xi64>
    %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %43 : tensor<1x8xi64>, tensor<1x8xi64>) outs(%47 : tensor<1x8xi64>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb2(%49: i64, %50: i64, %51: i64):
      %52 = arith.muli %49, %50 : i64
      linalg.yield %52 : i64
    } -> tensor<1x8xi64>
    %53 = tensor.concat dim(0) %0, %1 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.wte"} : (tensor<4096x3584xf32>, tensor<128x3584xf32>) -> tensor<4224x3584xf32>
    %54 = tensor.empty() : tensor<1x8x3584xf32>
    %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48 : tensor<1x8xi64>) outs(%54 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "embedding", prov.op = "embedding", prov.aten = "aten.embedding.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.wte"} {
    ^bb3(%56: i64, %57: f32):
      %58 = arith.index_cast %56 : i64 to index
      %59 = linalg.index 2 : index
      %60 = tensor.extract %53[%58, %59] : tensor<4224x3584xf32>
      linalg.yield %60 : f32
    } -> tensor<1x8x3584xf32>
    %61 = tensor.empty() : tensor<8xi64>
    %62 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%61 : tensor<8xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb4(%63: i64):
      %64 = linalg.index 0 : index
      %65 = arith.index_cast %64 : index to i64
      %66 = arith.constant 1 : i64
      %67 = arith.muli %65, %66 : i64
      %68 = arith.constant 0 : i64
      %69 = arith.addi %68, %67 : i64
      linalg.yield %69 : i64
    } -> tensor<8xi64>
    %70 = tensor.expand_shape %62 [[0 : i64, 1 : i64]] output_shape [1, 8] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<8xi64> into tensor<1x8xi64>
    %71 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model"} -3.40282347e+38 : f32
    %72 = tensor.splat %71 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model"} : tensor<8x9xf32>
    %73 = tensor.empty() : tensor<9xi64>
    %74 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%73 : tensor<9xi64>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb5(%75: i64):
      %76 = linalg.index 0 : index
      %77 = arith.index_cast %76 : index to i64
      %78 = arith.constant 1 : i64
      %79 = arith.muli %77, %78 : i64
      %80 = arith.constant 0 : i64
      %81 = arith.addi %80, %79 : i64
      linalg.yield %81 : i64
    } -> tensor<9xi64>
    %82 = tensor.expand_shape %74 [[0 : i64, 1 : i64]] output_shape [1, 9] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<9xi64> into tensor<1x9xi64>
    %83 = tensor.empty() : tensor<8xi64>
    %84 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%83 : tensor<8xi64>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb6(%85: i64):
      %86 = linalg.index 0 : index
      %87 = arith.index_cast %86 : index to i64
      %88 = arith.constant 1 : i64
      %89 = arith.muli %87, %88 : i64
      %90 = arith.constant 0 : i64
      %91 = arith.addi %90, %89 : i64
      linalg.yield %91 : i64
    } -> tensor<8xi64>
    %92 = tensor.expand_shape %84 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<8xi64> into tensor<8x1xi64>
    %93 = tensor.empty() : tensor<8x9xi64>
    %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %92 : tensor<1x9xi64>, tensor<8x1xi64>) outs(%93 : tensor<8x9xi64>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb7(%95: i64, %96: i64, %97: i64):
      %98 = arith.subi %95, %96 : i64
      linalg.yield %98 : i64
    } -> tensor<8x9xi64>
    %99 = arith.constant {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model"} 1 : i64
    %100 = tensor.splat %99 {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model"} : tensor<8x9xi64>
    %101 = tensor.empty() : tensor<8x9xi1>
    %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%94, %100 : tensor<8x9xi64>, tensor<8x9xi64>) outs(%101 : tensor<8x9xi1>) attrs =  {prov.region_id = "compare_1", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb8(%103: i64, %104: i64, %105: i1):
      %106 = arith.cmpi sge, %103, %104 : i64
      linalg.yield %106 : i1
    } -> tensor<8x9xi1>
    %107 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model"} 0.000000e+00 : f32
    %108 = tensor.splat %107 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model"} : tensor<f32>
    %109 = tensor.empty() : tensor<8x9xf32>
    %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%102, %72, %108 : tensor<8x9xi1>, tensor<8x9xf32>, tensor<f32>) outs(%109 : tensor<8x9xf32>) attrs =  {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb9(%111: i1, %112: f32, %113: f32, %114: f32):
      %115 = arith.select %111, %112, %113 : f32
      linalg.yield %115 : f32
    } -> tensor<8x9xf32>
    %116 = tensor.empty() : tensor<9xi64>
    %117 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%116 : tensor<9xi64>) attrs =  {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb10(%118: i64):
      %119 = linalg.index 0 : index
      %120 = arith.index_cast %119 : index to i64
      %121 = arith.constant 1 : i64
      %122 = arith.muli %120, %121 : i64
      %123 = arith.constant 0 : i64
      %124 = arith.addi %123, %122 : i64
      linalg.yield %124 : i64
    } -> tensor<9xi64>
    %125 = tensor.expand_shape %62 [[0 : i64, 1 : i64]] output_shape [8, 1] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int64", prov.module = "lm", prov.fqn = "lm.model"} : tensor<8xi64> into tensor<8x1xi64>
    %126 = tensor.empty() : tensor<8x9xi1>
    %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%117, %125 : tensor<9xi64>, tensor<8x1xi64>) outs(%126 : tensor<8x9xi1>) attrs =  {prov.region_id = "compare_2", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.gt.Tensor", prov.orig_dtype = "bool", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb11(%128: i64, %129: i64, %130: i1):
      %131 = arith.cmpi sgt, %128, %129 : i64
      linalg.yield %131 : i1
    } -> tensor<8x9xi1>
    %132 = tensor.empty() : tensor<8x9xf32>
    %133 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%110, %127 : tensor<8x9xf32>, tensor<8x9xi1>) outs(%132 : tensor<8x9xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model"} {
    ^bb12(%134: f32, %135: i1, %136: f32):
      %137 = arith.sitofp %135 : i1 to f32
      %138 = arith.mulf %134, %137 : f32
      linalg.yield %138 : f32
    } -> tensor<8x9xf32>
    %139 = tensor.expand_shape %32 [[0 : i64, 1 : i64]] output_shape [1, 64] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<64xf32> into tensor<1x64xf32>
    %140 = "tensor.extract_slice"(%139) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 64>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x64xf32>) -> tensor<1x64xf32>
    %141 = tensor.collapse_shape %140 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x64xf32> into tensor<64xf32>
    %142 = tensor.expand_shape %141 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 64, 1] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<64xf32> into tensor<1x64x1xf32>
    %143 = tensor.empty() : tensor<1x64x1xf32>
    %144 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%142 : tensor<1x64x1xf32>) outs(%143 : tensor<1x64x1xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb13(%145: f32, %146: f32):
      linalg.yield %145 : f32
    } -> tensor<1x64x1xf32>
    %147 = "tensor.extract_slice"(%70) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 8>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int64"} : (tensor<1x8xi64>) -> tensor<1x8xi64>
    %148 = tensor.collapse_shape %147 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<1x8xi64> into tensor<8xi64>
    %149 = tensor.expand_shape %148 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 8] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<8xi64> into tensor<1x1x8xi64>
    %150 = "tensor.extract_slice"(%149) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 8>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int64"} : (tensor<1x1x8xi64>) -> tensor<1x1x8xi64>
    %151 = tensor.empty() : tensor<1x1x8xf32>
    %152 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%150 : tensor<1x1x8xi64>) outs(%151 : tensor<1x1x8xf32>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb14(%153: i64, %154: f32):
      %155 = arith.sitofp %153 : i64 to f32
      linalg.yield %155 : f32
    } -> tensor<1x1x8xf32>
    %156 = tensor.empty() : tensor<1x64x1xf32>
    %157 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%144 : tensor<1x64x1xf32>) outs(%156 : tensor<1x64x1xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb15(%158: f32, %159: f32):
      linalg.yield %158 : f32
    } -> tensor<1x64x1xf32>
    %160 = tensor.empty() : tensor<1x1x8xf32>
    %161 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%152 : tensor<1x1x8xf32>) outs(%160 : tensor<1x1x8xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb16(%162: f32, %163: f32):
      linalg.yield %162 : f32
    } -> tensor<1x1x8xf32>
    %164 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %165 = tensor.splat %164 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} : tensor<1x64x8xf32>
    %166 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%157, %161 : tensor<1x64x1xf32>, tensor<1x1x8xf32>) outs(%165 : tensor<1x64x8xf32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32"} {
    ^bb17(%167: f32, %168: f32, %169: f32):
      %170 = arith.mulf %167, %168 : f32
      %171 = arith.addf %169, %170 : f32
      linalg.yield %171 : f32
    } -> tensor<1x64x8xf32>
    %172 = tensor.empty() : tensor<1x8x64xf32>
    %173 = linalg.transpose ins(%166:tensor<1x64x8xf32>) outs(%172:tensor<1x8x64xf32>) permutation = [0, 2, 1]
    %174 = tensor.concat dim(2) %173, %173 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x8x128xf32>
    %175 = tensor.empty() : tensor<1x8x128xf32>
    %176 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%174 : tensor<1x8x128xf32>) outs(%175 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32"} {
    ^bb18(%177: f32, %178: f32):
      %179 = math.cos %177 : f32
      linalg.yield %179 : f32
    } -> tensor<1x8x128xf32>
    %180 = arith.constant {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
    %181 = tensor.splat %180 {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x128xf32>
    %182 = tensor.empty() : tensor<1x8x128xf32>
    %183 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%176, %181 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%182 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb19(%184: f32, %185: f32, %186: f32):
      %187 = arith.mulf %184, %185 : f32
      linalg.yield %187 : f32
    } -> tensor<1x8x128xf32>
    %188 = tensor.empty() : tensor<1x8x128xf32>
    %189 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%174 : tensor<1x8x128xf32>) outs(%188 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32"} {
    ^bb20(%190: f32, %191: f32):
      %192 = math.sin %190 : f32
      linalg.yield %192 : f32
    } -> tensor<1x8x128xf32>
    %193 = arith.constant {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
    %194 = tensor.splat %193 {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x128xf32>
    %195 = tensor.empty() : tensor<1x8x128xf32>
    %196 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%189, %194 : tensor<1x8x128xf32>, tensor<1x8x128xf32>) outs(%195 : tensor<1x8x128xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb21(%197: f32, %198: f32, %199: f32):
      %200 = arith.mulf %197, %198 : f32
      linalg.yield %200 : f32
    } -> tensor<1x8x128xf32>
    %201 = tensor.empty() : tensor<1x8x3584xf32>
    %202 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%55 : tensor<1x8x3584xf32>) outs(%201 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb22(%203: f32, %204: f32):
      %205 = arith.constant 2.000000e+00 : f32
      %206 = math.powf %203, %205 : f32
      linalg.yield %206 : f32
    } -> tensor<1x8x3584xf32>
    %207 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %208 = tensor.splat %207 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %209 = linalg.reduce ins(%202:tensor<1x8x3584xf32>) outs(%208:tensor<1x8xf32>) dimensions = [2]
    (%210: f32, %211: f32) {
      %212 = arith.addf %210, %211 : f32
      linalg.yield %212 : f32
    }
    %213 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %214 = tensor.splat %213 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %215 = tensor.empty() : tensor<1x8xf32>
    %216 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%209, %214 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%215 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb23(%217: f32, %218: f32, %219: f32):
      %220 = arith.divf %217, %218 : f32
      linalg.yield %220 : f32
    } -> tensor<1x8xf32>
    %221 = tensor.collapse_shape %216 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %222 = tensor.expand_shape %221 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %223 = arith.constant {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %224 = tensor.splat %223 {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %225 = tensor.empty() : tensor<1x8x1xf32>
    %226 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%222, %224 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%225 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb24(%227: f32, %228: f32, %229: f32):
      %230 = arith.addf %227, %228 : f32
      linalg.yield %230 : f32
    } -> tensor<1x8x1xf32>
    %231 = tensor.empty() : tensor<1x8x1xf32>
    %232 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%226 : tensor<1x8x1xf32>) outs(%231 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb25(%233: f32, %234: f32):
      %235 = math.rsqrt %233 : f32
      linalg.yield %235 : f32
    } -> tensor<1x8x1xf32>
    %236 = tensor.empty() : tensor<1x8x3584xf32>
    %237 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%55, %232 : tensor<1x8x3584xf32>, tensor<1x8x1xf32>) outs(%236 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb26(%238: f32, %239: f32, %240: f32):
      %241 = arith.mulf %238, %239 : f32
      linalg.yield %241 : f32
    } -> tensor<1x8x3584xf32>
    %242 = tensor.empty() : tensor<1x8x3584xf32>
    %243 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2, %237 : tensor<3584xf32>, tensor<1x8x3584xf32>) outs(%242 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.attn_norm"} {
    ^bb27(%244: f32, %245: f32, %246: f32):
      %247 = arith.mulf %244, %245 : f32
      linalg.yield %247 : f32
    } -> tensor<1x8x3584xf32>
    %248 = tensor.collapse_shape %243 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn.att_proj"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %249 = tensor.expand_shape %248 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn.att_proj"} : tensor<28672xf32> into tensor<8x3584xf32>
    %250 = tensor.empty() : tensor<3584x4608xf32>
    %251 = linalg.transpose ins(%4:tensor<4608x3584xf32>) outs(%250:tensor<3584x4608xf32>) permutation = [1, 0]
    %252 = tensor.empty() : tensor<8x4608xf32>
    %253 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %254 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%253 : f32) outs(%252 : tensor<8x4608xf32>) -> tensor<8x4608xf32>
    %255 = linalg.matmul {prov.region_id = "matmul_1", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn.att_proj", prov.transposed_b = "true"} ins(%249, %251 : tensor<8x3584xf32>, tensor<3584x4608xf32>) outs(%254 : tensor<8x4608xf32>) -> tensor<8x4608xf32>
    %256 = tensor.empty() : tensor<8x4608xf32>
    %257 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%255, %5 : tensor<8x4608xf32>, tensor<4608xf32>) outs(%256 : tensor<8x4608xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn.att_proj"} {
    ^bb28(%258: f32, %259: f32, %260: f32):
      %261 = arith.addf %258, %259 : f32
      linalg.yield %261 : f32
    } -> tensor<8x4608xf32>
    %262 = tensor.collapse_shape %257 [[0 : i64, 1 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn.att_proj"} : tensor<8x4608xf32> into tensor<36864xf32>
    %263 = tensor.expand_shape %262 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 4608] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn.att_proj"} : tensor<36864xf32> into tensor<1x8x4608xf32>
    %264 = "tensor.extract_slice"(%263) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x8x4608xf32>) -> tensor<1x8x3584xf32>
    %265 = "tensor.extract_slice"(%263) <{static_offsets = array<i64: 0, 0, 3584>, static_sizes = array<i64: 1, 8, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x8x4608xf32>) -> tensor<1x8x512xf32>
    %266 = "tensor.extract_slice"(%263) <{static_offsets = array<i64: 0, 0, 4096>, static_sizes = array<i64: 1, 8, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x8x4608xf32>) -> tensor<1x8x512xf32>
    %267 = tensor.collapse_shape %266 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x8x512xf32> into tensor<4096xf32>
    %268 = tensor.expand_shape %267 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 128] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<4096xf32> into tensor<1x8x4x128xf32>
    %269 = tensor.collapse_shape %264 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %270 = tensor.expand_shape %269 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 128] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<28672xf32> into tensor<1x8x28x128xf32>
    %271 = tensor.collapse_shape %265 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x8x512xf32> into tensor<4096xf32>
    %272 = tensor.expand_shape %271 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 128] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<4096xf32> into tensor<1x8x4x128xf32>
    %273 = tensor.empty() : tensor<1x28x8x128xf32>
    %274 = linalg.transpose ins(%270:tensor<1x8x28x128xf32>) outs(%273:tensor<1x28x8x128xf32>) permutation = [0, 2, 1, 3]
    %275 = tensor.empty() : tensor<1x4x8x128xf32>
    %276 = linalg.transpose ins(%272:tensor<1x8x4x128xf32>) outs(%275:tensor<1x4x8x128xf32>) permutation = [0, 2, 1, 3]
    %277 = tensor.empty() : tensor<1x4x8x128xf32>
    %278 = linalg.transpose ins(%268:tensor<1x8x4x128xf32>) outs(%277:tensor<1x4x8x128xf32>) permutation = [0, 2, 1, 3]
    %279 = tensor.collapse_shape %183 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %280 = tensor.expand_shape %279 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 128] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1024xf32> into tensor<1x1x8x128xf32>
    %281 = tensor.collapse_shape %196 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %282 = tensor.expand_shape %281 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 128] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1024xf32> into tensor<1x1x8x128xf32>
    %283 = tensor.empty() : tensor<1x28x8x128xf32>
    %284 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%274, %280 : tensor<1x28x8x128xf32>, tensor<1x1x8x128xf32>) outs(%283 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb29(%285: f32, %286: f32, %287: f32):
      %288 = arith.mulf %285, %286 : f32
      linalg.yield %288 : f32
    } -> tensor<1x28x8x128xf32>
    %289 = "tensor.extract_slice"(%274) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 28, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x28x8x128xf32>) -> tensor<1x28x8x64xf32>
    %290 = "tensor.extract_slice"(%274) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 28, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x28x8x128xf32>) -> tensor<1x28x8x64xf32>
    %291 = tensor.empty() : tensor<1x28x8x64xf32>
    %292 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%290 : tensor<1x28x8x64xf32>) outs(%291 : tensor<1x28x8x64xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb30(%293: f32, %294: f32):
      %295 = arith.negf %293 : f32
      linalg.yield %295 : f32
    } -> tensor<1x28x8x64xf32>
    %296 = tensor.concat dim(3) %292, %289 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x28x8x64xf32>, tensor<1x28x8x64xf32>) -> tensor<1x28x8x128xf32>
    %297 = tensor.empty() : tensor<1x28x8x128xf32>
    %298 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%296, %282 : tensor<1x28x8x128xf32>, tensor<1x1x8x128xf32>) outs(%297 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb31(%299: f32, %300: f32, %301: f32):
      %302 = arith.mulf %299, %300 : f32
      linalg.yield %302 : f32
    } -> tensor<1x28x8x128xf32>
    %303 = tensor.empty() : tensor<1x28x8x128xf32>
    %304 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%284, %298 : tensor<1x28x8x128xf32>, tensor<1x28x8x128xf32>) outs(%303 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb32(%305: f32, %306: f32, %307: f32):
      %308 = arith.addf %305, %306 : f32
      linalg.yield %308 : f32
    } -> tensor<1x28x8x128xf32>
    %309 = tensor.empty() : tensor<1x4x8x128xf32>
    %310 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%276, %280 : tensor<1x4x8x128xf32>, tensor<1x1x8x128xf32>) outs(%309 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb33(%311: f32, %312: f32, %313: f32):
      %314 = arith.mulf %311, %312 : f32
      linalg.yield %314 : f32
    } -> tensor<1x4x8x128xf32>
    %315 = "tensor.extract_slice"(%276) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x64xf32>
    %316 = "tensor.extract_slice"(%276) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x64xf32>
    %317 = tensor.empty() : tensor<1x4x8x64xf32>
    %318 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%316 : tensor<1x4x8x64xf32>) outs(%317 : tensor<1x4x8x64xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb34(%319: f32, %320: f32):
      %321 = arith.negf %319 : f32
      linalg.yield %321 : f32
    } -> tensor<1x4x8x64xf32>
    %322 = tensor.concat dim(3) %318, %315 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x4x8x64xf32>, tensor<1x4x8x64xf32>) -> tensor<1x4x8x128xf32>
    %323 = tensor.empty() : tensor<1x4x8x128xf32>
    %324 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%322, %282 : tensor<1x4x8x128xf32>, tensor<1x1x8x128xf32>) outs(%323 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb35(%325: f32, %326: f32, %327: f32):
      %328 = arith.mulf %325, %326 : f32
      linalg.yield %328 : f32
    } -> tensor<1x4x8x128xf32>
    %329 = tensor.empty() : tensor<1x4x8x128xf32>
    %330 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%310, %324 : tensor<1x4x8x128xf32>, tensor<1x4x8x128xf32>) outs(%329 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb36(%331: f32, %332: f32, %333: f32):
      %334 = arith.addf %331, %332 : f32
      linalg.yield %334 : f32
    } -> tensor<1x4x8x128xf32>
    %335 = "tensor.extract_slice"(%330) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32>
    %336 = "tensor.extract_slice"(%335) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32>
    %337 = tensor.collapse_shape %336 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x4x8x128xf32> into tensor<4096xf32>
    %338 = tensor.expand_shape %337 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 8, 128] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<4096xf32> into tensor<1x4x1x8x128xf32>
    %339 = "tensor.extract_slice"(%338) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_9", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x4x1x8x128xf32>) -> tensor<1x4x1x8x128xf32>
    %340 = "tensor.extract_slice"(%339) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_10", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x4x1x8x128xf32>) -> tensor<1x4x1x8x128xf32>
    %341 = tensor.empty() : tensor<1x4x7x8x128xf32>
    %342 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%340 : tensor<1x4x1x8x128xf32>) outs(%341 : tensor<1x4x7x8x128xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb37(%343: f32, %344: f32):
      linalg.yield %343 : f32
    } -> tensor<1x4x7x8x128xf32>
    %345 = tensor.collapse_shape %342 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x4x7x8x128xf32> into tensor<28672xf32>
    %346 = tensor.expand_shape %345 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %347 = "tensor.extract_slice"(%278) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_11", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32>
    %348 = "tensor.extract_slice"(%347) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_12", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32>
    %349 = tensor.collapse_shape %348 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x4x8x128xf32> into tensor<4096xf32>
    %350 = tensor.expand_shape %349 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 8, 128] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<4096xf32> into tensor<1x4x1x8x128xf32>
    %351 = "tensor.extract_slice"(%350) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_13", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x4x1x8x128xf32>) -> tensor<1x4x1x8x128xf32>
    %352 = "tensor.extract_slice"(%351) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_14", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x4x1x8x128xf32>) -> tensor<1x4x1x8x128xf32>
    %353 = tensor.empty() : tensor<1x4x7x8x128xf32>
    %354 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%352 : tensor<1x4x1x8x128xf32>) outs(%353 : tensor<1x4x7x8x128xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb38(%355: f32, %356: f32):
      linalg.yield %355 : f32
    } -> tensor<1x4x7x8x128xf32>
    %357 = tensor.collapse_shape %354 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x4x7x8x128xf32> into tensor<28672xf32>
    %358 = tensor.expand_shape %357 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %359 = tensor.empty() : tensor<1x28x128x8xf32>
    %360 = linalg.transpose ins(%346:tensor<1x28x8x128xf32>) outs(%359:tensor<1x28x128x8xf32>) permutation = [0, 1, 3, 2]
    %361 = tensor.empty() : tensor<1x28x8x128xf32>
    %362 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%304 : tensor<1x28x8x128xf32>) outs(%361 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb39(%363: f32, %364: f32):
      linalg.yield %363 : f32
    } -> tensor<1x28x8x128xf32>
    %365 = tensor.collapse_shape %362 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
    %366 = tensor.expand_shape %365 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 128] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<28672xf32> into tensor<28x8x128xf32>
    %367 = tensor.empty() : tensor<1x28x128x8xf32>
    %368 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%360 : tensor<1x28x128x8xf32>) outs(%367 : tensor<1x28x128x8xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb40(%369: f32, %370: f32):
      linalg.yield %369 : f32
    } -> tensor<1x28x128x8xf32>
    %371 = tensor.collapse_shape %368 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x28x128x8xf32> into tensor<28672xf32>
    %372 = tensor.expand_shape %371 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 128, 8] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<28672xf32> into tensor<28x128x8xf32>
    %373 = arith.constant {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} 0.000000e+00 : f32
    %374 = tensor.splat %373 {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<28x8x8xf32>
    %375 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%366, %372 : tensor<28x8x128xf32>, tensor<28x128x8xf32>) outs(%374 : tensor<28x8x8xf32>) attrs =  {prov.region_id = "matmul_2", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb41(%376: f32, %377: f32, %378: f32):
      %379 = arith.mulf %376, %377 : f32
      %380 = arith.addf %378, %379 : f32
      linalg.yield %380 : f32
    } -> tensor<28x8x8xf32>
    %381 = tensor.collapse_shape %375 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<28x8x8xf32> into tensor<1792xf32>
    %382 = tensor.expand_shape %381 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 8] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1792xf32> into tensor<1x28x8x8xf32>
    %383 = arith.constant {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} 0.0883883461 : f32
    %384 = tensor.splat %383 {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x28x8x8xf32>
    %385 = tensor.empty() : tensor<1x28x8x8xf32>
    %386 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%382, %384 : tensor<1x28x8x8xf32>, tensor<1x28x8x8xf32>) outs(%385 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb42(%387: f32, %388: f32, %389: f32):
      %390 = arith.mulf %387, %388 : f32
      linalg.yield %390 : f32
    } -> tensor<1x28x8x8xf32>
    %391 = tensor.collapse_shape %133 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<8x9xf32> into tensor<72xf32>
    %392 = tensor.expand_shape %391 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 9] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<72xf32> into tensor<1x8x9xf32>
    %393 = tensor.collapse_shape %392 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x8x9xf32> into tensor<72xf32>
    %394 = tensor.expand_shape %393 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 9] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<72xf32> into tensor<1x1x8x9xf32>
    %395 = "tensor.extract_slice"(%394) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_15", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %396 = "tensor.extract_slice"(%395) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_16", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %397 = tensor.empty() : tensor<1x1x8x9xf32>
    %398 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%396 : tensor<1x1x8x9xf32>) outs(%397 : tensor<1x1x8x9xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb43(%399: f32, %400: f32):
      linalg.yield %399 : f32
    } -> tensor<1x1x8x9xf32>
    %401 = "tensor.extract_slice"(%398) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_17", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %402 = "tensor.extract_slice"(%401) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_18", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %403 = "tensor.extract_slice"(%402) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_19", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %404 = "tensor.extract_slice"(%403) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_20", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x8xf32>
    %405 = tensor.empty() : tensor<1x28x8x8xf32>
    %406 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%386, %404 : tensor<1x28x8x8xf32>, tensor<1x1x8x8xf32>) outs(%405 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb44(%407: f32, %408: f32, %409: f32):
      %410 = arith.addf %407, %408 : f32
      linalg.yield %410 : f32
    } -> tensor<1x28x8x8xf32>
    %411 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} 0xff800000 : f32
    %412 = tensor.splat %411 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x28x8xf32>
    %413 = linalg.reduce ins(%406:tensor<1x28x8x8xf32>) outs(%412:tensor<1x28x8xf32>) dimensions = [3]
    (%414: f32, %415: f32) {
      %416 = arith.maximumf %414, %415 : f32
      linalg.yield %416 : f32
    }
    %417 = tensor.collapse_shape %413 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x28x8xf32> into tensor<224xf32>
    %418 = tensor.expand_shape %417 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %419 = tensor.empty() : tensor<1x28x8x8xf32>
    %420 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%406, %418 : tensor<1x28x8x8xf32>, tensor<1x28x8x1xf32>) outs(%419 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb45(%421: f32, %422: f32, %423: f32):
      %424 = arith.subf %421, %422 : f32
      linalg.yield %424 : f32
    } -> tensor<1x28x8x8xf32>
    %425 = tensor.empty() : tensor<1x28x8x8xf32>
    %426 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%420 : tensor<1x28x8x8xf32>) outs(%425 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb46(%427: f32, %428: f32):
      %429 = math.exp %427 : f32
      linalg.yield %429 : f32
    } -> tensor<1x28x8x8xf32>
    %430 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} 0.000000e+00 : f32
    %431 = tensor.splat %430 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x28x8xf32>
    %432 = linalg.reduce ins(%426:tensor<1x28x8x8xf32>) outs(%431:tensor<1x28x8xf32>) dimensions = [3]
    (%433: f32, %434: f32) {
      %435 = arith.addf %433, %434 : f32
      linalg.yield %435 : f32
    }
    %436 = tensor.collapse_shape %432 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x28x8xf32> into tensor<224xf32>
    %437 = tensor.expand_shape %436 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %438 = tensor.empty() : tensor<1x28x8x8xf32>
    %439 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%426, %437 : tensor<1x28x8x8xf32>, tensor<1x28x8x1xf32>) outs(%438 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb47(%440: f32, %441: f32, %442: f32):
      %443 = arith.divf %440, %441 : f32
      linalg.yield %443 : f32
    } -> tensor<1x28x8x8xf32>
    %444 = tensor.empty() : tensor<1x28x8x8xf32>
    %445 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%439 : tensor<1x28x8x8xf32>) outs(%444 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb48(%446: f32, %447: f32):
      linalg.yield %446 : f32
    } -> tensor<1x28x8x8xf32>
    %448 = tensor.collapse_shape %445 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x28x8x8xf32> into tensor<1792xf32>
    %449 = tensor.expand_shape %448 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 8] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1792xf32> into tensor<28x8x8xf32>
    %450 = tensor.empty() : tensor<1x28x8x128xf32>
    %451 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%358 : tensor<1x28x8x128xf32>) outs(%450 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb49(%452: f32, %453: f32):
      linalg.yield %452 : f32
    } -> tensor<1x28x8x128xf32>
    %454 = tensor.collapse_shape %451 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
    %455 = tensor.expand_shape %454 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 128] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<28672xf32> into tensor<28x8x128xf32>
    %456 = arith.constant {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} 0.000000e+00 : f32
    %457 = tensor.splat %456 {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<28x8x128xf32>
    %458 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%449, %455 : tensor<28x8x8xf32>, tensor<28x8x128xf32>) outs(%457 : tensor<28x8x128xf32>) attrs =  {prov.region_id = "matmul_3", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} {
    ^bb50(%459: f32, %460: f32, %461: f32):
      %462 = arith.mulf %459, %460 : f32
      %463 = arith.addf %461, %462 : f32
      linalg.yield %463 : f32
    } -> tensor<28x8x128xf32>
    %464 = tensor.collapse_shape %458 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<28x8x128xf32> into tensor<28672xf32>
    %465 = tensor.expand_shape %464 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %466 = tensor.empty() : tensor<1x8x28x128xf32>
    %467 = linalg.transpose ins(%465:tensor<1x28x8x128xf32>) outs(%466:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
    %468 = tensor.collapse_shape %467 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<1x8x28x128xf32> into tensor<28672xf32>
    %469 = tensor.expand_shape %468 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %470 = tensor.empty() : tensor<3584x3584xf32>
    %471 = linalg.transpose ins(%6:tensor<3584x3584xf32>) outs(%470:tensor<3584x3584xf32>) permutation = [1, 0]
    %472 = tensor.collapse_shape %469 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn.attn_out"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %473 = tensor.expand_shape %472 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn.attn_out"} : tensor<28672xf32> into tensor<8x3584xf32>
    %474 = tensor.empty() : tensor<8x3584xf32>
    %475 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %476 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%475 : f32) outs(%474 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %477 = linalg.matmul {prov.region_id = "matmul_4", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn.attn_out", prov.transposed_b = "true"} ins(%473, %471 : tensor<8x3584xf32>, tensor<3584x3584xf32>) outs(%476 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %478 = tensor.collapse_shape %477 [[0 : i64, 1 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn.attn_out"} : tensor<8x3584xf32> into tensor<28672xf32>
    %479 = tensor.expand_shape %478 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.self_attn.attn_out"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %480 = tensor.empty() : tensor<1x8x3584xf32>
    %481 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%55, %479 : tensor<1x8x3584xf32>, tensor<1x8x3584xf32>) outs(%480 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0"} {
    ^bb51(%482: f32, %483: f32, %484: f32):
      %485 = arith.addf %482, %483 : f32
      linalg.yield %485 : f32
    } -> tensor<1x8x3584xf32>
    %486 = tensor.empty() : tensor<1x8x3584xf32>
    %487 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%481 : tensor<1x8x3584xf32>) outs(%486 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb52(%488: f32, %489: f32):
      %490 = arith.constant 2.000000e+00 : f32
      %491 = math.powf %488, %490 : f32
      linalg.yield %491 : f32
    } -> tensor<1x8x3584xf32>
    %492 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %493 = tensor.splat %492 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %494 = linalg.reduce ins(%487:tensor<1x8x3584xf32>) outs(%493:tensor<1x8xf32>) dimensions = [2]
    (%495: f32, %496: f32) {
      %497 = arith.addf %495, %496 : f32
      linalg.yield %497 : f32
    }
    %498 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %499 = tensor.splat %498 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %500 = tensor.empty() : tensor<1x8xf32>
    %501 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%494, %499 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%500 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb53(%502: f32, %503: f32, %504: f32):
      %505 = arith.divf %502, %503 : f32
      linalg.yield %505 : f32
    } -> tensor<1x8xf32>
    %506 = tensor.collapse_shape %501 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %507 = tensor.expand_shape %506 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %508 = arith.constant {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %509 = tensor.splat %508 {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %510 = tensor.empty() : tensor<1x8x1xf32>
    %511 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%507, %509 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%510 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb54(%512: f32, %513: f32, %514: f32):
      %515 = arith.addf %512, %513 : f32
      linalg.yield %515 : f32
    } -> tensor<1x8x1xf32>
    %516 = tensor.empty() : tensor<1x8x1xf32>
    %517 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%511 : tensor<1x8x1xf32>) outs(%516 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb55(%518: f32, %519: f32):
      %520 = math.rsqrt %518 : f32
      linalg.yield %520 : f32
    } -> tensor<1x8x1xf32>
    %521 = tensor.empty() : tensor<1x8x3584xf32>
    %522 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%481, %517 : tensor<1x8x3584xf32>, tensor<1x8x1xf32>) outs(%521 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb56(%523: f32, %524: f32, %525: f32):
      %526 = arith.mulf %523, %524 : f32
      linalg.yield %526 : f32
    } -> tensor<1x8x3584xf32>
    %527 = tensor.empty() : tensor<1x8x3584xf32>
    %528 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %522 : tensor<3584xf32>, tensor<1x8x3584xf32>) outs(%527 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.ff_norm"} {
    ^bb57(%529: f32, %530: f32, %531: f32):
      %532 = arith.mulf %529, %530 : f32
      linalg.yield %532 : f32
    } -> tensor<1x8x3584xf32>
    %533 = tensor.empty() : tensor<3584x37888xf32>
    %534 = linalg.transpose ins(%7:tensor<37888x3584xf32>) outs(%533:tensor<3584x37888xf32>) permutation = [1, 0]
    %535 = tensor.collapse_shape %528 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.mlp.ff_proj"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %536 = tensor.expand_shape %535 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.mlp.ff_proj"} : tensor<28672xf32> into tensor<8x3584xf32>
    %537 = tensor.empty() : tensor<8x37888xf32>
    %538 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %539 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%538 : f32) outs(%537 : tensor<8x37888xf32>) -> tensor<8x37888xf32>
    %540 = linalg.matmul {prov.region_id = "matmul_5", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.mlp.ff_proj", prov.transposed_b = "true"} ins(%536, %534 : tensor<8x3584xf32>, tensor<3584x37888xf32>) outs(%539 : tensor<8x37888xf32>) -> tensor<8x37888xf32>
    %541 = tensor.collapse_shape %540 [[0 : i64, 1 : i64]] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.mlp.ff_proj"} : tensor<8x37888xf32> into tensor<303104xf32>
    %542 = tensor.expand_shape %541 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 37888] {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.mlp.ff_proj"} : tensor<303104xf32> into tensor<1x8x37888xf32>
    %543 = "tensor.extract_slice"(%542) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.mlp"} : (tensor<1x8x37888xf32>) -> tensor<1x8x18944xf32>
    %544 = "tensor.extract_slice"(%542) <{static_offsets = array<i64: 0, 0, 18944>, static_sizes = array<i64: 1, 8, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.mlp"} : (tensor<1x8x37888xf32>) -> tensor<1x8x18944xf32>
    %545 = tensor.empty() : tensor<1x8x18944xf32>
    %546 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%544 : tensor<1x8x18944xf32>) outs(%545 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.mlp.act"} {
    ^bb58(%547: f32, %548: f32):
      %549 = arith.constant 1.000000e+00 : f32
      %550 = arith.negf %547 : f32
      %551 = math.exp %550 : f32
      %552 = arith.addf %549, %551 : f32
      %553 = arith.divf %549, %552 : f32
      linalg.yield %553 : f32
    } -> tensor<1x8x18944xf32>
    %554 = tensor.empty() : tensor<1x8x18944xf32>
    %555 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%544, %546 : tensor<1x8x18944xf32>, tensor<1x8x18944xf32>) outs(%554 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.mlp.act"} {
    ^bb59(%556: f32, %557: f32, %558: f32):
      %559 = arith.mulf %556, %557 : f32
      linalg.yield %559 : f32
    } -> tensor<1x8x18944xf32>
    %560 = tensor.empty() : tensor<1x8x18944xf32>
    %561 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%555, %543 : tensor<1x8x18944xf32>, tensor<1x8x18944xf32>) outs(%560 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.mlp"} {
    ^bb60(%562: f32, %563: f32, %564: f32):
      %565 = arith.mulf %562, %563 : f32
      linalg.yield %565 : f32
    } -> tensor<1x8x18944xf32>
    %566 = tensor.empty() : tensor<18944x3584xf32>
    %567 = linalg.transpose ins(%8:tensor<3584x18944xf32>) outs(%566:tensor<18944x3584xf32>) permutation = [1, 0]
    %568 = tensor.collapse_shape %561 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.mlp.ff_out"} : tensor<1x8x18944xf32> into tensor<151552xf32>
    %569 = tensor.expand_shape %568 [[0 : i64, 1 : i64]] output_shape [8, 18944] {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.mlp.ff_out"} : tensor<151552xf32> into tensor<8x18944xf32>
    %570 = tensor.empty() : tensor<8x3584xf32>
    %571 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %572 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%571 : f32) outs(%570 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %573 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.mlp.ff_out", prov.transposed_b = "true"} ins(%569, %567 : tensor<8x18944xf32>, tensor<18944x3584xf32>) outs(%572 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %574 = tensor.collapse_shape %573 [[0 : i64, 1 : i64]] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.mlp.ff_out"} : tensor<8x3584xf32> into tensor<28672xf32>
    %575 = tensor.expand_shape %574 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0.mlp.ff_out"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %576 = tensor.empty() : tensor<1x8x3584xf32>
    %577 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%481, %575 : tensor<1x8x3584xf32>, tensor<1x8x3584xf32>) outs(%576 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).0"} {
    ^bb61(%578: f32, %579: f32, %580: f32):
      %581 = arith.addf %578, %579 : f32
      linalg.yield %581 : f32
    } -> tensor<1x8x3584xf32>
    %582 = tensor.empty() : tensor<1x8x3584xf32>
    %583 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%577 : tensor<1x8x3584xf32>) outs(%582 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb62(%584: f32, %585: f32):
      %586 = arith.constant 2.000000e+00 : f32
      %587 = math.powf %584, %586 : f32
      linalg.yield %587 : f32
    } -> tensor<1x8x3584xf32>
    %588 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %589 = tensor.splat %588 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %590 = linalg.reduce ins(%583:tensor<1x8x3584xf32>) outs(%589:tensor<1x8xf32>) dimensions = [2]
    (%591: f32, %592: f32) {
      %593 = arith.addf %591, %592 : f32
      linalg.yield %593 : f32
    }
    %594 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %595 = tensor.splat %594 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %596 = tensor.empty() : tensor<1x8xf32>
    %597 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%590, %595 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%596 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb63(%598: f32, %599: f32, %600: f32):
      %601 = arith.divf %598, %599 : f32
      linalg.yield %601 : f32
    } -> tensor<1x8xf32>
    %602 = tensor.collapse_shape %597 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %603 = tensor.expand_shape %602 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %604 = arith.constant {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %605 = tensor.splat %604 {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %606 = tensor.empty() : tensor<1x8x1xf32>
    %607 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%603, %605 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%606 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb64(%608: f32, %609: f32, %610: f32):
      %611 = arith.addf %608, %609 : f32
      linalg.yield %611 : f32
    } -> tensor<1x8x1xf32>
    %612 = tensor.empty() : tensor<1x8x1xf32>
    %613 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%607 : tensor<1x8x1xf32>) outs(%612 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb65(%614: f32, %615: f32):
      %616 = math.rsqrt %614 : f32
      linalg.yield %616 : f32
    } -> tensor<1x8x1xf32>
    %617 = tensor.empty() : tensor<1x8x3584xf32>
    %618 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%577, %613 : tensor<1x8x3584xf32>, tensor<1x8x1xf32>) outs(%617 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb66(%619: f32, %620: f32, %621: f32):
      %622 = arith.mulf %619, %620 : f32
      linalg.yield %622 : f32
    } -> tensor<1x8x3584xf32>
    %623 = tensor.empty() : tensor<1x8x3584xf32>
    %624 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %618 : tensor<3584xf32>, tensor<1x8x3584xf32>) outs(%623 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.attn_norm"} {
    ^bb67(%625: f32, %626: f32, %627: f32):
      %628 = arith.mulf %625, %626 : f32
      linalg.yield %628 : f32
    } -> tensor<1x8x3584xf32>
    %629 = tensor.collapse_shape %624 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn.att_proj"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %630 = tensor.expand_shape %629 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_24", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn.att_proj"} : tensor<28672xf32> into tensor<8x3584xf32>
    %631 = tensor.empty() : tensor<3584x4608xf32>
    %632 = linalg.transpose ins(%11:tensor<4608x3584xf32>) outs(%631:tensor<3584x4608xf32>) permutation = [1, 0]
    %633 = tensor.empty() : tensor<8x4608xf32>
    %634 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %635 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%634 : f32) outs(%633 : tensor<8x4608xf32>) -> tensor<8x4608xf32>
    %636 = linalg.matmul {prov.region_id = "matmul_7", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn.att_proj", prov.transposed_b = "true"} ins(%630, %632 : tensor<8x3584xf32>, tensor<3584x4608xf32>) outs(%635 : tensor<8x4608xf32>) -> tensor<8x4608xf32>
    %637 = tensor.empty() : tensor<8x4608xf32>
    %638 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%636, %12 : tensor<8x4608xf32>, tensor<4608xf32>) outs(%637 : tensor<8x4608xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn.att_proj"} {
    ^bb68(%639: f32, %640: f32, %641: f32):
      %642 = arith.addf %639, %640 : f32
      linalg.yield %642 : f32
    } -> tensor<8x4608xf32>
    %643 = tensor.collapse_shape %638 [[0 : i64, 1 : i64]] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn.att_proj"} : tensor<8x4608xf32> into tensor<36864xf32>
    %644 = tensor.expand_shape %643 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 4608] {prov.region_id = "view_25", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn.att_proj"} : tensor<36864xf32> into tensor<1x8x4608xf32>
    %645 = "tensor.extract_slice"(%644) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x8x4608xf32>) -> tensor<1x8x3584xf32>
    %646 = "tensor.extract_slice"(%644) <{static_offsets = array<i64: 0, 0, 3584>, static_sizes = array<i64: 1, 8, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x8x4608xf32>) -> tensor<1x8x512xf32>
    %647 = "tensor.extract_slice"(%644) <{static_offsets = array<i64: 0, 0, 4096>, static_sizes = array<i64: 1, 8, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x8x4608xf32>) -> tensor<1x8x512xf32>
    %648 = tensor.collapse_shape %647 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x8x512xf32> into tensor<4096xf32>
    %649 = tensor.expand_shape %648 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 128] {prov.region_id = "view_26", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<4096xf32> into tensor<1x8x4x128xf32>
    %650 = tensor.collapse_shape %645 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %651 = tensor.expand_shape %650 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 128] {prov.region_id = "view_27", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<28672xf32> into tensor<1x8x28x128xf32>
    %652 = tensor.collapse_shape %646 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x8x512xf32> into tensor<4096xf32>
    %653 = tensor.expand_shape %652 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 128] {prov.region_id = "view_28", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<4096xf32> into tensor<1x8x4x128xf32>
    %654 = tensor.empty() : tensor<1x28x8x128xf32>
    %655 = linalg.transpose ins(%651:tensor<1x8x28x128xf32>) outs(%654:tensor<1x28x8x128xf32>) permutation = [0, 2, 1, 3]
    %656 = tensor.empty() : tensor<1x4x8x128xf32>
    %657 = linalg.transpose ins(%653:tensor<1x8x4x128xf32>) outs(%656:tensor<1x4x8x128xf32>) permutation = [0, 2, 1, 3]
    %658 = tensor.empty() : tensor<1x4x8x128xf32>
    %659 = linalg.transpose ins(%649:tensor<1x8x4x128xf32>) outs(%658:tensor<1x4x8x128xf32>) permutation = [0, 2, 1, 3]
    %660 = tensor.collapse_shape %183 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %661 = tensor.expand_shape %660 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 128] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1024xf32> into tensor<1x1x8x128xf32>
    %662 = tensor.collapse_shape %196 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %663 = tensor.expand_shape %662 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 128] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1024xf32> into tensor<1x1x8x128xf32>
    %664 = tensor.empty() : tensor<1x28x8x128xf32>
    %665 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%655, %661 : tensor<1x28x8x128xf32>, tensor<1x1x8x128xf32>) outs(%664 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb69(%666: f32, %667: f32, %668: f32):
      %669 = arith.mulf %666, %667 : f32
      linalg.yield %669 : f32
    } -> tensor<1x28x8x128xf32>
    %670 = "tensor.extract_slice"(%655) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 28, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_21", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x28x8x128xf32>) -> tensor<1x28x8x64xf32>
    %671 = "tensor.extract_slice"(%655) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 28, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_22", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x28x8x128xf32>) -> tensor<1x28x8x64xf32>
    %672 = tensor.empty() : tensor<1x28x8x64xf32>
    %673 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%671 : tensor<1x28x8x64xf32>) outs(%672 : tensor<1x28x8x64xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb70(%674: f32, %675: f32):
      %676 = arith.negf %674 : f32
      linalg.yield %676 : f32
    } -> tensor<1x28x8x64xf32>
    %677 = tensor.concat dim(3) %673, %670 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x28x8x64xf32>, tensor<1x28x8x64xf32>) -> tensor<1x28x8x128xf32>
    %678 = tensor.empty() : tensor<1x28x8x128xf32>
    %679 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%677, %663 : tensor<1x28x8x128xf32>, tensor<1x1x8x128xf32>) outs(%678 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb71(%680: f32, %681: f32, %682: f32):
      %683 = arith.mulf %680, %681 : f32
      linalg.yield %683 : f32
    } -> tensor<1x28x8x128xf32>
    %684 = tensor.empty() : tensor<1x28x8x128xf32>
    %685 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%665, %679 : tensor<1x28x8x128xf32>, tensor<1x28x8x128xf32>) outs(%684 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb72(%686: f32, %687: f32, %688: f32):
      %689 = arith.addf %686, %687 : f32
      linalg.yield %689 : f32
    } -> tensor<1x28x8x128xf32>
    %690 = tensor.empty() : tensor<1x4x8x128xf32>
    %691 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%657, %661 : tensor<1x4x8x128xf32>, tensor<1x1x8x128xf32>) outs(%690 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb73(%692: f32, %693: f32, %694: f32):
      %695 = arith.mulf %692, %693 : f32
      linalg.yield %695 : f32
    } -> tensor<1x4x8x128xf32>
    %696 = "tensor.extract_slice"(%657) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_23", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x64xf32>
    %697 = "tensor.extract_slice"(%657) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_24", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x64xf32>
    %698 = tensor.empty() : tensor<1x4x8x64xf32>
    %699 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%697 : tensor<1x4x8x64xf32>) outs(%698 : tensor<1x4x8x64xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb74(%700: f32, %701: f32):
      %702 = arith.negf %700 : f32
      linalg.yield %702 : f32
    } -> tensor<1x4x8x64xf32>
    %703 = tensor.concat dim(3) %699, %696 {prov.region_id = "cat_5", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x4x8x64xf32>, tensor<1x4x8x64xf32>) -> tensor<1x4x8x128xf32>
    %704 = tensor.empty() : tensor<1x4x8x128xf32>
    %705 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%703, %663 : tensor<1x4x8x128xf32>, tensor<1x1x8x128xf32>) outs(%704 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb75(%706: f32, %707: f32, %708: f32):
      %709 = arith.mulf %706, %707 : f32
      linalg.yield %709 : f32
    } -> tensor<1x4x8x128xf32>
    %710 = tensor.empty() : tensor<1x4x8x128xf32>
    %711 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%691, %705 : tensor<1x4x8x128xf32>, tensor<1x4x8x128xf32>) outs(%710 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb76(%712: f32, %713: f32, %714: f32):
      %715 = arith.addf %712, %713 : f32
      linalg.yield %715 : f32
    } -> tensor<1x4x8x128xf32>
    %716 = "tensor.extract_slice"(%711) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_25", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32>
    %717 = "tensor.extract_slice"(%716) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_26", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32>
    %718 = tensor.collapse_shape %717 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x4x8x128xf32> into tensor<4096xf32>
    %719 = tensor.expand_shape %718 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 8, 128] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<4096xf32> into tensor<1x4x1x8x128xf32>
    %720 = "tensor.extract_slice"(%719) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_27", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x4x1x8x128xf32>) -> tensor<1x4x1x8x128xf32>
    %721 = "tensor.extract_slice"(%720) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_28", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x4x1x8x128xf32>) -> tensor<1x4x1x8x128xf32>
    %722 = tensor.empty() : tensor<1x4x7x8x128xf32>
    %723 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%721 : tensor<1x4x1x8x128xf32>) outs(%722 : tensor<1x4x7x8x128xf32>) attrs =  {prov.region_id = "expand_10", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb77(%724: f32, %725: f32):
      linalg.yield %724 : f32
    } -> tensor<1x4x7x8x128xf32>
    %726 = tensor.collapse_shape %723 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x4x7x8x128xf32> into tensor<28672xf32>
    %727 = tensor.expand_shape %726 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_29", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %728 = "tensor.extract_slice"(%659) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_29", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32>
    %729 = "tensor.extract_slice"(%728) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_30", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32>
    %730 = tensor.collapse_shape %729 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x4x8x128xf32> into tensor<4096xf32>
    %731 = tensor.expand_shape %730 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 8, 128] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<4096xf32> into tensor<1x4x1x8x128xf32>
    %732 = "tensor.extract_slice"(%731) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_31", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x4x1x8x128xf32>) -> tensor<1x4x1x8x128xf32>
    %733 = "tensor.extract_slice"(%732) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_32", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x4x1x8x128xf32>) -> tensor<1x4x1x8x128xf32>
    %734 = tensor.empty() : tensor<1x4x7x8x128xf32>
    %735 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%733 : tensor<1x4x1x8x128xf32>) outs(%734 : tensor<1x4x7x8x128xf32>) attrs =  {prov.region_id = "expand_11", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb78(%736: f32, %737: f32):
      linalg.yield %736 : f32
    } -> tensor<1x4x7x8x128xf32>
    %738 = tensor.collapse_shape %735 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x4x7x8x128xf32> into tensor<28672xf32>
    %739 = tensor.expand_shape %738 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_30", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %740 = tensor.empty() : tensor<1x28x128x8xf32>
    %741 = linalg.transpose ins(%727:tensor<1x28x8x128xf32>) outs(%740:tensor<1x28x128x8xf32>) permutation = [0, 1, 3, 2]
    %742 = tensor.empty() : tensor<1x28x8x128xf32>
    %743 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%685 : tensor<1x28x8x128xf32>) outs(%742 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "expand_12", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb79(%744: f32, %745: f32):
      linalg.yield %744 : f32
    } -> tensor<1x28x8x128xf32>
    %746 = tensor.collapse_shape %743 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
    %747 = tensor.expand_shape %746 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 128] {prov.region_id = "view_31", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<28672xf32> into tensor<28x8x128xf32>
    %748 = tensor.empty() : tensor<1x28x128x8xf32>
    %749 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%741 : tensor<1x28x128x8xf32>) outs(%748 : tensor<1x28x128x8xf32>) attrs =  {prov.region_id = "expand_13", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb80(%750: f32, %751: f32):
      linalg.yield %750 : f32
    } -> tensor<1x28x128x8xf32>
    %752 = tensor.collapse_shape %749 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x28x128x8xf32> into tensor<28672xf32>
    %753 = tensor.expand_shape %752 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 128, 8] {prov.region_id = "view_32", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<28672xf32> into tensor<28x128x8xf32>
    %754 = arith.constant {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} 0.000000e+00 : f32
    %755 = tensor.splat %754 {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<28x8x8xf32>
    %756 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%747, %753 : tensor<28x8x128xf32>, tensor<28x128x8xf32>) outs(%755 : tensor<28x8x8xf32>) attrs =  {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb81(%757: f32, %758: f32, %759: f32):
      %760 = arith.mulf %757, %758 : f32
      %761 = arith.addf %759, %760 : f32
      linalg.yield %761 : f32
    } -> tensor<28x8x8xf32>
    %762 = tensor.collapse_shape %756 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<28x8x8xf32> into tensor<1792xf32>
    %763 = tensor.expand_shape %762 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 8] {prov.region_id = "view_33", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1792xf32> into tensor<1x28x8x8xf32>
    %764 = arith.constant {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} 0.0883883461 : f32
    %765 = tensor.splat %764 {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x28x8x8xf32>
    %766 = tensor.empty() : tensor<1x28x8x8xf32>
    %767 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%763, %765 : tensor<1x28x8x8xf32>, tensor<1x28x8x8xf32>) outs(%766 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb82(%768: f32, %769: f32, %770: f32):
      %771 = arith.mulf %768, %769 : f32
      linalg.yield %771 : f32
    } -> tensor<1x28x8x8xf32>
    %772 = tensor.collapse_shape %133 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<8x9xf32> into tensor<72xf32>
    %773 = tensor.expand_shape %772 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 9] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<72xf32> into tensor<1x8x9xf32>
    %774 = tensor.collapse_shape %773 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x8x9xf32> into tensor<72xf32>
    %775 = tensor.expand_shape %774 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 9] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<72xf32> into tensor<1x1x8x9xf32>
    %776 = "tensor.extract_slice"(%775) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_33", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %777 = "tensor.extract_slice"(%776) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_34", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %778 = tensor.empty() : tensor<1x1x8x9xf32>
    %779 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%777 : tensor<1x1x8x9xf32>) outs(%778 : tensor<1x1x8x9xf32>) attrs =  {prov.region_id = "expand_14", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb83(%780: f32, %781: f32):
      linalg.yield %780 : f32
    } -> tensor<1x1x8x9xf32>
    %782 = "tensor.extract_slice"(%779) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_35", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %783 = "tensor.extract_slice"(%782) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_36", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %784 = "tensor.extract_slice"(%783) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_37", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %785 = "tensor.extract_slice"(%784) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_38", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x8xf32>
    %786 = tensor.empty() : tensor<1x28x8x8xf32>
    %787 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%767, %785 : tensor<1x28x8x8xf32>, tensor<1x1x8x8xf32>) outs(%786 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb84(%788: f32, %789: f32, %790: f32):
      %791 = arith.addf %788, %789 : f32
      linalg.yield %791 : f32
    } -> tensor<1x28x8x8xf32>
    %792 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} 0xff800000 : f32
    %793 = tensor.splat %792 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x28x8xf32>
    %794 = linalg.reduce ins(%787:tensor<1x28x8x8xf32>) outs(%793:tensor<1x28x8xf32>) dimensions = [3]
    (%795: f32, %796: f32) {
      %797 = arith.maximumf %795, %796 : f32
      linalg.yield %797 : f32
    }
    %798 = tensor.collapse_shape %794 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x28x8xf32> into tensor<224xf32>
    %799 = tensor.expand_shape %798 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %800 = tensor.empty() : tensor<1x28x8x8xf32>
    %801 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%787, %799 : tensor<1x28x8x8xf32>, tensor<1x28x8x1xf32>) outs(%800 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb85(%802: f32, %803: f32, %804: f32):
      %805 = arith.subf %802, %803 : f32
      linalg.yield %805 : f32
    } -> tensor<1x28x8x8xf32>
    %806 = tensor.empty() : tensor<1x28x8x8xf32>
    %807 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%801 : tensor<1x28x8x8xf32>) outs(%806 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb86(%808: f32, %809: f32):
      %810 = math.exp %808 : f32
      linalg.yield %810 : f32
    } -> tensor<1x28x8x8xf32>
    %811 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} 0.000000e+00 : f32
    %812 = tensor.splat %811 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x28x8xf32>
    %813 = linalg.reduce ins(%807:tensor<1x28x8x8xf32>) outs(%812:tensor<1x28x8xf32>) dimensions = [3]
    (%814: f32, %815: f32) {
      %816 = arith.addf %814, %815 : f32
      linalg.yield %816 : f32
    }
    %817 = tensor.collapse_shape %813 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x28x8xf32> into tensor<224xf32>
    %818 = tensor.expand_shape %817 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %819 = tensor.empty() : tensor<1x28x8x8xf32>
    %820 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%807, %818 : tensor<1x28x8x8xf32>, tensor<1x28x8x1xf32>) outs(%819 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb87(%821: f32, %822: f32, %823: f32):
      %824 = arith.divf %821, %822 : f32
      linalg.yield %824 : f32
    } -> tensor<1x28x8x8xf32>
    %825 = tensor.empty() : tensor<1x28x8x8xf32>
    %826 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%820 : tensor<1x28x8x8xf32>) outs(%825 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "expand_15", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb88(%827: f32, %828: f32):
      linalg.yield %827 : f32
    } -> tensor<1x28x8x8xf32>
    %829 = tensor.collapse_shape %826 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x28x8x8xf32> into tensor<1792xf32>
    %830 = tensor.expand_shape %829 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 8] {prov.region_id = "view_34", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1792xf32> into tensor<28x8x8xf32>
    %831 = tensor.empty() : tensor<1x28x8x128xf32>
    %832 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%739 : tensor<1x28x8x128xf32>) outs(%831 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "expand_16", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb89(%833: f32, %834: f32):
      linalg.yield %833 : f32
    } -> tensor<1x28x8x128xf32>
    %835 = tensor.collapse_shape %832 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
    %836 = tensor.expand_shape %835 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 128] {prov.region_id = "view_35", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<28672xf32> into tensor<28x8x128xf32>
    %837 = arith.constant {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} 0.000000e+00 : f32
    %838 = tensor.splat %837 {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<28x8x128xf32>
    %839 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%830, %836 : tensor<28x8x8xf32>, tensor<28x8x128xf32>) outs(%838 : tensor<28x8x128xf32>) attrs =  {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} {
    ^bb90(%840: f32, %841: f32, %842: f32):
      %843 = arith.mulf %840, %841 : f32
      %844 = arith.addf %842, %843 : f32
      linalg.yield %844 : f32
    } -> tensor<28x8x128xf32>
    %845 = tensor.collapse_shape %839 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<28x8x128xf32> into tensor<28672xf32>
    %846 = tensor.expand_shape %845 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_36", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %847 = tensor.empty() : tensor<1x8x28x128xf32>
    %848 = linalg.transpose ins(%846:tensor<1x28x8x128xf32>) outs(%847:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
    %849 = tensor.collapse_shape %848 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<1x8x28x128xf32> into tensor<28672xf32>
    %850 = tensor.expand_shape %849 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_37", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %851 = tensor.empty() : tensor<3584x3584xf32>
    %852 = linalg.transpose ins(%13:tensor<3584x3584xf32>) outs(%851:tensor<3584x3584xf32>) permutation = [1, 0]
    %853 = tensor.collapse_shape %850 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn.attn_out"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %854 = tensor.expand_shape %853 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_38", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn.attn_out"} : tensor<28672xf32> into tensor<8x3584xf32>
    %855 = tensor.empty() : tensor<8x3584xf32>
    %856 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %857 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%856 : f32) outs(%855 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %858 = linalg.matmul {prov.region_id = "matmul_10", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn.attn_out", prov.transposed_b = "true"} ins(%854, %852 : tensor<8x3584xf32>, tensor<3584x3584xf32>) outs(%857 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %859 = tensor.collapse_shape %858 [[0 : i64, 1 : i64]] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn.attn_out"} : tensor<8x3584xf32> into tensor<28672xf32>
    %860 = tensor.expand_shape %859 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_39", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.self_attn.attn_out"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %861 = tensor.empty() : tensor<1x8x3584xf32>
    %862 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%577, %860 : tensor<1x8x3584xf32>, tensor<1x8x3584xf32>) outs(%861 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1"} {
    ^bb91(%863: f32, %864: f32, %865: f32):
      %866 = arith.addf %863, %864 : f32
      linalg.yield %866 : f32
    } -> tensor<1x8x3584xf32>
    %867 = tensor.empty() : tensor<1x8x3584xf32>
    %868 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%862 : tensor<1x8x3584xf32>) outs(%867 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb92(%869: f32, %870: f32):
      %871 = arith.constant 2.000000e+00 : f32
      %872 = math.powf %869, %871 : f32
      linalg.yield %872 : f32
    } -> tensor<1x8x3584xf32>
    %873 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %874 = tensor.splat %873 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %875 = linalg.reduce ins(%868:tensor<1x8x3584xf32>) outs(%874:tensor<1x8xf32>) dimensions = [2]
    (%876: f32, %877: f32) {
      %878 = arith.addf %876, %877 : f32
      linalg.yield %878 : f32
    }
    %879 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %880 = tensor.splat %879 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %881 = tensor.empty() : tensor<1x8xf32>
    %882 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%875, %880 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%881 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb93(%883: f32, %884: f32, %885: f32):
      %886 = arith.divf %883, %884 : f32
      linalg.yield %886 : f32
    } -> tensor<1x8xf32>
    %887 = tensor.collapse_shape %882 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %888 = tensor.expand_shape %887 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %889 = arith.constant {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %890 = tensor.splat %889 {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %891 = tensor.empty() : tensor<1x8x1xf32>
    %892 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%888, %890 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%891 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb94(%893: f32, %894: f32, %895: f32):
      %896 = arith.addf %893, %894 : f32
      linalg.yield %896 : f32
    } -> tensor<1x8x1xf32>
    %897 = tensor.empty() : tensor<1x8x1xf32>
    %898 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%892 : tensor<1x8x1xf32>) outs(%897 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb95(%899: f32, %900: f32):
      %901 = math.rsqrt %899 : f32
      linalg.yield %901 : f32
    } -> tensor<1x8x1xf32>
    %902 = tensor.empty() : tensor<1x8x3584xf32>
    %903 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%862, %898 : tensor<1x8x3584xf32>, tensor<1x8x1xf32>) outs(%902 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb96(%904: f32, %905: f32, %906: f32):
      %907 = arith.mulf %904, %905 : f32
      linalg.yield %907 : f32
    } -> tensor<1x8x3584xf32>
    %908 = tensor.empty() : tensor<1x8x3584xf32>
    %909 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %903 : tensor<3584xf32>, tensor<1x8x3584xf32>) outs(%908 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.ff_norm"} {
    ^bb97(%910: f32, %911: f32, %912: f32):
      %913 = arith.mulf %910, %911 : f32
      linalg.yield %913 : f32
    } -> tensor<1x8x3584xf32>
    %914 = tensor.empty() : tensor<3584x37888xf32>
    %915 = linalg.transpose ins(%14:tensor<37888x3584xf32>) outs(%914:tensor<3584x37888xf32>) permutation = [1, 0]
    %916 = tensor.collapse_shape %909 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.mlp.ff_proj"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %917 = tensor.expand_shape %916 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_40", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.mlp.ff_proj"} : tensor<28672xf32> into tensor<8x3584xf32>
    %918 = tensor.empty() : tensor<8x37888xf32>
    %919 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %920 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%919 : f32) outs(%918 : tensor<8x37888xf32>) -> tensor<8x37888xf32>
    %921 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.mlp.ff_proj", prov.transposed_b = "true"} ins(%917, %915 : tensor<8x3584xf32>, tensor<3584x37888xf32>) outs(%920 : tensor<8x37888xf32>) -> tensor<8x37888xf32>
    %922 = tensor.collapse_shape %921 [[0 : i64, 1 : i64]] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.mlp.ff_proj"} : tensor<8x37888xf32> into tensor<303104xf32>
    %923 = tensor.expand_shape %922 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 37888] {prov.region_id = "view_41", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.mlp.ff_proj"} : tensor<303104xf32> into tensor<1x8x37888xf32>
    %924 = "tensor.extract_slice"(%923) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.mlp"} : (tensor<1x8x37888xf32>) -> tensor<1x8x18944xf32>
    %925 = "tensor.extract_slice"(%923) <{static_offsets = array<i64: 0, 0, 18944>, static_sizes = array<i64: 1, 8, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.mlp"} : (tensor<1x8x37888xf32>) -> tensor<1x8x18944xf32>
    %926 = tensor.empty() : tensor<1x8x18944xf32>
    %927 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%925 : tensor<1x8x18944xf32>) outs(%926 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.mlp.act"} {
    ^bb98(%928: f32, %929: f32):
      %930 = arith.constant 1.000000e+00 : f32
      %931 = arith.negf %928 : f32
      %932 = math.exp %931 : f32
      %933 = arith.addf %930, %932 : f32
      %934 = arith.divf %930, %933 : f32
      linalg.yield %934 : f32
    } -> tensor<1x8x18944xf32>
    %935 = tensor.empty() : tensor<1x8x18944xf32>
    %936 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%925, %927 : tensor<1x8x18944xf32>, tensor<1x8x18944xf32>) outs(%935 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.mlp.act"} {
    ^bb99(%937: f32, %938: f32, %939: f32):
      %940 = arith.mulf %937, %938 : f32
      linalg.yield %940 : f32
    } -> tensor<1x8x18944xf32>
    %941 = tensor.empty() : tensor<1x8x18944xf32>
    %942 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%936, %924 : tensor<1x8x18944xf32>, tensor<1x8x18944xf32>) outs(%941 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.mlp"} {
    ^bb100(%943: f32, %944: f32, %945: f32):
      %946 = arith.mulf %943, %944 : f32
      linalg.yield %946 : f32
    } -> tensor<1x8x18944xf32>
    %947 = tensor.empty() : tensor<18944x3584xf32>
    %948 = linalg.transpose ins(%15:tensor<3584x18944xf32>) outs(%947:tensor<18944x3584xf32>) permutation = [1, 0]
    %949 = tensor.collapse_shape %942 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.mlp.ff_out"} : tensor<1x8x18944xf32> into tensor<151552xf32>
    %950 = tensor.expand_shape %949 [[0 : i64, 1 : i64]] output_shape [8, 18944] {prov.region_id = "view_42", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.mlp.ff_out"} : tensor<151552xf32> into tensor<8x18944xf32>
    %951 = tensor.empty() : tensor<8x3584xf32>
    %952 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %953 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%952 : f32) outs(%951 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %954 = linalg.matmul {prov.region_id = "matmul_12", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.mlp.ff_out", prov.transposed_b = "true"} ins(%950, %948 : tensor<8x18944xf32>, tensor<18944x3584xf32>) outs(%953 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %955 = tensor.collapse_shape %954 [[0 : i64, 1 : i64]] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.mlp.ff_out"} : tensor<8x3584xf32> into tensor<28672xf32>
    %956 = tensor.expand_shape %955 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_43", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1.mlp.ff_out"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %957 = tensor.empty() : tensor<1x8x3584xf32>
    %958 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%862, %956 : tensor<1x8x3584xf32>, tensor<1x8x3584xf32>) outs(%957 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).1"} {
    ^bb101(%959: f32, %960: f32, %961: f32):
      %962 = arith.addf %959, %960 : f32
      linalg.yield %962 : f32
    } -> tensor<1x8x3584xf32>
    %963 = tensor.empty() : tensor<1x8x3584xf32>
    %964 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%958 : tensor<1x8x3584xf32>) outs(%963 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb102(%965: f32, %966: f32):
      %967 = arith.constant 2.000000e+00 : f32
      %968 = math.powf %965, %967 : f32
      linalg.yield %968 : f32
    } -> tensor<1x8x3584xf32>
    %969 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %970 = tensor.splat %969 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %971 = linalg.reduce ins(%964:tensor<1x8x3584xf32>) outs(%970:tensor<1x8xf32>) dimensions = [2]
    (%972: f32, %973: f32) {
      %974 = arith.addf %972, %973 : f32
      linalg.yield %974 : f32
    }
    %975 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %976 = tensor.splat %975 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %977 = tensor.empty() : tensor<1x8xf32>
    %978 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%971, %976 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%977 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb103(%979: f32, %980: f32, %981: f32):
      %982 = arith.divf %979, %980 : f32
      linalg.yield %982 : f32
    } -> tensor<1x8xf32>
    %983 = tensor.collapse_shape %978 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %984 = tensor.expand_shape %983 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %985 = arith.constant {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %986 = tensor.splat %985 {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %987 = tensor.empty() : tensor<1x8x1xf32>
    %988 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%984, %986 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%987 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb104(%989: f32, %990: f32, %991: f32):
      %992 = arith.addf %989, %990 : f32
      linalg.yield %992 : f32
    } -> tensor<1x8x1xf32>
    %993 = tensor.empty() : tensor<1x8x1xf32>
    %994 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%988 : tensor<1x8x1xf32>) outs(%993 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb105(%995: f32, %996: f32):
      %997 = math.rsqrt %995 : f32
      linalg.yield %997 : f32
    } -> tensor<1x8x1xf32>
    %998 = tensor.empty() : tensor<1x8x3584xf32>
    %999 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%958, %994 : tensor<1x8x3584xf32>, tensor<1x8x1xf32>) outs(%998 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb106(%1000: f32, %1001: f32, %1002: f32):
      %1003 = arith.mulf %1000, %1001 : f32
      linalg.yield %1003 : f32
    } -> tensor<1x8x3584xf32>
    %1004 = tensor.empty() : tensor<1x8x3584xf32>
    %1005 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16, %999 : tensor<3584xf32>, tensor<1x8x3584xf32>) outs(%1004 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.attn_norm"} {
    ^bb107(%1006: f32, %1007: f32, %1008: f32):
      %1009 = arith.mulf %1006, %1007 : f32
      linalg.yield %1009 : f32
    } -> tensor<1x8x3584xf32>
    %1010 = tensor.collapse_shape %1005 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn.att_proj"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %1011 = tensor.expand_shape %1010 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_44", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn.att_proj"} : tensor<28672xf32> into tensor<8x3584xf32>
    %1012 = tensor.empty() : tensor<3584x4608xf32>
    %1013 = linalg.transpose ins(%18:tensor<4608x3584xf32>) outs(%1012:tensor<3584x4608xf32>) permutation = [1, 0]
    %1014 = tensor.empty() : tensor<8x4608xf32>
    %1015 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %1016 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%1015 : f32) outs(%1014 : tensor<8x4608xf32>) -> tensor<8x4608xf32>
    %1017 = linalg.matmul {prov.region_id = "matmul_13", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn.att_proj", prov.transposed_b = "true"} ins(%1011, %1013 : tensor<8x3584xf32>, tensor<3584x4608xf32>) outs(%1016 : tensor<8x4608xf32>) -> tensor<8x4608xf32>
    %1018 = tensor.empty() : tensor<8x4608xf32>
    %1019 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1017, %19 : tensor<8x4608xf32>, tensor<4608xf32>) outs(%1018 : tensor<8x4608xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn.att_proj"} {
    ^bb108(%1020: f32, %1021: f32, %1022: f32):
      %1023 = arith.addf %1020, %1021 : f32
      linalg.yield %1023 : f32
    } -> tensor<8x4608xf32>
    %1024 = tensor.collapse_shape %1019 [[0 : i64, 1 : i64]] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn.att_proj"} : tensor<8x4608xf32> into tensor<36864xf32>
    %1025 = tensor.expand_shape %1024 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 4608] {prov.region_id = "view_45", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn.att_proj"} : tensor<36864xf32> into tensor<1x8x4608xf32>
    %1026 = "tensor.extract_slice"(%1025) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_4", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x8x4608xf32>) -> tensor<1x8x3584xf32>
    %1027 = "tensor.extract_slice"(%1025) <{static_offsets = array<i64: 0, 0, 3584>, static_sizes = array<i64: 1, 8, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_4", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x8x4608xf32>) -> tensor<1x8x512xf32>
    %1028 = "tensor.extract_slice"(%1025) <{static_offsets = array<i64: 0, 0, 4096>, static_sizes = array<i64: 1, 8, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_4", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x8x4608xf32>) -> tensor<1x8x512xf32>
    %1029 = tensor.collapse_shape %1028 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x8x512xf32> into tensor<4096xf32>
    %1030 = tensor.expand_shape %1029 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 128] {prov.region_id = "view_46", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<4096xf32> into tensor<1x8x4x128xf32>
    %1031 = tensor.collapse_shape %1026 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %1032 = tensor.expand_shape %1031 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 128] {prov.region_id = "view_47", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<28672xf32> into tensor<1x8x28x128xf32>
    %1033 = tensor.collapse_shape %1027 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x8x512xf32> into tensor<4096xf32>
    %1034 = tensor.expand_shape %1033 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 128] {prov.region_id = "view_48", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<4096xf32> into tensor<1x8x4x128xf32>
    %1035 = tensor.empty() : tensor<1x28x8x128xf32>
    %1036 = linalg.transpose ins(%1032:tensor<1x8x28x128xf32>) outs(%1035:tensor<1x28x8x128xf32>) permutation = [0, 2, 1, 3]
    %1037 = tensor.empty() : tensor<1x4x8x128xf32>
    %1038 = linalg.transpose ins(%1034:tensor<1x8x4x128xf32>) outs(%1037:tensor<1x4x8x128xf32>) permutation = [0, 2, 1, 3]
    %1039 = tensor.empty() : tensor<1x4x8x128xf32>
    %1040 = linalg.transpose ins(%1030:tensor<1x8x4x128xf32>) outs(%1039:tensor<1x4x8x128xf32>) permutation = [0, 2, 1, 3]
    %1041 = tensor.collapse_shape %183 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1042 = tensor.expand_shape %1041 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 128] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1024xf32> into tensor<1x1x8x128xf32>
    %1043 = tensor.collapse_shape %196 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1044 = tensor.expand_shape %1043 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 128] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1024xf32> into tensor<1x1x8x128xf32>
    %1045 = tensor.empty() : tensor<1x28x8x128xf32>
    %1046 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1036, %1042 : tensor<1x28x8x128xf32>, tensor<1x1x8x128xf32>) outs(%1045 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb109(%1047: f32, %1048: f32, %1049: f32):
      %1050 = arith.mulf %1047, %1048 : f32
      linalg.yield %1050 : f32
    } -> tensor<1x28x8x128xf32>
    %1051 = "tensor.extract_slice"(%1036) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 28, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_39", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x28x8x128xf32>) -> tensor<1x28x8x64xf32>
    %1052 = "tensor.extract_slice"(%1036) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 28, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_40", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x28x8x128xf32>) -> tensor<1x28x8x64xf32>
    %1053 = tensor.empty() : tensor<1x28x8x64xf32>
    %1054 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1052 : tensor<1x28x8x64xf32>) outs(%1053 : tensor<1x28x8x64xf32>) attrs =  {prov.region_id = "neg_4", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb110(%1055: f32, %1056: f32):
      %1057 = arith.negf %1055 : f32
      linalg.yield %1057 : f32
    } -> tensor<1x28x8x64xf32>
    %1058 = tensor.concat dim(3) %1054, %1051 {prov.region_id = "cat_6", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x28x8x64xf32>, tensor<1x28x8x64xf32>) -> tensor<1x28x8x128xf32>
    %1059 = tensor.empty() : tensor<1x28x8x128xf32>
    %1060 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1058, %1044 : tensor<1x28x8x128xf32>, tensor<1x1x8x128xf32>) outs(%1059 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb111(%1061: f32, %1062: f32, %1063: f32):
      %1064 = arith.mulf %1061, %1062 : f32
      linalg.yield %1064 : f32
    } -> tensor<1x28x8x128xf32>
    %1065 = tensor.empty() : tensor<1x28x8x128xf32>
    %1066 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1046, %1060 : tensor<1x28x8x128xf32>, tensor<1x28x8x128xf32>) outs(%1065 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb112(%1067: f32, %1068: f32, %1069: f32):
      %1070 = arith.addf %1067, %1068 : f32
      linalg.yield %1070 : f32
    } -> tensor<1x28x8x128xf32>
    %1071 = tensor.empty() : tensor<1x4x8x128xf32>
    %1072 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1038, %1042 : tensor<1x4x8x128xf32>, tensor<1x1x8x128xf32>) outs(%1071 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb113(%1073: f32, %1074: f32, %1075: f32):
      %1076 = arith.mulf %1073, %1074 : f32
      linalg.yield %1076 : f32
    } -> tensor<1x4x8x128xf32>
    %1077 = "tensor.extract_slice"(%1038) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_41", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x64xf32>
    %1078 = "tensor.extract_slice"(%1038) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_42", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x64xf32>
    %1079 = tensor.empty() : tensor<1x4x8x64xf32>
    %1080 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1078 : tensor<1x4x8x64xf32>) outs(%1079 : tensor<1x4x8x64xf32>) attrs =  {prov.region_id = "neg_5", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb114(%1081: f32, %1082: f32):
      %1083 = arith.negf %1081 : f32
      linalg.yield %1083 : f32
    } -> tensor<1x4x8x64xf32>
    %1084 = tensor.concat dim(3) %1080, %1077 {prov.region_id = "cat_7", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x4x8x64xf32>, tensor<1x4x8x64xf32>) -> tensor<1x4x8x128xf32>
    %1085 = tensor.empty() : tensor<1x4x8x128xf32>
    %1086 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1084, %1044 : tensor<1x4x8x128xf32>, tensor<1x1x8x128xf32>) outs(%1085 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb115(%1087: f32, %1088: f32, %1089: f32):
      %1090 = arith.mulf %1087, %1088 : f32
      linalg.yield %1090 : f32
    } -> tensor<1x4x8x128xf32>
    %1091 = tensor.empty() : tensor<1x4x8x128xf32>
    %1092 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1072, %1086 : tensor<1x4x8x128xf32>, tensor<1x4x8x128xf32>) outs(%1091 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb116(%1093: f32, %1094: f32, %1095: f32):
      %1096 = arith.addf %1093, %1094 : f32
      linalg.yield %1096 : f32
    } -> tensor<1x4x8x128xf32>
    %1097 = "tensor.extract_slice"(%1092) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_43", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32>
    %1098 = "tensor.extract_slice"(%1097) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_44", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32>
    %1099 = tensor.collapse_shape %1098 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x4x8x128xf32> into tensor<4096xf32>
    %1100 = tensor.expand_shape %1099 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 8, 128] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<4096xf32> into tensor<1x4x1x8x128xf32>
    %1101 = "tensor.extract_slice"(%1100) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_45", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x4x1x8x128xf32>) -> tensor<1x4x1x8x128xf32>
    %1102 = "tensor.extract_slice"(%1101) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_46", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x4x1x8x128xf32>) -> tensor<1x4x1x8x128xf32>
    %1103 = tensor.empty() : tensor<1x4x7x8x128xf32>
    %1104 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1102 : tensor<1x4x1x8x128xf32>) outs(%1103 : tensor<1x4x7x8x128xf32>) attrs =  {prov.region_id = "expand_17", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb117(%1105: f32, %1106: f32):
      linalg.yield %1105 : f32
    } -> tensor<1x4x7x8x128xf32>
    %1107 = tensor.collapse_shape %1104 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x4x7x8x128xf32> into tensor<28672xf32>
    %1108 = tensor.expand_shape %1107 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_49", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %1109 = "tensor.extract_slice"(%1040) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_47", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32>
    %1110 = "tensor.extract_slice"(%1109) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_48", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32>
    %1111 = tensor.collapse_shape %1110 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x4x8x128xf32> into tensor<4096xf32>
    %1112 = tensor.expand_shape %1111 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 8, 128] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<4096xf32> into tensor<1x4x1x8x128xf32>
    %1113 = "tensor.extract_slice"(%1112) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_49", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x4x1x8x128xf32>) -> tensor<1x4x1x8x128xf32>
    %1114 = "tensor.extract_slice"(%1113) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_50", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x4x1x8x128xf32>) -> tensor<1x4x1x8x128xf32>
    %1115 = tensor.empty() : tensor<1x4x7x8x128xf32>
    %1116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1114 : tensor<1x4x1x8x128xf32>) outs(%1115 : tensor<1x4x7x8x128xf32>) attrs =  {prov.region_id = "expand_18", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb118(%1117: f32, %1118: f32):
      linalg.yield %1117 : f32
    } -> tensor<1x4x7x8x128xf32>
    %1119 = tensor.collapse_shape %1116 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x4x7x8x128xf32> into tensor<28672xf32>
    %1120 = tensor.expand_shape %1119 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_50", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %1121 = tensor.empty() : tensor<1x28x128x8xf32>
    %1122 = linalg.transpose ins(%1108:tensor<1x28x8x128xf32>) outs(%1121:tensor<1x28x128x8xf32>) permutation = [0, 1, 3, 2]
    %1123 = tensor.empty() : tensor<1x28x8x128xf32>
    %1124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1066 : tensor<1x28x8x128xf32>) outs(%1123 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "expand_19", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb119(%1125: f32, %1126: f32):
      linalg.yield %1125 : f32
    } -> tensor<1x28x8x128xf32>
    %1127 = tensor.collapse_shape %1124 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
    %1128 = tensor.expand_shape %1127 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 128] {prov.region_id = "view_51", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<28672xf32> into tensor<28x8x128xf32>
    %1129 = tensor.empty() : tensor<1x28x128x8xf32>
    %1130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1122 : tensor<1x28x128x8xf32>) outs(%1129 : tensor<1x28x128x8xf32>) attrs =  {prov.region_id = "expand_20", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb120(%1131: f32, %1132: f32):
      linalg.yield %1131 : f32
    } -> tensor<1x28x128x8xf32>
    %1133 = tensor.collapse_shape %1130 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x28x128x8xf32> into tensor<28672xf32>
    %1134 = tensor.expand_shape %1133 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 128, 8] {prov.region_id = "view_52", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<28672xf32> into tensor<28x128x8xf32>
    %1135 = arith.constant {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} 0.000000e+00 : f32
    %1136 = tensor.splat %1135 {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<28x8x8xf32>
    %1137 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1128, %1134 : tensor<28x8x128xf32>, tensor<28x128x8xf32>) outs(%1136 : tensor<28x8x8xf32>) attrs =  {prov.region_id = "matmul_14", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb121(%1138: f32, %1139: f32, %1140: f32):
      %1141 = arith.mulf %1138, %1139 : f32
      %1142 = arith.addf %1140, %1141 : f32
      linalg.yield %1142 : f32
    } -> tensor<28x8x8xf32>
    %1143 = tensor.collapse_shape %1137 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<28x8x8xf32> into tensor<1792xf32>
    %1144 = tensor.expand_shape %1143 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 8] {prov.region_id = "view_53", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1792xf32> into tensor<1x28x8x8xf32>
    %1145 = arith.constant {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} 0.0883883461 : f32
    %1146 = tensor.splat %1145 {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x28x8x8xf32>
    %1147 = tensor.empty() : tensor<1x28x8x8xf32>
    %1148 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1144, %1146 : tensor<1x28x8x8xf32>, tensor<1x28x8x8xf32>) outs(%1147 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb122(%1149: f32, %1150: f32, %1151: f32):
      %1152 = arith.mulf %1149, %1150 : f32
      linalg.yield %1152 : f32
    } -> tensor<1x28x8x8xf32>
    %1153 = tensor.collapse_shape %133 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<8x9xf32> into tensor<72xf32>
    %1154 = tensor.expand_shape %1153 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 9] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<72xf32> into tensor<1x8x9xf32>
    %1155 = tensor.collapse_shape %1154 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x8x9xf32> into tensor<72xf32>
    %1156 = tensor.expand_shape %1155 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 9] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<72xf32> into tensor<1x1x8x9xf32>
    %1157 = "tensor.extract_slice"(%1156) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_51", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %1158 = "tensor.extract_slice"(%1157) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_52", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %1159 = tensor.empty() : tensor<1x1x8x9xf32>
    %1160 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1158 : tensor<1x1x8x9xf32>) outs(%1159 : tensor<1x1x8x9xf32>) attrs =  {prov.region_id = "expand_21", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb123(%1161: f32, %1162: f32):
      linalg.yield %1161 : f32
    } -> tensor<1x1x8x9xf32>
    %1163 = "tensor.extract_slice"(%1160) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_53", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %1164 = "tensor.extract_slice"(%1163) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_54", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %1165 = "tensor.extract_slice"(%1164) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_55", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %1166 = "tensor.extract_slice"(%1165) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_56", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x8xf32>
    %1167 = tensor.empty() : tensor<1x28x8x8xf32>
    %1168 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1148, %1166 : tensor<1x28x8x8xf32>, tensor<1x1x8x8xf32>) outs(%1167 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb124(%1169: f32, %1170: f32, %1171: f32):
      %1172 = arith.addf %1169, %1170 : f32
      linalg.yield %1172 : f32
    } -> tensor<1x28x8x8xf32>
    %1173 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} 0xff800000 : f32
    %1174 = tensor.splat %1173 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x28x8xf32>
    %1175 = linalg.reduce ins(%1168:tensor<1x28x8x8xf32>) outs(%1174:tensor<1x28x8xf32>) dimensions = [3]
    (%1176: f32, %1177: f32) {
      %1178 = arith.maximumf %1176, %1177 : f32
      linalg.yield %1178 : f32
    }
    %1179 = tensor.collapse_shape %1175 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x28x8xf32> into tensor<224xf32>
    %1180 = tensor.expand_shape %1179 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %1181 = tensor.empty() : tensor<1x28x8x8xf32>
    %1182 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1168, %1180 : tensor<1x28x8x8xf32>, tensor<1x28x8x1xf32>) outs(%1181 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb125(%1183: f32, %1184: f32, %1185: f32):
      %1186 = arith.subf %1183, %1184 : f32
      linalg.yield %1186 : f32
    } -> tensor<1x28x8x8xf32>
    %1187 = tensor.empty() : tensor<1x28x8x8xf32>
    %1188 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1182 : tensor<1x28x8x8xf32>) outs(%1187 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb126(%1189: f32, %1190: f32):
      %1191 = math.exp %1189 : f32
      linalg.yield %1191 : f32
    } -> tensor<1x28x8x8xf32>
    %1192 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} 0.000000e+00 : f32
    %1193 = tensor.splat %1192 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x28x8xf32>
    %1194 = linalg.reduce ins(%1188:tensor<1x28x8x8xf32>) outs(%1193:tensor<1x28x8xf32>) dimensions = [3]
    (%1195: f32, %1196: f32) {
      %1197 = arith.addf %1195, %1196 : f32
      linalg.yield %1197 : f32
    }
    %1198 = tensor.collapse_shape %1194 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x28x8xf32> into tensor<224xf32>
    %1199 = tensor.expand_shape %1198 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %1200 = tensor.empty() : tensor<1x28x8x8xf32>
    %1201 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1188, %1199 : tensor<1x28x8x8xf32>, tensor<1x28x8x1xf32>) outs(%1200 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb127(%1202: f32, %1203: f32, %1204: f32):
      %1205 = arith.divf %1202, %1203 : f32
      linalg.yield %1205 : f32
    } -> tensor<1x28x8x8xf32>
    %1206 = tensor.empty() : tensor<1x28x8x8xf32>
    %1207 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1201 : tensor<1x28x8x8xf32>) outs(%1206 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "expand_22", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb128(%1208: f32, %1209: f32):
      linalg.yield %1208 : f32
    } -> tensor<1x28x8x8xf32>
    %1210 = tensor.collapse_shape %1207 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x28x8x8xf32> into tensor<1792xf32>
    %1211 = tensor.expand_shape %1210 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 8] {prov.region_id = "view_54", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1792xf32> into tensor<28x8x8xf32>
    %1212 = tensor.empty() : tensor<1x28x8x128xf32>
    %1213 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1120 : tensor<1x28x8x128xf32>) outs(%1212 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "expand_23", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb129(%1214: f32, %1215: f32):
      linalg.yield %1214 : f32
    } -> tensor<1x28x8x128xf32>
    %1216 = tensor.collapse_shape %1213 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
    %1217 = tensor.expand_shape %1216 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 128] {prov.region_id = "view_55", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<28672xf32> into tensor<28x8x128xf32>
    %1218 = arith.constant {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} 0.000000e+00 : f32
    %1219 = tensor.splat %1218 {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<28x8x128xf32>
    %1220 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1211, %1217 : tensor<28x8x8xf32>, tensor<28x8x128xf32>) outs(%1219 : tensor<28x8x128xf32>) attrs =  {prov.region_id = "matmul_15", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} {
    ^bb130(%1221: f32, %1222: f32, %1223: f32):
      %1224 = arith.mulf %1221, %1222 : f32
      %1225 = arith.addf %1223, %1224 : f32
      linalg.yield %1225 : f32
    } -> tensor<28x8x128xf32>
    %1226 = tensor.collapse_shape %1220 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<28x8x128xf32> into tensor<28672xf32>
    %1227 = tensor.expand_shape %1226 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_56", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %1228 = tensor.empty() : tensor<1x8x28x128xf32>
    %1229 = linalg.transpose ins(%1227:tensor<1x28x8x128xf32>) outs(%1228:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
    %1230 = tensor.collapse_shape %1229 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<1x8x28x128xf32> into tensor<28672xf32>
    %1231 = tensor.expand_shape %1230 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_57", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %1232 = tensor.empty() : tensor<3584x3584xf32>
    %1233 = linalg.transpose ins(%20:tensor<3584x3584xf32>) outs(%1232:tensor<3584x3584xf32>) permutation = [1, 0]
    %1234 = tensor.collapse_shape %1231 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn.attn_out"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %1235 = tensor.expand_shape %1234 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_58", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn.attn_out"} : tensor<28672xf32> into tensor<8x3584xf32>
    %1236 = tensor.empty() : tensor<8x3584xf32>
    %1237 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %1238 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%1237 : f32) outs(%1236 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %1239 = linalg.matmul {prov.region_id = "matmul_16", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn.attn_out", prov.transposed_b = "true"} ins(%1235, %1233 : tensor<8x3584xf32>, tensor<3584x3584xf32>) outs(%1238 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %1240 = tensor.collapse_shape %1239 [[0 : i64, 1 : i64]] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn.attn_out"} : tensor<8x3584xf32> into tensor<28672xf32>
    %1241 = tensor.expand_shape %1240 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_59", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.self_attn.attn_out"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %1242 = tensor.empty() : tensor<1x8x3584xf32>
    %1243 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%958, %1241 : tensor<1x8x3584xf32>, tensor<1x8x3584xf32>) outs(%1242 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2"} {
    ^bb131(%1244: f32, %1245: f32, %1246: f32):
      %1247 = arith.addf %1244, %1245 : f32
      linalg.yield %1247 : f32
    } -> tensor<1x8x3584xf32>
    %1248 = tensor.empty() : tensor<1x8x3584xf32>
    %1249 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1243 : tensor<1x8x3584xf32>) outs(%1248 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb132(%1250: f32, %1251: f32):
      %1252 = arith.constant 2.000000e+00 : f32
      %1253 = math.powf %1250, %1252 : f32
      linalg.yield %1253 : f32
    } -> tensor<1x8x3584xf32>
    %1254 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %1255 = tensor.splat %1254 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %1256 = linalg.reduce ins(%1249:tensor<1x8x3584xf32>) outs(%1255:tensor<1x8xf32>) dimensions = [2]
    (%1257: f32, %1258: f32) {
      %1259 = arith.addf %1257, %1258 : f32
      linalg.yield %1259 : f32
    }
    %1260 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %1261 = tensor.splat %1260 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %1262 = tensor.empty() : tensor<1x8xf32>
    %1263 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1256, %1261 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%1262 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb133(%1264: f32, %1265: f32, %1266: f32):
      %1267 = arith.divf %1264, %1265 : f32
      linalg.yield %1267 : f32
    } -> tensor<1x8xf32>
    %1268 = tensor.collapse_shape %1263 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %1269 = tensor.expand_shape %1268 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %1270 = arith.constant {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %1271 = tensor.splat %1270 {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %1272 = tensor.empty() : tensor<1x8x1xf32>
    %1273 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1269, %1271 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%1272 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb134(%1274: f32, %1275: f32, %1276: f32):
      %1277 = arith.addf %1274, %1275 : f32
      linalg.yield %1277 : f32
    } -> tensor<1x8x1xf32>
    %1278 = tensor.empty() : tensor<1x8x1xf32>
    %1279 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1273 : tensor<1x8x1xf32>) outs(%1278 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_5", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb135(%1280: f32, %1281: f32):
      %1282 = math.rsqrt %1280 : f32
      linalg.yield %1282 : f32
    } -> tensor<1x8x1xf32>
    %1283 = tensor.empty() : tensor<1x8x3584xf32>
    %1284 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1243, %1279 : tensor<1x8x3584xf32>, tensor<1x8x1xf32>) outs(%1283 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb136(%1285: f32, %1286: f32, %1287: f32):
      %1288 = arith.mulf %1285, %1286 : f32
      linalg.yield %1288 : f32
    } -> tensor<1x8x3584xf32>
    %1289 = tensor.empty() : tensor<1x8x3584xf32>
    %1290 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %1284 : tensor<3584xf32>, tensor<1x8x3584xf32>) outs(%1289 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_34", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.ff_norm"} {
    ^bb137(%1291: f32, %1292: f32, %1293: f32):
      %1294 = arith.mulf %1291, %1292 : f32
      linalg.yield %1294 : f32
    } -> tensor<1x8x3584xf32>
    %1295 = tensor.empty() : tensor<3584x37888xf32>
    %1296 = linalg.transpose ins(%21:tensor<37888x3584xf32>) outs(%1295:tensor<3584x37888xf32>) permutation = [1, 0]
    %1297 = tensor.collapse_shape %1290 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.mlp.ff_proj"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %1298 = tensor.expand_shape %1297 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_60", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.mlp.ff_proj"} : tensor<28672xf32> into tensor<8x3584xf32>
    %1299 = tensor.empty() : tensor<8x37888xf32>
    %1300 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %1301 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%1300 : f32) outs(%1299 : tensor<8x37888xf32>) -> tensor<8x37888xf32>
    %1302 = linalg.matmul {prov.region_id = "matmul_17", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.mlp.ff_proj", prov.transposed_b = "true"} ins(%1298, %1296 : tensor<8x3584xf32>, tensor<3584x37888xf32>) outs(%1301 : tensor<8x37888xf32>) -> tensor<8x37888xf32>
    %1303 = tensor.collapse_shape %1302 [[0 : i64, 1 : i64]] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.mlp.ff_proj"} : tensor<8x37888xf32> into tensor<303104xf32>
    %1304 = tensor.expand_shape %1303 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 37888] {prov.region_id = "view_61", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.mlp.ff_proj"} : tensor<303104xf32> into tensor<1x8x37888xf32>
    %1305 = "tensor.extract_slice"(%1304) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_5", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.mlp"} : (tensor<1x8x37888xf32>) -> tensor<1x8x18944xf32>
    %1306 = "tensor.extract_slice"(%1304) <{static_offsets = array<i64: 0, 0, 18944>, static_sizes = array<i64: 1, 8, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_5", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.mlp"} : (tensor<1x8x37888xf32>) -> tensor<1x8x18944xf32>
    %1307 = tensor.empty() : tensor<1x8x18944xf32>
    %1308 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1306 : tensor<1x8x18944xf32>) outs(%1307 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "sigmoid_2", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.mlp.act"} {
    ^bb138(%1309: f32, %1310: f32):
      %1311 = arith.constant 1.000000e+00 : f32
      %1312 = arith.negf %1309 : f32
      %1313 = math.exp %1312 : f32
      %1314 = arith.addf %1311, %1313 : f32
      %1315 = arith.divf %1311, %1314 : f32
      linalg.yield %1315 : f32
    } -> tensor<1x8x18944xf32>
    %1316 = tensor.empty() : tensor<1x8x18944xf32>
    %1317 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1306, %1308 : tensor<1x8x18944xf32>, tensor<1x8x18944xf32>) outs(%1316 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "mul_35", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.mlp.act"} {
    ^bb139(%1318: f32, %1319: f32, %1320: f32):
      %1321 = arith.mulf %1318, %1319 : f32
      linalg.yield %1321 : f32
    } -> tensor<1x8x18944xf32>
    %1322 = tensor.empty() : tensor<1x8x18944xf32>
    %1323 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1317, %1305 : tensor<1x8x18944xf32>, tensor<1x8x18944xf32>) outs(%1322 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "mul_36", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.mlp"} {
    ^bb140(%1324: f32, %1325: f32, %1326: f32):
      %1327 = arith.mulf %1324, %1325 : f32
      linalg.yield %1327 : f32
    } -> tensor<1x8x18944xf32>
    %1328 = tensor.empty() : tensor<18944x3584xf32>
    %1329 = linalg.transpose ins(%22:tensor<3584x18944xf32>) outs(%1328:tensor<18944x3584xf32>) permutation = [1, 0]
    %1330 = tensor.collapse_shape %1323 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.mlp.ff_out"} : tensor<1x8x18944xf32> into tensor<151552xf32>
    %1331 = tensor.expand_shape %1330 [[0 : i64, 1 : i64]] output_shape [8, 18944] {prov.region_id = "view_62", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.mlp.ff_out"} : tensor<151552xf32> into tensor<8x18944xf32>
    %1332 = tensor.empty() : tensor<8x3584xf32>
    %1333 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %1334 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%1333 : f32) outs(%1332 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %1335 = linalg.matmul {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.mlp.ff_out", prov.transposed_b = "true"} ins(%1331, %1329 : tensor<8x18944xf32>, tensor<18944x3584xf32>) outs(%1334 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %1336 = tensor.collapse_shape %1335 [[0 : i64, 1 : i64]] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.mlp.ff_out"} : tensor<8x3584xf32> into tensor<28672xf32>
    %1337 = tensor.expand_shape %1336 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_63", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2.mlp.ff_out"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %1338 = tensor.empty() : tensor<1x8x3584xf32>
    %1339 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1243, %1337 : tensor<1x8x3584xf32>, tensor<1x8x3584xf32>) outs(%1338 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).2"} {
    ^bb141(%1340: f32, %1341: f32, %1342: f32):
      %1343 = arith.addf %1340, %1341 : f32
      linalg.yield %1343 : f32
    } -> tensor<1x8x3584xf32>
    %1344 = tensor.empty() : tensor<1x8x3584xf32>
    %1345 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1339 : tensor<1x8x3584xf32>) outs(%1344 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb142(%1346: f32, %1347: f32):
      %1348 = arith.constant 2.000000e+00 : f32
      %1349 = math.powf %1346, %1348 : f32
      linalg.yield %1349 : f32
    } -> tensor<1x8x3584xf32>
    %1350 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %1351 = tensor.splat %1350 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %1352 = linalg.reduce ins(%1345:tensor<1x8x3584xf32>) outs(%1351:tensor<1x8xf32>) dimensions = [2]
    (%1353: f32, %1354: f32) {
      %1355 = arith.addf %1353, %1354 : f32
      linalg.yield %1355 : f32
    }
    %1356 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %1357 = tensor.splat %1356 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %1358 = tensor.empty() : tensor<1x8xf32>
    %1359 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1352, %1357 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%1358 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb143(%1360: f32, %1361: f32, %1362: f32):
      %1363 = arith.divf %1360, %1361 : f32
      linalg.yield %1363 : f32
    } -> tensor<1x8xf32>
    %1364 = tensor.collapse_shape %1359 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %1365 = tensor.expand_shape %1364 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %1366 = arith.constant {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %1367 = tensor.splat %1366 {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %1368 = tensor.empty() : tensor<1x8x1xf32>
    %1369 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1365, %1367 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%1368 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb144(%1370: f32, %1371: f32, %1372: f32):
      %1373 = arith.addf %1370, %1371 : f32
      linalg.yield %1373 : f32
    } -> tensor<1x8x1xf32>
    %1374 = tensor.empty() : tensor<1x8x1xf32>
    %1375 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1369 : tensor<1x8x1xf32>) outs(%1374 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_6", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb145(%1376: f32, %1377: f32):
      %1378 = math.rsqrt %1376 : f32
      linalg.yield %1378 : f32
    } -> tensor<1x8x1xf32>
    %1379 = tensor.empty() : tensor<1x8x3584xf32>
    %1380 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1339, %1375 : tensor<1x8x3584xf32>, tensor<1x8x1xf32>) outs(%1379 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb146(%1381: f32, %1382: f32, %1383: f32):
      %1384 = arith.mulf %1381, %1382 : f32
      linalg.yield %1384 : f32
    } -> tensor<1x8x3584xf32>
    %1385 = tensor.empty() : tensor<1x8x3584xf32>
    %1386 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %1380 : tensor<3584xf32>, tensor<1x8x3584xf32>) outs(%1385 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_38", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.attn_norm"} {
    ^bb147(%1387: f32, %1388: f32, %1389: f32):
      %1390 = arith.mulf %1387, %1388 : f32
      linalg.yield %1390 : f32
    } -> tensor<1x8x3584xf32>
    %1391 = tensor.collapse_shape %1386 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn.att_proj"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %1392 = tensor.expand_shape %1391 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_64", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn.att_proj"} : tensor<28672xf32> into tensor<8x3584xf32>
    %1393 = tensor.empty() : tensor<3584x4608xf32>
    %1394 = linalg.transpose ins(%25:tensor<4608x3584xf32>) outs(%1393:tensor<3584x4608xf32>) permutation = [1, 0]
    %1395 = tensor.empty() : tensor<8x4608xf32>
    %1396 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %1397 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%1396 : f32) outs(%1395 : tensor<8x4608xf32>) -> tensor<8x4608xf32>
    %1398 = linalg.matmul {prov.region_id = "matmul_19", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn.att_proj", prov.transposed_b = "true"} ins(%1392, %1394 : tensor<8x3584xf32>, tensor<3584x4608xf32>) outs(%1397 : tensor<8x4608xf32>) -> tensor<8x4608xf32>
    %1399 = tensor.empty() : tensor<8x4608xf32>
    %1400 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1398, %26 : tensor<8x4608xf32>, tensor<4608xf32>) outs(%1399 : tensor<8x4608xf32>) attrs =  {prov.region_id = "add_25", prov._pattern_hint = "addmm", prov.op = "addmm", prov.family = "contraction", prov.aten = "aten.addmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn.att_proj"} {
    ^bb148(%1401: f32, %1402: f32, %1403: f32):
      %1404 = arith.addf %1401, %1402 : f32
      linalg.yield %1404 : f32
    } -> tensor<8x4608xf32>
    %1405 = tensor.collapse_shape %1400 [[0 : i64, 1 : i64]] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn.att_proj"} : tensor<8x4608xf32> into tensor<36864xf32>
    %1406 = tensor.expand_shape %1405 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 4608] {prov.region_id = "view_65", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn.att_proj"} : tensor<36864xf32> into tensor<1x8x4608xf32>
    %1407 = "tensor.extract_slice"(%1406) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_6", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x8x4608xf32>) -> tensor<1x8x3584xf32>
    %1408 = "tensor.extract_slice"(%1406) <{static_offsets = array<i64: 0, 0, 3584>, static_sizes = array<i64: 1, 8, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_6", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x8x4608xf32>) -> tensor<1x8x512xf32>
    %1409 = "tensor.extract_slice"(%1406) <{static_offsets = array<i64: 0, 0, 4096>, static_sizes = array<i64: 1, 8, 512>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_6", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x8x4608xf32>) -> tensor<1x8x512xf32>
    %1410 = tensor.collapse_shape %1409 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x8x512xf32> into tensor<4096xf32>
    %1411 = tensor.expand_shape %1410 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 128] {prov.region_id = "view_66", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<4096xf32> into tensor<1x8x4x128xf32>
    %1412 = tensor.collapse_shape %1407 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %1413 = tensor.expand_shape %1412 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 128] {prov.region_id = "view_67", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<28672xf32> into tensor<1x8x28x128xf32>
    %1414 = tensor.collapse_shape %1408 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x8x512xf32> into tensor<4096xf32>
    %1415 = tensor.expand_shape %1414 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 4, 128] {prov.region_id = "view_68", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<4096xf32> into tensor<1x8x4x128xf32>
    %1416 = tensor.empty() : tensor<1x28x8x128xf32>
    %1417 = linalg.transpose ins(%1413:tensor<1x8x28x128xf32>) outs(%1416:tensor<1x28x8x128xf32>) permutation = [0, 2, 1, 3]
    %1418 = tensor.empty() : tensor<1x4x8x128xf32>
    %1419 = linalg.transpose ins(%1415:tensor<1x8x4x128xf32>) outs(%1418:tensor<1x4x8x128xf32>) permutation = [0, 2, 1, 3]
    %1420 = tensor.empty() : tensor<1x4x8x128xf32>
    %1421 = linalg.transpose ins(%1411:tensor<1x8x4x128xf32>) outs(%1420:tensor<1x4x8x128xf32>) permutation = [0, 2, 1, 3]
    %1422 = tensor.collapse_shape %183 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1423 = tensor.expand_shape %1422 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 128] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1024xf32> into tensor<1x1x8x128xf32>
    %1424 = tensor.collapse_shape %196 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x8x128xf32> into tensor<1024xf32>
    %1425 = tensor.expand_shape %1424 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 128] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1024xf32> into tensor<1x1x8x128xf32>
    %1426 = tensor.empty() : tensor<1x28x8x128xf32>
    %1427 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1417, %1423 : tensor<1x28x8x128xf32>, tensor<1x1x8x128xf32>) outs(%1426 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_39", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb149(%1428: f32, %1429: f32, %1430: f32):
      %1431 = arith.mulf %1428, %1429 : f32
      linalg.yield %1431 : f32
    } -> tensor<1x28x8x128xf32>
    %1432 = "tensor.extract_slice"(%1417) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 28, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_57", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x28x8x128xf32>) -> tensor<1x28x8x64xf32>
    %1433 = "tensor.extract_slice"(%1417) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 28, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_58", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x28x8x128xf32>) -> tensor<1x28x8x64xf32>
    %1434 = tensor.empty() : tensor<1x28x8x64xf32>
    %1435 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1433 : tensor<1x28x8x64xf32>) outs(%1434 : tensor<1x28x8x64xf32>) attrs =  {prov.region_id = "neg_6", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb150(%1436: f32, %1437: f32):
      %1438 = arith.negf %1436 : f32
      linalg.yield %1438 : f32
    } -> tensor<1x28x8x64xf32>
    %1439 = tensor.concat dim(3) %1435, %1432 {prov.region_id = "cat_8", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x28x8x64xf32>, tensor<1x28x8x64xf32>) -> tensor<1x28x8x128xf32>
    %1440 = tensor.empty() : tensor<1x28x8x128xf32>
    %1441 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1439, %1425 : tensor<1x28x8x128xf32>, tensor<1x1x8x128xf32>) outs(%1440 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_40", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb151(%1442: f32, %1443: f32, %1444: f32):
      %1445 = arith.mulf %1442, %1443 : f32
      linalg.yield %1445 : f32
    } -> tensor<1x28x8x128xf32>
    %1446 = tensor.empty() : tensor<1x28x8x128xf32>
    %1447 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1427, %1441 : tensor<1x28x8x128xf32>, tensor<1x28x8x128xf32>) outs(%1446 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "add_26", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb152(%1448: f32, %1449: f32, %1450: f32):
      %1451 = arith.addf %1448, %1449 : f32
      linalg.yield %1451 : f32
    } -> tensor<1x28x8x128xf32>
    %1452 = tensor.empty() : tensor<1x4x8x128xf32>
    %1453 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1419, %1423 : tensor<1x4x8x128xf32>, tensor<1x1x8x128xf32>) outs(%1452 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb153(%1454: f32, %1455: f32, %1456: f32):
      %1457 = arith.mulf %1454, %1455 : f32
      linalg.yield %1457 : f32
    } -> tensor<1x4x8x128xf32>
    %1458 = "tensor.extract_slice"(%1419) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_59", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x64xf32>
    %1459 = "tensor.extract_slice"(%1419) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 4, 8, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_60", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x64xf32>
    %1460 = tensor.empty() : tensor<1x4x8x64xf32>
    %1461 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1459 : tensor<1x4x8x64xf32>) outs(%1460 : tensor<1x4x8x64xf32>) attrs =  {prov.region_id = "neg_7", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb154(%1462: f32, %1463: f32):
      %1464 = arith.negf %1462 : f32
      linalg.yield %1464 : f32
    } -> tensor<1x4x8x64xf32>
    %1465 = tensor.concat dim(3) %1461, %1458 {prov.region_id = "cat_9", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x4x8x64xf32>, tensor<1x4x8x64xf32>) -> tensor<1x4x8x128xf32>
    %1466 = tensor.empty() : tensor<1x4x8x128xf32>
    %1467 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1465, %1425 : tensor<1x4x8x128xf32>, tensor<1x1x8x128xf32>) outs(%1466 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "mul_42", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb155(%1468: f32, %1469: f32, %1470: f32):
      %1471 = arith.mulf %1468, %1469 : f32
      linalg.yield %1471 : f32
    } -> tensor<1x4x8x128xf32>
    %1472 = tensor.empty() : tensor<1x4x8x128xf32>
    %1473 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1453, %1467 : tensor<1x4x8x128xf32>, tensor<1x4x8x128xf32>) outs(%1472 : tensor<1x4x8x128xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb156(%1474: f32, %1475: f32, %1476: f32):
      %1477 = arith.addf %1474, %1475 : f32
      linalg.yield %1477 : f32
    } -> tensor<1x4x8x128xf32>
    %1478 = "tensor.extract_slice"(%1473) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_61", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32>
    %1479 = "tensor.extract_slice"(%1478) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_62", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32>
    %1480 = tensor.collapse_shape %1479 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_26", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x4x8x128xf32> into tensor<4096xf32>
    %1481 = tensor.expand_shape %1480 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 8, 128] {prov.region_id = "unsqueeze_26", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<4096xf32> into tensor<1x4x1x8x128xf32>
    %1482 = "tensor.extract_slice"(%1481) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_63", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x4x1x8x128xf32>) -> tensor<1x4x1x8x128xf32>
    %1483 = "tensor.extract_slice"(%1482) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_64", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x4x1x8x128xf32>) -> tensor<1x4x1x8x128xf32>
    %1484 = tensor.empty() : tensor<1x4x7x8x128xf32>
    %1485 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1483 : tensor<1x4x1x8x128xf32>) outs(%1484 : tensor<1x4x7x8x128xf32>) attrs =  {prov.region_id = "expand_24", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb157(%1486: f32, %1487: f32):
      linalg.yield %1486 : f32
    } -> tensor<1x4x7x8x128xf32>
    %1488 = tensor.collapse_shape %1485 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x4x7x8x128xf32> into tensor<28672xf32>
    %1489 = tensor.expand_shape %1488 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_69", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %1490 = "tensor.extract_slice"(%1421) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_65", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32>
    %1491 = "tensor.extract_slice"(%1490) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 8, 128>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_66", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32>
    %1492 = tensor.collapse_shape %1491 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_27", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x4x8x128xf32> into tensor<4096xf32>
    %1493 = tensor.expand_shape %1492 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4, 1, 8, 128] {prov.region_id = "unsqueeze_27", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<4096xf32> into tensor<1x4x1x8x128xf32>
    %1494 = "tensor.extract_slice"(%1493) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_67", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x4x1x8x128xf32>) -> tensor<1x4x1x8x128xf32>
    %1495 = "tensor.extract_slice"(%1494) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 4, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_68", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x4x1x8x128xf32>) -> tensor<1x4x1x8x128xf32>
    %1496 = tensor.empty() : tensor<1x4x7x8x128xf32>
    %1497 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1495 : tensor<1x4x1x8x128xf32>) outs(%1496 : tensor<1x4x7x8x128xf32>) attrs =  {prov.region_id = "expand_25", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb158(%1498: f32, %1499: f32):
      linalg.yield %1498 : f32
    } -> tensor<1x4x7x8x128xf32>
    %1500 = tensor.collapse_shape %1497 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x4x7x8x128xf32> into tensor<28672xf32>
    %1501 = tensor.expand_shape %1500 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_70", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %1502 = tensor.empty() : tensor<1x28x128x8xf32>
    %1503 = linalg.transpose ins(%1489:tensor<1x28x8x128xf32>) outs(%1502:tensor<1x28x128x8xf32>) permutation = [0, 1, 3, 2]
    %1504 = tensor.empty() : tensor<1x28x8x128xf32>
    %1505 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1447 : tensor<1x28x8x128xf32>) outs(%1504 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "expand_26", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb159(%1506: f32, %1507: f32):
      linalg.yield %1506 : f32
    } -> tensor<1x28x8x128xf32>
    %1508 = tensor.collapse_shape %1505 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
    %1509 = tensor.expand_shape %1508 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 128] {prov.region_id = "view_71", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<28672xf32> into tensor<28x8x128xf32>
    %1510 = tensor.empty() : tensor<1x28x128x8xf32>
    %1511 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1503 : tensor<1x28x128x8xf32>) outs(%1510 : tensor<1x28x128x8xf32>) attrs =  {prov.region_id = "expand_27", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb160(%1512: f32, %1513: f32):
      linalg.yield %1512 : f32
    } -> tensor<1x28x128x8xf32>
    %1514 = tensor.collapse_shape %1511 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x28x128x8xf32> into tensor<28672xf32>
    %1515 = tensor.expand_shape %1514 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 128, 8] {prov.region_id = "view_72", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<28672xf32> into tensor<28x128x8xf32>
    %1516 = arith.constant {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} 0.000000e+00 : f32
    %1517 = tensor.splat %1516 {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<28x8x8xf32>
    %1518 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1509, %1515 : tensor<28x8x128xf32>, tensor<28x128x8xf32>) outs(%1517 : tensor<28x8x8xf32>) attrs =  {prov.region_id = "matmul_20", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb161(%1519: f32, %1520: f32, %1521: f32):
      %1522 = arith.mulf %1519, %1520 : f32
      %1523 = arith.addf %1521, %1522 : f32
      linalg.yield %1523 : f32
    } -> tensor<28x8x8xf32>
    %1524 = tensor.collapse_shape %1518 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<28x8x8xf32> into tensor<1792xf32>
    %1525 = tensor.expand_shape %1524 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 8] {prov.region_id = "view_73", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1792xf32> into tensor<1x28x8x8xf32>
    %1526 = arith.constant {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} 0.0883883461 : f32
    %1527 = tensor.splat %1526 {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x28x8x8xf32>
    %1528 = tensor.empty() : tensor<1x28x8x8xf32>
    %1529 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1525, %1527 : tensor<1x28x8x8xf32>, tensor<1x28x8x8xf32>) outs(%1528 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb162(%1530: f32, %1531: f32, %1532: f32):
      %1533 = arith.mulf %1530, %1531 : f32
      linalg.yield %1533 : f32
    } -> tensor<1x28x8x8xf32>
    %1534 = tensor.collapse_shape %133 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_28", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<8x9xf32> into tensor<72xf32>
    %1535 = tensor.expand_shape %1534 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 9] {prov.region_id = "unsqueeze_28", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<72xf32> into tensor<1x8x9xf32>
    %1536 = tensor.collapse_shape %1535 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_29", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x8x9xf32> into tensor<72xf32>
    %1537 = tensor.expand_shape %1536 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 8, 9] {prov.region_id = "unsqueeze_29", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<72xf32> into tensor<1x1x8x9xf32>
    %1538 = "tensor.extract_slice"(%1537) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_69", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %1539 = "tensor.extract_slice"(%1538) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_70", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %1540 = tensor.empty() : tensor<1x1x8x9xf32>
    %1541 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1539 : tensor<1x1x8x9xf32>) outs(%1540 : tensor<1x1x8x9xf32>) attrs =  {prov.region_id = "expand_28", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb163(%1542: f32, %1543: f32):
      linalg.yield %1542 : f32
    } -> tensor<1x1x8x9xf32>
    %1544 = "tensor.extract_slice"(%1541) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_71", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %1545 = "tensor.extract_slice"(%1544) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_72", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %1546 = "tensor.extract_slice"(%1545) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 9>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_73", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x9xf32>
    %1547 = "tensor.extract_slice"(%1546) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 8, 8>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_74", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : (tensor<1x1x8x9xf32>) -> tensor<1x1x8x8xf32>
    %1548 = tensor.empty() : tensor<1x28x8x8xf32>
    %1549 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1529, %1547 : tensor<1x28x8x8xf32>, tensor<1x1x8x8xf32>) outs(%1548 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb164(%1550: f32, %1551: f32, %1552: f32):
      %1553 = arith.addf %1550, %1551 : f32
      linalg.yield %1553 : f32
    } -> tensor<1x28x8x8xf32>
    %1554 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} 0xff800000 : f32
    %1555 = tensor.splat %1554 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x28x8xf32>
    %1556 = linalg.reduce ins(%1549:tensor<1x28x8x8xf32>) outs(%1555:tensor<1x28x8xf32>) dimensions = [3]
    (%1557: f32, %1558: f32) {
      %1559 = arith.maximumf %1557, %1558 : f32
      linalg.yield %1559 : f32
    }
    %1560 = tensor.collapse_shape %1556 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x28x8xf32> into tensor<224xf32>
    %1561 = tensor.expand_shape %1560 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %1562 = tensor.empty() : tensor<1x28x8x8xf32>
    %1563 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1549, %1561 : tensor<1x28x8x8xf32>, tensor<1x28x8x1xf32>) outs(%1562 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb165(%1564: f32, %1565: f32, %1566: f32):
      %1567 = arith.subf %1564, %1565 : f32
      linalg.yield %1567 : f32
    } -> tensor<1x28x8x8xf32>
    %1568 = tensor.empty() : tensor<1x28x8x8xf32>
    %1569 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1563 : tensor<1x28x8x8xf32>) outs(%1568 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb166(%1570: f32, %1571: f32):
      %1572 = math.exp %1570 : f32
      linalg.yield %1572 : f32
    } -> tensor<1x28x8x8xf32>
    %1573 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} 0.000000e+00 : f32
    %1574 = tensor.splat %1573 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x28x8xf32>
    %1575 = linalg.reduce ins(%1569:tensor<1x28x8x8xf32>) outs(%1574:tensor<1x28x8xf32>) dimensions = [3]
    (%1576: f32, %1577: f32) {
      %1578 = arith.addf %1576, %1577 : f32
      linalg.yield %1578 : f32
    }
    %1579 = tensor.collapse_shape %1575 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x28x8xf32> into tensor<224xf32>
    %1580 = tensor.expand_shape %1579 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<224xf32> into tensor<1x28x8x1xf32>
    %1581 = tensor.empty() : tensor<1x28x8x8xf32>
    %1582 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1569, %1580 : tensor<1x28x8x8xf32>, tensor<1x28x8x1xf32>) outs(%1581 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb167(%1583: f32, %1584: f32, %1585: f32):
      %1586 = arith.divf %1583, %1584 : f32
      linalg.yield %1586 : f32
    } -> tensor<1x28x8x8xf32>
    %1587 = tensor.empty() : tensor<1x28x8x8xf32>
    %1588 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1582 : tensor<1x28x8x8xf32>) outs(%1587 : tensor<1x28x8x8xf32>) attrs =  {prov.region_id = "expand_29", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb168(%1589: f32, %1590: f32):
      linalg.yield %1589 : f32
    } -> tensor<1x28x8x8xf32>
    %1591 = tensor.collapse_shape %1588 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x28x8x8xf32> into tensor<1792xf32>
    %1592 = tensor.expand_shape %1591 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 8] {prov.region_id = "view_74", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1792xf32> into tensor<28x8x8xf32>
    %1593 = tensor.empty() : tensor<1x28x8x128xf32>
    %1594 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1501 : tensor<1x28x8x128xf32>) outs(%1593 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "expand_30", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb169(%1595: f32, %1596: f32):
      linalg.yield %1595 : f32
    } -> tensor<1x28x8x128xf32>
    %1597 = tensor.collapse_shape %1594 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
    %1598 = tensor.expand_shape %1597 [[0 : i64, 1 : i64, 2 : i64]] output_shape [28, 8, 128] {prov.region_id = "view_75", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<28672xf32> into tensor<28x8x128xf32>
    %1599 = arith.constant {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} 0.000000e+00 : f32
    %1600 = tensor.splat %1599 {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<28x8x128xf32>
    %1601 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%1592, %1598 : tensor<28x8x8xf32>, tensor<28x8x128xf32>) outs(%1600 : tensor<28x8x128xf32>) attrs =  {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "batch_matmul", prov.op = "batch_matmul", prov.aten = "aten.bmm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} {
    ^bb170(%1602: f32, %1603: f32, %1604: f32):
      %1605 = arith.mulf %1602, %1603 : f32
      %1606 = arith.addf %1604, %1605 : f32
      linalg.yield %1606 : f32
    } -> tensor<28x8x128xf32>
    %1607 = tensor.collapse_shape %1601 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<28x8x128xf32> into tensor<28672xf32>
    %1608 = tensor.expand_shape %1607 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_76", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
    %1609 = tensor.empty() : tensor<1x8x28x128xf32>
    %1610 = linalg.transpose ins(%1608:tensor<1x28x8x128xf32>) outs(%1609:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
    %1611 = tensor.collapse_shape %1610 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<1x8x28x128xf32> into tensor<28672xf32>
    %1612 = tensor.expand_shape %1611 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_77", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %1613 = tensor.empty() : tensor<3584x3584xf32>
    %1614 = linalg.transpose ins(%27:tensor<3584x3584xf32>) outs(%1613:tensor<3584x3584xf32>) permutation = [1, 0]
    %1615 = tensor.collapse_shape %1612 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn.attn_out"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %1616 = tensor.expand_shape %1615 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_78", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn.attn_out"} : tensor<28672xf32> into tensor<8x3584xf32>
    %1617 = tensor.empty() : tensor<8x3584xf32>
    %1618 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %1619 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%1618 : f32) outs(%1617 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %1620 = linalg.matmul {prov.region_id = "matmul_22", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn.attn_out", prov.transposed_b = "true"} ins(%1616, %1614 : tensor<8x3584xf32>, tensor<3584x3584xf32>) outs(%1619 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %1621 = tensor.collapse_shape %1620 [[0 : i64, 1 : i64]] {prov.region_id = "view_79", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn.attn_out"} : tensor<8x3584xf32> into tensor<28672xf32>
    %1622 = tensor.expand_shape %1621 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_79", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.self_attn.attn_out"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %1623 = tensor.empty() : tensor<1x8x3584xf32>
    %1624 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1339, %1622 : tensor<1x8x3584xf32>, tensor<1x8x3584xf32>) outs(%1623 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "add_29", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3"} {
    ^bb171(%1625: f32, %1626: f32, %1627: f32):
      %1628 = arith.addf %1625, %1626 : f32
      linalg.yield %1628 : f32
    } -> tensor<1x8x3584xf32>
    %1629 = tensor.empty() : tensor<1x8x3584xf32>
    %1630 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1624 : tensor<1x8x3584xf32>) outs(%1629 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb172(%1631: f32, %1632: f32):
      %1633 = arith.constant 2.000000e+00 : f32
      %1634 = math.powf %1631, %1633 : f32
      linalg.yield %1634 : f32
    } -> tensor<1x8x3584xf32>
    %1635 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %1636 = tensor.splat %1635 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %1637 = linalg.reduce ins(%1630:tensor<1x8x3584xf32>) outs(%1636:tensor<1x8xf32>) dimensions = [2]
    (%1638: f32, %1639: f32) {
      %1640 = arith.addf %1638, %1639 : f32
      linalg.yield %1640 : f32
    }
    %1641 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %1642 = tensor.splat %1641 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %1643 = tensor.empty() : tensor<1x8xf32>
    %1644 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1637, %1642 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%1643 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb173(%1645: f32, %1646: f32, %1647: f32):
      %1648 = arith.divf %1645, %1646 : f32
      linalg.yield %1648 : f32
    } -> tensor<1x8xf32>
    %1649 = tensor.collapse_shape %1644 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %1650 = tensor.expand_shape %1649 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %1651 = arith.constant {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %1652 = tensor.splat %1651 {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %1653 = tensor.empty() : tensor<1x8x1xf32>
    %1654 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1650, %1652 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%1653 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb174(%1655: f32, %1656: f32, %1657: f32):
      %1658 = arith.addf %1655, %1656 : f32
      linalg.yield %1658 : f32
    } -> tensor<1x8x1xf32>
    %1659 = tensor.empty() : tensor<1x8x1xf32>
    %1660 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1654 : tensor<1x8x1xf32>) outs(%1659 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_7", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb175(%1661: f32, %1662: f32):
      %1663 = math.rsqrt %1661 : f32
      linalg.yield %1663 : f32
    } -> tensor<1x8x1xf32>
    %1664 = tensor.empty() : tensor<1x8x3584xf32>
    %1665 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1624, %1660 : tensor<1x8x3584xf32>, tensor<1x8x1xf32>) outs(%1664 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_44", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb176(%1666: f32, %1667: f32, %1668: f32):
      %1669 = arith.mulf %1666, %1667 : f32
      linalg.yield %1669 : f32
    } -> tensor<1x8x3584xf32>
    %1670 = tensor.empty() : tensor<1x8x3584xf32>
    %1671 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %1665 : tensor<3584xf32>, tensor<1x8x3584xf32>) outs(%1670 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.ff_norm"} {
    ^bb177(%1672: f32, %1673: f32, %1674: f32):
      %1675 = arith.mulf %1672, %1673 : f32
      linalg.yield %1675 : f32
    } -> tensor<1x8x3584xf32>
    %1676 = tensor.empty() : tensor<3584x37888xf32>
    %1677 = linalg.transpose ins(%28:tensor<37888x3584xf32>) outs(%1676:tensor<3584x37888xf32>) permutation = [1, 0]
    %1678 = tensor.collapse_shape %1671 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_80", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.mlp.ff_proj"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %1679 = tensor.expand_shape %1678 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_80", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.mlp.ff_proj"} : tensor<28672xf32> into tensor<8x3584xf32>
    %1680 = tensor.empty() : tensor<8x37888xf32>
    %1681 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %1682 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%1681 : f32) outs(%1680 : tensor<8x37888xf32>) -> tensor<8x37888xf32>
    %1683 = linalg.matmul {prov.region_id = "matmul_23", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.mlp.ff_proj", prov.transposed_b = "true"} ins(%1679, %1677 : tensor<8x3584xf32>, tensor<3584x37888xf32>) outs(%1682 : tensor<8x37888xf32>) -> tensor<8x37888xf32>
    %1684 = tensor.collapse_shape %1683 [[0 : i64, 1 : i64]] {prov.region_id = "view_81", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.mlp.ff_proj"} : tensor<8x37888xf32> into tensor<303104xf32>
    %1685 = tensor.expand_shape %1684 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 37888] {prov.region_id = "view_81", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.mlp.ff_proj"} : tensor<303104xf32> into tensor<1x8x37888xf32>
    %1686 = "tensor.extract_slice"(%1685) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_7", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.mlp"} : (tensor<1x8x37888xf32>) -> tensor<1x8x18944xf32>
    %1687 = "tensor.extract_slice"(%1685) <{static_offsets = array<i64: 0, 0, 18944>, static_sizes = array<i64: 1, 8, 18944>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_7", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.mlp"} : (tensor<1x8x37888xf32>) -> tensor<1x8x18944xf32>
    %1688 = tensor.empty() : tensor<1x8x18944xf32>
    %1689 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1687 : tensor<1x8x18944xf32>) outs(%1688 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "sigmoid_3", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.mlp.act"} {
    ^bb178(%1690: f32, %1691: f32):
      %1692 = arith.constant 1.000000e+00 : f32
      %1693 = arith.negf %1690 : f32
      %1694 = math.exp %1693 : f32
      %1695 = arith.addf %1692, %1694 : f32
      %1696 = arith.divf %1692, %1695 : f32
      linalg.yield %1696 : f32
    } -> tensor<1x8x18944xf32>
    %1697 = tensor.empty() : tensor<1x8x18944xf32>
    %1698 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1687, %1689 : tensor<1x8x18944xf32>, tensor<1x8x18944xf32>) outs(%1697 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "mul_46", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.mlp.act"} {
    ^bb179(%1699: f32, %1700: f32, %1701: f32):
      %1702 = arith.mulf %1699, %1700 : f32
      linalg.yield %1702 : f32
    } -> tensor<1x8x18944xf32>
    %1703 = tensor.empty() : tensor<1x8x18944xf32>
    %1704 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1698, %1686 : tensor<1x8x18944xf32>, tensor<1x8x18944xf32>) outs(%1703 : tensor<1x8x18944xf32>) attrs =  {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.mlp"} {
    ^bb180(%1705: f32, %1706: f32, %1707: f32):
      %1708 = arith.mulf %1705, %1706 : f32
      linalg.yield %1708 : f32
    } -> tensor<1x8x18944xf32>
    %1709 = tensor.empty() : tensor<18944x3584xf32>
    %1710 = linalg.transpose ins(%29:tensor<3584x18944xf32>) outs(%1709:tensor<18944x3584xf32>) permutation = [1, 0]
    %1711 = tensor.collapse_shape %1704 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_82", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.mlp.ff_out"} : tensor<1x8x18944xf32> into tensor<151552xf32>
    %1712 = tensor.expand_shape %1711 [[0 : i64, 1 : i64]] output_shape [8, 18944] {prov.region_id = "view_82", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.mlp.ff_out"} : tensor<151552xf32> into tensor<8x18944xf32>
    %1713 = tensor.empty() : tensor<8x3584xf32>
    %1714 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %1715 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%1714 : f32) outs(%1713 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %1716 = linalg.matmul {prov.region_id = "matmul_24", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.mlp.ff_out", prov.transposed_b = "true"} ins(%1712, %1710 : tensor<8x18944xf32>, tensor<18944x3584xf32>) outs(%1715 : tensor<8x3584xf32>) -> tensor<8x3584xf32>
    %1717 = tensor.collapse_shape %1716 [[0 : i64, 1 : i64]] {prov.region_id = "view_83", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.mlp.ff_out"} : tensor<8x3584xf32> into tensor<28672xf32>
    %1718 = tensor.expand_shape %1717 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 3584] {prov.region_id = "view_83", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3.mlp.ff_out"} : tensor<28672xf32> into tensor<1x8x3584xf32>
    %1719 = tensor.empty() : tensor<1x8x3584xf32>
    %1720 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1624, %1718 : tensor<1x8x3584xf32>, tensor<1x8x3584xf32>) outs(%1719 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.blocks.slice(None, 4, None).3"} {
    ^bb181(%1721: f32, %1722: f32, %1723: f32):
      %1724 = arith.addf %1721, %1722 : f32
      linalg.yield %1724 : f32
    } -> tensor<1x8x3584xf32>
    %1725 = tensor.empty() : tensor<1x8x3584xf32>
    %1726 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1720 : tensor<1x8x3584xf32>) outs(%1725 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb182(%1727: f32, %1728: f32):
      %1729 = arith.constant 2.000000e+00 : f32
      %1730 = math.powf %1727, %1729 : f32
      linalg.yield %1730 : f32
    } -> tensor<1x8x3584xf32>
    %1731 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %1732 = tensor.splat %1731 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %1733 = linalg.reduce ins(%1726:tensor<1x8x3584xf32>) outs(%1732:tensor<1x8xf32>) dimensions = [2]
    (%1734: f32, %1735: f32) {
      %1736 = arith.addf %1734, %1735 : f32
      linalg.yield %1736 : f32
    }
    %1737 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.584000e+03 : f32
    %1738 = tensor.splat %1737 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32>
    %1739 = tensor.empty() : tensor<1x8xf32>
    %1740 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1733, %1738 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%1739 : tensor<1x8xf32>) attrs =  {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb183(%1741: f32, %1742: f32, %1743: f32):
      %1744 = arith.divf %1741, %1742 : f32
      linalg.yield %1744 : f32
    } -> tensor<1x8xf32>
    %1745 = tensor.collapse_shape %1740 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x8xf32> into tensor<8xf32>
    %1746 = tensor.expand_shape %1745 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 1] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<8xf32> into tensor<1x8x1xf32>
    %1747 = arith.constant {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %1748 = tensor.splat %1747 {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x1xf32>
    %1749 = tensor.empty() : tensor<1x8x1xf32>
    %1750 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1746, %1748 : tensor<1x8x1xf32>, tensor<1x8x1xf32>) outs(%1749 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb184(%1751: f32, %1752: f32, %1753: f32):
      %1754 = arith.addf %1751, %1752 : f32
      linalg.yield %1754 : f32
    } -> tensor<1x8x1xf32>
    %1755 = tensor.empty() : tensor<1x8x1xf32>
    %1756 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1750 : tensor<1x8x1xf32>) outs(%1755 : tensor<1x8x1xf32>) attrs =  {prov.region_id = "rsqrt_8", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb185(%1757: f32, %1758: f32):
      %1759 = math.rsqrt %1757 : f32
      linalg.yield %1759 : f32
    } -> tensor<1x8x1xf32>
    %1760 = tensor.empty() : tensor<1x8x3584xf32>
    %1761 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1720, %1756 : tensor<1x8x3584xf32>, tensor<1x8x1xf32>) outs(%1760 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_48", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb186(%1762: f32, %1763: f32, %1764: f32):
      %1765 = arith.mulf %1762, %1763 : f32
      linalg.yield %1765 : f32
    } -> tensor<1x8x3584xf32>
    %1766 = tensor.empty() : tensor<1x8x3584xf32>
    %1767 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %1761 : tensor<3584xf32>, tensor<1x8x3584xf32>) outs(%1766 : tensor<1x8x3584xf32>) attrs =  {prov.region_id = "mul_49", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.model.ln_f"} {
    ^bb187(%1768: f32, %1769: f32, %1770: f32):
      %1771 = arith.mulf %1768, %1769 : f32
      linalg.yield %1771 : f32
    } -> tensor<1x8x3584xf32>
    %1772 = "tensor.extract_slice"(%1767) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_75", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : (tensor<1x8x3584xf32>) -> tensor<1x8x3584xf32>
    %1773 = "tensor.extract_slice"(%1772) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_76", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : (tensor<1x8x3584xf32>) -> tensor<1x8x3584xf32>
    %1774 = "tensor.extract_slice"(%1773) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 8, 3584>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_77", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm"} : (tensor<1x8x3584xf32>) -> tensor<1x8x3584xf32>
    %1775 = tensor.empty() : tensor<3584x4096xf32>
    %1776 = linalg.transpose ins(%31:tensor<4096x3584xf32>) outs(%1775:tensor<3584x4096xf32>) permutation = [1, 0]
    %1777 = tensor.collapse_shape %1774 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_84", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.lm_head"} : tensor<1x8x3584xf32> into tensor<28672xf32>
    %1778 = tensor.expand_shape %1777 [[0 : i64, 1 : i64]] output_shape [8, 3584] {prov.region_id = "view_84", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.lm_head"} : tensor<28672xf32> into tensor<8x3584xf32>
    %1779 = tensor.empty() : tensor<8x4096xf32>
    %1780 = arith.constant {prov.module = "lm"} 0.000000e+00 : f32
    %1781 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "lm"} ins(%1780 : f32) outs(%1779 : tensor<8x4096xf32>) -> tensor<8x4096xf32>
    %1782 = linalg.matmul {prov.region_id = "matmul_25", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.lm_head", prov.transposed_b = "true"} ins(%1778, %1776 : tensor<8x3584xf32>, tensor<3584x4096xf32>) outs(%1781 : tensor<8x4096xf32>) -> tensor<8x4096xf32>
    %1783 = tensor.collapse_shape %1782 [[0 : i64, 1 : i64]] {prov.region_id = "view_85", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.lm_head"} : tensor<8x4096xf32> into tensor<32768xf32>
    %1784 = tensor.expand_shape %1783 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 8, 4096] {prov.region_id = "view_85", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32", prov.module = "lm", prov.fqn = "lm.lm_head"} : tensor<32768xf32> into tensor<1x8x4096xf32>
    func.return %1784 : tensor<1x8x4096xf32>
  }
}
