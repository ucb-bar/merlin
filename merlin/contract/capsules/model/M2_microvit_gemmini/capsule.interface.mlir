builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "int8_weight_only"} {
  func.func @forward(%0: tensor<32xf32>, %1: tensor<32xf32>, %2: tensor<32xf32>, %3: tensor<32xf32>, %4: tensor<48x1x3x3xf32>, %5: tensor<48xf32>, %6: tensor<36x32xi8>, %7: tensor<32x32xi8>, %8: tensor<32x64xi8>, %9: tensor<32x32xi8>, %10: tensor<32x48xi8>, %11: tensor<48x32xi8>, %12: tensor<32x32xi8>, %13: tensor<48x64xi8>, %14: tensor<16x64xi8>, %15: tensor<64xf32>, %16: tensor<1x16xi8>, %17: tensor<1x16xf32>, %18: tensor<16x16xi8>, %19: tensor<1x1x16x16xf32>, %20: tensor<1x16xf32>) -> tensor<1x16xf32> {
    %21 = tensor.empty() : tensor<4xi64>
    %22 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%21 : tensor<4xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb0(%23: i64):
      %24 = linalg.index 0 : index
      %25 = arith.index_cast %24 : index to i64
      %26 = arith.constant 4 : i64
      %27 = arith.muli %25, %26 : i64
      %28 = arith.constant 0 : i64
      %29 = arith.addi %28, %27 : i64
      linalg.yield %29 : i64
    } -> tensor<4xi64>
    %30 = tensor.expand_shape %22 [[0 : i64, 1 : i64]] output_shape [1, 4] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<4xi64> into tensor<1x4xi64>
    %31 = tensor.empty() : tensor<6xi64>
    %32 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%31 : tensor<6xi64>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb1(%33: i64):
      %34 = linalg.index 0 : index
      %35 = arith.index_cast %34 : index to i64
      %36 = arith.constant 1 : i64
      %37 = arith.muli %35, %36 : i64
      %38 = arith.constant 0 : i64
      %39 = arith.addi %38, %37 : i64
      linalg.yield %39 : i64
    } -> tensor<6xi64>
    %40 = tensor.expand_shape %32 [[0 : i64, 1 : i64]] output_shape [6, 1] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<6xi64> into tensor<6x1xi64>
    %41 = tensor.empty() : tensor<6x4xi64>
    %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%30, %40 : tensor<1x4xi64>, tensor<6x1xi64>) outs(%41 : tensor<6x4xi64>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb2(%43: i64, %44: i64, %45: i64):
      %46 = arith.addi %43, %44 : i64
      linalg.yield %46 : i64
    } -> tensor<6x4xi64>
    %47 = tensor.empty() : tensor<4xi64>
    %48 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%47 : tensor<4xi64>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb3(%49: i64):
      %50 = linalg.index 0 : index
      %51 = arith.index_cast %50 : index to i64
      %52 = arith.constant 4 : i64
      %53 = arith.muli %51, %52 : i64
      %54 = arith.constant 0 : i64
      %55 = arith.addi %54, %53 : i64
      linalg.yield %55 : i64
    } -> tensor<4xi64>
    %56 = tensor.expand_shape %48 [[0 : i64, 1 : i64]] output_shape [1, 4] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<4xi64> into tensor<1x4xi64>
    %57 = tensor.empty() : tensor<6xi64>
    %58 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%57 : tensor<6xi64>) attrs =  {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb4(%59: i64):
      %60 = linalg.index 0 : index
      %61 = arith.index_cast %60 : index to i64
      %62 = arith.constant 1 : i64
      %63 = arith.muli %61, %62 : i64
      %64 = arith.constant 0 : i64
      %65 = arith.addi %64, %63 : i64
      linalg.yield %65 : i64
    } -> tensor<6xi64>
    %66 = tensor.expand_shape %58 [[0 : i64, 1 : i64]] output_shape [6, 1] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<6xi64> into tensor<6x1xi64>
    %67 = tensor.empty() : tensor<6x4xi64>
    %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%56, %66 : tensor<1x4xi64>, tensor<6x1xi64>) outs(%67 : tensor<6x4xi64>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb5(%69: i64, %70: i64, %71: i64):
      %72 = arith.addi %69, %70 : i64
      linalg.yield %72 : i64
    } -> tensor<6x4xi64>
    %73 = arith.constant {prov.region_id = "pad_0", prov.family = "layout", prov._pattern_hint = "pad", prov.op = "pad", prov.aten = "aten.constant_pad_nd.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %74 = tensor.splat %73 {prov.region_id = "pad_0", prov.family = "layout", prov._pattern_hint = "pad", prov.op = "pad", prov.aten = "aten.constant_pad_nd.default", prov.orig_dtype = "float32"} : tensor<1x1x18x18xf32>
    %75 = "tensor.insert_slice"(%19, %74) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 1, 16, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "pad_0", prov.family = "layout", prov._pattern_hint = "pad", prov.op = "pad", prov.aten = "aten.constant_pad_nd.default", prov.orig_dtype = "float32"} : (tensor<1x1x16x16xf32>, tensor<1x1x18x18xf32>) -> tensor<1x1x18x18xf32>
    %76 = tensor.collapse_shape %42 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<6x4xi64> into tensor<24xi64>
    %77 = tensor.expand_shape %76 [[0 : i64, 1 : i64, 2 : i64]] output_shape [6, 4, 1] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<24xi64> into tensor<6x4x1xi64>
    %78 = tensor.collapse_shape %77 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<6x4x1xi64> into tensor<24xi64>
    %79 = tensor.expand_shape %78 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [6, 4, 1, 1] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<24xi64> into tensor<6x4x1x1xi64>
    %80 = tensor.empty() : tensor<1x1x6x4x6x4xf32>
    %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, 0, 0)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%79, %68 : tensor<6x4x1x1xi64>, tensor<6x4xi64>) outs(%80 : tensor<1x1x6x4x6x4xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32"} {
    ^bb6(%82: i64, %83: i64, %84: f32):
      %85 = linalg.index 0 : index
      %86 = linalg.index 1 : index
      %87 = arith.index_cast %82 : i64 to index
      %88 = arith.index_cast %83 : i64 to index
      %89 = tensor.extract %75[%85, %86, %87, %88] : tensor<1x1x18x18xf32>
      linalg.yield %89 : f32
    } -> tensor<1x1x6x4x6x4xf32>
    %90 = tensor.empty() : tensor<1x1x6x6x4x4xf32>
    %91 = linalg.transpose ins(%81:tensor<1x1x6x4x6x4xf32>) outs(%90:tensor<1x1x6x6x4x4xf32>) permutation = [0, 1, 2, 4, 3, 5]
    %92 = tensor.collapse_shape %91 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x6x6x4x4xf32> into tensor<576xf32>
    %93 = tensor.expand_shape %92 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 36, 16] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<576xf32> into tensor<1x36x16xf32>
    %94 = tensor.empty() : tensor<1x16x36xf32>
    %95 = linalg.transpose ins(%93:tensor<1x36x16xf32>) outs(%94:tensor<1x16x36xf32>) permutation = [0, 2, 1]
    %96 = tensor.collapse_shape %95 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x16x36xf32> into tensor<576xf32>
    %97 = tensor.expand_shape %96 [[0 : i64, 1 : i64]] output_shape [16, 36] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<576xf32> into tensor<16x36xf32>
    %98 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 8.000000e-01 : f32
    %99 = tensor.splat %98 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x36xf32>
    %100 = tensor.empty() : tensor<16x36xf32>
    %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%97, %99 : tensor<16x36xf32>, tensor<16x36xf32>) outs(%100 : tensor<16x36xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb7(%102: f32, %103: f32, %104: f32):
      %105 = arith.mulf %102, %103 : f32
      linalg.yield %105 : f32
    } -> tensor<16x36xf32>
    %106 = tensor.empty() : tensor<16x36xf32>
    %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101 : tensor<16x36xf32>) outs(%106 : tensor<16x36xf32>) attrs =  {prov.region_id = "tanh_0", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb8(%108: f32, %109: f32):
      %110 = math.tanh %108 : f32
      linalg.yield %110 : f32
    } -> tensor<16x36xf32>
    %111 = arith.constant {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %112 = tensor.splat %111 {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x36xf32>
    %113 = tensor.empty() : tensor<16x36xf32>
    %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%107, %112 : tensor<16x36xf32>, tensor<16x36xf32>) outs(%113 : tensor<16x36xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb9(%115: f32, %116: f32, %117: f32):
      %118 = arith.mulf %115, %116 : f32
      linalg.yield %118 : f32
    } -> tensor<16x36xf32>
    %119 = tensor.empty() : tensor<16x36xi8>
    %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%114 : tensor<16x36xf32>) outs(%119 : tensor<16x36xi8>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb10(%121: f32, %122: i8):
      %123 = arith.fptosi %121 : f32 to i8
      linalg.yield %123 : i8
    } -> tensor<16x36xi8>
    %124 = tensor.empty() : tensor<16x32xi8>
    %125 = arith.constant 0 : i8
    %126 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%125 : i8) outs(%124 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %127 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%120, %6 : tensor<16x36xi8>, tensor<36x32xi8>) outs(%126 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %128 = tensor.empty() : tensor<16x32xf32>
    %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%127 : tensor<16x32xi8>) outs(%128 : tensor<16x32xf32>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb11(%130: i8, %131: f32):
      %132 = arith.sitofp %130 : i8 to f32
      linalg.yield %132 : f32
    } -> tensor<16x32xf32>
    %133 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} 0.000000e+00 : f32
    %134 = tensor.splat %133 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} : tensor<16xf32>
    %135 = linalg.reduce ins(%129:tensor<16x32xf32>) outs(%134:tensor<16xf32>) dimensions = [1]
    (%136: f32, %137: f32) {
      %138 = arith.addf %136, %137 : f32
      linalg.yield %138 : f32
    }
    %139 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} 3.200000e+01 : f32
    %140 = tensor.splat %139 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} : tensor<16xf32>
    %141 = tensor.empty() : tensor<16xf32>
    %142 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%135, %140 : tensor<16xf32>, tensor<16xf32>) outs(%141 : tensor<16xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb12(%143: f32, %144: f32, %145: f32):
      %146 = arith.divf %143, %144 : f32
      linalg.yield %146 : f32
    } -> tensor<16xf32>
    %147 = tensor.expand_shape %142 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} : tensor<16xf32> into tensor<16x1xf32>
    %148 = tensor.empty() : tensor<16x32xf32>
    %149 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%129, %147 : tensor<16x32xf32>, tensor<16x1xf32>) outs(%148 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb13(%150: f32, %151: f32, %152: f32):
      %153 = arith.subf %150, %151 : f32
      linalg.yield %153 : f32
    } -> tensor<16x32xf32>
    %154 = tensor.empty() : tensor<16x32xf32>
    %155 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%149, %149 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%154 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb14(%156: f32, %157: f32, %158: f32):
      %159 = arith.mulf %156, %157 : f32
      linalg.yield %159 : f32
    } -> tensor<16x32xf32>
    %160 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} 0.000000e+00 : f32
    %161 = tensor.splat %160 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} : tensor<16xf32>
    %162 = linalg.reduce ins(%155:tensor<16x32xf32>) outs(%161:tensor<16xf32>) dimensions = [1]
    (%163: f32, %164: f32) {
      %165 = arith.addf %163, %164 : f32
      linalg.yield %165 : f32
    }
    %166 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} 3.200000e+01 : f32
    %167 = tensor.splat %166 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} : tensor<16xf32>
    %168 = tensor.empty() : tensor<16xf32>
    %169 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%162, %167 : tensor<16xf32>, tensor<16xf32>) outs(%168 : tensor<16xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb15(%170: f32, %171: f32, %172: f32):
      %173 = arith.divf %170, %171 : f32
      linalg.yield %173 : f32
    } -> tensor<16xf32>
    %174 = tensor.expand_shape %169 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} : tensor<16xf32> into tensor<16x1xf32>
    %175 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} 1.000000e-05 : f32
    %176 = tensor.splat %175 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} : tensor<16x1xf32>
    %177 = tensor.empty() : tensor<16x1xf32>
    %178 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%174, %176 : tensor<16x1xf32>, tensor<16x1xf32>) outs(%177 : tensor<16x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb16(%179: f32, %180: f32, %181: f32):
      %182 = arith.addf %179, %180 : f32
      linalg.yield %182 : f32
    } -> tensor<16x1xf32>
    %183 = tensor.empty() : tensor<16x1xf32>
    %184 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%178 : tensor<16x1xf32>) outs(%183 : tensor<16x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb17(%185: f32, %186: f32):
      %187 = math.rsqrt %185 : f32
      linalg.yield %187 : f32
    } -> tensor<16x1xf32>
    %188 = tensor.empty() : tensor<16x32xf32>
    %189 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%149, %184 : tensor<16x32xf32>, tensor<16x1xf32>) outs(%188 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb18(%190: f32, %191: f32, %192: f32):
      %193 = arith.mulf %190, %191 : f32
      linalg.yield %193 : f32
    } -> tensor<16x32xf32>
    %194 = tensor.empty() : tensor<16x32xf32>
    %195 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%189, %0 : tensor<16x32xf32>, tensor<32xf32>) outs(%194 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb19(%196: f32, %197: f32, %198: f32):
      %199 = arith.mulf %196, %197 : f32
      linalg.yield %199 : f32
    } -> tensor<16x32xf32>
    %200 = tensor.empty() : tensor<16x32xf32>
    %201 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%195, %1 : tensor<16x32xf32>, tensor<32xf32>) outs(%200 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb20(%202: f32, %203: f32, %204: f32):
      %205 = arith.addf %202, %203 : f32
      linalg.yield %205 : f32
    } -> tensor<16x32xf32>
    %206 = arith.constant {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.100000e+00 : f32
    %207 = tensor.splat %206 {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %208 = tensor.empty() : tensor<16x32xf32>
    %209 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%201, %207 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%208 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb21(%210: f32, %211: f32, %212: f32):
      %213 = arith.mulf %210, %211 : f32
      linalg.yield %213 : f32
    } -> tensor<16x32xf32>
    %214 = tensor.empty() : tensor<16x32xf32>
    %215 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%209 : tensor<16x32xf32>) outs(%214 : tensor<16x32xf32>) attrs =  {prov.region_id = "tanh_1", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb22(%216: f32, %217: f32):
      %218 = math.tanh %216 : f32
      linalg.yield %218 : f32
    } -> tensor<16x32xf32>
    %219 = arith.constant {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %220 = tensor.splat %219 {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %221 = tensor.empty() : tensor<16x32xf32>
    %222 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%215, %220 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%221 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb23(%223: f32, %224: f32, %225: f32):
      %226 = arith.mulf %223, %224 : f32
      linalg.yield %226 : f32
    } -> tensor<16x32xf32>
    %227 = tensor.empty() : tensor<16x32xi8>
    %228 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%222 : tensor<16x32xf32>) outs(%227 : tensor<16x32xi8>) attrs =  {prov.region_id = "dtype_cast_2", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb24(%229: f32, %230: i8):
      %231 = arith.fptosi %229 : f32 to i8
      linalg.yield %231 : i8
    } -> tensor<16x32xi8>
    %232 = tensor.collapse_shape %228 [[0 : i64, 1 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int8"} : tensor<16x32xi8> into tensor<512xi8>
    %233 = tensor.expand_shape %232 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 4, 32] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int8"} : tensor<512xi8> into tensor<1x4x4x32xi8>
    %234 = tensor.empty() : tensor<1x32x4x4xi8>
    %235 = linalg.transpose ins(%233:tensor<1x4x4x32xi8>) outs(%234:tensor<1x32x4x4xi8>) permutation = [0, 3, 1, 2]
    %236 = tensor.collapse_shape %235 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int8"} : tensor<1x32x4x4xi8> into tensor<512xi8>
    %237 = tensor.expand_shape %236 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] output_shape [1, 32, 2, 2, 2, 2] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int8"} : tensor<512xi8> into tensor<1x32x2x2x2x2xi8>
    %238 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "int8"} -128 : i8
    %239 = tensor.splat %238 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "int8"} : tensor<1x32x2x2x2xi8>
    %240 = linalg.reduce ins(%237:tensor<1x32x2x2x2x2xi8>) outs(%239:tensor<1x32x2x2x2xi8>) dimensions = [5]
    (%241: i8, %242: i8) {
      %243 = arith.maxsi %241, %242 : i8
      linalg.yield %243 : i8
    }
    %244 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "int8"} -128 : i8
    %245 = tensor.splat %244 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "int8"} : tensor<1x32x2x2xi8>
    %246 = linalg.reduce ins(%240:tensor<1x32x2x2x2xi8>) outs(%245:tensor<1x32x2x2xi8>) dimensions = [3]
    (%247: i8, %248: i8) {
      %249 = arith.maxsi %247, %248 : i8
      linalg.yield %249 : i8
    }
    %250 = tensor.empty() : tensor<1x2x2x32xi8>
    %251 = linalg.transpose ins(%246:tensor<1x32x2x2xi8>) outs(%250:tensor<1x2x2x32xi8>) permutation = [0, 2, 3, 1]
    %252 = tensor.collapse_shape %251 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int8"} : tensor<1x2x2x32xi8> into tensor<128xi8>
    %253 = tensor.expand_shape %252 [[0 : i64, 1 : i64]] output_shape [4, 32] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int8"} : tensor<128xi8> into tensor<4x32xi8>
    %254 = tensor.empty() : tensor<4x32xf32>
    %255 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%253 : tensor<4x32xi8>) outs(%254 : tensor<4x32xf32>) attrs =  {prov.region_id = "dtype_cast_3", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb25(%256: i8, %257: f32):
      %258 = arith.sitofp %256 : i8 to f32
      linalg.yield %258 : f32
    } -> tensor<4x32xf32>
    %259 = arith.constant {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 9.000000e-01 : f32
    %260 = tensor.splat %259 {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<4x32xf32>
    %261 = tensor.empty() : tensor<4x32xf32>
    %262 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%255, %260 : tensor<4x32xf32>, tensor<4x32xf32>) outs(%261 : tensor<4x32xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb26(%263: f32, %264: f32, %265: f32):
      %266 = arith.mulf %263, %264 : f32
      linalg.yield %266 : f32
    } -> tensor<4x32xf32>
    %267 = tensor.empty() : tensor<4x32xf32>
    %268 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%262 : tensor<4x32xf32>) outs(%267 : tensor<4x32xf32>) attrs =  {prov.region_id = "tanh_2", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb27(%269: f32, %270: f32):
      %271 = math.tanh %269 : f32
      linalg.yield %271 : f32
    } -> tensor<4x32xf32>
    %272 = arith.constant {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %273 = tensor.splat %272 {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<4x32xf32>
    %274 = tensor.empty() : tensor<4x32xf32>
    %275 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%268, %273 : tensor<4x32xf32>, tensor<4x32xf32>) outs(%274 : tensor<4x32xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb28(%276: f32, %277: f32, %278: f32):
      %279 = arith.mulf %276, %277 : f32
      linalg.yield %279 : f32
    } -> tensor<4x32xf32>
    %280 = tensor.empty() : tensor<4x32xi8>
    %281 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%275 : tensor<4x32xf32>) outs(%280 : tensor<4x32xi8>) attrs =  {prov.region_id = "dtype_cast_4", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb29(%282: f32, %283: i8):
      %284 = arith.fptosi %282 : f32 to i8
      linalg.yield %284 : i8
    } -> tensor<4x32xi8>
    %285 = tensor.empty() : tensor<16x32xi8>
    %286 = arith.constant 0 : i8
    %287 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%286 : i8) outs(%285 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %288 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%228, %7 : tensor<16x32xi8>, tensor<32x32xi8>) outs(%287 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %289 = tensor.empty() : tensor<16x32xf32>
    %290 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%288 : tensor<16x32xi8>) outs(%289 : tensor<16x32xf32>) attrs =  {prov.region_id = "dtype_cast_5", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb30(%291: i8, %292: f32):
      %293 = arith.sitofp %291 : i8 to f32
      linalg.yield %293 : f32
    } -> tensor<16x32xf32>
    %294 = arith.constant {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.500000e-01 : f32
    %295 = tensor.splat %294 {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %296 = tensor.empty() : tensor<16x32xf32>
    %297 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%290, %295 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%296 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb31(%298: f32, %299: f32, %300: f32):
      %301 = arith.mulf %298, %299 : f32
      linalg.yield %301 : f32
    } -> tensor<16x32xf32>
    %302 = tensor.empty() : tensor<16x32xf32>
    %303 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%297 : tensor<16x32xf32>) outs(%302 : tensor<16x32xf32>) attrs =  {prov.region_id = "tanh_3", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb32(%304: f32, %305: f32):
      %306 = math.tanh %304 : f32
      linalg.yield %306 : f32
    } -> tensor<16x32xf32>
    %307 = arith.constant {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %308 = tensor.splat %307 {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %309 = tensor.empty() : tensor<16x32xf32>
    %310 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%303, %308 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%309 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb33(%311: f32, %312: f32, %313: f32):
      %314 = arith.mulf %311, %312 : f32
      linalg.yield %314 : f32
    } -> tensor<16x32xf32>
    %315 = tensor.empty() : tensor<16x32xi8>
    %316 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%310 : tensor<16x32xf32>) outs(%315 : tensor<16x32xi8>) attrs =  {prov.region_id = "dtype_cast_6", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb34(%317: f32, %318: i8):
      %319 = arith.fptosi %317 : f32 to i8
      linalg.yield %319 : i8
    } -> tensor<16x32xi8>
    %320 = tensor.empty() : tensor<4x64xi8>
    %321 = arith.constant 0 : i8
    %322 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%321 : i8) outs(%320 : tensor<4x64xi8>) -> tensor<4x64xi8>
    %323 = linalg.matmul {prov.region_id = "matmul_2", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%281, %8 : tensor<4x32xi8>, tensor<32x64xi8>) outs(%322 : tensor<4x64xi8>) -> tensor<4x64xi8>
    %324 = tensor.empty() : tensor<4x64xf32>
    %325 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%323 : tensor<4x64xi8>) outs(%324 : tensor<4x64xf32>) attrs =  {prov.region_id = "dtype_cast_7", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb35(%326: i8, %327: f32):
      %328 = arith.sitofp %326 : i8 to f32
      linalg.yield %328 : f32
    } -> tensor<4x64xf32>
    %329 = arith.constant {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.500000e-01 : f32
    %330 = tensor.splat %329 {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<4x64xf32>
    %331 = tensor.empty() : tensor<4x64xf32>
    %332 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%325, %330 : tensor<4x64xf32>, tensor<4x64xf32>) outs(%331 : tensor<4x64xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb36(%333: f32, %334: f32, %335: f32):
      %336 = arith.mulf %333, %334 : f32
      linalg.yield %336 : f32
    } -> tensor<4x64xf32>
    %337 = tensor.empty() : tensor<4x64xf32>
    %338 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%332 : tensor<4x64xf32>) outs(%337 : tensor<4x64xf32>) attrs =  {prov.region_id = "tanh_4", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb37(%339: f32, %340: f32):
      %341 = math.tanh %339 : f32
      linalg.yield %341 : f32
    } -> tensor<4x64xf32>
    %342 = arith.constant {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %343 = tensor.splat %342 {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<4x64xf32>
    %344 = tensor.empty() : tensor<4x64xf32>
    %345 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%338, %343 : tensor<4x64xf32>, tensor<4x64xf32>) outs(%344 : tensor<4x64xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb38(%346: f32, %347: f32, %348: f32):
      %349 = arith.mulf %346, %347 : f32
      linalg.yield %349 : f32
    } -> tensor<4x64xf32>
    %350 = tensor.empty() : tensor<4x64xi8>
    %351 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%345 : tensor<4x64xf32>) outs(%350 : tensor<4x64xi8>) attrs =  {prov.region_id = "dtype_cast_8", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb39(%352: f32, %353: i8):
      %354 = arith.fptosi %352 : f32 to i8
      linalg.yield %354 : i8
    } -> tensor<4x64xi8>
    %355 = "tensor.extract_slice"(%351) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 4, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int8"} : (tensor<4x64xi8>) -> tensor<4x32xi8>
    %356 = "tensor.extract_slice"(%351) <{static_offsets = array<i64: 0, 32>, static_sizes = array<i64: 4, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int8"} : (tensor<4x64xi8>) -> tensor<4x32xi8>
    %357 = tensor.empty() : tensor<32x4xi8>
    %358 = linalg.transpose ins(%355:tensor<4x32xi8>) outs(%357:tensor<32x4xi8>) permutation = [1, 0]
    %359 = tensor.empty() : tensor<16x4xi8>
    %360 = arith.constant 0 : i8
    %361 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%360 : i8) outs(%359 : tensor<16x4xi8>) -> tensor<16x4xi8>
    %362 = linalg.matmul {prov.region_id = "matmul_3", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8", prov.transposed_b = "true"} ins(%316, %358 : tensor<16x32xi8>, tensor<32x4xi8>) outs(%361 : tensor<16x4xi8>) -> tensor<16x4xi8>
    %363 = tensor.empty() : tensor<16x4xf32>
    %364 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%362 : tensor<16x4xi8>) outs(%363 : tensor<16x4xf32>) attrs =  {prov.region_id = "dtype_cast_9", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb40(%365: i8, %366: f32):
      %367 = arith.sitofp %365 : i8 to f32
      linalg.yield %367 : f32
    } -> tensor<16x4xf32>
    %368 = arith.constant {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.176776692 : f32
    %369 = tensor.splat %368 {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x4xf32>
    %370 = tensor.empty() : tensor<16x4xf32>
    %371 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%364, %369 : tensor<16x4xf32>, tensor<16x4xf32>) outs(%370 : tensor<16x4xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb41(%372: f32, %373: f32, %374: f32):
      %375 = arith.mulf %372, %373 : f32
      linalg.yield %375 : f32
    } -> tensor<16x4xf32>
    %376 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %377 = tensor.splat %376 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<16xf32>
    %378 = linalg.reduce ins(%371:tensor<16x4xf32>) outs(%377:tensor<16xf32>) dimensions = [1]
    (%379: f32, %380: f32) {
      %381 = arith.maximumf %379, %380 : f32
      linalg.yield %381 : f32
    }
    %382 = tensor.expand_shape %378 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<16xf32> into tensor<16x1xf32>
    %383 = tensor.empty() : tensor<16x4xf32>
    %384 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%371, %382 : tensor<16x4xf32>, tensor<16x1xf32>) outs(%383 : tensor<16x4xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb42(%385: f32, %386: f32, %387: f32):
      %388 = arith.subf %385, %386 : f32
      linalg.yield %388 : f32
    } -> tensor<16x4xf32>
    %389 = tensor.empty() : tensor<16x4xf32>
    %390 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%384 : tensor<16x4xf32>) outs(%389 : tensor<16x4xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb43(%391: f32, %392: f32):
      %393 = math.exp %391 : f32
      linalg.yield %393 : f32
    } -> tensor<16x4xf32>
    %394 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %395 = tensor.splat %394 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<16xf32>
    %396 = linalg.reduce ins(%390:tensor<16x4xf32>) outs(%395:tensor<16xf32>) dimensions = [1]
    (%397: f32, %398: f32) {
      %399 = arith.addf %397, %398 : f32
      linalg.yield %399 : f32
    }
    %400 = tensor.expand_shape %396 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<16xf32> into tensor<16x1xf32>
    %401 = tensor.empty() : tensor<16x4xf32>
    %402 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%390, %400 : tensor<16x4xf32>, tensor<16x1xf32>) outs(%401 : tensor<16x4xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb44(%403: f32, %404: f32, %405: f32):
      %406 = arith.divf %403, %404 : f32
      linalg.yield %406 : f32
    } -> tensor<16x4xf32>
    %407 = arith.constant {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+01 : f32
    %408 = tensor.splat %407 {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x4xf32>
    %409 = tensor.empty() : tensor<16x4xf32>
    %410 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%402, %408 : tensor<16x4xf32>, tensor<16x4xf32>) outs(%409 : tensor<16x4xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb45(%411: f32, %412: f32, %413: f32):
      %414 = arith.mulf %411, %412 : f32
      linalg.yield %414 : f32
    } -> tensor<16x4xf32>
    %415 = tensor.empty() : tensor<16x4xi8>
    %416 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%410 : tensor<16x4xf32>) outs(%415 : tensor<16x4xi8>) attrs =  {prov.region_id = "dtype_cast_10", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb46(%417: f32, %418: i8):
      %419 = arith.fptosi %417 : f32 to i8
      linalg.yield %419 : i8
    } -> tensor<16x4xi8>
    %420 = tensor.empty() : tensor<16x32xi8>
    %421 = arith.constant 0 : i8
    %422 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%421 : i8) outs(%420 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %423 = linalg.matmul {prov.region_id = "matmul_4", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%416, %356 : tensor<16x4xi8>, tensor<4x32xi8>) outs(%422 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %424 = tensor.empty() : tensor<16x32xf32>
    %425 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%423 : tensor<16x32xi8>) outs(%424 : tensor<16x32xf32>) attrs =  {prov.region_id = "dtype_cast_11", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb47(%426: i8, %427: f32):
      %428 = arith.sitofp %426 : i8 to f32
      linalg.yield %428 : f32
    } -> tensor<16x32xf32>
    %429 = arith.constant {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 4.000000e-02 : f32
    %430 = tensor.splat %429 {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %431 = tensor.empty() : tensor<16x32xf32>
    %432 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%425, %430 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%431 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb48(%433: f32, %434: f32, %435: f32):
      %436 = arith.mulf %433, %434 : f32
      linalg.yield %436 : f32
    } -> tensor<16x32xf32>
    %437 = tensor.empty() : tensor<16x32xf32>
    %438 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%432 : tensor<16x32xf32>) outs(%437 : tensor<16x32xf32>) attrs =  {prov.region_id = "tanh_5", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb49(%439: f32, %440: f32):
      %441 = math.tanh %439 : f32
      linalg.yield %441 : f32
    } -> tensor<16x32xf32>
    %442 = arith.constant {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %443 = tensor.splat %442 {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %444 = tensor.empty() : tensor<16x32xf32>
    %445 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%438, %443 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%444 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb50(%446: f32, %447: f32, %448: f32):
      %449 = arith.mulf %446, %447 : f32
      linalg.yield %449 : f32
    } -> tensor<16x32xf32>
    %450 = tensor.empty() : tensor<16x32xi8>
    %451 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%445 : tensor<16x32xf32>) outs(%450 : tensor<16x32xi8>) attrs =  {prov.region_id = "dtype_cast_12", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb51(%452: f32, %453: i8):
      %454 = arith.fptosi %452 : f32 to i8
      linalg.yield %454 : i8
    } -> tensor<16x32xi8>
    %455 = tensor.empty() : tensor<16x32xi8>
    %456 = arith.constant 0 : i8
    %457 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%456 : i8) outs(%455 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %458 = linalg.matmul {prov.region_id = "matmul_5", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%451, %9 : tensor<16x32xi8>, tensor<32x32xi8>) outs(%457 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %459 = tensor.empty() : tensor<16x32xf32>
    %460 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%458 : tensor<16x32xi8>) outs(%459 : tensor<16x32xf32>) attrs =  {prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb52(%461: i8, %462: f32):
      %463 = arith.sitofp %461 : i8 to f32
      linalg.yield %463 : f32
    } -> tensor<16x32xf32>
    %464 = arith.constant {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.500000e-01 : f32
    %465 = tensor.splat %464 {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %466 = tensor.empty() : tensor<16x32xf32>
    %467 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%460, %465 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%466 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb53(%468: f32, %469: f32, %470: f32):
      %471 = arith.mulf %468, %469 : f32
      linalg.yield %471 : f32
    } -> tensor<16x32xf32>
    %472 = tensor.empty() : tensor<16x32xf32>
    %473 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%467 : tensor<16x32xf32>) outs(%472 : tensor<16x32xf32>) attrs =  {prov.region_id = "tanh_6", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb54(%474: f32, %475: f32):
      %476 = math.tanh %474 : f32
      linalg.yield %476 : f32
    } -> tensor<16x32xf32>
    %477 = arith.constant {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %478 = tensor.splat %477 {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %479 = tensor.empty() : tensor<16x32xf32>
    %480 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%473, %478 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%479 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb55(%481: f32, %482: f32, %483: f32):
      %484 = arith.mulf %481, %482 : f32
      linalg.yield %484 : f32
    } -> tensor<16x32xf32>
    %485 = tensor.empty() : tensor<16x32xi8>
    %486 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%480 : tensor<16x32xf32>) outs(%485 : tensor<16x32xi8>) attrs =  {prov.region_id = "dtype_cast_14", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb56(%487: f32, %488: i8):
      %489 = arith.fptosi %487 : f32 to i8
      linalg.yield %489 : i8
    } -> tensor<16x32xi8>
    %490 = tensor.empty() : tensor<16x32xi8>
    %491 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%228, %486 : tensor<16x32xi8>, tensor<16x32xi8>) outs(%490 : tensor<16x32xi8>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int8"} {
    ^bb57(%492: i8, %493: i8, %494: i8):
      %495 = arith.addi %492, %493 : i8
      linalg.yield %495 : i8
    } -> tensor<16x32xi8>
    %496 = tensor.empty() : tensor<16x32xf32>
    %497 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%491 : tensor<16x32xi8>) outs(%496 : tensor<16x32xf32>) attrs =  {prov.region_id = "dtype_cast_15", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb58(%498: i8, %499: f32):
      %500 = arith.sitofp %498 : i8 to f32
      linalg.yield %500 : f32
    } -> tensor<16x32xf32>
    %501 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} 0.000000e+00 : f32
    %502 = tensor.splat %501 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} : tensor<16xf32>
    %503 = linalg.reduce ins(%497:tensor<16x32xf32>) outs(%502:tensor<16xf32>) dimensions = [1]
    (%504: f32, %505: f32) {
      %506 = arith.addf %504, %505 : f32
      linalg.yield %506 : f32
    }
    %507 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} 3.200000e+01 : f32
    %508 = tensor.splat %507 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} : tensor<16xf32>
    %509 = tensor.empty() : tensor<16xf32>
    %510 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%503, %508 : tensor<16xf32>, tensor<16xf32>) outs(%509 : tensor<16xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb59(%511: f32, %512: f32, %513: f32):
      %514 = arith.divf %511, %512 : f32
      linalg.yield %514 : f32
    } -> tensor<16xf32>
    %515 = tensor.expand_shape %510 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} : tensor<16xf32> into tensor<16x1xf32>
    %516 = tensor.empty() : tensor<16x32xf32>
    %517 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%497, %515 : tensor<16x32xf32>, tensor<16x1xf32>) outs(%516 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb60(%518: f32, %519: f32, %520: f32):
      %521 = arith.subf %518, %519 : f32
      linalg.yield %521 : f32
    } -> tensor<16x32xf32>
    %522 = tensor.empty() : tensor<16x32xf32>
    %523 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%517, %517 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%522 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb61(%524: f32, %525: f32, %526: f32):
      %527 = arith.mulf %524, %525 : f32
      linalg.yield %527 : f32
    } -> tensor<16x32xf32>
    %528 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} 0.000000e+00 : f32
    %529 = tensor.splat %528 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} : tensor<16xf32>
    %530 = linalg.reduce ins(%523:tensor<16x32xf32>) outs(%529:tensor<16xf32>) dimensions = [1]
    (%531: f32, %532: f32) {
      %533 = arith.addf %531, %532 : f32
      linalg.yield %533 : f32
    }
    %534 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} 3.200000e+01 : f32
    %535 = tensor.splat %534 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} : tensor<16xf32>
    %536 = tensor.empty() : tensor<16xf32>
    %537 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%530, %535 : tensor<16xf32>, tensor<16xf32>) outs(%536 : tensor<16xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb62(%538: f32, %539: f32, %540: f32):
      %541 = arith.divf %538, %539 : f32
      linalg.yield %541 : f32
    } -> tensor<16xf32>
    %542 = tensor.expand_shape %537 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} : tensor<16xf32> into tensor<16x1xf32>
    %543 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} 1.000000e-05 : f32
    %544 = tensor.splat %543 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} : tensor<16x1xf32>
    %545 = tensor.empty() : tensor<16x1xf32>
    %546 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%542, %544 : tensor<16x1xf32>, tensor<16x1xf32>) outs(%545 : tensor<16x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb63(%547: f32, %548: f32, %549: f32):
      %550 = arith.addf %547, %548 : f32
      linalg.yield %550 : f32
    } -> tensor<16x1xf32>
    %551 = tensor.empty() : tensor<16x1xf32>
    %552 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%546 : tensor<16x1xf32>) outs(%551 : tensor<16x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb64(%553: f32, %554: f32):
      %555 = math.rsqrt %553 : f32
      linalg.yield %555 : f32
    } -> tensor<16x1xf32>
    %556 = tensor.empty() : tensor<16x32xf32>
    %557 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%517, %552 : tensor<16x32xf32>, tensor<16x1xf32>) outs(%556 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb65(%558: f32, %559: f32, %560: f32):
      %561 = arith.mulf %558, %559 : f32
      linalg.yield %561 : f32
    } -> tensor<16x32xf32>
    %562 = tensor.empty() : tensor<16x32xf32>
    %563 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%557, %2 : tensor<16x32xf32>, tensor<32xf32>) outs(%562 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb66(%564: f32, %565: f32, %566: f32):
      %567 = arith.mulf %564, %565 : f32
      linalg.yield %567 : f32
    } -> tensor<16x32xf32>
    %568 = tensor.empty() : tensor<16x32xf32>
    %569 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%563, %3 : tensor<16x32xf32>, tensor<32xf32>) outs(%568 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb67(%570: f32, %571: f32, %572: f32):
      %573 = arith.addf %570, %571 : f32
      linalg.yield %573 : f32
    } -> tensor<16x32xf32>
    %574 = arith.constant {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.100000e+00 : f32
    %575 = tensor.splat %574 {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %576 = tensor.empty() : tensor<16x32xf32>
    %577 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%569, %575 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%576 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb68(%578: f32, %579: f32, %580: f32):
      %581 = arith.mulf %578, %579 : f32
      linalg.yield %581 : f32
    } -> tensor<16x32xf32>
    %582 = tensor.empty() : tensor<16x32xf32>
    %583 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%577 : tensor<16x32xf32>) outs(%582 : tensor<16x32xf32>) attrs =  {prov.region_id = "tanh_7", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb69(%584: f32, %585: f32):
      %586 = math.tanh %584 : f32
      linalg.yield %586 : f32
    } -> tensor<16x32xf32>
    %587 = arith.constant {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %588 = tensor.splat %587 {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %589 = tensor.empty() : tensor<16x32xf32>
    %590 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%583, %588 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%589 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb70(%591: f32, %592: f32, %593: f32):
      %594 = arith.mulf %591, %592 : f32
      linalg.yield %594 : f32
    } -> tensor<16x32xf32>
    %595 = tensor.empty() : tensor<16x32xi8>
    %596 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%590 : tensor<16x32xf32>) outs(%595 : tensor<16x32xi8>) attrs =  {prov.region_id = "dtype_cast_16", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb71(%597: f32, %598: i8):
      %599 = arith.fptosi %597 : f32 to i8
      linalg.yield %599 : i8
    } -> tensor<16x32xi8>
    %600 = tensor.empty() : tensor<16x48xi8>
    %601 = arith.constant 0 : i8
    %602 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%601 : i8) outs(%600 : tensor<16x48xi8>) -> tensor<16x48xi8>
    %603 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%596, %10 : tensor<16x32xi8>, tensor<32x48xi8>) outs(%602 : tensor<16x48xi8>) -> tensor<16x48xi8>
    %604 = tensor.empty() : tensor<16x48xf32>
    %605 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%603 : tensor<16x48xi8>) outs(%604 : tensor<16x48xf32>) attrs =  {prov.region_id = "dtype_cast_17", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb72(%606: i8, %607: f32):
      %608 = arith.sitofp %606 : i8 to f32
      linalg.yield %608 : f32
    } -> tensor<16x48xf32>
    %609 = arith.constant {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e-01 : f32
    %610 = tensor.splat %609 {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x48xf32>
    %611 = tensor.empty() : tensor<16x48xf32>
    %612 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%605, %610 : tensor<16x48xf32>, tensor<16x48xf32>) outs(%611 : tensor<16x48xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb73(%613: f32, %614: f32, %615: f32):
      %616 = arith.mulf %613, %614 : f32
      linalg.yield %616 : f32
    } -> tensor<16x48xf32>
    %617 = tensor.collapse_shape %612 [[0 : i64, 1 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16x48xf32> into tensor<768xf32>
    %618 = tensor.expand_shape %617 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 4, 48] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<768xf32> into tensor<1x4x4x48xf32>
    %619 = tensor.empty() : tensor<1x48x4x4xf32>
    %620 = linalg.transpose ins(%618:tensor<1x4x4x48xf32>) outs(%619:tensor<1x48x4x4xf32>) permutation = [0, 3, 1, 2]
    %621 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} 0.000000e+00 : f32
    %622 = tensor.splat %621 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<1x48x6x6xf32>
    %623 = "tensor.insert_slice"(%620, %622) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 48, 4, 4>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : (tensor<1x48x4x4xf32>, tensor<1x48x6x6xf32>) -> tensor<1x48x6x6xf32>
    %624 = tensor.empty() : tensor<48x1x3x3x1x4x4xf32>
    %625 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, (d0 + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%623 : tensor<1x48x6x6xf32>) outs(%624 : tensor<48x1x3x3x1x4x4xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} {
    ^bb74(%626: f32, %627: f32):
      linalg.yield %626 : f32
    } -> tensor<48x1x3x3x1x4x4xf32>
    %628 = tensor.collapse_shape %625 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<48x1x3x3x1x4x4xf32> into tensor<6912xf32>
    %629 = tensor.expand_shape %628 [[0 : i64, 1 : i64, 2 : i64]] output_shape [48, 9, 16] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<6912xf32> into tensor<48x9x16xf32>
    %630 = tensor.collapse_shape %4 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<48x1x3x3xf32> into tensor<432xf32>
    %631 = tensor.expand_shape %630 [[0 : i64, 1 : i64, 2 : i64]] output_shape [48, 1, 9] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<432xf32> into tensor<48x1x9xf32>
    %632 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} 0.000000e+00 : f32
    %633 = tensor.splat %632 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<48x1x16xf32>
    %634 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%631, %629 : tensor<48x1x9xf32>, tensor<48x9x16xf32>) outs(%633 : tensor<48x1x16xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} {
    ^bb75(%635: f32, %636: f32, %637: f32):
      %638 = arith.mulf %635, %636 : f32
      %639 = arith.addf %637, %638 : f32
      linalg.yield %639 : f32
    } -> tensor<48x1x16xf32>
    %640 = tensor.collapse_shape %634 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<48x1x16xf32> into tensor<768xf32>
    %641 = tensor.expand_shape %640 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [48, 1, 4, 4] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<768xf32> into tensor<48x1x4x4xf32>
    %642 = tensor.collapse_shape %641 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<48x1x4x4xf32> into tensor<768xf32>
    %643 = tensor.expand_shape %642 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 48, 4, 4] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<768xf32> into tensor<1x48x4x4xf32>
    %644 = tensor.empty() : tensor<1x48x4x4xf32>
    %645 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%643, %5 : tensor<1x48x4x4xf32>, tensor<48xf32>) outs(%644 : tensor<1x48x4x4xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} {
    ^bb76(%646: f32, %647: f32, %648: f32):
      %649 = arith.addf %646, %647 : f32
      linalg.yield %649 : f32
    } -> tensor<1x48x4x4xf32>
    %650 = tensor.empty() : tensor<1x48x4x4xf32>
    %651 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%645 : tensor<1x48x4x4xf32>) outs(%650 : tensor<1x48x4x4xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32"} {
    ^bb77(%652: f32, %653: f32):
      %654 = arith.constant 5.000000e-01 : f32
      %655 = arith.constant 1.000000e+00 : f32
      %656 = arith.constant 0.707106769 : f32
      %657 = arith.mulf %652, %656 : f32
      %658 = math.erf %657 : f32
      %659 = arith.addf %655, %658 : f32
      %660 = arith.mulf %654, %652 : f32
      %661 = arith.mulf %660, %659 : f32
      linalg.yield %661 : f32
    } -> tensor<1x48x4x4xf32>
    %662 = tensor.empty() : tensor<1x4x4x48xf32>
    %663 = linalg.transpose ins(%651:tensor<1x48x4x4xf32>) outs(%662:tensor<1x4x4x48xf32>) permutation = [0, 2, 3, 1]
    %664 = tensor.collapse_shape %663 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x4x48xf32> into tensor<768xf32>
    %665 = tensor.expand_shape %664 [[0 : i64, 1 : i64]] output_shape [16, 48] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<768xf32> into tensor<16x48xf32>
    %666 = arith.constant {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 4.000000e+00 : f32
    %667 = tensor.splat %666 {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x48xf32>
    %668 = tensor.empty() : tensor<16x48xf32>
    %669 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%665, %667 : tensor<16x48xf32>, tensor<16x48xf32>) outs(%668 : tensor<16x48xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb78(%670: f32, %671: f32, %672: f32):
      %673 = arith.mulf %670, %671 : f32
      linalg.yield %673 : f32
    } -> tensor<16x48xf32>
    %674 = tensor.empty() : tensor<16x48xf32>
    %675 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%669 : tensor<16x48xf32>) outs(%674 : tensor<16x48xf32>) attrs =  {prov.region_id = "tanh_8", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb79(%676: f32, %677: f32):
      %678 = math.tanh %676 : f32
      linalg.yield %678 : f32
    } -> tensor<16x48xf32>
    %679 = arith.constant {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 2.000000e+00 : f32
    %680 = tensor.splat %679 {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x48xf32>
    %681 = tensor.empty() : tensor<16x48xf32>
    %682 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%675, %680 : tensor<16x48xf32>, tensor<16x48xf32>) outs(%681 : tensor<16x48xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb80(%683: f32, %684: f32, %685: f32):
      %686 = arith.mulf %683, %684 : f32
      linalg.yield %686 : f32
    } -> tensor<16x48xf32>
    %687 = tensor.empty() : tensor<16x48xi8>
    %688 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%682 : tensor<16x48xf32>) outs(%687 : tensor<16x48xi8>) attrs =  {prov.region_id = "dtype_cast_18", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb81(%689: f32, %690: i8):
      %691 = arith.fptosi %689 : f32 to i8
      linalg.yield %691 : i8
    } -> tensor<16x48xi8>
    %692 = tensor.empty() : tensor<16x32xi8>
    %693 = arith.constant 0 : i8
    %694 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%693 : i8) outs(%692 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %695 = linalg.matmul {prov.region_id = "matmul_7", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%688, %11 : tensor<16x48xi8>, tensor<48x32xi8>) outs(%694 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %696 = tensor.empty() : tensor<16x32xf32>
    %697 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%695 : tensor<16x32xi8>) outs(%696 : tensor<16x32xf32>) attrs =  {prov.region_id = "dtype_cast_19", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb82(%698: i8, %699: f32):
      %700 = arith.sitofp %698 : i8 to f32
      linalg.yield %700 : f32
    } -> tensor<16x32xf32>
    %701 = arith.constant {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e-01 : f32
    %702 = tensor.splat %701 {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %703 = tensor.empty() : tensor<16x32xf32>
    %704 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%697, %702 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%703 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb83(%705: f32, %706: f32, %707: f32):
      %708 = arith.mulf %705, %706 : f32
      linalg.yield %708 : f32
    } -> tensor<16x32xf32>
    %709 = tensor.empty() : tensor<16x32xf32>
    %710 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%569, %704 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%709 : tensor<16x32xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb84(%711: f32, %712: f32, %713: f32):
      %714 = arith.addf %711, %712 : f32
      linalg.yield %714 : f32
    } -> tensor<16x32xf32>
    %715 = tensor.collapse_shape %710 [[0 : i64, 1 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16x32xf32> into tensor<512xf32>
    %716 = tensor.expand_shape %715 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 4, 32] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x4x4x32xf32>
    %717 = tensor.empty() : tensor<1x32x4x4xf32>
    %718 = linalg.transpose ins(%716:tensor<1x4x4x32xf32>) outs(%717:tensor<1x32x4x4xf32>) permutation = [0, 3, 1, 2]
    %719 = tensor.collapse_shape %718 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x4x4xf32> into tensor<512xf32>
    %720 = tensor.expand_shape %719 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] output_shape [1, 8, 2, 2, 4, 4] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x8x2x2x4x4xf32>
    %721 = tensor.empty() : tensor<1x8x4x2x4x2xf32>
    %722 = linalg.transpose ins(%720:tensor<1x8x2x2x4x4xf32>) outs(%721:tensor<1x8x4x2x4x2xf32>) permutation = [0, 1, 4, 2, 5, 3]
    %723 = tensor.collapse_shape %722 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x4x2x4x2xf32> into tensor<512xf32>
    %724 = tensor.expand_shape %723 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 8, 8] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x8x8x8xf32>
    %725 = tensor.collapse_shape %724 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x8x8xf32> into tensor<512xf32>
    %726 = tensor.expand_shape %725 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] output_shape [1, 8, 2, 4, 2, 4] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x8x2x4x2x4xf32>
    %727 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %728 = tensor.splat %727 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} : tensor<1x8x2x4x2xf32>
    %729 = linalg.reduce ins(%726:tensor<1x8x2x4x2x4xf32>) outs(%728:tensor<1x8x2x4x2xf32>) dimensions = [5]
    (%730: f32, %731: f32) {
      %732 = arith.maximumf %730, %731 : f32
      linalg.yield %732 : f32
    }
    %733 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %734 = tensor.splat %733 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} : tensor<1x8x2x2xf32>
    %735 = linalg.reduce ins(%729:tensor<1x8x2x4x2xf32>) outs(%734:tensor<1x8x2x2xf32>) dimensions = [3]
    (%736: f32, %737: f32) {
      %738 = arith.maximumf %736, %737 : f32
      linalg.yield %738 : f32
    }
    %739 = tensor.collapse_shape %735 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x2x2xf32> into tensor<32xf32>
    %740 = tensor.expand_shape %739 [[0 : i64, 1 : i64]] output_shape [1, 32] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32xf32>
    %741 = arith.constant {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 6.000000e-01 : f32
    %742 = tensor.splat %741 {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x32xf32>
    %743 = tensor.empty() : tensor<1x32xf32>
    %744 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%740, %742 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%743 : tensor<1x32xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb85(%745: f32, %746: f32, %747: f32):
      %748 = arith.mulf %745, %746 : f32
      linalg.yield %748 : f32
    } -> tensor<1x32xf32>
    %749 = tensor.empty() : tensor<1x32xf32>
    %750 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%744 : tensor<1x32xf32>) outs(%749 : tensor<1x32xf32>) attrs =  {prov.region_id = "tanh_9", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb86(%751: f32, %752: f32):
      %753 = math.tanh %751 : f32
      linalg.yield %753 : f32
    } -> tensor<1x32xf32>
    %754 = arith.constant {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %755 = tensor.splat %754 {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x32xf32>
    %756 = tensor.empty() : tensor<1x32xf32>
    %757 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%750, %755 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%756 : tensor<1x32xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb87(%758: f32, %759: f32, %760: f32):
      %761 = arith.mulf %758, %759 : f32
      linalg.yield %761 : f32
    } -> tensor<1x32xf32>
    %762 = tensor.empty() : tensor<1x32xi8>
    %763 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%757 : tensor<1x32xf32>) outs(%762 : tensor<1x32xi8>) attrs =  {prov.region_id = "dtype_cast_20", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb88(%764: f32, %765: i8):
      %766 = arith.fptosi %764 : f32 to i8
      linalg.yield %766 : i8
    } -> tensor<1x32xi8>
    %767 = tensor.empty() : tensor<1x32xi8>
    %768 = arith.constant 0 : i8
    %769 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%768 : i8) outs(%767 : tensor<1x32xi8>) -> tensor<1x32xi8>
    %770 = linalg.matmul {prov.region_id = "matmul_8", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%763, %12 : tensor<1x32xi8>, tensor<32x32xi8>) outs(%769 : tensor<1x32xi8>) -> tensor<1x32xi8>
    %771 = tensor.empty() : tensor<1x32xf32>
    %772 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%770 : tensor<1x32xi8>) outs(%771 : tensor<1x32xf32>) attrs =  {prov.region_id = "dtype_cast_21", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb89(%773: i8, %774: f32):
      %775 = arith.sitofp %773 : i8 to f32
      linalg.yield %775 : f32
    } -> tensor<1x32xf32>
    %776 = arith.constant {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e-01 : f32
    %777 = tensor.splat %776 {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x32xf32>
    %778 = tensor.empty() : tensor<1x32xf32>
    %779 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%772, %777 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%778 : tensor<1x32xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb90(%780: f32, %781: f32, %782: f32):
      %783 = arith.mulf %780, %781 : f32
      linalg.yield %783 : f32
    } -> tensor<1x32xf32>
    %784 = tensor.concat dim(1) %779, %20 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x32xf32>, tensor<1x16xf32>) -> tensor<1x48xf32>
    %785 = arith.constant {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.200000e+00 : f32
    %786 = tensor.splat %785 {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x48xf32>
    %787 = tensor.empty() : tensor<1x48xf32>
    %788 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%784, %786 : tensor<1x48xf32>, tensor<1x48xf32>) outs(%787 : tensor<1x48xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb91(%789: f32, %790: f32, %791: f32):
      %792 = arith.mulf %789, %790 : f32
      linalg.yield %792 : f32
    } -> tensor<1x48xf32>
    %793 = tensor.empty() : tensor<1x48xf32>
    %794 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%788 : tensor<1x48xf32>) outs(%793 : tensor<1x48xf32>) attrs =  {prov.region_id = "tanh_10", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb92(%795: f32, %796: f32):
      %797 = math.tanh %795 : f32
      linalg.yield %797 : f32
    } -> tensor<1x48xf32>
    %798 = arith.constant {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 2.000000e+00 : f32
    %799 = tensor.splat %798 {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x48xf32>
    %800 = tensor.empty() : tensor<1x48xf32>
    %801 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%794, %799 : tensor<1x48xf32>, tensor<1x48xf32>) outs(%800 : tensor<1x48xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb93(%802: f32, %803: f32, %804: f32):
      %805 = arith.mulf %802, %803 : f32
      linalg.yield %805 : f32
    } -> tensor<1x48xf32>
    %806 = tensor.empty() : tensor<1x48xi8>
    %807 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%801 : tensor<1x48xf32>) outs(%806 : tensor<1x48xi8>) attrs =  {prov.region_id = "dtype_cast_22", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb94(%808: f32, %809: i8):
      %810 = arith.fptosi %808 : f32 to i8
      linalg.yield %810 : i8
    } -> tensor<1x48xi8>
    %811 = tensor.empty() : tensor<1x64xi8>
    %812 = arith.constant 0 : i8
    %813 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%812 : i8) outs(%811 : tensor<1x64xi8>) -> tensor<1x64xi8>
    %814 = linalg.matmul {prov.region_id = "matmul_9", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%807, %13 : tensor<1x48xi8>, tensor<48x64xi8>) outs(%813 : tensor<1x64xi8>) -> tensor<1x64xi8>
    %815 = tensor.empty() : tensor<1x64xf32>
    %816 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%814 : tensor<1x64xi8>) outs(%815 : tensor<1x64xf32>) attrs =  {prov.region_id = "dtype_cast_23", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb95(%817: i8, %818: f32):
      %819 = arith.sitofp %817 : i8 to f32
      linalg.yield %819 : f32
    } -> tensor<1x64xf32>
    %820 = tensor.empty() : tensor<1x64xi8>
    %821 = arith.constant 0 : i8
    %822 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%821 : i8) outs(%820 : tensor<1x64xi8>) -> tensor<1x64xi8>
    %823 = linalg.matmul {prov.region_id = "matmul_10", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%16, %14 : tensor<1x16xi8>, tensor<16x64xi8>) outs(%822 : tensor<1x64xi8>) -> tensor<1x64xi8>
    %824 = tensor.empty() : tensor<1x64xf32>
    %825 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%823 : tensor<1x64xi8>) outs(%824 : tensor<1x64xf32>) attrs =  {prov.region_id = "dtype_cast_24", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb96(%826: i8, %827: f32):
      %828 = arith.sitofp %826 : i8 to f32
      linalg.yield %828 : f32
    } -> tensor<1x64xf32>
    %829 = tensor.empty() : tensor<1x64xf32>
    %830 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%816, %825 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%829 : tensor<1x64xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb97(%831: f32, %832: f32, %833: f32):
      %834 = arith.addf %831, %832 : f32
      linalg.yield %834 : f32
    } -> tensor<1x64xf32>
    %835 = arith.constant {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.200000e-01 : f32
    %836 = tensor.splat %835 {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x64xf32>
    %837 = tensor.empty() : tensor<1x64xf32>
    %838 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%830, %836 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%837 : tensor<1x64xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb98(%839: f32, %840: f32, %841: f32):
      %842 = arith.mulf %839, %840 : f32
      linalg.yield %842 : f32
    } -> tensor<1x64xf32>
    %843 = tensor.empty() : tensor<1x64xf32>
    %844 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%838, %15 : tensor<1x64xf32>, tensor<64xf32>) outs(%843 : tensor<1x64xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb99(%845: f32, %846: f32, %847: f32):
      %848 = arith.addf %845, %846 : f32
      linalg.yield %848 : f32
    } -> tensor<1x64xf32>
    %849 = "tensor.extract_slice"(%844) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 16>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x64xf32>) -> tensor<1x16xf32>
    %850 = "tensor.extract_slice"(%844) <{static_offsets = array<i64: 0, 16>, static_sizes = array<i64: 1, 16>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x64xf32>) -> tensor<1x16xf32>
    %851 = "tensor.extract_slice"(%844) <{static_offsets = array<i64: 0, 32>, static_sizes = array<i64: 1, 16>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x64xf32>) -> tensor<1x16xf32>
    %852 = "tensor.extract_slice"(%844) <{static_offsets = array<i64: 0, 48>, static_sizes = array<i64: 1, 16>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x64xf32>) -> tensor<1x16xf32>
    %853 = tensor.empty() : tensor<1x16xf32>
    %854 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%850 : tensor<1x16xf32>) outs(%853 : tensor<1x16xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32"} {
    ^bb100(%855: f32, %856: f32):
      %857 = arith.constant 1.000000e+00 : f32
      %858 = arith.negf %855 : f32
      %859 = math.exp %858 : f32
      %860 = arith.addf %857, %859 : f32
      %861 = arith.divf %857, %860 : f32
      linalg.yield %861 : f32
    } -> tensor<1x16xf32>
    %862 = tensor.empty() : tensor<1x16xf32>
    %863 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%854, %17 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%862 : tensor<1x16xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb101(%864: f32, %865: f32, %866: f32):
      %867 = arith.mulf %864, %865 : f32
      linalg.yield %867 : f32
    } -> tensor<1x16xf32>
    %868 = tensor.empty() : tensor<1x16xf32>
    %869 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%849 : tensor<1x16xf32>) outs(%868 : tensor<1x16xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32"} {
    ^bb102(%870: f32, %871: f32):
      %872 = arith.constant 1.000000e+00 : f32
      %873 = arith.negf %870 : f32
      %874 = math.exp %873 : f32
      %875 = arith.addf %872, %874 : f32
      %876 = arith.divf %872, %875 : f32
      linalg.yield %876 : f32
    } -> tensor<1x16xf32>
    %877 = tensor.empty() : tensor<1x16xf32>
    %878 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%851 : tensor<1x16xf32>) outs(%877 : tensor<1x16xf32>) attrs =  {prov.region_id = "tanh_11", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb103(%879: f32, %880: f32):
      %881 = math.tanh %879 : f32
      linalg.yield %881 : f32
    } -> tensor<1x16xf32>
    %882 = tensor.empty() : tensor<1x16xf32>
    %883 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%869, %878 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%882 : tensor<1x16xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb104(%884: f32, %885: f32, %886: f32):
      %887 = arith.mulf %884, %885 : f32
      linalg.yield %887 : f32
    } -> tensor<1x16xf32>
    %888 = tensor.empty() : tensor<1x16xf32>
    %889 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%863, %883 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%888 : tensor<1x16xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb105(%890: f32, %891: f32, %892: f32):
      %893 = arith.addf %890, %891 : f32
      linalg.yield %893 : f32
    } -> tensor<1x16xf32>
    %894 = tensor.empty() : tensor<1x16xf32>
    %895 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%852 : tensor<1x16xf32>) outs(%894 : tensor<1x16xf32>) attrs =  {prov.region_id = "sigmoid_2", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32"} {
    ^bb106(%896: f32, %897: f32):
      %898 = arith.constant 1.000000e+00 : f32
      %899 = arith.negf %896 : f32
      %900 = math.exp %899 : f32
      %901 = arith.addf %898, %900 : f32
      %902 = arith.divf %898, %901 : f32
      linalg.yield %902 : f32
    } -> tensor<1x16xf32>
    %903 = tensor.empty() : tensor<1x16xf32>
    %904 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%889 : tensor<1x16xf32>) outs(%903 : tensor<1x16xf32>) attrs =  {prov.region_id = "tanh_12", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb107(%905: f32, %906: f32):
      %907 = math.tanh %905 : f32
      linalg.yield %907 : f32
    } -> tensor<1x16xf32>
    %908 = tensor.empty() : tensor<1x16xf32>
    %909 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%895, %904 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%908 : tensor<1x16xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb108(%910: f32, %911: f32, %912: f32):
      %913 = arith.mulf %910, %911 : f32
      linalg.yield %913 : f32
    } -> tensor<1x16xf32>
    %914 = arith.constant {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 7.000000e+00 : f32
    %915 = tensor.splat %914 {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x16xf32>
    %916 = tensor.empty() : tensor<1x16xf32>
    %917 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%909, %915 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%916 : tensor<1x16xf32>) attrs =  {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb109(%918: f32, %919: f32, %920: f32):
      %921 = arith.mulf %918, %919 : f32
      linalg.yield %921 : f32
    } -> tensor<1x16xf32>
    %922 = tensor.empty() : tensor<1x16xi8>
    %923 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%917 : tensor<1x16xf32>) outs(%922 : tensor<1x16xi8>) attrs =  {prov.region_id = "dtype_cast_25", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb110(%924: f32, %925: i8):
      %926 = arith.fptosi %924 : f32 to i8
      linalg.yield %926 : i8
    } -> tensor<1x16xi8>
    %927 = tensor.empty() : tensor<1x16xi8>
    %928 = arith.constant 0 : i8
    %929 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%928 : i8) outs(%927 : tensor<1x16xi8>) -> tensor<1x16xi8>
    %930 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%923, %18 : tensor<1x16xi8>, tensor<16x16xi8>) outs(%929 : tensor<1x16xi8>) -> tensor<1x16xi8>
    %931 = tensor.empty() : tensor<1x16xf32>
    %932 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%930 : tensor<1x16xi8>) outs(%931 : tensor<1x16xf32>) attrs =  {prov.region_id = "dtype_cast_26", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb111(%933: i8, %934: f32):
      %935 = arith.sitofp %933 : i8 to f32
      linalg.yield %935 : f32
    } -> tensor<1x16xf32>
    %936 = arith.constant {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.200000e-01 : f32
    %937 = tensor.splat %936 {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x16xf32>
    %938 = tensor.empty() : tensor<1x16xf32>
    %939 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%932, %937 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%938 : tensor<1x16xf32>) attrs =  {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb112(%940: f32, %941: f32, %942: f32):
      %943 = arith.mulf %940, %941 : f32
      linalg.yield %943 : f32
    } -> tensor<1x16xf32>
    func.return %939 : tensor<1x16xf32>
  }
}
