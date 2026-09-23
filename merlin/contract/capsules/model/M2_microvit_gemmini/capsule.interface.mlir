builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "int8_static_act_int8_weight"} {
  func.func @forward(%0: tensor<32xf32>, %1: tensor<32xf32>, %2: tensor<32xf32>, %3: tensor<32xf32>, %4: tensor<48xf32>, %5: tensor<36x32xi8>, %6: tensor<32x32xi8>, %7: tensor<32x64xi8>, %8: tensor<32x32xi8>, %9: tensor<32x48xi8>, %10: tensor<48x32xi8>, %11: tensor<32x32xi8>, %12: tensor<48x64xi8>, %13: tensor<16x64xi8>, %14: tensor<64xf32>, %15: tensor<1x16xi8>, %16: tensor<1x16xf32>, %17: tensor<16x16xi8>, %18: tensor<48x1x3x3xi8>, %19: tensor<1x1x16x16xf32>, %20: tensor<1x16xf32>) -> tensor<1x16xf32> {
    %21 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.0129223242 : f32
    %22 = tensor.splat %21 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %23 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %24 = tensor.splat %23 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %25 = "quant_ext.dequantize_per_tensor"(%18, %22, %24) <{quant_min = -127 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_0", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<48x1x3x3xi8>, tensor<f32>, tensor<i64>) -> tensor<48x1x3x3xf32>
    %26 = tensor.empty() : tensor<4xi64>
    %27 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%26 : tensor<4xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb0(%28: i64):
      %29 = linalg.index 0 : index
      %30 = arith.index_cast %29 : index to i64
      %31 = arith.constant 4 : i64
      %32 = arith.muli %30, %31 : i64
      %33 = arith.constant 0 : i64
      %34 = arith.addi %33, %32 : i64
      linalg.yield %34 : i64
    } -> tensor<4xi64>
    %35 = tensor.expand_shape %27 [[0 : i64, 1 : i64]] output_shape [1, 4] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<4xi64> into tensor<1x4xi64>
    %36 = tensor.empty() : tensor<6xi64>
    %37 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%36 : tensor<6xi64>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb1(%38: i64):
      %39 = linalg.index 0 : index
      %40 = arith.index_cast %39 : index to i64
      %41 = arith.constant 1 : i64
      %42 = arith.muli %40, %41 : i64
      %43 = arith.constant 0 : i64
      %44 = arith.addi %43, %42 : i64
      linalg.yield %44 : i64
    } -> tensor<6xi64>
    %45 = tensor.expand_shape %37 [[0 : i64, 1 : i64]] output_shape [6, 1] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<6xi64> into tensor<6x1xi64>
    %46 = tensor.empty() : tensor<6x4xi64>
    %47 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%35, %45 : tensor<1x4xi64>, tensor<6x1xi64>) outs(%46 : tensor<6x4xi64>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb2(%48: i64, %49: i64, %50: i64):
      %51 = arith.addi %48, %49 : i64
      linalg.yield %51 : i64
    } -> tensor<6x4xi64>
    %52 = tensor.empty() : tensor<4xi64>
    %53 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%52 : tensor<4xi64>) attrs =  {prov.region_id = "iota_2", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb3(%54: i64):
      %55 = linalg.index 0 : index
      %56 = arith.index_cast %55 : index to i64
      %57 = arith.constant 4 : i64
      %58 = arith.muli %56, %57 : i64
      %59 = arith.constant 0 : i64
      %60 = arith.addi %59, %58 : i64
      linalg.yield %60 : i64
    } -> tensor<4xi64>
    %61 = tensor.expand_shape %53 [[0 : i64, 1 : i64]] output_shape [1, 4] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<4xi64> into tensor<1x4xi64>
    %62 = tensor.empty() : tensor<6xi64>
    %63 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%62 : tensor<6xi64>) attrs =  {prov.region_id = "iota_3", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb4(%64: i64):
      %65 = linalg.index 0 : index
      %66 = arith.index_cast %65 : index to i64
      %67 = arith.constant 1 : i64
      %68 = arith.muli %66, %67 : i64
      %69 = arith.constant 0 : i64
      %70 = arith.addi %69, %68 : i64
      linalg.yield %70 : i64
    } -> tensor<6xi64>
    %71 = tensor.expand_shape %63 [[0 : i64, 1 : i64]] output_shape [6, 1] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<6xi64> into tensor<6x1xi64>
    %72 = tensor.empty() : tensor<6x4xi64>
    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%61, %71 : tensor<1x4xi64>, tensor<6x1xi64>) outs(%72 : tensor<6x4xi64>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb5(%74: i64, %75: i64, %76: i64):
      %77 = arith.addi %74, %75 : i64
      linalg.yield %77 : i64
    } -> tensor<6x4xi64>
    %78 = arith.constant {prov.region_id = "pad_0", prov.family = "layout", prov._pattern_hint = "pad", prov.op = "pad", prov.aten = "aten.constant_pad_nd.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %79 = tensor.splat %78 {prov.region_id = "pad_0", prov.family = "layout", prov._pattern_hint = "pad", prov.op = "pad", prov.aten = "aten.constant_pad_nd.default", prov.orig_dtype = "float32"} : tensor<1x1x18x18xf32>
    %80 = "tensor.insert_slice"(%19, %79) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 1, 16, 16>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "pad_0", prov.family = "layout", prov._pattern_hint = "pad", prov.op = "pad", prov.aten = "aten.constant_pad_nd.default", prov.orig_dtype = "float32"} : (tensor<1x1x16x16xf32>, tensor<1x1x18x18xf32>) -> tensor<1x1x18x18xf32>
    %81 = tensor.collapse_shape %47 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<6x4xi64> into tensor<24xi64>
    %82 = tensor.expand_shape %81 [[0 : i64, 1 : i64, 2 : i64]] output_shape [6, 4, 1] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<24xi64> into tensor<6x4x1xi64>
    %83 = tensor.collapse_shape %82 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<6x4x1xi64> into tensor<24xi64>
    %84 = tensor.expand_shape %83 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [6, 4, 1, 1] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<24xi64> into tensor<6x4x1x1xi64>
    %85 = tensor.empty() : tensor<1x1x6x4x6x4xf32>
    %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3, 0, 0)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%84, %73 : tensor<6x4x1x1xi64>, tensor<6x4xi64>) outs(%85 : tensor<1x1x6x4x6x4xf32>) attrs =  {prov.region_id = "gather_0", prov.family = "gather_scatter", prov._pattern_hint = "index_gather", prov.op = "index_gather", prov.aten = "aten.index.Tensor", prov.orig_dtype = "float32"} {
    ^bb6(%87: i64, %88: i64, %89: f32):
      %90 = linalg.index 0 : index
      %91 = linalg.index 1 : index
      %92 = arith.index_cast %87 : i64 to index
      %93 = arith.index_cast %88 : i64 to index
      %94 = tensor.extract %80[%90, %91, %92, %93] : tensor<1x1x18x18xf32>
      linalg.yield %94 : f32
    } -> tensor<1x1x6x4x6x4xf32>
    %95 = tensor.empty() : tensor<1x1x6x6x4x4xf32>
    %96 = linalg.transpose ins(%86:tensor<1x1x6x4x6x4xf32>) outs(%95:tensor<1x1x6x6x4x4xf32>) permutation = [0, 1, 2, 4, 3, 5]
    %97 = tensor.collapse_shape %96 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x6x6x4x4xf32> into tensor<576xf32>
    %98 = tensor.expand_shape %97 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 36, 16] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<576xf32> into tensor<1x36x16xf32>
    %99 = tensor.empty() : tensor<1x16x36xf32>
    %100 = linalg.transpose ins(%98:tensor<1x36x16xf32>) outs(%99:tensor<1x16x36xf32>) permutation = [0, 2, 1]
    %101 = tensor.collapse_shape %100 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x16x36xf32> into tensor<576xf32>
    %102 = tensor.expand_shape %101 [[0 : i64, 1 : i64]] output_shape [16, 36] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<576xf32> into tensor<16x36xf32>
    %103 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 8.000000e-01 : f32
    %104 = tensor.splat %103 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x36xf32>
    %105 = tensor.empty() : tensor<16x36xf32>
    %106 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%102, %104 : tensor<16x36xf32>, tensor<16x36xf32>) outs(%105 : tensor<16x36xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb7(%107: f32, %108: f32, %109: f32):
      %110 = arith.mulf %107, %108 : f32
      linalg.yield %110 : f32
    } -> tensor<16x36xf32>
    %111 = tensor.empty() : tensor<16x36xf32>
    %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106 : tensor<16x36xf32>) outs(%111 : tensor<16x36xf32>) attrs =  {prov.region_id = "tanh_0", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb8(%113: f32, %114: f32):
      %115 = math.tanh %113 : f32
      linalg.yield %115 : f32
    } -> tensor<16x36xf32>
    %116 = arith.constant {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %117 = tensor.splat %116 {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x36xf32>
    %118 = tensor.empty() : tensor<16x36xf32>
    %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%112, %117 : tensor<16x36xf32>, tensor<16x36xf32>) outs(%118 : tensor<16x36xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb9(%120: f32, %121: f32, %122: f32):
      %123 = arith.mulf %120, %121 : f32
      linalg.yield %123 : f32
    } -> tensor<16x36xf32>
    %124 = tensor.empty() : tensor<16x36xi8>
    %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%119 : tensor<16x36xf32>) outs(%124 : tensor<16x36xi8>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb10(%126: f32, %127: i8):
      %128 = arith.fptosi %126 : f32 to i8
      linalg.yield %128 : i8
    } -> tensor<16x36xi8>
    %129 = tensor.empty() : tensor<16x32xi8>
    %130 = arith.constant 0 : i8
    %131 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%130 : i8) outs(%129 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %132 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%125, %5 : tensor<16x36xi8>, tensor<36x32xi8>) outs(%131 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %133 = tensor.empty() : tensor<16x32xf32>
    %134 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%132 : tensor<16x32xi8>) outs(%133 : tensor<16x32xf32>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb11(%135: i8, %136: f32):
      %137 = arith.sitofp %135 : i8 to f32
      linalg.yield %137 : f32
    } -> tensor<16x32xf32>
    %138 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} 0.000000e+00 : f32
    %139 = tensor.splat %138 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} : tensor<16xf32>
    %140 = linalg.reduce ins(%134:tensor<16x32xf32>) outs(%139:tensor<16xf32>) dimensions = [1]
    (%141: f32, %142: f32) {
      %143 = arith.addf %141, %142 : f32
      linalg.yield %143 : f32
    }
    %144 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} 3.200000e+01 : f32
    %145 = tensor.splat %144 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} : tensor<16xf32>
    %146 = tensor.empty() : tensor<16xf32>
    %147 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%140, %145 : tensor<16xf32>, tensor<16xf32>) outs(%146 : tensor<16xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb12(%148: f32, %149: f32, %150: f32):
      %151 = arith.divf %148, %149 : f32
      linalg.yield %151 : f32
    } -> tensor<16xf32>
    %152 = tensor.expand_shape %147 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} : tensor<16xf32> into tensor<16x1xf32>
    %153 = tensor.empty() : tensor<16x32xf32>
    %154 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%134, %152 : tensor<16x32xf32>, tensor<16x1xf32>) outs(%153 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb13(%155: f32, %156: f32, %157: f32):
      %158 = arith.subf %155, %156 : f32
      linalg.yield %158 : f32
    } -> tensor<16x32xf32>
    %159 = tensor.empty() : tensor<16x32xf32>
    %160 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%154, %154 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%159 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb14(%161: f32, %162: f32, %163: f32):
      %164 = arith.mulf %161, %162 : f32
      linalg.yield %164 : f32
    } -> tensor<16x32xf32>
    %165 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} 0.000000e+00 : f32
    %166 = tensor.splat %165 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} : tensor<16xf32>
    %167 = linalg.reduce ins(%160:tensor<16x32xf32>) outs(%166:tensor<16xf32>) dimensions = [1]
    (%168: f32, %169: f32) {
      %170 = arith.addf %168, %169 : f32
      linalg.yield %170 : f32
    }
    %171 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} 3.200000e+01 : f32
    %172 = tensor.splat %171 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} : tensor<16xf32>
    %173 = tensor.empty() : tensor<16xf32>
    %174 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%167, %172 : tensor<16xf32>, tensor<16xf32>) outs(%173 : tensor<16xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb15(%175: f32, %176: f32, %177: f32):
      %178 = arith.divf %175, %176 : f32
      linalg.yield %178 : f32
    } -> tensor<16xf32>
    %179 = tensor.expand_shape %174 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} : tensor<16xf32> into tensor<16x1xf32>
    %180 = arith.constant {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} 1.000000e-05 : f32
    %181 = tensor.splat %180 {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} : tensor<16x1xf32>
    %182 = tensor.empty() : tensor<16x1xf32>
    %183 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%179, %181 : tensor<16x1xf32>, tensor<16x1xf32>) outs(%182 : tensor<16x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb16(%184: f32, %185: f32, %186: f32):
      %187 = arith.addf %184, %185 : f32
      linalg.yield %187 : f32
    } -> tensor<16x1xf32>
    %188 = tensor.empty() : tensor<16x1xf32>
    %189 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%183 : tensor<16x1xf32>) outs(%188 : tensor<16x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb17(%190: f32, %191: f32):
      %192 = math.rsqrt %190 : f32
      linalg.yield %192 : f32
    } -> tensor<16x1xf32>
    %193 = tensor.empty() : tensor<16x32xf32>
    %194 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%154, %189 : tensor<16x32xf32>, tensor<16x1xf32>) outs(%193 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb18(%195: f32, %196: f32, %197: f32):
      %198 = arith.mulf %195, %196 : f32
      linalg.yield %198 : f32
    } -> tensor<16x32xf32>
    %199 = tensor.empty() : tensor<16x32xf32>
    %200 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%194, %0 : tensor<16x32xf32>, tensor<32xf32>) outs(%199 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb19(%201: f32, %202: f32, %203: f32):
      %204 = arith.mulf %201, %202 : f32
      linalg.yield %204 : f32
    } -> tensor<16x32xf32>
    %205 = tensor.empty() : tensor<16x32xf32>
    %206 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%200, %1 : tensor<16x32xf32>, tensor<32xf32>) outs(%205 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln1", prov.fqn = "ln1"} {
    ^bb20(%207: f32, %208: f32, %209: f32):
      %210 = arith.addf %207, %208 : f32
      linalg.yield %210 : f32
    } -> tensor<16x32xf32>
    %211 = arith.constant {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.100000e+00 : f32
    %212 = tensor.splat %211 {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %213 = tensor.empty() : tensor<16x32xf32>
    %214 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%206, %212 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%213 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb21(%215: f32, %216: f32, %217: f32):
      %218 = arith.mulf %215, %216 : f32
      linalg.yield %218 : f32
    } -> tensor<16x32xf32>
    %219 = tensor.empty() : tensor<16x32xf32>
    %220 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%214 : tensor<16x32xf32>) outs(%219 : tensor<16x32xf32>) attrs =  {prov.region_id = "tanh_1", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb22(%221: f32, %222: f32):
      %223 = math.tanh %221 : f32
      linalg.yield %223 : f32
    } -> tensor<16x32xf32>
    %224 = arith.constant {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %225 = tensor.splat %224 {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %226 = tensor.empty() : tensor<16x32xf32>
    %227 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%220, %225 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%226 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb23(%228: f32, %229: f32, %230: f32):
      %231 = arith.mulf %228, %229 : f32
      linalg.yield %231 : f32
    } -> tensor<16x32xf32>
    %232 = tensor.empty() : tensor<16x32xi8>
    %233 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%227 : tensor<16x32xf32>) outs(%232 : tensor<16x32xi8>) attrs =  {prov.region_id = "dtype_cast_2", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb24(%234: f32, %235: i8):
      %236 = arith.fptosi %234 : f32 to i8
      linalg.yield %236 : i8
    } -> tensor<16x32xi8>
    %237 = tensor.collapse_shape %233 [[0 : i64, 1 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int8"} : tensor<16x32xi8> into tensor<512xi8>
    %238 = tensor.expand_shape %237 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 4, 32] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int8"} : tensor<512xi8> into tensor<1x4x4x32xi8>
    %239 = tensor.empty() : tensor<1x32x4x4xi8>
    %240 = linalg.transpose ins(%238:tensor<1x4x4x32xi8>) outs(%239:tensor<1x32x4x4xi8>) permutation = [0, 3, 1, 2]
    %241 = tensor.collapse_shape %240 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int8"} : tensor<1x32x4x4xi8> into tensor<512xi8>
    %242 = tensor.expand_shape %241 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] output_shape [1, 32, 2, 2, 2, 2] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int8"} : tensor<512xi8> into tensor<1x32x2x2x2x2xi8>
    %243 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "int8"} -128 : i8
    %244 = tensor.splat %243 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "int8"} : tensor<1x32x2x2x2xi8>
    %245 = linalg.reduce ins(%242:tensor<1x32x2x2x2x2xi8>) outs(%244:tensor<1x32x2x2x2xi8>) dimensions = [5]
    (%246: i8, %247: i8) {
      %248 = arith.maxsi %246, %247 : i8
      linalg.yield %248 : i8
    }
    %249 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "int8"} -128 : i8
    %250 = tensor.splat %249 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "int8"} : tensor<1x32x2x2xi8>
    %251 = linalg.reduce ins(%245:tensor<1x32x2x2x2xi8>) outs(%250:tensor<1x32x2x2xi8>) dimensions = [3]
    (%252: i8, %253: i8) {
      %254 = arith.maxsi %252, %253 : i8
      linalg.yield %254 : i8
    }
    %255 = tensor.empty() : tensor<1x2x2x32xi8>
    %256 = linalg.transpose ins(%251:tensor<1x32x2x2xi8>) outs(%255:tensor<1x2x2x32xi8>) permutation = [0, 2, 3, 1]
    %257 = tensor.collapse_shape %256 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int8"} : tensor<1x2x2x32xi8> into tensor<128xi8>
    %258 = tensor.expand_shape %257 [[0 : i64, 1 : i64]] output_shape [4, 32] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "int8"} : tensor<128xi8> into tensor<4x32xi8>
    %259 = tensor.empty() : tensor<4x32xf32>
    %260 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%258 : tensor<4x32xi8>) outs(%259 : tensor<4x32xf32>) attrs =  {prov.region_id = "dtype_cast_3", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb25(%261: i8, %262: f32):
      %263 = arith.sitofp %261 : i8 to f32
      linalg.yield %263 : f32
    } -> tensor<4x32xf32>
    %264 = arith.constant {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 9.000000e-01 : f32
    %265 = tensor.splat %264 {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<4x32xf32>
    %266 = tensor.empty() : tensor<4x32xf32>
    %267 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%260, %265 : tensor<4x32xf32>, tensor<4x32xf32>) outs(%266 : tensor<4x32xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb26(%268: f32, %269: f32, %270: f32):
      %271 = arith.mulf %268, %269 : f32
      linalg.yield %271 : f32
    } -> tensor<4x32xf32>
    %272 = tensor.empty() : tensor<4x32xf32>
    %273 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%267 : tensor<4x32xf32>) outs(%272 : tensor<4x32xf32>) attrs =  {prov.region_id = "tanh_2", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb27(%274: f32, %275: f32):
      %276 = math.tanh %274 : f32
      linalg.yield %276 : f32
    } -> tensor<4x32xf32>
    %277 = arith.constant {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %278 = tensor.splat %277 {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<4x32xf32>
    %279 = tensor.empty() : tensor<4x32xf32>
    %280 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%273, %278 : tensor<4x32xf32>, tensor<4x32xf32>) outs(%279 : tensor<4x32xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb28(%281: f32, %282: f32, %283: f32):
      %284 = arith.mulf %281, %282 : f32
      linalg.yield %284 : f32
    } -> tensor<4x32xf32>
    %285 = tensor.empty() : tensor<4x32xi8>
    %286 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%280 : tensor<4x32xf32>) outs(%285 : tensor<4x32xi8>) attrs =  {prov.region_id = "dtype_cast_4", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb29(%287: f32, %288: i8):
      %289 = arith.fptosi %287 : f32 to i8
      linalg.yield %289 : i8
    } -> tensor<4x32xi8>
    %290 = tensor.empty() : tensor<16x32xi8>
    %291 = arith.constant 0 : i8
    %292 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%291 : i8) outs(%290 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %293 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%233, %6 : tensor<16x32xi8>, tensor<32x32xi8>) outs(%292 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %294 = tensor.empty() : tensor<16x32xf32>
    %295 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%293 : tensor<16x32xi8>) outs(%294 : tensor<16x32xf32>) attrs =  {prov.region_id = "dtype_cast_5", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb30(%296: i8, %297: f32):
      %298 = arith.sitofp %296 : i8 to f32
      linalg.yield %298 : f32
    } -> tensor<16x32xf32>
    %299 = arith.constant {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.500000e-01 : f32
    %300 = tensor.splat %299 {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %301 = tensor.empty() : tensor<16x32xf32>
    %302 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%295, %300 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%301 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb31(%303: f32, %304: f32, %305: f32):
      %306 = arith.mulf %303, %304 : f32
      linalg.yield %306 : f32
    } -> tensor<16x32xf32>
    %307 = tensor.empty() : tensor<16x32xf32>
    %308 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%302 : tensor<16x32xf32>) outs(%307 : tensor<16x32xf32>) attrs =  {prov.region_id = "tanh_3", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb32(%309: f32, %310: f32):
      %311 = math.tanh %309 : f32
      linalg.yield %311 : f32
    } -> tensor<16x32xf32>
    %312 = arith.constant {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %313 = tensor.splat %312 {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %314 = tensor.empty() : tensor<16x32xf32>
    %315 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%308, %313 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%314 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb33(%316: f32, %317: f32, %318: f32):
      %319 = arith.mulf %316, %317 : f32
      linalg.yield %319 : f32
    } -> tensor<16x32xf32>
    %320 = tensor.empty() : tensor<16x32xi8>
    %321 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%315 : tensor<16x32xf32>) outs(%320 : tensor<16x32xi8>) attrs =  {prov.region_id = "dtype_cast_6", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb34(%322: f32, %323: i8):
      %324 = arith.fptosi %322 : f32 to i8
      linalg.yield %324 : i8
    } -> tensor<16x32xi8>
    %325 = tensor.empty() : tensor<4x64xi8>
    %326 = arith.constant 0 : i8
    %327 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%326 : i8) outs(%325 : tensor<4x64xi8>) -> tensor<4x64xi8>
    %328 = linalg.matmul {prov.region_id = "matmul_2", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%286, %7 : tensor<4x32xi8>, tensor<32x64xi8>) outs(%327 : tensor<4x64xi8>) -> tensor<4x64xi8>
    %329 = tensor.empty() : tensor<4x64xf32>
    %330 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%328 : tensor<4x64xi8>) outs(%329 : tensor<4x64xf32>) attrs =  {prov.region_id = "dtype_cast_7", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb35(%331: i8, %332: f32):
      %333 = arith.sitofp %331 : i8 to f32
      linalg.yield %333 : f32
    } -> tensor<4x64xf32>
    %334 = arith.constant {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.500000e-01 : f32
    %335 = tensor.splat %334 {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<4x64xf32>
    %336 = tensor.empty() : tensor<4x64xf32>
    %337 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%330, %335 : tensor<4x64xf32>, tensor<4x64xf32>) outs(%336 : tensor<4x64xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb36(%338: f32, %339: f32, %340: f32):
      %341 = arith.mulf %338, %339 : f32
      linalg.yield %341 : f32
    } -> tensor<4x64xf32>
    %342 = tensor.empty() : tensor<4x64xf32>
    %343 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%337 : tensor<4x64xf32>) outs(%342 : tensor<4x64xf32>) attrs =  {prov.region_id = "tanh_4", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb37(%344: f32, %345: f32):
      %346 = math.tanh %344 : f32
      linalg.yield %346 : f32
    } -> tensor<4x64xf32>
    %347 = arith.constant {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %348 = tensor.splat %347 {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<4x64xf32>
    %349 = tensor.empty() : tensor<4x64xf32>
    %350 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%343, %348 : tensor<4x64xf32>, tensor<4x64xf32>) outs(%349 : tensor<4x64xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb38(%351: f32, %352: f32, %353: f32):
      %354 = arith.mulf %351, %352 : f32
      linalg.yield %354 : f32
    } -> tensor<4x64xf32>
    %355 = tensor.empty() : tensor<4x64xi8>
    %356 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%350 : tensor<4x64xf32>) outs(%355 : tensor<4x64xi8>) attrs =  {prov.region_id = "dtype_cast_8", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb39(%357: f32, %358: i8):
      %359 = arith.fptosi %357 : f32 to i8
      linalg.yield %359 : i8
    } -> tensor<4x64xi8>
    %360 = "tensor.extract_slice"(%356) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 4, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int8"} : (tensor<4x64xi8>) -> tensor<4x32xi8>
    %361 = "tensor.extract_slice"(%356) <{static_offsets = array<i64: 0, 32>, static_sizes = array<i64: 4, 32>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "int8"} : (tensor<4x64xi8>) -> tensor<4x32xi8>
    %362 = tensor.empty() : tensor<32x4xi8>
    %363 = linalg.transpose ins(%360:tensor<4x32xi8>) outs(%362:tensor<32x4xi8>) permutation = [1, 0]
    %364 = tensor.empty() : tensor<16x4xi8>
    %365 = arith.constant 0 : i8
    %366 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%365 : i8) outs(%364 : tensor<16x4xi8>) -> tensor<16x4xi8>
    %367 = linalg.matmul {prov.region_id = "matmul_3", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8", prov.transposed_b = "true"} ins(%321, %363 : tensor<16x32xi8>, tensor<32x4xi8>) outs(%366 : tensor<16x4xi8>) -> tensor<16x4xi8>
    %368 = tensor.empty() : tensor<16x4xf32>
    %369 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%367 : tensor<16x4xi8>) outs(%368 : tensor<16x4xf32>) attrs =  {prov.region_id = "dtype_cast_9", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb40(%370: i8, %371: f32):
      %372 = arith.sitofp %370 : i8 to f32
      linalg.yield %372 : f32
    } -> tensor<16x4xf32>
    %373 = arith.constant {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.176776692 : f32
    %374 = tensor.splat %373 {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x4xf32>
    %375 = tensor.empty() : tensor<16x4xf32>
    %376 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%369, %374 : tensor<16x4xf32>, tensor<16x4xf32>) outs(%375 : tensor<16x4xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb41(%377: f32, %378: f32, %379: f32):
      %380 = arith.mulf %377, %378 : f32
      linalg.yield %380 : f32
    } -> tensor<16x4xf32>
    %381 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %382 = tensor.splat %381 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<16xf32>
    %383 = linalg.reduce ins(%376:tensor<16x4xf32>) outs(%382:tensor<16xf32>) dimensions = [1]
    (%384: f32, %385: f32) {
      %386 = arith.maximumf %384, %385 : f32
      linalg.yield %386 : f32
    }
    %387 = tensor.expand_shape %383 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<16xf32> into tensor<16x1xf32>
    %388 = tensor.empty() : tensor<16x4xf32>
    %389 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%376, %387 : tensor<16x4xf32>, tensor<16x1xf32>) outs(%388 : tensor<16x4xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb42(%390: f32, %391: f32, %392: f32):
      %393 = arith.subf %390, %391 : f32
      linalg.yield %393 : f32
    } -> tensor<16x4xf32>
    %394 = tensor.empty() : tensor<16x4xf32>
    %395 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%389 : tensor<16x4xf32>) outs(%394 : tensor<16x4xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb43(%396: f32, %397: f32):
      %398 = math.exp %396 : f32
      linalg.yield %398 : f32
    } -> tensor<16x4xf32>
    %399 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %400 = tensor.splat %399 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<16xf32>
    %401 = linalg.reduce ins(%395:tensor<16x4xf32>) outs(%400:tensor<16xf32>) dimensions = [1]
    (%402: f32, %403: f32) {
      %404 = arith.addf %402, %403 : f32
      linalg.yield %404 : f32
    }
    %405 = tensor.expand_shape %401 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<16xf32> into tensor<16x1xf32>
    %406 = tensor.empty() : tensor<16x4xf32>
    %407 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%395, %405 : tensor<16x4xf32>, tensor<16x1xf32>) outs(%406 : tensor<16x4xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb44(%408: f32, %409: f32, %410: f32):
      %411 = arith.divf %408, %409 : f32
      linalg.yield %411 : f32
    } -> tensor<16x4xf32>
    %412 = arith.constant {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+01 : f32
    %413 = tensor.splat %412 {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x4xf32>
    %414 = tensor.empty() : tensor<16x4xf32>
    %415 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%407, %413 : tensor<16x4xf32>, tensor<16x4xf32>) outs(%414 : tensor<16x4xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb45(%416: f32, %417: f32, %418: f32):
      %419 = arith.mulf %416, %417 : f32
      linalg.yield %419 : f32
    } -> tensor<16x4xf32>
    %420 = tensor.empty() : tensor<16x4xi8>
    %421 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%415 : tensor<16x4xf32>) outs(%420 : tensor<16x4xi8>) attrs =  {prov.region_id = "dtype_cast_10", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb46(%422: f32, %423: i8):
      %424 = arith.fptosi %422 : f32 to i8
      linalg.yield %424 : i8
    } -> tensor<16x4xi8>
    %425 = tensor.empty() : tensor<16x32xi8>
    %426 = arith.constant 0 : i8
    %427 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%426 : i8) outs(%425 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %428 = linalg.matmul {prov.region_id = "matmul_4", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%421, %361 : tensor<16x4xi8>, tensor<4x32xi8>) outs(%427 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %429 = tensor.empty() : tensor<16x32xf32>
    %430 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%428 : tensor<16x32xi8>) outs(%429 : tensor<16x32xf32>) attrs =  {prov.region_id = "dtype_cast_11", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb47(%431: i8, %432: f32):
      %433 = arith.sitofp %431 : i8 to f32
      linalg.yield %433 : f32
    } -> tensor<16x32xf32>
    %434 = arith.constant {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 4.000000e-02 : f32
    %435 = tensor.splat %434 {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %436 = tensor.empty() : tensor<16x32xf32>
    %437 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%430, %435 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%436 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb48(%438: f32, %439: f32, %440: f32):
      %441 = arith.mulf %438, %439 : f32
      linalg.yield %441 : f32
    } -> tensor<16x32xf32>
    %442 = tensor.empty() : tensor<16x32xf32>
    %443 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%437 : tensor<16x32xf32>) outs(%442 : tensor<16x32xf32>) attrs =  {prov.region_id = "tanh_5", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb49(%444: f32, %445: f32):
      %446 = math.tanh %444 : f32
      linalg.yield %446 : f32
    } -> tensor<16x32xf32>
    %447 = arith.constant {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %448 = tensor.splat %447 {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %449 = tensor.empty() : tensor<16x32xf32>
    %450 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%443, %448 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%449 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb50(%451: f32, %452: f32, %453: f32):
      %454 = arith.mulf %451, %452 : f32
      linalg.yield %454 : f32
    } -> tensor<16x32xf32>
    %455 = tensor.empty() : tensor<16x32xi8>
    %456 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%450 : tensor<16x32xf32>) outs(%455 : tensor<16x32xi8>) attrs =  {prov.region_id = "dtype_cast_12", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb51(%457: f32, %458: i8):
      %459 = arith.fptosi %457 : f32 to i8
      linalg.yield %459 : i8
    } -> tensor<16x32xi8>
    %460 = tensor.empty() : tensor<16x32xi8>
    %461 = arith.constant 0 : i8
    %462 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%461 : i8) outs(%460 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %463 = linalg.matmul {prov.region_id = "matmul_5", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%456, %8 : tensor<16x32xi8>, tensor<32x32xi8>) outs(%462 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %464 = tensor.empty() : tensor<16x32xf32>
    %465 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%463 : tensor<16x32xi8>) outs(%464 : tensor<16x32xf32>) attrs =  {prov.region_id = "dtype_cast_13", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb52(%466: i8, %467: f32):
      %468 = arith.sitofp %466 : i8 to f32
      linalg.yield %468 : f32
    } -> tensor<16x32xf32>
    %469 = arith.constant {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.500000e-01 : f32
    %470 = tensor.splat %469 {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %471 = tensor.empty() : tensor<16x32xf32>
    %472 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%465, %470 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%471 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb53(%473: f32, %474: f32, %475: f32):
      %476 = arith.mulf %473, %474 : f32
      linalg.yield %476 : f32
    } -> tensor<16x32xf32>
    %477 = tensor.empty() : tensor<16x32xf32>
    %478 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%472 : tensor<16x32xf32>) outs(%477 : tensor<16x32xf32>) attrs =  {prov.region_id = "tanh_6", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb54(%479: f32, %480: f32):
      %481 = math.tanh %479 : f32
      linalg.yield %481 : f32
    } -> tensor<16x32xf32>
    %482 = arith.constant {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %483 = tensor.splat %482 {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %484 = tensor.empty() : tensor<16x32xf32>
    %485 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%478, %483 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%484 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb55(%486: f32, %487: f32, %488: f32):
      %489 = arith.mulf %486, %487 : f32
      linalg.yield %489 : f32
    } -> tensor<16x32xf32>
    %490 = tensor.empty() : tensor<16x32xi8>
    %491 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%485 : tensor<16x32xf32>) outs(%490 : tensor<16x32xi8>) attrs =  {prov.region_id = "dtype_cast_14", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb56(%492: f32, %493: i8):
      %494 = arith.fptosi %492 : f32 to i8
      linalg.yield %494 : i8
    } -> tensor<16x32xi8>
    %495 = tensor.empty() : tensor<16x32xi8>
    %496 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%233, %491 : tensor<16x32xi8>, tensor<16x32xi8>) outs(%495 : tensor<16x32xi8>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int8"} {
    ^bb57(%497: i8, %498: i8, %499: i8):
      %500 = arith.addi %497, %498 : i8
      linalg.yield %500 : i8
    } -> tensor<16x32xi8>
    %501 = tensor.empty() : tensor<16x32xf32>
    %502 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%496 : tensor<16x32xi8>) outs(%501 : tensor<16x32xf32>) attrs =  {prov.region_id = "dtype_cast_15", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb58(%503: i8, %504: f32):
      %505 = arith.sitofp %503 : i8 to f32
      linalg.yield %505 : f32
    } -> tensor<16x32xf32>
    %506 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} 0.000000e+00 : f32
    %507 = tensor.splat %506 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} : tensor<16xf32>
    %508 = linalg.reduce ins(%502:tensor<16x32xf32>) outs(%507:tensor<16xf32>) dimensions = [1]
    (%509: f32, %510: f32) {
      %511 = arith.addf %509, %510 : f32
      linalg.yield %511 : f32
    }
    %512 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} 3.200000e+01 : f32
    %513 = tensor.splat %512 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} : tensor<16xf32>
    %514 = tensor.empty() : tensor<16xf32>
    %515 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%508, %513 : tensor<16xf32>, tensor<16xf32>) outs(%514 : tensor<16xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb59(%516: f32, %517: f32, %518: f32):
      %519 = arith.divf %516, %517 : f32
      linalg.yield %519 : f32
    } -> tensor<16xf32>
    %520 = tensor.expand_shape %515 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} : tensor<16xf32> into tensor<16x1xf32>
    %521 = tensor.empty() : tensor<16x32xf32>
    %522 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%502, %520 : tensor<16x32xf32>, tensor<16x1xf32>) outs(%521 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb60(%523: f32, %524: f32, %525: f32):
      %526 = arith.subf %523, %524 : f32
      linalg.yield %526 : f32
    } -> tensor<16x32xf32>
    %527 = tensor.empty() : tensor<16x32xf32>
    %528 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%522, %522 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%527 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb61(%529: f32, %530: f32, %531: f32):
      %532 = arith.mulf %529, %530 : f32
      linalg.yield %532 : f32
    } -> tensor<16x32xf32>
    %533 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} 0.000000e+00 : f32
    %534 = tensor.splat %533 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} : tensor<16xf32>
    %535 = linalg.reduce ins(%528:tensor<16x32xf32>) outs(%534:tensor<16xf32>) dimensions = [1]
    (%536: f32, %537: f32) {
      %538 = arith.addf %536, %537 : f32
      linalg.yield %538 : f32
    }
    %539 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} 3.200000e+01 : f32
    %540 = tensor.splat %539 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} : tensor<16xf32>
    %541 = tensor.empty() : tensor<16xf32>
    %542 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%535, %540 : tensor<16xf32>, tensor<16xf32>) outs(%541 : tensor<16xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb62(%543: f32, %544: f32, %545: f32):
      %546 = arith.divf %543, %544 : f32
      linalg.yield %546 : f32
    } -> tensor<16xf32>
    %547 = tensor.expand_shape %542 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} : tensor<16xf32> into tensor<16x1xf32>
    %548 = arith.constant {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} 1.000000e-05 : f32
    %549 = tensor.splat %548 {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} : tensor<16x1xf32>
    %550 = tensor.empty() : tensor<16x1xf32>
    %551 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%547, %549 : tensor<16x1xf32>, tensor<16x1xf32>) outs(%550 : tensor<16x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb63(%552: f32, %553: f32, %554: f32):
      %555 = arith.addf %552, %553 : f32
      linalg.yield %555 : f32
    } -> tensor<16x1xf32>
    %556 = tensor.empty() : tensor<16x1xf32>
    %557 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%551 : tensor<16x1xf32>) outs(%556 : tensor<16x1xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb64(%558: f32, %559: f32):
      %560 = math.rsqrt %558 : f32
      linalg.yield %560 : f32
    } -> tensor<16x1xf32>
    %561 = tensor.empty() : tensor<16x32xf32>
    %562 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%522, %557 : tensor<16x32xf32>, tensor<16x1xf32>) outs(%561 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb65(%563: f32, %564: f32, %565: f32):
      %566 = arith.mulf %563, %564 : f32
      linalg.yield %566 : f32
    } -> tensor<16x32xf32>
    %567 = tensor.empty() : tensor<16x32xf32>
    %568 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%562, %2 : tensor<16x32xf32>, tensor<32xf32>) outs(%567 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb66(%569: f32, %570: f32, %571: f32):
      %572 = arith.mulf %569, %570 : f32
      linalg.yield %572 : f32
    } -> tensor<16x32xf32>
    %573 = tensor.empty() : tensor<16x32xf32>
    %574 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%568, %3 : tensor<16x32xf32>, tensor<32xf32>) outs(%573 : tensor<16x32xf32>) attrs =  {prov.region_id = "layer_norm_1", prov.family = "normalization", prov._pattern_hint = "layer_norm", prov.op = "layer_norm", prov.aten = "aten.native_layer_norm.default", prov.orig_dtype = "float32", prov.module = "ln2", prov.fqn = "ln2"} {
    ^bb67(%575: f32, %576: f32, %577: f32):
      %578 = arith.addf %575, %576 : f32
      linalg.yield %578 : f32
    } -> tensor<16x32xf32>
    %579 = arith.constant {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.100000e+00 : f32
    %580 = tensor.splat %579 {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %581 = tensor.empty() : tensor<16x32xf32>
    %582 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%574, %580 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%581 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb68(%583: f32, %584: f32, %585: f32):
      %586 = arith.mulf %583, %584 : f32
      linalg.yield %586 : f32
    } -> tensor<16x32xf32>
    %587 = tensor.empty() : tensor<16x32xf32>
    %588 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%582 : tensor<16x32xf32>) outs(%587 : tensor<16x32xf32>) attrs =  {prov.region_id = "tanh_7", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb69(%589: f32, %590: f32):
      %591 = math.tanh %589 : f32
      linalg.yield %591 : f32
    } -> tensor<16x32xf32>
    %592 = arith.constant {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %593 = tensor.splat %592 {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %594 = tensor.empty() : tensor<16x32xf32>
    %595 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%588, %593 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%594 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb70(%596: f32, %597: f32, %598: f32):
      %599 = arith.mulf %596, %597 : f32
      linalg.yield %599 : f32
    } -> tensor<16x32xf32>
    %600 = tensor.empty() : tensor<16x32xi8>
    %601 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%595 : tensor<16x32xf32>) outs(%600 : tensor<16x32xi8>) attrs =  {prov.region_id = "dtype_cast_16", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb71(%602: f32, %603: i8):
      %604 = arith.fptosi %602 : f32 to i8
      linalg.yield %604 : i8
    } -> tensor<16x32xi8>
    %605 = tensor.empty() : tensor<16x48xi8>
    %606 = arith.constant 0 : i8
    %607 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%606 : i8) outs(%605 : tensor<16x48xi8>) -> tensor<16x48xi8>
    %608 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%601, %9 : tensor<16x32xi8>, tensor<32x48xi8>) outs(%607 : tensor<16x48xi8>) -> tensor<16x48xi8>
    %609 = tensor.empty() : tensor<16x48xf32>
    %610 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%608 : tensor<16x48xi8>) outs(%609 : tensor<16x48xf32>) attrs =  {prov.region_id = "dtype_cast_17", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb72(%611: i8, %612: f32):
      %613 = arith.sitofp %611 : i8 to f32
      linalg.yield %613 : f32
    } -> tensor<16x48xf32>
    %614 = arith.constant {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e-01 : f32
    %615 = tensor.splat %614 {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x48xf32>
    %616 = tensor.empty() : tensor<16x48xf32>
    %617 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%610, %615 : tensor<16x48xf32>, tensor<16x48xf32>) outs(%616 : tensor<16x48xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb73(%618: f32, %619: f32, %620: f32):
      %621 = arith.mulf %618, %619 : f32
      linalg.yield %621 : f32
    } -> tensor<16x48xf32>
    %622 = tensor.collapse_shape %617 [[0 : i64, 1 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16x48xf32> into tensor<768xf32>
    %623 = tensor.expand_shape %622 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 4, 48] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<768xf32> into tensor<1x4x4x48xf32>
    %624 = tensor.empty() : tensor<1x48x4x4xf32>
    %625 = linalg.transpose ins(%623:tensor<1x4x4x48xf32>) outs(%624:tensor<1x48x4x4xf32>) permutation = [0, 3, 1, 2]
    %626 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0.017254902 : f32
    %627 = tensor.splat %626 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<f32>
    %628 = arith.constant {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} 0 : i64
    %629 = tensor.splat %628 {prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : tensor<i64>
    %630 = "quant_ext.quantize_per_tensor"(%625, %627, %629) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> {prov.region_id = "quantize_0", prov._pattern_hint = "quantize_per_tensor", prov.op = "quantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.quantize_per_tensor.default", prov.orig_dtype = "int8"} : (tensor<1x48x4x4xf32>, tensor<f32>, tensor<i64>) -> tensor<1x48x4x4xi8>
    %631 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0.017254902 : f32
    %632 = tensor.splat %631 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<f32>
    %633 = arith.constant {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} 0 : i64
    %634 = tensor.splat %633 {prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : tensor<i64>
    %635 = "quant_ext.dequantize_per_tensor"(%630, %632, %634) <{quant_min = -128 : i64, quant_max = 127 : i64}> {prov.region_id = "dequantize_1", prov._pattern_hint = "dequantize_per_tensor", prov.op = "dequantize_per_tensor", prov.family = "quantize", prov.aten = "quantized_decomposed.dequantize_per_tensor.default", prov.orig_dtype = "float32"} : (tensor<1x48x4x4xi8>, tensor<f32>, tensor<i64>) -> tensor<1x48x4x4xf32>
    %636 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} 0.000000e+00 : f32
    %637 = tensor.splat %636 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<1x48x6x6xf32>
    %638 = "tensor.insert_slice"(%635, %637) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 48, 4, 4>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : (tensor<1x48x4x4xf32>, tensor<1x48x6x6xf32>) -> tensor<1x48x6x6xf32>
    %639 = tensor.empty() : tensor<48x1x3x3x1x4x4xf32>
    %640 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, (d0 + d1), (d5 + d2), (d6 + d3))>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%638 : tensor<1x48x6x6xf32>) outs(%639 : tensor<48x1x3x3x1x4x4xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} {
    ^bb74(%641: f32, %642: f32):
      linalg.yield %641 : f32
    } -> tensor<48x1x3x3x1x4x4xf32>
    %643 = tensor.collapse_shape %640 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<48x1x3x3x1x4x4xf32> into tensor<6912xf32>
    %644 = tensor.expand_shape %643 [[0 : i64, 1 : i64, 2 : i64]] output_shape [48, 9, 16] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<6912xf32> into tensor<48x9x16xf32>
    %645 = tensor.collapse_shape %25 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<48x1x3x3xf32> into tensor<432xf32>
    %646 = tensor.expand_shape %645 [[0 : i64, 1 : i64, 2 : i64]] output_shape [48, 1, 9] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<432xf32> into tensor<48x1x9xf32>
    %647 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} 0.000000e+00 : f32
    %648 = tensor.splat %647 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<48x1x16xf32>
    %649 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%646, %644 : tensor<48x1x9xf32>, tensor<48x9x16xf32>) outs(%648 : tensor<48x1x16xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} {
    ^bb75(%650: f32, %651: f32, %652: f32):
      %653 = arith.mulf %650, %651 : f32
      %654 = arith.addf %652, %653 : f32
      linalg.yield %654 : f32
    } -> tensor<48x1x16xf32>
    %655 = tensor.collapse_shape %649 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<48x1x16xf32> into tensor<768xf32>
    %656 = tensor.expand_shape %655 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [48, 1, 4, 4] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<768xf32> into tensor<48x1x4x4xf32>
    %657 = tensor.collapse_shape %656 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<48x1x4x4xf32> into tensor<768xf32>
    %658 = tensor.expand_shape %657 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 48, 4, 4] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} : tensor<768xf32> into tensor<1x48x4x4xf32>
    %659 = tensor.empty() : tensor<1x48x4x4xf32>
    %660 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%658, %4 : tensor<1x48x4x4xf32>, tensor<48xf32>) outs(%659 : tensor<1x48x4x4xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.convolution.default", prov.orig_dtype = "float32", prov.module = "dw", prov.fqn = "dw"} {
    ^bb76(%661: f32, %662: f32, %663: f32):
      %664 = arith.addf %661, %662 : f32
      linalg.yield %664 : f32
    } -> tensor<1x48x4x4xf32>
    %665 = tensor.empty() : tensor<1x48x4x4xf32>
    %666 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%660 : tensor<1x48x4x4xf32>) outs(%665 : tensor<1x48x4x4xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32"} {
    ^bb77(%667: f32, %668: f32):
      %669 = arith.constant 5.000000e-01 : f32
      %670 = arith.constant 1.000000e+00 : f32
      %671 = arith.constant 0.707106769 : f32
      %672 = arith.mulf %667, %671 : f32
      %673 = math.erf %672 : f32
      %674 = arith.addf %670, %673 : f32
      %675 = arith.mulf %669, %667 : f32
      %676 = arith.mulf %675, %674 : f32
      linalg.yield %676 : f32
    } -> tensor<1x48x4x4xf32>
    %677 = tensor.empty() : tensor<1x4x4x48xf32>
    %678 = linalg.transpose ins(%666:tensor<1x48x4x4xf32>) outs(%677:tensor<1x4x4x48xf32>) permutation = [0, 2, 3, 1]
    %679 = tensor.collapse_shape %678 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x4x4x48xf32> into tensor<768xf32>
    %680 = tensor.expand_shape %679 [[0 : i64, 1 : i64]] output_shape [16, 48] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<768xf32> into tensor<16x48xf32>
    %681 = arith.constant {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 4.000000e+00 : f32
    %682 = tensor.splat %681 {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x48xf32>
    %683 = tensor.empty() : tensor<16x48xf32>
    %684 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%680, %682 : tensor<16x48xf32>, tensor<16x48xf32>) outs(%683 : tensor<16x48xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb78(%685: f32, %686: f32, %687: f32):
      %688 = arith.mulf %685, %686 : f32
      linalg.yield %688 : f32
    } -> tensor<16x48xf32>
    %689 = tensor.empty() : tensor<16x48xf32>
    %690 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%684 : tensor<16x48xf32>) outs(%689 : tensor<16x48xf32>) attrs =  {prov.region_id = "tanh_8", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb79(%691: f32, %692: f32):
      %693 = math.tanh %691 : f32
      linalg.yield %693 : f32
    } -> tensor<16x48xf32>
    %694 = arith.constant {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 2.000000e+00 : f32
    %695 = tensor.splat %694 {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x48xf32>
    %696 = tensor.empty() : tensor<16x48xf32>
    %697 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%690, %695 : tensor<16x48xf32>, tensor<16x48xf32>) outs(%696 : tensor<16x48xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb80(%698: f32, %699: f32, %700: f32):
      %701 = arith.mulf %698, %699 : f32
      linalg.yield %701 : f32
    } -> tensor<16x48xf32>
    %702 = tensor.empty() : tensor<16x48xi8>
    %703 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%697 : tensor<16x48xf32>) outs(%702 : tensor<16x48xi8>) attrs =  {prov.region_id = "dtype_cast_18", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb81(%704: f32, %705: i8):
      %706 = arith.fptosi %704 : f32 to i8
      linalg.yield %706 : i8
    } -> tensor<16x48xi8>
    %707 = tensor.empty() : tensor<16x32xi8>
    %708 = arith.constant 0 : i8
    %709 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%708 : i8) outs(%707 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %710 = linalg.matmul {prov.region_id = "matmul_7", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%703, %10 : tensor<16x48xi8>, tensor<48x32xi8>) outs(%709 : tensor<16x32xi8>) -> tensor<16x32xi8>
    %711 = tensor.empty() : tensor<16x32xf32>
    %712 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%710 : tensor<16x32xi8>) outs(%711 : tensor<16x32xf32>) attrs =  {prov.region_id = "dtype_cast_19", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb82(%713: i8, %714: f32):
      %715 = arith.sitofp %713 : i8 to f32
      linalg.yield %715 : f32
    } -> tensor<16x32xf32>
    %716 = arith.constant {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e-01 : f32
    %717 = tensor.splat %716 {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<16x32xf32>
    %718 = tensor.empty() : tensor<16x32xf32>
    %719 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%712, %717 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%718 : tensor<16x32xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb83(%720: f32, %721: f32, %722: f32):
      %723 = arith.mulf %720, %721 : f32
      linalg.yield %723 : f32
    } -> tensor<16x32xf32>
    %724 = tensor.empty() : tensor<16x32xf32>
    %725 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%574, %719 : tensor<16x32xf32>, tensor<16x32xf32>) outs(%724 : tensor<16x32xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb84(%726: f32, %727: f32, %728: f32):
      %729 = arith.addf %726, %727 : f32
      linalg.yield %729 : f32
    } -> tensor<16x32xf32>
    %730 = tensor.collapse_shape %725 [[0 : i64, 1 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<16x32xf32> into tensor<512xf32>
    %731 = tensor.expand_shape %730 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 4, 4, 32] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x4x4x32xf32>
    %732 = tensor.empty() : tensor<1x32x4x4xf32>
    %733 = linalg.transpose ins(%731:tensor<1x4x4x32xf32>) outs(%732:tensor<1x32x4x4xf32>) permutation = [0, 3, 1, 2]
    %734 = tensor.collapse_shape %733 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x4x4xf32> into tensor<512xf32>
    %735 = tensor.expand_shape %734 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] output_shape [1, 8, 2, 2, 4, 4] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x8x2x2x4x4xf32>
    %736 = tensor.empty() : tensor<1x8x4x2x4x2xf32>
    %737 = linalg.transpose ins(%735:tensor<1x8x2x2x4x4xf32>) outs(%736:tensor<1x8x4x2x4x2xf32>) permutation = [0, 1, 4, 2, 5, 3]
    %738 = tensor.collapse_shape %737 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x4x2x4x2xf32> into tensor<512xf32>
    %739 = tensor.expand_shape %738 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 8, 8] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x8x8x8xf32>
    %740 = tensor.collapse_shape %739 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x8x8xf32> into tensor<512xf32>
    %741 = tensor.expand_shape %740 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] output_shape [1, 8, 2, 4, 2, 4] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<512xf32> into tensor<1x8x2x4x2x4xf32>
    %742 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %743 = tensor.splat %742 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} : tensor<1x8x2x4x2xf32>
    %744 = linalg.reduce ins(%741:tensor<1x8x2x4x2x4xf32>) outs(%743:tensor<1x8x2x4x2xf32>) dimensions = [5]
    (%745: f32, %746: f32) {
      %747 = arith.maximumf %745, %746 : f32
      linalg.yield %747 : f32
    }
    %748 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %749 = tensor.splat %748 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} : tensor<1x8x2x2xf32>
    %750 = linalg.reduce ins(%744:tensor<1x8x2x4x2xf32>) outs(%749:tensor<1x8x2x2xf32>) dimensions = [3]
    (%751: f32, %752: f32) {
      %753 = arith.maximumf %751, %752 : f32
      linalg.yield %753 : f32
    }
    %754 = tensor.collapse_shape %750 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x8x2x2xf32> into tensor<32xf32>
    %755 = tensor.expand_shape %754 [[0 : i64, 1 : i64]] output_shape [1, 32] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32xf32>
    %756 = arith.constant {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 6.000000e-01 : f32
    %757 = tensor.splat %756 {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x32xf32>
    %758 = tensor.empty() : tensor<1x32xf32>
    %759 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%755, %757 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%758 : tensor<1x32xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb85(%760: f32, %761: f32, %762: f32):
      %763 = arith.mulf %760, %761 : f32
      linalg.yield %763 : f32
    } -> tensor<1x32xf32>
    %764 = tensor.empty() : tensor<1x32xf32>
    %765 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%759 : tensor<1x32xf32>) outs(%764 : tensor<1x32xf32>) attrs =  {prov.region_id = "tanh_9", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb86(%766: f32, %767: f32):
      %768 = math.tanh %766 : f32
      linalg.yield %768 : f32
    } -> tensor<1x32xf32>
    %769 = arith.constant {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 3.000000e+00 : f32
    %770 = tensor.splat %769 {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x32xf32>
    %771 = tensor.empty() : tensor<1x32xf32>
    %772 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%765, %770 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%771 : tensor<1x32xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb87(%773: f32, %774: f32, %775: f32):
      %776 = arith.mulf %773, %774 : f32
      linalg.yield %776 : f32
    } -> tensor<1x32xf32>
    %777 = tensor.empty() : tensor<1x32xi8>
    %778 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%772 : tensor<1x32xf32>) outs(%777 : tensor<1x32xi8>) attrs =  {prov.region_id = "dtype_cast_20", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb88(%779: f32, %780: i8):
      %781 = arith.fptosi %779 : f32 to i8
      linalg.yield %781 : i8
    } -> tensor<1x32xi8>
    %782 = tensor.empty() : tensor<1x32xi8>
    %783 = arith.constant 0 : i8
    %784 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%783 : i8) outs(%782 : tensor<1x32xi8>) -> tensor<1x32xi8>
    %785 = linalg.matmul {prov.region_id = "matmul_8", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%778, %11 : tensor<1x32xi8>, tensor<32x32xi8>) outs(%784 : tensor<1x32xi8>) -> tensor<1x32xi8>
    %786 = tensor.empty() : tensor<1x32xf32>
    %787 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%785 : tensor<1x32xi8>) outs(%786 : tensor<1x32xf32>) attrs =  {prov.region_id = "dtype_cast_21", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb89(%788: i8, %789: f32):
      %790 = arith.sitofp %788 : i8 to f32
      linalg.yield %790 : f32
    } -> tensor<1x32xf32>
    %791 = arith.constant {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e-01 : f32
    %792 = tensor.splat %791 {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x32xf32>
    %793 = tensor.empty() : tensor<1x32xf32>
    %794 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%787, %792 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%793 : tensor<1x32xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb90(%795: f32, %796: f32, %797: f32):
      %798 = arith.mulf %795, %796 : f32
      linalg.yield %798 : f32
    } -> tensor<1x32xf32>
    %799 = tensor.concat dim(1) %794, %20 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x32xf32>, tensor<1x16xf32>) -> tensor<1x48xf32>
    %800 = arith.constant {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.200000e+00 : f32
    %801 = tensor.splat %800 {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x48xf32>
    %802 = tensor.empty() : tensor<1x48xf32>
    %803 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%799, %801 : tensor<1x48xf32>, tensor<1x48xf32>) outs(%802 : tensor<1x48xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb91(%804: f32, %805: f32, %806: f32):
      %807 = arith.mulf %804, %805 : f32
      linalg.yield %807 : f32
    } -> tensor<1x48xf32>
    %808 = tensor.empty() : tensor<1x48xf32>
    %809 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%803 : tensor<1x48xf32>) outs(%808 : tensor<1x48xf32>) attrs =  {prov.region_id = "tanh_10", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb92(%810: f32, %811: f32):
      %812 = math.tanh %810 : f32
      linalg.yield %812 : f32
    } -> tensor<1x48xf32>
    %813 = arith.constant {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 2.000000e+00 : f32
    %814 = tensor.splat %813 {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x48xf32>
    %815 = tensor.empty() : tensor<1x48xf32>
    %816 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%809, %814 : tensor<1x48xf32>, tensor<1x48xf32>) outs(%815 : tensor<1x48xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb93(%817: f32, %818: f32, %819: f32):
      %820 = arith.mulf %817, %818 : f32
      linalg.yield %820 : f32
    } -> tensor<1x48xf32>
    %821 = tensor.empty() : tensor<1x48xi8>
    %822 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%816 : tensor<1x48xf32>) outs(%821 : tensor<1x48xi8>) attrs =  {prov.region_id = "dtype_cast_22", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb94(%823: f32, %824: i8):
      %825 = arith.fptosi %823 : f32 to i8
      linalg.yield %825 : i8
    } -> tensor<1x48xi8>
    %826 = tensor.empty() : tensor<1x64xi8>
    %827 = arith.constant 0 : i8
    %828 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%827 : i8) outs(%826 : tensor<1x64xi8>) -> tensor<1x64xi8>
    %829 = linalg.matmul {prov.region_id = "matmul_9", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%822, %12 : tensor<1x48xi8>, tensor<48x64xi8>) outs(%828 : tensor<1x64xi8>) -> tensor<1x64xi8>
    %830 = tensor.empty() : tensor<1x64xf32>
    %831 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%829 : tensor<1x64xi8>) outs(%830 : tensor<1x64xf32>) attrs =  {prov.region_id = "dtype_cast_23", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb95(%832: i8, %833: f32):
      %834 = arith.sitofp %832 : i8 to f32
      linalg.yield %834 : f32
    } -> tensor<1x64xf32>
    %835 = tensor.empty() : tensor<1x64xi8>
    %836 = arith.constant 0 : i8
    %837 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%836 : i8) outs(%835 : tensor<1x64xi8>) -> tensor<1x64xi8>
    %838 = linalg.matmul {prov.region_id = "matmul_10", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%15, %13 : tensor<1x16xi8>, tensor<16x64xi8>) outs(%837 : tensor<1x64xi8>) -> tensor<1x64xi8>
    %839 = tensor.empty() : tensor<1x64xf32>
    %840 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%838 : tensor<1x64xi8>) outs(%839 : tensor<1x64xf32>) attrs =  {prov.region_id = "dtype_cast_24", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb96(%841: i8, %842: f32):
      %843 = arith.sitofp %841 : i8 to f32
      linalg.yield %843 : f32
    } -> tensor<1x64xf32>
    %844 = tensor.empty() : tensor<1x64xf32>
    %845 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%831, %840 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%844 : tensor<1x64xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb97(%846: f32, %847: f32, %848: f32):
      %849 = arith.addf %846, %847 : f32
      linalg.yield %849 : f32
    } -> tensor<1x64xf32>
    %850 = arith.constant {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.200000e-01 : f32
    %851 = tensor.splat %850 {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x64xf32>
    %852 = tensor.empty() : tensor<1x64xf32>
    %853 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%845, %851 : tensor<1x64xf32>, tensor<1x64xf32>) outs(%852 : tensor<1x64xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb98(%854: f32, %855: f32, %856: f32):
      %857 = arith.mulf %854, %855 : f32
      linalg.yield %857 : f32
    } -> tensor<1x64xf32>
    %858 = tensor.empty() : tensor<1x64xf32>
    %859 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%853, %14 : tensor<1x64xf32>, tensor<64xf32>) outs(%858 : tensor<1x64xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb99(%860: f32, %861: f32, %862: f32):
      %863 = arith.addf %860, %861 : f32
      linalg.yield %863 : f32
    } -> tensor<1x64xf32>
    %864 = "tensor.extract_slice"(%859) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 16>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x64xf32>) -> tensor<1x16xf32>
    %865 = "tensor.extract_slice"(%859) <{static_offsets = array<i64: 0, 16>, static_sizes = array<i64: 1, 16>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x64xf32>) -> tensor<1x16xf32>
    %866 = "tensor.extract_slice"(%859) <{static_offsets = array<i64: 0, 32>, static_sizes = array<i64: 1, 16>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x64xf32>) -> tensor<1x16xf32>
    %867 = "tensor.extract_slice"(%859) <{static_offsets = array<i64: 0, 48>, static_sizes = array<i64: 1, 16>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "split", prov.op = "split", prov.aten = "aten.split_with_sizes.default", prov.orig_dtype = "float32"} : (tensor<1x64xf32>) -> tensor<1x16xf32>
    %868 = tensor.empty() : tensor<1x16xf32>
    %869 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%865 : tensor<1x16xf32>) outs(%868 : tensor<1x16xf32>) attrs =  {prov.region_id = "sigmoid_0", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32"} {
    ^bb100(%870: f32, %871: f32):
      %872 = arith.constant 1.000000e+00 : f32
      %873 = arith.negf %870 : f32
      %874 = math.exp %873 : f32
      %875 = arith.addf %872, %874 : f32
      %876 = arith.divf %872, %875 : f32
      linalg.yield %876 : f32
    } -> tensor<1x16xf32>
    %877 = tensor.empty() : tensor<1x16xf32>
    %878 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%869, %16 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%877 : tensor<1x16xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb101(%879: f32, %880: f32, %881: f32):
      %882 = arith.mulf %879, %880 : f32
      linalg.yield %882 : f32
    } -> tensor<1x16xf32>
    %883 = tensor.empty() : tensor<1x16xf32>
    %884 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%864 : tensor<1x16xf32>) outs(%883 : tensor<1x16xf32>) attrs =  {prov.region_id = "sigmoid_1", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32"} {
    ^bb102(%885: f32, %886: f32):
      %887 = arith.constant 1.000000e+00 : f32
      %888 = arith.negf %885 : f32
      %889 = math.exp %888 : f32
      %890 = arith.addf %887, %889 : f32
      %891 = arith.divf %887, %890 : f32
      linalg.yield %891 : f32
    } -> tensor<1x16xf32>
    %892 = tensor.empty() : tensor<1x16xf32>
    %893 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%866 : tensor<1x16xf32>) outs(%892 : tensor<1x16xf32>) attrs =  {prov.region_id = "tanh_11", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb103(%894: f32, %895: f32):
      %896 = math.tanh %894 : f32
      linalg.yield %896 : f32
    } -> tensor<1x16xf32>
    %897 = tensor.empty() : tensor<1x16xf32>
    %898 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%884, %893 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%897 : tensor<1x16xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb104(%899: f32, %900: f32, %901: f32):
      %902 = arith.mulf %899, %900 : f32
      linalg.yield %902 : f32
    } -> tensor<1x16xf32>
    %903 = tensor.empty() : tensor<1x16xf32>
    %904 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%878, %898 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%903 : tensor<1x16xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb105(%905: f32, %906: f32, %907: f32):
      %908 = arith.addf %905, %906 : f32
      linalg.yield %908 : f32
    } -> tensor<1x16xf32>
    %909 = tensor.empty() : tensor<1x16xf32>
    %910 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%867 : tensor<1x16xf32>) outs(%909 : tensor<1x16xf32>) attrs =  {prov.region_id = "sigmoid_2", prov._pattern_hint = "sigmoid", prov.op = "sigmoid", prov.family = "elementwise", prov.aten = "aten.sigmoid.default", prov.orig_dtype = "float32"} {
    ^bb106(%911: f32, %912: f32):
      %913 = arith.constant 1.000000e+00 : f32
      %914 = arith.negf %911 : f32
      %915 = math.exp %914 : f32
      %916 = arith.addf %913, %915 : f32
      %917 = arith.divf %913, %916 : f32
      linalg.yield %917 : f32
    } -> tensor<1x16xf32>
    %918 = tensor.empty() : tensor<1x16xf32>
    %919 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%904 : tensor<1x16xf32>) outs(%918 : tensor<1x16xf32>) attrs =  {prov.region_id = "tanh_12", prov._pattern_hint = "tanh", prov.op = "tanh", prov.family = "elementwise", prov.aten = "aten.tanh.default", prov.orig_dtype = "float32"} {
    ^bb107(%920: f32, %921: f32):
      %922 = math.tanh %920 : f32
      linalg.yield %922 : f32
    } -> tensor<1x16xf32>
    %923 = tensor.empty() : tensor<1x16xf32>
    %924 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%910, %919 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%923 : tensor<1x16xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb108(%925: f32, %926: f32, %927: f32):
      %928 = arith.mulf %925, %926 : f32
      linalg.yield %928 : f32
    } -> tensor<1x16xf32>
    %929 = arith.constant {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 7.000000e+00 : f32
    %930 = tensor.splat %929 {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x16xf32>
    %931 = tensor.empty() : tensor<1x16xf32>
    %932 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%924, %930 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%931 : tensor<1x16xf32>) attrs =  {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb109(%933: f32, %934: f32, %935: f32):
      %936 = arith.mulf %933, %934 : f32
      linalg.yield %936 : f32
    } -> tensor<1x16xf32>
    %937 = tensor.empty() : tensor<1x16xi8>
    %938 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%932 : tensor<1x16xf32>) outs(%937 : tensor<1x16xi8>) attrs =  {prov.region_id = "dtype_cast_25", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int8"} {
    ^bb110(%939: f32, %940: i8):
      %941 = arith.fptosi %939 : f32 to i8
      linalg.yield %941 : i8
    } -> tensor<1x16xi8>
    %942 = tensor.empty() : tensor<1x16xi8>
    %943 = arith.constant 0 : i8
    %944 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%943 : i8) outs(%942 : tensor<1x16xi8>) -> tensor<1x16xi8>
    %945 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "int8"} ins(%938, %17 : tensor<1x16xi8>, tensor<16x16xi8>) outs(%944 : tensor<1x16xi8>) -> tensor<1x16xi8>
    %946 = tensor.empty() : tensor<1x16xf32>
    %947 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%945 : tensor<1x16xi8>) outs(%946 : tensor<1x16xf32>) attrs =  {prov.region_id = "dtype_cast_26", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb111(%948: i8, %949: f32):
      %950 = arith.sitofp %948 : i8 to f32
      linalg.yield %950 : f32
    } -> tensor<1x16xf32>
    %951 = arith.constant {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.200000e-01 : f32
    %952 = tensor.splat %951 {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x16xf32>
    %953 = tensor.empty() : tensor<1x16xf32>
    %954 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%947, %952 : tensor<1x16xf32>, tensor<1x16xf32>) outs(%953 : tensor<1x16xf32>) attrs =  {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb112(%955: f32, %956: f32, %957: f32):
      %958 = arith.mulf %955, %956 : f32
      linalg.yield %958 : f32
    } -> tensor<1x16xf32>
    func.return %954 : tensor<1x16xf32>
  }
}
