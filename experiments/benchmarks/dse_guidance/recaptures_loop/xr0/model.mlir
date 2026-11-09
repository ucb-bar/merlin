builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func private @aten_zeros_default() -> tensor<i64>
  func.func @forward(%0: tensor<6x1024xf32>, %1: tensor<3072x1024xf32>, %2: tensor<3072xf32>, %3: tensor<1024x1024xf32>, %4: tensor<128xf32>, %5: tensor<128xf32>, %6: tensor<4096x1024xf32>, %7: tensor<4096x1024xf32>, %8: tensor<1024x4096xf32>, %9: tensor<1024xf32>, %10: tensor<1024xf32>, %11: tensor<1024xf32>, %12: tensor<1024xf32>, %13: tensor<6x1024xf32>, %14: tensor<3072x1024xf32>, %15: tensor<3072xf32>, %16: tensor<1024x1024xf32>, %17: tensor<128xf32>, %18: tensor<128xf32>, %19: tensor<4096x1024xf32>, %20: tensor<4096x1024xf32>, %21: tensor<1024x4096xf32>, %22: tensor<1024xf32>, %23: tensor<1024xf32>, %24: tensor<1024xf32>, %25: tensor<1024xf32>, %26: tensor<1024x32xf32>, %27: tensor<1024x1024xf32>, %28: tensor<1024x32xf32>, %29: tensor<1024x1024xf32>, %30: tensor<32x1024xf32>, %31: tensor<32x32xf32>, %32: tensor<1024x256xf32>, %33: tensor<1024x1024xf32>, %34: tensor<6144x1024xf32>, %35: tensor<6144xf32>, %36: tensor<1x1024xf32>, %37: tensor<64xf32>, %38: tensor<1x30x32xf32>, %39: tensor<1x1x1xf32>, %40: tensor<1x30x32xf32>, %41: tensor<1x1x32xf32>, %42: tensor<1x32x128xf32>, %43: tensor<1x32x128xf32>, %44: tensor<1x1x32x48xi1>, %45: tensor<1x8x16x128xf32>, %46: tensor<1x8x16x128xf32>, %47: tensor<1x8x16x128xf32>, %48: tensor<1x8x16x128xf32>) -> tensor<1x30x32xf32> {
    %49 = tensor.empty() : tensor<32x1024xf32>
    %50 = linalg.transpose ins(%26:tensor<1024x32xf32>) outs(%49:tensor<32x1024xf32>) permutation = [1, 0]
    %51 = tensor.empty() : tensor<1x1x1024xf32>
    %52 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %53 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%52 : f32) outs(%51 : tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
    %54 = linalg.matmul {prov.region_id = "matmul_0", prov.dispatch_id = "matmul_0", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.model.state_projector.layers.0"} ins(%41, %50 : tensor<1x1x32xf32>, tensor<32x1024xf32>) outs(%53 : tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
    %55 = tensor.empty() : tensor<1x1x1024xf32>
    %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%54 : tensor<1x1x1024xf32>) outs(%55 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.model.state_projector.layers.1"} {
    ^bb0(%57: f32, %58: f32):
      %59 = arith.constant 5.000000e-01 : f32
      %60 = arith.constant 1.000000e+00 : f32
      %61 = arith.constant 0.707106769 : f32
      %62 = arith.mulf %57, %61 : f32
      %63 = math.erf %62 : f32
      %64 = arith.addf %60, %63 : f32
      %65 = arith.mulf %59, %57 : f32
      %66 = arith.mulf %65, %64 : f32
      linalg.yield %66 : f32
    } -> tensor<1x1x1024xf32>
    %67 = tensor.empty() : tensor<1024x1024xf32>
    %68 = linalg.transpose ins(%27:tensor<1024x1024xf32>) outs(%67:tensor<1024x1024xf32>) permutation = [1, 0]
    %69 = tensor.empty() : tensor<1x1x1024xf32>
    %70 = arith.constant {prov.module = "model"} 0.000000e+00 : f32
    %71 = linalg.fill {prov.op = "fill", prov.family = "fill", prov.module = "model"} ins(%70 : f32) outs(%69 : tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
    %72 = linalg.matmul {prov.region_id = "matmul_1", prov.dispatch_id = "matmul_1", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.model.state_projector.layers.2"} ins(%56, %68 : tensor<1x1x1024xf32>, tensor<1024x1024xf32>) outs(%71 : tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
    %73 = tensor.empty() : tensor<128xf32>
    %74 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%73 : tensor<128xf32>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start", prov.orig_dtype = "float32"} {
    ^bb1(%75: f32):
      %76 = linalg.index 0 : index
      %77 = arith.index_cast %76 : index to i64
      %78 = arith.sitofp %77 : i64 to f32
      %79 = arith.constant 1.000000e+00 : f32
      %80 = arith.mulf %78, %79 : f32
      %81 = arith.constant 0.000000e+00 : f32
      %82 = arith.addf %81, %80 : f32
      linalg.yield %82 : f32
    } -> tensor<128xf32>
    %83 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} -9.2103405 : f32
    %84 = tensor.splat %83 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<128xf32>
    %85 = tensor.empty() : tensor<128xf32>
    %86 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%74, %84 : tensor<128xf32>, tensor<128xf32>) outs(%85 : tensor<128xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb2(%87: f32, %88: f32, %89: f32):
      %90 = arith.mulf %87, %88 : f32
      linalg.yield %90 : f32
    } -> tensor<128xf32>
    %91 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 1.280000e+02 : f32
    %92 = tensor.splat %91 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<128xf32>
    %93 = tensor.empty() : tensor<128xf32>
    %94 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%86, %92 : tensor<128xf32>, tensor<128xf32>) outs(%93 : tensor<128xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb3(%95: f32, %96: f32, %97: f32):
      %98 = arith.divf %95, %96 : f32
      linalg.yield %98 : f32
    } -> tensor<128xf32>
    %99 = tensor.empty() : tensor<128xf32>
    %100 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%94 : tensor<128xf32>) outs(%99 : tensor<128xf32>) attrs =  {prov.region_id = "exp_0", prov._pattern_hint = "exp", prov.op = "exp", prov.family = "elementwise", prov.aten = "aten.exp.default", prov.orig_dtype = "float32"} {
    ^bb4(%101: f32, %102: f32):
      %103 = math.exp %101 : f32
      linalg.yield %103 : f32
    } -> tensor<128xf32>
    %104 = func.call @aten_zeros_default() {prov.region_id = "aten_zeros_default_0", prov.dispatch_id = "aten_zeros_default_0"} : () -> tensor<i64>
    %105 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 0 : index
    %106 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 5 : index
    %107 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 1 : index
    %108, %109 = scf.for %110 = %105 to %106 step %107 iter_args(%111 = %104, %112 = %38) -> (tensor<i64>, tensor<1x30x32xf32>) {
      %113 = tensor.empty() : tensor<f32>
      %114 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%111 : tensor<i64>) outs(%113 : tensor<f32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32"} {
      ^bb5(%115: i64, %116: f32):
        %117 = arith.sitofp %115 : i64 to f32
        linalg.yield %117 : f32
      } -> tensor<f32>
      %118 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 2.000000e-01 : f32
      %119 = tensor.splat %118 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<f32>
      %120 = tensor.empty() : tensor<f32>
      %121 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%114, %119 : tensor<f32>, tensor<f32>) outs(%120 : tensor<f32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb6(%122: f32, %123: f32, %124: f32):
        %125 = arith.mulf %122, %123 : f32
        linalg.yield %125 : f32
      } -> tensor<f32>
      %126 = tensor.extract %121[] : tensor<f32>
      %127 = tensor.from_elements %126 {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x1x1xf32>
      %128 = tensor.empty() : tensor<1x1x1xf32>
      %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%127 : tensor<1x1x1xf32>) outs(%128 : tensor<1x1x1xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb7(%130: f32, %131: f32):
        linalg.yield %130 : f32
      } -> tensor<1x1x1xf32>
      %132 = tensor.collapse_shape %129 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x1x1xf32> into tensor<1xf32>
      %133 = arith.constant {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 1.000000e+03 : f32
      %134 = tensor.splat %133 {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1xf32>
      %135 = tensor.empty() : tensor<1xf32>
      %136 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%132, %134 : tensor<1xf32>, tensor<1xf32>) outs(%135 : tensor<1xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb8(%137: f32, %138: f32, %139: f32):
        %140 = arith.mulf %137, %138 : f32
        linalg.yield %140 : f32
      } -> tensor<1xf32>
      %141 = "tensor.extract_slice"(%136) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1xf32>) -> tensor<1xf32>
      %142 = tensor.expand_shape %141 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1xf32>
      %143 = tensor.expand_shape %100 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x128xf32>
      %144 = tensor.empty() : tensor<1x128xf32>
      %145 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%142, %143 : tensor<1x1xf32>, tensor<1x128xf32>) outs(%144 : tensor<1x128xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb9(%146: f32, %147: f32, %148: f32):
        %149 = arith.mulf %146, %147 : f32
        linalg.yield %149 : f32
      } -> tensor<1x128xf32>
      %150 = tensor.empty() : tensor<1x128xf32>
      %151 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%145 : tensor<1x128xf32>) outs(%150 : tensor<1x128xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32"} {
      ^bb10(%152: f32, %153: f32):
        %154 = math.cos %152 : f32
        linalg.yield %154 : f32
      } -> tensor<1x128xf32>
      %155 = tensor.empty() : tensor<1x128xf32>
      %156 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%145 : tensor<1x128xf32>) outs(%155 : tensor<1x128xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32"} {
      ^bb11(%157: f32, %158: f32):
        %159 = math.sin %157 : f32
        linalg.yield %159 : f32
      } -> tensor<1x128xf32>
      %160 = tensor.concat dim(1) %151, %156 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x256xf32>
      %161 = tensor.empty() : tensor<256x1024xf32>
      %162 = linalg.transpose ins(%32:tensor<1024x256xf32>) outs(%161:tensor<256x1024xf32>) permutation = [1, 0]
      %163 = tensor.empty() : tensor<1x1024xf32>
      %164 = arith.constant 0.000000e+00 : f32
      %165 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%164 : f32) outs(%163 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %166 = linalg.matmul {prov.region_id = "matmul_0", prov.dispatch_id = "matmul_0", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%160, %162 : tensor<1x256xf32>, tensor<256x1024xf32>) outs(%165 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %167 = tensor.empty() : tensor<1x1024xf32>
      %168 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%166 : tensor<1x1024xf32>) outs(%167 : tensor<1x1024xf32>) attrs =  {prov.region_id = "silu_0", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb12(%169: f32, %170: f32):
        %171 = arith.constant 1.000000e+00 : f32
        %172 = arith.negf %169 : f32
        %173 = math.exp %172 : f32
        %174 = arith.addf %171, %173 : f32
        %175 = arith.divf %171, %174 : f32
        %176 = arith.mulf %169, %175 : f32
        linalg.yield %176 : f32
      } -> tensor<1x1024xf32>
      %177 = tensor.empty() : tensor<1024x1024xf32>
      %178 = linalg.transpose ins(%33:tensor<1024x1024xf32>) outs(%177:tensor<1024x1024xf32>) permutation = [1, 0]
      %179 = tensor.empty() : tensor<1x1024xf32>
      %180 = arith.constant 0.000000e+00 : f32
      %181 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%180 : f32) outs(%179 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %182 = linalg.matmul {prov.region_id = "matmul_1", prov.dispatch_id = "matmul_1", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%168, %178 : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%181 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %183 = "tensor.extract_slice"(%182) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %184 = tensor.collapse_shape %183 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %185 = tensor.expand_shape %184 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %186 = tensor.empty() : tensor<1024x6144xf32>
      %187 = linalg.transpose ins(%34:tensor<6144x1024xf32>) outs(%186:tensor<1024x6144xf32>) permutation = [1, 0]
      %188 = tensor.empty() : tensor<1x1x6144xf32>
      %189 = arith.constant 0.000000e+00 : f32
      %190 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%189 : f32) outs(%188 : tensor<1x1x6144xf32>) -> tensor<1x1x6144xf32>
      %191 = linalg.matmul {prov.region_id = "matmul_2", prov.dispatch_id = "matmul_2", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%185, %187 : tensor<1x1x1024xf32>, tensor<1024x6144xf32>) outs(%190 : tensor<1x1x6144xf32>) -> tensor<1x1x6144xf32>
      %192 = tensor.empty() : tensor<1x1x6144xf32>
      %193 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%191, %35 : tensor<1x1x6144xf32>, tensor<6144xf32>) outs(%192 : tensor<1x1x6144xf32>) attrs =  {prov.region_id = "add_0", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb13(%194: f32, %195: f32, %196: f32):
        %197 = arith.addf %194, %195 : f32
        linalg.yield %197 : f32
      } -> tensor<1x1x6144xf32>
      %198 = tensor.collapse_shape %193 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x1x6144xf32> into tensor<6144xf32>
      %199 = tensor.expand_shape %198 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1024] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<6144xf32> into tensor<1x6x1024xf32>
      %200 = tensor.empty() : tensor<1x30x32xf32>
      %201 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%112, %40 : tensor<1x30x32xf32>, tensor<1x30x32xf32>) outs(%200 : tensor<1x30x32xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb14(%202: f32, %203: f32, %204: f32):
        %205 = arith.mulf %202, %203 : f32
        linalg.yield %205 : f32
      } -> tensor<1x30x32xf32>
      %206 = tensor.empty() : tensor<32x1024xf32>
      %207 = linalg.transpose ins(%28:tensor<1024x32xf32>) outs(%206:tensor<32x1024xf32>) permutation = [1, 0]
      %208 = tensor.empty() : tensor<1x30x1024xf32>
      %209 = arith.constant 0.000000e+00 : f32
      %210 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%209 : f32) outs(%208 : tensor<1x30x1024xf32>) -> tensor<1x30x1024xf32>
      %211 = linalg.matmul {prov.region_id = "matmul_3", prov.dispatch_id = "matmul_3", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%201, %207 : tensor<1x30x32xf32>, tensor<32x1024xf32>) outs(%210 : tensor<1x30x1024xf32>) -> tensor<1x30x1024xf32>
      %212 = tensor.empty() : tensor<1x30x1024xf32>
      %213 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%211 : tensor<1x30x1024xf32>) outs(%212 : tensor<1x30x1024xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32"} {
      ^bb15(%214: f32, %215: f32):
        %216 = arith.constant 5.000000e-01 : f32
        %217 = arith.constant 1.000000e+00 : f32
        %218 = arith.constant 0.707106769 : f32
        %219 = arith.mulf %214, %218 : f32
        %220 = math.erf %219 : f32
        %221 = arith.addf %217, %220 : f32
        %222 = arith.mulf %216, %214 : f32
        %223 = arith.mulf %222, %221 : f32
        linalg.yield %223 : f32
      } -> tensor<1x30x1024xf32>
      %224 = tensor.empty() : tensor<1024x1024xf32>
      %225 = linalg.transpose ins(%29:tensor<1024x1024xf32>) outs(%224:tensor<1024x1024xf32>) permutation = [1, 0]
      %226 = tensor.empty() : tensor<1x30x1024xf32>
      %227 = arith.constant 0.000000e+00 : f32
      %228 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%227 : f32) outs(%226 : tensor<1x30x1024xf32>) -> tensor<1x30x1024xf32>
      %229 = linalg.matmul {prov.region_id = "matmul_4", prov.dispatch_id = "matmul_4", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%213, %225 : tensor<1x30x1024xf32>, tensor<1024x1024xf32>) outs(%228 : tensor<1x30x1024xf32>) -> tensor<1x30x1024xf32>
      %230 = tensor.collapse_shape %36 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %231 = tensor.expand_shape %230 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %232 = tensor.concat dim(1) %231, %72, %229 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x1x1024xf32>, tensor<1x1x1024xf32>, tensor<1x30x1024xf32>) -> tensor<1x32x1024xf32>
      %233 = tensor.collapse_shape %0 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<6x1024xf32> into tensor<6144xf32>
      %234 = tensor.expand_shape %233 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1024] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<6144xf32> into tensor<1x6x1024xf32>
      %235 = tensor.empty() : tensor<1x6x1024xf32>
      %236 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%234, %199 : tensor<1x6x1024xf32>, tensor<1x6x1024xf32>) outs(%235 : tensor<1x6x1024xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb16(%237: f32, %238: f32, %239: f32):
        %240 = arith.addf %237, %238 : f32
        linalg.yield %240 : f32
      } -> tensor<1x6x1024xf32>
      %241 = "tensor.extract_slice"(%236) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
      %242 = "tensor.extract_slice"(%236) <{static_offsets = array<i64: 0, 1, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
      %243 = "tensor.extract_slice"(%236) <{static_offsets = array<i64: 0, 2, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
      %244 = "tensor.extract_slice"(%236) <{static_offsets = array<i64: 0, 3, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
      %245 = "tensor.extract_slice"(%236) <{static_offsets = array<i64: 0, 4, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
      %246 = "tensor.extract_slice"(%236) <{static_offsets = array<i64: 0, 5, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
      %247 = tensor.empty() : tensor<1x32x1024xf32>
      %248 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%232 : tensor<1x32x1024xf32>) outs(%247 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb17(%249: f32, %250: f32):
        %251 = arith.constant 2.000000e+00 : f32
        %252 = math.powf %249, %251 : f32
        linalg.yield %252 : f32
      } -> tensor<1x32x1024xf32>
      %253 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %254 = tensor.splat %253 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32>
      %255 = linalg.reduce ins(%248:tensor<1x32x1024xf32>) outs(%254:tensor<1x32xf32>) dimensions = [2]
      (%256: f32, %257: f32) {
        %258 = arith.addf %256, %257 : f32
        linalg.yield %258 : f32
      }
      %259 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.024000e+03 : f32
      %260 = tensor.splat %259 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32>
      %261 = tensor.empty() : tensor<1x32xf32>
      %262 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%255, %260 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%261 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb18(%263: f32, %264: f32, %265: f32):
        %266 = arith.divf %263, %264 : f32
        linalg.yield %266 : f32
      } -> tensor<1x32xf32>
      %267 = tensor.collapse_shape %262 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
      %268 = tensor.expand_shape %267 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32x1xf32>
      %269 = arith.constant {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %270 = tensor.splat %269 {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x1xf32>
      %271 = tensor.empty() : tensor<1x32x1xf32>
      %272 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%268, %270 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%271 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb19(%273: f32, %274: f32, %275: f32):
        %276 = arith.addf %273, %274 : f32
        linalg.yield %276 : f32
      } -> tensor<1x32x1xf32>
      %277 = tensor.empty() : tensor<1x32x1xf32>
      %278 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%272 : tensor<1x32x1xf32>) outs(%277 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb20(%279: f32, %280: f32):
        %281 = math.rsqrt %279 : f32
        linalg.yield %281 : f32
      } -> tensor<1x32x1xf32>
      %282 = tensor.empty() : tensor<1x32x1024xf32>
      %283 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%232, %278 : tensor<1x32x1024xf32>, tensor<1x32x1xf32>) outs(%282 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb21(%284: f32, %285: f32, %286: f32):
        %287 = arith.mulf %284, %285 : f32
        linalg.yield %287 : f32
      } -> tensor<1x32x1024xf32>
      %288 = tensor.empty() : tensor<1x32x1024xf32>
      %289 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %283 : tensor<1024xf32>, tensor<1x32x1024xf32>) outs(%288 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb22(%290: f32, %291: f32, %292: f32):
        %293 = arith.mulf %290, %291 : f32
        linalg.yield %293 : f32
      } -> tensor<1x32x1024xf32>
      %294 = arith.constant {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
      %295 = tensor.splat %294 {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1024xf32>
      %296 = tensor.empty() : tensor<1x1x1024xf32>
      %297 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%242, %295 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%296 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb23(%298: f32, %299: f32, %300: f32):
        %301 = arith.addf %298, %299 : f32
        linalg.yield %301 : f32
      } -> tensor<1x1x1024xf32>
      %302 = tensor.empty() : tensor<1x32x1024xf32>
      %303 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%289, %297 : tensor<1x32x1024xf32>, tensor<1x1x1024xf32>) outs(%302 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb24(%304: f32, %305: f32, %306: f32):
        %307 = arith.mulf %304, %305 : f32
        linalg.yield %307 : f32
      } -> tensor<1x32x1024xf32>
      %308 = tensor.empty() : tensor<1x32x1024xf32>
      %309 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%303, %241 : tensor<1x32x1024xf32>, tensor<1x1x1024xf32>) outs(%308 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb25(%310: f32, %311: f32, %312: f32):
        %313 = arith.addf %310, %311 : f32
        linalg.yield %313 : f32
      } -> tensor<1x32x1024xf32>
      %314 = tensor.empty() : tensor<1024x3072xf32>
      %315 = linalg.transpose ins(%1:tensor<3072x1024xf32>) outs(%314:tensor<1024x3072xf32>) permutation = [1, 0]
      %316 = tensor.empty() : tensor<1x32x3072xf32>
      %317 = arith.constant 0.000000e+00 : f32
      %318 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%317 : f32) outs(%316 : tensor<1x32x3072xf32>) -> tensor<1x32x3072xf32>
      %319 = linalg.matmul {prov.region_id = "matmul_5", prov.dispatch_id = "matmul_5", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%309, %315 : tensor<1x32x1024xf32>, tensor<1024x3072xf32>) outs(%318 : tensor<1x32x3072xf32>) -> tensor<1x32x3072xf32>
      %320 = tensor.empty() : tensor<1x32x3072xf32>
      %321 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%319, %2 : tensor<1x32x3072xf32>, tensor<3072xf32>) outs(%320 : tensor<1x32x3072xf32>) attrs =  {prov.region_id = "add_5", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb26(%322: f32, %323: f32, %324: f32):
        %325 = arith.addf %322, %323 : f32
        linalg.yield %325 : f32
      } -> tensor<1x32x3072xf32>
      %326 = tensor.collapse_shape %321 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x3072xf32> into tensor<98304xf32>
      %327 = tensor.expand_shape %326 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 32, 3, 8, 128] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<98304xf32> into tensor<1x32x3x8x128xf32>
      %328 = "tensor.extract_slice"(%327) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 32, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<1x32x3x8x128xf32>) -> tensor<1x32x1x8x128xf32>
      %329 = tensor.collapse_shape %328 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x32x1x8x128xf32> into tensor<32768xf32>
      %330 = tensor.expand_shape %329 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 128] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<1x32x8x128xf32>
      %331 = "tensor.extract_slice"(%327) <{static_offsets = array<i64: 0, 0, 1, 0, 0>, static_sizes = array<i64: 1, 32, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<1x32x3x8x128xf32>) -> tensor<1x32x1x8x128xf32>
      %332 = tensor.collapse_shape %331 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x32x1x8x128xf32> into tensor<32768xf32>
      %333 = tensor.expand_shape %332 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 128] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<1x32x8x128xf32>
      %334 = "tensor.extract_slice"(%327) <{static_offsets = array<i64: 0, 0, 2, 0, 0>, static_sizes = array<i64: 1, 32, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<1x32x3x8x128xf32>) -> tensor<1x32x1x8x128xf32>
      %335 = tensor.collapse_shape %334 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x32x1x8x128xf32> into tensor<32768xf32>
      %336 = tensor.expand_shape %335 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 128] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<1x32x8x128xf32>
      %337 = tensor.empty() : tensor<1x32x8x128xf32>
      %338 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%330 : tensor<1x32x8x128xf32>) outs(%337 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb27(%339: f32, %340: f32):
        %341 = arith.constant 2.000000e+00 : f32
        %342 = math.powf %339, %341 : f32
        linalg.yield %342 : f32
      } -> tensor<1x32x8x128xf32>
      %343 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %344 = tensor.splat %343 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32x8xf32>
      %345 = linalg.reduce ins(%338:tensor<1x32x8x128xf32>) outs(%344:tensor<1x32x8xf32>) dimensions = [3]
      (%346: f32, %347: f32) {
        %348 = arith.addf %346, %347 : f32
        linalg.yield %348 : f32
      }
      %349 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %350 = tensor.splat %349 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32x8xf32>
      %351 = tensor.empty() : tensor<1x32x8xf32>
      %352 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%345, %350 : tensor<1x32x8xf32>, tensor<1x32x8xf32>) outs(%351 : tensor<1x32x8xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb28(%353: f32, %354: f32, %355: f32):
        %356 = arith.divf %353, %354 : f32
        linalg.yield %356 : f32
      } -> tensor<1x32x8xf32>
      %357 = tensor.collapse_shape %352 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32x8xf32> into tensor<256xf32>
      %358 = tensor.expand_shape %357 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x32x8x1xf32>
      %359 = arith.constant {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %360 = tensor.splat %359 {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x8x1xf32>
      %361 = tensor.empty() : tensor<1x32x8x1xf32>
      %362 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%358, %360 : tensor<1x32x8x1xf32>, tensor<1x32x8x1xf32>) outs(%361 : tensor<1x32x8x1xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb29(%363: f32, %364: f32, %365: f32):
        %366 = arith.addf %363, %364 : f32
        linalg.yield %366 : f32
      } -> tensor<1x32x8x1xf32>
      %367 = tensor.empty() : tensor<1x32x8x1xf32>
      %368 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%362 : tensor<1x32x8x1xf32>) outs(%367 : tensor<1x32x8x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb30(%369: f32, %370: f32):
        %371 = math.rsqrt %369 : f32
        linalg.yield %371 : f32
      } -> tensor<1x32x8x1xf32>
      %372 = tensor.empty() : tensor<1x32x8x128xf32>
      %373 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%330, %368 : tensor<1x32x8x128xf32>, tensor<1x32x8x1xf32>) outs(%372 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb31(%374: f32, %375: f32, %376: f32):
        %377 = arith.mulf %374, %375 : f32
        linalg.yield %377 : f32
      } -> tensor<1x32x8x128xf32>
      %378 = tensor.empty() : tensor<1x32x8x128xf32>
      %379 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %373 : tensor<128xf32>, tensor<1x32x8x128xf32>) outs(%378 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb32(%380: f32, %381: f32, %382: f32):
        %383 = arith.mulf %380, %381 : f32
        linalg.yield %383 : f32
      } -> tensor<1x32x8x128xf32>
      %384 = tensor.empty() : tensor<1x32x8x128xf32>
      %385 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%333 : tensor<1x32x8x128xf32>) outs(%384 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb33(%386: f32, %387: f32):
        %388 = arith.constant 2.000000e+00 : f32
        %389 = math.powf %386, %388 : f32
        linalg.yield %389 : f32
      } -> tensor<1x32x8x128xf32>
      %390 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %391 = tensor.splat %390 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32x8xf32>
      %392 = linalg.reduce ins(%385:tensor<1x32x8x128xf32>) outs(%391:tensor<1x32x8xf32>) dimensions = [3]
      (%393: f32, %394: f32) {
        %395 = arith.addf %393, %394 : f32
        linalg.yield %395 : f32
      }
      %396 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %397 = tensor.splat %396 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32x8xf32>
      %398 = tensor.empty() : tensor<1x32x8xf32>
      %399 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%392, %397 : tensor<1x32x8xf32>, tensor<1x32x8xf32>) outs(%398 : tensor<1x32x8xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb34(%400: f32, %401: f32, %402: f32):
        %403 = arith.divf %400, %401 : f32
        linalg.yield %403 : f32
      } -> tensor<1x32x8xf32>
      %404 = tensor.collapse_shape %399 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32x8xf32> into tensor<256xf32>
      %405 = tensor.expand_shape %404 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x32x8x1xf32>
      %406 = arith.constant {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %407 = tensor.splat %406 {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x8x1xf32>
      %408 = tensor.empty() : tensor<1x32x8x1xf32>
      %409 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%405, %407 : tensor<1x32x8x1xf32>, tensor<1x32x8x1xf32>) outs(%408 : tensor<1x32x8x1xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb35(%410: f32, %411: f32, %412: f32):
        %413 = arith.addf %410, %411 : f32
        linalg.yield %413 : f32
      } -> tensor<1x32x8x1xf32>
      %414 = tensor.empty() : tensor<1x32x8x1xf32>
      %415 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%409 : tensor<1x32x8x1xf32>) outs(%414 : tensor<1x32x8x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb36(%416: f32, %417: f32):
        %418 = math.rsqrt %416 : f32
        linalg.yield %418 : f32
      } -> tensor<1x32x8x1xf32>
      %419 = tensor.empty() : tensor<1x32x8x128xf32>
      %420 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%333, %415 : tensor<1x32x8x128xf32>, tensor<1x32x8x1xf32>) outs(%419 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb37(%421: f32, %422: f32, %423: f32):
        %424 = arith.mulf %421, %422 : f32
        linalg.yield %424 : f32
      } -> tensor<1x32x8x128xf32>
      %425 = tensor.empty() : tensor<1x32x8x128xf32>
      %426 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %420 : tensor<128xf32>, tensor<1x32x8x128xf32>) outs(%425 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb38(%427: f32, %428: f32, %429: f32):
        %430 = arith.mulf %427, %428 : f32
        linalg.yield %430 : f32
      } -> tensor<1x32x8x128xf32>
      %431 = tensor.empty() : tensor<1x8x32x128xf32>
      %432 = linalg.transpose ins(%379:tensor<1x32x8x128xf32>) outs(%431:tensor<1x8x32x128xf32>) permutation = [0, 2, 1, 3]
      %433 = tensor.empty() : tensor<1x8x32x128xf32>
      %434 = linalg.transpose ins(%426:tensor<1x32x8x128xf32>) outs(%433:tensor<1x8x32x128xf32>) permutation = [0, 2, 1, 3]
      %435 = tensor.empty() : tensor<1x8x32x128xf32>
      %436 = linalg.transpose ins(%336:tensor<1x32x8x128xf32>) outs(%435:tensor<1x8x32x128xf32>) permutation = [0, 2, 1, 3]
      %437 = tensor.collapse_shape %42 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32x128xf32> into tensor<4096xf32>
      %438 = tensor.expand_shape %437 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 128] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x1x32x128xf32>
      %439 = tensor.collapse_shape %43 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32x128xf32> into tensor<4096xf32>
      %440 = tensor.expand_shape %439 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 128] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x1x32x128xf32>
      %441 = tensor.empty() : tensor<1x8x32x128xf32>
      %442 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%432, %438 : tensor<1x8x32x128xf32>, tensor<1x1x32x128xf32>) outs(%441 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb39(%443: f32, %444: f32, %445: f32):
        %446 = arith.mulf %443, %444 : f32
        linalg.yield %446 : f32
      } -> tensor<1x8x32x128xf32>
      %447 = "tensor.extract_slice"(%432) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x64xf32>
      %448 = "tensor.extract_slice"(%432) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 8, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x64xf32>
      %449 = tensor.empty() : tensor<1x8x32x64xf32>
      %450 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%448 : tensor<1x8x32x64xf32>) outs(%449 : tensor<1x8x32x64xf32>) attrs =  {prov.region_id = "neg_0", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb40(%451: f32, %452: f32):
        %453 = arith.negf %451 : f32
        linalg.yield %453 : f32
      } -> tensor<1x8x32x64xf32>
      %454 = tensor.concat dim(3) %450, %447 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x8x32x64xf32>, tensor<1x8x32x64xf32>) -> tensor<1x8x32x128xf32>
      %455 = tensor.empty() : tensor<1x8x32x128xf32>
      %456 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%454, %440 : tensor<1x8x32x128xf32>, tensor<1x1x32x128xf32>) outs(%455 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb41(%457: f32, %458: f32, %459: f32):
        %460 = arith.mulf %457, %458 : f32
        linalg.yield %460 : f32
      } -> tensor<1x8x32x128xf32>
      %461 = tensor.empty() : tensor<1x8x32x128xf32>
      %462 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%442, %456 : tensor<1x8x32x128xf32>, tensor<1x8x32x128xf32>) outs(%461 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb42(%463: f32, %464: f32, %465: f32):
        %466 = arith.addf %463, %464 : f32
        linalg.yield %466 : f32
      } -> tensor<1x8x32x128xf32>
      %467 = tensor.empty() : tensor<1x8x32x128xf32>
      %468 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%434, %438 : tensor<1x8x32x128xf32>, tensor<1x1x32x128xf32>) outs(%467 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb43(%469: f32, %470: f32, %471: f32):
        %472 = arith.mulf %469, %470 : f32
        linalg.yield %472 : f32
      } -> tensor<1x8x32x128xf32>
      %473 = "tensor.extract_slice"(%434) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_4", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x64xf32>
      %474 = "tensor.extract_slice"(%434) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 8, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_5", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x64xf32>
      %475 = tensor.empty() : tensor<1x8x32x64xf32>
      %476 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%474 : tensor<1x8x32x64xf32>) outs(%475 : tensor<1x8x32x64xf32>) attrs =  {prov.region_id = "neg_1", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb44(%477: f32, %478: f32):
        %479 = arith.negf %477 : f32
        linalg.yield %479 : f32
      } -> tensor<1x8x32x64xf32>
      %480 = tensor.concat dim(3) %476, %473 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x8x32x64xf32>, tensor<1x8x32x64xf32>) -> tensor<1x8x32x128xf32>
      %481 = tensor.empty() : tensor<1x8x32x128xf32>
      %482 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%480, %440 : tensor<1x8x32x128xf32>, tensor<1x1x32x128xf32>) outs(%481 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb45(%483: f32, %484: f32, %485: f32):
        %486 = arith.mulf %483, %484 : f32
        linalg.yield %486 : f32
      } -> tensor<1x8x32x128xf32>
      %487 = tensor.empty() : tensor<1x8x32x128xf32>
      %488 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%468, %482 : tensor<1x8x32x128xf32>, tensor<1x8x32x128xf32>) outs(%487 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb46(%489: f32, %490: f32, %491: f32):
        %492 = arith.addf %489, %490 : f32
        linalg.yield %492 : f32
      } -> tensor<1x8x32x128xf32>
      %493 = tensor.concat dim(2) %45, %488 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x8x16x128xf32>, tensor<1x8x32x128xf32>) -> tensor<1x8x48x128xf32>
      %494 = tensor.concat dim(2) %46, %436 {prov.region_id = "cat_5", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x8x16x128xf32>, tensor<1x8x32x128xf32>) -> tensor<1x8x48x128xf32>
      %495 = arith.constant {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %496 = tensor.splat %495 {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x8x32x48xf32>
      %497 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%462, %493 : tensor<1x8x32x128xf32>, tensor<1x8x48x128xf32>) outs(%496 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb47(%498: f32, %499: f32, %500: f32):
        %501 = arith.mulf %498, %499 : f32
        %502 = arith.addf %500, %501 : f32
        linalg.yield %502 : f32
      } -> tensor<1x8x32x48xf32>
      %503 = tensor.empty() : tensor<1x8x32x48xf32>
      %504 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%497 : tensor<1x8x32x48xf32>) outs(%503 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb48(%505: f32, %506: f32):
        %507 = arith.constant 0.0883883461 : f32
        %508 = arith.mulf %505, %507 : f32
        linalg.yield %508 : f32
      } -> tensor<1x8x32x48xf32>
      %509 = tensor.empty() : tensor<1x8x32x48xf32>
      %510 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%504, %44 : tensor<1x8x32x48xf32>, tensor<1x1x32x48xi1>) outs(%509 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb49(%511: f32, %512: i1, %513: f32):
        %514 = arith.constant 0xff800000 : f32
        %515 = arith.select %512, %511, %514 : f32
        linalg.yield %515 : f32
      } -> tensor<1x8x32x48xf32>
      %516 = arith.constant {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %517 = tensor.splat %516 {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x8x32xf32>
      %518 = linalg.reduce ins(%510:tensor<1x8x32x48xf32>) outs(%517:tensor<1x8x32xf32>) dimensions = [3]
      (%519: f32, %520: f32) {
        %521 = arith.maximumf %519, %520 : f32
        linalg.yield %521 : f32
      }
      %522 = tensor.collapse_shape %518 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x8x32xf32> into tensor<256xf32>
      %523 = tensor.expand_shape %522 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x8x32x1xf32>
      %524 = tensor.empty() : tensor<1x8x32x48xf32>
      %525 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%510, %523 : tensor<1x8x32x48xf32>, tensor<1x8x32x1xf32>) outs(%524 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb50(%526: f32, %527: f32, %528: f32):
        %529 = arith.subf %526, %527 : f32
        linalg.yield %529 : f32
      } -> tensor<1x8x32x48xf32>
      %530 = tensor.empty() : tensor<1x8x32x48xf32>
      %531 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%525 : tensor<1x8x32x48xf32>) outs(%530 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb51(%532: f32, %533: f32):
        %534 = math.exp %532 : f32
        linalg.yield %534 : f32
      } -> tensor<1x8x32x48xf32>
      %535 = arith.constant {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %536 = tensor.splat %535 {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x8x32xf32>
      %537 = linalg.reduce ins(%531:tensor<1x8x32x48xf32>) outs(%536:tensor<1x8x32xf32>) dimensions = [3]
      (%538: f32, %539: f32) {
        %540 = arith.addf %538, %539 : f32
        linalg.yield %540 : f32
      }
      %541 = tensor.collapse_shape %537 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x8x32xf32> into tensor<256xf32>
      %542 = tensor.expand_shape %541 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x8x32x1xf32>
      %543 = tensor.empty() : tensor<1x8x32x48xf32>
      %544 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%531, %542 : tensor<1x8x32x48xf32>, tensor<1x8x32x1xf32>) outs(%543 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb52(%545: f32, %546: f32, %547: f32):
        %548 = arith.divf %545, %546 : f32
        linalg.yield %548 : f32
      } -> tensor<1x8x32x48xf32>
      %549 = arith.constant {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %550 = tensor.splat %549 {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x8x32x128xf32>
      %551 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%544, %494 : tensor<1x8x32x48xf32>, tensor<1x8x48x128xf32>) outs(%550 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb53(%552: f32, %553: f32, %554: f32):
        %555 = arith.mulf %552, %553 : f32
        %556 = arith.addf %554, %555 : f32
        linalg.yield %556 : f32
      } -> tensor<1x8x32x128xf32>
      %557 = tensor.empty() : tensor<1x32x8x128xf32>
      %558 = linalg.transpose ins(%551:tensor<1x8x32x128xf32>) outs(%557:tensor<1x32x8x128xf32>) permutation = [0, 2, 1, 3]
      %559 = tensor.collapse_shape %558 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x8x128xf32> into tensor<32768xf32>
      %560 = tensor.expand_shape %559 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1024] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<1x32x1024xf32>
      %561 = tensor.empty() : tensor<1024x1024xf32>
      %562 = linalg.transpose ins(%3:tensor<1024x1024xf32>) outs(%561:tensor<1024x1024xf32>) permutation = [1, 0]
      %563 = tensor.empty() : tensor<1x32x1024xf32>
      %564 = arith.constant 0.000000e+00 : f32
      %565 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%564 : f32) outs(%563 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
      %566 = linalg.matmul {prov.region_id = "matmul_7", prov.dispatch_id = "matmul_7", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%560, %562 : tensor<1x32x1024xf32>, tensor<1024x1024xf32>) outs(%565 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
      %567 = tensor.empty() : tensor<1x32x1024xf32>
      %568 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%243, %566 : tensor<1x1x1024xf32>, tensor<1x32x1024xf32>) outs(%567 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb54(%569: f32, %570: f32, %571: f32):
        %572 = arith.mulf %569, %570 : f32
        linalg.yield %572 : f32
      } -> tensor<1x32x1024xf32>
      %573 = tensor.empty() : tensor<1x32x1024xf32>
      %574 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%232, %568 : tensor<1x32x1024xf32>, tensor<1x32x1024xf32>) outs(%573 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb55(%575: f32, %576: f32, %577: f32):
        %578 = arith.addf %575, %576 : f32
        linalg.yield %578 : f32
      } -> tensor<1x32x1024xf32>
      %579 = tensor.empty() : tensor<1x32x1024xf32>
      %580 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%574 : tensor<1x32x1024xf32>) outs(%579 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb56(%581: f32, %582: f32):
        %583 = arith.constant 2.000000e+00 : f32
        %584 = math.powf %581, %583 : f32
        linalg.yield %584 : f32
      } -> tensor<1x32x1024xf32>
      %585 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %586 = tensor.splat %585 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32>
      %587 = linalg.reduce ins(%580:tensor<1x32x1024xf32>) outs(%586:tensor<1x32xf32>) dimensions = [2]
      (%588: f32, %589: f32) {
        %590 = arith.addf %588, %589 : f32
        linalg.yield %590 : f32
      }
      %591 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.024000e+03 : f32
      %592 = tensor.splat %591 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32>
      %593 = tensor.empty() : tensor<1x32xf32>
      %594 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%587, %592 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%593 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb57(%595: f32, %596: f32, %597: f32):
        %598 = arith.divf %595, %596 : f32
        linalg.yield %598 : f32
      } -> tensor<1x32xf32>
      %599 = tensor.collapse_shape %594 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
      %600 = tensor.expand_shape %599 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32x1xf32>
      %601 = arith.constant {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %602 = tensor.splat %601 {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x1xf32>
      %603 = tensor.empty() : tensor<1x32x1xf32>
      %604 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%600, %602 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%603 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb58(%605: f32, %606: f32, %607: f32):
        %608 = arith.addf %605, %606 : f32
        linalg.yield %608 : f32
      } -> tensor<1x32x1xf32>
      %609 = tensor.empty() : tensor<1x32x1xf32>
      %610 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%604 : tensor<1x32x1xf32>) outs(%609 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb59(%611: f32, %612: f32):
        %613 = math.rsqrt %611 : f32
        linalg.yield %613 : f32
      } -> tensor<1x32x1xf32>
      %614 = tensor.empty() : tensor<1x32x1024xf32>
      %615 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%574, %610 : tensor<1x32x1024xf32>, tensor<1x32x1xf32>) outs(%614 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb60(%616: f32, %617: f32, %618: f32):
        %619 = arith.mulf %616, %617 : f32
        linalg.yield %619 : f32
      } -> tensor<1x32x1024xf32>
      %620 = tensor.empty() : tensor<1x32x1024xf32>
      %621 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %615 : tensor<1024xf32>, tensor<1x32x1024xf32>) outs(%620 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb61(%622: f32, %623: f32, %624: f32):
        %625 = arith.mulf %622, %623 : f32
        linalg.yield %625 : f32
      } -> tensor<1x32x1024xf32>
      %626 = tensor.empty() : tensor<1x32x1024xf32>
      %627 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%621 : tensor<1x32x1024xf32>) outs(%626 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb62(%628: f32, %629: f32):
        %630 = arith.constant 2.000000e+00 : f32
        %631 = math.powf %628, %630 : f32
        linalg.yield %631 : f32
      } -> tensor<1x32x1024xf32>
      %632 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %633 = tensor.splat %632 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32>
      %634 = linalg.reduce ins(%627:tensor<1x32x1024xf32>) outs(%633:tensor<1x32xf32>) dimensions = [2]
      (%635: f32, %636: f32) {
        %637 = arith.addf %635, %636 : f32
        linalg.yield %637 : f32
      }
      %638 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.024000e+03 : f32
      %639 = tensor.splat %638 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32>
      %640 = tensor.empty() : tensor<1x32xf32>
      %641 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%634, %639 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%640 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb63(%642: f32, %643: f32, %644: f32):
        %645 = arith.divf %642, %643 : f32
        linalg.yield %645 : f32
      } -> tensor<1x32xf32>
      %646 = tensor.collapse_shape %641 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
      %647 = tensor.expand_shape %646 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32x1xf32>
      %648 = arith.constant {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %649 = tensor.splat %648 {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x1xf32>
      %650 = tensor.empty() : tensor<1x32x1xf32>
      %651 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%647, %649 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%650 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb64(%652: f32, %653: f32, %654: f32):
        %655 = arith.addf %652, %653 : f32
        linalg.yield %655 : f32
      } -> tensor<1x32x1xf32>
      %656 = tensor.empty() : tensor<1x32x1xf32>
      %657 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%651 : tensor<1x32x1xf32>) outs(%656 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb65(%658: f32, %659: f32):
        %660 = math.rsqrt %658 : f32
        linalg.yield %660 : f32
      } -> tensor<1x32x1xf32>
      %661 = tensor.empty() : tensor<1x32x1024xf32>
      %662 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%621, %657 : tensor<1x32x1024xf32>, tensor<1x32x1xf32>) outs(%661 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb66(%663: f32, %664: f32, %665: f32):
        %666 = arith.mulf %663, %664 : f32
        linalg.yield %666 : f32
      } -> tensor<1x32x1024xf32>
      %667 = tensor.empty() : tensor<1x32x1024xf32>
      %668 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11, %662 : tensor<1024xf32>, tensor<1x32x1024xf32>) outs(%667 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb67(%669: f32, %670: f32, %671: f32):
        %672 = arith.mulf %669, %670 : f32
        linalg.yield %672 : f32
      } -> tensor<1x32x1024xf32>
      %673 = arith.constant {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
      %674 = tensor.splat %673 {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1024xf32>
      %675 = tensor.empty() : tensor<1x1x1024xf32>
      %676 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%245, %674 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%675 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb68(%677: f32, %678: f32, %679: f32):
        %680 = arith.addf %677, %678 : f32
        linalg.yield %680 : f32
      } -> tensor<1x1x1024xf32>
      %681 = tensor.empty() : tensor<1x32x1024xf32>
      %682 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%668, %676 : tensor<1x32x1024xf32>, tensor<1x1x1024xf32>) outs(%681 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb69(%683: f32, %684: f32, %685: f32):
        %686 = arith.mulf %683, %684 : f32
        linalg.yield %686 : f32
      } -> tensor<1x32x1024xf32>
      %687 = tensor.empty() : tensor<1x32x1024xf32>
      %688 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%682, %244 : tensor<1x32x1024xf32>, tensor<1x1x1024xf32>) outs(%687 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb70(%689: f32, %690: f32, %691: f32):
        %692 = arith.addf %689, %690 : f32
        linalg.yield %692 : f32
      } -> tensor<1x32x1024xf32>
      %693 = tensor.empty() : tensor<1024x4096xf32>
      %694 = linalg.transpose ins(%6:tensor<4096x1024xf32>) outs(%693:tensor<1024x4096xf32>) permutation = [1, 0]
      %695 = tensor.empty() : tensor<1x32x4096xf32>
      %696 = arith.constant 0.000000e+00 : f32
      %697 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%696 : f32) outs(%695 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
      %698 = linalg.matmul {prov.region_id = "matmul_8", prov.dispatch_id = "matmul_8", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%688, %694 : tensor<1x32x1024xf32>, tensor<1024x4096xf32>) outs(%697 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
      %699 = tensor.empty() : tensor<1x32x4096xf32>
      %700 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%698 : tensor<1x32x4096xf32>) outs(%699 : tensor<1x32x4096xf32>) attrs =  {prov.region_id = "silu_1", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb71(%701: f32, %702: f32):
        %703 = arith.constant 1.000000e+00 : f32
        %704 = arith.negf %701 : f32
        %705 = math.exp %704 : f32
        %706 = arith.addf %703, %705 : f32
        %707 = arith.divf %703, %706 : f32
        %708 = arith.mulf %701, %707 : f32
        linalg.yield %708 : f32
      } -> tensor<1x32x4096xf32>
      %709 = tensor.empty() : tensor<1024x4096xf32>
      %710 = linalg.transpose ins(%7:tensor<4096x1024xf32>) outs(%709:tensor<1024x4096xf32>) permutation = [1, 0]
      %711 = tensor.empty() : tensor<1x32x4096xf32>
      %712 = arith.constant 0.000000e+00 : f32
      %713 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%712 : f32) outs(%711 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
      %714 = linalg.matmul {prov.region_id = "matmul_9", prov.dispatch_id = "matmul_9", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%688, %710 : tensor<1x32x1024xf32>, tensor<1024x4096xf32>) outs(%713 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
      %715 = tensor.empty() : tensor<1x32x4096xf32>
      %716 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%700, %714 : tensor<1x32x4096xf32>, tensor<1x32x4096xf32>) outs(%715 : tensor<1x32x4096xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb72(%717: f32, %718: f32, %719: f32):
        %720 = arith.mulf %717, %718 : f32
        linalg.yield %720 : f32
      } -> tensor<1x32x4096xf32>
      %721 = tensor.empty() : tensor<4096x1024xf32>
      %722 = linalg.transpose ins(%8:tensor<1024x4096xf32>) outs(%721:tensor<4096x1024xf32>) permutation = [1, 0]
      %723 = tensor.empty() : tensor<1x32x1024xf32>
      %724 = arith.constant 0.000000e+00 : f32
      %725 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%724 : f32) outs(%723 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
      %726 = linalg.matmul {prov.region_id = "matmul_10", prov.dispatch_id = "matmul_10", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%716, %722 : tensor<1x32x4096xf32>, tensor<4096x1024xf32>) outs(%725 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
      %727 = tensor.empty() : tensor<1x32x1024xf32>
      %728 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%246, %726 : tensor<1x1x1024xf32>, tensor<1x32x1024xf32>) outs(%727 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb73(%729: f32, %730: f32, %731: f32):
        %732 = arith.mulf %729, %730 : f32
        linalg.yield %732 : f32
      } -> tensor<1x32x1024xf32>
      %733 = tensor.empty() : tensor<1x32x1024xf32>
      %734 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%621, %728 : tensor<1x32x1024xf32>, tensor<1x32x1024xf32>) outs(%733 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb74(%735: f32, %736: f32, %737: f32):
        %738 = arith.addf %735, %736 : f32
        linalg.yield %738 : f32
      } -> tensor<1x32x1024xf32>
      %739 = tensor.empty() : tensor<1x32x1024xf32>
      %740 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%734 : tensor<1x32x1024xf32>) outs(%739 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb75(%741: f32, %742: f32):
        %743 = arith.constant 2.000000e+00 : f32
        %744 = math.powf %741, %743 : f32
        linalg.yield %744 : f32
      } -> tensor<1x32x1024xf32>
      %745 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %746 = tensor.splat %745 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32>
      %747 = linalg.reduce ins(%740:tensor<1x32x1024xf32>) outs(%746:tensor<1x32xf32>) dimensions = [2]
      (%748: f32, %749: f32) {
        %750 = arith.addf %748, %749 : f32
        linalg.yield %750 : f32
      }
      %751 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.024000e+03 : f32
      %752 = tensor.splat %751 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32>
      %753 = tensor.empty() : tensor<1x32xf32>
      %754 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%747, %752 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%753 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb76(%755: f32, %756: f32, %757: f32):
        %758 = arith.divf %755, %756 : f32
        linalg.yield %758 : f32
      } -> tensor<1x32xf32>
      %759 = tensor.collapse_shape %754 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
      %760 = tensor.expand_shape %759 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32x1xf32>
      %761 = arith.constant {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %762 = tensor.splat %761 {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x1xf32>
      %763 = tensor.empty() : tensor<1x32x1xf32>
      %764 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%760, %762 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%763 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb77(%765: f32, %766: f32, %767: f32):
        %768 = arith.addf %765, %766 : f32
        linalg.yield %768 : f32
      } -> tensor<1x32x1xf32>
      %769 = tensor.empty() : tensor<1x32x1xf32>
      %770 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%764 : tensor<1x32x1xf32>) outs(%769 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_5", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb78(%771: f32, %772: f32):
        %773 = math.rsqrt %771 : f32
        linalg.yield %773 : f32
      } -> tensor<1x32x1xf32>
      %774 = tensor.empty() : tensor<1x32x1024xf32>
      %775 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%734, %770 : tensor<1x32x1024xf32>, tensor<1x32x1xf32>) outs(%774 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb79(%776: f32, %777: f32, %778: f32):
        %779 = arith.mulf %776, %777 : f32
        linalg.yield %779 : f32
      } -> tensor<1x32x1024xf32>
      %780 = tensor.empty() : tensor<1x32x1024xf32>
      %781 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12, %775 : tensor<1024xf32>, tensor<1x32x1024xf32>) outs(%780 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb80(%782: f32, %783: f32, %784: f32):
        %785 = arith.mulf %782, %783 : f32
        linalg.yield %785 : f32
      } -> tensor<1x32x1024xf32>
      %786 = tensor.collapse_shape %13 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<6x1024xf32> into tensor<6144xf32>
      %787 = tensor.expand_shape %786 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 6, 1024] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<6144xf32> into tensor<1x6x1024xf32>
      %788 = tensor.empty() : tensor<1x6x1024xf32>
      %789 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%787, %199 : tensor<1x6x1024xf32>, tensor<1x6x1024xf32>) outs(%788 : tensor<1x6x1024xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb81(%790: f32, %791: f32, %792: f32):
        %793 = arith.addf %790, %791 : f32
        linalg.yield %793 : f32
      } -> tensor<1x6x1024xf32>
      %794 = "tensor.extract_slice"(%789) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
      %795 = "tensor.extract_slice"(%789) <{static_offsets = array<i64: 0, 1, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
      %796 = "tensor.extract_slice"(%789) <{static_offsets = array<i64: 0, 2, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
      %797 = "tensor.extract_slice"(%789) <{static_offsets = array<i64: 0, 3, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
      %798 = "tensor.extract_slice"(%789) <{static_offsets = array<i64: 0, 4, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
      %799 = "tensor.extract_slice"(%789) <{static_offsets = array<i64: 0, 5, 0>, static_sizes = array<i64: 1, 1, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x6x1024xf32>) -> tensor<1x1x1024xf32>
      %800 = tensor.empty() : tensor<1x32x1024xf32>
      %801 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%781 : tensor<1x32x1024xf32>) outs(%800 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb82(%802: f32, %803: f32):
        %804 = arith.constant 2.000000e+00 : f32
        %805 = math.powf %802, %804 : f32
        linalg.yield %805 : f32
      } -> tensor<1x32x1024xf32>
      %806 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %807 = tensor.splat %806 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32>
      %808 = linalg.reduce ins(%801:tensor<1x32x1024xf32>) outs(%807:tensor<1x32xf32>) dimensions = [2]
      (%809: f32, %810: f32) {
        %811 = arith.addf %809, %810 : f32
        linalg.yield %811 : f32
      }
      %812 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.024000e+03 : f32
      %813 = tensor.splat %812 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32>
      %814 = tensor.empty() : tensor<1x32xf32>
      %815 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%808, %813 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%814 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb83(%816: f32, %817: f32, %818: f32):
        %819 = arith.divf %816, %817 : f32
        linalg.yield %819 : f32
      } -> tensor<1x32xf32>
      %820 = tensor.collapse_shape %815 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
      %821 = tensor.expand_shape %820 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32x1xf32>
      %822 = arith.constant {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %823 = tensor.splat %822 {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x1xf32>
      %824 = tensor.empty() : tensor<1x32x1xf32>
      %825 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%821, %823 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%824 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb84(%826: f32, %827: f32, %828: f32):
        %829 = arith.addf %826, %827 : f32
        linalg.yield %829 : f32
      } -> tensor<1x32x1xf32>
      %830 = tensor.empty() : tensor<1x32x1xf32>
      %831 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%825 : tensor<1x32x1xf32>) outs(%830 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_6", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb85(%832: f32, %833: f32):
        %834 = math.rsqrt %832 : f32
        linalg.yield %834 : f32
      } -> tensor<1x32x1xf32>
      %835 = tensor.empty() : tensor<1x32x1024xf32>
      %836 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%781, %831 : tensor<1x32x1024xf32>, tensor<1x32x1xf32>) outs(%835 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb86(%837: f32, %838: f32, %839: f32):
        %840 = arith.mulf %837, %838 : f32
        linalg.yield %840 : f32
      } -> tensor<1x32x1024xf32>
      %841 = tensor.empty() : tensor<1x32x1024xf32>
      %842 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22, %836 : tensor<1024xf32>, tensor<1x32x1024xf32>) outs(%841 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb87(%843: f32, %844: f32, %845: f32):
        %846 = arith.mulf %843, %844 : f32
        linalg.yield %846 : f32
      } -> tensor<1x32x1024xf32>
      %847 = arith.constant {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
      %848 = tensor.splat %847 {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1024xf32>
      %849 = tensor.empty() : tensor<1x1x1024xf32>
      %850 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%795, %848 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%849 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb88(%851: f32, %852: f32, %853: f32):
        %854 = arith.addf %851, %852 : f32
        linalg.yield %854 : f32
      } -> tensor<1x1x1024xf32>
      %855 = tensor.empty() : tensor<1x32x1024xf32>
      %856 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%842, %850 : tensor<1x32x1024xf32>, tensor<1x1x1024xf32>) outs(%855 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb89(%857: f32, %858: f32, %859: f32):
        %860 = arith.mulf %857, %858 : f32
        linalg.yield %860 : f32
      } -> tensor<1x32x1024xf32>
      %861 = tensor.empty() : tensor<1x32x1024xf32>
      %862 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%856, %794 : tensor<1x32x1024xf32>, tensor<1x1x1024xf32>) outs(%861 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb90(%863: f32, %864: f32, %865: f32):
        %866 = arith.addf %863, %864 : f32
        linalg.yield %866 : f32
      } -> tensor<1x32x1024xf32>
      %867 = tensor.empty() : tensor<1024x3072xf32>
      %868 = linalg.transpose ins(%14:tensor<3072x1024xf32>) outs(%867:tensor<1024x3072xf32>) permutation = [1, 0]
      %869 = tensor.empty() : tensor<1x32x3072xf32>
      %870 = arith.constant 0.000000e+00 : f32
      %871 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%870 : f32) outs(%869 : tensor<1x32x3072xf32>) -> tensor<1x32x3072xf32>
      %872 = linalg.matmul {prov.region_id = "matmul_11", prov.dispatch_id = "matmul_11", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%862, %868 : tensor<1x32x1024xf32>, tensor<1024x3072xf32>) outs(%871 : tensor<1x32x3072xf32>) -> tensor<1x32x3072xf32>
      %873 = tensor.empty() : tensor<1x32x3072xf32>
      %874 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%872, %15 : tensor<1x32x3072xf32>, tensor<3072xf32>) outs(%873 : tensor<1x32x3072xf32>) attrs =  {prov.region_id = "add_21", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb91(%875: f32, %876: f32, %877: f32):
        %878 = arith.addf %875, %876 : f32
        linalg.yield %878 : f32
      } -> tensor<1x32x3072xf32>
      %879 = tensor.collapse_shape %874 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x3072xf32> into tensor<98304xf32>
      %880 = tensor.expand_shape %879 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 32, 3, 8, 128] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<98304xf32> into tensor<1x32x3x8x128xf32>
      %881 = "tensor.extract_slice"(%880) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 32, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<1x32x3x8x128xf32>) -> tensor<1x32x1x8x128xf32>
      %882 = tensor.collapse_shape %881 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x32x1x8x128xf32> into tensor<32768xf32>
      %883 = tensor.expand_shape %882 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 128] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<1x32x8x128xf32>
      %884 = "tensor.extract_slice"(%880) <{static_offsets = array<i64: 0, 0, 1, 0, 0>, static_sizes = array<i64: 1, 32, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<1x32x3x8x128xf32>) -> tensor<1x32x1x8x128xf32>
      %885 = tensor.collapse_shape %884 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x32x1x8x128xf32> into tensor<32768xf32>
      %886 = tensor.expand_shape %885 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 128] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<1x32x8x128xf32>
      %887 = "tensor.extract_slice"(%880) <{static_offsets = array<i64: 0, 0, 2, 0, 0>, static_sizes = array<i64: 1, 32, 1, 8, 128>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<1x32x3x8x128xf32>) -> tensor<1x32x1x8x128xf32>
      %888 = tensor.collapse_shape %887 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x32x1x8x128xf32> into tensor<32768xf32>
      %889 = tensor.expand_shape %888 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 128] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<1x32x8x128xf32>
      %890 = tensor.empty() : tensor<1x32x8x128xf32>
      %891 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%883 : tensor<1x32x8x128xf32>) outs(%890 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb92(%892: f32, %893: f32):
        %894 = arith.constant 2.000000e+00 : f32
        %895 = math.powf %892, %894 : f32
        linalg.yield %895 : f32
      } -> tensor<1x32x8x128xf32>
      %896 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %897 = tensor.splat %896 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32x8xf32>
      %898 = linalg.reduce ins(%891:tensor<1x32x8x128xf32>) outs(%897:tensor<1x32x8xf32>) dimensions = [3]
      (%899: f32, %900: f32) {
        %901 = arith.addf %899, %900 : f32
        linalg.yield %901 : f32
      }
      %902 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %903 = tensor.splat %902 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32x8xf32>
      %904 = tensor.empty() : tensor<1x32x8xf32>
      %905 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%898, %903 : tensor<1x32x8xf32>, tensor<1x32x8xf32>) outs(%904 : tensor<1x32x8xf32>) attrs =  {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb93(%906: f32, %907: f32, %908: f32):
        %909 = arith.divf %906, %907 : f32
        linalg.yield %909 : f32
      } -> tensor<1x32x8xf32>
      %910 = tensor.collapse_shape %905 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32x8xf32> into tensor<256xf32>
      %911 = tensor.expand_shape %910 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 1] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x32x8x1xf32>
      %912 = arith.constant {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %913 = tensor.splat %912 {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x8x1xf32>
      %914 = tensor.empty() : tensor<1x32x8x1xf32>
      %915 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%911, %913 : tensor<1x32x8x1xf32>, tensor<1x32x8x1xf32>) outs(%914 : tensor<1x32x8x1xf32>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb94(%916: f32, %917: f32, %918: f32):
        %919 = arith.addf %916, %917 : f32
        linalg.yield %919 : f32
      } -> tensor<1x32x8x1xf32>
      %920 = tensor.empty() : tensor<1x32x8x1xf32>
      %921 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%915 : tensor<1x32x8x1xf32>) outs(%920 : tensor<1x32x8x1xf32>) attrs =  {prov.region_id = "rsqrt_7", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb95(%922: f32, %923: f32):
        %924 = math.rsqrt %922 : f32
        linalg.yield %924 : f32
      } -> tensor<1x32x8x1xf32>
      %925 = tensor.empty() : tensor<1x32x8x128xf32>
      %926 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%883, %921 : tensor<1x32x8x128xf32>, tensor<1x32x8x1xf32>) outs(%925 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb96(%927: f32, %928: f32, %929: f32):
        %930 = arith.mulf %927, %928 : f32
        linalg.yield %930 : f32
      } -> tensor<1x32x8x128xf32>
      %931 = tensor.empty() : tensor<1x32x8x128xf32>
      %932 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%17, %926 : tensor<128xf32>, tensor<1x32x8x128xf32>) outs(%931 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb97(%933: f32, %934: f32, %935: f32):
        %936 = arith.mulf %933, %934 : f32
        linalg.yield %936 : f32
      } -> tensor<1x32x8x128xf32>
      %937 = tensor.empty() : tensor<1x32x8x128xf32>
      %938 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%886 : tensor<1x32x8x128xf32>) outs(%937 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb98(%939: f32, %940: f32):
        %941 = arith.constant 2.000000e+00 : f32
        %942 = math.powf %939, %941 : f32
        linalg.yield %942 : f32
      } -> tensor<1x32x8x128xf32>
      %943 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %944 = tensor.splat %943 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32x8xf32>
      %945 = linalg.reduce ins(%938:tensor<1x32x8x128xf32>) outs(%944:tensor<1x32x8xf32>) dimensions = [3]
      (%946: f32, %947: f32) {
        %948 = arith.addf %946, %947 : f32
        linalg.yield %948 : f32
      }
      %949 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %950 = tensor.splat %949 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32x8xf32>
      %951 = tensor.empty() : tensor<1x32x8xf32>
      %952 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%945, %950 : tensor<1x32x8xf32>, tensor<1x32x8xf32>) outs(%951 : tensor<1x32x8xf32>) attrs =  {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb99(%953: f32, %954: f32, %955: f32):
        %956 = arith.divf %953, %954 : f32
        linalg.yield %956 : f32
      } -> tensor<1x32x8xf32>
      %957 = tensor.collapse_shape %952 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32x8xf32> into tensor<256xf32>
      %958 = tensor.expand_shape %957 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 8, 1] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x32x8x1xf32>
      %959 = arith.constant {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %960 = tensor.splat %959 {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x8x1xf32>
      %961 = tensor.empty() : tensor<1x32x8x1xf32>
      %962 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%958, %960 : tensor<1x32x8x1xf32>, tensor<1x32x8x1xf32>) outs(%961 : tensor<1x32x8x1xf32>) attrs =  {prov.region_id = "add_23", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb100(%963: f32, %964: f32, %965: f32):
        %966 = arith.addf %963, %964 : f32
        linalg.yield %966 : f32
      } -> tensor<1x32x8x1xf32>
      %967 = tensor.empty() : tensor<1x32x8x1xf32>
      %968 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%962 : tensor<1x32x8x1xf32>) outs(%967 : tensor<1x32x8x1xf32>) attrs =  {prov.region_id = "rsqrt_8", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb101(%969: f32, %970: f32):
        %971 = math.rsqrt %969 : f32
        linalg.yield %971 : f32
      } -> tensor<1x32x8x1xf32>
      %972 = tensor.empty() : tensor<1x32x8x128xf32>
      %973 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%886, %968 : tensor<1x32x8x128xf32>, tensor<1x32x8x1xf32>) outs(%972 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb102(%974: f32, %975: f32, %976: f32):
        %977 = arith.mulf %974, %975 : f32
        linalg.yield %977 : f32
      } -> tensor<1x32x8x128xf32>
      %978 = tensor.empty() : tensor<1x32x8x128xf32>
      %979 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%18, %973 : tensor<128xf32>, tensor<1x32x8x128xf32>) outs(%978 : tensor<1x32x8x128xf32>) attrs =  {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb103(%980: f32, %981: f32, %982: f32):
        %983 = arith.mulf %980, %981 : f32
        linalg.yield %983 : f32
      } -> tensor<1x32x8x128xf32>
      %984 = tensor.empty() : tensor<1x8x32x128xf32>
      %985 = linalg.transpose ins(%932:tensor<1x32x8x128xf32>) outs(%984:tensor<1x8x32x128xf32>) permutation = [0, 2, 1, 3]
      %986 = tensor.empty() : tensor<1x8x32x128xf32>
      %987 = linalg.transpose ins(%979:tensor<1x32x8x128xf32>) outs(%986:tensor<1x8x32x128xf32>) permutation = [0, 2, 1, 3]
      %988 = tensor.empty() : tensor<1x8x32x128xf32>
      %989 = linalg.transpose ins(%889:tensor<1x32x8x128xf32>) outs(%988:tensor<1x8x32x128xf32>) permutation = [0, 2, 1, 3]
      %990 = tensor.collapse_shape %42 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32x128xf32> into tensor<4096xf32>
      %991 = tensor.expand_shape %990 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 128] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x1x32x128xf32>
      %992 = tensor.collapse_shape %43 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x32x128xf32> into tensor<4096xf32>
      %993 = tensor.expand_shape %992 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 32, 128] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<4096xf32> into tensor<1x1x32x128xf32>
      %994 = tensor.empty() : tensor<1x8x32x128xf32>
      %995 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%985, %991 : tensor<1x8x32x128xf32>, tensor<1x1x32x128xf32>) outs(%994 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb104(%996: f32, %997: f32, %998: f32):
        %999 = arith.mulf %996, %997 : f32
        linalg.yield %999 : f32
      } -> tensor<1x8x32x128xf32>
      %1000 = "tensor.extract_slice"(%985) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_6", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x64xf32>
      %1001 = "tensor.extract_slice"(%985) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 8, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_7", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x64xf32>
      %1002 = tensor.empty() : tensor<1x8x32x64xf32>
      %1003 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1001 : tensor<1x8x32x64xf32>) outs(%1002 : tensor<1x8x32x64xf32>) attrs =  {prov.region_id = "neg_2", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb105(%1004: f32, %1005: f32):
        %1006 = arith.negf %1004 : f32
        linalg.yield %1006 : f32
      } -> tensor<1x8x32x64xf32>
      %1007 = tensor.concat dim(3) %1003, %1000 {prov.region_id = "cat_6", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x8x32x64xf32>, tensor<1x8x32x64xf32>) -> tensor<1x8x32x128xf32>
      %1008 = tensor.empty() : tensor<1x8x32x128xf32>
      %1009 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1007, %993 : tensor<1x8x32x128xf32>, tensor<1x1x32x128xf32>) outs(%1008 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb106(%1010: f32, %1011: f32, %1012: f32):
        %1013 = arith.mulf %1010, %1011 : f32
        linalg.yield %1013 : f32
      } -> tensor<1x8x32x128xf32>
      %1014 = tensor.empty() : tensor<1x8x32x128xf32>
      %1015 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%995, %1009 : tensor<1x8x32x128xf32>, tensor<1x8x32x128xf32>) outs(%1014 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb107(%1016: f32, %1017: f32, %1018: f32):
        %1019 = arith.addf %1016, %1017 : f32
        linalg.yield %1019 : f32
      } -> tensor<1x8x32x128xf32>
      %1020 = tensor.empty() : tensor<1x8x32x128xf32>
      %1021 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%987, %991 : tensor<1x8x32x128xf32>, tensor<1x1x32x128xf32>) outs(%1020 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "mul_34", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb108(%1022: f32, %1023: f32, %1024: f32):
        %1025 = arith.mulf %1022, %1023 : f32
        linalg.yield %1025 : f32
      } -> tensor<1x8x32x128xf32>
      %1026 = "tensor.extract_slice"(%987) <{static_offsets = array<i64: 0, 0, 0, 0>, static_sizes = array<i64: 1, 8, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_8", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x64xf32>
      %1027 = "tensor.extract_slice"(%987) <{static_offsets = array<i64: 0, 0, 0, 64>, static_sizes = array<i64: 1, 8, 32, 64>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_9", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x8x32x128xf32>) -> tensor<1x8x32x64xf32>
      %1028 = tensor.empty() : tensor<1x8x32x64xf32>
      %1029 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1027 : tensor<1x8x32x64xf32>) outs(%1028 : tensor<1x8x32x64xf32>) attrs =  {prov.region_id = "neg_3", prov._pattern_hint = "neg", prov.op = "neg", prov.family = "elementwise", prov.aten = "aten.neg.default", prov.orig_dtype = "float32"} {
      ^bb109(%1030: f32, %1031: f32):
        %1032 = arith.negf %1030 : f32
        linalg.yield %1032 : f32
      } -> tensor<1x8x32x64xf32>
      %1033 = tensor.concat dim(3) %1029, %1026 {prov.region_id = "cat_7", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x8x32x64xf32>, tensor<1x8x32x64xf32>) -> tensor<1x8x32x128xf32>
      %1034 = tensor.empty() : tensor<1x8x32x128xf32>
      %1035 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1033, %993 : tensor<1x8x32x128xf32>, tensor<1x1x32x128xf32>) outs(%1034 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "mul_35", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb110(%1036: f32, %1037: f32, %1038: f32):
        %1039 = arith.mulf %1036, %1037 : f32
        linalg.yield %1039 : f32
      } -> tensor<1x8x32x128xf32>
      %1040 = tensor.empty() : tensor<1x8x32x128xf32>
      %1041 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1021, %1035 : tensor<1x8x32x128xf32>, tensor<1x8x32x128xf32>) outs(%1040 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb111(%1042: f32, %1043: f32, %1044: f32):
        %1045 = arith.addf %1042, %1043 : f32
        linalg.yield %1045 : f32
      } -> tensor<1x8x32x128xf32>
      %1046 = tensor.concat dim(2) %47, %1041 {prov.region_id = "cat_8", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x8x16x128xf32>, tensor<1x8x32x128xf32>) -> tensor<1x8x48x128xf32>
      %1047 = tensor.concat dim(2) %48, %989 {prov.region_id = "cat_9", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x8x16x128xf32>, tensor<1x8x32x128xf32>) -> tensor<1x8x48x128xf32>
      %1048 = arith.constant {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1049 = tensor.splat %1048 {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x8x32x48xf32>
      %1050 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1015, %1046 : tensor<1x8x32x128xf32>, tensor<1x8x48x128xf32>) outs(%1049 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb112(%1051: f32, %1052: f32, %1053: f32):
        %1054 = arith.mulf %1051, %1052 : f32
        %1055 = arith.addf %1053, %1054 : f32
        linalg.yield %1055 : f32
      } -> tensor<1x8x32x48xf32>
      %1056 = tensor.empty() : tensor<1x8x32x48xf32>
      %1057 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1050 : tensor<1x8x32x48xf32>) outs(%1056 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb113(%1058: f32, %1059: f32):
        %1060 = arith.constant 0.0883883461 : f32
        %1061 = arith.mulf %1058, %1060 : f32
        linalg.yield %1061 : f32
      } -> tensor<1x8x32x48xf32>
      %1062 = tensor.empty() : tensor<1x8x32x48xf32>
      %1063 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1057, %44 : tensor<1x8x32x48xf32>, tensor<1x1x32x48xi1>) outs(%1062 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb114(%1064: f32, %1065: i1, %1066: f32):
        %1067 = arith.constant 0xff800000 : f32
        %1068 = arith.select %1065, %1064, %1067 : f32
        linalg.yield %1068 : f32
      } -> tensor<1x8x32x48xf32>
      %1069 = arith.constant {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %1070 = tensor.splat %1069 {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x8x32xf32>
      %1071 = linalg.reduce ins(%1063:tensor<1x8x32x48xf32>) outs(%1070:tensor<1x8x32xf32>) dimensions = [3]
      (%1072: f32, %1073: f32) {
        %1074 = arith.maximumf %1072, %1073 : f32
        linalg.yield %1074 : f32
      }
      %1075 = tensor.collapse_shape %1071 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x8x32xf32> into tensor<256xf32>
      %1076 = tensor.expand_shape %1075 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x8x32x1xf32>
      %1077 = tensor.empty() : tensor<1x8x32x48xf32>
      %1078 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1063, %1076 : tensor<1x8x32x48xf32>, tensor<1x8x32x1xf32>) outs(%1077 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb115(%1079: f32, %1080: f32, %1081: f32):
        %1082 = arith.subf %1079, %1080 : f32
        linalg.yield %1082 : f32
      } -> tensor<1x8x32x48xf32>
      %1083 = tensor.empty() : tensor<1x8x32x48xf32>
      %1084 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1078 : tensor<1x8x32x48xf32>) outs(%1083 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb116(%1085: f32, %1086: f32):
        %1087 = math.exp %1085 : f32
        linalg.yield %1087 : f32
      } -> tensor<1x8x32x48xf32>
      %1088 = arith.constant {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1089 = tensor.splat %1088 {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x8x32xf32>
      %1090 = linalg.reduce ins(%1084:tensor<1x8x32x48xf32>) outs(%1089:tensor<1x8x32xf32>) dimensions = [3]
      (%1091: f32, %1092: f32) {
        %1093 = arith.addf %1091, %1092 : f32
        linalg.yield %1093 : f32
      }
      %1094 = tensor.collapse_shape %1090 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x8x32xf32> into tensor<256xf32>
      %1095 = tensor.expand_shape %1094 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 32, 1] {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<256xf32> into tensor<1x8x32x1xf32>
      %1096 = tensor.empty() : tensor<1x8x32x48xf32>
      %1097 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1084, %1095 : tensor<1x8x32x48xf32>, tensor<1x8x32x1xf32>) outs(%1096 : tensor<1x8x32x48xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb117(%1098: f32, %1099: f32, %1100: f32):
        %1101 = arith.divf %1098, %1099 : f32
        linalg.yield %1101 : f32
      } -> tensor<1x8x32x48xf32>
      %1102 = arith.constant {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1103 = tensor.splat %1102 {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x8x32x128xf32>
      %1104 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1097, %1047 : tensor<1x8x32x48xf32>, tensor<1x8x48x128xf32>) outs(%1103 : tensor<1x8x32x128xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb118(%1105: f32, %1106: f32, %1107: f32):
        %1108 = arith.mulf %1105, %1106 : f32
        %1109 = arith.addf %1107, %1108 : f32
        linalg.yield %1109 : f32
      } -> tensor<1x8x32x128xf32>
      %1110 = tensor.empty() : tensor<1x32x8x128xf32>
      %1111 = linalg.transpose ins(%1104:tensor<1x8x32x128xf32>) outs(%1110:tensor<1x32x8x128xf32>) permutation = [0, 2, 1, 3]
      %1112 = tensor.collapse_shape %1111 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x32x8x128xf32> into tensor<32768xf32>
      %1113 = tensor.expand_shape %1112 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1024] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<1x32x1024xf32>
      %1114 = tensor.empty() : tensor<1024x1024xf32>
      %1115 = linalg.transpose ins(%16:tensor<1024x1024xf32>) outs(%1114:tensor<1024x1024xf32>) permutation = [1, 0]
      %1116 = tensor.empty() : tensor<1x32x1024xf32>
      %1117 = arith.constant 0.000000e+00 : f32
      %1118 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1117 : f32) outs(%1116 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
      %1119 = linalg.matmul {prov.region_id = "matmul_13", prov.dispatch_id = "matmul_13", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1113, %1115 : tensor<1x32x1024xf32>, tensor<1024x1024xf32>) outs(%1118 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
      %1120 = tensor.empty() : tensor<1x32x1024xf32>
      %1121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%796, %1119 : tensor<1x1x1024xf32>, tensor<1x32x1024xf32>) outs(%1120 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_36", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb119(%1122: f32, %1123: f32, %1124: f32):
        %1125 = arith.mulf %1122, %1123 : f32
        linalg.yield %1125 : f32
      } -> tensor<1x32x1024xf32>
      %1126 = tensor.empty() : tensor<1x32x1024xf32>
      %1127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%781, %1121 : tensor<1x32x1024xf32>, tensor<1x32x1024xf32>) outs(%1126 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "add_26", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb120(%1128: f32, %1129: f32, %1130: f32):
        %1131 = arith.addf %1128, %1129 : f32
        linalg.yield %1131 : f32
      } -> tensor<1x32x1024xf32>
      %1132 = tensor.empty() : tensor<1x32x1024xf32>
      %1133 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1127 : tensor<1x32x1024xf32>) outs(%1132 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "pow_9", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb121(%1134: f32, %1135: f32):
        %1136 = arith.constant 2.000000e+00 : f32
        %1137 = math.powf %1134, %1136 : f32
        linalg.yield %1137 : f32
      } -> tensor<1x32x1024xf32>
      %1138 = arith.constant {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1139 = tensor.splat %1138 {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32>
      %1140 = linalg.reduce ins(%1133:tensor<1x32x1024xf32>) outs(%1139:tensor<1x32xf32>) dimensions = [2]
      (%1141: f32, %1142: f32) {
        %1143 = arith.addf %1141, %1142 : f32
        linalg.yield %1143 : f32
      }
      %1144 = arith.constant {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.024000e+03 : f32
      %1145 = tensor.splat %1144 {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32>
      %1146 = tensor.empty() : tensor<1x32xf32>
      %1147 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1140, %1145 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1146 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb122(%1148: f32, %1149: f32, %1150: f32):
        %1151 = arith.divf %1148, %1149 : f32
        linalg.yield %1151 : f32
      } -> tensor<1x32xf32>
      %1152 = tensor.collapse_shape %1147 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
      %1153 = tensor.expand_shape %1152 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32x1xf32>
      %1154 = arith.constant {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %1155 = tensor.splat %1154 {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x1xf32>
      %1156 = tensor.empty() : tensor<1x32x1xf32>
      %1157 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1153, %1155 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1156 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb123(%1158: f32, %1159: f32, %1160: f32):
        %1161 = arith.addf %1158, %1159 : f32
        linalg.yield %1161 : f32
      } -> tensor<1x32x1xf32>
      %1162 = tensor.empty() : tensor<1x32x1xf32>
      %1163 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1157 : tensor<1x32x1xf32>) outs(%1162 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_9", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb124(%1164: f32, %1165: f32):
        %1166 = math.rsqrt %1164 : f32
        linalg.yield %1166 : f32
      } -> tensor<1x32x1xf32>
      %1167 = tensor.empty() : tensor<1x32x1024xf32>
      %1168 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1127, %1163 : tensor<1x32x1024xf32>, tensor<1x32x1xf32>) outs(%1167 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb125(%1169: f32, %1170: f32, %1171: f32):
        %1172 = arith.mulf %1169, %1170 : f32
        linalg.yield %1172 : f32
      } -> tensor<1x32x1024xf32>
      %1173 = tensor.empty() : tensor<1x32x1024xf32>
      %1174 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %1168 : tensor<1024xf32>, tensor<1x32x1024xf32>) outs(%1173 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_38", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb126(%1175: f32, %1176: f32, %1177: f32):
        %1178 = arith.mulf %1175, %1176 : f32
        linalg.yield %1178 : f32
      } -> tensor<1x32x1024xf32>
      %1179 = tensor.empty() : tensor<1x32x1024xf32>
      %1180 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1174 : tensor<1x32x1024xf32>) outs(%1179 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "pow_10", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb127(%1181: f32, %1182: f32):
        %1183 = arith.constant 2.000000e+00 : f32
        %1184 = math.powf %1181, %1183 : f32
        linalg.yield %1184 : f32
      } -> tensor<1x32x1024xf32>
      %1185 = arith.constant {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1186 = tensor.splat %1185 {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32>
      %1187 = linalg.reduce ins(%1180:tensor<1x32x1024xf32>) outs(%1186:tensor<1x32xf32>) dimensions = [2]
      (%1188: f32, %1189: f32) {
        %1190 = arith.addf %1188, %1189 : f32
        linalg.yield %1190 : f32
      }
      %1191 = arith.constant {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.024000e+03 : f32
      %1192 = tensor.splat %1191 {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32>
      %1193 = tensor.empty() : tensor<1x32xf32>
      %1194 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1187, %1192 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1193 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb128(%1195: f32, %1196: f32, %1197: f32):
        %1198 = arith.divf %1195, %1196 : f32
        linalg.yield %1198 : f32
      } -> tensor<1x32xf32>
      %1199 = tensor.collapse_shape %1194 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
      %1200 = tensor.expand_shape %1199 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32x1xf32>
      %1201 = arith.constant {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %1202 = tensor.splat %1201 {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x1xf32>
      %1203 = tensor.empty() : tensor<1x32x1xf32>
      %1204 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1200, %1202 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1203 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb129(%1205: f32, %1206: f32, %1207: f32):
        %1208 = arith.addf %1205, %1206 : f32
        linalg.yield %1208 : f32
      } -> tensor<1x32x1xf32>
      %1209 = tensor.empty() : tensor<1x32x1xf32>
      %1210 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1204 : tensor<1x32x1xf32>) outs(%1209 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_10", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb130(%1211: f32, %1212: f32):
        %1213 = math.rsqrt %1211 : f32
        linalg.yield %1213 : f32
      } -> tensor<1x32x1xf32>
      %1214 = tensor.empty() : tensor<1x32x1024xf32>
      %1215 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1174, %1210 : tensor<1x32x1024xf32>, tensor<1x32x1xf32>) outs(%1214 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_39", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb131(%1216: f32, %1217: f32, %1218: f32):
        %1219 = arith.mulf %1216, %1217 : f32
        linalg.yield %1219 : f32
      } -> tensor<1x32x1024xf32>
      %1220 = tensor.empty() : tensor<1x32x1024xf32>
      %1221 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %1215 : tensor<1024xf32>, tensor<1x32x1024xf32>) outs(%1220 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_40", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb132(%1222: f32, %1223: f32, %1224: f32):
        %1225 = arith.mulf %1222, %1223 : f32
        linalg.yield %1225 : f32
      } -> tensor<1x32x1024xf32>
      %1226 = arith.constant {prov.region_id = "add_29", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
      %1227 = tensor.splat %1226 {prov.region_id = "add_29", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1024xf32>
      %1228 = tensor.empty() : tensor<1x1x1024xf32>
      %1229 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%798, %1227 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%1228 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_29", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb133(%1230: f32, %1231: f32, %1232: f32):
        %1233 = arith.addf %1230, %1231 : f32
        linalg.yield %1233 : f32
      } -> tensor<1x1x1024xf32>
      %1234 = tensor.empty() : tensor<1x32x1024xf32>
      %1235 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1221, %1229 : tensor<1x32x1024xf32>, tensor<1x1x1024xf32>) outs(%1234 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb134(%1236: f32, %1237: f32, %1238: f32):
        %1239 = arith.mulf %1236, %1237 : f32
        linalg.yield %1239 : f32
      } -> tensor<1x32x1024xf32>
      %1240 = tensor.empty() : tensor<1x32x1024xf32>
      %1241 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1235, %797 : tensor<1x32x1024xf32>, tensor<1x1x1024xf32>) outs(%1240 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb135(%1242: f32, %1243: f32, %1244: f32):
        %1245 = arith.addf %1242, %1243 : f32
        linalg.yield %1245 : f32
      } -> tensor<1x32x1024xf32>
      %1246 = tensor.empty() : tensor<1024x4096xf32>
      %1247 = linalg.transpose ins(%19:tensor<4096x1024xf32>) outs(%1246:tensor<1024x4096xf32>) permutation = [1, 0]
      %1248 = tensor.empty() : tensor<1x32x4096xf32>
      %1249 = arith.constant 0.000000e+00 : f32
      %1250 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1249 : f32) outs(%1248 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
      %1251 = linalg.matmul {prov.region_id = "matmul_14", prov.dispatch_id = "matmul_14", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1241, %1247 : tensor<1x32x1024xf32>, tensor<1024x4096xf32>) outs(%1250 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
      %1252 = tensor.empty() : tensor<1x32x4096xf32>
      %1253 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1251 : tensor<1x32x4096xf32>) outs(%1252 : tensor<1x32x4096xf32>) attrs =  {prov.region_id = "silu_2", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb136(%1254: f32, %1255: f32):
        %1256 = arith.constant 1.000000e+00 : f32
        %1257 = arith.negf %1254 : f32
        %1258 = math.exp %1257 : f32
        %1259 = arith.addf %1256, %1258 : f32
        %1260 = arith.divf %1256, %1259 : f32
        %1261 = arith.mulf %1254, %1260 : f32
        linalg.yield %1261 : f32
      } -> tensor<1x32x4096xf32>
      %1262 = tensor.empty() : tensor<1024x4096xf32>
      %1263 = linalg.transpose ins(%20:tensor<4096x1024xf32>) outs(%1262:tensor<1024x4096xf32>) permutation = [1, 0]
      %1264 = tensor.empty() : tensor<1x32x4096xf32>
      %1265 = arith.constant 0.000000e+00 : f32
      %1266 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1265 : f32) outs(%1264 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
      %1267 = linalg.matmul {prov.region_id = "matmul_15", prov.dispatch_id = "matmul_15", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1241, %1263 : tensor<1x32x1024xf32>, tensor<1024x4096xf32>) outs(%1266 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
      %1268 = tensor.empty() : tensor<1x32x4096xf32>
      %1269 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1253, %1267 : tensor<1x32x4096xf32>, tensor<1x32x4096xf32>) outs(%1268 : tensor<1x32x4096xf32>) attrs =  {prov.region_id = "mul_42", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb137(%1270: f32, %1271: f32, %1272: f32):
        %1273 = arith.mulf %1270, %1271 : f32
        linalg.yield %1273 : f32
      } -> tensor<1x32x4096xf32>
      %1274 = tensor.empty() : tensor<4096x1024xf32>
      %1275 = linalg.transpose ins(%21:tensor<1024x4096xf32>) outs(%1274:tensor<4096x1024xf32>) permutation = [1, 0]
      %1276 = tensor.empty() : tensor<1x32x1024xf32>
      %1277 = arith.constant 0.000000e+00 : f32
      %1278 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1277 : f32) outs(%1276 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
      %1279 = linalg.matmul {prov.region_id = "matmul_16", prov.dispatch_id = "matmul_16", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1269, %1275 : tensor<1x32x4096xf32>, tensor<4096x1024xf32>) outs(%1278 : tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
      %1280 = tensor.empty() : tensor<1x32x1024xf32>
      %1281 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%799, %1279 : tensor<1x1x1024xf32>, tensor<1x32x1024xf32>) outs(%1280 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb138(%1282: f32, %1283: f32, %1284: f32):
        %1285 = arith.mulf %1282, %1283 : f32
        linalg.yield %1285 : f32
      } -> tensor<1x32x1024xf32>
      %1286 = tensor.empty() : tensor<1x32x1024xf32>
      %1287 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1174, %1281 : tensor<1x32x1024xf32>, tensor<1x32x1024xf32>) outs(%1286 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb139(%1288: f32, %1289: f32, %1290: f32):
        %1291 = arith.addf %1288, %1289 : f32
        linalg.yield %1291 : f32
      } -> tensor<1x32x1024xf32>
      %1292 = tensor.empty() : tensor<1x32x1024xf32>
      %1293 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1287 : tensor<1x32x1024xf32>) outs(%1292 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "pow_11", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb140(%1294: f32, %1295: f32):
        %1296 = arith.constant 2.000000e+00 : f32
        %1297 = math.powf %1294, %1296 : f32
        linalg.yield %1297 : f32
      } -> tensor<1x32x1024xf32>
      %1298 = arith.constant {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1299 = tensor.splat %1298 {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32>
      %1300 = linalg.reduce ins(%1293:tensor<1x32x1024xf32>) outs(%1299:tensor<1x32xf32>) dimensions = [2]
      (%1301: f32, %1302: f32) {
        %1303 = arith.addf %1301, %1302 : f32
        linalg.yield %1303 : f32
      }
      %1304 = arith.constant {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.024000e+03 : f32
      %1305 = tensor.splat %1304 {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32>
      %1306 = tensor.empty() : tensor<1x32xf32>
      %1307 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1300, %1305 : tensor<1x32xf32>, tensor<1x32xf32>) outs(%1306 : tensor<1x32xf32>) attrs =  {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb141(%1308: f32, %1309: f32, %1310: f32):
        %1311 = arith.divf %1308, %1309 : f32
        linalg.yield %1311 : f32
      } -> tensor<1x32xf32>
      %1312 = tensor.collapse_shape %1307 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x32xf32> into tensor<32xf32>
      %1313 = tensor.expand_shape %1312 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 32, 1] {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<1x32x1xf32>
      %1314 = arith.constant {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
      %1315 = tensor.splat %1314 {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x32x1xf32>
      %1316 = tensor.empty() : tensor<1x32x1xf32>
      %1317 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1313, %1315 : tensor<1x32x1xf32>, tensor<1x32x1xf32>) outs(%1316 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb142(%1318: f32, %1319: f32, %1320: f32):
        %1321 = arith.addf %1318, %1319 : f32
        linalg.yield %1321 : f32
      } -> tensor<1x32x1xf32>
      %1322 = tensor.empty() : tensor<1x32x1xf32>
      %1323 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1317 : tensor<1x32x1xf32>) outs(%1322 : tensor<1x32x1xf32>) attrs =  {prov.region_id = "rsqrt_11", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb143(%1324: f32, %1325: f32):
        %1326 = math.rsqrt %1324 : f32
        linalg.yield %1326 : f32
      } -> tensor<1x32x1xf32>
      %1327 = tensor.empty() : tensor<1x32x1024xf32>
      %1328 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1287, %1323 : tensor<1x32x1024xf32>, tensor<1x32x1xf32>) outs(%1327 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_44", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb144(%1329: f32, %1330: f32, %1331: f32):
        %1332 = arith.mulf %1329, %1330 : f32
        linalg.yield %1332 : f32
      } -> tensor<1x32x1024xf32>
      %1333 = tensor.empty() : tensor<1x32x1024xf32>
      %1334 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %1328 : tensor<1024xf32>, tensor<1x32x1024xf32>) outs(%1333 : tensor<1x32x1024xf32>) attrs =  {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb145(%1335: f32, %1336: f32, %1337: f32):
        %1338 = arith.mulf %1335, %1336 : f32
        linalg.yield %1338 : f32
      } -> tensor<1x32x1024xf32>
      %1339 = "tensor.extract_slice"(%1334) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_10", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32>
      %1340 = "tensor.extract_slice"(%1339) <{static_offsets = array<i64: 0, 2, 0>, static_sizes = array<i64: 1, 30, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_11", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x32x1024xf32>) -> tensor<1x30x1024xf32>
      %1341 = "tensor.extract_slice"(%1340) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 30, 1024>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_12", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x30x1024xf32>) -> tensor<1x30x1024xf32>
      %1342 = tensor.empty() : tensor<1024x32xf32>
      %1343 = linalg.transpose ins(%30:tensor<32x1024xf32>) outs(%1342:tensor<1024x32xf32>) permutation = [1, 0]
      %1344 = tensor.empty() : tensor<1x30x32xf32>
      %1345 = arith.constant 0.000000e+00 : f32
      %1346 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1345 : f32) outs(%1344 : tensor<1x30x32xf32>) -> tensor<1x30x32xf32>
      %1347 = linalg.matmul {prov.region_id = "matmul_17", prov.dispatch_id = "matmul_17", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1341, %1343 : tensor<1x30x1024xf32>, tensor<1024x32xf32>) outs(%1346 : tensor<1x30x32xf32>) -> tensor<1x30x32xf32>
      %1348 = tensor.empty() : tensor<1x30x32xf32>
      %1349 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1347 : tensor<1x30x32xf32>) outs(%1348 : tensor<1x30x32xf32>) attrs =  {prov.region_id = "gelu_1", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32"} {
      ^bb146(%1350: f32, %1351: f32):
        %1352 = arith.constant 5.000000e-01 : f32
        %1353 = arith.constant 1.000000e+00 : f32
        %1354 = arith.constant 0.707106769 : f32
        %1355 = arith.mulf %1350, %1354 : f32
        %1356 = math.erf %1355 : f32
        %1357 = arith.addf %1353, %1356 : f32
        %1358 = arith.mulf %1352, %1350 : f32
        %1359 = arith.mulf %1358, %1357 : f32
        linalg.yield %1359 : f32
      } -> tensor<1x30x32xf32>
      %1360 = tensor.empty() : tensor<32x32xf32>
      %1361 = linalg.transpose ins(%31:tensor<32x32xf32>) outs(%1360:tensor<32x32xf32>) permutation = [1, 0]
      %1362 = tensor.empty() : tensor<1x30x32xf32>
      %1363 = arith.constant 0.000000e+00 : f32
      %1364 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1363 : f32) outs(%1362 : tensor<1x30x32xf32>) -> tensor<1x30x32xf32>
      %1365 = linalg.matmul {prov.region_id = "matmul_18", prov.dispatch_id = "matmul_18", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1349, %1361 : tensor<1x30x32xf32>, tensor<32x32xf32>) outs(%1364 : tensor<1x30x32xf32>) -> tensor<1x30x32xf32>
      %1366 = arith.constant {prov.region_id = "mul_46", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 2.000000e-01 : f32
      %1367 = tensor.splat %1366 {prov.region_id = "mul_46", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x30x32xf32>
      %1368 = tensor.empty() : tensor<1x30x32xf32>
      %1369 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1365, %1367 : tensor<1x30x32xf32>, tensor<1x30x32xf32>) outs(%1368 : tensor<1x30x32xf32>) attrs =  {prov.region_id = "mul_46", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb147(%1370: f32, %1371: f32, %1372: f32):
        %1373 = arith.mulf %1370, %1371 : f32
        linalg.yield %1373 : f32
      } -> tensor<1x30x32xf32>
      %1374 = tensor.empty() : tensor<1x30x32xf32>
      %1375 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%112, %1369 : tensor<1x30x32xf32>, tensor<1x30x32xf32>) outs(%1374 : tensor<1x30x32xf32>) attrs =  {prov.region_id = "add_33", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb148(%1376: f32, %1377: f32, %1378: f32):
        %1379 = arith.addf %1376, %1377 : f32
        linalg.yield %1379 : f32
      } -> tensor<1x30x32xf32>
      %1380 = arith.constant {prov.region_id = "add_34", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} 1 : i64
      %1381 = tensor.splat %1380 {prov.region_id = "add_34", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} : tensor<i64>
      %1382 = tensor.empty() : tensor<i64>
      %1383 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%111, %1381 : tensor<i64>, tensor<i64>) outs(%1382 : tensor<i64>) attrs =  {prov.region_id = "add_34", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb149(%1384: i64, %1385: i64, %1386: i64):
        %1387 = arith.addi %1384, %1385 : i64
        linalg.yield %1387 : i64
      } -> tensor<i64>
      scf.yield %1383, %1375 : tensor<i64>, tensor<1x30x32xf32>
    }
    func.return %109 : tensor<1x30x32xf32>
  }
}
