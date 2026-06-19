builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func private @aten_zeros_default() -> tensor<i64>
  func.func private @aten_zeros_like_default(tensor<1x64x128xf32>) -> tensor<1x64x128xf32>
  func.func private @aten_index_select_default_wl0(tensor<5xf32>, tensor<1xi64>) -> tensor<1xf32>
  func.func private @aten_rms_norm_default_wl1(tensor<1x67x2048xf32>, tensor<2048xf32>) -> tensor<1x67x2048xf32>
  func.func private @aten_rms_norm_default_1_wl2(tensor<1x32x67x64xf32>, tensor<64xf32>) -> tensor<1x32x67x64xf32>
  func.func private @aten_rms_norm_default_2_wl3(tensor<1x32x32x64xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>
  func.func private @aten_rms_norm_default_3_wl4(tensor<1x32x4096x64xf32>, tensor<64xf32>) -> tensor<1x32x4096x64xf32>
  func.func @forward(%0: tensor<1x67x2048xf32>, %1: tensor<1x1024x2048xf32>, %2: tensor<1x4096x2048xf32>, %3: tensor<2048x256xf32>, %4: tensor<2048xf32>, %5: tensor<2048x2048xf32>, %6: tensor<2048xf32>, %7: tensor<2048x256xf32>, %8: tensor<2048xf32>, %9: tensor<2048x2048xf32>, %10: tensor<2048xf32>, %11: tensor<2048xf32>, %12: tensor<6144x2048xf32>, %13: tensor<6144xf32>, %14: tensor<64xf32>, %15: tensor<64xf32>, %16: tensor<2048x2048xf32>, %17: tensor<2048xf32>, %18: tensor<2048x2048xf32>, %19: tensor<2048xf32>, %20: tensor<4096x2048xf32>, %21: tensor<4096xf32>, %22: tensor<64xf32>, %23: tensor<64xf32>, %24: tensor<2048x2048xf32>, %25: tensor<2048xf32>, %26: tensor<2048xf32>, %27: tensor<2048x2048xf32>, %28: tensor<2048xf32>, %29: tensor<2048x2048xf32>, %30: tensor<2048xf32>, %31: tensor<2048xf32>, %32: tensor<2048xf32>, %33: tensor<6144x2048xf32>, %34: tensor<6144xf32>, %35: tensor<64xf32>, %36: tensor<64xf32>, %37: tensor<2048x2048xf32>, %38: tensor<2048xf32>, %39: tensor<2048x2048xf32>, %40: tensor<2048xf32>, %41: tensor<4096x2048xf32>, %42: tensor<4096xf32>, %43: tensor<64xf32>, %44: tensor<64xf32>, %45: tensor<2048x2048xf32>, %46: tensor<2048xf32>, %47: tensor<2048xf32>, %48: tensor<2048x2048xf32>, %49: tensor<2048xf32>, %50: tensor<2048x2048xf32>, %51: tensor<2048xf32>, %52: tensor<2048xf32>, %53: tensor<2048xf32>, %54: tensor<2048x2048xf32>, %55: tensor<2048xf32>, %56: tensor<128x2048xf32>, %57: tensor<128xf32>, %58: tensor<2048x256xf32>, %59: tensor<2048xf32>, %60: tensor<5xf32>, %61: tensor<5xf32>, %62: tensor<5xf32>, %63: tensor<5xf32>, %64: tensor<5xf32>, %65: tensor<5xf32>, %66: tensor<1x64x128xf32>, %67: tensor<1xf32>, %68: tensor<1x1x2048xf32>, %69: tensor<1x1x128xf32>, %70: tensor<1x32x2048xf32>, %71: tensor<1x4096x2048xf32>, %72: tensor<1x32xi1>) -> tensor<1x64x128xf32> {
    %73 = tensor.empty() : tensor<1x64x128xf32>
    %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%69 : tensor<1x1x128xf32>) outs(%73 : tensor<1x64x128xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
    ^bb0(%75: f32, %76: f32):
      linalg.yield %75 : f32
    } -> tensor<1x64x128xf32>
    %77 = func.call @aten_zeros_default() {prov.region_id = "aten_zeros_default_0", prov.dispatch_id = "aten_zeros_default_0"} : () -> tensor<i64>
    %78 = func.call @aten_zeros_like_default(%66) {prov.region_id = "aten_zeros_like_default_0", prov.dispatch_id = "aten_zeros_like_default_0"} : (tensor<1x64x128xf32>) -> tensor<1x64x128xf32>
    %79 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 0 : index
    %80 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 5 : index
    %81 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 1 : index
    %82, %83, %84 = scf.for %85 = %79 to %80 step %81 iter_args(%86 = %77, %87 = %66, %88 = %78) -> (tensor<i64>, tensor<1x64x128xf32>, tensor<1x64x128xf32>) {
      %89 = tensor.concat dim(2) %87, %74 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x64x128xf32>, tensor<1x64x128xf32>) -> tensor<1x64x256xf32>
      %90 = tensor.empty() : tensor<256x2048xf32>
      %91 = linalg.transpose ins(%58:tensor<2048x256xf32>) outs(%90:tensor<256x2048xf32>) permutation = [1, 0]
      %92 = tensor.empty() : tensor<1x64x2048xf32>
      %93 = arith.constant 0.000000e+00 : f32
      %94 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%93 : f32) outs(%92 : tensor<1x64x2048xf32>) -> tensor<1x64x2048xf32>
      %95 = linalg.matmul {prov.region_id = "matmul_0", prov.dispatch_id = "matmul_0", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%89, %91 : tensor<1x64x256xf32>, tensor<256x2048xf32>) outs(%94 : tensor<1x64x2048xf32>) -> tensor<1x64x2048xf32>
      %96 = tensor.empty() : tensor<1x64x2048xf32>
      %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95, %59 : tensor<1x64x2048xf32>, tensor<2048xf32>) outs(%96 : tensor<1x64x2048xf32>) attrs =  {prov.region_id = "add_0", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb1(%98: f32, %99: f32, %100: f32):
        %101 = arith.addf %98, %99 : f32
        linalg.yield %101 : f32
      } -> tensor<1x64x2048xf32>
      %102 = tensor.concat dim(1) %68, %97 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x1x2048xf32>, tensor<1x64x2048xf32>) -> tensor<1x65x2048xf32>
      %103 = tensor.extract %86[] : tensor<i64>
      %104 = tensor.from_elements %103 {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "int64"} : tensor<1xi64>
      %105 = func.call @aten_index_select_default_wl0(%65, %104) {prov.region_id = "aten_index_select_default_0", prov.dispatch_id = "aten_index_select_default_0"} : (tensor<5xf32>, tensor<1xi64>) -> tensor<1xf32>
      %106 = arith.constant {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} 0 : index
      %107 = tensor.extract %105[%106] : tensor<1xf32>
      %108 = tensor.from_elements %107 {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<f32>
      %109 = tensor.extract %108[] : tensor<f32>
      %110 = tensor.from_elements %109 {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1xf32>
      %111 = tensor.empty() : tensor<128xf32>
      %112 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%111 : tensor<128xf32>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start", prov.orig_dtype = "float32"} {
      ^bb2(%113: f32):
        %114 = linalg.index 0 : index
        %115 = arith.index_cast %114 : index to i64
        %116 = arith.sitofp %115 : i64 to f32
        %117 = arith.constant 1.000000e+00 : f32
        %118 = arith.mulf %116, %117 : f32
        %119 = arith.constant 0.000000e+00 : f32
        %120 = arith.addf %119, %118 : f32
        linalg.yield %120 : f32
      } -> tensor<128xf32>
      %121 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} -9.2103405 : f32
      %122 = tensor.splat %121 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<128xf32>
      %123 = tensor.empty() : tensor<128xf32>
      %124 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%112, %122 : tensor<128xf32>, tensor<128xf32>) outs(%123 : tensor<128xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb3(%125: f32, %126: f32, %127: f32):
        %128 = arith.mulf %125, %126 : f32
        linalg.yield %128 : f32
      } -> tensor<128xf32>
      %129 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %130 = tensor.splat %129 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<128xf32>
      %131 = tensor.empty() : tensor<128xf32>
      %132 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%124, %130 : tensor<128xf32>, tensor<128xf32>) outs(%131 : tensor<128xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
      ^bb4(%133: f32, %134: f32, %135: f32):
        %136 = arith.divf %133, %134 : f32
        linalg.yield %136 : f32
      } -> tensor<128xf32>
      %137 = tensor.empty() : tensor<128xf32>
      %138 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%132 : tensor<128xf32>) outs(%137 : tensor<128xf32>) attrs =  {prov.region_id = "exp_0", prov._pattern_hint = "exp", prov.op = "exp", prov.family = "elementwise", prov.aten = "aten.exp.default", prov.orig_dtype = "float32"} {
      ^bb5(%139: f32, %140: f32):
        %141 = math.exp %139 : f32
        linalg.yield %141 : f32
      } -> tensor<128xf32>
      %142 = tensor.expand_shape %110 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1xf32>
      %143 = tensor.expand_shape %138 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x128xf32>
      %144 = tensor.empty() : tensor<1x128xf32>
      %145 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%142, %143 : tensor<1x1xf32>, tensor<1x128xf32>) outs(%144 : tensor<1x128xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb6(%146: f32, %147: f32, %148: f32):
        %149 = arith.mulf %146, %147 : f32
        linalg.yield %149 : f32
      } -> tensor<1x128xf32>
      %150 = tensor.empty() : tensor<1x128xf32>
      %151 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%145 : tensor<1x128xf32>) outs(%150 : tensor<1x128xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32"} {
      ^bb7(%152: f32, %153: f32):
        %154 = math.cos %152 : f32
        linalg.yield %154 : f32
      } -> tensor<1x128xf32>
      %155 = tensor.empty() : tensor<1x128xf32>
      %156 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%145 : tensor<1x128xf32>) outs(%155 : tensor<1x128xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32"} {
      ^bb8(%157: f32, %158: f32):
        %159 = math.sin %157 : f32
        linalg.yield %159 : f32
      } -> tensor<1x128xf32>
      %160 = tensor.concat dim(1) %151, %156 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x256xf32>
      %161 = tensor.empty() : tensor<256x2048xf32>
      %162 = linalg.transpose ins(%3:tensor<2048x256xf32>) outs(%161:tensor<256x2048xf32>) permutation = [1, 0]
      %163 = tensor.empty() : tensor<1x2048xf32>
      %164 = arith.constant 0.000000e+00 : f32
      %165 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%164 : f32) outs(%163 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %166 = linalg.matmul {prov.region_id = "matmul_1", prov.dispatch_id = "matmul_1", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%160, %162 : tensor<1x256xf32>, tensor<256x2048xf32>) outs(%165 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %167 = tensor.empty() : tensor<1x2048xf32>
      %168 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%166, %4 : tensor<1x2048xf32>, tensor<2048xf32>) outs(%167 : tensor<1x2048xf32>) attrs =  {prov.region_id = "add_1", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb9(%169: f32, %170: f32, %171: f32):
        %172 = arith.addf %169, %170 : f32
        linalg.yield %172 : f32
      } -> tensor<1x2048xf32>
      %173 = tensor.empty() : tensor<1x2048xf32>
      %174 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%168 : tensor<1x2048xf32>) outs(%173 : tensor<1x2048xf32>) attrs =  {prov.region_id = "silu_0", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb10(%175: f32, %176: f32):
        %177 = arith.constant 1.000000e+00 : f32
        %178 = arith.negf %175 : f32
        %179 = math.exp %178 : f32
        %180 = arith.addf %177, %179 : f32
        %181 = arith.divf %177, %180 : f32
        %182 = arith.mulf %175, %181 : f32
        linalg.yield %182 : f32
      } -> tensor<1x2048xf32>
      %183 = tensor.empty() : tensor<2048x2048xf32>
      %184 = linalg.transpose ins(%5:tensor<2048x2048xf32>) outs(%183:tensor<2048x2048xf32>) permutation = [1, 0]
      %185 = tensor.empty() : tensor<1x2048xf32>
      %186 = arith.constant 0.000000e+00 : f32
      %187 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%186 : f32) outs(%185 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %188 = linalg.matmul {prov.region_id = "matmul_2", prov.dispatch_id = "matmul_2", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%174, %184 : tensor<1x2048xf32>, tensor<2048x2048xf32>) outs(%187 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %189 = tensor.empty() : tensor<1x2048xf32>
      %190 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%188, %6 : tensor<1x2048xf32>, tensor<2048xf32>) outs(%189 : tensor<1x2048xf32>) attrs =  {prov.region_id = "add_2", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb11(%191: f32, %192: f32, %193: f32):
        %194 = arith.addf %191, %192 : f32
        linalg.yield %194 : f32
      } -> tensor<1x2048xf32>
      %195 = tensor.collapse_shape %190 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x2048xf32> into tensor<2048xf32>
      %196 = tensor.expand_shape %195 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 2048] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x1x2048xf32>
      %197 = tensor.empty() : tensor<128xf32>
      %198 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%197 : tensor<128xf32>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start", prov.orig_dtype = "float32"} {
      ^bb12(%199: f32):
        %200 = linalg.index 0 : index
        %201 = arith.index_cast %200 : index to i64
        %202 = arith.sitofp %201 : i64 to f32
        %203 = arith.constant 1.000000e+00 : f32
        %204 = arith.mulf %202, %203 : f32
        %205 = arith.constant 0.000000e+00 : f32
        %206 = arith.addf %205, %204 : f32
        linalg.yield %206 : f32
      } -> tensor<128xf32>
      %207 = arith.constant {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} -9.2103405 : f32
      %208 = tensor.splat %207 {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<128xf32>
      %209 = tensor.empty() : tensor<128xf32>
      %210 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%198, %208 : tensor<128xf32>, tensor<128xf32>) outs(%209 : tensor<128xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb13(%211: f32, %212: f32, %213: f32):
        %214 = arith.mulf %211, %212 : f32
        linalg.yield %214 : f32
      } -> tensor<128xf32>
      %215 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %216 = tensor.splat %215 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<128xf32>
      %217 = tensor.empty() : tensor<128xf32>
      %218 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%210, %216 : tensor<128xf32>, tensor<128xf32>) outs(%217 : tensor<128xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
      ^bb14(%219: f32, %220: f32, %221: f32):
        %222 = arith.divf %219, %220 : f32
        linalg.yield %222 : f32
      } -> tensor<128xf32>
      %223 = tensor.empty() : tensor<128xf32>
      %224 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%218 : tensor<128xf32>) outs(%223 : tensor<128xf32>) attrs =  {prov.region_id = "exp_1", prov._pattern_hint = "exp", prov.op = "exp", prov.family = "elementwise", prov.aten = "aten.exp.default", prov.orig_dtype = "float32"} {
      ^bb15(%225: f32, %226: f32):
        %227 = math.exp %225 : f32
        linalg.yield %227 : f32
      } -> tensor<128xf32>
      %228 = tensor.expand_shape %67 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1xf32>
      %229 = tensor.expand_shape %224 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x128xf32>
      %230 = tensor.empty() : tensor<1x128xf32>
      %231 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%228, %229 : tensor<1x1xf32>, tensor<1x128xf32>) outs(%230 : tensor<1x128xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb16(%232: f32, %233: f32, %234: f32):
        %235 = arith.mulf %232, %233 : f32
        linalg.yield %235 : f32
      } -> tensor<1x128xf32>
      %236 = tensor.empty() : tensor<1x128xf32>
      %237 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%231 : tensor<1x128xf32>) outs(%236 : tensor<1x128xf32>) attrs =  {prov.region_id = "cos_1", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32"} {
      ^bb17(%238: f32, %239: f32):
        %240 = math.cos %238 : f32
        linalg.yield %240 : f32
      } -> tensor<1x128xf32>
      %241 = tensor.empty() : tensor<1x128xf32>
      %242 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%231 : tensor<1x128xf32>) outs(%241 : tensor<1x128xf32>) attrs =  {prov.region_id = "sin_1", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32"} {
      ^bb18(%243: f32, %244: f32):
        %245 = math.sin %243 : f32
        linalg.yield %245 : f32
      } -> tensor<1x128xf32>
      %246 = tensor.concat dim(1) %237, %242 {prov.region_id = "cat_3", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x256xf32>
      %247 = tensor.empty() : tensor<256x2048xf32>
      %248 = linalg.transpose ins(%7:tensor<2048x256xf32>) outs(%247:tensor<256x2048xf32>) permutation = [1, 0]
      %249 = tensor.empty() : tensor<1x2048xf32>
      %250 = arith.constant 0.000000e+00 : f32
      %251 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%250 : f32) outs(%249 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %252 = linalg.matmul {prov.region_id = "matmul_3", prov.dispatch_id = "matmul_3", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%246, %248 : tensor<1x256xf32>, tensor<256x2048xf32>) outs(%251 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %253 = tensor.empty() : tensor<1x2048xf32>
      %254 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%252, %8 : tensor<1x2048xf32>, tensor<2048xf32>) outs(%253 : tensor<1x2048xf32>) attrs =  {prov.region_id = "add_3", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb19(%255: f32, %256: f32, %257: f32):
        %258 = arith.addf %255, %256 : f32
        linalg.yield %258 : f32
      } -> tensor<1x2048xf32>
      %259 = tensor.empty() : tensor<1x2048xf32>
      %260 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%254 : tensor<1x2048xf32>) outs(%259 : tensor<1x2048xf32>) attrs =  {prov.region_id = "silu_1", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb20(%261: f32, %262: f32):
        %263 = arith.constant 1.000000e+00 : f32
        %264 = arith.negf %261 : f32
        %265 = math.exp %264 : f32
        %266 = arith.addf %263, %265 : f32
        %267 = arith.divf %263, %266 : f32
        %268 = arith.mulf %261, %267 : f32
        linalg.yield %268 : f32
      } -> tensor<1x2048xf32>
      %269 = tensor.empty() : tensor<2048x2048xf32>
      %270 = linalg.transpose ins(%9:tensor<2048x2048xf32>) outs(%269:tensor<2048x2048xf32>) permutation = [1, 0]
      %271 = tensor.empty() : tensor<1x2048xf32>
      %272 = arith.constant 0.000000e+00 : f32
      %273 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%272 : f32) outs(%271 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %274 = linalg.matmul {prov.region_id = "matmul_4", prov.dispatch_id = "matmul_4", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%260, %270 : tensor<1x2048xf32>, tensor<2048x2048xf32>) outs(%273 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %275 = tensor.empty() : tensor<1x2048xf32>
      %276 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%274, %10 : tensor<1x2048xf32>, tensor<2048xf32>) outs(%275 : tensor<1x2048xf32>) attrs =  {prov.region_id = "add_4", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb21(%277: f32, %278: f32, %279: f32):
        %280 = arith.addf %277, %278 : f32
        linalg.yield %280 : f32
      } -> tensor<1x2048xf32>
      %281 = tensor.collapse_shape %276 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x2048xf32> into tensor<2048xf32>
      %282 = tensor.expand_shape %281 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 2048] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x1x2048xf32>
      %283 = tensor.empty() : tensor<1x1x2048xf32>
      %284 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%196 : tensor<1x1x2048xf32>) outs(%283 : tensor<1x1x2048xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb22(%285: f32, %286: f32):
        linalg.yield %285 : f32
      } -> tensor<1x1x2048xf32>
      %287 = tensor.concat dim(1) %284, %282, %102 {prov.region_id = "cat_4", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x1x2048xf32>, tensor<1x1x2048xf32>, tensor<1x65x2048xf32>) -> tensor<1x67x2048xf32>
      %288 = tensor.empty() : tensor<1x67x2048xf32>
      %289 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%287, %0 : tensor<1x67x2048xf32>, tensor<1x67x2048xf32>) outs(%288 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb23(%290: f32, %291: f32, %292: f32):
        %293 = arith.addf %290, %291 : f32
        linalg.yield %293 : f32
      } -> tensor<1x67x2048xf32>
      %294 = "tensor.extract_slice"(%1) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 2048>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x1024x2048xf32>) -> tensor<1x32x2048xf32>
      %295 = tensor.empty() : tensor<1x32x2048xf32>
      %296 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%70, %294 : tensor<1x32x2048xf32>, tensor<1x32x2048xf32>) outs(%295 : tensor<1x32x2048xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb24(%297: f32, %298: f32, %299: f32):
        %300 = arith.addf %297, %298 : f32
        linalg.yield %300 : f32
      } -> tensor<1x32x2048xf32>
      %301 = tensor.empty() : tensor<1x4096x2048xf32>
      %302 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%71, %2 : tensor<1x4096x2048xf32>, tensor<1x4096x2048xf32>) outs(%301 : tensor<1x4096x2048xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb25(%303: f32, %304: f32, %305: f32):
        %306 = arith.addf %303, %304 : f32
        linalg.yield %306 : f32
      } -> tensor<1x4096x2048xf32>
      %307 = func.call @aten_rms_norm_default_wl1(%289, %11) {prov.region_id = "aten_rms_norm_default_0", prov.dispatch_id = "aten_rms_norm_default_0"} : (tensor<1x67x2048xf32>, tensor<2048xf32>) -> tensor<1x67x2048xf32>
      %308 = tensor.empty() : tensor<2048x6144xf32>
      %309 = linalg.transpose ins(%12:tensor<6144x2048xf32>) outs(%308:tensor<2048x6144xf32>) permutation = [1, 0]
      %310 = tensor.empty() : tensor<1x67x6144xf32>
      %311 = arith.constant 0.000000e+00 : f32
      %312 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%311 : f32) outs(%310 : tensor<1x67x6144xf32>) -> tensor<1x67x6144xf32>
      %313 = linalg.matmul {prov.region_id = "matmul_5", prov.dispatch_id = "matmul_5", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%307, %309 : tensor<1x67x2048xf32>, tensor<2048x6144xf32>) outs(%312 : tensor<1x67x6144xf32>) -> tensor<1x67x6144xf32>
      %314 = tensor.empty() : tensor<1x67x6144xf32>
      %315 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%313, %13 : tensor<1x67x6144xf32>, tensor<6144xf32>) outs(%314 : tensor<1x67x6144xf32>) attrs =  {prov.region_id = "add_8", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb26(%316: f32, %317: f32, %318: f32):
        %319 = arith.addf %316, %317 : f32
        linalg.yield %319 : f32
      } -> tensor<1x67x6144xf32>
      %320 = tensor.collapse_shape %315 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x67x6144xf32> into tensor<411648xf32>
      %321 = tensor.expand_shape %320 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 67, 3, 32, 64] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<411648xf32> into tensor<1x67x3x32x64xf32>
      %322 = tensor.empty() : tensor<3x1x32x67x64xf32>
      %323 = linalg.transpose ins(%321:tensor<1x67x3x32x64xf32>) outs(%322:tensor<3x1x32x67x64xf32>) permutation = [2, 0, 3, 1, 4]
      %324 = "tensor.extract_slice"(%323) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 67, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<3x1x32x67x64xf32>) -> tensor<1x1x32x67x64xf32>
      %325 = tensor.collapse_shape %324 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x1x32x67x64xf32> into tensor<137216xf32>
      %326 = tensor.expand_shape %325 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 64] {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<137216xf32> into tensor<1x32x67x64xf32>
      %327 = "tensor.extract_slice"(%323) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 67, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<3x1x32x67x64xf32>) -> tensor<1x1x32x67x64xf32>
      %328 = tensor.collapse_shape %327 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x1x32x67x64xf32> into tensor<137216xf32>
      %329 = tensor.expand_shape %328 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 64] {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<137216xf32> into tensor<1x32x67x64xf32>
      %330 = "tensor.extract_slice"(%323) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 67, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<3x1x32x67x64xf32>) -> tensor<1x1x32x67x64xf32>
      %331 = tensor.collapse_shape %330 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x1x32x67x64xf32> into tensor<137216xf32>
      %332 = tensor.expand_shape %331 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 64] {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<137216xf32> into tensor<1x32x67x64xf32>
      %333 = func.call @aten_rms_norm_default_1_wl2(%326, %14) {prov.region_id = "aten_rms_norm_default_1_0", prov.dispatch_id = "aten_rms_norm_default_1_0"} : (tensor<1x32x67x64xf32>, tensor<64xf32>) -> tensor<1x32x67x64xf32>
      %334 = func.call @aten_rms_norm_default_1_wl2(%329, %15) {prov.region_id = "aten_rms_norm_default_1_1", prov.dispatch_id = "aten_rms_norm_default_1_1"} : (tensor<1x32x67x64xf32>, tensor<64xf32>) -> tensor<1x32x67x64xf32>
      %335 = arith.constant {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %336 = tensor.splat %335 {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67x67xf32>
      %337 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%333, %334 : tensor<1x32x67x64xf32>, tensor<1x32x67x64xf32>) outs(%336 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb27(%338: f32, %339: f32, %340: f32):
        %341 = arith.mulf %338, %339 : f32
        %342 = arith.addf %340, %341 : f32
        linalg.yield %342 : f32
      } -> tensor<1x32x67x67xf32>
      %343 = tensor.empty() : tensor<1x32x67x67xf32>
      %344 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%337 : tensor<1x32x67x67xf32>) outs(%343 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb28(%345: f32, %346: f32):
        %347 = arith.constant 1.250000e-01 : f32
        %348 = arith.mulf %345, %347 : f32
        linalg.yield %348 : f32
      } -> tensor<1x32x67x67xf32>
      %349 = arith.constant {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %350 = tensor.splat %349 {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67xf32>
      %351 = linalg.reduce ins(%344:tensor<1x32x67x67xf32>) outs(%350:tensor<1x32x67xf32>) dimensions = [3]
      (%352: f32, %353: f32) {
        %354 = arith.maximumf %352, %353 : f32
        linalg.yield %354 : f32
      }
      %355 = tensor.collapse_shape %351 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67xf32> into tensor<2144xf32>
      %356 = tensor.expand_shape %355 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
      %357 = tensor.empty() : tensor<1x32x67x67xf32>
      %358 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%344, %356 : tensor<1x32x67x67xf32>, tensor<1x32x67x1xf32>) outs(%357 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb29(%359: f32, %360: f32, %361: f32):
        %362 = arith.subf %359, %360 : f32
        linalg.yield %362 : f32
      } -> tensor<1x32x67x67xf32>
      %363 = tensor.empty() : tensor<1x32x67x67xf32>
      %364 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%358 : tensor<1x32x67x67xf32>) outs(%363 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb30(%365: f32, %366: f32):
        %367 = math.exp %365 : f32
        linalg.yield %367 : f32
      } -> tensor<1x32x67x67xf32>
      %368 = arith.constant {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %369 = tensor.splat %368 {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67xf32>
      %370 = linalg.reduce ins(%364:tensor<1x32x67x67xf32>) outs(%369:tensor<1x32x67xf32>) dimensions = [3]
      (%371: f32, %372: f32) {
        %373 = arith.addf %371, %372 : f32
        linalg.yield %373 : f32
      }
      %374 = tensor.collapse_shape %370 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67xf32> into tensor<2144xf32>
      %375 = tensor.expand_shape %374 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
      %376 = tensor.empty() : tensor<1x32x67x67xf32>
      %377 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%364, %375 : tensor<1x32x67x67xf32>, tensor<1x32x67x1xf32>) outs(%376 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb31(%378: f32, %379: f32, %380: f32):
        %381 = arith.divf %378, %379 : f32
        linalg.yield %381 : f32
      } -> tensor<1x32x67x67xf32>
      %382 = arith.constant {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %383 = tensor.splat %382 {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67x64xf32>
      %384 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%377, %332 : tensor<1x32x67x67xf32>, tensor<1x32x67x64xf32>) outs(%383 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "attention_0", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb32(%385: f32, %386: f32, %387: f32):
        %388 = arith.mulf %385, %386 : f32
        %389 = arith.addf %387, %388 : f32
        linalg.yield %389 : f32
      } -> tensor<1x32x67x64xf32>
      %390 = tensor.empty() : tensor<1x67x32x64xf32>
      %391 = linalg.transpose ins(%384:tensor<1x32x67x64xf32>) outs(%390:tensor<1x67x32x64xf32>) permutation = [0, 2, 1, 3]
      %392 = tensor.collapse_shape %391 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x67x32x64xf32> into tensor<137216xf32>
      %393 = tensor.expand_shape %392 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<137216xf32> into tensor<1x67x2048xf32>
      %394 = tensor.empty() : tensor<2048x2048xf32>
      %395 = linalg.transpose ins(%16:tensor<2048x2048xf32>) outs(%394:tensor<2048x2048xf32>) permutation = [1, 0]
      %396 = tensor.empty() : tensor<1x67x2048xf32>
      %397 = arith.constant 0.000000e+00 : f32
      %398 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%397 : f32) outs(%396 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %399 = linalg.matmul {prov.region_id = "matmul_7", prov.dispatch_id = "matmul_7", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%393, %395 : tensor<1x67x2048xf32>, tensor<2048x2048xf32>) outs(%398 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %400 = tensor.empty() : tensor<1x67x2048xf32>
      %401 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%399, %17 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%400 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_9", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb33(%402: f32, %403: f32, %404: f32):
        %405 = arith.addf %402, %403 : f32
        linalg.yield %405 : f32
      } -> tensor<1x67x2048xf32>
      %406 = tensor.empty() : tensor<1x67x2048xf32>
      %407 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%401, %289 : tensor<1x67x2048xf32>, tensor<1x67x2048xf32>) outs(%406 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb34(%408: f32, %409: f32, %410: f32):
        %411 = arith.addf %408, %409 : f32
        linalg.yield %411 : f32
      } -> tensor<1x67x2048xf32>
      %412 = func.call @aten_rms_norm_default_wl1(%407, %26) {prov.region_id = "aten_rms_norm_default_1", prov.dispatch_id = "aten_rms_norm_default_1"} : (tensor<1x67x2048xf32>, tensor<2048xf32>) -> tensor<1x67x2048xf32>
      %413 = tensor.empty() : tensor<2048x2048xf32>
      %414 = linalg.transpose ins(%18:tensor<2048x2048xf32>) outs(%413:tensor<2048x2048xf32>) permutation = [1, 0]
      %415 = tensor.empty() : tensor<1x67x2048xf32>
      %416 = arith.constant 0.000000e+00 : f32
      %417 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%416 : f32) outs(%415 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %418 = linalg.matmul {prov.region_id = "matmul_8", prov.dispatch_id = "matmul_8", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%412, %414 : tensor<1x67x2048xf32>, tensor<2048x2048xf32>) outs(%417 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %419 = tensor.empty() : tensor<1x67x2048xf32>
      %420 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%418, %19 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%419 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_11", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb35(%421: f32, %422: f32, %423: f32):
        %424 = arith.addf %421, %422 : f32
        linalg.yield %424 : f32
      } -> tensor<1x67x2048xf32>
      %425 = tensor.collapse_shape %420 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x67x2048xf32> into tensor<137216xf32>
      %426 = tensor.expand_shape %425 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 67, 32, 64] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<137216xf32> into tensor<1x67x32x64xf32>
      %427 = tensor.empty() : tensor<1x32x67x64xf32>
      %428 = linalg.transpose ins(%426:tensor<1x67x32x64xf32>) outs(%427:tensor<1x32x67x64xf32>) permutation = [0, 2, 1, 3]
      %429 = tensor.empty() : tensor<2048x4096xf32>
      %430 = linalg.transpose ins(%20:tensor<4096x2048xf32>) outs(%429:tensor<2048x4096xf32>) permutation = [1, 0]
      %431 = tensor.empty() : tensor<1x32x4096xf32>
      %432 = arith.constant 0.000000e+00 : f32
      %433 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%432 : f32) outs(%431 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
      %434 = linalg.matmul {prov.region_id = "matmul_9", prov.dispatch_id = "matmul_9", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%296, %430 : tensor<1x32x2048xf32>, tensor<2048x4096xf32>) outs(%433 : tensor<1x32x4096xf32>) -> tensor<1x32x4096xf32>
      %435 = tensor.empty() : tensor<1x32x4096xf32>
      %436 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%434, %21 : tensor<1x32x4096xf32>, tensor<4096xf32>) outs(%435 : tensor<1x32x4096xf32>) attrs =  {prov.region_id = "add_12", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb36(%437: f32, %438: f32, %439: f32):
        %440 = arith.addf %437, %438 : f32
        linalg.yield %440 : f32
      } -> tensor<1x32x4096xf32>
      %441 = tensor.collapse_shape %436 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x32x4096xf32> into tensor<131072xf32>
      %442 = tensor.expand_shape %441 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 32, 2, 32, 64] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<131072xf32> into tensor<1x32x2x32x64xf32>
      %443 = tensor.empty() : tensor<2x1x32x32x64xf32>
      %444 = linalg.transpose ins(%442:tensor<1x32x2x32x64xf32>) outs(%443:tensor<2x1x32x32x64xf32>) permutation = [2, 0, 3, 1, 4]
      %445 = "tensor.extract_slice"(%444) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<2x1x32x32x64xf32>) -> tensor<1x1x32x32x64xf32>
      %446 = tensor.collapse_shape %445 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x1x32x32x64xf32> into tensor<65536xf32>
      %447 = tensor.expand_shape %446 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 32, 64] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<65536xf32> into tensor<1x32x32x64xf32>
      %448 = "tensor.extract_slice"(%444) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 32, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<2x1x32x32x64xf32>) -> tensor<1x1x32x32x64xf32>
      %449 = tensor.collapse_shape %448 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x1x32x32x64xf32> into tensor<65536xf32>
      %450 = tensor.expand_shape %449 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 32, 64] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<65536xf32> into tensor<1x32x32x64xf32>
      %451 = func.call @aten_rms_norm_default_1_wl2(%428, %22) {prov.region_id = "aten_rms_norm_default_1_2", prov.dispatch_id = "aten_rms_norm_default_1_2"} : (tensor<1x32x67x64xf32>, tensor<64xf32>) -> tensor<1x32x67x64xf32>
      %452 = func.call @aten_rms_norm_default_2_wl3(%447, %23) {prov.region_id = "aten_rms_norm_default_2_0", prov.dispatch_id = "aten_rms_norm_default_2_0"} : (tensor<1x32x32x64xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>
      %453 = tensor.collapse_shape %72 [[0 : i64, 1 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "bool"} : tensor<1x32xi1> into tensor<32xi1>
      %454 = tensor.expand_shape %453 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1, 1, 32] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "bool"} : tensor<32xi1> into tensor<1x1x1x32xi1>
      %455 = tensor.empty() : tensor<1x1x67x32xi1>
      %456 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%454 : tensor<1x1x1x32xi1>) outs(%455 : tensor<1x1x67x32xi1>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "bool"} {
      ^bb37(%457: i1, %458: i1):
        linalg.yield %457 : i1
      } -> tensor<1x1x67x32xi1>
      %459 = arith.constant {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %460 = tensor.splat %459 {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67x32xf32>
      %461 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%451, %452 : tensor<1x32x67x64xf32>, tensor<1x32x32x64xf32>) outs(%460 : tensor<1x32x67x32xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb38(%462: f32, %463: f32, %464: f32):
        %465 = arith.mulf %462, %463 : f32
        %466 = arith.addf %464, %465 : f32
        linalg.yield %466 : f32
      } -> tensor<1x32x67x32xf32>
      %467 = tensor.empty() : tensor<1x32x67x32xf32>
      %468 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%461 : tensor<1x32x67x32xf32>) outs(%467 : tensor<1x32x67x32xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb39(%469: f32, %470: f32):
        %471 = arith.constant 1.250000e-01 : f32
        %472 = arith.mulf %469, %471 : f32
        linalg.yield %472 : f32
      } -> tensor<1x32x67x32xf32>
      %473 = tensor.empty() : tensor<1x32x67x32xf32>
      %474 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%468, %456 : tensor<1x32x67x32xf32>, tensor<1x1x67x32xi1>) outs(%473 : tensor<1x32x67x32xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb40(%475: f32, %476: i1, %477: f32):
        %478 = arith.constant 0xff800000 : f32
        %479 = arith.select %476, %475, %478 : f32
        linalg.yield %479 : f32
      } -> tensor<1x32x67x32xf32>
      %480 = arith.constant {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %481 = tensor.splat %480 {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67xf32>
      %482 = linalg.reduce ins(%474:tensor<1x32x67x32xf32>) outs(%481:tensor<1x32x67xf32>) dimensions = [3]
      (%483: f32, %484: f32) {
        %485 = arith.maximumf %483, %484 : f32
        linalg.yield %485 : f32
      }
      %486 = tensor.collapse_shape %482 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67xf32> into tensor<2144xf32>
      %487 = tensor.expand_shape %486 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
      %488 = tensor.empty() : tensor<1x32x67x32xf32>
      %489 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%474, %487 : tensor<1x32x67x32xf32>, tensor<1x32x67x1xf32>) outs(%488 : tensor<1x32x67x32xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb41(%490: f32, %491: f32, %492: f32):
        %493 = arith.subf %490, %491 : f32
        linalg.yield %493 : f32
      } -> tensor<1x32x67x32xf32>
      %494 = tensor.empty() : tensor<1x32x67x32xf32>
      %495 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%489 : tensor<1x32x67x32xf32>) outs(%494 : tensor<1x32x67x32xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb42(%496: f32, %497: f32):
        %498 = math.exp %496 : f32
        linalg.yield %498 : f32
      } -> tensor<1x32x67x32xf32>
      %499 = arith.constant {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %500 = tensor.splat %499 {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67xf32>
      %501 = linalg.reduce ins(%495:tensor<1x32x67x32xf32>) outs(%500:tensor<1x32x67xf32>) dimensions = [3]
      (%502: f32, %503: f32) {
        %504 = arith.addf %502, %503 : f32
        linalg.yield %504 : f32
      }
      %505 = tensor.collapse_shape %501 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67xf32> into tensor<2144xf32>
      %506 = tensor.expand_shape %505 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
      %507 = tensor.empty() : tensor<1x32x67x32xf32>
      %508 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%495, %506 : tensor<1x32x67x32xf32>, tensor<1x32x67x1xf32>) outs(%507 : tensor<1x32x67x32xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb43(%509: f32, %510: f32, %511: f32):
        %512 = arith.divf %509, %510 : f32
        linalg.yield %512 : f32
      } -> tensor<1x32x67x32xf32>
      %513 = arith.constant {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %514 = tensor.splat %513 {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67x64xf32>
      %515 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%508, %450 : tensor<1x32x67x32xf32>, tensor<1x32x32x64xf32>) outs(%514 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "attention_1", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb44(%516: f32, %517: f32, %518: f32):
        %519 = arith.mulf %516, %517 : f32
        %520 = arith.addf %518, %519 : f32
        linalg.yield %520 : f32
      } -> tensor<1x32x67x64xf32>
      %521 = tensor.empty() : tensor<1x67x32x64xf32>
      %522 = linalg.transpose ins(%515:tensor<1x32x67x64xf32>) outs(%521:tensor<1x67x32x64xf32>) permutation = [0, 2, 1, 3]
      %523 = tensor.collapse_shape %522 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x67x32x64xf32> into tensor<137216xf32>
      %524 = tensor.expand_shape %523 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<137216xf32> into tensor<1x67x2048xf32>
      %525 = tensor.empty() : tensor<2048x2048xf32>
      %526 = linalg.transpose ins(%24:tensor<2048x2048xf32>) outs(%525:tensor<2048x2048xf32>) permutation = [1, 0]
      %527 = tensor.empty() : tensor<1x67x2048xf32>
      %528 = arith.constant 0.000000e+00 : f32
      %529 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%528 : f32) outs(%527 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %530 = linalg.matmul {prov.region_id = "matmul_11", prov.dispatch_id = "matmul_11", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%524, %526 : tensor<1x67x2048xf32>, tensor<2048x2048xf32>) outs(%529 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %531 = tensor.empty() : tensor<1x67x2048xf32>
      %532 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%530, %25 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%531 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_13", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb45(%533: f32, %534: f32, %535: f32):
        %536 = arith.addf %533, %534 : f32
        linalg.yield %536 : f32
      } -> tensor<1x67x2048xf32>
      %537 = tensor.empty() : tensor<1x67x2048xf32>
      %538 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%532, %407 : tensor<1x67x2048xf32>, tensor<1x67x2048xf32>) outs(%537 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb46(%539: f32, %540: f32, %541: f32):
        %542 = arith.addf %539, %540 : f32
        linalg.yield %542 : f32
      } -> tensor<1x67x2048xf32>
      %543 = func.call @aten_rms_norm_default_wl1(%538, %31) {prov.region_id = "aten_rms_norm_default_2", prov.dispatch_id = "aten_rms_norm_default_2"} : (tensor<1x67x2048xf32>, tensor<2048xf32>) -> tensor<1x67x2048xf32>
      %544 = tensor.empty() : tensor<2048x2048xf32>
      %545 = linalg.transpose ins(%27:tensor<2048x2048xf32>) outs(%544:tensor<2048x2048xf32>) permutation = [1, 0]
      %546 = tensor.empty() : tensor<1x67x2048xf32>
      %547 = arith.constant 0.000000e+00 : f32
      %548 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%547 : f32) outs(%546 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %549 = linalg.matmul {prov.region_id = "matmul_12", prov.dispatch_id = "matmul_12", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%543, %545 : tensor<1x67x2048xf32>, tensor<2048x2048xf32>) outs(%548 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %550 = tensor.empty() : tensor<1x67x2048xf32>
      %551 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%549, %28 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%550 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_15", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb47(%552: f32, %553: f32, %554: f32):
        %555 = arith.addf %552, %553 : f32
        linalg.yield %555 : f32
      } -> tensor<1x67x2048xf32>
      %556 = tensor.empty() : tensor<1x67x2048xf32>
      %557 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%551 : tensor<1x67x2048xf32>) outs(%556 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32"} {
      ^bb48(%558: f32, %559: f32):
        %560 = arith.constant 5.000000e-01 : f32
        %561 = arith.constant 1.000000e+00 : f32
        %562 = arith.constant 0.707106769 : f32
        %563 = arith.mulf %558, %562 : f32
        %564 = math.erf %563 : f32
        %565 = arith.addf %561, %564 : f32
        %566 = arith.mulf %560, %558 : f32
        %567 = arith.mulf %566, %565 : f32
        linalg.yield %567 : f32
      } -> tensor<1x67x2048xf32>
      %568 = tensor.empty() : tensor<2048x2048xf32>
      %569 = linalg.transpose ins(%29:tensor<2048x2048xf32>) outs(%568:tensor<2048x2048xf32>) permutation = [1, 0]
      %570 = tensor.empty() : tensor<1x67x2048xf32>
      %571 = arith.constant 0.000000e+00 : f32
      %572 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%571 : f32) outs(%570 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %573 = linalg.matmul {prov.region_id = "matmul_13", prov.dispatch_id = "matmul_13", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%557, %569 : tensor<1x67x2048xf32>, tensor<2048x2048xf32>) outs(%572 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %574 = tensor.empty() : tensor<1x67x2048xf32>
      %575 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%573, %30 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%574 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_16", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb49(%576: f32, %577: f32, %578: f32):
        %579 = arith.addf %576, %577 : f32
        linalg.yield %579 : f32
      } -> tensor<1x67x2048xf32>
      %580 = tensor.empty() : tensor<1x67x2048xf32>
      %581 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%575, %538 : tensor<1x67x2048xf32>, tensor<1x67x2048xf32>) outs(%580 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb50(%582: f32, %583: f32, %584: f32):
        %585 = arith.addf %582, %583 : f32
        linalg.yield %585 : f32
      } -> tensor<1x67x2048xf32>
      %586 = func.call @aten_rms_norm_default_wl1(%581, %32) {prov.region_id = "aten_rms_norm_default_3", prov.dispatch_id = "aten_rms_norm_default_3"} : (tensor<1x67x2048xf32>, tensor<2048xf32>) -> tensor<1x67x2048xf32>
      %587 = tensor.empty() : tensor<2048x6144xf32>
      %588 = linalg.transpose ins(%33:tensor<6144x2048xf32>) outs(%587:tensor<2048x6144xf32>) permutation = [1, 0]
      %589 = tensor.empty() : tensor<1x67x6144xf32>
      %590 = arith.constant 0.000000e+00 : f32
      %591 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%590 : f32) outs(%589 : tensor<1x67x6144xf32>) -> tensor<1x67x6144xf32>
      %592 = linalg.matmul {prov.region_id = "matmul_14", prov.dispatch_id = "matmul_14", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%586, %588 : tensor<1x67x2048xf32>, tensor<2048x6144xf32>) outs(%591 : tensor<1x67x6144xf32>) -> tensor<1x67x6144xf32>
      %593 = tensor.empty() : tensor<1x67x6144xf32>
      %594 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%592, %34 : tensor<1x67x6144xf32>, tensor<6144xf32>) outs(%593 : tensor<1x67x6144xf32>) attrs =  {prov.region_id = "add_18", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb51(%595: f32, %596: f32, %597: f32):
        %598 = arith.addf %595, %596 : f32
        linalg.yield %598 : f32
      } -> tensor<1x67x6144xf32>
      %599 = tensor.collapse_shape %594 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x67x6144xf32> into tensor<411648xf32>
      %600 = tensor.expand_shape %599 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 67, 3, 32, 64] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<411648xf32> into tensor<1x67x3x32x64xf32>
      %601 = tensor.empty() : tensor<3x1x32x67x64xf32>
      %602 = linalg.transpose ins(%600:tensor<1x67x3x32x64xf32>) outs(%601:tensor<3x1x32x67x64xf32>) permutation = [2, 0, 3, 1, 4]
      %603 = "tensor.extract_slice"(%602) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 67, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<3x1x32x67x64xf32>) -> tensor<1x1x32x67x64xf32>
      %604 = tensor.collapse_shape %603 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x1x32x67x64xf32> into tensor<137216xf32>
      %605 = tensor.expand_shape %604 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 64] {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<137216xf32> into tensor<1x32x67x64xf32>
      %606 = "tensor.extract_slice"(%602) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 67, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<3x1x32x67x64xf32>) -> tensor<1x1x32x67x64xf32>
      %607 = tensor.collapse_shape %606 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x1x32x67x64xf32> into tensor<137216xf32>
      %608 = tensor.expand_shape %607 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 64] {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<137216xf32> into tensor<1x32x67x64xf32>
      %609 = "tensor.extract_slice"(%602) <{static_offsets = array<i64: 2, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 67, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<3x1x32x67x64xf32>) -> tensor<1x1x32x67x64xf32>
      %610 = tensor.collapse_shape %609 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x1x32x67x64xf32> into tensor<137216xf32>
      %611 = tensor.expand_shape %610 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 64] {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<137216xf32> into tensor<1x32x67x64xf32>
      %612 = func.call @aten_rms_norm_default_1_wl2(%605, %35) {prov.region_id = "aten_rms_norm_default_1_3", prov.dispatch_id = "aten_rms_norm_default_1_3"} : (tensor<1x32x67x64xf32>, tensor<64xf32>) -> tensor<1x32x67x64xf32>
      %613 = func.call @aten_rms_norm_default_1_wl2(%608, %36) {prov.region_id = "aten_rms_norm_default_1_4", prov.dispatch_id = "aten_rms_norm_default_1_4"} : (tensor<1x32x67x64xf32>, tensor<64xf32>) -> tensor<1x32x67x64xf32>
      %614 = arith.constant {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %615 = tensor.splat %614 {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67x67xf32>
      %616 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%612, %613 : tensor<1x32x67x64xf32>, tensor<1x32x67x64xf32>) outs(%615 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb52(%617: f32, %618: f32, %619: f32):
        %620 = arith.mulf %617, %618 : f32
        %621 = arith.addf %619, %620 : f32
        linalg.yield %621 : f32
      } -> tensor<1x32x67x67xf32>
      %622 = tensor.empty() : tensor<1x32x67x67xf32>
      %623 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%616 : tensor<1x32x67x67xf32>) outs(%622 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb53(%624: f32, %625: f32):
        %626 = arith.constant 1.250000e-01 : f32
        %627 = arith.mulf %624, %626 : f32
        linalg.yield %627 : f32
      } -> tensor<1x32x67x67xf32>
      %628 = arith.constant {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %629 = tensor.splat %628 {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67xf32>
      %630 = linalg.reduce ins(%623:tensor<1x32x67x67xf32>) outs(%629:tensor<1x32x67xf32>) dimensions = [3]
      (%631: f32, %632: f32) {
        %633 = arith.maximumf %631, %632 : f32
        linalg.yield %633 : f32
      }
      %634 = tensor.collapse_shape %630 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67xf32> into tensor<2144xf32>
      %635 = tensor.expand_shape %634 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
      %636 = tensor.empty() : tensor<1x32x67x67xf32>
      %637 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%623, %635 : tensor<1x32x67x67xf32>, tensor<1x32x67x1xf32>) outs(%636 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb54(%638: f32, %639: f32, %640: f32):
        %641 = arith.subf %638, %639 : f32
        linalg.yield %641 : f32
      } -> tensor<1x32x67x67xf32>
      %642 = tensor.empty() : tensor<1x32x67x67xf32>
      %643 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%637 : tensor<1x32x67x67xf32>) outs(%642 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb55(%644: f32, %645: f32):
        %646 = math.exp %644 : f32
        linalg.yield %646 : f32
      } -> tensor<1x32x67x67xf32>
      %647 = arith.constant {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %648 = tensor.splat %647 {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67xf32>
      %649 = linalg.reduce ins(%643:tensor<1x32x67x67xf32>) outs(%648:tensor<1x32x67xf32>) dimensions = [3]
      (%650: f32, %651: f32) {
        %652 = arith.addf %650, %651 : f32
        linalg.yield %652 : f32
      }
      %653 = tensor.collapse_shape %649 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67xf32> into tensor<2144xf32>
      %654 = tensor.expand_shape %653 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
      %655 = tensor.empty() : tensor<1x32x67x67xf32>
      %656 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%643, %654 : tensor<1x32x67x67xf32>, tensor<1x32x67x1xf32>) outs(%655 : tensor<1x32x67x67xf32>) attrs =  {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb56(%657: f32, %658: f32, %659: f32):
        %660 = arith.divf %657, %658 : f32
        linalg.yield %660 : f32
      } -> tensor<1x32x67x67xf32>
      %661 = arith.constant {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %662 = tensor.splat %661 {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67x64xf32>
      %663 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%656, %611 : tensor<1x32x67x67xf32>, tensor<1x32x67x64xf32>) outs(%662 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "attention_2", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb57(%664: f32, %665: f32, %666: f32):
        %667 = arith.mulf %664, %665 : f32
        %668 = arith.addf %666, %667 : f32
        linalg.yield %668 : f32
      } -> tensor<1x32x67x64xf32>
      %669 = tensor.empty() : tensor<1x67x32x64xf32>
      %670 = linalg.transpose ins(%663:tensor<1x32x67x64xf32>) outs(%669:tensor<1x67x32x64xf32>) permutation = [0, 2, 1, 3]
      %671 = tensor.collapse_shape %670 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x67x32x64xf32> into tensor<137216xf32>
      %672 = tensor.expand_shape %671 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<137216xf32> into tensor<1x67x2048xf32>
      %673 = tensor.empty() : tensor<2048x2048xf32>
      %674 = linalg.transpose ins(%37:tensor<2048x2048xf32>) outs(%673:tensor<2048x2048xf32>) permutation = [1, 0]
      %675 = tensor.empty() : tensor<1x67x2048xf32>
      %676 = arith.constant 0.000000e+00 : f32
      %677 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%676 : f32) outs(%675 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %678 = linalg.matmul {prov.region_id = "matmul_16", prov.dispatch_id = "matmul_16", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%672, %674 : tensor<1x67x2048xf32>, tensor<2048x2048xf32>) outs(%677 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %679 = tensor.empty() : tensor<1x67x2048xf32>
      %680 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%678, %38 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%679 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_19", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb58(%681: f32, %682: f32, %683: f32):
        %684 = arith.addf %681, %682 : f32
        linalg.yield %684 : f32
      } -> tensor<1x67x2048xf32>
      %685 = tensor.empty() : tensor<1x67x2048xf32>
      %686 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%680, %581 : tensor<1x67x2048xf32>, tensor<1x67x2048xf32>) outs(%685 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb59(%687: f32, %688: f32, %689: f32):
        %690 = arith.addf %687, %688 : f32
        linalg.yield %690 : f32
      } -> tensor<1x67x2048xf32>
      %691 = func.call @aten_rms_norm_default_wl1(%686, %47) {prov.region_id = "aten_rms_norm_default_4", prov.dispatch_id = "aten_rms_norm_default_4"} : (tensor<1x67x2048xf32>, tensor<2048xf32>) -> tensor<1x67x2048xf32>
      %692 = tensor.empty() : tensor<2048x2048xf32>
      %693 = linalg.transpose ins(%39:tensor<2048x2048xf32>) outs(%692:tensor<2048x2048xf32>) permutation = [1, 0]
      %694 = tensor.empty() : tensor<1x67x2048xf32>
      %695 = arith.constant 0.000000e+00 : f32
      %696 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%695 : f32) outs(%694 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %697 = linalg.matmul {prov.region_id = "matmul_17", prov.dispatch_id = "matmul_17", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%691, %693 : tensor<1x67x2048xf32>, tensor<2048x2048xf32>) outs(%696 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %698 = tensor.empty() : tensor<1x67x2048xf32>
      %699 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%697, %40 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%698 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_21", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb60(%700: f32, %701: f32, %702: f32):
        %703 = arith.addf %700, %701 : f32
        linalg.yield %703 : f32
      } -> tensor<1x67x2048xf32>
      %704 = tensor.collapse_shape %699 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x67x2048xf32> into tensor<137216xf32>
      %705 = tensor.expand_shape %704 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 67, 32, 64] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<137216xf32> into tensor<1x67x32x64xf32>
      %706 = tensor.empty() : tensor<1x32x67x64xf32>
      %707 = linalg.transpose ins(%705:tensor<1x67x32x64xf32>) outs(%706:tensor<1x32x67x64xf32>) permutation = [0, 2, 1, 3]
      %708 = tensor.empty() : tensor<2048x4096xf32>
      %709 = linalg.transpose ins(%41:tensor<4096x2048xf32>) outs(%708:tensor<2048x4096xf32>) permutation = [1, 0]
      %710 = tensor.empty() : tensor<1x4096x4096xf32>
      %711 = arith.constant 0.000000e+00 : f32
      %712 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%711 : f32) outs(%710 : tensor<1x4096x4096xf32>) -> tensor<1x4096x4096xf32>
      %713 = linalg.matmul {prov.region_id = "matmul_18", prov.dispatch_id = "matmul_18", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%302, %709 : tensor<1x4096x2048xf32>, tensor<2048x4096xf32>) outs(%712 : tensor<1x4096x4096xf32>) -> tensor<1x4096x4096xf32>
      %714 = tensor.empty() : tensor<1x4096x4096xf32>
      %715 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%713, %42 : tensor<1x4096x4096xf32>, tensor<4096xf32>) outs(%714 : tensor<1x4096x4096xf32>) attrs =  {prov.region_id = "add_22", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb61(%716: f32, %717: f32, %718: f32):
        %719 = arith.addf %716, %717 : f32
        linalg.yield %719 : f32
      } -> tensor<1x4096x4096xf32>
      %720 = tensor.collapse_shape %715 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x4096x4096xf32> into tensor<16777216xf32>
      %721 = tensor.expand_shape %720 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 4096, 2, 32, 64] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<16777216xf32> into tensor<1x4096x2x32x64xf32>
      %722 = tensor.empty() : tensor<2x1x32x4096x64xf32>
      %723 = linalg.transpose ins(%721:tensor<1x4096x2x32x64xf32>) outs(%722:tensor<2x1x32x4096x64xf32>) permutation = [2, 0, 3, 1, 4]
      %724 = "tensor.extract_slice"(%723) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 4096, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<2x1x32x4096x64xf32>) -> tensor<1x1x32x4096x64xf32>
      %725 = tensor.collapse_shape %724 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x1x32x4096x64xf32> into tensor<8388608xf32>
      %726 = tensor.expand_shape %725 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4096, 64] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<8388608xf32> into tensor<1x32x4096x64xf32>
      %727 = "tensor.extract_slice"(%723) <{static_offsets = array<i64: 1, 0, 0, 0, 0>, static_sizes = array<i64: 1, 1, 32, 4096, 64>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<2x1x32x4096x64xf32>) -> tensor<1x1x32x4096x64xf32>
      %728 = tensor.collapse_shape %727 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x1x32x4096x64xf32> into tensor<8388608xf32>
      %729 = tensor.expand_shape %728 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 4096, 64] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<8388608xf32> into tensor<1x32x4096x64xf32>
      %730 = func.call @aten_rms_norm_default_1_wl2(%707, %43) {prov.region_id = "aten_rms_norm_default_1_5", prov.dispatch_id = "aten_rms_norm_default_1_5"} : (tensor<1x32x67x64xf32>, tensor<64xf32>) -> tensor<1x32x67x64xf32>
      %731 = func.call @aten_rms_norm_default_3_wl4(%726, %44) {prov.region_id = "aten_rms_norm_default_3_0", prov.dispatch_id = "aten_rms_norm_default_3_0"} : (tensor<1x32x4096x64xf32>, tensor<64xf32>) -> tensor<1x32x4096x64xf32>
      %732 = arith.constant {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %733 = tensor.splat %732 {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67x4096xf32>
      %734 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%730, %731 : tensor<1x32x67x64xf32>, tensor<1x32x4096x64xf32>) outs(%733 : tensor<1x32x67x4096xf32>) attrs =  {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb62(%735: f32, %736: f32, %737: f32):
        %738 = arith.mulf %735, %736 : f32
        %739 = arith.addf %737, %738 : f32
        linalg.yield %739 : f32
      } -> tensor<1x32x67x4096xf32>
      %740 = tensor.empty() : tensor<1x32x67x4096xf32>
      %741 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%734 : tensor<1x32x67x4096xf32>) outs(%740 : tensor<1x32x67x4096xf32>) attrs =  {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb63(%742: f32, %743: f32):
        %744 = arith.constant 1.250000e-01 : f32
        %745 = arith.mulf %742, %744 : f32
        linalg.yield %745 : f32
      } -> tensor<1x32x67x4096xf32>
      %746 = arith.constant {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0xff800000 : f32
      %747 = tensor.splat %746 {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67xf32>
      %748 = linalg.reduce ins(%741:tensor<1x32x67x4096xf32>) outs(%747:tensor<1x32x67xf32>) dimensions = [3]
      (%749: f32, %750: f32) {
        %751 = arith.maximumf %749, %750 : f32
        linalg.yield %751 : f32
      }
      %752 = tensor.collapse_shape %748 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67xf32> into tensor<2144xf32>
      %753 = tensor.expand_shape %752 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
      %754 = tensor.empty() : tensor<1x32x67x4096xf32>
      %755 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%741, %753 : tensor<1x32x67x4096xf32>, tensor<1x32x67x1xf32>) outs(%754 : tensor<1x32x67x4096xf32>) attrs =  {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb64(%756: f32, %757: f32, %758: f32):
        %759 = arith.subf %756, %757 : f32
        linalg.yield %759 : f32
      } -> tensor<1x32x67x4096xf32>
      %760 = tensor.empty() : tensor<1x32x67x4096xf32>
      %761 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%755 : tensor<1x32x67x4096xf32>) outs(%760 : tensor<1x32x67x4096xf32>) attrs =  {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb65(%762: f32, %763: f32):
        %764 = math.exp %762 : f32
        linalg.yield %764 : f32
      } -> tensor<1x32x67x4096xf32>
      %765 = arith.constant {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %766 = tensor.splat %765 {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67xf32>
      %767 = linalg.reduce ins(%761:tensor<1x32x67x4096xf32>) outs(%766:tensor<1x32x67xf32>) dimensions = [3]
      (%768: f32, %769: f32) {
        %770 = arith.addf %768, %769 : f32
        linalg.yield %770 : f32
      }
      %771 = tensor.collapse_shape %767 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67xf32> into tensor<2144xf32>
      %772 = tensor.expand_shape %771 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 32, 67, 1] {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<2144xf32> into tensor<1x32x67x1xf32>
      %773 = tensor.empty() : tensor<1x32x67x4096xf32>
      %774 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%761, %772 : tensor<1x32x67x4096xf32>, tensor<1x32x67x1xf32>) outs(%773 : tensor<1x32x67x4096xf32>) attrs =  {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb66(%775: f32, %776: f32, %777: f32):
        %778 = arith.divf %775, %776 : f32
        linalg.yield %778 : f32
      } -> tensor<1x32x67x4096xf32>
      %779 = arith.constant {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %780 = tensor.splat %779 {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} : tensor<1x32x67x64xf32>
      %781 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%774, %729 : tensor<1x32x67x4096xf32>, tensor<1x32x4096x64xf32>) outs(%780 : tensor<1x32x67x64xf32>) attrs =  {prov.region_id = "attention_3", prov.family = "attention", prov._pattern_hint = "sdpa", prov.op = "sdpa", prov.aten = "aten.scaled_dot_product_attention.default", prov.orig_dtype = "float32"} {
      ^bb67(%782: f32, %783: f32, %784: f32):
        %785 = arith.mulf %782, %783 : f32
        %786 = arith.addf %784, %785 : f32
        linalg.yield %786 : f32
      } -> tensor<1x32x67x64xf32>
      %787 = tensor.empty() : tensor<1x67x32x64xf32>
      %788 = linalg.transpose ins(%781:tensor<1x32x67x64xf32>) outs(%787:tensor<1x67x32x64xf32>) permutation = [0, 2, 1, 3]
      %789 = tensor.collapse_shape %788 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x67x32x64xf32> into tensor<137216xf32>
      %790 = tensor.expand_shape %789 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 67, 2048] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<137216xf32> into tensor<1x67x2048xf32>
      %791 = tensor.empty() : tensor<2048x2048xf32>
      %792 = linalg.transpose ins(%45:tensor<2048x2048xf32>) outs(%791:tensor<2048x2048xf32>) permutation = [1, 0]
      %793 = tensor.empty() : tensor<1x67x2048xf32>
      %794 = arith.constant 0.000000e+00 : f32
      %795 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%794 : f32) outs(%793 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %796 = linalg.matmul {prov.region_id = "matmul_20", prov.dispatch_id = "matmul_20", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%790, %792 : tensor<1x67x2048xf32>, tensor<2048x2048xf32>) outs(%795 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %797 = tensor.empty() : tensor<1x67x2048xf32>
      %798 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%796, %46 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%797 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_23", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb68(%799: f32, %800: f32, %801: f32):
        %802 = arith.addf %799, %800 : f32
        linalg.yield %802 : f32
      } -> tensor<1x67x2048xf32>
      %803 = tensor.empty() : tensor<1x67x2048xf32>
      %804 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%798, %686 : tensor<1x67x2048xf32>, tensor<1x67x2048xf32>) outs(%803 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb69(%805: f32, %806: f32, %807: f32):
        %808 = arith.addf %805, %806 : f32
        linalg.yield %808 : f32
      } -> tensor<1x67x2048xf32>
      %809 = func.call @aten_rms_norm_default_wl1(%804, %52) {prov.region_id = "aten_rms_norm_default_5", prov.dispatch_id = "aten_rms_norm_default_5"} : (tensor<1x67x2048xf32>, tensor<2048xf32>) -> tensor<1x67x2048xf32>
      %810 = tensor.empty() : tensor<2048x2048xf32>
      %811 = linalg.transpose ins(%48:tensor<2048x2048xf32>) outs(%810:tensor<2048x2048xf32>) permutation = [1, 0]
      %812 = tensor.empty() : tensor<1x67x2048xf32>
      %813 = arith.constant 0.000000e+00 : f32
      %814 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%813 : f32) outs(%812 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %815 = linalg.matmul {prov.region_id = "matmul_21", prov.dispatch_id = "matmul_21", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%809, %811 : tensor<1x67x2048xf32>, tensor<2048x2048xf32>) outs(%814 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %816 = tensor.empty() : tensor<1x67x2048xf32>
      %817 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%815, %49 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%816 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_25", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb70(%818: f32, %819: f32, %820: f32):
        %821 = arith.addf %818, %819 : f32
        linalg.yield %821 : f32
      } -> tensor<1x67x2048xf32>
      %822 = tensor.empty() : tensor<1x67x2048xf32>
      %823 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%817 : tensor<1x67x2048xf32>) outs(%822 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "gelu_1", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32"} {
      ^bb71(%824: f32, %825: f32):
        %826 = arith.constant 5.000000e-01 : f32
        %827 = arith.constant 1.000000e+00 : f32
        %828 = arith.constant 0.707106769 : f32
        %829 = arith.mulf %824, %828 : f32
        %830 = math.erf %829 : f32
        %831 = arith.addf %827, %830 : f32
        %832 = arith.mulf %826, %824 : f32
        %833 = arith.mulf %832, %831 : f32
        linalg.yield %833 : f32
      } -> tensor<1x67x2048xf32>
      %834 = tensor.empty() : tensor<2048x2048xf32>
      %835 = linalg.transpose ins(%50:tensor<2048x2048xf32>) outs(%834:tensor<2048x2048xf32>) permutation = [1, 0]
      %836 = tensor.empty() : tensor<1x67x2048xf32>
      %837 = arith.constant 0.000000e+00 : f32
      %838 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%837 : f32) outs(%836 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %839 = linalg.matmul {prov.region_id = "matmul_22", prov.dispatch_id = "matmul_22", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%823, %835 : tensor<1x67x2048xf32>, tensor<2048x2048xf32>) outs(%838 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %840 = tensor.empty() : tensor<1x67x2048xf32>
      %841 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%839, %51 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%840 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_26", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb72(%842: f32, %843: f32, %844: f32):
        %845 = arith.addf %842, %843 : f32
        linalg.yield %845 : f32
      } -> tensor<1x67x2048xf32>
      %846 = tensor.empty() : tensor<1x67x2048xf32>
      %847 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%841, %804 : tensor<1x67x2048xf32>, tensor<1x67x2048xf32>) outs(%846 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb73(%848: f32, %849: f32, %850: f32):
        %851 = arith.addf %848, %849 : f32
        linalg.yield %851 : f32
      } -> tensor<1x67x2048xf32>
      %852 = func.call @aten_rms_norm_default_wl1(%847, %53) {prov.region_id = "aten_rms_norm_default_6", prov.dispatch_id = "aten_rms_norm_default_6"} : (tensor<1x67x2048xf32>, tensor<2048xf32>) -> tensor<1x67x2048xf32>
      %853 = tensor.empty() : tensor<2048x2048xf32>
      %854 = linalg.transpose ins(%54:tensor<2048x2048xf32>) outs(%853:tensor<2048x2048xf32>) permutation = [1, 0]
      %855 = tensor.empty() : tensor<1x67x2048xf32>
      %856 = arith.constant 0.000000e+00 : f32
      %857 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%856 : f32) outs(%855 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %858 = linalg.matmul {prov.region_id = "matmul_23", prov.dispatch_id = "matmul_23", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%852, %854 : tensor<1x67x2048xf32>, tensor<2048x2048xf32>) outs(%857 : tensor<1x67x2048xf32>) -> tensor<1x67x2048xf32>
      %859 = tensor.empty() : tensor<1x67x2048xf32>
      %860 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%858, %55 : tensor<1x67x2048xf32>, tensor<2048xf32>) outs(%859 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "add_28", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb74(%861: f32, %862: f32, %863: f32):
        %864 = arith.addf %861, %862 : f32
        linalg.yield %864 : f32
      } -> tensor<1x67x2048xf32>
      %865 = tensor.empty() : tensor<1x67x2048xf32>
      %866 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%860 : tensor<1x67x2048xf32>) outs(%865 : tensor<1x67x2048xf32>) attrs =  {prov.region_id = "gelu_2", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32"} {
      ^bb75(%867: f32, %868: f32):
        %869 = arith.constant 5.000000e-01 : f32
        %870 = arith.constant 1.000000e+00 : f32
        %871 = arith.constant 0.707106769 : f32
        %872 = arith.mulf %867, %871 : f32
        %873 = math.erf %872 : f32
        %874 = arith.addf %870, %873 : f32
        %875 = arith.mulf %869, %867 : f32
        %876 = arith.mulf %875, %874 : f32
        linalg.yield %876 : f32
      } -> tensor<1x67x2048xf32>
      %877 = tensor.empty() : tensor<2048x128xf32>
      %878 = linalg.transpose ins(%56:tensor<128x2048xf32>) outs(%877:tensor<2048x128xf32>) permutation = [1, 0]
      %879 = tensor.empty() : tensor<1x67x128xf32>
      %880 = arith.constant 0.000000e+00 : f32
      %881 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%880 : f32) outs(%879 : tensor<1x67x128xf32>) -> tensor<1x67x128xf32>
      %882 = linalg.matmul {prov.region_id = "matmul_24", prov.dispatch_id = "matmul_24", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%866, %878 : tensor<1x67x2048xf32>, tensor<2048x128xf32>) outs(%881 : tensor<1x67x128xf32>) -> tensor<1x67x128xf32>
      %883 = tensor.empty() : tensor<1x67x128xf32>
      %884 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%882, %57 : tensor<1x67x128xf32>, tensor<128xf32>) outs(%883 : tensor<1x67x128xf32>) attrs =  {prov.region_id = "add_29", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb76(%885: f32, %886: f32, %887: f32):
        %888 = arith.addf %885, %886 : f32
        linalg.yield %888 : f32
      } -> tensor<1x67x128xf32>
      %889 = "tensor.extract_slice"(%884) <{static_offsets = array<i64: 0, 3, 0>, static_sizes = array<i64: 1, 64, 128>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x67x128xf32>) -> tensor<1x64x128xf32>
      %890 = tensor.extract %86[] : tensor<i64>
      %891 = tensor.from_elements %890 {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "int64"} : tensor<1xi64>
      %892 = func.call @aten_index_select_default_wl0(%60, %891) {prov.region_id = "aten_index_select_default_1", prov.dispatch_id = "aten_index_select_default_1"} : (tensor<5xf32>, tensor<1xi64>) -> tensor<1xf32>
      %893 = arith.constant {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} 0 : index
      %894 = tensor.extract %892[%893] : tensor<1xf32>
      %895 = tensor.from_elements %894 {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<f32>
      %896 = tensor.extract %86[] : tensor<i64>
      %897 = tensor.from_elements %896 {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "int64"} : tensor<1xi64>
      %898 = func.call @aten_index_select_default_wl0(%61, %897) {prov.region_id = "aten_index_select_default_2", prov.dispatch_id = "aten_index_select_default_2"} : (tensor<5xf32>, tensor<1xi64>) -> tensor<1xf32>
      %899 = arith.constant {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} 0 : index
      %900 = tensor.extract %898[%899] : tensor<1xf32>
      %901 = tensor.from_elements %900 {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<f32>
      %902 = tensor.extract %86[] : tensor<i64>
      %903 = tensor.from_elements %902 {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "int64"} : tensor<1xi64>
      %904 = func.call @aten_index_select_default_wl0(%62, %903) {prov.region_id = "aten_index_select_default_3", prov.dispatch_id = "aten_index_select_default_3"} : (tensor<5xf32>, tensor<1xi64>) -> tensor<1xf32>
      %905 = arith.constant {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} 0 : index
      %906 = tensor.extract %904[%905] : tensor<1xf32>
      %907 = tensor.from_elements %906 {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<f32>
      %908 = tensor.extract %86[] : tensor<i64>
      %909 = tensor.from_elements %908 {prov.region_id = "view_20", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "int64"} : tensor<1xi64>
      %910 = func.call @aten_index_select_default_wl0(%63, %909) {prov.region_id = "aten_index_select_default_4", prov.dispatch_id = "aten_index_select_default_4"} : (tensor<5xf32>, tensor<1xi64>) -> tensor<1xf32>
      %911 = arith.constant {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} 0 : index
      %912 = tensor.extract %910[%911] : tensor<1xf32>
      %913 = tensor.from_elements %912 {prov.region_id = "view_21", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<f32>
      %914 = tensor.extract %86[] : tensor<i64>
      %915 = tensor.from_elements %914 {prov.region_id = "view_22", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "int64"} : tensor<1xi64>
      %916 = func.call @aten_index_select_default_wl0(%64, %915) {prov.region_id = "aten_index_select_default_5", prov.dispatch_id = "aten_index_select_default_5"} : (tensor<5xf32>, tensor<1xi64>) -> tensor<1xf32>
      %917 = arith.constant {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} 0 : index
      %918 = tensor.extract %916[%917] : tensor<1xf32>
      %919 = tensor.from_elements %918 {prov.region_id = "view_23", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<f32>
      %920 = tensor.empty() : tensor<1x64x128xf32>
      %921 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%889, %88 : tensor<1x64x128xf32>, tensor<1x64x128xf32>) outs(%920 : tensor<1x64x128xf32>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} {
      ^bb77(%922: f32, %923: f32, %924: f32):
        %925 = arith.subf %922, %923 : f32
        linalg.yield %925 : f32
      } -> tensor<1x64x128xf32>
      %926 = tensor.empty() : tensor<1x64x128xf32>
      %927 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%913, %921 : tensor<f32>, tensor<1x64x128xf32>) outs(%926 : tensor<1x64x128xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb78(%928: f32, %929: f32, %930: f32):
        %931 = arith.mulf %928, %929 : f32
        linalg.yield %931 : f32
      } -> tensor<1x64x128xf32>
      %932 = tensor.empty() : tensor<1x64x128xf32>
      %933 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%895, %87 : tensor<f32>, tensor<1x64x128xf32>) outs(%932 : tensor<1x64x128xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb79(%934: f32, %935: f32, %936: f32):
        %937 = arith.mulf %934, %935 : f32
        linalg.yield %937 : f32
      } -> tensor<1x64x128xf32>
      %938 = tensor.empty() : tensor<1x64x128xf32>
      %939 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%901, %889 : tensor<f32>, tensor<1x64x128xf32>) outs(%938 : tensor<1x64x128xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb80(%940: f32, %941: f32, %942: f32):
        %943 = arith.mulf %940, %941 : f32
        linalg.yield %943 : f32
      } -> tensor<1x64x128xf32>
      %944 = tensor.empty() : tensor<1x64x128xf32>
      %945 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%933, %939 : tensor<1x64x128xf32>, tensor<1x64x128xf32>) outs(%944 : tensor<1x64x128xf32>) attrs =  {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb81(%946: f32, %947: f32, %948: f32):
        %949 = arith.addf %946, %947 : f32
        linalg.yield %949 : f32
      } -> tensor<1x64x128xf32>
      %950 = tensor.empty() : tensor<1x64x128xf32>
      %951 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%907, %927 : tensor<f32>, tensor<1x64x128xf32>) outs(%950 : tensor<1x64x128xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb82(%952: f32, %953: f32, %954: f32):
        %955 = arith.mulf %952, %953 : f32
        linalg.yield %955 : f32
      } -> tensor<1x64x128xf32>
      %956 = tensor.empty() : tensor<1x64x128xf32>
      %957 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%919, %951 : tensor<f32>, tensor<1x64x128xf32>) outs(%956 : tensor<1x64x128xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb83(%958: f32, %959: f32, %960: f32):
        %961 = arith.mulf %958, %959 : f32
        linalg.yield %961 : f32
      } -> tensor<1x64x128xf32>
      %962 = tensor.empty() : tensor<1x64x128xf32>
      %963 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%945, %957 : tensor<1x64x128xf32>, tensor<1x64x128xf32>) outs(%962 : tensor<1x64x128xf32>) attrs =  {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb84(%964: f32, %965: f32, %966: f32):
        %967 = arith.addf %964, %965 : f32
        linalg.yield %967 : f32
      } -> tensor<1x64x128xf32>
      %968 = arith.constant {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} 1 : i64
      %969 = tensor.splat %968 {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} : tensor<i64>
      %970 = tensor.empty() : tensor<i64>
      %971 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%86, %969 : tensor<i64>, tensor<i64>) outs(%970 : tensor<i64>) attrs =  {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb85(%972: i64, %973: i64, %974: i64):
        %975 = arith.addi %972, %973 : i64
        linalg.yield %975 : i64
      } -> tensor<i64>
      scf.yield %971, %963, %889 : tensor<i64>, tensor<1x64x128xf32>, tensor<1x64x128xf32>
    }
    %976 = tensor.empty() : tensor<1x64x128xf32>
    %977 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83, %74 : tensor<1x64x128xf32>, tensor<1x64x128xf32>) outs(%976 : tensor<1x64x128xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb86(%978: f32, %979: f32, %980: f32):
      %981 = arith.mulf %978, %979 : f32
      linalg.yield %981 : f32
    } -> tensor<1x64x128xf32>
    func.return %977 : tensor<1x64x128xf32>
  }
}
