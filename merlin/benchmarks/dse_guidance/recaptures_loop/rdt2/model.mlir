builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func private @aten_zeros_default() -> tensor<i64>
  func.func private @aten_type_as_default_wl0(tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
  func.func private @aten_type_as_default_1_wl1(tensor<1x28x8x128xf32>, tensor<1x28x8x128xf32>) -> tensor<1x28x8x128xf32>
  func.func private @aten_type_as_default_2_wl2(tensor<1x28x4x128xf32>, tensor<1x28x4x128xf32>) -> tensor<1x28x4x128xf32>
  func.func private @aten_type_as_default_3_wl3(tensor<1x8x28x28xf32>, tensor<1x8x28x128xf32>) -> tensor<1x8x28x28xf32>
  func.func private @aten_type_as_default_4_wl4(tensor<1x8x28x64xf32>, tensor<1x8x28x128xf32>) -> tensor<1x8x28x64xf32>
  func.func @forward(%0: tensor<1x4x1024xf32>, %1: tensor<1x28x1024xf32>, %2: tensor<1x1x1024xf32>, %3: tensor<1024x256xf32>, %4: tensor<1024xf32>, %5: tensor<1024x1024xf32>, %6: tensor<1024xf32>, %7: tensor<1024xf32>, %8: tensor<1024x1024xf32>, %9: tensor<1024x1024xf32>, %10: tensor<1024x1024xf32>, %11: tensor<128xf32>, %12: tensor<128xf32>, %13: tensor<1024xf32>, %14: tensor<1024xf32>, %15: tensor<1024x1024xf32>, %16: tensor<1024x1024xf32>, %17: tensor<1024x1024xf32>, %18: tensor<128xf32>, %19: tensor<128xf32>, %20: tensor<1024xf32>, %21: tensor<2816x1024xf32>, %22: tensor<1024x2816xf32>, %23: tensor<2816x1024xf32>, %24: tensor<9216x2048xf32>, %25: tensor<9216xf32>, %26: tensor<1024xf32>, %27: tensor<1024x1024xf32>, %28: tensor<1024x1024xf32>, %29: tensor<1024x1024xf32>, %30: tensor<128xf32>, %31: tensor<128xf32>, %32: tensor<1024xf32>, %33: tensor<1024xf32>, %34: tensor<1024x1024xf32>, %35: tensor<1024x1024xf32>, %36: tensor<1024x1024xf32>, %37: tensor<128xf32>, %38: tensor<128xf32>, %39: tensor<1024xf32>, %40: tensor<2816x1024xf32>, %41: tensor<1024x2816xf32>, %42: tensor<2816x1024xf32>, %43: tensor<9216x2048xf32>, %44: tensor<9216xf32>, %45: tensor<1024xf32>, %46: tensor<4096x1024xf32>, %47: tensor<4096xf32>, %48: tensor<20x4096xf32>, %49: tensor<20xf32>, %50: tensor<2048x2048xf32>, %51: tensor<2048xf32>, %52: tensor<1024x20xf32>, %53: tensor<1024xf32>, %54: tensor<1024x1024xf32>, %55: tensor<1024xf32>, %56: tensor<1024x1024xf32>, %57: tensor<1024xf32>, %58: tensor<1x24x20xf32>, %59: tensor<1x1x1024xf32>, %60: tensor<1x4x64x128xf32>, %61: tensor<1x4x64x128xf32>, %62: tensor<1x4x64x128xf32>, %63: tensor<1x4x64x128xf32>) -> tensor<1x24x20xf32> {
    %64 = tensor.empty() : tensor<128xf32>
    %65 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%64 : tensor<128xf32>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start", prov.orig_dtype = "float32"} {
    ^bb0(%66: f32):
      %67 = linalg.index 0 : index
      %68 = arith.index_cast %67 : index to i64
      %69 = arith.sitofp %68 : i64 to f32
      %70 = arith.constant 1.000000e+00 : f32
      %71 = arith.mulf %69, %70 : f32
      %72 = arith.constant 0.000000e+00 : f32
      %73 = arith.addf %72, %71 : f32
      linalg.yield %73 : f32
    } -> tensor<128xf32>
    %74 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} -9.2103405 : f32
    %75 = tensor.splat %74 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<128xf32>
    %76 = tensor.empty() : tensor<128xf32>
    %77 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%65, %75 : tensor<128xf32>, tensor<128xf32>) outs(%76 : tensor<128xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb1(%78: f32, %79: f32, %80: f32):
      %81 = arith.mulf %78, %79 : f32
      linalg.yield %81 : f32
    } -> tensor<128xf32>
    %82 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 1.280000e+02 : f32
    %83 = tensor.splat %82 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<128xf32>
    %84 = tensor.empty() : tensor<128xf32>
    %85 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%77, %83 : tensor<128xf32>, tensor<128xf32>) outs(%84 : tensor<128xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb2(%86: f32, %87: f32, %88: f32):
      %89 = arith.divf %86, %87 : f32
      linalg.yield %89 : f32
    } -> tensor<128xf32>
    %90 = tensor.empty() : tensor<128xf32>
    %91 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%85 : tensor<128xf32>) outs(%90 : tensor<128xf32>) attrs =  {prov.region_id = "exp_0", prov._pattern_hint = "exp", prov.op = "exp", prov.family = "elementwise", prov.aten = "aten.exp.default", prov.orig_dtype = "float32"} {
    ^bb3(%92: f32, %93: f32):
      %94 = math.exp %92 : f32
      linalg.yield %94 : f32
    } -> tensor<128xf32>
    %95 = func.call @aten_zeros_default() {prov.region_id = "aten_zeros_default_0", prov.dispatch_id = "aten_zeros_default_0"} : () -> tensor<i64>
    %96 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 0 : index
    %97 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 5 : index
    %98 = arith.constant {prov.op = "while_loop", prov.family = "loop"} 1 : index
    %99, %100 = scf.for %101 = %96 to %97 step %98 iter_args(%102 = %95, %103 = %58) -> (tensor<i64>, tensor<1x24x20xf32>) {
      %104 = tensor.empty() : tensor<f32>
      %105 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%102 : tensor<i64>) outs(%104 : tensor<f32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32"} {
      ^bb4(%106: i64, %107: f32):
        %108 = arith.sitofp %106 : i64 to f32
        linalg.yield %108 : f32
      } -> tensor<f32>
      %109 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 2.000000e-01 : f32
      %110 = tensor.splat %109 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<f32>
      %111 = tensor.empty() : tensor<f32>
      %112 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%105, %110 : tensor<f32>, tensor<f32>) outs(%111 : tensor<f32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb5(%113: f32, %114: f32, %115: f32):
        %116 = arith.mulf %113, %114 : f32
        linalg.yield %116 : f32
      } -> tensor<f32>
      %117 = tensor.extract %112[] : tensor<f32>
      %118 = tensor.from_elements %117 {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1xf32>
      %119 = tensor.expand_shape %118 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1xf32> into tensor<1x1xf32>
      %120 = tensor.expand_shape %91 [[0 : i64, 1 : i64]] output_shape [1, 128] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<128xf32> into tensor<1x128xf32>
      %121 = tensor.empty() : tensor<1x128xf32>
      %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%119, %120 : tensor<1x1xf32>, tensor<1x128xf32>) outs(%121 : tensor<1x128xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb6(%123: f32, %124: f32, %125: f32):
        %126 = arith.mulf %123, %124 : f32
        linalg.yield %126 : f32
      } -> tensor<1x128xf32>
      %127 = tensor.empty() : tensor<1x128xf32>
      %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%122 : tensor<1x128xf32>) outs(%127 : tensor<1x128xf32>) attrs =  {prov.region_id = "cos_0", prov._pattern_hint = "cos", prov.op = "cos", prov.family = "elementwise", prov.aten = "aten.cos.default", prov.orig_dtype = "float32"} {
      ^bb7(%129: f32, %130: f32):
        %131 = math.cos %129 : f32
        linalg.yield %131 : f32
      } -> tensor<1x128xf32>
      %132 = tensor.empty() : tensor<1x128xf32>
      %133 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%122 : tensor<1x128xf32>) outs(%132 : tensor<1x128xf32>) attrs =  {prov.region_id = "sin_0", prov._pattern_hint = "sin", prov.op = "sin", prov.family = "elementwise", prov.aten = "aten.sin.default", prov.orig_dtype = "float32"} {
      ^bb8(%134: f32, %135: f32):
        %136 = math.sin %134 : f32
        linalg.yield %136 : f32
      } -> tensor<1x128xf32>
      %137 = tensor.concat dim(1) %128, %133 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x256xf32>
      %138 = tensor.empty() : tensor<256x1024xf32>
      %139 = linalg.transpose ins(%3:tensor<1024x256xf32>) outs(%138:tensor<256x1024xf32>) permutation = [1, 0]
      %140 = tensor.empty() : tensor<1x1024xf32>
      %141 = arith.constant 0.000000e+00 : f32
      %142 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%141 : f32) outs(%140 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %143 = linalg.matmul {prov.region_id = "matmul_0", prov.dispatch_id = "matmul_0", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%137, %139 : tensor<1x256xf32>, tensor<256x1024xf32>) outs(%142 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %144 = tensor.empty() : tensor<1x1024xf32>
      %145 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%143, %4 : tensor<1x1024xf32>, tensor<1024xf32>) outs(%144 : tensor<1x1024xf32>) attrs =  {prov.region_id = "add_0", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb9(%146: f32, %147: f32, %148: f32):
        %149 = arith.addf %146, %147 : f32
        linalg.yield %149 : f32
      } -> tensor<1x1024xf32>
      %150 = tensor.empty() : tensor<1x1024xf32>
      %151 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%145 : tensor<1x1024xf32>) outs(%150 : tensor<1x1024xf32>) attrs =  {prov.region_id = "silu_0", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb10(%152: f32, %153: f32):
        %154 = arith.constant 1.000000e+00 : f32
        %155 = arith.negf %152 : f32
        %156 = math.exp %155 : f32
        %157 = arith.addf %154, %156 : f32
        %158 = arith.divf %154, %157 : f32
        %159 = arith.mulf %152, %158 : f32
        linalg.yield %159 : f32
      } -> tensor<1x1024xf32>
      %160 = tensor.empty() : tensor<1024x1024xf32>
      %161 = linalg.transpose ins(%5:tensor<1024x1024xf32>) outs(%160:tensor<1024x1024xf32>) permutation = [1, 0]
      %162 = tensor.empty() : tensor<1x1024xf32>
      %163 = arith.constant 0.000000e+00 : f32
      %164 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%163 : f32) outs(%162 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %165 = linalg.matmul {prov.region_id = "matmul_1", prov.dispatch_id = "matmul_1", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%151, %161 : tensor<1x1024xf32>, tensor<1024x1024xf32>) outs(%164 : tensor<1x1024xf32>) -> tensor<1x1024xf32>
      %166 = tensor.empty() : tensor<1x1024xf32>
      %167 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%165, %6 : tensor<1x1024xf32>, tensor<1024xf32>) outs(%166 : tensor<1x1024xf32>) attrs =  {prov.region_id = "add_1", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb11(%168: f32, %169: f32, %170: f32):
        %171 = arith.addf %168, %169 : f32
        linalg.yield %171 : f32
      } -> tensor<1x1024xf32>
      %172 = tensor.empty() : tensor<20x1024xf32>
      %173 = linalg.transpose ins(%52:tensor<1024x20xf32>) outs(%172:tensor<20x1024xf32>) permutation = [1, 0]
      %174 = tensor.empty() : tensor<1x24x1024xf32>
      %175 = arith.constant 0.000000e+00 : f32
      %176 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%175 : f32) outs(%174 : tensor<1x24x1024xf32>) -> tensor<1x24x1024xf32>
      %177 = linalg.matmul {prov.region_id = "matmul_2", prov.dispatch_id = "matmul_2", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%103, %173 : tensor<1x24x20xf32>, tensor<20x1024xf32>) outs(%176 : tensor<1x24x1024xf32>) -> tensor<1x24x1024xf32>
      %178 = tensor.empty() : tensor<1x24x1024xf32>
      %179 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%177, %53 : tensor<1x24x1024xf32>, tensor<1024xf32>) outs(%178 : tensor<1x24x1024xf32>) attrs =  {prov.region_id = "add_2", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb12(%180: f32, %181: f32, %182: f32):
        %183 = arith.addf %180, %181 : f32
        linalg.yield %183 : f32
      } -> tensor<1x24x1024xf32>
      %184 = tensor.empty() : tensor<1x24x1024xf32>
      %185 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%179 : tensor<1x24x1024xf32>) outs(%184 : tensor<1x24x1024xf32>) attrs =  {prov.region_id = "silu_1", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb13(%186: f32, %187: f32):
        %188 = arith.constant 1.000000e+00 : f32
        %189 = arith.negf %186 : f32
        %190 = math.exp %189 : f32
        %191 = arith.addf %188, %190 : f32
        %192 = arith.divf %188, %191 : f32
        %193 = arith.mulf %186, %192 : f32
        linalg.yield %193 : f32
      } -> tensor<1x24x1024xf32>
      %194 = tensor.empty() : tensor<1024x1024xf32>
      %195 = linalg.transpose ins(%54:tensor<1024x1024xf32>) outs(%194:tensor<1024x1024xf32>) permutation = [1, 0]
      %196 = tensor.empty() : tensor<1x24x1024xf32>
      %197 = arith.constant 0.000000e+00 : f32
      %198 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%197 : f32) outs(%196 : tensor<1x24x1024xf32>) -> tensor<1x24x1024xf32>
      %199 = linalg.matmul {prov.region_id = "matmul_3", prov.dispatch_id = "matmul_3", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%185, %195 : tensor<1x24x1024xf32>, tensor<1024x1024xf32>) outs(%198 : tensor<1x24x1024xf32>) -> tensor<1x24x1024xf32>
      %200 = tensor.empty() : tensor<1x24x1024xf32>
      %201 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%199, %55 : tensor<1x24x1024xf32>, tensor<1024xf32>) outs(%200 : tensor<1x24x1024xf32>) attrs =  {prov.region_id = "add_3", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb14(%202: f32, %203: f32, %204: f32):
        %205 = arith.addf %202, %203 : f32
        linalg.yield %205 : f32
      } -> tensor<1x24x1024xf32>
      %206 = tensor.empty() : tensor<1x24x1024xf32>
      %207 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%201 : tensor<1x24x1024xf32>) outs(%206 : tensor<1x24x1024xf32>) attrs =  {prov.region_id = "silu_2", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb15(%208: f32, %209: f32):
        %210 = arith.constant 1.000000e+00 : f32
        %211 = arith.negf %208 : f32
        %212 = math.exp %211 : f32
        %213 = arith.addf %210, %212 : f32
        %214 = arith.divf %210, %213 : f32
        %215 = arith.mulf %208, %214 : f32
        linalg.yield %215 : f32
      } -> tensor<1x24x1024xf32>
      %216 = tensor.empty() : tensor<1024x1024xf32>
      %217 = linalg.transpose ins(%56:tensor<1024x1024xf32>) outs(%216:tensor<1024x1024xf32>) permutation = [1, 0]
      %218 = tensor.empty() : tensor<1x24x1024xf32>
      %219 = arith.constant 0.000000e+00 : f32
      %220 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%219 : f32) outs(%218 : tensor<1x24x1024xf32>) -> tensor<1x24x1024xf32>
      %221 = linalg.matmul {prov.region_id = "matmul_4", prov.dispatch_id = "matmul_4", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%207, %217 : tensor<1x24x1024xf32>, tensor<1024x1024xf32>) outs(%220 : tensor<1x24x1024xf32>) -> tensor<1x24x1024xf32>
      %222 = tensor.empty() : tensor<1x24x1024xf32>
      %223 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%221, %57 : tensor<1x24x1024xf32>, tensor<1024xf32>) outs(%222 : tensor<1x24x1024xf32>) attrs =  {prov.region_id = "add_4", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb16(%224: f32, %225: f32, %226: f32):
        %227 = arith.addf %224, %225 : f32
        linalg.yield %227 : f32
      } -> tensor<1x24x1024xf32>
      %228 = tensor.empty() : tensor<1x1024xf32>
      %229 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%167 : tensor<1x1024xf32>) outs(%228 : tensor<1x1024xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb17(%230: f32, %231: f32):
        linalg.yield %230 : f32
      } -> tensor<1x1024xf32>
      %232 = tensor.empty() : tensor<1x1x1024xf32>
      %233 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %2 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%232 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb18(%234: f32, %235: f32, %236: f32):
        %237 = arith.addf %234, %235 : f32
        linalg.yield %237 : f32
      } -> tensor<1x1x1024xf32>
      %238 = tensor.collapse_shape %229 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %239 = tensor.expand_shape %238 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_2", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %240 = tensor.concat dim(1) %239, %233 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x2x1024xf32>
      %241 = tensor.collapse_shape %240 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x2x1024xf32> into tensor<2048xf32>
      %242 = tensor.expand_shape %241 [[0 : i64, 1 : i64]] output_shape [1, 2048] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<2048xf32> into tensor<1x2048xf32>
      %243 = tensor.empty() : tensor<1x4x1024xf32>
      %244 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<1x4x1024xf32>) outs(%243 : tensor<1x4x1024xf32>) attrs =  {prov.region_id = "expand_1", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb19(%245: f32, %246: f32):
        linalg.yield %245 : f32
      } -> tensor<1x4x1024xf32>
      %247 = tensor.concat dim(1) %223, %244 {prov.region_id = "cat_2", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1x24x1024xf32>, tensor<1x4x1024xf32>) -> tensor<1x28x1024xf32>
      %248 = tensor.empty() : tensor<1x28x1024xf32>
      %249 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%247, %1 : tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) outs(%248 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb20(%250: f32, %251: f32, %252: f32):
        %253 = arith.addf %250, %251 : f32
        linalg.yield %253 : f32
      } -> tensor<1x28x1024xf32>
      %254 = tensor.empty() : tensor<1x64x4x128xf32>
      %255 = linalg.transpose ins(%60:tensor<1x4x64x128xf32>) outs(%254:tensor<1x64x4x128xf32>) permutation = [0, 2, 1, 3]
      %256 = tensor.empty() : tensor<1x64x4x128xf32>
      %257 = linalg.transpose ins(%61:tensor<1x4x64x128xf32>) outs(%256:tensor<1x64x4x128xf32>) permutation = [0, 2, 1, 3]
      %258 = tensor.empty() : tensor<1x2048xf32>
      %259 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%242 : tensor<1x2048xf32>) outs(%258 : tensor<1x2048xf32>) attrs =  {prov.region_id = "silu_3", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb21(%260: f32, %261: f32):
        %262 = arith.constant 1.000000e+00 : f32
        %263 = arith.negf %260 : f32
        %264 = math.exp %263 : f32
        %265 = arith.addf %262, %264 : f32
        %266 = arith.divf %262, %265 : f32
        %267 = arith.mulf %260, %266 : f32
        linalg.yield %267 : f32
      } -> tensor<1x2048xf32>
      %268 = tensor.empty() : tensor<2048x9216xf32>
      %269 = linalg.transpose ins(%24:tensor<9216x2048xf32>) outs(%268:tensor<2048x9216xf32>) permutation = [1, 0]
      %270 = tensor.empty() : tensor<1x9216xf32>
      %271 = arith.constant 0.000000e+00 : f32
      %272 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%271 : f32) outs(%270 : tensor<1x9216xf32>) -> tensor<1x9216xf32>
      %273 = linalg.matmul {prov.region_id = "matmul_5", prov.dispatch_id = "matmul_5", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%259, %269 : tensor<1x2048xf32>, tensor<2048x9216xf32>) outs(%272 : tensor<1x9216xf32>) -> tensor<1x9216xf32>
      %274 = tensor.empty() : tensor<1x9216xf32>
      %275 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%273, %25 : tensor<1x9216xf32>, tensor<9216xf32>) outs(%274 : tensor<1x9216xf32>) attrs =  {prov.region_id = "add_7", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb22(%276: f32, %277: f32, %278: f32):
        %279 = arith.addf %276, %277 : f32
        linalg.yield %279 : f32
      } -> tensor<1x9216xf32>
      %280 = "tensor.extract_slice"(%275) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %281 = "tensor.extract_slice"(%275) <{static_offsets = array<i64: 0, 1024>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %282 = "tensor.extract_slice"(%275) <{static_offsets = array<i64: 0, 2048>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %283 = "tensor.extract_slice"(%275) <{static_offsets = array<i64: 0, 3072>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %284 = "tensor.extract_slice"(%275) <{static_offsets = array<i64: 0, 4096>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %285 = "tensor.extract_slice"(%275) <{static_offsets = array<i64: 0, 5120>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %286 = "tensor.extract_slice"(%275) <{static_offsets = array<i64: 0, 6144>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %287 = "tensor.extract_slice"(%275) <{static_offsets = array<i64: 0, 7168>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %288 = "tensor.extract_slice"(%275) <{static_offsets = array<i64: 0, 8192>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_0", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %289 = tensor.collapse_shape %282 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %290 = tensor.expand_shape %289 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_3", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %291 = tensor.empty() : tensor<1x28x1024xf32>
      %292 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%249 : tensor<1x28x1024xf32>) outs(%291 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb23(%293: f32, %294: f32):
        %295 = arith.constant 2.000000e+00 : f32
        %296 = math.powf %293, %295 : f32
        linalg.yield %296 : f32
      } -> tensor<1x28x1024xf32>
      %297 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %298 = tensor.splat %297 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32>
      %299 = linalg.reduce ins(%292:tensor<1x28x1024xf32>) outs(%298:tensor<1x28xf32>) dimensions = [2]
      (%300: f32, %301: f32) {
        %302 = arith.addf %300, %301 : f32
        linalg.yield %302 : f32
      }
      %303 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.024000e+03 : f32
      %304 = tensor.splat %303 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32>
      %305 = tensor.empty() : tensor<1x28xf32>
      %306 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%299, %304 : tensor<1x28xf32>, tensor<1x28xf32>) outs(%305 : tensor<1x28xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb24(%307: f32, %308: f32, %309: f32):
        %310 = arith.divf %307, %308 : f32
        linalg.yield %310 : f32
      } -> tensor<1x28xf32>
      %311 = tensor.collapse_shape %306 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32> into tensor<28xf32>
      %312 = tensor.expand_shape %311 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<28xf32> into tensor<1x28x1xf32>
      %313 = arith.constant {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %314 = tensor.splat %313 {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x1xf32>
      %315 = tensor.empty() : tensor<1x28x1xf32>
      %316 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%312, %314 : tensor<1x28x1xf32>, tensor<1x28x1xf32>) outs(%315 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb25(%317: f32, %318: f32, %319: f32):
        %320 = arith.addf %317, %318 : f32
        linalg.yield %320 : f32
      } -> tensor<1x28x1xf32>
      %321 = tensor.empty() : tensor<1x28x1xf32>
      %322 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%316 : tensor<1x28x1xf32>) outs(%321 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb26(%323: f32, %324: f32):
        %325 = math.rsqrt %323 : f32
        linalg.yield %325 : f32
      } -> tensor<1x28x1xf32>
      %326 = tensor.empty() : tensor<1x28x1024xf32>
      %327 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%249, %322 : tensor<1x28x1024xf32>, tensor<1x28x1xf32>) outs(%326 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb27(%328: f32, %329: f32, %330: f32):
        %331 = arith.mulf %328, %329 : f32
        linalg.yield %331 : f32
      } -> tensor<1x28x1024xf32>
      %332 = func.call @aten_type_as_default_wl0(%327, %249) {prov.region_id = "aten_type_as_default_0", prov.dispatch_id = "aten_type_as_default_0"} : (tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %333 = tensor.empty() : tensor<1x28x1024xf32>
      %334 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%332, %7 : tensor<1x28x1024xf32>, tensor<1024xf32>) outs(%333 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb28(%335: f32, %336: f32, %337: f32):
        %338 = arith.mulf %335, %336 : f32
        linalg.yield %338 : f32
      } -> tensor<1x28x1024xf32>
      %339 = tensor.collapse_shape %281 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %340 = tensor.expand_shape %339 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_4", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %341 = arith.constant {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
      %342 = tensor.splat %341 {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1024xf32>
      %343 = tensor.empty() : tensor<1x1x1024xf32>
      %344 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%340, %342 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%343 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb29(%345: f32, %346: f32, %347: f32):
        %348 = arith.addf %345, %346 : f32
        linalg.yield %348 : f32
      } -> tensor<1x1x1024xf32>
      %349 = tensor.empty() : tensor<1x28x1024xf32>
      %350 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%334, %344 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%349 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb30(%351: f32, %352: f32, %353: f32):
        %354 = arith.mulf %351, %352 : f32
        linalg.yield %354 : f32
      } -> tensor<1x28x1024xf32>
      %355 = tensor.collapse_shape %280 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %356 = tensor.expand_shape %355 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_5", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %357 = tensor.empty() : tensor<1x28x1024xf32>
      %358 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%350, %356 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%357 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb31(%359: f32, %360: f32, %361: f32):
        %362 = arith.addf %359, %360 : f32
        linalg.yield %362 : f32
      } -> tensor<1x28x1024xf32>
      %363 = tensor.empty() : tensor<1024x1024xf32>
      %364 = linalg.transpose ins(%8:tensor<1024x1024xf32>) outs(%363:tensor<1024x1024xf32>) permutation = [1, 0]
      %365 = tensor.empty() : tensor<1x28x1024xf32>
      %366 = arith.constant 0.000000e+00 : f32
      %367 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%366 : f32) outs(%365 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %368 = linalg.matmul {prov.region_id = "matmul_6", prov.dispatch_id = "matmul_6", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%358, %364 : tensor<1x28x1024xf32>, tensor<1024x1024xf32>) outs(%367 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %369 = tensor.collapse_shape %368 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x1024xf32> into tensor<28672xf32>
      %370 = tensor.expand_shape %369 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
      %371 = tensor.empty() : tensor<1024x1024xf32>
      %372 = linalg.transpose ins(%9:tensor<1024x1024xf32>) outs(%371:tensor<1024x1024xf32>) permutation = [1, 0]
      %373 = tensor.empty() : tensor<1x28x1024xf32>
      %374 = arith.constant 0.000000e+00 : f32
      %375 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%374 : f32) outs(%373 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %376 = linalg.matmul {prov.region_id = "matmul_7", prov.dispatch_id = "matmul_7", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%358, %372 : tensor<1x28x1024xf32>, tensor<1024x1024xf32>) outs(%375 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %377 = tensor.collapse_shape %376 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x1024xf32> into tensor<28672xf32>
      %378 = tensor.expand_shape %377 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 28, 4, 128, 2] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x4x128x2xf32>
      %379 = "tensor.extract_slice"(%378) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 28, 4, 128, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<1x28x4x128x2xf32>) -> tensor<1x28x4x128x1xf32>
      %380 = tensor.collapse_shape %379 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x28x4x128x1xf32> into tensor<14336xf32>
      %381 = tensor.expand_shape %380 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 4, 128] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<14336xf32> into tensor<1x28x4x128xf32>
      %382 = "tensor.extract_slice"(%378) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 28, 4, 128, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<1x28x4x128x2xf32>) -> tensor<1x28x4x128x1xf32>
      %383 = tensor.collapse_shape %382 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x28x4x128x1xf32> into tensor<14336xf32>
      %384 = tensor.expand_shape %383 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 4, 128] {prov.region_id = "split_1", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<14336xf32> into tensor<1x28x4x128xf32>
      %385 = tensor.empty() : tensor<1x28x8x128xf32>
      %386 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%370 : tensor<1x28x8x128xf32>) outs(%385 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb32(%387: f32, %388: f32):
        %389 = arith.constant 2.000000e+00 : f32
        %390 = math.powf %387, %389 : f32
        linalg.yield %390 : f32
      } -> tensor<1x28x8x128xf32>
      %391 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %392 = tensor.splat %391 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x8xf32>
      %393 = linalg.reduce ins(%386:tensor<1x28x8x128xf32>) outs(%392:tensor<1x28x8xf32>) dimensions = [3]
      (%394: f32, %395: f32) {
        %396 = arith.addf %394, %395 : f32
        linalg.yield %396 : f32
      }
      %397 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %398 = tensor.splat %397 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x8xf32>
      %399 = tensor.empty() : tensor<1x28x8xf32>
      %400 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%393, %398 : tensor<1x28x8xf32>, tensor<1x28x8xf32>) outs(%399 : tensor<1x28x8xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb33(%401: f32, %402: f32, %403: f32):
        %404 = arith.divf %401, %402 : f32
        linalg.yield %404 : f32
      } -> tensor<1x28x8xf32>
      %405 = tensor.collapse_shape %400 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x8xf32> into tensor<224xf32>
      %406 = tensor.expand_shape %405 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x28x8x1xf32>
      %407 = arith.constant {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %408 = tensor.splat %407 {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x8x1xf32>
      %409 = tensor.empty() : tensor<1x28x8x1xf32>
      %410 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%406, %408 : tensor<1x28x8x1xf32>, tensor<1x28x8x1xf32>) outs(%409 : tensor<1x28x8x1xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb34(%411: f32, %412: f32, %413: f32):
        %414 = arith.addf %411, %412 : f32
        linalg.yield %414 : f32
      } -> tensor<1x28x8x1xf32>
      %415 = tensor.empty() : tensor<1x28x8x1xf32>
      %416 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%410 : tensor<1x28x8x1xf32>) outs(%415 : tensor<1x28x8x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb35(%417: f32, %418: f32):
        %419 = math.rsqrt %417 : f32
        linalg.yield %419 : f32
      } -> tensor<1x28x8x1xf32>
      %420 = tensor.empty() : tensor<1x28x8x128xf32>
      %421 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%370, %416 : tensor<1x28x8x128xf32>, tensor<1x28x8x1xf32>) outs(%420 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb36(%422: f32, %423: f32, %424: f32):
        %425 = arith.mulf %422, %423 : f32
        linalg.yield %425 : f32
      } -> tensor<1x28x8x128xf32>
      %426 = func.call @aten_type_as_default_1_wl1(%421, %370) {prov.region_id = "aten_type_as_default_1_0", prov.dispatch_id = "aten_type_as_default_1_0"} : (tensor<1x28x8x128xf32>, tensor<1x28x8x128xf32>) -> tensor<1x28x8x128xf32>
      %427 = tensor.empty() : tensor<1x28x8x128xf32>
      %428 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%426, %11 : tensor<1x28x8x128xf32>, tensor<128xf32>) outs(%427 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb37(%429: f32, %430: f32, %431: f32):
        %432 = arith.mulf %429, %430 : f32
        linalg.yield %432 : f32
      } -> tensor<1x28x8x128xf32>
      %433 = tensor.empty() : tensor<1x28x4x128xf32>
      %434 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%381 : tensor<1x28x4x128xf32>) outs(%433 : tensor<1x28x4x128xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb38(%435: f32, %436: f32):
        %437 = arith.constant 2.000000e+00 : f32
        %438 = math.powf %435, %437 : f32
        linalg.yield %438 : f32
      } -> tensor<1x28x4x128xf32>
      %439 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %440 = tensor.splat %439 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x4xf32>
      %441 = linalg.reduce ins(%434:tensor<1x28x4x128xf32>) outs(%440:tensor<1x28x4xf32>) dimensions = [3]
      (%442: f32, %443: f32) {
        %444 = arith.addf %442, %443 : f32
        linalg.yield %444 : f32
      }
      %445 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %446 = tensor.splat %445 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x4xf32>
      %447 = tensor.empty() : tensor<1x28x4xf32>
      %448 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%441, %446 : tensor<1x28x4xf32>, tensor<1x28x4xf32>) outs(%447 : tensor<1x28x4xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb39(%449: f32, %450: f32, %451: f32):
        %452 = arith.divf %449, %450 : f32
        linalg.yield %452 : f32
      } -> tensor<1x28x4xf32>
      %453 = tensor.collapse_shape %448 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x4xf32> into tensor<112xf32>
      %454 = tensor.expand_shape %453 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 4, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<112xf32> into tensor<1x28x4x1xf32>
      %455 = arith.constant {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %456 = tensor.splat %455 {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x4x1xf32>
      %457 = tensor.empty() : tensor<1x28x4x1xf32>
      %458 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%454, %456 : tensor<1x28x4x1xf32>, tensor<1x28x4x1xf32>) outs(%457 : tensor<1x28x4x1xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb40(%459: f32, %460: f32, %461: f32):
        %462 = arith.addf %459, %460 : f32
        linalg.yield %462 : f32
      } -> tensor<1x28x4x1xf32>
      %463 = tensor.empty() : tensor<1x28x4x1xf32>
      %464 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%458 : tensor<1x28x4x1xf32>) outs(%463 : tensor<1x28x4x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb41(%465: f32, %466: f32):
        %467 = math.rsqrt %465 : f32
        linalg.yield %467 : f32
      } -> tensor<1x28x4x1xf32>
      %468 = tensor.empty() : tensor<1x28x4x128xf32>
      %469 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%381, %464 : tensor<1x28x4x128xf32>, tensor<1x28x4x1xf32>) outs(%468 : tensor<1x28x4x128xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb42(%470: f32, %471: f32, %472: f32):
        %473 = arith.mulf %470, %471 : f32
        linalg.yield %473 : f32
      } -> tensor<1x28x4x128xf32>
      %474 = func.call @aten_type_as_default_2_wl2(%469, %381) {prov.region_id = "aten_type_as_default_2_0", prov.dispatch_id = "aten_type_as_default_2_0"} : (tensor<1x28x4x128xf32>, tensor<1x28x4x128xf32>) -> tensor<1x28x4x128xf32>
      %475 = tensor.empty() : tensor<1x28x4x128xf32>
      %476 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%474, %12 : tensor<1x28x4x128xf32>, tensor<128xf32>) outs(%475 : tensor<1x28x4x128xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb43(%477: f32, %478: f32, %479: f32):
        %480 = arith.mulf %477, %478 : f32
        linalg.yield %480 : f32
      } -> tensor<1x28x4x128xf32>
      %481 = tensor.collapse_shape %476 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x28x4x128xf32> into tensor<14336xf32>
      %482 = tensor.expand_shape %481 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 28, 4, 1, 128] {prov.region_id = "unsqueeze_6", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<14336xf32> into tensor<1x28x4x1x128xf32>
      %483 = tensor.empty() : tensor<1x28x4x2x128xf32>
      %484 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%482 : tensor<1x28x4x1x128xf32>) outs(%483 : tensor<1x28x4x2x128xf32>) attrs =  {prov.region_id = "expand_2", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb44(%485: f32, %486: f32):
        linalg.yield %485 : f32
      } -> tensor<1x28x4x2x128xf32>
      %487 = tensor.collapse_shape %484 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x28x4x2x128xf32> into tensor<28672xf32>
      %488 = tensor.expand_shape %487 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
      %489 = tensor.collapse_shape %384 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x28x4x128xf32> into tensor<14336xf32>
      %490 = tensor.expand_shape %489 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 28, 4, 1, 128] {prov.region_id = "unsqueeze_7", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<14336xf32> into tensor<1x28x4x1x128xf32>
      %491 = tensor.empty() : tensor<1x28x4x2x128xf32>
      %492 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%490 : tensor<1x28x4x1x128xf32>) outs(%491 : tensor<1x28x4x2x128xf32>) attrs =  {prov.region_id = "expand_3", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb45(%493: f32, %494: f32):
        linalg.yield %493 : f32
      } -> tensor<1x28x4x2x128xf32>
      %495 = tensor.collapse_shape %492 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x28x4x2x128xf32> into tensor<28672xf32>
      %496 = tensor.expand_shape %495 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
      %497 = tensor.empty() : tensor<1x8x28x128xf32>
      %498 = linalg.transpose ins(%428:tensor<1x28x8x128xf32>) outs(%497:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
      %499 = tensor.empty() : tensor<1x8x28x128xf32>
      %500 = linalg.transpose ins(%488:tensor<1x28x8x128xf32>) outs(%499:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
      %501 = tensor.empty() : tensor<1x8x28x128xf32>
      %502 = linalg.transpose ins(%496:tensor<1x28x8x128xf32>) outs(%501:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
      %503 = tensor.empty() : tensor<1x8x128x28xf32>
      %504 = linalg.transpose ins(%500:tensor<1x8x28x128xf32>) outs(%503:tensor<1x8x128x28xf32>) permutation = [0, 1, 3, 2]
      %505 = arith.constant {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %506 = tensor.splat %505 {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x28x28xf32>
      %507 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%498, %504 : tensor<1x8x28x128xf32>, tensor<1x8x128x28xf32>) outs(%506 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "matmul_8", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb46(%508: f32, %509: f32, %510: f32):
        %511 = arith.mulf %508, %509 : f32
        %512 = arith.addf %510, %511 : f32
        linalg.yield %512 : f32
      } -> tensor<1x8x28x28xf32>
      %513 = arith.constant {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.0883883461 : f32
      %514 = tensor.splat %513 {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x28x28xf32>
      %515 = tensor.empty() : tensor<1x8x28x28xf32>
      %516 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%507, %514 : tensor<1x8x28x28xf32>, tensor<1x8x28x28xf32>) outs(%515 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "mul_9", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb47(%517: f32, %518: f32, %519: f32):
        %520 = arith.mulf %517, %518 : f32
        linalg.yield %520 : f32
      } -> tensor<1x8x28x28xf32>
      %521 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0xff800000 : f32
      %522 = tensor.splat %521 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x28xf32>
      %523 = linalg.reduce ins(%516:tensor<1x8x28x28xf32>) outs(%522:tensor<1x8x28xf32>) dimensions = [3]
      (%524: f32, %525: f32) {
        %526 = arith.maximumf %524, %525 : f32
        linalg.yield %526 : f32
      }
      %527 = tensor.collapse_shape %523 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x28xf32> into tensor<224xf32>
      %528 = tensor.expand_shape %527 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x8x28x1xf32>
      %529 = tensor.empty() : tensor<1x8x28x28xf32>
      %530 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%516, %528 : tensor<1x8x28x28xf32>, tensor<1x8x28x1xf32>) outs(%529 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb48(%531: f32, %532: f32, %533: f32):
        %534 = arith.subf %531, %532 : f32
        linalg.yield %534 : f32
      } -> tensor<1x8x28x28xf32>
      %535 = tensor.empty() : tensor<1x8x28x28xf32>
      %536 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%530 : tensor<1x8x28x28xf32>) outs(%535 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb49(%537: f32, %538: f32):
        %539 = math.exp %537 : f32
        linalg.yield %539 : f32
      } -> tensor<1x8x28x28xf32>
      %540 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %541 = tensor.splat %540 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x28xf32>
      %542 = linalg.reduce ins(%536:tensor<1x8x28x28xf32>) outs(%541:tensor<1x8x28xf32>) dimensions = [3]
      (%543: f32, %544: f32) {
        %545 = arith.addf %543, %544 : f32
        linalg.yield %545 : f32
      }
      %546 = tensor.collapse_shape %542 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x28xf32> into tensor<224xf32>
      %547 = tensor.expand_shape %546 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x8x28x1xf32>
      %548 = tensor.empty() : tensor<1x8x28x28xf32>
      %549 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%536, %547 : tensor<1x8x28x28xf32>, tensor<1x8x28x1xf32>) outs(%548 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb50(%550: f32, %551: f32, %552: f32):
        %553 = arith.divf %550, %551 : f32
        linalg.yield %553 : f32
      } -> tensor<1x8x28x28xf32>
      %554 = func.call @aten_type_as_default_3_wl3(%549, %498) {prov.region_id = "aten_type_as_default_3_0", prov.dispatch_id = "aten_type_as_default_3_0"} : (tensor<1x8x28x28xf32>, tensor<1x8x28x128xf32>) -> tensor<1x8x28x28xf32>
      %555 = arith.constant {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %556 = tensor.splat %555 {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x28x128xf32>
      %557 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%554, %502 : tensor<1x8x28x28xf32>, tensor<1x8x28x128xf32>) outs(%556 : tensor<1x8x28x128xf32>) attrs =  {prov.region_id = "matmul_9", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb51(%558: f32, %559: f32, %560: f32):
        %561 = arith.mulf %558, %559 : f32
        %562 = arith.addf %560, %561 : f32
        linalg.yield %562 : f32
      } -> tensor<1x8x28x128xf32>
      %563 = tensor.empty() : tensor<1x28x8x128xf32>
      %564 = linalg.transpose ins(%557:tensor<1x8x28x128xf32>) outs(%563:tensor<1x28x8x128xf32>) permutation = [0, 2, 1, 3]
      %565 = tensor.collapse_shape %564 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
      %566 = tensor.expand_shape %565 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_6", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x1024xf32>
      %567 = tensor.empty() : tensor<1024x1024xf32>
      %568 = linalg.transpose ins(%10:tensor<1024x1024xf32>) outs(%567:tensor<1024x1024xf32>) permutation = [1, 0]
      %569 = tensor.empty() : tensor<1x28x1024xf32>
      %570 = arith.constant 0.000000e+00 : f32
      %571 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%570 : f32) outs(%569 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %572 = linalg.matmul {prov.region_id = "matmul_10", prov.dispatch_id = "matmul_10", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%566, %568 : tensor<1x28x1024xf32>, tensor<1024x1024xf32>) outs(%571 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %573 = tensor.empty() : tensor<1x28x1024xf32>
      %574 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%290, %572 : tensor<1x1x1024xf32>, tensor<1x28x1024xf32>) outs(%573 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_10", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb52(%575: f32, %576: f32, %577: f32):
        %578 = arith.mulf %575, %576 : f32
        linalg.yield %578 : f32
      } -> tensor<1x28x1024xf32>
      %579 = tensor.empty() : tensor<1x28x1024xf32>
      %580 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%249, %574 : tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) outs(%579 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb53(%581: f32, %582: f32, %583: f32):
        %584 = arith.addf %581, %582 : f32
        linalg.yield %584 : f32
      } -> tensor<1x28x1024xf32>
      %585 = tensor.collapse_shape %285 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %586 = tensor.expand_shape %585 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_8", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %587 = tensor.empty() : tensor<1x28x1024xf32>
      %588 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%580 : tensor<1x28x1024xf32>) outs(%587 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "pow_3", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb54(%589: f32, %590: f32):
        %591 = arith.constant 2.000000e+00 : f32
        %592 = math.powf %589, %591 : f32
        linalg.yield %592 : f32
      } -> tensor<1x28x1024xf32>
      %593 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %594 = tensor.splat %593 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32>
      %595 = linalg.reduce ins(%588:tensor<1x28x1024xf32>) outs(%594:tensor<1x28xf32>) dimensions = [2]
      (%596: f32, %597: f32) {
        %598 = arith.addf %596, %597 : f32
        linalg.yield %598 : f32
      }
      %599 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.024000e+03 : f32
      %600 = tensor.splat %599 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32>
      %601 = tensor.empty() : tensor<1x28xf32>
      %602 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%595, %600 : tensor<1x28xf32>, tensor<1x28xf32>) outs(%601 : tensor<1x28xf32>) attrs =  {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb55(%603: f32, %604: f32, %605: f32):
        %606 = arith.divf %603, %604 : f32
        linalg.yield %606 : f32
      } -> tensor<1x28xf32>
      %607 = tensor.collapse_shape %602 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32> into tensor<28xf32>
      %608 = tensor.expand_shape %607 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<28xf32> into tensor<1x28x1xf32>
      %609 = arith.constant {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %610 = tensor.splat %609 {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x1xf32>
      %611 = tensor.empty() : tensor<1x28x1xf32>
      %612 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%608, %610 : tensor<1x28x1xf32>, tensor<1x28x1xf32>) outs(%611 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb56(%613: f32, %614: f32, %615: f32):
        %616 = arith.addf %613, %614 : f32
        linalg.yield %616 : f32
      } -> tensor<1x28x1xf32>
      %617 = tensor.empty() : tensor<1x28x1xf32>
      %618 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%612 : tensor<1x28x1xf32>) outs(%617 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "rsqrt_3", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb57(%619: f32, %620: f32):
        %621 = math.rsqrt %619 : f32
        linalg.yield %621 : f32
      } -> tensor<1x28x1xf32>
      %622 = tensor.empty() : tensor<1x28x1024xf32>
      %623 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%580, %618 : tensor<1x28x1024xf32>, tensor<1x28x1xf32>) outs(%622 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_11", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb58(%624: f32, %625: f32, %626: f32):
        %627 = arith.mulf %624, %625 : f32
        linalg.yield %627 : f32
      } -> tensor<1x28x1024xf32>
      %628 = func.call @aten_type_as_default_wl0(%623, %580) {prov.region_id = "aten_type_as_default_1", prov.dispatch_id = "aten_type_as_default_1"} : (tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %629 = tensor.empty() : tensor<1x28x1024xf32>
      %630 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%628, %13 : tensor<1x28x1024xf32>, tensor<1024xf32>) outs(%629 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_12", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb59(%631: f32, %632: f32, %633: f32):
        %634 = arith.mulf %631, %632 : f32
        linalg.yield %634 : f32
      } -> tensor<1x28x1024xf32>
      %635 = tensor.collapse_shape %284 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %636 = tensor.expand_shape %635 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_9", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %637 = arith.constant {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
      %638 = tensor.splat %637 {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1024xf32>
      %639 = tensor.empty() : tensor<1x1x1024xf32>
      %640 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%636, %638 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%639 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb60(%641: f32, %642: f32, %643: f32):
        %644 = arith.addf %641, %642 : f32
        linalg.yield %644 : f32
      } -> tensor<1x1x1024xf32>
      %645 = tensor.empty() : tensor<1x28x1024xf32>
      %646 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%630, %640 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%645 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_13", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb61(%647: f32, %648: f32, %649: f32):
        %650 = arith.mulf %647, %648 : f32
        linalg.yield %650 : f32
      } -> tensor<1x28x1024xf32>
      %651 = tensor.collapse_shape %283 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %652 = tensor.expand_shape %651 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_10", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %653 = tensor.empty() : tensor<1x28x1024xf32>
      %654 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%646, %652 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%653 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb62(%655: f32, %656: f32, %657: f32):
        %658 = arith.addf %655, %656 : f32
        linalg.yield %658 : f32
      } -> tensor<1x28x1024xf32>
      %659 = tensor.empty() : tensor<1024x1024xf32>
      %660 = linalg.transpose ins(%15:tensor<1024x1024xf32>) outs(%659:tensor<1024x1024xf32>) permutation = [1, 0]
      %661 = tensor.empty() : tensor<1x28x1024xf32>
      %662 = arith.constant 0.000000e+00 : f32
      %663 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%662 : f32) outs(%661 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %664 = linalg.matmul {prov.region_id = "matmul_11", prov.dispatch_id = "matmul_11", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%654, %660 : tensor<1x28x1024xf32>, tensor<1024x1024xf32>) outs(%663 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %665 = tensor.collapse_shape %664 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x1024xf32> into tensor<28672xf32>
      %666 = tensor.expand_shape %665 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_7", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
      %667 = tensor.empty() : tensor<1x28x8x128xf32>
      %668 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%666 : tensor<1x28x8x128xf32>) outs(%667 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "pow_4", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb63(%669: f32, %670: f32):
        %671 = arith.constant 2.000000e+00 : f32
        %672 = math.powf %669, %671 : f32
        linalg.yield %672 : f32
      } -> tensor<1x28x8x128xf32>
      %673 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %674 = tensor.splat %673 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x8xf32>
      %675 = linalg.reduce ins(%668:tensor<1x28x8x128xf32>) outs(%674:tensor<1x28x8xf32>) dimensions = [3]
      (%676: f32, %677: f32) {
        %678 = arith.addf %676, %677 : f32
        linalg.yield %678 : f32
      }
      %679 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %680 = tensor.splat %679 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x8xf32>
      %681 = tensor.empty() : tensor<1x28x8xf32>
      %682 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%675, %680 : tensor<1x28x8xf32>, tensor<1x28x8xf32>) outs(%681 : tensor<1x28x8xf32>) attrs =  {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb64(%683: f32, %684: f32, %685: f32):
        %686 = arith.divf %683, %684 : f32
        linalg.yield %686 : f32
      } -> tensor<1x28x8xf32>
      %687 = tensor.collapse_shape %682 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x8xf32> into tensor<224xf32>
      %688 = tensor.expand_shape %687 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x28x8x1xf32>
      %689 = arith.constant {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %690 = tensor.splat %689 {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x8x1xf32>
      %691 = tensor.empty() : tensor<1x28x8x1xf32>
      %692 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%688, %690 : tensor<1x28x8x1xf32>, tensor<1x28x8x1xf32>) outs(%691 : tensor<1x28x8x1xf32>) attrs =  {prov.region_id = "add_17", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb65(%693: f32, %694: f32, %695: f32):
        %696 = arith.addf %693, %694 : f32
        linalg.yield %696 : f32
      } -> tensor<1x28x8x1xf32>
      %697 = tensor.empty() : tensor<1x28x8x1xf32>
      %698 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%692 : tensor<1x28x8x1xf32>) outs(%697 : tensor<1x28x8x1xf32>) attrs =  {prov.region_id = "rsqrt_4", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb66(%699: f32, %700: f32):
        %701 = math.rsqrt %699 : f32
        linalg.yield %701 : f32
      } -> tensor<1x28x8x1xf32>
      %702 = tensor.empty() : tensor<1x28x8x128xf32>
      %703 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%666, %698 : tensor<1x28x8x128xf32>, tensor<1x28x8x1xf32>) outs(%702 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_14", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb67(%704: f32, %705: f32, %706: f32):
        %707 = arith.mulf %704, %705 : f32
        linalg.yield %707 : f32
      } -> tensor<1x28x8x128xf32>
      %708 = func.call @aten_type_as_default_1_wl1(%703, %666) {prov.region_id = "aten_type_as_default_1_1", prov.dispatch_id = "aten_type_as_default_1_1"} : (tensor<1x28x8x128xf32>, tensor<1x28x8x128xf32>) -> tensor<1x28x8x128xf32>
      %709 = tensor.empty() : tensor<1x28x8x128xf32>
      %710 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%708, %18 : tensor<1x28x8x128xf32>, tensor<128xf32>) outs(%709 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_15", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb68(%711: f32, %712: f32, %713: f32):
        %714 = arith.mulf %711, %712 : f32
        linalg.yield %714 : f32
      } -> tensor<1x28x8x128xf32>
      %715 = tensor.collapse_shape %255 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x64x4x128xf32> into tensor<32768xf32>
      %716 = tensor.expand_shape %715 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 64, 4, 1, 128] {prov.region_id = "unsqueeze_11", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<1x64x4x1x128xf32>
      %717 = tensor.empty() : tensor<1x64x4x2x128xf32>
      %718 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%716 : tensor<1x64x4x1x128xf32>) outs(%717 : tensor<1x64x4x2x128xf32>) attrs =  {prov.region_id = "expand_4", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb69(%719: f32, %720: f32):
        linalg.yield %719 : f32
      } -> tensor<1x64x4x2x128xf32>
      %721 = tensor.collapse_shape %718 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x64x4x2x128xf32> into tensor<65536xf32>
      %722 = tensor.expand_shape %721 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 128] {prov.region_id = "view_8", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<65536xf32> into tensor<1x64x8x128xf32>
      %723 = tensor.collapse_shape %257 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x64x4x128xf32> into tensor<32768xf32>
      %724 = tensor.expand_shape %723 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 64, 4, 1, 128] {prov.region_id = "unsqueeze_12", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<1x64x4x1x128xf32>
      %725 = tensor.empty() : tensor<1x64x4x2x128xf32>
      %726 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%724 : tensor<1x64x4x1x128xf32>) outs(%725 : tensor<1x64x4x2x128xf32>) attrs =  {prov.region_id = "expand_5", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb70(%727: f32, %728: f32):
        linalg.yield %727 : f32
      } -> tensor<1x64x4x2x128xf32>
      %729 = tensor.collapse_shape %726 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x64x4x2x128xf32> into tensor<65536xf32>
      %730 = tensor.expand_shape %729 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 128] {prov.region_id = "view_9", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<65536xf32> into tensor<1x64x8x128xf32>
      %731 = tensor.empty() : tensor<1x8x28x128xf32>
      %732 = linalg.transpose ins(%710:tensor<1x28x8x128xf32>) outs(%731:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
      %733 = tensor.empty() : tensor<1x8x64x128xf32>
      %734 = linalg.transpose ins(%722:tensor<1x64x8x128xf32>) outs(%733:tensor<1x8x64x128xf32>) permutation = [0, 2, 1, 3]
      %735 = tensor.empty() : tensor<1x8x64x128xf32>
      %736 = linalg.transpose ins(%730:tensor<1x64x8x128xf32>) outs(%735:tensor<1x8x64x128xf32>) permutation = [0, 2, 1, 3]
      %737 = tensor.empty() : tensor<1x8x128x64xf32>
      %738 = linalg.transpose ins(%734:tensor<1x8x64x128xf32>) outs(%737:tensor<1x8x128x64xf32>) permutation = [0, 1, 3, 2]
      %739 = arith.constant {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %740 = tensor.splat %739 {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x28x64xf32>
      %741 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%732, %738 : tensor<1x8x28x128xf32>, tensor<1x8x128x64xf32>) outs(%740 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "matmul_12", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb71(%742: f32, %743: f32, %744: f32):
        %745 = arith.mulf %742, %743 : f32
        %746 = arith.addf %744, %745 : f32
        linalg.yield %746 : f32
      } -> tensor<1x8x28x64xf32>
      %747 = arith.constant {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.0883883461 : f32
      %748 = tensor.splat %747 {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x28x64xf32>
      %749 = tensor.empty() : tensor<1x8x28x64xf32>
      %750 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%741, %748 : tensor<1x8x28x64xf32>, tensor<1x8x28x64xf32>) outs(%749 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "mul_16", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb72(%751: f32, %752: f32, %753: f32):
        %754 = arith.mulf %751, %752 : f32
        linalg.yield %754 : f32
      } -> tensor<1x8x28x64xf32>
      %755 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0xff800000 : f32
      %756 = tensor.splat %755 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x28xf32>
      %757 = linalg.reduce ins(%750:tensor<1x8x28x64xf32>) outs(%756:tensor<1x8x28xf32>) dimensions = [3]
      (%758: f32, %759: f32) {
        %760 = arith.maximumf %758, %759 : f32
        linalg.yield %760 : f32
      }
      %761 = tensor.collapse_shape %757 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x28xf32> into tensor<224xf32>
      %762 = tensor.expand_shape %761 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x8x28x1xf32>
      %763 = tensor.empty() : tensor<1x8x28x64xf32>
      %764 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%750, %762 : tensor<1x8x28x64xf32>, tensor<1x8x28x1xf32>) outs(%763 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb73(%765: f32, %766: f32, %767: f32):
        %768 = arith.subf %765, %766 : f32
        linalg.yield %768 : f32
      } -> tensor<1x8x28x64xf32>
      %769 = tensor.empty() : tensor<1x8x28x64xf32>
      %770 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%764 : tensor<1x8x28x64xf32>) outs(%769 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb74(%771: f32, %772: f32):
        %773 = math.exp %771 : f32
        linalg.yield %773 : f32
      } -> tensor<1x8x28x64xf32>
      %774 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %775 = tensor.splat %774 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x28xf32>
      %776 = linalg.reduce ins(%770:tensor<1x8x28x64xf32>) outs(%775:tensor<1x8x28xf32>) dimensions = [3]
      (%777: f32, %778: f32) {
        %779 = arith.addf %777, %778 : f32
        linalg.yield %779 : f32
      }
      %780 = tensor.collapse_shape %776 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x28xf32> into tensor<224xf32>
      %781 = tensor.expand_shape %780 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x8x28x1xf32>
      %782 = tensor.empty() : tensor<1x8x28x64xf32>
      %783 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%770, %781 : tensor<1x8x28x64xf32>, tensor<1x8x28x1xf32>) outs(%782 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb75(%784: f32, %785: f32, %786: f32):
        %787 = arith.divf %784, %785 : f32
        linalg.yield %787 : f32
      } -> tensor<1x8x28x64xf32>
      %788 = func.call @aten_type_as_default_4_wl4(%783, %732) {prov.region_id = "aten_type_as_default_4_0", prov.dispatch_id = "aten_type_as_default_4_0"} : (tensor<1x8x28x64xf32>, tensor<1x8x28x128xf32>) -> tensor<1x8x28x64xf32>
      %789 = arith.constant {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %790 = tensor.splat %789 {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x28x128xf32>
      %791 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%788, %736 : tensor<1x8x28x64xf32>, tensor<1x8x64x128xf32>) outs(%790 : tensor<1x8x28x128xf32>) attrs =  {prov.region_id = "matmul_13", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb76(%792: f32, %793: f32, %794: f32):
        %795 = arith.mulf %792, %793 : f32
        %796 = arith.addf %794, %795 : f32
        linalg.yield %796 : f32
      } -> tensor<1x8x28x128xf32>
      %797 = tensor.empty() : tensor<1x28x8x128xf32>
      %798 = linalg.transpose ins(%791:tensor<1x8x28x128xf32>) outs(%797:tensor<1x28x8x128xf32>) permutation = [0, 2, 1, 3]
      %799 = tensor.collapse_shape %798 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
      %800 = tensor.expand_shape %799 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_10", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x1024xf32>
      %801 = tensor.empty() : tensor<1024x1024xf32>
      %802 = linalg.transpose ins(%17:tensor<1024x1024xf32>) outs(%801:tensor<1024x1024xf32>) permutation = [1, 0]
      %803 = tensor.empty() : tensor<1x28x1024xf32>
      %804 = arith.constant 0.000000e+00 : f32
      %805 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%804 : f32) outs(%803 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %806 = linalg.matmul {prov.region_id = "matmul_14", prov.dispatch_id = "matmul_14", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%800, %802 : tensor<1x28x1024xf32>, tensor<1024x1024xf32>) outs(%805 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %807 = tensor.empty() : tensor<1x28x1024xf32>
      %808 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%586, %806 : tensor<1x1x1024xf32>, tensor<1x28x1024xf32>) outs(%807 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_17", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb77(%809: f32, %810: f32, %811: f32):
        %812 = arith.mulf %809, %810 : f32
        linalg.yield %812 : f32
      } -> tensor<1x28x1024xf32>
      %813 = tensor.empty() : tensor<1x28x1024xf32>
      %814 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%580, %808 : tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) outs(%813 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_18", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb78(%815: f32, %816: f32, %817: f32):
        %818 = arith.addf %815, %816 : f32
        linalg.yield %818 : f32
      } -> tensor<1x28x1024xf32>
      %819 = tensor.collapse_shape %288 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %820 = tensor.expand_shape %819 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_13", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %821 = tensor.empty() : tensor<1x28x1024xf32>
      %822 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%814 : tensor<1x28x1024xf32>) outs(%821 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "pow_5", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb79(%823: f32, %824: f32):
        %825 = arith.constant 2.000000e+00 : f32
        %826 = math.powf %823, %825 : f32
        linalg.yield %826 : f32
      } -> tensor<1x28x1024xf32>
      %827 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %828 = tensor.splat %827 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32>
      %829 = linalg.reduce ins(%822:tensor<1x28x1024xf32>) outs(%828:tensor<1x28xf32>) dimensions = [2]
      (%830: f32, %831: f32) {
        %832 = arith.addf %830, %831 : f32
        linalg.yield %832 : f32
      }
      %833 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.024000e+03 : f32
      %834 = tensor.splat %833 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32>
      %835 = tensor.empty() : tensor<1x28xf32>
      %836 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%829, %834 : tensor<1x28xf32>, tensor<1x28xf32>) outs(%835 : tensor<1x28xf32>) attrs =  {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb80(%837: f32, %838: f32, %839: f32):
        %840 = arith.divf %837, %838 : f32
        linalg.yield %840 : f32
      } -> tensor<1x28xf32>
      %841 = tensor.collapse_shape %836 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32> into tensor<28xf32>
      %842 = tensor.expand_shape %841 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<28xf32> into tensor<1x28x1xf32>
      %843 = arith.constant {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %844 = tensor.splat %843 {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x1xf32>
      %845 = tensor.empty() : tensor<1x28x1xf32>
      %846 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%842, %844 : tensor<1x28x1xf32>, tensor<1x28x1xf32>) outs(%845 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "add_19", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb81(%847: f32, %848: f32, %849: f32):
        %850 = arith.addf %847, %848 : f32
        linalg.yield %850 : f32
      } -> tensor<1x28x1xf32>
      %851 = tensor.empty() : tensor<1x28x1xf32>
      %852 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%846 : tensor<1x28x1xf32>) outs(%851 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "rsqrt_5", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb82(%853: f32, %854: f32):
        %855 = math.rsqrt %853 : f32
        linalg.yield %855 : f32
      } -> tensor<1x28x1xf32>
      %856 = tensor.empty() : tensor<1x28x1024xf32>
      %857 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%814, %852 : tensor<1x28x1024xf32>, tensor<1x28x1xf32>) outs(%856 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_18", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb83(%858: f32, %859: f32, %860: f32):
        %861 = arith.mulf %858, %859 : f32
        linalg.yield %861 : f32
      } -> tensor<1x28x1024xf32>
      %862 = func.call @aten_type_as_default_wl0(%857, %814) {prov.region_id = "aten_type_as_default_2", prov.dispatch_id = "aten_type_as_default_2"} : (tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %863 = tensor.empty() : tensor<1x28x1024xf32>
      %864 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%862, %20 : tensor<1x28x1024xf32>, tensor<1024xf32>) outs(%863 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_19", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb84(%865: f32, %866: f32, %867: f32):
        %868 = arith.mulf %865, %866 : f32
        linalg.yield %868 : f32
      } -> tensor<1x28x1024xf32>
      %869 = tensor.collapse_shape %287 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %870 = tensor.expand_shape %869 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_14", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %871 = arith.constant {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
      %872 = tensor.splat %871 {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1024xf32>
      %873 = tensor.empty() : tensor<1x1x1024xf32>
      %874 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%870, %872 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%873 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_20", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb85(%875: f32, %876: f32, %877: f32):
        %878 = arith.addf %875, %876 : f32
        linalg.yield %878 : f32
      } -> tensor<1x1x1024xf32>
      %879 = tensor.empty() : tensor<1x28x1024xf32>
      %880 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%864, %874 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%879 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_20", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb86(%881: f32, %882: f32, %883: f32):
        %884 = arith.mulf %881, %882 : f32
        linalg.yield %884 : f32
      } -> tensor<1x28x1024xf32>
      %885 = tensor.collapse_shape %286 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %886 = tensor.expand_shape %885 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_15", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %887 = tensor.empty() : tensor<1x28x1024xf32>
      %888 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%880, %886 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%887 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_21", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb87(%889: f32, %890: f32, %891: f32):
        %892 = arith.addf %889, %890 : f32
        linalg.yield %892 : f32
      } -> tensor<1x28x1024xf32>
      %893 = tensor.empty() : tensor<1024x2816xf32>
      %894 = linalg.transpose ins(%21:tensor<2816x1024xf32>) outs(%893:tensor<1024x2816xf32>) permutation = [1, 0]
      %895 = tensor.empty() : tensor<1x28x2816xf32>
      %896 = arith.constant 0.000000e+00 : f32
      %897 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%896 : f32) outs(%895 : tensor<1x28x2816xf32>) -> tensor<1x28x2816xf32>
      %898 = linalg.matmul {prov.region_id = "matmul_15", prov.dispatch_id = "matmul_15", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%888, %894 : tensor<1x28x1024xf32>, tensor<1024x2816xf32>) outs(%897 : tensor<1x28x2816xf32>) -> tensor<1x28x2816xf32>
      %899 = tensor.empty() : tensor<1x28x2816xf32>
      %900 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%898 : tensor<1x28x2816xf32>) outs(%899 : tensor<1x28x2816xf32>) attrs =  {prov.region_id = "silu_4", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb88(%901: f32, %902: f32):
        %903 = arith.constant 1.000000e+00 : f32
        %904 = arith.negf %901 : f32
        %905 = math.exp %904 : f32
        %906 = arith.addf %903, %905 : f32
        %907 = arith.divf %903, %906 : f32
        %908 = arith.mulf %901, %907 : f32
        linalg.yield %908 : f32
      } -> tensor<1x28x2816xf32>
      %909 = tensor.empty() : tensor<1024x2816xf32>
      %910 = linalg.transpose ins(%23:tensor<2816x1024xf32>) outs(%909:tensor<1024x2816xf32>) permutation = [1, 0]
      %911 = tensor.empty() : tensor<1x28x2816xf32>
      %912 = arith.constant 0.000000e+00 : f32
      %913 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%912 : f32) outs(%911 : tensor<1x28x2816xf32>) -> tensor<1x28x2816xf32>
      %914 = linalg.matmul {prov.region_id = "matmul_16", prov.dispatch_id = "matmul_16", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%888, %910 : tensor<1x28x1024xf32>, tensor<1024x2816xf32>) outs(%913 : tensor<1x28x2816xf32>) -> tensor<1x28x2816xf32>
      %915 = tensor.empty() : tensor<1x28x2816xf32>
      %916 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%900, %914 : tensor<1x28x2816xf32>, tensor<1x28x2816xf32>) outs(%915 : tensor<1x28x2816xf32>) attrs =  {prov.region_id = "mul_21", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb89(%917: f32, %918: f32, %919: f32):
        %920 = arith.mulf %917, %918 : f32
        linalg.yield %920 : f32
      } -> tensor<1x28x2816xf32>
      %921 = tensor.empty() : tensor<2816x1024xf32>
      %922 = linalg.transpose ins(%22:tensor<1024x2816xf32>) outs(%921:tensor<2816x1024xf32>) permutation = [1, 0]
      %923 = tensor.empty() : tensor<1x28x1024xf32>
      %924 = arith.constant 0.000000e+00 : f32
      %925 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%924 : f32) outs(%923 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %926 = linalg.matmul {prov.region_id = "matmul_17", prov.dispatch_id = "matmul_17", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%916, %922 : tensor<1x28x2816xf32>, tensor<2816x1024xf32>) outs(%925 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %927 = tensor.empty() : tensor<1x28x1024xf32>
      %928 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%820, %926 : tensor<1x1x1024xf32>, tensor<1x28x1024xf32>) outs(%927 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_22", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb90(%929: f32, %930: f32, %931: f32):
        %932 = arith.mulf %929, %930 : f32
        linalg.yield %932 : f32
      } -> tensor<1x28x1024xf32>
      %933 = tensor.empty() : tensor<1x28x1024xf32>
      %934 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%814, %928 : tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) outs(%933 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_22", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb91(%935: f32, %936: f32, %937: f32):
        %938 = arith.addf %935, %936 : f32
        linalg.yield %938 : f32
      } -> tensor<1x28x1024xf32>
      %939 = tensor.empty() : tensor<1x64x4x128xf32>
      %940 = linalg.transpose ins(%62:tensor<1x4x64x128xf32>) outs(%939:tensor<1x64x4x128xf32>) permutation = [0, 2, 1, 3]
      %941 = tensor.empty() : tensor<1x64x4x128xf32>
      %942 = linalg.transpose ins(%63:tensor<1x4x64x128xf32>) outs(%941:tensor<1x64x4x128xf32>) permutation = [0, 2, 1, 3]
      %943 = tensor.empty() : tensor<1x2048xf32>
      %944 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%242 : tensor<1x2048xf32>) outs(%943 : tensor<1x2048xf32>) attrs =  {prov.region_id = "silu_5", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb92(%945: f32, %946: f32):
        %947 = arith.constant 1.000000e+00 : f32
        %948 = arith.negf %945 : f32
        %949 = math.exp %948 : f32
        %950 = arith.addf %947, %949 : f32
        %951 = arith.divf %947, %950 : f32
        %952 = arith.mulf %945, %951 : f32
        linalg.yield %952 : f32
      } -> tensor<1x2048xf32>
      %953 = tensor.empty() : tensor<2048x9216xf32>
      %954 = linalg.transpose ins(%43:tensor<9216x2048xf32>) outs(%953:tensor<2048x9216xf32>) permutation = [1, 0]
      %955 = tensor.empty() : tensor<1x9216xf32>
      %956 = arith.constant 0.000000e+00 : f32
      %957 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%956 : f32) outs(%955 : tensor<1x9216xf32>) -> tensor<1x9216xf32>
      %958 = linalg.matmul {prov.region_id = "matmul_18", prov.dispatch_id = "matmul_18", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%944, %954 : tensor<1x2048xf32>, tensor<2048x9216xf32>) outs(%957 : tensor<1x9216xf32>) -> tensor<1x9216xf32>
      %959 = tensor.empty() : tensor<1x9216xf32>
      %960 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%958, %44 : tensor<1x9216xf32>, tensor<9216xf32>) outs(%959 : tensor<1x9216xf32>) attrs =  {prov.region_id = "add_23", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb93(%961: f32, %962: f32, %963: f32):
        %964 = arith.addf %961, %962 : f32
        linalg.yield %964 : f32
      } -> tensor<1x9216xf32>
      %965 = "tensor.extract_slice"(%960) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %966 = "tensor.extract_slice"(%960) <{static_offsets = array<i64: 0, 1024>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %967 = "tensor.extract_slice"(%960) <{static_offsets = array<i64: 0, 2048>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %968 = "tensor.extract_slice"(%960) <{static_offsets = array<i64: 0, 3072>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %969 = "tensor.extract_slice"(%960) <{static_offsets = array<i64: 0, 4096>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %970 = "tensor.extract_slice"(%960) <{static_offsets = array<i64: 0, 5120>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %971 = "tensor.extract_slice"(%960) <{static_offsets = array<i64: 0, 6144>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %972 = "tensor.extract_slice"(%960) <{static_offsets = array<i64: 0, 7168>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %973 = "tensor.extract_slice"(%960) <{static_offsets = array<i64: 0, 8192>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_2", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x9216xf32>) -> tensor<1x1024xf32>
      %974 = tensor.collapse_shape %967 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %975 = tensor.expand_shape %974 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_16", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %976 = tensor.empty() : tensor<1x28x1024xf32>
      %977 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%934 : tensor<1x28x1024xf32>) outs(%976 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "pow_6", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb94(%978: f32, %979: f32):
        %980 = arith.constant 2.000000e+00 : f32
        %981 = math.powf %978, %980 : f32
        linalg.yield %981 : f32
      } -> tensor<1x28x1024xf32>
      %982 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %983 = tensor.splat %982 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32>
      %984 = linalg.reduce ins(%977:tensor<1x28x1024xf32>) outs(%983:tensor<1x28xf32>) dimensions = [2]
      (%985: f32, %986: f32) {
        %987 = arith.addf %985, %986 : f32
        linalg.yield %987 : f32
      }
      %988 = arith.constant {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.024000e+03 : f32
      %989 = tensor.splat %988 {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32>
      %990 = tensor.empty() : tensor<1x28xf32>
      %991 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%984, %989 : tensor<1x28xf32>, tensor<1x28xf32>) outs(%990 : tensor<1x28xf32>) attrs =  {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb95(%992: f32, %993: f32, %994: f32):
        %995 = arith.divf %992, %993 : f32
        linalg.yield %995 : f32
      } -> tensor<1x28xf32>
      %996 = tensor.collapse_shape %991 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32> into tensor<28xf32>
      %997 = tensor.expand_shape %996 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1] {prov.region_id = "reduce_6", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<28xf32> into tensor<1x28x1xf32>
      %998 = arith.constant {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %999 = tensor.splat %998 {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x1xf32>
      %1000 = tensor.empty() : tensor<1x28x1xf32>
      %1001 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%997, %999 : tensor<1x28x1xf32>, tensor<1x28x1xf32>) outs(%1000 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "add_24", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb96(%1002: f32, %1003: f32, %1004: f32):
        %1005 = arith.addf %1002, %1003 : f32
        linalg.yield %1005 : f32
      } -> tensor<1x28x1xf32>
      %1006 = tensor.empty() : tensor<1x28x1xf32>
      %1007 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1001 : tensor<1x28x1xf32>) outs(%1006 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "rsqrt_6", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb97(%1008: f32, %1009: f32):
        %1010 = math.rsqrt %1008 : f32
        linalg.yield %1010 : f32
      } -> tensor<1x28x1xf32>
      %1011 = tensor.empty() : tensor<1x28x1024xf32>
      %1012 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%934, %1007 : tensor<1x28x1024xf32>, tensor<1x28x1xf32>) outs(%1011 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_23", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb98(%1013: f32, %1014: f32, %1015: f32):
        %1016 = arith.mulf %1013, %1014 : f32
        linalg.yield %1016 : f32
      } -> tensor<1x28x1024xf32>
      %1017 = func.call @aten_type_as_default_wl0(%1012, %934) {prov.region_id = "aten_type_as_default_3", prov.dispatch_id = "aten_type_as_default_3"} : (tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %1018 = tensor.empty() : tensor<1x28x1024xf32>
      %1019 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1017, %26 : tensor<1x28x1024xf32>, tensor<1024xf32>) outs(%1018 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_24", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb99(%1020: f32, %1021: f32, %1022: f32):
        %1023 = arith.mulf %1020, %1021 : f32
        linalg.yield %1023 : f32
      } -> tensor<1x28x1024xf32>
      %1024 = tensor.collapse_shape %966 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %1025 = tensor.expand_shape %1024 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_17", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %1026 = arith.constant {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
      %1027 = tensor.splat %1026 {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1024xf32>
      %1028 = tensor.empty() : tensor<1x1x1024xf32>
      %1029 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1025, %1027 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%1028 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_25", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb100(%1030: f32, %1031: f32, %1032: f32):
        %1033 = arith.addf %1030, %1031 : f32
        linalg.yield %1033 : f32
      } -> tensor<1x1x1024xf32>
      %1034 = tensor.empty() : tensor<1x28x1024xf32>
      %1035 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1019, %1029 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%1034 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_25", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb101(%1036: f32, %1037: f32, %1038: f32):
        %1039 = arith.mulf %1036, %1037 : f32
        linalg.yield %1039 : f32
      } -> tensor<1x28x1024xf32>
      %1040 = tensor.collapse_shape %965 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %1041 = tensor.expand_shape %1040 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_18", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %1042 = tensor.empty() : tensor<1x28x1024xf32>
      %1043 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1035, %1041 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%1042 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_26", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb102(%1044: f32, %1045: f32, %1046: f32):
        %1047 = arith.addf %1044, %1045 : f32
        linalg.yield %1047 : f32
      } -> tensor<1x28x1024xf32>
      %1048 = tensor.empty() : tensor<1024x1024xf32>
      %1049 = linalg.transpose ins(%27:tensor<1024x1024xf32>) outs(%1048:tensor<1024x1024xf32>) permutation = [1, 0]
      %1050 = tensor.empty() : tensor<1x28x1024xf32>
      %1051 = arith.constant 0.000000e+00 : f32
      %1052 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1051 : f32) outs(%1050 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %1053 = linalg.matmul {prov.region_id = "matmul_19", prov.dispatch_id = "matmul_19", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1043, %1049 : tensor<1x28x1024xf32>, tensor<1024x1024xf32>) outs(%1052 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %1054 = tensor.collapse_shape %1053 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x1024xf32> into tensor<28672xf32>
      %1055 = tensor.expand_shape %1054 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_11", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
      %1056 = tensor.empty() : tensor<1024x1024xf32>
      %1057 = linalg.transpose ins(%28:tensor<1024x1024xf32>) outs(%1056:tensor<1024x1024xf32>) permutation = [1, 0]
      %1058 = tensor.empty() : tensor<1x28x1024xf32>
      %1059 = arith.constant 0.000000e+00 : f32
      %1060 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1059 : f32) outs(%1058 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %1061 = linalg.matmul {prov.region_id = "matmul_20", prov.dispatch_id = "matmul_20", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1043, %1057 : tensor<1x28x1024xf32>, tensor<1024x1024xf32>) outs(%1060 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %1062 = tensor.collapse_shape %1061 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x1024xf32> into tensor<28672xf32>
      %1063 = tensor.expand_shape %1062 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 28, 4, 128, 2] {prov.region_id = "view_12", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x4x128x2xf32>
      %1064 = "tensor.extract_slice"(%1063) <{static_offsets = array<i64: 0, 0, 0, 0, 0>, static_sizes = array<i64: 1, 28, 4, 128, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<1x28x4x128x2xf32>) -> tensor<1x28x4x128x1xf32>
      %1065 = tensor.collapse_shape %1064 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x28x4x128x1xf32> into tensor<14336xf32>
      %1066 = tensor.expand_shape %1065 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 4, 128] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<14336xf32> into tensor<1x28x4x128xf32>
      %1067 = "tensor.extract_slice"(%1063) <{static_offsets = array<i64: 0, 0, 0, 0, 1>, static_sizes = array<i64: 1, 28, 4, 128, 1>, static_strides = array<i64: 1, 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "slice", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : (tensor<1x28x4x128x2xf32>) -> tensor<1x28x4x128x1xf32>
      %1068 = tensor.collapse_shape %1067 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<1x28x4x128x1xf32> into tensor<14336xf32>
      %1069 = tensor.expand_shape %1068 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 4, 128] {prov.region_id = "split_3", prov.family = "layout", prov._pattern_hint = "unbind", prov.op = "reshape", prov.aten = "aten.unbind.int", prov.orig_dtype = "float32"} : tensor<14336xf32> into tensor<1x28x4x128xf32>
      %1070 = tensor.empty() : tensor<1x28x8x128xf32>
      %1071 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1055 : tensor<1x28x8x128xf32>) outs(%1070 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "pow_7", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb103(%1072: f32, %1073: f32):
        %1074 = arith.constant 2.000000e+00 : f32
        %1075 = math.powf %1072, %1074 : f32
        linalg.yield %1075 : f32
      } -> tensor<1x28x8x128xf32>
      %1076 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1077 = tensor.splat %1076 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x8xf32>
      %1078 = linalg.reduce ins(%1071:tensor<1x28x8x128xf32>) outs(%1077:tensor<1x28x8xf32>) dimensions = [3]
      (%1079: f32, %1080: f32) {
        %1081 = arith.addf %1079, %1080 : f32
        linalg.yield %1081 : f32
      }
      %1082 = arith.constant {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %1083 = tensor.splat %1082 {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x8xf32>
      %1084 = tensor.empty() : tensor<1x28x8xf32>
      %1085 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1078, %1083 : tensor<1x28x8xf32>, tensor<1x28x8xf32>) outs(%1084 : tensor<1x28x8xf32>) attrs =  {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb104(%1086: f32, %1087: f32, %1088: f32):
        %1089 = arith.divf %1086, %1087 : f32
        linalg.yield %1089 : f32
      } -> tensor<1x28x8xf32>
      %1090 = tensor.collapse_shape %1085 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x8xf32> into tensor<224xf32>
      %1091 = tensor.expand_shape %1090 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "reduce_7", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x28x8x1xf32>
      %1092 = arith.constant {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %1093 = tensor.splat %1092 {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x8x1xf32>
      %1094 = tensor.empty() : tensor<1x28x8x1xf32>
      %1095 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1091, %1093 : tensor<1x28x8x1xf32>, tensor<1x28x8x1xf32>) outs(%1094 : tensor<1x28x8x1xf32>) attrs =  {prov.region_id = "add_27", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb105(%1096: f32, %1097: f32, %1098: f32):
        %1099 = arith.addf %1096, %1097 : f32
        linalg.yield %1099 : f32
      } -> tensor<1x28x8x1xf32>
      %1100 = tensor.empty() : tensor<1x28x8x1xf32>
      %1101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1095 : tensor<1x28x8x1xf32>) outs(%1100 : tensor<1x28x8x1xf32>) attrs =  {prov.region_id = "rsqrt_7", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb106(%1102: f32, %1103: f32):
        %1104 = math.rsqrt %1102 : f32
        linalg.yield %1104 : f32
      } -> tensor<1x28x8x1xf32>
      %1105 = tensor.empty() : tensor<1x28x8x128xf32>
      %1106 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1055, %1101 : tensor<1x28x8x128xf32>, tensor<1x28x8x1xf32>) outs(%1105 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_26", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb107(%1107: f32, %1108: f32, %1109: f32):
        %1110 = arith.mulf %1107, %1108 : f32
        linalg.yield %1110 : f32
      } -> tensor<1x28x8x128xf32>
      %1111 = func.call @aten_type_as_default_1_wl1(%1106, %1055) {prov.region_id = "aten_type_as_default_1_2", prov.dispatch_id = "aten_type_as_default_1_2"} : (tensor<1x28x8x128xf32>, tensor<1x28x8x128xf32>) -> tensor<1x28x8x128xf32>
      %1112 = tensor.empty() : tensor<1x28x8x128xf32>
      %1113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1111, %30 : tensor<1x28x8x128xf32>, tensor<128xf32>) outs(%1112 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_27", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb108(%1114: f32, %1115: f32, %1116: f32):
        %1117 = arith.mulf %1114, %1115 : f32
        linalg.yield %1117 : f32
      } -> tensor<1x28x8x128xf32>
      %1118 = tensor.empty() : tensor<1x28x4x128xf32>
      %1119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1066 : tensor<1x28x4x128xf32>) outs(%1118 : tensor<1x28x4x128xf32>) attrs =  {prov.region_id = "pow_8", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb109(%1120: f32, %1121: f32):
        %1122 = arith.constant 2.000000e+00 : f32
        %1123 = math.powf %1120, %1122 : f32
        linalg.yield %1123 : f32
      } -> tensor<1x28x4x128xf32>
      %1124 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1125 = tensor.splat %1124 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x4xf32>
      %1126 = linalg.reduce ins(%1119:tensor<1x28x4x128xf32>) outs(%1125:tensor<1x28x4xf32>) dimensions = [3]
      (%1127: f32, %1128: f32) {
        %1129 = arith.addf %1127, %1128 : f32
        linalg.yield %1129 : f32
      }
      %1130 = arith.constant {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %1131 = tensor.splat %1130 {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x4xf32>
      %1132 = tensor.empty() : tensor<1x28x4xf32>
      %1133 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1126, %1131 : tensor<1x28x4xf32>, tensor<1x28x4xf32>) outs(%1132 : tensor<1x28x4xf32>) attrs =  {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb110(%1134: f32, %1135: f32, %1136: f32):
        %1137 = arith.divf %1134, %1135 : f32
        linalg.yield %1137 : f32
      } -> tensor<1x28x4xf32>
      %1138 = tensor.collapse_shape %1133 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x4xf32> into tensor<112xf32>
      %1139 = tensor.expand_shape %1138 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 4, 1] {prov.region_id = "reduce_8", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<112xf32> into tensor<1x28x4x1xf32>
      %1140 = arith.constant {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %1141 = tensor.splat %1140 {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x4x1xf32>
      %1142 = tensor.empty() : tensor<1x28x4x1xf32>
      %1143 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1139, %1141 : tensor<1x28x4x1xf32>, tensor<1x28x4x1xf32>) outs(%1142 : tensor<1x28x4x1xf32>) attrs =  {prov.region_id = "add_28", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb111(%1144: f32, %1145: f32, %1146: f32):
        %1147 = arith.addf %1144, %1145 : f32
        linalg.yield %1147 : f32
      } -> tensor<1x28x4x1xf32>
      %1148 = tensor.empty() : tensor<1x28x4x1xf32>
      %1149 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1143 : tensor<1x28x4x1xf32>) outs(%1148 : tensor<1x28x4x1xf32>) attrs =  {prov.region_id = "rsqrt_8", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb112(%1150: f32, %1151: f32):
        %1152 = math.rsqrt %1150 : f32
        linalg.yield %1152 : f32
      } -> tensor<1x28x4x1xf32>
      %1153 = tensor.empty() : tensor<1x28x4x128xf32>
      %1154 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1066, %1149 : tensor<1x28x4x128xf32>, tensor<1x28x4x1xf32>) outs(%1153 : tensor<1x28x4x128xf32>) attrs =  {prov.region_id = "mul_28", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb113(%1155: f32, %1156: f32, %1157: f32):
        %1158 = arith.mulf %1155, %1156 : f32
        linalg.yield %1158 : f32
      } -> tensor<1x28x4x128xf32>
      %1159 = func.call @aten_type_as_default_2_wl2(%1154, %1066) {prov.region_id = "aten_type_as_default_2_1", prov.dispatch_id = "aten_type_as_default_2_1"} : (tensor<1x28x4x128xf32>, tensor<1x28x4x128xf32>) -> tensor<1x28x4x128xf32>
      %1160 = tensor.empty() : tensor<1x28x4x128xf32>
      %1161 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1159, %31 : tensor<1x28x4x128xf32>, tensor<128xf32>) outs(%1160 : tensor<1x28x4x128xf32>) attrs =  {prov.region_id = "mul_29", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb114(%1162: f32, %1163: f32, %1164: f32):
        %1165 = arith.mulf %1162, %1163 : f32
        linalg.yield %1165 : f32
      } -> tensor<1x28x4x128xf32>
      %1166 = tensor.collapse_shape %1161 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x28x4x128xf32> into tensor<14336xf32>
      %1167 = tensor.expand_shape %1166 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 28, 4, 1, 128] {prov.region_id = "unsqueeze_19", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<14336xf32> into tensor<1x28x4x1x128xf32>
      %1168 = tensor.empty() : tensor<1x28x4x2x128xf32>
      %1169 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1167 : tensor<1x28x4x1x128xf32>) outs(%1168 : tensor<1x28x4x2x128xf32>) attrs =  {prov.region_id = "expand_6", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb115(%1170: f32, %1171: f32):
        linalg.yield %1170 : f32
      } -> tensor<1x28x4x2x128xf32>
      %1172 = tensor.collapse_shape %1169 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x28x4x2x128xf32> into tensor<28672xf32>
      %1173 = tensor.expand_shape %1172 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_13", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
      %1174 = tensor.collapse_shape %1069 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x28x4x128xf32> into tensor<14336xf32>
      %1175 = tensor.expand_shape %1174 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 28, 4, 1, 128] {prov.region_id = "unsqueeze_20", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<14336xf32> into tensor<1x28x4x1x128xf32>
      %1176 = tensor.empty() : tensor<1x28x4x2x128xf32>
      %1177 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1175 : tensor<1x28x4x1x128xf32>) outs(%1176 : tensor<1x28x4x2x128xf32>) attrs =  {prov.region_id = "expand_7", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb116(%1178: f32, %1179: f32):
        linalg.yield %1178 : f32
      } -> tensor<1x28x4x2x128xf32>
      %1180 = tensor.collapse_shape %1177 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x28x4x2x128xf32> into tensor<28672xf32>
      %1181 = tensor.expand_shape %1180 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_14", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
      %1182 = tensor.empty() : tensor<1x8x28x128xf32>
      %1183 = linalg.transpose ins(%1113:tensor<1x28x8x128xf32>) outs(%1182:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
      %1184 = tensor.empty() : tensor<1x8x28x128xf32>
      %1185 = linalg.transpose ins(%1173:tensor<1x28x8x128xf32>) outs(%1184:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
      %1186 = tensor.empty() : tensor<1x8x28x128xf32>
      %1187 = linalg.transpose ins(%1181:tensor<1x28x8x128xf32>) outs(%1186:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
      %1188 = tensor.empty() : tensor<1x8x128x28xf32>
      %1189 = linalg.transpose ins(%1185:tensor<1x8x28x128xf32>) outs(%1188:tensor<1x8x128x28xf32>) permutation = [0, 1, 3, 2]
      %1190 = arith.constant {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1191 = tensor.splat %1190 {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x28x28xf32>
      %1192 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1183, %1189 : tensor<1x8x28x128xf32>, tensor<1x8x128x28xf32>) outs(%1191 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "matmul_21", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb117(%1193: f32, %1194: f32, %1195: f32):
        %1196 = arith.mulf %1193, %1194 : f32
        %1197 = arith.addf %1195, %1196 : f32
        linalg.yield %1197 : f32
      } -> tensor<1x8x28x28xf32>
      %1198 = arith.constant {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.0883883461 : f32
      %1199 = tensor.splat %1198 {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x28x28xf32>
      %1200 = tensor.empty() : tensor<1x8x28x28xf32>
      %1201 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1192, %1199 : tensor<1x8x28x28xf32>, tensor<1x8x28x28xf32>) outs(%1200 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "mul_30", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb118(%1202: f32, %1203: f32, %1204: f32):
        %1205 = arith.mulf %1202, %1203 : f32
        linalg.yield %1205 : f32
      } -> tensor<1x8x28x28xf32>
      %1206 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0xff800000 : f32
      %1207 = tensor.splat %1206 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x28xf32>
      %1208 = linalg.reduce ins(%1201:tensor<1x8x28x28xf32>) outs(%1207:tensor<1x8x28xf32>) dimensions = [3]
      (%1209: f32, %1210: f32) {
        %1211 = arith.maximumf %1209, %1210 : f32
        linalg.yield %1211 : f32
      }
      %1212 = tensor.collapse_shape %1208 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x28xf32> into tensor<224xf32>
      %1213 = tensor.expand_shape %1212 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x8x28x1xf32>
      %1214 = tensor.empty() : tensor<1x8x28x28xf32>
      %1215 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1201, %1213 : tensor<1x8x28x28xf32>, tensor<1x8x28x1xf32>) outs(%1214 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb119(%1216: f32, %1217: f32, %1218: f32):
        %1219 = arith.subf %1216, %1217 : f32
        linalg.yield %1219 : f32
      } -> tensor<1x8x28x28xf32>
      %1220 = tensor.empty() : tensor<1x8x28x28xf32>
      %1221 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1215 : tensor<1x8x28x28xf32>) outs(%1220 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb120(%1222: f32, %1223: f32):
        %1224 = math.exp %1222 : f32
        linalg.yield %1224 : f32
      } -> tensor<1x8x28x28xf32>
      %1225 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1226 = tensor.splat %1225 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x28xf32>
      %1227 = linalg.reduce ins(%1221:tensor<1x8x28x28xf32>) outs(%1226:tensor<1x8x28xf32>) dimensions = [3]
      (%1228: f32, %1229: f32) {
        %1230 = arith.addf %1228, %1229 : f32
        linalg.yield %1230 : f32
      }
      %1231 = tensor.collapse_shape %1227 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x28xf32> into tensor<224xf32>
      %1232 = tensor.expand_shape %1231 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x8x28x1xf32>
      %1233 = tensor.empty() : tensor<1x8x28x28xf32>
      %1234 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1221, %1232 : tensor<1x8x28x28xf32>, tensor<1x8x28x1xf32>) outs(%1233 : tensor<1x8x28x28xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb121(%1235: f32, %1236: f32, %1237: f32):
        %1238 = arith.divf %1235, %1236 : f32
        linalg.yield %1238 : f32
      } -> tensor<1x8x28x28xf32>
      %1239 = func.call @aten_type_as_default_3_wl3(%1234, %1183) {prov.region_id = "aten_type_as_default_3_1", prov.dispatch_id = "aten_type_as_default_3_1"} : (tensor<1x8x28x28xf32>, tensor<1x8x28x128xf32>) -> tensor<1x8x28x28xf32>
      %1240 = arith.constant {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1241 = tensor.splat %1240 {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x28x128xf32>
      %1242 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1239, %1187 : tensor<1x8x28x28xf32>, tensor<1x8x28x128xf32>) outs(%1241 : tensor<1x8x28x128xf32>) attrs =  {prov.region_id = "matmul_22", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb122(%1243: f32, %1244: f32, %1245: f32):
        %1246 = arith.mulf %1243, %1244 : f32
        %1247 = arith.addf %1245, %1246 : f32
        linalg.yield %1247 : f32
      } -> tensor<1x8x28x128xf32>
      %1248 = tensor.empty() : tensor<1x28x8x128xf32>
      %1249 = linalg.transpose ins(%1242:tensor<1x8x28x128xf32>) outs(%1248:tensor<1x28x8x128xf32>) permutation = [0, 2, 1, 3]
      %1250 = tensor.collapse_shape %1249 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
      %1251 = tensor.expand_shape %1250 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_15", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x1024xf32>
      %1252 = tensor.empty() : tensor<1024x1024xf32>
      %1253 = linalg.transpose ins(%29:tensor<1024x1024xf32>) outs(%1252:tensor<1024x1024xf32>) permutation = [1, 0]
      %1254 = tensor.empty() : tensor<1x28x1024xf32>
      %1255 = arith.constant 0.000000e+00 : f32
      %1256 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1255 : f32) outs(%1254 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %1257 = linalg.matmul {prov.region_id = "matmul_23", prov.dispatch_id = "matmul_23", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1251, %1253 : tensor<1x28x1024xf32>, tensor<1024x1024xf32>) outs(%1256 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %1258 = tensor.empty() : tensor<1x28x1024xf32>
      %1259 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%975, %1257 : tensor<1x1x1024xf32>, tensor<1x28x1024xf32>) outs(%1258 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_31", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb123(%1260: f32, %1261: f32, %1262: f32):
        %1263 = arith.mulf %1260, %1261 : f32
        linalg.yield %1263 : f32
      } -> tensor<1x28x1024xf32>
      %1264 = tensor.empty() : tensor<1x28x1024xf32>
      %1265 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%934, %1259 : tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) outs(%1264 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_29", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb124(%1266: f32, %1267: f32, %1268: f32):
        %1269 = arith.addf %1266, %1267 : f32
        linalg.yield %1269 : f32
      } -> tensor<1x28x1024xf32>
      %1270 = tensor.collapse_shape %970 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %1271 = tensor.expand_shape %1270 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_21", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %1272 = tensor.empty() : tensor<1x28x1024xf32>
      %1273 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1265 : tensor<1x28x1024xf32>) outs(%1272 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "pow_9", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb125(%1274: f32, %1275: f32):
        %1276 = arith.constant 2.000000e+00 : f32
        %1277 = math.powf %1274, %1276 : f32
        linalg.yield %1277 : f32
      } -> tensor<1x28x1024xf32>
      %1278 = arith.constant {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1279 = tensor.splat %1278 {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32>
      %1280 = linalg.reduce ins(%1273:tensor<1x28x1024xf32>) outs(%1279:tensor<1x28xf32>) dimensions = [2]
      (%1281: f32, %1282: f32) {
        %1283 = arith.addf %1281, %1282 : f32
        linalg.yield %1283 : f32
      }
      %1284 = arith.constant {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.024000e+03 : f32
      %1285 = tensor.splat %1284 {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32>
      %1286 = tensor.empty() : tensor<1x28xf32>
      %1287 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1280, %1285 : tensor<1x28xf32>, tensor<1x28xf32>) outs(%1286 : tensor<1x28xf32>) attrs =  {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb126(%1288: f32, %1289: f32, %1290: f32):
        %1291 = arith.divf %1288, %1289 : f32
        linalg.yield %1291 : f32
      } -> tensor<1x28xf32>
      %1292 = tensor.collapse_shape %1287 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32> into tensor<28xf32>
      %1293 = tensor.expand_shape %1292 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1] {prov.region_id = "reduce_9", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<28xf32> into tensor<1x28x1xf32>
      %1294 = arith.constant {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %1295 = tensor.splat %1294 {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x1xf32>
      %1296 = tensor.empty() : tensor<1x28x1xf32>
      %1297 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1293, %1295 : tensor<1x28x1xf32>, tensor<1x28x1xf32>) outs(%1296 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "add_30", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb127(%1298: f32, %1299: f32, %1300: f32):
        %1301 = arith.addf %1298, %1299 : f32
        linalg.yield %1301 : f32
      } -> tensor<1x28x1xf32>
      %1302 = tensor.empty() : tensor<1x28x1xf32>
      %1303 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1297 : tensor<1x28x1xf32>) outs(%1302 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "rsqrt_9", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb128(%1304: f32, %1305: f32):
        %1306 = math.rsqrt %1304 : f32
        linalg.yield %1306 : f32
      } -> tensor<1x28x1xf32>
      %1307 = tensor.empty() : tensor<1x28x1024xf32>
      %1308 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1265, %1303 : tensor<1x28x1024xf32>, tensor<1x28x1xf32>) outs(%1307 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_32", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb129(%1309: f32, %1310: f32, %1311: f32):
        %1312 = arith.mulf %1309, %1310 : f32
        linalg.yield %1312 : f32
      } -> tensor<1x28x1024xf32>
      %1313 = func.call @aten_type_as_default_wl0(%1308, %1265) {prov.region_id = "aten_type_as_default_4", prov.dispatch_id = "aten_type_as_default_4"} : (tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %1314 = tensor.empty() : tensor<1x28x1024xf32>
      %1315 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1313, %32 : tensor<1x28x1024xf32>, tensor<1024xf32>) outs(%1314 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_33", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb130(%1316: f32, %1317: f32, %1318: f32):
        %1319 = arith.mulf %1316, %1317 : f32
        linalg.yield %1319 : f32
      } -> tensor<1x28x1024xf32>
      %1320 = tensor.collapse_shape %969 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %1321 = tensor.expand_shape %1320 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_22", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %1322 = arith.constant {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
      %1323 = tensor.splat %1322 {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1024xf32>
      %1324 = tensor.empty() : tensor<1x1x1024xf32>
      %1325 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1321, %1323 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%1324 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_31", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb131(%1326: f32, %1327: f32, %1328: f32):
        %1329 = arith.addf %1326, %1327 : f32
        linalg.yield %1329 : f32
      } -> tensor<1x1x1024xf32>
      %1330 = tensor.empty() : tensor<1x28x1024xf32>
      %1331 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1315, %1325 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%1330 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_34", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb132(%1332: f32, %1333: f32, %1334: f32):
        %1335 = arith.mulf %1332, %1333 : f32
        linalg.yield %1335 : f32
      } -> tensor<1x28x1024xf32>
      %1336 = tensor.collapse_shape %968 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %1337 = tensor.expand_shape %1336 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_23", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %1338 = tensor.empty() : tensor<1x28x1024xf32>
      %1339 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1331, %1337 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%1338 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_32", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb133(%1340: f32, %1341: f32, %1342: f32):
        %1343 = arith.addf %1340, %1341 : f32
        linalg.yield %1343 : f32
      } -> tensor<1x28x1024xf32>
      %1344 = tensor.empty() : tensor<1024x1024xf32>
      %1345 = linalg.transpose ins(%34:tensor<1024x1024xf32>) outs(%1344:tensor<1024x1024xf32>) permutation = [1, 0]
      %1346 = tensor.empty() : tensor<1x28x1024xf32>
      %1347 = arith.constant 0.000000e+00 : f32
      %1348 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1347 : f32) outs(%1346 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %1349 = linalg.matmul {prov.region_id = "matmul_24", prov.dispatch_id = "matmul_24", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1339, %1345 : tensor<1x28x1024xf32>, tensor<1024x1024xf32>) outs(%1348 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %1350 = tensor.collapse_shape %1349 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x1024xf32> into tensor<28672xf32>
      %1351 = tensor.expand_shape %1350 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 128] {prov.region_id = "view_16", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x8x128xf32>
      %1352 = tensor.empty() : tensor<1x28x8x128xf32>
      %1353 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1351 : tensor<1x28x8x128xf32>) outs(%1352 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "pow_10", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb134(%1354: f32, %1355: f32):
        %1356 = arith.constant 2.000000e+00 : f32
        %1357 = math.powf %1354, %1356 : f32
        linalg.yield %1357 : f32
      } -> tensor<1x28x8x128xf32>
      %1358 = arith.constant {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1359 = tensor.splat %1358 {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x8xf32>
      %1360 = linalg.reduce ins(%1353:tensor<1x28x8x128xf32>) outs(%1359:tensor<1x28x8xf32>) dimensions = [3]
      (%1361: f32, %1362: f32) {
        %1363 = arith.addf %1361, %1362 : f32
        linalg.yield %1363 : f32
      }
      %1364 = arith.constant {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.280000e+02 : f32
      %1365 = tensor.splat %1364 {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x8xf32>
      %1366 = tensor.empty() : tensor<1x28x8xf32>
      %1367 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1360, %1365 : tensor<1x28x8xf32>, tensor<1x28x8xf32>) outs(%1366 : tensor<1x28x8xf32>) attrs =  {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb135(%1368: f32, %1369: f32, %1370: f32):
        %1371 = arith.divf %1368, %1369 : f32
        linalg.yield %1371 : f32
      } -> tensor<1x28x8xf32>
      %1372 = tensor.collapse_shape %1367 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28x8xf32> into tensor<224xf32>
      %1373 = tensor.expand_shape %1372 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 28, 8, 1] {prov.region_id = "reduce_10", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x28x8x1xf32>
      %1374 = arith.constant {prov.region_id = "add_33", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %1375 = tensor.splat %1374 {prov.region_id = "add_33", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x8x1xf32>
      %1376 = tensor.empty() : tensor<1x28x8x1xf32>
      %1377 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1373, %1375 : tensor<1x28x8x1xf32>, tensor<1x28x8x1xf32>) outs(%1376 : tensor<1x28x8x1xf32>) attrs =  {prov.region_id = "add_33", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb136(%1378: f32, %1379: f32, %1380: f32):
        %1381 = arith.addf %1378, %1379 : f32
        linalg.yield %1381 : f32
      } -> tensor<1x28x8x1xf32>
      %1382 = tensor.empty() : tensor<1x28x8x1xf32>
      %1383 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1377 : tensor<1x28x8x1xf32>) outs(%1382 : tensor<1x28x8x1xf32>) attrs =  {prov.region_id = "rsqrt_10", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb137(%1384: f32, %1385: f32):
        %1386 = math.rsqrt %1384 : f32
        linalg.yield %1386 : f32
      } -> tensor<1x28x8x1xf32>
      %1387 = tensor.empty() : tensor<1x28x8x128xf32>
      %1388 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1351, %1383 : tensor<1x28x8x128xf32>, tensor<1x28x8x1xf32>) outs(%1387 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_35", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb138(%1389: f32, %1390: f32, %1391: f32):
        %1392 = arith.mulf %1389, %1390 : f32
        linalg.yield %1392 : f32
      } -> tensor<1x28x8x128xf32>
      %1393 = func.call @aten_type_as_default_1_wl1(%1388, %1351) {prov.region_id = "aten_type_as_default_1_3", prov.dispatch_id = "aten_type_as_default_1_3"} : (tensor<1x28x8x128xf32>, tensor<1x28x8x128xf32>) -> tensor<1x28x8x128xf32>
      %1394 = tensor.empty() : tensor<1x28x8x128xf32>
      %1395 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1393, %37 : tensor<1x28x8x128xf32>, tensor<128xf32>) outs(%1394 : tensor<1x28x8x128xf32>) attrs =  {prov.region_id = "mul_36", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb139(%1396: f32, %1397: f32, %1398: f32):
        %1399 = arith.mulf %1396, %1397 : f32
        linalg.yield %1399 : f32
      } -> tensor<1x28x8x128xf32>
      %1400 = tensor.collapse_shape %940 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x64x4x128xf32> into tensor<32768xf32>
      %1401 = tensor.expand_shape %1400 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 64, 4, 1, 128] {prov.region_id = "unsqueeze_24", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<1x64x4x1x128xf32>
      %1402 = tensor.empty() : tensor<1x64x4x2x128xf32>
      %1403 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1401 : tensor<1x64x4x1x128xf32>) outs(%1402 : tensor<1x64x4x2x128xf32>) attrs =  {prov.region_id = "expand_8", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb140(%1404: f32, %1405: f32):
        linalg.yield %1404 : f32
      } -> tensor<1x64x4x2x128xf32>
      %1406 = tensor.collapse_shape %1403 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x64x4x2x128xf32> into tensor<65536xf32>
      %1407 = tensor.expand_shape %1406 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 128] {prov.region_id = "view_17", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<65536xf32> into tensor<1x64x8x128xf32>
      %1408 = tensor.collapse_shape %942 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x64x4x128xf32> into tensor<32768xf32>
      %1409 = tensor.expand_shape %1408 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] output_shape [1, 64, 4, 1, 128] {prov.region_id = "unsqueeze_25", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<32768xf32> into tensor<1x64x4x1x128xf32>
      %1410 = tensor.empty() : tensor<1x64x4x2x128xf32>
      %1411 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1409 : tensor<1x64x4x1x128xf32>) outs(%1410 : tensor<1x64x4x2x128xf32>) attrs =  {prov.region_id = "expand_9", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32"} {
      ^bb141(%1412: f32, %1413: f32):
        linalg.yield %1412 : f32
      } -> tensor<1x64x4x2x128xf32>
      %1414 = tensor.collapse_shape %1411 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<1x64x4x2x128xf32> into tensor<65536xf32>
      %1415 = tensor.expand_shape %1414 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 8, 128] {prov.region_id = "view_18", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32"} : tensor<65536xf32> into tensor<1x64x8x128xf32>
      %1416 = tensor.empty() : tensor<1x8x28x128xf32>
      %1417 = linalg.transpose ins(%1395:tensor<1x28x8x128xf32>) outs(%1416:tensor<1x8x28x128xf32>) permutation = [0, 2, 1, 3]
      %1418 = tensor.empty() : tensor<1x8x64x128xf32>
      %1419 = linalg.transpose ins(%1407:tensor<1x64x8x128xf32>) outs(%1418:tensor<1x8x64x128xf32>) permutation = [0, 2, 1, 3]
      %1420 = tensor.empty() : tensor<1x8x64x128xf32>
      %1421 = linalg.transpose ins(%1415:tensor<1x64x8x128xf32>) outs(%1420:tensor<1x8x64x128xf32>) permutation = [0, 2, 1, 3]
      %1422 = tensor.empty() : tensor<1x8x128x64xf32>
      %1423 = linalg.transpose ins(%1419:tensor<1x8x64x128xf32>) outs(%1422:tensor<1x8x128x64xf32>) permutation = [0, 1, 3, 2]
      %1424 = arith.constant {prov.region_id = "matmul_25", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1425 = tensor.splat %1424 {prov.region_id = "matmul_25", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x28x64xf32>
      %1426 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1417, %1423 : tensor<1x8x28x128xf32>, tensor<1x8x128x64xf32>) outs(%1425 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "matmul_25", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb142(%1427: f32, %1428: f32, %1429: f32):
        %1430 = arith.mulf %1427, %1428 : f32
        %1431 = arith.addf %1429, %1430 : f32
        linalg.yield %1431 : f32
      } -> tensor<1x8x28x64xf32>
      %1432 = arith.constant {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.0883883461 : f32
      %1433 = tensor.splat %1432 {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x8x28x64xf32>
      %1434 = tensor.empty() : tensor<1x8x28x64xf32>
      %1435 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1426, %1433 : tensor<1x8x28x64xf32>, tensor<1x8x28x64xf32>) outs(%1434 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "mul_37", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb143(%1436: f32, %1437: f32, %1438: f32):
        %1439 = arith.mulf %1436, %1437 : f32
        linalg.yield %1439 : f32
      } -> tensor<1x8x28x64xf32>
      %1440 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0xff800000 : f32
      %1441 = tensor.splat %1440 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x28xf32>
      %1442 = linalg.reduce ins(%1435:tensor<1x8x28x64xf32>) outs(%1441:tensor<1x8x28xf32>) dimensions = [3]
      (%1443: f32, %1444: f32) {
        %1445 = arith.maximumf %1443, %1444 : f32
        linalg.yield %1445 : f32
      }
      %1446 = tensor.collapse_shape %1442 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x28xf32> into tensor<224xf32>
      %1447 = tensor.expand_shape %1446 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x8x28x1xf32>
      %1448 = tensor.empty() : tensor<1x8x28x64xf32>
      %1449 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1435, %1447 : tensor<1x8x28x64xf32>, tensor<1x8x28x1xf32>) outs(%1448 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb144(%1450: f32, %1451: f32, %1452: f32):
        %1453 = arith.subf %1450, %1451 : f32
        linalg.yield %1453 : f32
      } -> tensor<1x8x28x64xf32>
      %1454 = tensor.empty() : tensor<1x8x28x64xf32>
      %1455 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1449 : tensor<1x8x28x64xf32>) outs(%1454 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb145(%1456: f32, %1457: f32):
        %1458 = math.exp %1456 : f32
        linalg.yield %1458 : f32
      } -> tensor<1x8x28x64xf32>
      %1459 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1460 = tensor.splat %1459 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x28xf32>
      %1461 = linalg.reduce ins(%1455:tensor<1x8x28x64xf32>) outs(%1460:tensor<1x8x28xf32>) dimensions = [3]
      (%1462: f32, %1463: f32) {
        %1464 = arith.addf %1462, %1463 : f32
        linalg.yield %1464 : f32
      }
      %1465 = tensor.collapse_shape %1461 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<1x8x28xf32> into tensor<224xf32>
      %1466 = tensor.expand_shape %1465 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 28, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} : tensor<224xf32> into tensor<1x8x28x1xf32>
      %1467 = tensor.empty() : tensor<1x8x28x64xf32>
      %1468 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1455, %1466 : tensor<1x8x28x64xf32>, tensor<1x8x28x1xf32>) outs(%1467 : tensor<1x8x28x64xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten.softmax.int", prov.orig_dtype = "float32"} {
      ^bb146(%1469: f32, %1470: f32, %1471: f32):
        %1472 = arith.divf %1469, %1470 : f32
        linalg.yield %1472 : f32
      } -> tensor<1x8x28x64xf32>
      %1473 = func.call @aten_type_as_default_4_wl4(%1468, %1417) {prov.region_id = "aten_type_as_default_4_1", prov.dispatch_id = "aten_type_as_default_4_1"} : (tensor<1x8x28x64xf32>, tensor<1x8x28x128xf32>) -> tensor<1x8x28x64xf32>
      %1474 = arith.constant {prov.region_id = "matmul_26", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1475 = tensor.splat %1474 {prov.region_id = "matmul_26", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} : tensor<1x8x28x128xf32>
      %1476 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1473, %1421 : tensor<1x8x28x64xf32>, tensor<1x8x64x128xf32>) outs(%1475 : tensor<1x8x28x128xf32>) attrs =  {prov.region_id = "matmul_26", prov.family = "contraction", prov._pattern_hint = "matmul", prov.op = "matmul", prov.aten = "aten.matmul.default", prov.orig_dtype = "float32"} {
      ^bb147(%1477: f32, %1478: f32, %1479: f32):
        %1480 = arith.mulf %1477, %1478 : f32
        %1481 = arith.addf %1479, %1480 : f32
        linalg.yield %1481 : f32
      } -> tensor<1x8x28x128xf32>
      %1482 = tensor.empty() : tensor<1x28x8x128xf32>
      %1483 = linalg.transpose ins(%1476:tensor<1x8x28x128xf32>) outs(%1482:tensor<1x28x8x128xf32>) permutation = [0, 2, 1, 3]
      %1484 = tensor.collapse_shape %1483 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<1x28x8x128xf32> into tensor<28672xf32>
      %1485 = tensor.expand_shape %1484 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1024] {prov.region_id = "view_19", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<28672xf32> into tensor<1x28x1024xf32>
      %1486 = tensor.empty() : tensor<1024x1024xf32>
      %1487 = linalg.transpose ins(%36:tensor<1024x1024xf32>) outs(%1486:tensor<1024x1024xf32>) permutation = [1, 0]
      %1488 = tensor.empty() : tensor<1x28x1024xf32>
      %1489 = arith.constant 0.000000e+00 : f32
      %1490 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1489 : f32) outs(%1488 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %1491 = linalg.matmul {prov.region_id = "matmul_27", prov.dispatch_id = "matmul_27", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1485, %1487 : tensor<1x28x1024xf32>, tensor<1024x1024xf32>) outs(%1490 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %1492 = tensor.empty() : tensor<1x28x1024xf32>
      %1493 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1271, %1491 : tensor<1x1x1024xf32>, tensor<1x28x1024xf32>) outs(%1492 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_38", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb148(%1494: f32, %1495: f32, %1496: f32):
        %1497 = arith.mulf %1494, %1495 : f32
        linalg.yield %1497 : f32
      } -> tensor<1x28x1024xf32>
      %1498 = tensor.empty() : tensor<1x28x1024xf32>
      %1499 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1265, %1493 : tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) outs(%1498 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_34", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb149(%1500: f32, %1501: f32, %1502: f32):
        %1503 = arith.addf %1500, %1501 : f32
        linalg.yield %1503 : f32
      } -> tensor<1x28x1024xf32>
      %1504 = tensor.collapse_shape %973 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_26", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %1505 = tensor.expand_shape %1504 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_26", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %1506 = tensor.empty() : tensor<1x28x1024xf32>
      %1507 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1499 : tensor<1x28x1024xf32>) outs(%1506 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "pow_11", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb150(%1508: f32, %1509: f32):
        %1510 = arith.constant 2.000000e+00 : f32
        %1511 = math.powf %1508, %1510 : f32
        linalg.yield %1511 : f32
      } -> tensor<1x28x1024xf32>
      %1512 = arith.constant {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1513 = tensor.splat %1512 {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32>
      %1514 = linalg.reduce ins(%1507:tensor<1x28x1024xf32>) outs(%1513:tensor<1x28xf32>) dimensions = [2]
      (%1515: f32, %1516: f32) {
        %1517 = arith.addf %1515, %1516 : f32
        linalg.yield %1517 : f32
      }
      %1518 = arith.constant {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.024000e+03 : f32
      %1519 = tensor.splat %1518 {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32>
      %1520 = tensor.empty() : tensor<1x28xf32>
      %1521 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1514, %1519 : tensor<1x28xf32>, tensor<1x28xf32>) outs(%1520 : tensor<1x28xf32>) attrs =  {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb151(%1522: f32, %1523: f32, %1524: f32):
        %1525 = arith.divf %1522, %1523 : f32
        linalg.yield %1525 : f32
      } -> tensor<1x28xf32>
      %1526 = tensor.collapse_shape %1521 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32> into tensor<28xf32>
      %1527 = tensor.expand_shape %1526 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1] {prov.region_id = "reduce_11", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<28xf32> into tensor<1x28x1xf32>
      %1528 = arith.constant {prov.region_id = "add_35", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %1529 = tensor.splat %1528 {prov.region_id = "add_35", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x1xf32>
      %1530 = tensor.empty() : tensor<1x28x1xf32>
      %1531 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1527, %1529 : tensor<1x28x1xf32>, tensor<1x28x1xf32>) outs(%1530 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "add_35", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb152(%1532: f32, %1533: f32, %1534: f32):
        %1535 = arith.addf %1532, %1533 : f32
        linalg.yield %1535 : f32
      } -> tensor<1x28x1xf32>
      %1536 = tensor.empty() : tensor<1x28x1xf32>
      %1537 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1531 : tensor<1x28x1xf32>) outs(%1536 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "rsqrt_11", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb153(%1538: f32, %1539: f32):
        %1540 = math.rsqrt %1538 : f32
        linalg.yield %1540 : f32
      } -> tensor<1x28x1xf32>
      %1541 = tensor.empty() : tensor<1x28x1024xf32>
      %1542 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1499, %1537 : tensor<1x28x1024xf32>, tensor<1x28x1xf32>) outs(%1541 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_39", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb154(%1543: f32, %1544: f32, %1545: f32):
        %1546 = arith.mulf %1543, %1544 : f32
        linalg.yield %1546 : f32
      } -> tensor<1x28x1024xf32>
      %1547 = func.call @aten_type_as_default_wl0(%1542, %1499) {prov.region_id = "aten_type_as_default_5", prov.dispatch_id = "aten_type_as_default_5"} : (tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %1548 = tensor.empty() : tensor<1x28x1024xf32>
      %1549 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1547, %39 : tensor<1x28x1024xf32>, tensor<1024xf32>) outs(%1548 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_40", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb155(%1550: f32, %1551: f32, %1552: f32):
        %1553 = arith.mulf %1550, %1551 : f32
        linalg.yield %1553 : f32
      } -> tensor<1x28x1024xf32>
      %1554 = tensor.collapse_shape %972 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_27", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %1555 = tensor.expand_shape %1554 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_27", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %1556 = arith.constant {prov.region_id = "add_36", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
      %1557 = tensor.splat %1556 {prov.region_id = "add_36", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1024xf32>
      %1558 = tensor.empty() : tensor<1x1x1024xf32>
      %1559 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1555, %1557 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%1558 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_36", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb156(%1560: f32, %1561: f32, %1562: f32):
        %1563 = arith.addf %1560, %1561 : f32
        linalg.yield %1563 : f32
      } -> tensor<1x1x1024xf32>
      %1564 = tensor.empty() : tensor<1x28x1024xf32>
      %1565 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1549, %1559 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%1564 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_41", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb157(%1566: f32, %1567: f32, %1568: f32):
        %1569 = arith.mulf %1566, %1567 : f32
        linalg.yield %1569 : f32
      } -> tensor<1x28x1024xf32>
      %1570 = tensor.collapse_shape %971 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_28", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %1571 = tensor.expand_shape %1570 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_28", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %1572 = tensor.empty() : tensor<1x28x1024xf32>
      %1573 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1565, %1571 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%1572 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_37", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb158(%1574: f32, %1575: f32, %1576: f32):
        %1577 = arith.addf %1574, %1575 : f32
        linalg.yield %1577 : f32
      } -> tensor<1x28x1024xf32>
      %1578 = tensor.empty() : tensor<1024x2816xf32>
      %1579 = linalg.transpose ins(%40:tensor<2816x1024xf32>) outs(%1578:tensor<1024x2816xf32>) permutation = [1, 0]
      %1580 = tensor.empty() : tensor<1x28x2816xf32>
      %1581 = arith.constant 0.000000e+00 : f32
      %1582 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1581 : f32) outs(%1580 : tensor<1x28x2816xf32>) -> tensor<1x28x2816xf32>
      %1583 = linalg.matmul {prov.region_id = "matmul_28", prov.dispatch_id = "matmul_28", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1573, %1579 : tensor<1x28x1024xf32>, tensor<1024x2816xf32>) outs(%1582 : tensor<1x28x2816xf32>) -> tensor<1x28x2816xf32>
      %1584 = tensor.empty() : tensor<1x28x2816xf32>
      %1585 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1583 : tensor<1x28x2816xf32>) outs(%1584 : tensor<1x28x2816xf32>) attrs =  {prov.region_id = "silu_6", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb159(%1586: f32, %1587: f32):
        %1588 = arith.constant 1.000000e+00 : f32
        %1589 = arith.negf %1586 : f32
        %1590 = math.exp %1589 : f32
        %1591 = arith.addf %1588, %1590 : f32
        %1592 = arith.divf %1588, %1591 : f32
        %1593 = arith.mulf %1586, %1592 : f32
        linalg.yield %1593 : f32
      } -> tensor<1x28x2816xf32>
      %1594 = tensor.empty() : tensor<1024x2816xf32>
      %1595 = linalg.transpose ins(%42:tensor<2816x1024xf32>) outs(%1594:tensor<1024x2816xf32>) permutation = [1, 0]
      %1596 = tensor.empty() : tensor<1x28x2816xf32>
      %1597 = arith.constant 0.000000e+00 : f32
      %1598 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1597 : f32) outs(%1596 : tensor<1x28x2816xf32>) -> tensor<1x28x2816xf32>
      %1599 = linalg.matmul {prov.region_id = "matmul_29", prov.dispatch_id = "matmul_29", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1573, %1595 : tensor<1x28x1024xf32>, tensor<1024x2816xf32>) outs(%1598 : tensor<1x28x2816xf32>) -> tensor<1x28x2816xf32>
      %1600 = tensor.empty() : tensor<1x28x2816xf32>
      %1601 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1585, %1599 : tensor<1x28x2816xf32>, tensor<1x28x2816xf32>) outs(%1600 : tensor<1x28x2816xf32>) attrs =  {prov.region_id = "mul_42", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb160(%1602: f32, %1603: f32, %1604: f32):
        %1605 = arith.mulf %1602, %1603 : f32
        linalg.yield %1605 : f32
      } -> tensor<1x28x2816xf32>
      %1606 = tensor.empty() : tensor<2816x1024xf32>
      %1607 = linalg.transpose ins(%41:tensor<1024x2816xf32>) outs(%1606:tensor<2816x1024xf32>) permutation = [1, 0]
      %1608 = tensor.empty() : tensor<1x28x1024xf32>
      %1609 = arith.constant 0.000000e+00 : f32
      %1610 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1609 : f32) outs(%1608 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %1611 = linalg.matmul {prov.region_id = "matmul_30", prov.dispatch_id = "matmul_30", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1601, %1607 : tensor<1x28x2816xf32>, tensor<2816x1024xf32>) outs(%1610 : tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %1612 = tensor.empty() : tensor<1x28x1024xf32>
      %1613 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1505, %1611 : tensor<1x1x1024xf32>, tensor<1x28x1024xf32>) outs(%1612 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_43", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb161(%1614: f32, %1615: f32, %1616: f32):
        %1617 = arith.mulf %1614, %1615 : f32
        linalg.yield %1617 : f32
      } -> tensor<1x28x1024xf32>
      %1618 = tensor.empty() : tensor<1x28x1024xf32>
      %1619 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1499, %1613 : tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) outs(%1618 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_38", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb162(%1620: f32, %1621: f32, %1622: f32):
        %1623 = arith.addf %1620, %1621 : f32
        linalg.yield %1623 : f32
      } -> tensor<1x28x1024xf32>
      %1624 = tensor.empty() : tensor<1x2048xf32>
      %1625 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%242 : tensor<1x2048xf32>) outs(%1624 : tensor<1x2048xf32>) attrs =  {prov.region_id = "silu_7", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb163(%1626: f32, %1627: f32):
        %1628 = arith.constant 1.000000e+00 : f32
        %1629 = arith.negf %1626 : f32
        %1630 = math.exp %1629 : f32
        %1631 = arith.addf %1628, %1630 : f32
        %1632 = arith.divf %1628, %1631 : f32
        %1633 = arith.mulf %1626, %1632 : f32
        linalg.yield %1633 : f32
      } -> tensor<1x2048xf32>
      %1634 = tensor.empty() : tensor<2048x2048xf32>
      %1635 = linalg.transpose ins(%50:tensor<2048x2048xf32>) outs(%1634:tensor<2048x2048xf32>) permutation = [1, 0]
      %1636 = tensor.empty() : tensor<1x2048xf32>
      %1637 = arith.constant 0.000000e+00 : f32
      %1638 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1637 : f32) outs(%1636 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %1639 = linalg.matmul {prov.region_id = "matmul_31", prov.dispatch_id = "matmul_31", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1625, %1635 : tensor<1x2048xf32>, tensor<2048x2048xf32>) outs(%1638 : tensor<1x2048xf32>) -> tensor<1x2048xf32>
      %1640 = tensor.empty() : tensor<1x2048xf32>
      %1641 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1639, %51 : tensor<1x2048xf32>, tensor<2048xf32>) outs(%1640 : tensor<1x2048xf32>) attrs =  {prov.region_id = "add_39", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb164(%1642: f32, %1643: f32, %1644: f32):
        %1645 = arith.addf %1642, %1643 : f32
        linalg.yield %1645 : f32
      } -> tensor<1x2048xf32>
      %1646 = "tensor.extract_slice"(%1641) <{static_offsets = array<i64: 0, 0>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_4", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x2048xf32>) -> tensor<1x1024xf32>
      %1647 = "tensor.extract_slice"(%1641) <{static_offsets = array<i64: 0, 1024>, static_sizes = array<i64: 1, 1024>, static_strides = array<i64: 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "split_4", prov.family = "layout", prov._pattern_hint = "chunk", prov.op = "slice", prov.aten = "aten.chunk.default", prov.orig_dtype = "float32"} : (tensor<1x2048xf32>) -> tensor<1x1024xf32>
      %1648 = tensor.empty() : tensor<1x28x1024xf32>
      %1649 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1619 : tensor<1x28x1024xf32>) outs(%1648 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "pow_12", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
      ^bb165(%1650: f32, %1651: f32):
        %1652 = arith.constant 2.000000e+00 : f32
        %1653 = math.powf %1650, %1652 : f32
        linalg.yield %1653 : f32
      } -> tensor<1x28x1024xf32>
      %1654 = arith.constant {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
      %1655 = tensor.splat %1654 {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32>
      %1656 = linalg.reduce ins(%1649:tensor<1x28x1024xf32>) outs(%1655:tensor<1x28xf32>) dimensions = [2]
      (%1657: f32, %1658: f32) {
        %1659 = arith.addf %1657, %1658 : f32
        linalg.yield %1659 : f32
      }
      %1660 = arith.constant {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 1.024000e+03 : f32
      %1661 = tensor.splat %1660 {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32>
      %1662 = tensor.empty() : tensor<1x28xf32>
      %1663 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1656, %1661 : tensor<1x28xf32>, tensor<1x28xf32>) outs(%1662 : tensor<1x28xf32>) attrs =  {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
      ^bb166(%1664: f32, %1665: f32, %1666: f32):
        %1667 = arith.divf %1664, %1665 : f32
        linalg.yield %1667 : f32
      } -> tensor<1x28xf32>
      %1668 = tensor.collapse_shape %1663 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<1x28xf32> into tensor<28xf32>
      %1669 = tensor.expand_shape %1668 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 28, 1] {prov.region_id = "reduce_12", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<28xf32> into tensor<1x28x1xf32>
      %1670 = arith.constant {prov.region_id = "add_40", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-05 : f32
      %1671 = tensor.splat %1670 {prov.region_id = "add_40", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x28x1xf32>
      %1672 = tensor.empty() : tensor<1x28x1xf32>
      %1673 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1669, %1671 : tensor<1x28x1xf32>, tensor<1x28x1xf32>) outs(%1672 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "add_40", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb167(%1674: f32, %1675: f32, %1676: f32):
        %1677 = arith.addf %1674, %1675 : f32
        linalg.yield %1677 : f32
      } -> tensor<1x28x1xf32>
      %1678 = tensor.empty() : tensor<1x28x1xf32>
      %1679 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1673 : tensor<1x28x1xf32>) outs(%1678 : tensor<1x28x1xf32>) attrs =  {prov.region_id = "rsqrt_12", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
      ^bb168(%1680: f32, %1681: f32):
        %1682 = math.rsqrt %1680 : f32
        linalg.yield %1682 : f32
      } -> tensor<1x28x1xf32>
      %1683 = tensor.empty() : tensor<1x28x1024xf32>
      %1684 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1619, %1679 : tensor<1x28x1024xf32>, tensor<1x28x1xf32>) outs(%1683 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_44", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb169(%1685: f32, %1686: f32, %1687: f32):
        %1688 = arith.mulf %1685, %1686 : f32
        linalg.yield %1688 : f32
      } -> tensor<1x28x1024xf32>
      %1689 = func.call @aten_type_as_default_wl0(%1684, %1619) {prov.region_id = "aten_type_as_default_6", prov.dispatch_id = "aten_type_as_default_6"} : (tensor<1x28x1024xf32>, tensor<1x28x1024xf32>) -> tensor<1x28x1024xf32>
      %1690 = tensor.empty() : tensor<1x28x1024xf32>
      %1691 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1689, %45 : tensor<1x28x1024xf32>, tensor<1024xf32>) outs(%1690 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_45", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb170(%1692: f32, %1693: f32, %1694: f32):
        %1695 = arith.mulf %1692, %1693 : f32
        linalg.yield %1695 : f32
      } -> tensor<1x28x1024xf32>
      %1696 = tensor.collapse_shape %1647 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_29", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %1697 = tensor.expand_shape %1696 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_29", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %1698 = arith.constant {prov.region_id = "add_41", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e+00 : f32
      %1699 = tensor.splat %1698 {prov.region_id = "add_41", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<1x1x1024xf32>
      %1700 = tensor.empty() : tensor<1x1x1024xf32>
      %1701 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1697, %1699 : tensor<1x1x1024xf32>, tensor<1x1x1024xf32>) outs(%1700 : tensor<1x1x1024xf32>) attrs =  {prov.region_id = "add_41", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb171(%1702: f32, %1703: f32, %1704: f32):
        %1705 = arith.addf %1702, %1703 : f32
        linalg.yield %1705 : f32
      } -> tensor<1x1x1024xf32>
      %1706 = tensor.empty() : tensor<1x28x1024xf32>
      %1707 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1691, %1701 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%1706 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "mul_46", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb172(%1708: f32, %1709: f32, %1710: f32):
        %1711 = arith.mulf %1708, %1709 : f32
        linalg.yield %1711 : f32
      } -> tensor<1x28x1024xf32>
      %1712 = tensor.collapse_shape %1646 [[0 : i64, 1 : i64]] {prov.region_id = "unsqueeze_30", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1x1024xf32> into tensor<1024xf32>
      %1713 = tensor.expand_shape %1712 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 1, 1024] {prov.region_id = "unsqueeze_30", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<1x1x1024xf32>
      %1714 = tensor.empty() : tensor<1x28x1024xf32>
      %1715 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, 0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1707, %1713 : tensor<1x28x1024xf32>, tensor<1x1x1024xf32>) outs(%1714 : tensor<1x28x1024xf32>) attrs =  {prov.region_id = "add_42", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb173(%1716: f32, %1717: f32, %1718: f32):
        %1719 = arith.addf %1716, %1717 : f32
        linalg.yield %1719 : f32
      } -> tensor<1x28x1024xf32>
      %1720 = tensor.empty() : tensor<1024x4096xf32>
      %1721 = linalg.transpose ins(%46:tensor<4096x1024xf32>) outs(%1720:tensor<1024x4096xf32>) permutation = [1, 0]
      %1722 = tensor.empty() : tensor<1x28x4096xf32>
      %1723 = arith.constant 0.000000e+00 : f32
      %1724 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1723 : f32) outs(%1722 : tensor<1x28x4096xf32>) -> tensor<1x28x4096xf32>
      %1725 = linalg.matmul {prov.region_id = "matmul_32", prov.dispatch_id = "matmul_32", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1715, %1721 : tensor<1x28x1024xf32>, tensor<1024x4096xf32>) outs(%1724 : tensor<1x28x4096xf32>) -> tensor<1x28x4096xf32>
      %1726 = tensor.empty() : tensor<1x28x4096xf32>
      %1727 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1725, %47 : tensor<1x28x4096xf32>, tensor<4096xf32>) outs(%1726 : tensor<1x28x4096xf32>) attrs =  {prov.region_id = "add_43", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb174(%1728: f32, %1729: f32, %1730: f32):
        %1731 = arith.addf %1728, %1729 : f32
        linalg.yield %1731 : f32
      } -> tensor<1x28x4096xf32>
      %1732 = tensor.empty() : tensor<1x28x4096xf32>
      %1733 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1727 : tensor<1x28x4096xf32>) outs(%1732 : tensor<1x28x4096xf32>) attrs =  {prov.region_id = "silu_8", prov._pattern_hint = "silu", prov.op = "silu", prov.family = "elementwise", prov.aten = "aten.silu.default", prov.orig_dtype = "float32"} {
      ^bb175(%1734: f32, %1735: f32):
        %1736 = arith.constant 1.000000e+00 : f32
        %1737 = arith.negf %1734 : f32
        %1738 = math.exp %1737 : f32
        %1739 = arith.addf %1736, %1738 : f32
        %1740 = arith.divf %1736, %1739 : f32
        %1741 = arith.mulf %1734, %1740 : f32
        linalg.yield %1741 : f32
      } -> tensor<1x28x4096xf32>
      %1742 = tensor.empty() : tensor<4096x20xf32>
      %1743 = linalg.transpose ins(%48:tensor<20x4096xf32>) outs(%1742:tensor<4096x20xf32>) permutation = [1, 0]
      %1744 = tensor.empty() : tensor<1x28x20xf32>
      %1745 = arith.constant 0.000000e+00 : f32
      %1746 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%1745 : f32) outs(%1744 : tensor<1x28x20xf32>) -> tensor<1x28x20xf32>
      %1747 = linalg.matmul {prov.region_id = "matmul_33", prov.dispatch_id = "matmul_33", prov.transposed_b = "true", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} ins(%1733, %1743 : tensor<1x28x4096xf32>, tensor<4096x20xf32>) outs(%1746 : tensor<1x28x20xf32>) -> tensor<1x28x20xf32>
      %1748 = tensor.empty() : tensor<1x28x20xf32>
      %1749 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1747, %49 : tensor<1x28x20xf32>, tensor<20xf32>) outs(%1748 : tensor<1x28x20xf32>) attrs =  {prov.region_id = "add_44", prov.aten = "aten.linear.default", prov.orig_dtype = "float32"} {
      ^bb176(%1750: f32, %1751: f32, %1752: f32):
        %1753 = arith.addf %1750, %1751 : f32
        linalg.yield %1753 : f32
      } -> tensor<1x28x20xf32>
      %1754 = "tensor.extract_slice"(%1749) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 24, 20>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<1x28x20xf32>) -> tensor<1x24x20xf32>
      %1755 = arith.constant {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 2.000000e-01 : f32
      %1756 = tensor.splat %1755 {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<1x24x20xf32>
      %1757 = tensor.empty() : tensor<1x24x20xf32>
      %1758 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1754, %1756 : tensor<1x24x20xf32>, tensor<1x24x20xf32>) outs(%1757 : tensor<1x24x20xf32>) attrs =  {prov.region_id = "mul_47", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
      ^bb177(%1759: f32, %1760: f32, %1761: f32):
        %1762 = arith.mulf %1759, %1760 : f32
        linalg.yield %1762 : f32
      } -> tensor<1x24x20xf32>
      %1763 = tensor.empty() : tensor<1x24x20xf32>
      %1764 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1758, %103 : tensor<1x24x20xf32>, tensor<1x24x20xf32>) outs(%1763 : tensor<1x24x20xf32>) attrs =  {prov.region_id = "add_45", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
      ^bb178(%1765: f32, %1766: f32, %1767: f32):
        %1768 = arith.addf %1765, %1766 : f32
        linalg.yield %1768 : f32
      } -> tensor<1x24x20xf32>
      %1769 = arith.constant {prov.region_id = "add_46", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} 1 : i64
      %1770 = tensor.splat %1769 {prov.region_id = "add_46", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} : tensor<i64>
      %1771 = tensor.empty() : tensor<i64>
      %1772 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%102, %1770 : tensor<i64>, tensor<i64>) outs(%1771 : tensor<i64>) attrs =  {prov.region_id = "add_46", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
      ^bb179(%1773: i64, %1774: i64, %1775: i64):
        %1776 = arith.addi %1773, %1774 : i64
        linalg.yield %1776 : i64
      } -> tensor<i64>
      scf.yield %1772, %1764 : tensor<i64>, tensor<1x24x20xf32>
    }
    func.return %100 : tensor<1x24x20xf32>
  }
}
