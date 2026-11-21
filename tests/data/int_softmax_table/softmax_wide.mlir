builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<3x11200xf32>) -> tensor<3x11200xf32> {
    %1 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %2 = tensor.splat %1 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} : tensor<3xf32>
    %3 = linalg.reduce ins(%0:tensor<3x11200xf32>) outs(%2:tensor<3xf32>) dimensions = [1]
    (%4: f32, %5: f32) {
      %6 = arith.maximumf %4, %5 : f32
      linalg.yield %6 : f32
    }
    %7 = tensor.expand_shape %3 [[0 : i64, 1 : i64]] output_shape [3, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce", prov.op = "reduce", prov.aten = "aten.amax.default", prov.orig_dtype = "float32"} : tensor<3xf32> into tensor<3x1xf32>
    %8 = tensor.empty() : tensor<3x11200xf32>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %7 : tensor<3x11200xf32>, tensor<3x1xf32>) outs(%8 : tensor<3x11200xf32>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} {
    ^bb0(%10: f32, %11: f32, %12: f32):
      %13 = arith.subf %10, %11 : f32
      linalg.yield %13 : f32
    } -> tensor<3x11200xf32>
    %14 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 0.00270760618 : f32
    %15 = tensor.splat %14 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<3x11200xf32>
    %16 = tensor.empty() : tensor<3x11200xf32>
    %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %15 : tensor<3x11200xf32>, tensor<3x11200xf32>) outs(%16 : tensor<3x11200xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb1(%18: f32, %19: f32, %20: f32):
      %21 = arith.divf %18, %19 : f32
      linalg.yield %21 : f32
    } -> tensor<3x11200xf32>
    %22 = tensor.empty() : tensor<3x11200xf32>
    %23 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17 : tensor<3x11200xf32>) outs(%22 : tensor<3x11200xf32>) attrs =  {prov.region_id = "round_0", prov._pattern_hint = "round", prov.op = "round", prov.family = "elementwise", prov.aten = "aten.round.default", prov.orig_dtype = "float32"} {
    ^bb2(%24: f32, %25: f32):
      %26 = math.roundeven %24 : f32
      linalg.yield %26 : f32
    } -> tensor<3x11200xf32>
    %27 = tensor.empty() : tensor<3x11200xf32>
    %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23 : tensor<3x11200xf32>) outs(%27 : tensor<3x11200xf32>) attrs =  {prov.region_id = "minmax_0", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.clamp.default", prov.orig_dtype = "float32"} {
    ^bb3(%29: f32, %30: f32):
      %31 = arith.constant -1.108000e+04 : f32
      %32 = arith.maximumf %29, %31 : f32
      %33 = arith.constant 0.000000e+00 : f32
      %34 = arith.minimumf %32, %33 : f32
      linalg.yield %34 : f32
    } -> tensor<3x11200xf32>
    %35 = tensor.empty() : tensor<3x11200xi32>
    %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%28 : tensor<3x11200xf32>) outs(%35 : tensor<3x11200xi32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int32"} {
    ^bb4(%37: f32, %38: i32):
      %39 = arith.fptosi %37 : f32 to i32
      linalg.yield %39 : i32
    } -> tensor<3x11200xi32>
    %40 = tensor.empty() : tensor<3x11200xi64>
    %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%36 : tensor<3x11200xi32>) outs(%40 : tensor<3x11200xi64>) attrs =  {prov.region_id = "dtype_cast_1", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int64"} {
    ^bb5(%42: i32, %43: i64):
      %44 = arith.extsi %42 : i32 to i64
      linalg.yield %44 : i64
    } -> tensor<3x11200xi64>
    %45 = arith.constant {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64"} 0 : i64
    %46 = tensor.splat %45 {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64"} : tensor<3x11200xi64>
    %47 = tensor.empty() : tensor<3x11200xi64>
    %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%46, %41 : tensor<3x11200xi64>, tensor<3x11200xi64>) outs(%47 : tensor<3x11200xi64>) attrs =  {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64"} {
    ^bb6(%49: i64, %50: i64, %51: i64):
      %52 = arith.subi %49, %50 : i64
      linalg.yield %52 : i64
    } -> tensor<3x11200xi64>
    %53 = tensor.empty() : tensor<3x11200xi64>
    %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%48 : tensor<3x11200xi64>) outs(%53 : tensor<3x11200xi64>) attrs =  {prov.region_id = "bitwise_0", prov.family = "bitwise", prov._pattern_hint = "bitwise_right_shift", prov.op = "bitwise_right_shift", prov.aten = "aten.bitwise_right_shift.Tensor_Scalar", prov.orig_dtype = "int64"} {
    ^bb7(%55: i64, %56: i64):
      %57 = arith.constant 8 : i64
      %58 = arith.shrsi %55, %57 : i64
      linalg.yield %58 : i64
    } -> tensor<3x11200xi64>
    %59 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} 256 : i64
    %60 = tensor.splat %59 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} : tensor<3x11200xi64>
    %61 = tensor.empty() : tensor<3x11200xi64>
    %62 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%54, %60 : tensor<3x11200xi64>, tensor<3x11200xi64>) outs(%61 : tensor<3x11200xi64>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} {
    ^bb8(%63: i64, %64: i64, %65: i64):
      %66 = arith.muli %63, %64 : i64
      linalg.yield %66 : i64
    } -> tensor<3x11200xi64>
    %67 = tensor.empty() : tensor<3x11200xi64>
    %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%41, %62 : tensor<3x11200xi64>, tensor<3x11200xi64>) outs(%67 : tensor<3x11200xi64>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb9(%69: i64, %70: i64, %71: i64):
      %72 = arith.addi %69, %70 : i64
      linalg.yield %72 : i64
    } -> tensor<3x11200xi64>
    %73 = arith.constant {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} 500 : i64
    %74 = tensor.splat %73 {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} : tensor<3x11200xi64>
    %75 = tensor.empty() : tensor<3x11200xi64>
    %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%68, %74 : tensor<3x11200xi64>, tensor<3x11200xi64>) outs(%75 : tensor<3x11200xi64>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb10(%77: i64, %78: i64, %79: i64):
      %80 = arith.addi %77, %78 : i64
      linalg.yield %80 : i64
    } -> tensor<3x11200xi64>
    %81 = arith.constant {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} 2819 : i64
    %82 = tensor.splat %81 {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} : tensor<3x11200xi64>
    %83 = tensor.empty() : tensor<3x11200xi64>
    %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%76, %82 : tensor<3x11200xi64>, tensor<3x11200xi64>) outs(%83 : tensor<3x11200xi64>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} {
    ^bb11(%85: i64, %86: i64, %87: i64):
      %88 = arith.muli %85, %86 : i64
      linalg.yield %88 : i64
    } -> tensor<3x11200xi64>
    %89 = tensor.empty() : tensor<3x11200xi64>
    %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%84, %76 : tensor<3x11200xi64>, tensor<3x11200xi64>) outs(%89 : tensor<3x11200xi64>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} {
    ^bb12(%91: i64, %92: i64, %93: i64):
      %94 = arith.muli %91, %92 : i64
      linalg.yield %94 : i64
    } -> tensor<3x11200xi64>
    %95 = arith.constant {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} 369388222 : i64
    %96 = tensor.splat %95 {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} : tensor<3x11200xi64>
    %97 = tensor.empty() : tensor<3x11200xi64>
    %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%90, %96 : tensor<3x11200xi64>, tensor<3x11200xi64>) outs(%97 : tensor<3x11200xi64>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb13(%99: i64, %100: i64, %101: i64):
      %102 = arith.addi %99, %100 : i64
      linalg.yield %102 : i64
    } -> tensor<3x11200xi64>
    %103 = tensor.empty() : tensor<3x11200xi64>
    %104 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%98, %54 : tensor<3x11200xi64>, tensor<3x11200xi64>) outs(%103 : tensor<3x11200xi64>) attrs =  {prov.region_id = "bitwise_1", prov.family = "bitwise", prov._pattern_hint = "bitwise_right_shift", prov.op = "bitwise_right_shift", prov.aten = "aten.bitwise_right_shift.Tensor", prov.orig_dtype = "int64"} {
    ^bb14(%105: i64, %106: i64, %107: i64):
      %108 = arith.constant 63 : i64
      %109 = arith.minui %106, %108 : i64
      %110 = arith.shrsi %105, %109 : i64
      linalg.yield %110 : i64
    } -> tensor<3x11200xi64>
    %111 = arith.constant {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} 127 : i64
    %112 = tensor.splat %111 {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} : tensor<3x11200xi64>
    %113 = tensor.empty() : tensor<3x11200xi64>
    %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%104, %112 : tensor<3x11200xi64>, tensor<3x11200xi64>) outs(%113 : tensor<3x11200xi64>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "int64"} {
    ^bb15(%115: i64, %116: i64, %117: i64):
      %118 = arith.muli %115, %116 : i64
      linalg.yield %118 : i64
    } -> tensor<3x11200xi64>
    %119 = arith.constant {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} 537069111 : i64
    %120 = tensor.splat %119 {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} : tensor<3x11200xi64>
    %121 = tensor.empty() : tensor<3x11200xi64>
    %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%114, %120 : tensor<3x11200xi64>, tensor<3x11200xi64>) outs(%121 : tensor<3x11200xi64>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "int64"} {
    ^bb16(%123: i64, %124: i64, %125: i64):
      %126 = arith.addi %123, %124 : i64
      linalg.yield %126 : i64
    } -> tensor<3x11200xi64>
    %127 = tensor.empty() : tensor<3x11200xi64>
    %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%122 : tensor<3x11200xi64>) outs(%127 : tensor<3x11200xi64>) attrs =  {prov.region_id = "elementwise_0", prov.family = "elementwise", prov._pattern_hint = "floor_divide", prov.op = "floor_divide", prov.aten = "aten.div.Tensor_mode", prov.orig_dtype = "int64"} {
    ^bb17(%129: i64, %130: i64):
      %131 = arith.constant 1074138222 : i64
      %132 = arith.floordivsi %129, %131 : i64
      linalg.yield %132 : i64
    } -> tensor<3x11200xi64>
    %133 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "int64"} 0 : i64
    %134 = tensor.splat %133 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "int64"} : tensor<3xi64>
    %135 = linalg.reduce ins(%128:tensor<3x11200xi64>) outs(%134:tensor<3xi64>) dimensions = [1]
    (%136: i64, %137: i64) {
      %138 = arith.addi %136, %137 : i64
      linalg.yield %138 : i64
    }
    %139 = tensor.expand_shape %135 [[0 : i64, 1 : i64]] output_shape [3, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "int64"} : tensor<3xi64> into tensor<3x1xi64>
    %140 = tensor.empty() : tensor<3x11200xf32>
    %141 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%128 : tensor<3x11200xi64>) outs(%140 : tensor<3x11200xf32>) attrs =  {prov.region_id = "dtype_cast_2", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb18(%142: i64, %143: f32):
      %144 = arith.sitofp %142 : i64 to f32
      linalg.yield %144 : f32
    } -> tensor<3x11200xf32>
    %145 = tensor.empty() : tensor<3x1xf32>
    %146 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%139 : tensor<3x1xi64>) outs(%145 : tensor<3x1xf32>) attrs =  {prov.region_id = "dtype_cast_3", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "float32"} {
    ^bb19(%147: i64, %148: f32):
      %149 = arith.sitofp %147 : i64 to f32
      linalg.yield %149 : f32
    } -> tensor<3x1xf32>
    %150 = tensor.empty() : tensor<3x11200xf32>
    %151 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%141, %146 : tensor<3x11200xf32>, tensor<3x1xf32>) outs(%150 : tensor<3x11200xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb20(%152: f32, %153: f32, %154: f32):
      %155 = arith.divf %152, %153 : f32
      linalg.yield %155 : f32
    } -> tensor<3x11200xf32>
    func.return %151 : tensor<3x11200xf32>
  }
}
