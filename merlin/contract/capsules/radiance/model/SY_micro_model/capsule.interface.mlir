builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<3x32x32xf32>, %1: tensor<3x32x32xf32>, %2: tensor<3x32x32xf32>, %3: tensor<3x32x32xf32>, %4: tensor<3x32x32xf32>, %5: tensor<3x32x32xf32>, %6: tensor<32x32xf32>, %7: tensor<32x32xf32>, %8: tensor<32x32xf32>, %9: tensor<32x32xf32>, %10: tensor<32x32xf32>, %11: tensor<32x32xf32>, %12: tensor<32x32xf32>, %13: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %14 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %15 = tensor.collapse_shape %14 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %16 = tensor.expand_shape %15 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %17 = tensor.empty() : tensor<32x32xf32>
    %18 = arith.constant 0.000000e+00 : f32
    %19 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%18 : f32) outs(%17 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %20 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%13, %16 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %21 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %22 = tensor.collapse_shape %21 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %23 = tensor.expand_shape %22 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_1", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %24 = tensor.empty() : tensor<32x32xf32>
    %25 = arith.constant 0.000000e+00 : f32
    %26 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%25 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %27 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%13, %23 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %28 = "tensor.extract_slice"(%0) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %29 = tensor.collapse_shape %28 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %30 = tensor.expand_shape %29 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_2", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %31 = tensor.empty() : tensor<32x32xf32>
    %32 = arith.constant 0.000000e+00 : f32
    %33 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%32 : f32) outs(%31 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %34 = linalg.matmul {prov.region_id = "matmul_2", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%13, %30 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%33 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %35 = tensor.empty() : tensor<32x32xf32>
    %36 = linalg.transpose ins(%27:tensor<32x32xf32>) outs(%35:tensor<32x32xf32>) permutation = [1, 0]
    %37 = tensor.empty() : tensor<32x32xf32>
    %38 = arith.constant 0.000000e+00 : f32
    %39 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%38 : f32) outs(%37 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %40 = linalg.matmul {prov.region_id = "matmul_3", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%20, %36 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%39 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %41 = arith.constant {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.176776692 : f32
    %42 = tensor.splat %41 {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<32x32xf32>
    %43 = tensor.empty() : tensor<32x32xf32>
    %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%40, %42 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%43 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb0(%45: f32, %46: f32, %47: f32):
      %48 = arith.mulf %45, %46 : f32
      linalg.yield %48 : f32
    } -> tensor<32x32xf32>
    %49 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %50 = tensor.splat %49 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %51 = linalg.reduce ins(%44:tensor<32x32xf32>) outs(%50:tensor<32xf32>) dimensions = [1]
    (%52: f32, %53: f32) {
      %54 = arith.maximumf %52, %53 : f32
      linalg.yield %54 : f32
    }
    %55 = tensor.expand_shape %51 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %56 = tensor.empty() : tensor<32x32xf32>
    %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%44, %55 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%56 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb1(%58: f32, %59: f32, %60: f32):
      %61 = arith.subf %58, %59 : f32
      linalg.yield %61 : f32
    } -> tensor<32x32xf32>
    %62 = tensor.empty() : tensor<32x32xf32>
    %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%57 : tensor<32x32xf32>) outs(%62 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb2(%64: f32, %65: f32):
      %66 = math.exp %64 : f32
      linalg.yield %66 : f32
    } -> tensor<32x32xf32>
    %67 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %68 = tensor.splat %67 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %69 = linalg.reduce ins(%63:tensor<32x32xf32>) outs(%68:tensor<32xf32>) dimensions = [1]
    (%70: f32, %71: f32) {
      %72 = arith.addf %70, %71 : f32
      linalg.yield %72 : f32
    }
    %73 = tensor.expand_shape %69 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %74 = tensor.empty() : tensor<32x32xf32>
    %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %73 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%74 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb3(%76: f32, %77: f32, %78: f32):
      %79 = arith.divf %76, %77 : f32
      linalg.yield %79 : f32
    } -> tensor<32x32xf32>
    %80 = tensor.empty() : tensor<32x32xf32>
    %81 = arith.constant 0.000000e+00 : f32
    %82 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%81 : f32) outs(%80 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %83 = linalg.matmul {prov.region_id = "matmul_4", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%75, %34 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%82 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %84 = tensor.empty() : tensor<32x32xf32>
    %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%13, %83 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%84 : tensor<32x32xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb4(%86: f32, %87: f32, %88: f32):
      %89 = arith.addf %86, %87 : f32
      linalg.yield %89 : f32
    } -> tensor<32x32xf32>
    %90 = "tensor.extract_slice"(%1) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %91 = tensor.collapse_shape %90 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %92 = tensor.expand_shape %91 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_3", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %93 = tensor.empty() : tensor<32x32xf32>
    %94 = arith.constant 0.000000e+00 : f32
    %95 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%94 : f32) outs(%93 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %96 = linalg.matmul {prov.region_id = "matmul_5", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%85, %92 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%95 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %97 = "tensor.extract_slice"(%1) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %98 = tensor.collapse_shape %97 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %99 = tensor.expand_shape %98 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_4", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %100 = tensor.empty() : tensor<32x32xf32>
    %101 = arith.constant 0.000000e+00 : f32
    %102 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%101 : f32) outs(%100 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %103 = linalg.matmul {prov.region_id = "matmul_6", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%85, %99 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%102 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %104 = "tensor.extract_slice"(%1) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %105 = tensor.collapse_shape %104 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %106 = tensor.expand_shape %105 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_5", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %107 = tensor.empty() : tensor<32x32xf32>
    %108 = arith.constant 0.000000e+00 : f32
    %109 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%108 : f32) outs(%107 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %110 = linalg.matmul {prov.region_id = "matmul_7", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%85, %106 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%109 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %111 = tensor.empty() : tensor<32x32xf32>
    %112 = linalg.transpose ins(%103:tensor<32x32xf32>) outs(%111:tensor<32x32xf32>) permutation = [1, 0]
    %113 = tensor.empty() : tensor<32x32xf32>
    %114 = arith.constant 0.000000e+00 : f32
    %115 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%114 : f32) outs(%113 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %116 = linalg.matmul {prov.region_id = "matmul_8", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%96, %112 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%115 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %117 = arith.constant {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.176776692 : f32
    %118 = tensor.splat %117 {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<32x32xf32>
    %119 = tensor.empty() : tensor<32x32xf32>
    %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %118 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%119 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb5(%121: f32, %122: f32, %123: f32):
      %124 = arith.mulf %121, %122 : f32
      linalg.yield %124 : f32
    } -> tensor<32x32xf32>
    %125 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %126 = tensor.splat %125 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %127 = linalg.reduce ins(%120:tensor<32x32xf32>) outs(%126:tensor<32xf32>) dimensions = [1]
    (%128: f32, %129: f32) {
      %130 = arith.maximumf %128, %129 : f32
      linalg.yield %130 : f32
    }
    %131 = tensor.expand_shape %127 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %132 = tensor.empty() : tensor<32x32xf32>
    %133 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%120, %131 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%132 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb6(%134: f32, %135: f32, %136: f32):
      %137 = arith.subf %134, %135 : f32
      linalg.yield %137 : f32
    } -> tensor<32x32xf32>
    %138 = tensor.empty() : tensor<32x32xf32>
    %139 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%133 : tensor<32x32xf32>) outs(%138 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb7(%140: f32, %141: f32):
      %142 = math.exp %140 : f32
      linalg.yield %142 : f32
    } -> tensor<32x32xf32>
    %143 = arith.constant {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %144 = tensor.splat %143 {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %145 = linalg.reduce ins(%139:tensor<32x32xf32>) outs(%144:tensor<32xf32>) dimensions = [1]
    (%146: f32, %147: f32) {
      %148 = arith.addf %146, %147 : f32
      linalg.yield %148 : f32
    }
    %149 = tensor.expand_shape %145 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %150 = tensor.empty() : tensor<32x32xf32>
    %151 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%139, %149 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%150 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_1", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb8(%152: f32, %153: f32, %154: f32):
      %155 = arith.divf %152, %153 : f32
      linalg.yield %155 : f32
    } -> tensor<32x32xf32>
    %156 = tensor.empty() : tensor<32x32xf32>
    %157 = arith.constant 0.000000e+00 : f32
    %158 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%157 : f32) outs(%156 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %159 = linalg.matmul {prov.region_id = "matmul_9", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%151, %110 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%158 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %160 = tensor.empty() : tensor<32x32xf32>
    %161 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%85, %159 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%160 : tensor<32x32xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb9(%162: f32, %163: f32, %164: f32):
      %165 = arith.addf %162, %163 : f32
      linalg.yield %165 : f32
    } -> tensor<32x32xf32>
    %166 = "tensor.extract_slice"(%2) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %167 = tensor.collapse_shape %166 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %168 = tensor.expand_shape %167 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_6", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %169 = tensor.empty() : tensor<32x32xf32>
    %170 = arith.constant 0.000000e+00 : f32
    %171 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%170 : f32) outs(%169 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %172 = linalg.matmul {prov.region_id = "matmul_10", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%161, %168 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%171 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %173 = "tensor.extract_slice"(%2) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %174 = tensor.collapse_shape %173 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %175 = tensor.expand_shape %174 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_7", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %176 = tensor.empty() : tensor<32x32xf32>
    %177 = arith.constant 0.000000e+00 : f32
    %178 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%177 : f32) outs(%176 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %179 = linalg.matmul {prov.region_id = "matmul_11", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%161, %175 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%178 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %180 = "tensor.extract_slice"(%2) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_8", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %181 = tensor.collapse_shape %180 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_8", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %182 = tensor.expand_shape %181 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_8", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %183 = tensor.empty() : tensor<32x32xf32>
    %184 = arith.constant 0.000000e+00 : f32
    %185 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%184 : f32) outs(%183 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %186 = linalg.matmul {prov.region_id = "matmul_12", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%161, %182 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%185 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %187 = tensor.empty() : tensor<32x32xf32>
    %188 = linalg.transpose ins(%179:tensor<32x32xf32>) outs(%187:tensor<32x32xf32>) permutation = [1, 0]
    %189 = tensor.empty() : tensor<32x32xf32>
    %190 = arith.constant 0.000000e+00 : f32
    %191 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%190 : f32) outs(%189 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %192 = linalg.matmul {prov.region_id = "matmul_13", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%172, %188 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%191 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %193 = arith.constant {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.176776692 : f32
    %194 = tensor.splat %193 {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<32x32xf32>
    %195 = tensor.empty() : tensor<32x32xf32>
    %196 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%192, %194 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%195 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb10(%197: f32, %198: f32, %199: f32):
      %200 = arith.mulf %197, %198 : f32
      linalg.yield %200 : f32
    } -> tensor<32x32xf32>
    %201 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %202 = tensor.splat %201 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %203 = linalg.reduce ins(%196:tensor<32x32xf32>) outs(%202:tensor<32xf32>) dimensions = [1]
    (%204: f32, %205: f32) {
      %206 = arith.maximumf %204, %205 : f32
      linalg.yield %206 : f32
    }
    %207 = tensor.expand_shape %203 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %208 = tensor.empty() : tensor<32x32xf32>
    %209 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%196, %207 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%208 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb11(%210: f32, %211: f32, %212: f32):
      %213 = arith.subf %210, %211 : f32
      linalg.yield %213 : f32
    } -> tensor<32x32xf32>
    %214 = tensor.empty() : tensor<32x32xf32>
    %215 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%209 : tensor<32x32xf32>) outs(%214 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb12(%216: f32, %217: f32):
      %218 = math.exp %216 : f32
      linalg.yield %218 : f32
    } -> tensor<32x32xf32>
    %219 = arith.constant {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %220 = tensor.splat %219 {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %221 = linalg.reduce ins(%215:tensor<32x32xf32>) outs(%220:tensor<32xf32>) dimensions = [1]
    (%222: f32, %223: f32) {
      %224 = arith.addf %222, %223 : f32
      linalg.yield %224 : f32
    }
    %225 = tensor.expand_shape %221 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %226 = tensor.empty() : tensor<32x32xf32>
    %227 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%215, %225 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%226 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_2", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb13(%228: f32, %229: f32, %230: f32):
      %231 = arith.divf %228, %229 : f32
      linalg.yield %231 : f32
    } -> tensor<32x32xf32>
    %232 = tensor.empty() : tensor<32x32xf32>
    %233 = arith.constant 0.000000e+00 : f32
    %234 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%233 : f32) outs(%232 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %235 = linalg.matmul {prov.region_id = "matmul_14", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%227, %186 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%234 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %236 = tensor.empty() : tensor<32x32xf32>
    %237 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%161, %235 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%236 : tensor<32x32xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb14(%238: f32, %239: f32, %240: f32):
      %241 = arith.addf %238, %239 : f32
      linalg.yield %241 : f32
    } -> tensor<32x32xf32>
    %242 = "tensor.extract_slice"(%3) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_9", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %243 = tensor.collapse_shape %242 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_9", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %244 = tensor.expand_shape %243 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_9", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %245 = tensor.empty() : tensor<32x32xf32>
    %246 = arith.constant 0.000000e+00 : f32
    %247 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%246 : f32) outs(%245 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %248 = linalg.matmul {prov.region_id = "matmul_15", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%237, %244 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%247 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %249 = "tensor.extract_slice"(%3) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_10", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %250 = tensor.collapse_shape %249 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_10", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %251 = tensor.expand_shape %250 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_10", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %252 = tensor.empty() : tensor<32x32xf32>
    %253 = arith.constant 0.000000e+00 : f32
    %254 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%253 : f32) outs(%252 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %255 = linalg.matmul {prov.region_id = "matmul_16", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%237, %251 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%254 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %256 = "tensor.extract_slice"(%3) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_11", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %257 = tensor.collapse_shape %256 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_11", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %258 = tensor.expand_shape %257 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_11", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %259 = tensor.empty() : tensor<32x32xf32>
    %260 = arith.constant 0.000000e+00 : f32
    %261 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%260 : f32) outs(%259 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %262 = linalg.matmul {prov.region_id = "matmul_17", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%237, %258 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%261 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %263 = tensor.empty() : tensor<32x32xf32>
    %264 = linalg.transpose ins(%255:tensor<32x32xf32>) outs(%263:tensor<32x32xf32>) permutation = [1, 0]
    %265 = tensor.empty() : tensor<32x32xf32>
    %266 = arith.constant 0.000000e+00 : f32
    %267 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%266 : f32) outs(%265 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %268 = linalg.matmul {prov.region_id = "matmul_18", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%248, %264 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%267 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %269 = arith.constant {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.176776692 : f32
    %270 = tensor.splat %269 {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<32x32xf32>
    %271 = tensor.empty() : tensor<32x32xf32>
    %272 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%268, %270 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%271 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb15(%273: f32, %274: f32, %275: f32):
      %276 = arith.mulf %273, %274 : f32
      linalg.yield %276 : f32
    } -> tensor<32x32xf32>
    %277 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %278 = tensor.splat %277 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %279 = linalg.reduce ins(%272:tensor<32x32xf32>) outs(%278:tensor<32xf32>) dimensions = [1]
    (%280: f32, %281: f32) {
      %282 = arith.maximumf %280, %281 : f32
      linalg.yield %282 : f32
    }
    %283 = tensor.expand_shape %279 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %284 = tensor.empty() : tensor<32x32xf32>
    %285 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%272, %283 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%284 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb16(%286: f32, %287: f32, %288: f32):
      %289 = arith.subf %286, %287 : f32
      linalg.yield %289 : f32
    } -> tensor<32x32xf32>
    %290 = tensor.empty() : tensor<32x32xf32>
    %291 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%285 : tensor<32x32xf32>) outs(%290 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb17(%292: f32, %293: f32):
      %294 = math.exp %292 : f32
      linalg.yield %294 : f32
    } -> tensor<32x32xf32>
    %295 = arith.constant {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %296 = tensor.splat %295 {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %297 = linalg.reduce ins(%291:tensor<32x32xf32>) outs(%296:tensor<32xf32>) dimensions = [1]
    (%298: f32, %299: f32) {
      %300 = arith.addf %298, %299 : f32
      linalg.yield %300 : f32
    }
    %301 = tensor.expand_shape %297 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %302 = tensor.empty() : tensor<32x32xf32>
    %303 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%291, %301 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%302 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_3", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb18(%304: f32, %305: f32, %306: f32):
      %307 = arith.divf %304, %305 : f32
      linalg.yield %307 : f32
    } -> tensor<32x32xf32>
    %308 = tensor.empty() : tensor<32x32xf32>
    %309 = arith.constant 0.000000e+00 : f32
    %310 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%309 : f32) outs(%308 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %311 = linalg.matmul {prov.region_id = "matmul_19", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%303, %262 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%310 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %312 = tensor.empty() : tensor<32x32xf32>
    %313 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%237, %311 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%312 : tensor<32x32xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb19(%314: f32, %315: f32, %316: f32):
      %317 = arith.addf %314, %315 : f32
      linalg.yield %317 : f32
    } -> tensor<32x32xf32>
    %318 = "tensor.extract_slice"(%4) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_12", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %319 = tensor.collapse_shape %318 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_12", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %320 = tensor.expand_shape %319 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_12", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %321 = tensor.empty() : tensor<32x32xf32>
    %322 = arith.constant 0.000000e+00 : f32
    %323 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%322 : f32) outs(%321 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %324 = linalg.matmul {prov.region_id = "matmul_20", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%313, %320 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%323 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %325 = "tensor.extract_slice"(%4) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_13", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %326 = tensor.collapse_shape %325 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_13", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %327 = tensor.expand_shape %326 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_13", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %328 = tensor.empty() : tensor<32x32xf32>
    %329 = arith.constant 0.000000e+00 : f32
    %330 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%329 : f32) outs(%328 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %331 = linalg.matmul {prov.region_id = "matmul_21", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%313, %327 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%330 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %332 = "tensor.extract_slice"(%4) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_14", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %333 = tensor.collapse_shape %332 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_14", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %334 = tensor.expand_shape %333 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_14", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %335 = tensor.empty() : tensor<32x32xf32>
    %336 = arith.constant 0.000000e+00 : f32
    %337 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%336 : f32) outs(%335 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %338 = linalg.matmul {prov.region_id = "matmul_22", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%313, %334 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%337 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %339 = tensor.empty() : tensor<32x32xf32>
    %340 = linalg.transpose ins(%331:tensor<32x32xf32>) outs(%339:tensor<32x32xf32>) permutation = [1, 0]
    %341 = tensor.empty() : tensor<32x32xf32>
    %342 = arith.constant 0.000000e+00 : f32
    %343 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%342 : f32) outs(%341 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %344 = linalg.matmul {prov.region_id = "matmul_23", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%324, %340 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%343 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %345 = arith.constant {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.176776692 : f32
    %346 = tensor.splat %345 {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<32x32xf32>
    %347 = tensor.empty() : tensor<32x32xf32>
    %348 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%344, %346 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%347 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb20(%349: f32, %350: f32, %351: f32):
      %352 = arith.mulf %349, %350 : f32
      linalg.yield %352 : f32
    } -> tensor<32x32xf32>
    %353 = arith.constant {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %354 = tensor.splat %353 {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %355 = linalg.reduce ins(%348:tensor<32x32xf32>) outs(%354:tensor<32xf32>) dimensions = [1]
    (%356: f32, %357: f32) {
      %358 = arith.maximumf %356, %357 : f32
      linalg.yield %358 : f32
    }
    %359 = tensor.expand_shape %355 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %360 = tensor.empty() : tensor<32x32xf32>
    %361 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%348, %359 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%360 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb21(%362: f32, %363: f32, %364: f32):
      %365 = arith.subf %362, %363 : f32
      linalg.yield %365 : f32
    } -> tensor<32x32xf32>
    %366 = tensor.empty() : tensor<32x32xf32>
    %367 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%361 : tensor<32x32xf32>) outs(%366 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb22(%368: f32, %369: f32):
      %370 = math.exp %368 : f32
      linalg.yield %370 : f32
    } -> tensor<32x32xf32>
    %371 = arith.constant {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %372 = tensor.splat %371 {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %373 = linalg.reduce ins(%367:tensor<32x32xf32>) outs(%372:tensor<32xf32>) dimensions = [1]
    (%374: f32, %375: f32) {
      %376 = arith.addf %374, %375 : f32
      linalg.yield %376 : f32
    }
    %377 = tensor.expand_shape %373 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %378 = tensor.empty() : tensor<32x32xf32>
    %379 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%367, %377 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%378 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_4", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb23(%380: f32, %381: f32, %382: f32):
      %383 = arith.divf %380, %381 : f32
      linalg.yield %383 : f32
    } -> tensor<32x32xf32>
    %384 = tensor.empty() : tensor<32x32xf32>
    %385 = arith.constant 0.000000e+00 : f32
    %386 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%385 : f32) outs(%384 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %387 = linalg.matmul {prov.region_id = "matmul_24", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%379, %338 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%386 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %388 = tensor.empty() : tensor<32x32xf32>
    %389 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%313, %387 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%388 : tensor<32x32xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb24(%390: f32, %391: f32, %392: f32):
      %393 = arith.addf %390, %391 : f32
      linalg.yield %393 : f32
    } -> tensor<32x32xf32>
    %394 = "tensor.extract_slice"(%5) <{static_offsets = array<i64: 0, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_15", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %395 = tensor.collapse_shape %394 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_15", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %396 = tensor.expand_shape %395 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_15", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %397 = tensor.empty() : tensor<32x32xf32>
    %398 = arith.constant 0.000000e+00 : f32
    %399 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%398 : f32) outs(%397 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %400 = linalg.matmul {prov.region_id = "matmul_25", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%389, %396 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%399 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %401 = "tensor.extract_slice"(%5) <{static_offsets = array<i64: 1, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_16", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %402 = tensor.collapse_shape %401 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_16", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %403 = tensor.expand_shape %402 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_16", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %404 = tensor.empty() : tensor<32x32xf32>
    %405 = arith.constant 0.000000e+00 : f32
    %406 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%405 : f32) outs(%404 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %407 = linalg.matmul {prov.region_id = "matmul_26", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%389, %403 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%406 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %408 = "tensor.extract_slice"(%5) <{static_offsets = array<i64: 2, 0, 0>, static_sizes = array<i64: 1, 32, 32>, static_strides = array<i64: 1, 1, 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "select_17", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : (tensor<3x32x32xf32>) -> tensor<1x32x32xf32>
    %409 = tensor.collapse_shape %408 [[0 : i64, 1 : i64, 2 : i64]] {prov.region_id = "select_17", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1x32x32xf32> into tensor<1024xf32>
    %410 = tensor.expand_shape %409 [[0 : i64, 1 : i64]] output_shape [32, 32] {prov.region_id = "select_17", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.select.int", prov.orig_dtype = "float32"} : tensor<1024xf32> into tensor<32x32xf32>
    %411 = tensor.empty() : tensor<32x32xf32>
    %412 = arith.constant 0.000000e+00 : f32
    %413 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%412 : f32) outs(%411 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %414 = linalg.matmul {prov.region_id = "matmul_27", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%389, %410 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%413 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %415 = tensor.empty() : tensor<32x32xf32>
    %416 = linalg.transpose ins(%407:tensor<32x32xf32>) outs(%415:tensor<32x32xf32>) permutation = [1, 0]
    %417 = tensor.empty() : tensor<32x32xf32>
    %418 = arith.constant 0.000000e+00 : f32
    %419 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%418 : f32) outs(%417 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %420 = linalg.matmul {prov.region_id = "matmul_28", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32", prov.transposed_b = "true"} ins(%400, %416 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%419 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %421 = arith.constant {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} 0.176776692 : f32
    %422 = tensor.splat %421 {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} : tensor<32x32xf32>
    %423 = tensor.empty() : tensor<32x32xf32>
    %424 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%420, %422 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%423 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb25(%425: f32, %426: f32, %427: f32):
      %428 = arith.mulf %425, %426 : f32
      linalg.yield %428 : f32
    } -> tensor<32x32xf32>
    %429 = arith.constant {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %430 = tensor.splat %429 {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %431 = linalg.reduce ins(%424:tensor<32x32xf32>) outs(%430:tensor<32xf32>) dimensions = [1]
    (%432: f32, %433: f32) {
      %434 = arith.maximumf %432, %433 : f32
      linalg.yield %434 : f32
    }
    %435 = tensor.expand_shape %431 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %436 = tensor.empty() : tensor<32x32xf32>
    %437 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%424, %435 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%436 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb26(%438: f32, %439: f32, %440: f32):
      %441 = arith.subf %438, %439 : f32
      linalg.yield %441 : f32
    } -> tensor<32x32xf32>
    %442 = tensor.empty() : tensor<32x32xf32>
    %443 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%437 : tensor<32x32xf32>) outs(%442 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb27(%444: f32, %445: f32):
      %446 = math.exp %444 : f32
      linalg.yield %446 : f32
    } -> tensor<32x32xf32>
    %447 = arith.constant {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %448 = tensor.splat %447 {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %449 = linalg.reduce ins(%443:tensor<32x32xf32>) outs(%448:tensor<32xf32>) dimensions = [1]
    (%450: f32, %451: f32) {
      %452 = arith.addf %450, %451 : f32
      linalg.yield %452 : f32
    }
    %453 = tensor.expand_shape %449 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %454 = tensor.empty() : tensor<32x32xf32>
    %455 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%443, %453 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%454 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_5", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb28(%456: f32, %457: f32, %458: f32):
      %459 = arith.divf %456, %457 : f32
      linalg.yield %459 : f32
    } -> tensor<32x32xf32>
    %460 = tensor.empty() : tensor<32x32xf32>
    %461 = arith.constant 0.000000e+00 : f32
    %462 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%461 : f32) outs(%460 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %463 = linalg.matmul {prov.region_id = "matmul_29", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%455, %414 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%462 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %464 = tensor.empty() : tensor<32x32xf32>
    %465 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%389, %463 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%464 : tensor<32x32xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb29(%466: f32, %467: f32, %468: f32):
      %469 = arith.addf %466, %467 : f32
      linalg.yield %469 : f32
    } -> tensor<32x32xf32>
    %470 = tensor.empty() : tensor<32x32xf32>
    %471 = arith.constant 0.000000e+00 : f32
    %472 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%471 : f32) outs(%470 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %473 = linalg.matmul {prov.region_id = "matmul_30", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%465, %6 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%472 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %474 = tensor.empty() : tensor<32x32xf32>
    %475 = arith.constant 0.000000e+00 : f32
    %476 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%475 : f32) outs(%474 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %477 = linalg.matmul {prov.region_id = "matmul_31", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%473, %7 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%476 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %478 = tensor.empty() : tensor<32x32xf32>
    %479 = arith.constant 0.000000e+00 : f32
    %480 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%479 : f32) outs(%478 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %481 = linalg.matmul {prov.region_id = "matmul_32", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%477, %8 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%480 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %482 = tensor.empty() : tensor<32x32xf32>
    %483 = arith.constant 0.000000e+00 : f32
    %484 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%483 : f32) outs(%482 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %485 = linalg.matmul {prov.region_id = "matmul_33", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%481, %9 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%484 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %486 = tensor.empty() : tensor<32x32xf32>
    %487 = arith.constant 0.000000e+00 : f32
    %488 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%487 : f32) outs(%486 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %489 = linalg.matmul {prov.region_id = "matmul_34", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%485, %10 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%488 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %490 = tensor.empty() : tensor<32x32xf32>
    %491 = arith.constant 0.000000e+00 : f32
    %492 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%491 : f32) outs(%490 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %493 = linalg.matmul {prov.region_id = "matmul_35", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%489, %11 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%492 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %494 = tensor.empty() : tensor<32x32xf32>
    %495 = arith.constant 0.000000e+00 : f32
    %496 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%495 : f32) outs(%494 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %497 = linalg.matmul {prov.region_id = "matmul_36", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "float32"} ins(%493, %12 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%496 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %498 = tensor.empty() : tensor<32x32xf32>
    %499 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%497 : tensor<32x32xf32>) outs(%498 : tensor<32x32xf32>) attrs =  {prov.region_id = "gelu_0", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32"} {
    ^bb30(%500: f32, %501: f32):
      %502 = arith.constant 5.000000e-01 : f32
      %503 = arith.constant 1.000000e+00 : f32
      %504 = arith.constant 0.707106769 : f32
      %505 = arith.mulf %500, %504 : f32
      %506 = math.erf %505 : f32
      %507 = arith.addf %503, %506 : f32
      %508 = arith.mulf %502, %500 : f32
      %509 = arith.mulf %508, %507 : f32
      linalg.yield %509 : f32
    } -> tensor<32x32xf32>
    %510 = tensor.empty() : tensor<32x32xf32>
    %511 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%499 : tensor<32x32xf32>) outs(%510 : tensor<32x32xf32>) attrs =  {prov.region_id = "gelu_1", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32"} {
    ^bb31(%512: f32, %513: f32):
      %514 = arith.constant 5.000000e-01 : f32
      %515 = arith.constant 1.000000e+00 : f32
      %516 = arith.constant 0.707106769 : f32
      %517 = arith.mulf %512, %516 : f32
      %518 = math.erf %517 : f32
      %519 = arith.addf %515, %518 : f32
      %520 = arith.mulf %514, %512 : f32
      %521 = arith.mulf %520, %519 : f32
      linalg.yield %521 : f32
    } -> tensor<32x32xf32>
    %522 = tensor.empty() : tensor<32x32xf32>
    %523 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%511 : tensor<32x32xf32>) outs(%522 : tensor<32x32xf32>) attrs =  {prov.region_id = "gelu_2", prov._pattern_hint = "gelu", prov.op = "gelu", prov.family = "elementwise", prov.aten = "aten.gelu.default", prov.orig_dtype = "float32"} {
    ^bb32(%524: f32, %525: f32):
      %526 = arith.constant 5.000000e-01 : f32
      %527 = arith.constant 1.000000e+00 : f32
      %528 = arith.constant 0.707106769 : f32
      %529 = arith.mulf %524, %528 : f32
      %530 = math.erf %529 : f32
      %531 = arith.addf %527, %530 : f32
      %532 = arith.mulf %526, %524 : f32
      %533 = arith.mulf %532, %531 : f32
      linalg.yield %533 : f32
    } -> tensor<32x32xf32>
    %534 = tensor.empty() : tensor<32x32xf32>
    %535 = linalg.transpose ins(%523:tensor<32x32xf32>) outs(%534:tensor<32x32xf32>) permutation = [1, 0]
    %536 = tensor.empty() : tensor<32x32xf32>
    %537 = linalg.transpose ins(%535:tensor<32x32xf32>) outs(%536:tensor<32x32xf32>) permutation = [1, 0]
    %538 = tensor.empty() : tensor<32x32xf32>
    %539 = linalg.transpose ins(%537:tensor<32x32xf32>) outs(%538:tensor<32x32xf32>) permutation = [1, 0]
    %540 = tensor.empty() : tensor<32x32xf32>
    %541 = linalg.transpose ins(%539:tensor<32x32xf32>) outs(%540:tensor<32x32xf32>) permutation = [1, 0]
    %542 = tensor.empty() : tensor<32x32xf32>
    %543 = linalg.transpose ins(%541:tensor<32x32xf32>) outs(%542:tensor<32x32xf32>) permutation = [1, 0]
    %544 = tensor.empty() : tensor<32x32xf32>
    %545 = linalg.transpose ins(%543:tensor<32x32xf32>) outs(%544:tensor<32x32xf32>) permutation = [1, 0]
    %546 = tensor.empty() : tensor<32x32xf32>
    %547 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%545 : tensor<32x32xf32>) outs(%546 : tensor<32x32xf32>) attrs =  {prov.region_id = "pow_0", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb33(%548: f32, %549: f32):
      %550 = arith.constant 2.000000e+00 : f32
      %551 = math.powf %548, %550 : f32
      linalg.yield %551 : f32
    } -> tensor<32x32xf32>
    %552 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %553 = tensor.splat %552 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32>
    %554 = linalg.reduce ins(%547:tensor<32x32xf32>) outs(%553:tensor<32xf32>) dimensions = [1]
    (%555: f32, %556: f32) {
      %557 = arith.addf %555, %556 : f32
      linalg.yield %557 : f32
    }
    %558 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.200000e+01 : f32
    %559 = tensor.splat %558 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32>
    %560 = tensor.empty() : tensor<32xf32>
    %561 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%554, %559 : tensor<32xf32>, tensor<32xf32>) outs(%560 : tensor<32xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb34(%562: f32, %563: f32, %564: f32):
      %565 = arith.divf %562, %563 : f32
      linalg.yield %565 : f32
    } -> tensor<32xf32>
    %566 = tensor.expand_shape %561 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %567 = arith.constant {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %568 = tensor.splat %567 {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<32x1xf32>
    %569 = tensor.empty() : tensor<32x1xf32>
    %570 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%566, %568 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%569 : tensor<32x1xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb35(%571: f32, %572: f32, %573: f32):
      %574 = arith.addf %571, %572 : f32
      linalg.yield %574 : f32
    } -> tensor<32x1xf32>
    %575 = tensor.empty() : tensor<32x1xf32>
    %576 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%570 : tensor<32x1xf32>) outs(%575 : tensor<32x1xf32>) attrs =  {prov.region_id = "rsqrt_0", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb36(%577: f32, %578: f32):
      %579 = math.rsqrt %577 : f32
      linalg.yield %579 : f32
    } -> tensor<32x1xf32>
    %580 = tensor.empty() : tensor<32x32xf32>
    %581 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%545, %576 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%580 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_6", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb37(%582: f32, %583: f32, %584: f32):
      %585 = arith.mulf %582, %583 : f32
      linalg.yield %585 : f32
    } -> tensor<32x32xf32>
    %586 = tensor.empty() : tensor<32x32xf32>
    %587 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%581 : tensor<32x32xf32>) outs(%586 : tensor<32x32xf32>) attrs =  {prov.region_id = "pow_1", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb38(%588: f32, %589: f32):
      %590 = arith.constant 2.000000e+00 : f32
      %591 = math.powf %588, %590 : f32
      linalg.yield %591 : f32
    } -> tensor<32x32xf32>
    %592 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %593 = tensor.splat %592 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32>
    %594 = linalg.reduce ins(%587:tensor<32x32xf32>) outs(%593:tensor<32xf32>) dimensions = [1]
    (%595: f32, %596: f32) {
      %597 = arith.addf %595, %596 : f32
      linalg.yield %597 : f32
    }
    %598 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.200000e+01 : f32
    %599 = tensor.splat %598 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32>
    %600 = tensor.empty() : tensor<32xf32>
    %601 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%594, %599 : tensor<32xf32>, tensor<32xf32>) outs(%600 : tensor<32xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb39(%602: f32, %603: f32, %604: f32):
      %605 = arith.divf %602, %603 : f32
      linalg.yield %605 : f32
    } -> tensor<32xf32>
    %606 = tensor.expand_shape %601 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %607 = arith.constant {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %608 = tensor.splat %607 {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<32x1xf32>
    %609 = tensor.empty() : tensor<32x1xf32>
    %610 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%606, %608 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%609 : tensor<32x1xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb40(%611: f32, %612: f32, %613: f32):
      %614 = arith.addf %611, %612 : f32
      linalg.yield %614 : f32
    } -> tensor<32x1xf32>
    %615 = tensor.empty() : tensor<32x1xf32>
    %616 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%610 : tensor<32x1xf32>) outs(%615 : tensor<32x1xf32>) attrs =  {prov.region_id = "rsqrt_1", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb41(%617: f32, %618: f32):
      %619 = math.rsqrt %617 : f32
      linalg.yield %619 : f32
    } -> tensor<32x1xf32>
    %620 = tensor.empty() : tensor<32x32xf32>
    %621 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%581, %616 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%620 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_7", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb42(%622: f32, %623: f32, %624: f32):
      %625 = arith.mulf %622, %623 : f32
      linalg.yield %625 : f32
    } -> tensor<32x32xf32>
    %626 = tensor.empty() : tensor<32x32xf32>
    %627 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%621 : tensor<32x32xf32>) outs(%626 : tensor<32x32xf32>) attrs =  {prov.region_id = "pow_2", prov.family = "elementwise", prov._pattern_hint = "pow", prov.op = "pow", prov.aten = "aten.pow.Tensor_Scalar", prov.orig_dtype = "float32"} {
    ^bb43(%628: f32, %629: f32):
      %630 = arith.constant 2.000000e+00 : f32
      %631 = math.powf %628, %630 : f32
      linalg.yield %631 : f32
    } -> tensor<32x32xf32>
    %632 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %633 = tensor.splat %632 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32>
    %634 = linalg.reduce ins(%627:tensor<32x32xf32>) outs(%633:tensor<32xf32>) dimensions = [1]
    (%635: f32, %636: f32) {
      %637 = arith.addf %635, %636 : f32
      linalg.yield %637 : f32
    }
    %638 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} 3.200000e+01 : f32
    %639 = tensor.splat %638 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32>
    %640 = tensor.empty() : tensor<32xf32>
    %641 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%634, %639 : tensor<32xf32>, tensor<32xf32>) outs(%640 : tensor<32xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} {
    ^bb44(%642: f32, %643: f32, %644: f32):
      %645 = arith.divf %642, %643 : f32
      linalg.yield %645 : f32
    } -> tensor<32xf32>
    %646 = tensor.expand_shape %641 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_mean", prov.op = "reduce_mean", prov.aten = "aten.mean.dim", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %647 = arith.constant {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} 1.000000e-06 : f32
    %648 = tensor.splat %647 {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} : tensor<32x1xf32>
    %649 = tensor.empty() : tensor<32x1xf32>
    %650 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%646, %648 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%649 : tensor<32x1xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "float32"} {
    ^bb45(%651: f32, %652: f32, %653: f32):
      %654 = arith.addf %651, %652 : f32
      linalg.yield %654 : f32
    } -> tensor<32x1xf32>
    %655 = tensor.empty() : tensor<32x1xf32>
    %656 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%650 : tensor<32x1xf32>) outs(%655 : tensor<32x1xf32>) attrs =  {prov.region_id = "rsqrt_2", prov._pattern_hint = "rsqrt", prov.op = "rsqrt", prov.family = "elementwise", prov.aten = "aten.rsqrt.default", prov.orig_dtype = "float32"} {
    ^bb46(%657: f32, %658: f32):
      %659 = math.rsqrt %657 : f32
      linalg.yield %659 : f32
    } -> tensor<32x1xf32>
    %660 = tensor.empty() : tensor<32x32xf32>
    %661 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%621, %656 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%660 : tensor<32x32xf32>) attrs =  {prov.region_id = "mul_8", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32"} {
    ^bb47(%662: f32, %663: f32, %664: f32):
      %665 = arith.mulf %662, %663 : f32
      linalg.yield %665 : f32
    } -> tensor<32x32xf32>
    %666 = arith.constant {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %667 = tensor.splat %666 {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<32xf32>
    %668 = linalg.reduce ins(%661:tensor<32x32xf32>) outs(%667:tensor<32xf32>) dimensions = [1]
    (%669: f32, %670: f32) {
      %671 = arith.addf %669, %670 : f32
      linalg.yield %671 : f32
    }
    %672 = tensor.expand_shape %668 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "reduce_3", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %673 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 3.200000e+01 : f32
    %674 = tensor.splat %673 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<32x1xf32>
    %675 = tensor.empty() : tensor<32x1xf32>
    %676 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%672, %674 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%675 : tensor<32x1xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb48(%677: f32, %678: f32, %679: f32):
      %680 = arith.divf %677, %678 : f32
      linalg.yield %680 : f32
    } -> tensor<32x1xf32>
    %681 = tensor.empty() : tensor<32x32xf32>
    %682 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%661, %676 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%681 : tensor<32x32xf32>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} {
    ^bb49(%683: f32, %684: f32, %685: f32):
      %686 = arith.subf %683, %684 : f32
      linalg.yield %686 : f32
    } -> tensor<32x32xf32>
    %687 = arith.constant {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %688 = tensor.splat %687 {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<32xf32>
    %689 = linalg.reduce ins(%682:tensor<32x32xf32>) outs(%688:tensor<32xf32>) dimensions = [1]
    (%690: f32, %691: f32) {
      %692 = arith.addf %690, %691 : f32
      linalg.yield %692 : f32
    }
    %693 = tensor.expand_shape %689 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "reduce_4", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %694 = arith.constant {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 3.200000e+01 : f32
    %695 = tensor.splat %694 {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<32x1xf32>
    %696 = tensor.empty() : tensor<32x1xf32>
    %697 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%693, %695 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%696 : tensor<32x1xf32>) attrs =  {prov.region_id = "div_1", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb50(%698: f32, %699: f32, %700: f32):
      %701 = arith.divf %698, %699 : f32
      linalg.yield %701 : f32
    } -> tensor<32x1xf32>
    %702 = tensor.empty() : tensor<32x32xf32>
    %703 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%682, %697 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%702 : tensor<32x32xf32>) attrs =  {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} {
    ^bb51(%704: f32, %705: f32, %706: f32):
      %707 = arith.subf %704, %705 : f32
      linalg.yield %707 : f32
    } -> tensor<32x32xf32>
    %708 = arith.constant {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %709 = tensor.splat %708 {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<32xf32>
    %710 = linalg.reduce ins(%703:tensor<32x32xf32>) outs(%709:tensor<32xf32>) dimensions = [1]
    (%711: f32, %712: f32) {
      %713 = arith.addf %711, %712 : f32
      linalg.yield %713 : f32
    }
    %714 = tensor.expand_shape %710 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "reduce_5", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.sum.dim_IntList", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %715 = arith.constant {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} 3.200000e+01 : f32
    %716 = tensor.splat %715 {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} : tensor<32x1xf32>
    %717 = tensor.empty() : tensor<32x1xf32>
    %718 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%714, %716 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%717 : tensor<32x1xf32>) attrs =  {prov.region_id = "div_2", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "float32"} {
    ^bb52(%719: f32, %720: f32, %721: f32):
      %722 = arith.divf %719, %720 : f32
      linalg.yield %722 : f32
    } -> tensor<32x1xf32>
    %723 = tensor.empty() : tensor<32x32xf32>
    %724 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%703, %718 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%723 : tensor<32x32xf32>) attrs =  {prov.region_id = "sub_2", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "float32"} {
    ^bb53(%725: f32, %726: f32, %727: f32):
      %728 = arith.subf %725, %726 : f32
      linalg.yield %728 : f32
    } -> tensor<32x32xf32>
    %729 = arith.constant {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %730 = tensor.splat %729 {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %731 = linalg.reduce ins(%724:tensor<32x32xf32>) outs(%730:tensor<32xf32>) dimensions = [1]
    (%732: f32, %733: f32) {
      %734 = arith.maximumf %732, %733 : f32
      linalg.yield %734 : f32
    }
    %735 = tensor.expand_shape %731 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %736 = tensor.empty() : tensor<32x32xf32>
    %737 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%724, %735 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%736 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb54(%738: f32, %739: f32, %740: f32):
      %741 = arith.subf %738, %739 : f32
      linalg.yield %741 : f32
    } -> tensor<32x32xf32>
    %742 = tensor.empty() : tensor<32x32xf32>
    %743 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%737 : tensor<32x32xf32>) outs(%742 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb55(%744: f32, %745: f32):
      %746 = math.exp %744 : f32
      linalg.yield %746 : f32
    } -> tensor<32x32xf32>
    %747 = arith.constant {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %748 = tensor.splat %747 {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %749 = linalg.reduce ins(%743:tensor<32x32xf32>) outs(%748:tensor<32xf32>) dimensions = [1]
    (%750: f32, %751: f32) {
      %752 = arith.addf %750, %751 : f32
      linalg.yield %752 : f32
    }
    %753 = tensor.expand_shape %749 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %754 = tensor.empty() : tensor<32x32xf32>
    %755 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%743, %753 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%754 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_6", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb56(%756: f32, %757: f32, %758: f32):
      %759 = arith.divf %756, %757 : f32
      linalg.yield %759 : f32
    } -> tensor<32x32xf32>
    %760 = arith.constant {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %761 = tensor.splat %760 {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %762 = linalg.reduce ins(%755:tensor<32x32xf32>) outs(%761:tensor<32xf32>) dimensions = [1]
    (%763: f32, %764: f32) {
      %765 = arith.maximumf %763, %764 : f32
      linalg.yield %765 : f32
    }
    %766 = tensor.expand_shape %762 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %767 = tensor.empty() : tensor<32x32xf32>
    %768 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%755, %766 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%767 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb57(%769: f32, %770: f32, %771: f32):
      %772 = arith.subf %769, %770 : f32
      linalg.yield %772 : f32
    } -> tensor<32x32xf32>
    %773 = tensor.empty() : tensor<32x32xf32>
    %774 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%768 : tensor<32x32xf32>) outs(%773 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb58(%775: f32, %776: f32):
      %777 = math.exp %775 : f32
      linalg.yield %777 : f32
    } -> tensor<32x32xf32>
    %778 = arith.constant {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %779 = tensor.splat %778 {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %780 = linalg.reduce ins(%774:tensor<32x32xf32>) outs(%779:tensor<32xf32>) dimensions = [1]
    (%781: f32, %782: f32) {
      %783 = arith.addf %781, %782 : f32
      linalg.yield %783 : f32
    }
    %784 = tensor.expand_shape %780 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %785 = tensor.empty() : tensor<32x32xf32>
    %786 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%774, %784 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%785 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_7", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb59(%787: f32, %788: f32, %789: f32):
      %790 = arith.divf %787, %788 : f32
      linalg.yield %790 : f32
    } -> tensor<32x32xf32>
    %791 = arith.constant {prov.region_id = "softmax_8", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0xff800000 : f32
    %792 = tensor.splat %791 {prov.region_id = "softmax_8", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %793 = linalg.reduce ins(%786:tensor<32x32xf32>) outs(%792:tensor<32xf32>) dimensions = [1]
    (%794: f32, %795: f32) {
      %796 = arith.maximumf %794, %795 : f32
      linalg.yield %796 : f32
    }
    %797 = tensor.expand_shape %793 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_8", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %798 = tensor.empty() : tensor<32x32xf32>
    %799 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%786, %797 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%798 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_8", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb60(%800: f32, %801: f32, %802: f32):
      %803 = arith.subf %800, %801 : f32
      linalg.yield %803 : f32
    } -> tensor<32x32xf32>
    %804 = tensor.empty() : tensor<32x32xf32>
    %805 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%799 : tensor<32x32xf32>) outs(%804 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_8", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb61(%806: f32, %807: f32):
      %808 = math.exp %806 : f32
      linalg.yield %808 : f32
    } -> tensor<32x32xf32>
    %809 = arith.constant {prov.region_id = "softmax_8", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} 0.000000e+00 : f32
    %810 = tensor.splat %809 {prov.region_id = "softmax_8", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32>
    %811 = linalg.reduce ins(%805:tensor<32x32xf32>) outs(%810:tensor<32xf32>) dimensions = [1]
    (%812: f32, %813: f32) {
      %814 = arith.addf %812, %813 : f32
      linalg.yield %814 : f32
    }
    %815 = tensor.expand_shape %811 [[0 : i64, 1 : i64]] output_shape [32, 1] {prov.region_id = "softmax_8", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} : tensor<32xf32> into tensor<32x1xf32>
    %816 = tensor.empty() : tensor<32x32xf32>
    %817 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%805, %815 : tensor<32x32xf32>, tensor<32x1xf32>) outs(%816 : tensor<32x32xf32>) attrs =  {prov.region_id = "softmax_8", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "float32"} {
    ^bb62(%818: f32, %819: f32, %820: f32):
      %821 = arith.divf %818, %819 : f32
      linalg.yield %821 : f32
    } -> tensor<32x32xf32>
    func.return %817 : tensor<32x32xf32>
  }
}
