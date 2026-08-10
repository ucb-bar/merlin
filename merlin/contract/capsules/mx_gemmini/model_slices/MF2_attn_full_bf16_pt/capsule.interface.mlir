builtin.module attributes {prov.weights_file = "/tmp/capsule_m2m__6wsou4e/weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<16x64xbf16>, %1: tensor<16x64xbf16>, %2: tensor<16x64xbf16>) -> tensor<16x64xbf16> {
    %3 = tensor.empty() : tensor<64x16xbf16>
    %4 = linalg.transpose ins(%1:tensor<16x64xbf16>) outs(%3:tensor<64x16xbf16>) permutation = [1, 0]
    %5 = tensor.empty() : tensor<16x16xbf16>
    %6 = arith.constant 0.000000e+00 : bf16
    %7 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%6 : bf16) outs(%5 : tensor<16x16xbf16>) -> tensor<16x16xbf16>
    %8 = linalg.matmul {prov.region_id = "matmul_0", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "bfloat16", prov.transposed_b = "true"} ins(%0, %4 : tensor<16x64xbf16>, tensor<64x16xbf16>) outs(%7 : tensor<16x16xbf16>) -> tensor<16x16xbf16>
    %9 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} 8.000000e+00 : bf16
    %10 = tensor.splat %9 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} : tensor<16x16xbf16>
    %11 = tensor.empty() : tensor<16x16xbf16>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %10 : tensor<16x16xbf16>, tensor<16x16xbf16>) outs(%11 : tensor<16x16xbf16>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.div.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb0(%13: bf16, %14: bf16, %15: bf16):
      %16 = arith.divf %13, %14 : bf16
      linalg.yield %16 : bf16
    } -> tensor<16x16xbf16>
    %17 = arith.constant {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "bfloat16"} -3.389530e+38 : bf16
    %18 = tensor.splat %17 {prov.region_id = "fill_0", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.full.default", prov.orig_dtype = "bfloat16"} : tensor<16x16xbf16>
    %19 = tensor.empty() : tensor<16xi64>
    %20 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%19 : tensor<16xi64>) attrs =  {prov.region_id = "iota_0", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb1(%21: i64):
      %22 = linalg.index 0 : index
      %23 = arith.index_cast %22 : index to i64
      %24 = arith.constant 1 : i64
      %25 = arith.muli %23, %24 : i64
      %26 = arith.constant 0 : i64
      %27 = arith.addi %26, %25 : i64
      linalg.yield %27 : i64
    } -> tensor<16xi64>
    %28 = tensor.expand_shape %20 [[0 : i64, 1 : i64]] output_shape [1, 16] {prov.region_id = "unsqueeze_0", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<16xi64> into tensor<1x16xi64>
    %29 = tensor.empty() : tensor<16xi64>
    %30 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%29 : tensor<16xi64>) attrs =  {prov.region_id = "iota_1", prov.family = "iota", prov._pattern_hint = "arange", prov.op = "arange", prov.aten = "aten.arange.start_step", prov.orig_dtype = "int64"} {
    ^bb2(%31: i64):
      %32 = linalg.index 0 : index
      %33 = arith.index_cast %32 : index to i64
      %34 = arith.constant 1 : i64
      %35 = arith.muli %33, %34 : i64
      %36 = arith.constant 0 : i64
      %37 = arith.addi %36, %35 : i64
      linalg.yield %37 : i64
    } -> tensor<16xi64>
    %38 = tensor.expand_shape %30 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "unsqueeze_1", prov._pattern_hint = "unsqueeze", prov.op = "unsqueeze", prov.family = "layout", prov.aten = "aten.unsqueeze.default", prov.orig_dtype = "int64"} : tensor<16xi64> into tensor<16x1xi64>
    %39 = tensor.empty() : tensor<16x16xi64>
    %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%28, %38 : tensor<1x16xi64>, tensor<16x1xi64>) outs(%39 : tensor<16x16xi64>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.sub.Tensor", prov.orig_dtype = "int64"} {
    ^bb3(%41: i64, %42: i64, %43: i64):
      %44 = arith.subi %41, %42 : i64
      linalg.yield %44 : i64
    } -> tensor<16x16xi64>
    %45 = arith.constant {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool"} 1 : i64
    %46 = tensor.splat %45 {prov._pattern_hint = "compare", prov.op = "compare", prov.family = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool"} : tensor<16x16xi64>
    %47 = tensor.empty() : tensor<16x16xi1>
    %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%40, %46 : tensor<16x16xi64>, tensor<16x16xi64>) outs(%47 : tensor<16x16xi1>) attrs =  {prov.region_id = "compare_0", prov.family = "compare", prov._pattern_hint = "compare", prov.op = "compare", prov.aten = "aten.ge.Scalar", prov.orig_dtype = "bool"} {
    ^bb4(%49: i64, %50: i64, %51: i1):
      %52 = arith.cmpi sge, %49, %50 : i64
      linalg.yield %52 : i1
    } -> tensor<16x16xi1>
    %53 = arith.constant {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %54 = tensor.splat %53 {prov.region_id = "fill_1", prov.family = "fill", prov._pattern_hint = "fill", prov.op = "fill", prov.aten = "aten.scalar_tensor.default", prov.orig_dtype = "bfloat16"} : tensor<bf16>
    %55 = tensor.empty() : tensor<16x16xbf16>
    %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%48, %18, %54 : tensor<16x16xi1>, tensor<16x16xbf16>, tensor<bf16>) outs(%55 : tensor<16x16xbf16>) attrs =  {prov.region_id = "select_0", prov.family = "layout", prov._pattern_hint = "select", prov.op = "select", prov.aten = "aten.where.self", prov.orig_dtype = "bfloat16"} {
    ^bb5(%57: i1, %58: bf16, %59: bf16, %60: bf16):
      %61 = arith.select %57, %58, %59 : bf16
      linalg.yield %61 : bf16
    } -> tensor<16x16xbf16>
    %62 = tensor.empty() : tensor<16x16xbf16>
    %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%12, %56 : tensor<16x16xbf16>, tensor<16x16xbf16>) outs(%62 : tensor<16x16xbf16>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add.Tensor", prov.orig_dtype = "bfloat16"} {
    ^bb6(%64: bf16, %65: bf16, %66: bf16):
      %67 = arith.addf %64, %65 : bf16
      linalg.yield %67 : bf16
    } -> tensor<16x16xbf16>
    %68 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} 0xff80 : bf16
    %69 = tensor.splat %68 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %70 = linalg.reduce ins(%63:tensor<16x16xbf16>) outs(%69:tensor<16xbf16>) dimensions = [1]
    (%71: bf16, %72: bf16) {
      %73 = arith.maximumf %71, %72 : bf16
      linalg.yield %73 : bf16
    }
    %74 = tensor.expand_shape %70 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<16xbf16> into tensor<16x1xbf16>
    %75 = tensor.empty() : tensor<16x16xbf16>
    %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %74 : tensor<16x16xbf16>, tensor<16x1xbf16>) outs(%75 : tensor<16x16xbf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} {
    ^bb7(%77: bf16, %78: bf16, %79: bf16):
      %80 = arith.subf %77, %78 : bf16
      linalg.yield %80 : bf16
    } -> tensor<16x16xbf16>
    %81 = tensor.empty() : tensor<16x16xbf16>
    %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%76 : tensor<16x16xbf16>) outs(%81 : tensor<16x16xbf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} {
    ^bb8(%83: bf16, %84: bf16):
      %85 = math.exp %83 : bf16
      linalg.yield %85 : bf16
    } -> tensor<16x16xbf16>
    %86 = arith.constant {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} 0.000000e+00 : bf16
    %87 = tensor.splat %86 {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<16xbf16>
    %88 = linalg.reduce ins(%82:tensor<16x16xbf16>) outs(%87:tensor<16xbf16>) dimensions = [1]
    (%89: bf16, %90: bf16) {
      %91 = arith.addf %89, %90 : bf16
      linalg.yield %91 : bf16
    }
    %92 = tensor.expand_shape %88 [[0 : i64, 1 : i64]] output_shape [16, 1] {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} : tensor<16xbf16> into tensor<16x1xbf16>
    %93 = tensor.empty() : tensor<16x16xbf16>
    %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %92 : tensor<16x16xbf16>, tensor<16x1xbf16>) outs(%93 : tensor<16x16xbf16>) attrs =  {prov.region_id = "softmax_0", prov.family = "normalization", prov._pattern_hint = "softmax", prov.op = "softmax", prov.aten = "aten._softmax.default", prov.orig_dtype = "bfloat16"} {
    ^bb9(%95: bf16, %96: bf16, %97: bf16):
      %98 = arith.divf %95, %96 : bf16
      linalg.yield %98 : bf16
    } -> tensor<16x16xbf16>
    %99 = tensor.empty() : tensor<16x64xbf16>
    %100 = arith.constant 0.000000e+00 : bf16
    %101 = linalg.fill {prov.op = "fill", prov.family = "fill"} ins(%100 : bf16) outs(%99 : tensor<16x64xbf16>) -> tensor<16x64xbf16>
    %102 = linalg.matmul {prov.region_id = "matmul_1", prov.op = "matmul", prov.family = "contraction", prov.aten = "aten.mm.default", prov.orig_dtype = "bfloat16"} ins(%94, %2 : tensor<16x16xbf16>, tensor<16x64xbf16>) outs(%101 : tensor<16x64xbf16>) -> tensor<16x64xbf16>
    func.return %102 : tensor<16x64xbf16>
  }
}
