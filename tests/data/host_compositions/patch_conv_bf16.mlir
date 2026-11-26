builtin.module {
  func.func @forward(%0: tensor<8x3x4x4xbf16>, %1: tensor<8xbf16>, %2: tensor<5x3x2x2xbf16>, %3: tensor<1x3x8x8xbf16>) -> (tensor<1x8x2x2xbf16>, tensor<1x5x4x4xbf16>) {
    %4 = tensor.empty() : tensor<1x3x8x8xf32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<1x3x8x8xbf16>) outs(%4 : tensor<1x3x8x8xf32>) {
    ^bb0(%6: bf16, %7: f32):
      %8 = arith.extf %6 : bf16 to f32
      linalg.yield %8 : f32
    } -> tensor<1x3x8x8xf32>
    %9 = tensor.empty() : tensor<8x3x4x4xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0 : tensor<8x3x4x4xbf16>) outs(%9 : tensor<8x3x4x4xf32>) {
    ^bb1(%11: bf16, %12: f32):
      %13 = arith.extf %11 : bf16 to f32
      linalg.yield %13 : f32
    } -> tensor<8x3x4x4xf32>
    %14 = tensor.empty() : tensor<8xf32>
    %15 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1 : tensor<8xbf16>) outs(%14 : tensor<8xf32>) {
    ^bb2(%16: bf16, %17: f32):
      %18 = arith.extf %16 : bf16 to f32
      linalg.yield %18 : f32
    } -> tensor<8xf32>
    %19 = tensor.empty() : tensor<3x4x4x1x2x2xf32>
    %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 4) + d1), ((d5 * 4) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%5 : tensor<1x3x8x8xf32>) outs(%19 : tensor<3x4x4x1x2x2xf32>) {
    ^bb3(%21: f32, %22: f32):
      linalg.yield %21 : f32
    } -> tensor<3x4x4x1x2x2xf32>
    %23 = tensor.collapse_shape %20 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] : tensor<3x4x4x1x2x2xf32> into tensor<192xf32>
    %24 = tensor.expand_shape %23 [[0 : i64, 1 : i64]] output_shape [48, 4] : tensor<192xf32> into tensor<48x4xf32>
    %25 = tensor.collapse_shape %10 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<8x3x4x4xf32> into tensor<384xf32>
    %26 = tensor.expand_shape %25 [[0 : i64, 1 : i64]] output_shape [8, 48] : tensor<384xf32> into tensor<8x48xf32>
    %27 = arith.constant 0.000000e+00 : f32
    %28 = tensor.splat %27 : tensor<8x4xf32>
    %29 = linalg.matmul ins(%26, %24 : tensor<8x48xf32>, tensor<48x4xf32>) outs(%28 : tensor<8x4xf32>) -> tensor<8x4xf32>
    %30 = tensor.collapse_shape %29 [[0 : i64, 1 : i64]] : tensor<8x4xf32> into tensor<32xf32>
    %31 = tensor.expand_shape %30 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [8, 1, 2, 2] : tensor<32xf32> into tensor<8x1x2x2xf32>
    %32 = tensor.collapse_shape %31 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<8x1x2x2xf32> into tensor<32xf32>
    %33 = tensor.expand_shape %32 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 8, 2, 2] : tensor<32xf32> into tensor<1x8x2x2xf32>
    %34 = tensor.empty() : tensor<1x8x2x2xf32>
    %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33, %15 : tensor<1x8x2x2xf32>, tensor<8xf32>) outs(%34 : tensor<1x8x2x2xf32>) {
    ^bb4(%36: f32, %37: f32, %38: f32):
      %39 = arith.addf %36, %37 : f32
      linalg.yield %39 : f32
    } -> tensor<1x8x2x2xf32>
    %40 = tensor.empty() : tensor<1x8x2x2xbf16>
    %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%35 : tensor<1x8x2x2xf32>) outs(%40 : tensor<1x8x2x2xbf16>) {
    ^bb5(%42: f32, %43: bf16):
      %44 = arith.truncf %42 : f32 to bf16
      linalg.yield %44 : bf16
    } -> tensor<1x8x2x2xbf16>
    %45 = tensor.empty() : tensor<1x3x8x8xf32>
    %46 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<1x3x8x8xbf16>) outs(%45 : tensor<1x3x8x8xf32>) {
    ^bb6(%47: bf16, %48: f32):
      %49 = arith.extf %47 : bf16 to f32
      linalg.yield %49 : f32
    } -> tensor<1x3x8x8xf32>
    %50 = tensor.empty() : tensor<5x3x2x2xf32>
    %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<5x3x2x2xbf16>) outs(%50 : tensor<5x3x2x2xf32>) {
    ^bb7(%52: bf16, %53: f32):
      %54 = arith.extf %52 : bf16 to f32
      linalg.yield %54 : f32
    } -> tensor<5x3x2x2xf32>
    %55 = tensor.empty() : tensor<3x2x2x1x4x4xf32>
    %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 2) + d1), ((d5 * 2) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%46 : tensor<1x3x8x8xf32>) outs(%55 : tensor<3x2x2x1x4x4xf32>) {
    ^bb8(%57: f32, %58: f32):
      linalg.yield %57 : f32
    } -> tensor<3x2x2x1x4x4xf32>
    %59 = tensor.collapse_shape %56 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] : tensor<3x2x2x1x4x4xf32> into tensor<192xf32>
    %60 = tensor.expand_shape %59 [[0 : i64, 1 : i64]] output_shape [12, 16] : tensor<192xf32> into tensor<12x16xf32>
    %61 = tensor.collapse_shape %51 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<5x3x2x2xf32> into tensor<60xf32>
    %62 = tensor.expand_shape %61 [[0 : i64, 1 : i64]] output_shape [5, 12] : tensor<60xf32> into tensor<5x12xf32>
    %63 = arith.constant 0.000000e+00 : f32
    %64 = tensor.splat %63 : tensor<5x16xf32>
    %65 = linalg.matmul ins(%62, %60 : tensor<5x12xf32>, tensor<12x16xf32>) outs(%64 : tensor<5x16xf32>) -> tensor<5x16xf32>
    %66 = tensor.collapse_shape %65 [[0 : i64, 1 : i64]] : tensor<5x16xf32> into tensor<80xf32>
    %67 = tensor.expand_shape %66 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [5, 1, 4, 4] : tensor<80xf32> into tensor<5x1x4x4xf32>
    %68 = tensor.collapse_shape %67 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<5x1x4x4xf32> into tensor<80xf32>
    %69 = tensor.expand_shape %68 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 5, 4, 4] : tensor<80xf32> into tensor<1x5x4x4xf32>
    %70 = tensor.empty() : tensor<1x5x4x4xbf16>
    %71 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69 : tensor<1x5x4x4xf32>) outs(%70 : tensor<1x5x4x4xbf16>) {
    ^bb9(%72: f32, %73: bf16):
      %74 = arith.truncf %72 : f32 to bf16
      linalg.yield %74 : bf16
    } -> tensor<1x5x4x4xbf16>
    func.return %41, %71 : tensor<1x8x2x2xbf16>, tensor<1x5x4x4xbf16>
  }
}