builtin.module {
  func.func @forward(%0: tensor<8xbf16>, %1: tensor<8xbf16>, %2: tensor<2x3x8xbf16>) -> (tensor<2x3x8xbf16>, tensor<2x3x8xbf16>) {
    %3 = tensor.empty() : tensor<2x3x8xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<2x3x8xbf16>) outs(%3 : tensor<2x3x8xf32>) {
    ^bb0(%5: bf16, %6: f32):
      %7 = arith.extf %5 : bf16 to f32
      linalg.yield %7 : f32
    } -> tensor<2x3x8xf32>
    %8 = tensor.empty() : tensor<8xf32>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%0 : tensor<8xbf16>) outs(%8 : tensor<8xf32>) {
    ^bb1(%10: bf16, %11: f32):
      %12 = arith.extf %10 : bf16 to f32
      linalg.yield %12 : f32
    } -> tensor<8xf32>
    %13 = tensor.empty() : tensor<8xf32>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1 : tensor<8xbf16>) outs(%13 : tensor<8xf32>) {
    ^bb2(%15: bf16, %16: f32):
      %17 = arith.extf %15 : bf16 to f32
      linalg.yield %17 : f32
    } -> tensor<8xf32>
    %18 = arith.constant 0.000000e+00 : f32
    %19 = tensor.splat %18 : tensor<2x3xf32>
    %20 = linalg.reduce ins(%4:tensor<2x3x8xf32>) outs(%19:tensor<2x3xf32>) dimensions = [2]
    (%21: f32, %22: f32) {
      %23 = arith.addf %21, %22 : f32
      linalg.yield %23 : f32
    }
    %24 = arith.constant 8.000000e+00 : f32
    %25 = tensor.splat %24 : tensor<2x3xf32>
    %26 = tensor.empty() : tensor<2x3xf32>
    %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %25 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%26 : tensor<2x3xf32>) {
    ^bb3(%28: f32, %29: f32, %30: f32):
      %31 = arith.divf %28, %29 : f32
      linalg.yield %31 : f32
    } -> tensor<2x3xf32>
    %32 = tensor.collapse_shape %27 [[0 : i64, 1 : i64]] : tensor<2x3xf32> into tensor<6xf32>
    %33 = tensor.expand_shape %32 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 3, 1] : tensor<6xf32> into tensor<2x3x1xf32>
    %34 = tensor.empty() : tensor<2x3x8xf32>
    %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%4, %33 : tensor<2x3x8xf32>, tensor<2x3x1xf32>) outs(%34 : tensor<2x3x8xf32>) {
    ^bb4(%36: f32, %37: f32, %38: f32):
      %39 = arith.subf %36, %37 : f32
      linalg.yield %39 : f32
    } -> tensor<2x3x8xf32>
    %40 = tensor.empty() : tensor<2x3x8xf32>
    %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %35 : tensor<2x3x8xf32>, tensor<2x3x8xf32>) outs(%40 : tensor<2x3x8xf32>) {
    ^bb5(%42: f32, %43: f32, %44: f32):
      %45 = arith.mulf %42, %43 : f32
      linalg.yield %45 : f32
    } -> tensor<2x3x8xf32>
    %46 = arith.constant 0.000000e+00 : f32
    %47 = tensor.splat %46 : tensor<2x3xf32>
    %48 = linalg.reduce ins(%41:tensor<2x3x8xf32>) outs(%47:tensor<2x3xf32>) dimensions = [2]
    (%49: f32, %50: f32) {
      %51 = arith.addf %49, %50 : f32
      linalg.yield %51 : f32
    }
    %52 = arith.constant 8.000000e+00 : f32
    %53 = tensor.splat %52 : tensor<2x3xf32>
    %54 = tensor.empty() : tensor<2x3xf32>
    %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%48, %53 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%54 : tensor<2x3xf32>) {
    ^bb6(%56: f32, %57: f32, %58: f32):
      %59 = arith.divf %56, %57 : f32
      linalg.yield %59 : f32
    } -> tensor<2x3xf32>
    %60 = tensor.collapse_shape %55 [[0 : i64, 1 : i64]] : tensor<2x3xf32> into tensor<6xf32>
    %61 = tensor.expand_shape %60 [[0 : i64, 1 : i64, 2 : i64]] output_shape [2, 3, 1] : tensor<6xf32> into tensor<2x3x1xf32>
    %62 = arith.constant 1.000000e-06 : f32
    %63 = tensor.splat %62 : tensor<2x3x1xf32>
    %64 = tensor.empty() : tensor<2x3x1xf32>
    %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %63 : tensor<2x3x1xf32>, tensor<2x3x1xf32>) outs(%64 : tensor<2x3x1xf32>) {
    ^bb7(%66: f32, %67: f32, %68: f32):
      %69 = arith.addf %66, %67 : f32
      linalg.yield %69 : f32
    } -> tensor<2x3x1xf32>
    %70 = tensor.empty() : tensor<2x3x1xf32>
    %71 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%65 : tensor<2x3x1xf32>) outs(%70 : tensor<2x3x1xf32>) {
    ^bb8(%72: f32, %73: f32):
      %74 = math.rsqrt %72 : f32
      linalg.yield %74 : f32
    } -> tensor<2x3x1xf32>
    %75 = tensor.empty() : tensor<2x3x8xf32>
    %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %71 : tensor<2x3x8xf32>, tensor<2x3x1xf32>) outs(%75 : tensor<2x3x8xf32>) {
    ^bb9(%77: f32, %78: f32, %79: f32):
      %80 = arith.mulf %77, %78 : f32
      linalg.yield %80 : f32
    } -> tensor<2x3x8xf32>
    %81 = tensor.empty() : tensor<2x3x8xf32>
    %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76, %9 : tensor<2x3x8xf32>, tensor<8xf32>) outs(%81 : tensor<2x3x8xf32>) {
    ^bb10(%83: f32, %84: f32, %85: f32):
      %86 = arith.mulf %83, %84 : f32
      linalg.yield %86 : f32
    } -> tensor<2x3x8xf32>
    %87 = tensor.empty() : tensor<2x3x8xf32>
    %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %14 : tensor<2x3x8xf32>, tensor<8xf32>) outs(%87 : tensor<2x3x8xf32>) {
    ^bb11(%89: f32, %90: f32, %91: f32):
      %92 = arith.addf %89, %90 : f32
      linalg.yield %92 : f32
    } -> tensor<2x3x8xf32>
    %93 = tensor.empty() : tensor<2x3x8xbf16>
    %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88 : tensor<2x3x8xf32>) outs(%93 : tensor<2x3x8xbf16>) {
    ^bb12(%95: f32, %96: bf16):
      %97 = arith.truncf %95 : f32 to bf16
      linalg.yield %97 : bf16
    } -> tensor<2x3x8xbf16>
    %98 = tensor.empty() : tensor<2x3x8xf32>
    %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%94 : tensor<2x3x8xbf16>) outs(%98 : tensor<2x3x8xf32>) {
    ^bb13(%100: bf16, %101: f32):
      %102 = arith.extf %100 : bf16 to f32
      linalg.yield %102 : f32
    } -> tensor<2x3x8xf32>
    %103 = tensor.empty() : tensor<2x3x8xf32>
    %104 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%99 : tensor<2x3x8xf32>) outs(%103 : tensor<2x3x8xf32>) {
    ^bb14(%105: f32, %106: f32):
      %107 = arith.constant 5.000000e-01 : f32
      %108 = arith.constant 1.000000e+00 : f32
      %109 = arith.constant 4.471500e-02 : f32
      %110 = arith.constant 0.797884583 : f32
      %111 = arith.mulf %105, %105 : f32
      %112 = arith.mulf %111, %105 : f32
      %113 = arith.mulf %109, %112 : f32
      %114 = arith.addf %105, %113 : f32
      %115 = arith.mulf %110, %114 : f32
      %116 = math.tanh %115 : f32
      %117 = arith.addf %108, %116 : f32
      %118 = arith.mulf %107, %105 : f32
      %119 = arith.mulf %118, %117 : f32
      linalg.yield %119 : f32
    } -> tensor<2x3x8xf32>
    %120 = tensor.empty() : tensor<2x3x8xbf16>
    %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%104 : tensor<2x3x8xf32>) outs(%120 : tensor<2x3x8xbf16>) {
    ^bb15(%122: f32, %123: bf16):
      %124 = arith.truncf %122 : f32 to bf16
      linalg.yield %124 : bf16
    } -> tensor<2x3x8xbf16>
    %125 = tensor.empty() : tensor<2x3x8xf32>
    %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<2x3x8xbf16>) outs(%125 : tensor<2x3x8xf32>) {
    ^bb16(%127: bf16, %128: f32):
      %129 = arith.extf %127 : bf16 to f32
      linalg.yield %129 : f32
    } -> tensor<2x3x8xf32>
    %130 = tensor.empty() : tensor<2x3x8xf32>
    %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%126 : tensor<2x3x8xf32>) outs(%130 : tensor<2x3x8xf32>) {
    ^bb17(%132: f32, %133: f32):
      %134 = arith.constant 5.000000e-01 : f32
      %135 = arith.constant 1.000000e+00 : f32
      %136 = arith.constant 0.707106769 : f32
      %137 = arith.mulf %132, %136 : f32
      %138 = math.erf %137 : f32
      %139 = arith.addf %135, %138 : f32
      %140 = arith.mulf %134, %132 : f32
      %141 = arith.mulf %140, %139 : f32
      linalg.yield %141 : f32
    } -> tensor<2x3x8xf32>
    %142 = tensor.empty() : tensor<2x3x8xbf16>
    %143 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%131 : tensor<2x3x8xf32>) outs(%142 : tensor<2x3x8xbf16>) {
    ^bb18(%144: f32, %145: bf16):
      %146 = arith.truncf %144 : f32 to bf16
      linalg.yield %146 : bf16
    } -> tensor<2x3x8xbf16>
    func.return %121, %143 : tensor<2x3x8xbf16>, tensor<2x3x8xbf16>
  }
}