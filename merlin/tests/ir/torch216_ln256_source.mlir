builtin.module attributes {prov.quantization = "int8_dyn_act_int8_weight"} {
  func.func @forward(%0: tensor<1x196x256xf32>, %1: tensor<256xf32>, %2: tensor<256xf32>) -> tensor<1x196x256xf32> {
    %3 = arith.constant {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} 0.000000e+00 : f32
    %4 = tensor.splat %3 {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} : tensor<1x196xf32>
    %5 = "linalg.reduce"(%0, %4) <{dimensions = array<i64: 2>}> ({
    ^bb0(%6: f32, %7: f32):
      %8 = "arith.addf"(%6, %7) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%8) : (f32) -> ()
    }) {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} : (tensor<1x196x256xf32>, tensor<1x196xf32>) -> tensor<1x196xf32>
    %9 = arith.constant {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} 2.560000e+02 : f32
    %10 = tensor.splat %9 {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} : tensor<1x196xf32>
    %11 = tensor.empty() : tensor<1x196xf32>
    %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %10 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%11 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} {
    ^bb0(%13: f32, %14: f32, %15: f32):
      %16 = arith.divf %13, %14 : f32
      linalg.yield %16 : f32
    } -> tensor<1x196xf32>
    %17 = tensor.collapse_shape %12 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} : tensor<1x196xf32> into tensor<196xf32>
    %18 = tensor.expand_shape %17 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} : tensor<196xf32> into tensor<1x196x1xf32>
    %19 = tensor.empty() : tensor<1x196x256xf32>
    %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %18 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%19 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} {
    ^bb0(%21: f32, %22: f32, %23: f32):
      %24 = arith.subf %21, %22 : f32
      linalg.yield %24 : f32
    } -> tensor<1x196x256xf32>
    %25 = tensor.empty() : tensor<1x196x256xf32>
    %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20, %20 : tensor<1x196x256xf32>, tensor<1x196x256xf32>) outs(%25 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} {
    ^bb0(%27: f32, %28: f32, %29: f32):
      %30 = arith.mulf %27, %28 : f32
      linalg.yield %30 : f32
    } -> tensor<1x196x256xf32>
    %31 = arith.constant {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} 0.000000e+00 : f32
    %32 = tensor.splat %31 {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} : tensor<1x196xf32>
    %33 = "linalg.reduce"(%26, %32) <{dimensions = array<i64: 2>}> ({
    ^bb0(%34: f32, %35: f32):
      %36 = "arith.addf"(%34, %35) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%36) : (f32) -> ()
    }) {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} : (tensor<1x196x256xf32>, tensor<1x196xf32>) -> tensor<1x196xf32>
    %37 = arith.constant {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} 2.560000e+02 : f32
    %38 = tensor.splat %37 {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} : tensor<1x196xf32>
    %39 = tensor.empty() : tensor<1x196xf32>
    %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %38 : tensor<1x196xf32>, tensor<1x196xf32>) outs(%39 : tensor<1x196xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} {
    ^bb0(%41: f32, %42: f32, %43: f32):
      %44 = arith.divf %41, %42 : f32
      linalg.yield %44 : f32
    } -> tensor<1x196xf32>
    %45 = tensor.collapse_shape %40 [[0 : i64, 1 : i64]] {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} : tensor<1x196xf32> into tensor<196xf32>
    %46 = tensor.expand_shape %45 [[0 : i64, 1 : i64, 2 : i64]] output_shape [1, 196, 1] {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} : tensor<196xf32> into tensor<1x196x1xf32>
    %47 = arith.constant {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} 1.000000e-06 : f32
    %48 = tensor.splat %47 {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} : tensor<1x196x1xf32>
    %49 = tensor.empty() : tensor<1x196x1xf32>
    %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46, %48 : tensor<1x196x1xf32>, tensor<1x196x1xf32>) outs(%49 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} {
    ^bb0(%51: f32, %52: f32, %53: f32):
      %54 = arith.addf %51, %52 : f32
      linalg.yield %54 : f32
    } -> tensor<1x196x1xf32>
    %55 = tensor.empty() : tensor<1x196x1xf32>
    %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50 : tensor<1x196x1xf32>) outs(%55 : tensor<1x196x1xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} {
    ^bb0(%57: f32, %58: f32):
      %59 = math.rsqrt %57 : f32
      linalg.yield %59 : f32
    } -> tensor<1x196x1xf32>
    %60 = tensor.empty() : tensor<1x196x256xf32>
    %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20, %56 : tensor<1x196x256xf32>, tensor<1x196x1xf32>) outs(%60 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} {
    ^bb0(%62: f32, %63: f32, %64: f32):
      %65 = arith.mulf %62, %63 : f32
      linalg.yield %65 : f32
    } -> tensor<1x196x256xf32>
    %66 = tensor.empty() : tensor<1x196x256xf32>
    %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %1 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%66 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} {
    ^bb0(%68: f32, %69: f32, %70: f32):
      %71 = arith.mulf %68, %69 : f32
      linalg.yield %71 : f32
    } -> tensor<1x196x256xf32>
    %72 = tensor.empty() : tensor<1x196x256xf32>
    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%67, %2 : tensor<1x196x256xf32>, tensor<256xf32>) outs(%72 : tensor<1x196x256xf32>) attrs =  {prov.region_id = "layer_norm_0", prov.aten = "aten.native_layer_norm.default", prov.op = "layer_norm"} {
    ^bb0(%74: f32, %75: f32, %76: f32):
      %77 = arith.addf %74, %75 : f32
      linalg.yield %77 : f32
    } -> tensor<1x196x256xf32>
    func.return %73 : tensor<1x196x256xf32>
  }
}