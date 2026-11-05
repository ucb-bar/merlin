builtin.module attributes {m2m.result_layout = "{\22authority\22: \22same_conversion_exported_program\22, \22inputs\22: [{\22dtype\22: \22float32\22, \22requires_grad\22: false, \22shape\22: [2, 3], \22storage_offset\22: 0, \22stride\22: [3, 1]}, {\22dtype\22: \22float32\22, \22requires_grad\22: false, \22shape\22: [2, 3], \22storage_offset\22: 0, \22stride\22: [3, 1]}, {\22dtype\22: \22float32\22, \22minimum_storage_elements\22: 2, \22requires_grad\22: false, \22shape\22: [2, 1], \22storage_offset\22: 0, \22stride\22: [1, 1]}, {\22dtype\22: \22float32\22, \22minimum_storage_elements\22: 2, \22requires_grad\22: false, \22shape\22: [2, 1], \22storage_offset\22: 0, \22stride\22: [1, 1]}, {\22dtype\22: \22float32\22, \22requires_grad\22: false, \22shape\22: [3], \22storage_offset\22: 0, \22stride\22: [1]}, {\22dtype\22: \22float32\22, \22requires_grad\22: false, \22shape\22: [3], \22storage_offset\22: 0, \22stride\22: [1]}], \22results\22: [{\22alias_inputs\22: [], \22alias_results\22: [], \22dtype\22: \22float32\22, \22requires_grad\22: false, \22role\22: \22user_output\22, \22shape\22: [2, 3], \22storage_offset\22: 0, \22stride\22: [3, 1]}, {\22alias_inputs\22: [], \22alias_results\22: [], \22dtype\22: \22float32\22, \22requires_grad\22: false, \22role\22: \22user_output\22, \22shape\22: [3], \22storage_offset\22: 0, \22stride\22: [1]}, {\22alias_inputs\22: [], \22alias_results\22: [], \22dtype\22: \22float32\22, \22requires_grad\22: false, \22role\22: \22user_output\22, \22shape\22: [3], \22storage_offset\22: 0, \22stride\22: [1]}], \22schema_version\22: 1}", prov.weights_file = "/scratch/loren/backend-verify/out-c1/captures/case-a477ed23df1048211a560012ecd6dc96613064d532dda9d7c50830faf3887e85/capture-work/weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<2x3xf32>, %1: tensor<2x3xf32>, %2: tensor<2x1xf32>, %3: tensor<2x1xf32>, %4: tensor<3xf32>, %5: tensor<3xf32>) -> (tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>) {
    %6 = tensor.collapse_shape %2 [[0 : i64, 1 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.source_node_ids = ["g:prepared:root:n6"], prov.origin_node_ids = ["g:original:root:n6", "g:quantized:root:n6"], prov.trace_role = "lowering", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2x1xf32> into tensor<2xf32>
    %7 = "tensor.extract_slice"(%6) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_0", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.source_node_ids = ["g:prepared:root:n7"], prov.origin_node_ids = ["g:original:root:n7", "g:quantized:root:n7"], prov.trace_role = "lowering", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<2xf32>) -> tensor<1xf32>
    %8 = tensor.collapse_shape %2 [[0 : i64, 1 : i64]] {prov.region_id = "view_1", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.source_node_ids = ["g:prepared:root:n8"], prov.origin_node_ids = ["g:original:root:n8", "g:quantized:root:n8"], prov.trace_role = "lowering", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2x1xf32> into tensor<2xf32>
    %9 = "tensor.extract_slice"(%8) <{static_offsets = array<i64: 1>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_1", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.source_node_ids = ["g:prepared:root:n9"], prov.origin_node_ids = ["g:original:root:n9", "g:quantized:root:n9"], prov.trace_role = "lowering", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<2xf32>) -> tensor<1xf32>
    %10 = tensor.concat dim(0) %7, %9 {prov.region_id = "cat_0", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.source_node_ids = ["g:prepared:root:n10"], prov.origin_node_ids = ["g:original:root:n10", "g:quantized:root:n10"], prov.trace_role = "lowering", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %11 = tensor.expand_shape %10 [[0 : i64, 1 : i64]] output_shape [2, 1] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.source_node_ids = ["g:prepared:root:n11"], prov.origin_node_ids = ["g:original:root:n11", "g:quantized:root:n11"], prov.trace_role = "lowering", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2xf32> into tensor<2x1xf32>
    %12 = tensor.collapse_shape %3 [[0 : i64, 1 : i64]] {prov.region_id = "view_3", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.source_node_ids = ["g:prepared:root:n12"], prov.origin_node_ids = ["g:original:root:n12", "g:quantized:root:n12"], prov.trace_role = "lowering", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2x1xf32> into tensor<2xf32>
    %13 = "tensor.extract_slice"(%12) <{static_offsets = array<i64: 0>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_2", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.source_node_ids = ["g:prepared:root:n13"], prov.origin_node_ids = ["g:original:root:n13", "g:quantized:root:n13"], prov.trace_role = "lowering", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<2xf32>) -> tensor<1xf32>
    %14 = tensor.collapse_shape %3 [[0 : i64, 1 : i64]] {prov.region_id = "view_4", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.source_node_ids = ["g:prepared:root:n14"], prov.origin_node_ids = ["g:original:root:n14", "g:quantized:root:n14"], prov.trace_role = "lowering", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2x1xf32> into tensor<2xf32>
    %15 = "tensor.extract_slice"(%14) <{static_offsets = array<i64: 1>, static_sizes = array<i64: 1>, static_strides = array<i64: 1>, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> {prov.region_id = "slice_3", prov.family = "layout", prov._pattern_hint = "slice", prov.op = "slice", prov.source_node_ids = ["g:prepared:root:n15"], prov.origin_node_ids = ["g:original:root:n15", "g:quantized:root:n15"], prov.trace_role = "lowering", prov.aten = "aten.slice.Tensor", prov.orig_dtype = "float32"} : (tensor<2xf32>) -> tensor<1xf32>
    %16 = tensor.concat dim(0) %13, %15 {prov.region_id = "cat_1", prov.family = "concat", prov._pattern_hint = "cat", prov.op = "cat", prov.source_node_ids = ["g:prepared:root:n16"], prov.origin_node_ids = ["g:original:root:n16", "g:quantized:root:n16"], prov.trace_role = "lowering", prov.aten = "aten.cat.default", prov.orig_dtype = "float32"} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %17 = tensor.expand_shape %16 [[0 : i64, 1 : i64]] output_shape [2, 1] {prov.region_id = "view_5", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.source_node_ids = ["g:prepared:root:n17"], prov.origin_node_ids = ["g:original:root:n17", "g:quantized:root:n17"], prov.trace_role = "lowering", prov.aten = "aten.view.default", prov.orig_dtype = "float32"} : tensor<2xf32> into tensor<2x1xf32>
    %18 = tensor.empty() : tensor<2x3xf32>
    %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1, %11 : tensor<2x3xf32>, tensor<2x1xf32>) outs(%18 : tensor<2x3xf32>) attrs =  {prov.region_id = "sub_0", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.sub.Tensor", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} {
    ^bb0(%20: f32, %21: f32, %22: f32):
      %23 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} -1 : i64
      %24 = arith.sitofp %23 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i64 to f32
      %25 = math.fma %21, %24, %20 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %26 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} -4194304 : i32
      %27 = arith.bitcast %26 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %28 = arith.cmpf uno, %25, %25 : f32
      %29 = arith.select %28, %27, %25 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %30 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 4194304 : i32
      %31 = arith.bitcast %20 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %32 = arith.ori %31, %30 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %33 = arith.bitcast %32 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %34 = arith.cmpf uno, %20, %20 : f32
      %35 = arith.select %34, %33, %29 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %36 = arith.bitcast %24 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %37 = arith.ori %36, %30 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %38 = arith.bitcast %37 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %39 = arith.cmpf uno, %24, %24 : f32
      %40 = arith.select %39, %38, %35 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %41 = arith.bitcast %21 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %42 = arith.ori %41, %30 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %43 = arith.bitcast %42 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %44 = arith.cmpf uno, %21, %21 : f32
      %45 = arith.select %44, %43, %40 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %46 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} true
      %47 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} -4194304 : i32
      %48 = arith.bitcast %47 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %49 = arith.cmpf uno, %25, %25 : f32
      %50 = arith.select %49, %48, %25 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %51 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 4194304 : i32
      %52 = arith.bitcast %20 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %53 = arith.ori %52, %51 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %54 = arith.bitcast %53 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %55 = arith.cmpf uno, %20, %20 : f32
      %56 = arith.select %55, %54, %50 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %57 = arith.bitcast %21 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %58 = arith.ori %57, %51 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %59 = arith.bitcast %58 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %60 = arith.cmpf uno, %21, %21 : f32
      %61 = arith.select %60, %59, %56 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %62 = arith.bitcast %24 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %63 = arith.ori %62, %51 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %64 = arith.bitcast %63 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %65 = arith.cmpf uno, %24, %24 : f32
      %66 = arith.select %65, %64, %61 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %67 = arith.select %46, %45, %66 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %68 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 0 : index
      %69 = linalg.index 1 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %70 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 1 : index
      %71 = arith.muli %69, %70 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %72 = arith.addi %68, %71 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %73 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 1 : index
      %74 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 1 : index
      %75 = arith.cmpi uge, %72, %73 : index
      %76 = arith.cmpi ult, %72, %74 : index
      %77 = arith.andi %75, %76 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i1
      %78 = arith.mulf %21, %24 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %79 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} -4194304 : i32
      %80 = arith.bitcast %79 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %81 = arith.cmpf uno, %78, %78 : f32
      %82 = arith.select %81, %80, %78 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %83 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 4194304 : i32
      %84 = arith.bitcast %24 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %85 = arith.ori %84, %83 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %86 = arith.bitcast %85 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %87 = arith.cmpf uno, %24, %24 : f32
      %88 = arith.select %87, %86, %82 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %89 = arith.bitcast %21 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %90 = arith.ori %89, %83 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %91 = arith.bitcast %90 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %92 = arith.cmpf uno, %21, %21 : f32
      %93 = arith.select %92, %91, %88 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %94 = arith.addf %93, %20 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %95 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} -4194304 : i32
      %96 = arith.bitcast %95 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %97 = arith.cmpf uno, %94, %94 : f32
      %98 = arith.select %97, %96, %94 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %99 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 4194304 : i32
      %100 = arith.bitcast %20 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %101 = arith.ori %100, %99 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %102 = arith.bitcast %101 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %103 = arith.cmpf uno, %20, %20 : f32
      %104 = arith.select %103, %102, %98 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %105 = arith.bitcast %93 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %106 = arith.ori %105, %99 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %107 = arith.bitcast %106 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %108 = arith.cmpf uno, %93, %93 : f32
      %109 = arith.select %108, %107, %104 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %110 = arith.select %77, %109, %67 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %110 : f32
    } -> tensor<2x3xf32>
    %111 = tensor.empty() : tensor<2x3xf32>
    %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %17 : tensor<2x3xf32>, tensor<2x1xf32>) outs(%111 : tensor<2x3xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.mul.Tensor", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} {
    ^bb1(%113: f32, %114: f32, %115: f32):
      %116 = arith.mulf %113, %114 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %116 : f32
    } -> tensor<2x3xf32>
    %117 = tensor.empty() : tensor<2x3xf32>
    %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0, %4 : tensor<2x3xf32>, tensor<3xf32>) outs(%117 : tensor<2x3xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.mul.Tensor", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} {
    ^bb2(%119: f32, %120: f32, %121: f32):
      %122 = arith.mulf %119, %120 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %122 : f32
    } -> tensor<2x3xf32>
    %123 = arith.constant {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.mul.Tensor", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 3.000000e+00 : f32
    %124 = tensor.splat %123 {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.mul.Tensor", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : tensor<2x3xf32>
    %125 = tensor.empty() : tensor<2x3xf32>
    %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%118, %124 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%125 : tensor<2x3xf32>) attrs =  {prov.region_id = "mul_2", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.mul.Tensor", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} {
    ^bb3(%127: f32, %128: f32, %129: f32):
      %130 = arith.mulf %127, %128 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %130 : f32
    } -> tensor<2x3xf32>
    %131 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.sum.dim_IntList", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 0.000000e+00 : f32
    %132 = tensor.splat %131 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.sum.dim_IntList", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : tensor<2xf32>
    %133 = linalg.reduce ins(%118:tensor<2x3xf32>) outs(%132:tensor<2xf32>) dimensions = [1]
    (%134: f32, %135: f32) {
      %136 = arith.addf %134, %135 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %136 : f32
    }
    %137 = tensor.empty() : tensor<2xf32>
    %138 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%133 : tensor<2xf32>) outs(%137 : tensor<2xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.sum.dim_IntList", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} {
    ^bb4(%139: f32, %140: f32):
      %141 = arith.bitcast %139 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %142 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 4194304 : i32
      %143 = arith.ori %141, %142 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %144 = arith.bitcast %143 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %145 = arith.cmpf uno, %139, %139 : f32
      %146 = arith.select %145, %144, %139 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %146 : f32
    } -> tensor<2xf32>
    %147 = tensor.expand_shape %138 [[0 : i64, 1 : i64]] output_shape [2, 1] {prov.region_id = "reduce_0", prov.family = "layout", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.sum.dim_IntList", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : tensor<2xf32> into tensor<2x1xf32>
    %148 = tensor.empty() : tensor<2x3xf32>
    %149 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%118, %112 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%148 : tensor<2x3xf32>) attrs =  {prov.region_id = "mul_3", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.mul.Tensor", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} {
    ^bb5(%150: f32, %151: f32, %152: f32):
      %153 = arith.mulf %150, %151 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %153 : f32
    } -> tensor<2x3xf32>
    %154 = arith.constant {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.sum.dim_IntList", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 0.000000e+00 : f32
    %155 = tensor.splat %154 {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.sum.dim_IntList", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : tensor<2xf32>
    %156 = linalg.reduce ins(%149:tensor<2x3xf32>) outs(%155:tensor<2xf32>) dimensions = [1]
    (%157: f32, %158: f32) {
      %159 = arith.addf %157, %158 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %159 : f32
    }
    %160 = tensor.empty() : tensor<2xf32>
    %161 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%156 : tensor<2xf32>) outs(%160 : tensor<2xf32>) attrs =  {prov.region_id = "reduce_1", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.sum.dim_IntList", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} {
    ^bb6(%162: f32, %163: f32):
      %164 = arith.bitcast %162 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %165 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 4194304 : i32
      %166 = arith.ori %164, %165 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %167 = arith.bitcast %166 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %168 = arith.cmpf uno, %162, %162 : f32
      %169 = arith.select %168, %167, %162 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %169 : f32
    } -> tensor<2xf32>
    %170 = tensor.expand_shape %161 [[0 : i64, 1 : i64]] output_shape [2, 1] {prov.region_id = "reduce_1", prov.family = "layout", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.sum.dim_IntList", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : tensor<2xf32> into tensor<2x1xf32>
    %171 = tensor.empty() : tensor<2x3xf32>
    %172 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%112, %170 : tensor<2x3xf32>, tensor<2x1xf32>) outs(%171 : tensor<2x3xf32>) attrs =  {prov.region_id = "mul_4", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.mul.Tensor", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} {
    ^bb7(%173: f32, %174: f32, %175: f32):
      %176 = arith.mulf %173, %174 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %176 : f32
    } -> tensor<2x3xf32>
    %177 = tensor.empty() : tensor<2x3xf32>
    %178 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%126, %147 : tensor<2x3xf32>, tensor<2x1xf32>) outs(%177 : tensor<2x3xf32>) attrs =  {prov.region_id = "sub_1", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.sub.Tensor", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} {
    ^bb8(%179: f32, %180: f32, %181: f32):
      %182 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} -1 : i64
      %183 = arith.sitofp %182 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i64 to f32
      %184 = math.fma %180, %183, %179 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %185 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} -4194304 : i32
      %186 = arith.bitcast %185 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %187 = arith.cmpf uno, %184, %184 : f32
      %188 = arith.select %187, %186, %184 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %189 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 4194304 : i32
      %190 = arith.bitcast %179 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %191 = arith.ori %190, %189 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %192 = arith.bitcast %191 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %193 = arith.cmpf uno, %179, %179 : f32
      %194 = arith.select %193, %192, %188 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %195 = arith.bitcast %183 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %196 = arith.ori %195, %189 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %197 = arith.bitcast %196 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %198 = arith.cmpf uno, %183, %183 : f32
      %199 = arith.select %198, %197, %194 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %200 = arith.bitcast %180 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %201 = arith.ori %200, %189 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %202 = arith.bitcast %201 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %203 = arith.cmpf uno, %180, %180 : f32
      %204 = arith.select %203, %202, %199 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %205 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} true
      %206 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} -4194304 : i32
      %207 = arith.bitcast %206 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %208 = arith.cmpf uno, %184, %184 : f32
      %209 = arith.select %208, %207, %184 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %210 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 4194304 : i32
      %211 = arith.bitcast %179 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %212 = arith.ori %211, %210 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %213 = arith.bitcast %212 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %214 = arith.cmpf uno, %179, %179 : f32
      %215 = arith.select %214, %213, %209 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %216 = arith.bitcast %180 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %217 = arith.ori %216, %210 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %218 = arith.bitcast %217 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %219 = arith.cmpf uno, %180, %180 : f32
      %220 = arith.select %219, %218, %215 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %221 = arith.bitcast %183 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %222 = arith.ori %221, %210 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %223 = arith.bitcast %222 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %224 = arith.cmpf uno, %183, %183 : f32
      %225 = arith.select %224, %223, %220 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %226 = arith.select %205, %204, %225 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %227 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 0 : index
      %228 = linalg.index 1 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %229 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 1 : index
      %230 = arith.muli %228, %229 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %231 = arith.addi %227, %230 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %232 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 1 : index
      %233 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 1 : index
      %234 = arith.cmpi uge, %231, %232 : index
      %235 = arith.cmpi ult, %231, %233 : index
      %236 = arith.andi %234, %235 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i1
      %237 = arith.mulf %180, %183 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %238 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} -4194304 : i32
      %239 = arith.bitcast %238 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %240 = arith.cmpf uno, %237, %237 : f32
      %241 = arith.select %240, %239, %237 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %242 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 4194304 : i32
      %243 = arith.bitcast %183 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %244 = arith.ori %243, %242 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %245 = arith.bitcast %244 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %246 = arith.cmpf uno, %183, %183 : f32
      %247 = arith.select %246, %245, %241 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %248 = arith.bitcast %180 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %249 = arith.ori %248, %242 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %250 = arith.bitcast %249 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %251 = arith.cmpf uno, %180, %180 : f32
      %252 = arith.select %251, %250, %247 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %253 = arith.addf %252, %179 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %254 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} -4194304 : i32
      %255 = arith.bitcast %254 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %256 = arith.cmpf uno, %253, %253 : f32
      %257 = arith.select %256, %255, %253 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %258 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 4194304 : i32
      %259 = arith.bitcast %179 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %260 = arith.ori %259, %258 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %261 = arith.bitcast %260 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %262 = arith.cmpf uno, %179, %179 : f32
      %263 = arith.select %262, %261, %257 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %264 = arith.bitcast %252 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %265 = arith.ori %264, %258 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %266 = arith.bitcast %265 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %267 = arith.cmpf uno, %252, %252 : f32
      %268 = arith.select %267, %266, %263 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %269 = arith.select %236, %268, %226 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %269 : f32
    } -> tensor<2x3xf32>
    %270 = tensor.empty() : tensor<2x3xf32>
    %271 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%178, %172 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%270 : tensor<2x3xf32>) attrs =  {prov.region_id = "sub_2", prov._pattern_hint = "sub", prov.op = "sub", prov.family = "elementwise", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.sub.Tensor", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} {
    ^bb9(%272: f32, %273: f32, %274: f32):
      %275 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} -1 : i64
      %276 = arith.sitofp %275 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i64 to f32
      %277 = math.fma %273, %276, %272 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %278 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} -4194304 : i32
      %279 = arith.bitcast %278 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %280 = arith.cmpf uno, %277, %277 : f32
      %281 = arith.select %280, %279, %277 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %282 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 4194304 : i32
      %283 = arith.bitcast %272 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %284 = arith.ori %283, %282 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %285 = arith.bitcast %284 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %286 = arith.cmpf uno, %272, %272 : f32
      %287 = arith.select %286, %285, %281 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %288 = arith.bitcast %276 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %289 = arith.ori %288, %282 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %290 = arith.bitcast %289 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %291 = arith.cmpf uno, %276, %276 : f32
      %292 = arith.select %291, %290, %287 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %293 = arith.bitcast %273 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %294 = arith.ori %293, %282 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %295 = arith.bitcast %294 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %296 = arith.cmpf uno, %273, %273 : f32
      %297 = arith.select %296, %295, %292 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %298 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 0 : index
      %299 = linalg.index 1 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %300 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 1 : index
      %301 = arith.muli %299, %300 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %302 = arith.addi %298, %301 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %303 = linalg.index 0 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %304 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 3 : index
      %305 = arith.muli %303, %304 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %306 = arith.addi %302, %305 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %307 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 0 : index
      %308 = arith.cmpi ult, %306, %307 : index
      %309 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 0 : index
      %310 = linalg.index 1 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %311 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 1 : index
      %312 = arith.muli %310, %311 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %313 = arith.addi %309, %312 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %314 = linalg.index 0 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %315 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 3 : index
      %316 = arith.muli %314, %315 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %317 = arith.addi %313, %316 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %318 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 4 : index
      %319 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 6 : index
      %320 = arith.cmpi uge, %317, %318 : index
      %321 = arith.cmpi ult, %317, %319 : index
      %322 = arith.andi %320, %321 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i1
      %323 = arith.ori %308, %322 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i1
      %324 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} -4194304 : i32
      %325 = arith.bitcast %324 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %326 = arith.cmpf uno, %277, %277 : f32
      %327 = arith.select %326, %325, %277 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %328 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 4194304 : i32
      %329 = arith.bitcast %272 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %330 = arith.ori %329, %328 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %331 = arith.bitcast %330 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %332 = arith.cmpf uno, %272, %272 : f32
      %333 = arith.select %332, %331, %327 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %334 = arith.bitcast %273 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %335 = arith.ori %334, %328 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %336 = arith.bitcast %335 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %337 = arith.cmpf uno, %273, %273 : f32
      %338 = arith.select %337, %336, %333 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %339 = arith.bitcast %276 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %340 = arith.ori %339, %328 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %341 = arith.bitcast %340 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %342 = arith.cmpf uno, %276, %276 : f32
      %343 = arith.select %342, %341, %338 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %344 = arith.select %323, %297, %343 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %345 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 0 : index
      %346 = linalg.index 1 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %347 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 1 : index
      %348 = arith.muli %346, %347 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %349 = arith.addi %345, %348 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %350 = linalg.index 0 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %351 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 3 : index
      %352 = arith.muli %350, %351 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %353 = arith.addi %349, %352 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %354 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 1 : index
      %355 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 1 : index
      %356 = arith.cmpi uge, %353, %354 : index
      %357 = arith.cmpi ult, %353, %355 : index
      %358 = arith.andi %356, %357 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i1
      %359 = arith.mulf %273, %276 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %360 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} -4194304 : i32
      %361 = arith.bitcast %360 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %362 = arith.cmpf uno, %359, %359 : f32
      %363 = arith.select %362, %361, %359 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %364 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 4194304 : i32
      %365 = arith.bitcast %276 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %366 = arith.ori %365, %364 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %367 = arith.bitcast %366 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %368 = arith.cmpf uno, %276, %276 : f32
      %369 = arith.select %368, %367, %363 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %370 = arith.bitcast %273 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %371 = arith.ori %370, %364 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %372 = arith.bitcast %371 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %373 = arith.cmpf uno, %273, %273 : f32
      %374 = arith.select %373, %372, %369 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %375 = arith.addf %374, %272 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %376 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} -4194304 : i32
      %377 = arith.bitcast %376 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %378 = arith.cmpf uno, %375, %375 : f32
      %379 = arith.select %378, %377, %375 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %380 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 4194304 : i32
      %381 = arith.bitcast %272 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %382 = arith.ori %381, %380 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %383 = arith.bitcast %382 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %384 = arith.cmpf uno, %272, %272 : f32
      %385 = arith.select %384, %383, %379 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %386 = arith.bitcast %374 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %387 = arith.ori %386, %380 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %388 = arith.bitcast %387 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %389 = arith.cmpf uno, %374, %374 : f32
      %390 = arith.select %389, %388, %385 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      %391 = arith.select %358, %390, %344 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %391 : f32
    } -> tensor<2x3xf32>
    %392 = arith.constant {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.div.Tensor", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 3.000000e+00 : f32
    %393 = tensor.splat %392 {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.div.Tensor", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : tensor<2x1xf32>
    %394 = tensor.empty() : tensor<2x1xf32>
    %395 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %393 : tensor<2x1xf32>, tensor<2x1xf32>) outs(%394 : tensor<2x1xf32>) attrs =  {prov.region_id = "div_0", prov._pattern_hint = "div", prov.op = "div", prov.family = "elementwise", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.div.Tensor", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} {
    ^bb10(%396: f32, %397: f32, %398: f32):
      %399 = arith.divf %396, %397 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %399 : f32
    } -> tensor<2x1xf32>
    %400 = tensor.empty() : tensor<2x3xf32>
    %401 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%395, %271 : tensor<2x1xf32>, tensor<2x3xf32>) outs(%400 : tensor<2x3xf32>) attrs =  {prov.region_id = "mul_5", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.mul.Tensor", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} {
    ^bb11(%402: f32, %403: f32, %404: f32):
      %405 = arith.mulf %402, %403 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %405 : f32
    } -> tensor<2x3xf32>
    %406 = arith.constant {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.sum.dim_IntList", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 0.000000e+00 : f32
    %407 = tensor.splat %406 {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.sum.dim_IntList", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : tensor<3xf32>
    %408 = linalg.reduce ins(%0:tensor<2x3xf32>) outs(%407:tensor<3xf32>) dimensions = [0]
    (%409: f32, %410: f32) {
      %411 = arith.addf %409, %410 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %411 : f32
    }
    %412 = tensor.empty() : tensor<3xf32>
    %413 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%408 : tensor<3xf32>) outs(%412 : tensor<3xf32>) attrs =  {prov.region_id = "reduce_2", prov.family = "reduce", prov._pattern_hint = "reduce_sum", prov.op = "reduce_sum", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32", prov.lowered_aten = "aten.sum.dim_IntList", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} {
    ^bb12(%414: f32, %415: f32):
      %416 = arith.bitcast %414 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32 to i32
      %417 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 4194304 : i32
      %418 = arith.ori %416, %417 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32
      %419 = arith.bitcast %418 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : i32 to f32
      %420 = arith.cmpf uno, %414, %414 : f32
      %421 = arith.select %420, %419, %414 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %421 : f32
    } -> tensor<3xf32>
    %422 = tensor.empty() : tensor<3xf32>
    %423 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%422 : tensor<3xf32>) attrs =  {prov._pattern_hint = "composite", prov.op = "composite", prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering", prov.aten = "aten.native_layer_norm_backward.default", prov.orig_dtype = "float32"} {
    ^bb13(%424: f32):
      %425 = linalg.index 0 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
      %426 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 0.000000e+00 : f32
      %427 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 0 : index
      %428 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 2 : index
      %429 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 1 : index
      %430 = scf.for %431 = %427 to %428 step %429 iter_args(%432 = %426) -> (f32) {
        %433 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 2 : index
        %434 = arith.remui %431, %433 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
        %435 = tensor.extract %1[%434, %425] : tensor<2x3xf32>
        %436 = tensor.extract %0[%434, %425] : tensor<2x3xf32>
        %437 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 1 : index
        %438 = arith.remui %431, %437 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
        %439 = arith.divui %431, %437 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
        %440 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 2 : index
        %441 = arith.remui %439, %440 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
        %442 = tensor.extract %17[%441, %438] : tensor<2x1xf32>
        %443 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 1 : index
        %444 = arith.remui %431, %443 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
        %445 = arith.divui %431, %443 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
        %446 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 2 : index
        %447 = arith.remui %445, %446 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : index
        %448 = tensor.extract %11[%447, %444] : tensor<2x1xf32>
        %449 = arith.negf %442 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
        %450 = arith.mulf %449, %448 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
        %451 = math.fma %442, %435, %450 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
        %452 = math.fma %436, %451, %432 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
        scf.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %452 : f32
      }
      %453 = arith.constant {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} 0.000000e+00 : f32
      %454 = arith.addf %453, %430 {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} : f32
      linalg.yield {prov.source_node_ids = ["g:prepared:root:n18"], prov.origin_node_ids = ["g:original:root:n18", "g:quantized:root:n18"], prov.trace_role = "lowering"} %454 : f32
    } -> tensor<3xf32>
    func.return %401, %423, %413 : tensor<2x3xf32>, tensor<3xf32>, tensor<3xf32>
  }
}
