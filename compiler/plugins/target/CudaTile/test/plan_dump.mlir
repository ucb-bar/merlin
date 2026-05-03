// RUN: iree-compile %s --iree-hal-target-backends=cuda_tile --iree-cuda-tile-enable-codegen=true --iree-cuda-tile-dump-kernel-plan-to=- --compile-to=executable-targets -o %t.mlir | FileCheck %s

func.func @matmul(%lhs: tensor<4x8xf32>, %rhs: tensor<8x4xf32>) -> tensor<4x4xf32> {
  %empty = tensor.empty() : tensor<4x4xf32>
  %zero = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%zero: f32) outs(%empty: tensor<4x4xf32>) -> tensor<4x4xf32>
  %result = linalg.matmul
      ins(%lhs, %rhs: tensor<4x8xf32>, tensor<8x4xf32>)
      outs(%fill: tensor<4x4xf32>)
      {lowering_config = #iree_gpu.lowering_config<{workgroup = [4, 4, 8]}>}
      -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

#map1 = affine_map<(d0) -> (d0)>

func.func @elementwise(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>) -> tensor<8xf32> {
  %empty = tensor.empty() : tensor<8xf32>
  %result = linalg.generic {
      indexing_maps = [#map1, #map1, #map1],
      iterator_types = ["parallel"]}
      ins(%lhs, %rhs: tensor<8xf32>, tensor<8xf32>)
      outs(%empty: tensor<8xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
  } -> tensor<8xf32>
  return %result : tensor<8xf32>
}

#reduce_in = affine_map<(d0, d1) -> (d0, d1)>
#reduce_out = affine_map<(d0, d1) -> (d0)>

func.func @reduce_sum(%input: tensor<4x8xf32>) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  %zero = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%zero: f32) outs(%empty: tensor<4xf32>) -> tensor<4xf32>
  %result = linalg.generic {
      indexing_maps = [#reduce_in, #reduce_out],
      iterator_types = ["parallel", "reduction"]}
      ins(%input: tensor<4x8xf32>)
      outs(%fill: tensor<4xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %sum = arith.addf %x, %acc : f32
      linalg.yield %sum : f32
  } -> tensor<4xf32>
  return %result : tensor<4xf32>
}

#conv2d_input = affine_map<(n, oh, ow, oc, kh, kw, ic) -> (n, oh + kh, ow + kw, ic)>
#conv2d_filter = affine_map<(n, oh, ow, oc, kh, kw, ic) -> (kh, kw, ic, oc)>
#conv2d_output = affine_map<(n, oh, ow, oc, kh, kw, ic) -> (n, oh, ow, oc)>

func.func @conv2d_direct(%input: tensor<1x5x6x2xf32>, %filter: tensor<3x2x2x4xf32>) -> tensor<1x3x5x4xf32> {
  %empty = tensor.empty() : tensor<1x3x5x4xf32>
  %zero = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%zero: f32) outs(%empty: tensor<1x3x5x4xf32>) -> tensor<1x3x5x4xf32>
  %result = linalg.generic {
      indexing_maps = [#conv2d_input, #conv2d_filter, #conv2d_output],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
      ins(%input, %filter: tensor<1x5x6x2xf32>, tensor<3x2x2x4xf32>)
      outs(%fill: tensor<1x3x5x4xf32>) {
    ^bb0(%x: f32, %w: f32, %acc: f32):
      %prod = arith.mulf %x, %w : f32
      %sum = arith.addf %prod, %acc : f32
      linalg.yield %sum : f32
  } -> tensor<1x3x5x4xf32>
  return %result : tensor<1x3x5x4xf32>
}

#conv1d_input = affine_map<(n, ow, oc, kw, ic) -> (n, ow + kw, ic)>
#conv1d_filter = affine_map<(n, ow, oc, kw, ic) -> (kw, ic, oc)>
#conv1d_output = affine_map<(n, ow, oc, kw, ic) -> (n, ow, oc)>

func.func @conv1d_unsupported(%input: tensor<1x8x2xf32>, %filter: tensor<3x2x4xf32>) -> tensor<1x6x4xf32> {
  %empty = tensor.empty() : tensor<1x6x4xf32>
  %zero = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%zero: f32) outs(%empty: tensor<1x6x4xf32>) -> tensor<1x6x4xf32>
  %result = linalg.generic {
      indexing_maps = [#conv1d_input, #conv1d_filter, #conv1d_output],
      iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]}
      ins(%input, %filter: tensor<1x8x2xf32>, tensor<3x2x4xf32>)
      outs(%fill: tensor<1x6x4xf32>) {
    ^bb0(%x: f32, %w: f32, %acc: f32):
      %prod = arith.mulf %x, %w : f32
      %sum = arith.addf %prod, %acc : f32
      linalg.yield %sum : f32
  } -> tensor<1x6x4xf32>
  return %result : tensor<1x6x4xf32>
}

#conv2d_pointwise_input = affine_map<(n, oh, ow, oc, kh, kw, ic) -> (n, oh, ow, ic)>
#conv2d_pointwise_filter = affine_map<(n, oh, ow, oc, kh, kw, ic) -> (kh, kw, ic, oc)>
#conv2d_pointwise_output = affine_map<(n, oh, ow, oc, kh, kw, ic) -> (n, oh, ow, oc)>

func.func @conv2d_pointwise(%input: tensor<1x3x5x2xf32>, %filter: tensor<1x1x2x4xf32>) -> tensor<1x3x5x4xf32> {
  %empty = tensor.empty() : tensor<1x3x5x4xf32>
  %zero = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%zero: f32) outs(%empty: tensor<1x3x5x4xf32>) -> tensor<1x3x5x4xf32>
  %result = linalg.generic {
      indexing_maps = [#conv2d_pointwise_input, #conv2d_pointwise_filter, #conv2d_pointwise_output],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
      ins(%input, %filter: tensor<1x3x5x2xf32>, tensor<1x1x2x4xf32>)
      outs(%fill: tensor<1x3x5x4xf32>) {
    ^bb0(%x: f32, %w: f32, %acc: f32):
      %prod = arith.mulf %x, %w : f32
      %sum = arith.addf %prod, %acc : f32
      linalg.yield %sum : f32
  } -> tensor<1x3x5x4xf32>
  return %result : tensor<1x3x5x4xf32>
}

// CHECK-LABEL: cuda_tile.kernel_plan {
// CHECK:   kernel_class = "elementwise"
// CHECK:   kind = elementwise
// CHECK:   semantic = map
// CHECK:   lowering_strategy = elementwise
// CHECK:   primary_op = "linalg.generic"
// CHECK:   op_counts = {map = 1, reduction = 0, contraction = 0, windowed_reduction = 0}

// CHECK-LABEL: cuda_tile.kernel_plan {
// CHECK:   kernel_class = "reduce"
// CHECK:   kind = reduction
// CHECK:   semantic = reduction
// CHECK:   lowering_strategy = reduction
// CHECK:   primary_op = "linalg.generic"
// CHECK:   op_counts = {map = 0, reduction = 1, contraction = 0, windowed_reduction = 0}
// CHECK:   loops = {parallel = [0], reduction = [1], tensor_reduction_dims = [1]}

// CHECK-LABEL: cuda_tile.kernel_plan {
// CHECK:   kernel_class = "matmul"
// CHECK:   semantic = contraction
// CHECK:   lowering_strategy = matmul
// CHECK:   primary_op = "linalg.generic"
// CHECK:   shapes = {src = [8, 2], dst = [6, 4], element_type = f32}
// CHECK:   contraction = {valid = true, m = 6, n = 4, k = 2
// CHECK-SAME: rhs_shape = [3, 2, 4]

// CHECK-LABEL: cuda_tile.kernel_plan {
// CHECK:   semantic = windowed_reduction
// CHECK:   lowering_strategy = direct_conv2d
// CHECK:   primary_op = "linalg.generic"
// CHECK:   conv = {mode = direct_conv2d
// CHECK-SAME: spatial_rank = 2
// CHECK-SAME: input_shape = [1, 5, 6, 2]
// CHECK-SAME: filter_shape = [3, 2, 2, 4]
// CHECK-SAME: output_shape = [1, 3, 5, 4]
// CHECK-SAME: strides = [1, 1]
// CHECK-SAME: dilations = [1, 1]

// CHECK-LABEL: cuda_tile.kernel_plan {
// CHECK:   kernel_class = "matmul"
// CHECK:   kind = matmul
// CHECK:   semantic = contraction
// CHECK:   primary_op = "linalg.matmul"
// CHECK:   op_counts = {map = 0, reduction = 0, contraction = 1, windowed_reduction = 0}
// CHECK:   shapes = {src = [4, 8], dst = [4, 4], element_type = f32}
// CHECK:   loops = {parallel = [0, 1], reduction = [2], tensor_reduction_dims = [1]}
// CHECK:   conv = {mode = not_conv
// CHECK:   contraction = {valid = true, m = 4, n = 4, k = 8
// CHECK-SAME: lhs_shape = [4, 8]
// CHECK-SAME: rhs_shape = [8, 4]
// CHECK-SAME: result_shape = [4, 4]
// CHECK-SAME: has_schedule_tiles = true
// CHECK-SAME: schedule_tiles = [4, 4, 8]
// CHECK-SAME: schedule_source = iree_gpu
// CHECK:   schedule = {source = iree_gpu
// CHECK-SAME: workgroup_tiles = [4, 4, 8]
// CHECK-SAME: has_lowering_config = true
// CHECK-SAME: has_iree_gpu_lowering_config = true

// CHECK-LABEL: cuda_tile.kernel_plan {
// CHECK:   kernel_class = "matmul"
// CHECK:   semantic = contraction
// CHECK:   lowering_strategy = matmul
// CHECK:   primary_op = "linalg.generic"
// CHECK:   shapes = {src = [15, 2], dst = [15, 4], element_type = f32}
// CHECK:   contraction = {valid = true, m = 15, n = 4, k = 2
// CHECK-SAME: rhs_shape = [2, 4]
