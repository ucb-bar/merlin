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

// CHECK: cuda_tile.kernel_plan {
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
// CHECK:   operands = [
