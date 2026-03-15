// RUN: iree-compile %s --iree-input-type=none --iree-hal-target-backends=llvm-cpu --compile-to=global-optimization --iree-plugin=npu --iree-npu-enable | FileCheck %s

func.func @main(%lhs: tensor<16x1024xf8E4M3FN>, %rhs: tensor<1024x128xf8E4M3FN>) -> tensor<16x128xf32> {
  %c0_f32 = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<16x128xf32>
  %init = linalg.fill ins(%c0_f32 : f32) outs(%empty : tensor<16x128xf32>) -> tensor<16x128xf32>
  %0 = linalg.matmul
      ins(%lhs, %rhs : tensor<16x1024xf8E4M3FN>, tensor<1024x128xf8E4M3FN>)
      outs(%init : tensor<16x128xf32>)
    -> tensor<16x128xf32>
  return %0 : tensor<16x128xf32>
}

// CHECK-LABEL: util.func public @main
// CHECK: npu_isa.matmul_mxu0
