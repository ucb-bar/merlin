// RUN: iree-opt %s --iree-plugin=gemmini --pass-pipeline='builtin.module(func.func(gemmini-convert-to-gemmini))' | FileCheck %s

func.func @no_convert_fp8(%arg0: tensor<4xf8E4M3FNUZ>) -> tensor<4xf8E4M3FNUZ> {
  %empty = tensor.empty() : tensor<4xf8E4M3FNUZ>
  %0 = linalg.add
      ins(%arg0, %arg0 : tensor<4xf8E4M3FNUZ>, tensor<4xf8E4M3FNUZ>)
      outs(%empty : tensor<4xf8E4M3FNUZ>) -> tensor<4xf8E4M3FNUZ>
  return %0 : tensor<4xf8E4M3FNUZ>
}

// CHECK-LABEL: func.func @no_convert_fp8
// CHECK: linalg.add
// CHECK-NOT: gemmini.
