// RUN: iree-opt --split-input-file --verify-diagnostics %s --iree-plugin=npu --pass-pipeline='builtin.module(npu-verify-ukernel-symbols)'

module {
  func.func @valid_kernel(%lhs: tensor<64x32xf8E4M3FN>, %rhs: tensor<32x16xf8E4M3FN>) -> tensor<64x16xf32> {
    %0 = npu_kernel.ukernel_generic "npu_uk_matmul_f8E4M3FN_f8E4M3FN_f32"(%lhs, %rhs)
      : tensor<64x32xf8E4M3FN>, tensor<32x16xf8E4M3FN> -> tensor<64x16xf32>
    return %0 : tensor<64x16xf32>
  }
}

// -----

module {
  func.func @invalid_unknown_symbol(%lhs: tensor<64x32xf8E4M3FN>, %rhs: tensor<32x16xf8E4M3FN>) -> tensor<64x16xf32> {
    // expected-error @+1 {{unknown ukernel symbol family: 'npu_uk_unknown_family'}}
    %0 = npu_kernel.ukernel_generic "npu_uk_unknown_family"(%lhs, %rhs)
      : tensor<64x32xf8E4M3FN>, tensor<32x16xf8E4M3FN> -> tensor<64x16xf32>
    return %0 : tensor<64x16xf32>
  }
}
