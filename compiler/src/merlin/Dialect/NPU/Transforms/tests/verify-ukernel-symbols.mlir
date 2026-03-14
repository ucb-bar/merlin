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
  func.func @valid_batched_kernel(%lhs: tensor<2x64x32xbf16>, %rhs: tensor<2x32x16xbf16>) -> tensor<2x64x16xbf16> {
    %0 = npu_kernel.ukernel_generic "npu_uk_matmul_bf16_bf16_bf16"(%lhs, %rhs)
      : tensor<2x64x32xbf16>, tensor<2x32x16xbf16> -> tensor<2x64x16xbf16>
    return %0 : tensor<2x64x16xbf16>
  }
}

// -----

module {
  func.func @valid_attention_symbol(%q: tensor<1x2x4xbf16>, %k: tensor<1x2x4xbf16>, %v: tensor<1x2x4xbf16>) -> tensor<1x2x4xbf16> {
    %0 = npu_kernel.ukernel_generic "npu_uk_gemma_attention_bf16_bf16"(%q, %k, %v)
      : tensor<1x2x4xbf16>, tensor<1x2x4xbf16>, tensor<1x2x4xbf16> -> tensor<1x2x4xbf16>
    return %0 : tensor<1x2x4xbf16>
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
