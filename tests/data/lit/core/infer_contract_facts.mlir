
// RUN: %merlin-opt %s -p merlin-infer-contract-facts | %filecheck %s
//
// OBLIGATION partition/eligibility -- the pass must emit the capability, the immutability
// assumption, the requirement, and a proof token for it. Checked on the `contract` dialect, whose
// namespace is fixed in-tree; never on a generated target dialect.

// CHECK: contract.capability
// CHECK: contract.assume
// CHECK-SAME: immutable
// CHECK: contract.require
// CHECK: contract.prove
builtin.module {
  func.func @repeated_rhs_matmul(%0: tensor<64x128xi8>, %1: tensor<64x128xi8>, %2: tensor<64x128xi8>, %3: tensor<64x128xi8>, %4: tensor<128x64xi8>) -> (tensor<64x64xi32>, tensor<64x64xi32>, tensor<64x64xi32>, tensor<64x64xi32>) {
    %5 = arith.constant 0 : i32
    %6 = tensor.empty() : tensor<64x64xi32>
    %7 = linalg.quantized_matmul ins(%0, %4, %5, %5 : tensor<64x128xi8>, tensor<128x64xi8>, i32, i32) outs(%6 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %8 = tensor.empty() : tensor<64x64xi32>
    %9 = linalg.quantized_matmul ins(%1, %4, %5, %5 : tensor<64x128xi8>, tensor<128x64xi8>, i32, i32) outs(%8 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %10 = tensor.empty() : tensor<64x64xi32>
    %11 = linalg.quantized_matmul ins(%2, %4, %5, %5 : tensor<64x128xi8>, tensor<128x64xi8>, i32, i32) outs(%10 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %12 = tensor.empty() : tensor<64x64xi32>
    %13 = linalg.quantized_matmul ins(%3, %4, %5, %5 : tensor<64x128xi8>, tensor<128x64xi8>, i32, i32) outs(%12 : tensor<64x64xi32>) -> tensor<64x64xi32>
    func.return %7, %9, %11, %13 : tensor<64x64xi32>, tensor<64x64xi32>, tensor<64x64xi32>, tensor<64x64xi32>
  }
}
