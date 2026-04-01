// 3-layer MLP with constant weights: x → W1 → W2 → W3 → output
// All weights are compile-time constants, enabling weight pre-packing.
// With encoding propagation, intermediate tensors should stay packed.
func.func @main(%x: tensor<1024x1024xi8>) -> tensor<1024x1024xi32> {
  %w1 = arith.constant dense<2> : tensor<1024x1024xi8>
  %w2 = arith.constant dense<1> : tensor<1024x1024xi8>

  %c0 = arith.constant 0 : i32

  // Layer 1: x * W1
  %acc1_init = tensor.empty() : tensor<1024x1024xi32>
  %acc1 = linalg.fill ins(%c0 : i32) outs(%acc1_init : tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
  %y1 = linalg.matmul ins(%x, %w1 : tensor<1024x1024xi8>, tensor<1024x1024xi8>)
                       outs(%acc1 : tensor<1024x1024xi32>) -> tensor<1024x1024xi32>

  return %y1 : tensor<1024x1024xi32>
}
