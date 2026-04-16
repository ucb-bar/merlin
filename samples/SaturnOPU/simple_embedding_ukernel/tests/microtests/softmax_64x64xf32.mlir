// Phase D6.5: isolate linalg.softmax (attention softmax).
// vit d7 and d19 use linalg.softmax. Lowering uses max-reduction +
// exp + div; combines vfredmax.vs (scalarized) + vfdiv + math.exp.
// Separate hang candidate from raw rsqrt.
func.func @main(%x: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %init = tensor.empty() : tensor<64x64xf32>
  %0 = linalg.softmax dimension(1)
       ins(%x : tensor<64x64xf32>) outs(%init : tensor<64x64xf32>)
       -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}
