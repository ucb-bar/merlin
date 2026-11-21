// ToyNPU example: matmul + fused epilogue (illustrative, not yet compilable).
//
//   %acc = toynpu.matmul %A, %w
//        : tensor<64x128xi8>, !toynpu.resident_tensor<128x64xi8>
//        -> !toynpu.accumulator<64x64xi32>
//   %y = toynpu.commit %acc {epilogue = ["bias", "requant", "relu"]}
//        : !toynpu.accumulator<64x64xi32> -> tensor<64x64xi8>
