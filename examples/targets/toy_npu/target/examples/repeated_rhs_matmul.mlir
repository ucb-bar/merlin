// ToyNPU example: repeated RHS matmul (illustrative, not yet compilable).
// for i in 1..R: Y_i = A_i @ W   ; W immutable, resident.
//
// Sketch of intended lowering once the toynpu dialect exists:
//
//   %w = toynpu.res_pack %W : tensor<128x64xi8> -> !toynpu.resident_tensor<128x64xi8>
//   scf.for %i = %c0 to %R step %c1 {
//     %acc = toynpu.matmul %A_i, %w
//          : tensor<64x128xi8>, !toynpu.resident_tensor<128x64xi8>
//          -> !toynpu.accumulator<64x64xi32>
//     %y = toynpu.commit %acc : !toynpu.accumulator<64x64xi32> -> tensor<64x64xi8>
//   }
//   toynpu.evict %w : !toynpu.resident_tensor<128x64xi8>
