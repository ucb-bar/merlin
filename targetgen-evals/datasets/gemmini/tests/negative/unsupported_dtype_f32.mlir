// RUN: xdsl-opt %s -p gemmini-verify 2>&1 | FileCheck %s
//
// FUTURE: Verifies that gemmini.pack with f32 input is rejected.
// Gemmini only supports i8/ui8 as the scratchpad element type.
//
// CHECK: error: gemmini.pack: unsupported element type 'f32'; expected i8 or ui8

func.func @unsupported_dtype_f32(
    %A: memref<16x16xf32>,
    %B: memref<16x16xf32>,
    %C: memref<16x16xf32>
) {
  // TODO: replace with actual dialect op that should be rejected.
  // %a_tile = gemmini.pack %A {...}  <- must fail: f32 not supported in scratchpad
  return
}
