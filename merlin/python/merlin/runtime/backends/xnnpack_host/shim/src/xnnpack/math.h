// Minimal XNNPACK math.h shim providing the scalar helpers the f32-gemm scalar
// microkernel references (muladd / min / max). NOT the real XNNPACK math.h.
#ifndef MERLIN_XNNHOST_MATH_H
#define MERLIN_XNNHOST_MATH_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"

XNN_INLINE static float math_muladd_f32(float x, float y, float acc) {
  return x * y + acc;
}
XNN_INLINE static float math_min_f32(float a, float b) {
  return a < b ? a : b;
}
XNN_INLINE static float math_max_f32(float a, float b) {
  return a > b ? a : b;
}

#endif  // MERLIN_XNNHOST_MATH_H
