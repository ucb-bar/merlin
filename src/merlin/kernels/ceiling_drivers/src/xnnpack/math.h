// Minimal XNNPACK math.h shim. The qd8 RVV GEMM 1x4v kernel #includes this but
// (at the 1x4v unroll) references no xnn math_* helpers; provide the standard
// libm + the common macros so it compiles standalone. NOT the full math.h.
#ifndef MERLIN_CEILING_XNN_MATH_H
#define MERLIN_CEILING_XNN_MATH_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"

#endif  // MERLIN_CEILING_XNN_MATH_H
