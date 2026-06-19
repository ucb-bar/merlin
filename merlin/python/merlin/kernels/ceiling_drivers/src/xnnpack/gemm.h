// Minimal XNNPACK gemm.h shim so the f32 / qd8 RVV GEMM microkernels compile
// standalone for the ceiling driver. Provides only the surface the kernel
// bodies reference: XNN_UNLIKELY, restrict, and the microparam structs
// (xnn_f32_default_params, xnn_f32_minmax_params, xnn_qd8_quantization_params).
// This is NOT the real XNNPACK gemm.h.
#ifndef MERLIN_CEILING_XNN_GEMM_H
#define MERLIN_CEILING_XNN_GEMM_H

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

#endif  // MERLIN_CEILING_XNN_GEMM_H
