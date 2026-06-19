// Minimal XNNPACK gemm.h shim so the f32 RVV GEMM microkernel compiles
// standalone for the ceiling driver. Provides only the surface the
// f32-gemm-1x4v-rvv.c body references: XNN_UNLIKELY, restrict, and the
// xnn_f32_default_params struct (a dummy in upstream XNNPACK; see
// src/xnnpack/microparams.h). This is NOT the real XNNPACK gemm.h.
#ifndef MERLIN_CEILING_XNN_GEMM_H
#define MERLIN_CEILING_XNN_GEMM_H

#include <stddef.h>
#include <stdint.h>

#ifndef XNN_UNLIKELY
#define XNN_UNLIKELY(condition) (__builtin_expect(!!(condition), 0))
#endif

// upstream src/xnnpack/microparams.h: dummy single-member struct
struct xnn_f32_default_params {
  char _;
};

#endif // MERLIN_CEILING_XNN_GEMM_H
