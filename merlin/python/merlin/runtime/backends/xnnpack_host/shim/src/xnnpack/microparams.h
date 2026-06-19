// Minimal XNNPACK microparams.h shim: only the f32 minmax param struct the scalar
// f32-gemm microkernel reads. NOT the real XNNPACK microparams.h.
#ifndef MERLIN_XNNHOST_MICROPARAMS_H
#define MERLIN_XNNHOST_MICROPARAMS_H

#include <stdint.h>

struct xnn_f32_minmax_params {
  struct {
    float min;
    float max;
  } scalar;
};

#endif  // MERLIN_XNNHOST_MICROPARAMS_H
