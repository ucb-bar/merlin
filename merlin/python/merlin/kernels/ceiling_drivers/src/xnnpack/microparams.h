// Minimal XNNPACK microparams.h shim for the ceiling drivers. Verbatim copies of
// the only param structs the RVV microkernels we race reference (layout taken
// from upstream src/xnnpack/microparams.h). NOT the full microparams header.
#ifndef MERLIN_CEILING_XNN_MICROPARAMS_H
#define MERLIN_CEILING_XNN_MICROPARAMS_H

#include <stdint.h>

// elementwise activations (vgelu / vsigmoid / vtanh): dummy single-member param.
struct xnn_f32_default_params {
  char _;
};

// f32 GEMM/dwconv output clamp.
struct xnn_f32_minmax_params {
  struct {
    float min;
    float max;
  } scalar;
};

// per-row dynamic activation quantization params for qd8 GEMM.
struct xnn_qd8_quantization_params {
  int32_t zero_point;
  float inv_scale;
};

#endif  // MERLIN_CEILING_XNN_MICROPARAMS_H
