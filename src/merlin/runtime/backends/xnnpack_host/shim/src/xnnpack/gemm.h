// Minimal XNNPACK gemm.h shim so the scalar f32-gemm microkernel compiles standalone
// for the host xnnpack_host backend. NOT the real XNNPACK gemm.h.
#ifndef MERLIN_XNNHOST_GEMM_H
#define MERLIN_XNNHOST_GEMM_H

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

#endif  // MERLIN_XNNHOST_GEMM_H
