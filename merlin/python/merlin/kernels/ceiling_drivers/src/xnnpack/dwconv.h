// Minimal XNNPACK dwconv.h shim for the f32 dwconv RVV ceiling driver. Provides
// XNN_UNPREDICTABLE (via common.h) and the xnn_f32_default_params struct (via
// microparams.h) the unipass dwconv kernel references. NOT the real dwconv.h.
#ifndef MERLIN_CEILING_XNN_DWCONV_H
#define MERLIN_CEILING_XNN_DWCONV_H

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

#endif  // MERLIN_CEILING_XNN_DWCONV_H
