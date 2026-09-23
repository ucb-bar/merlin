// Minimal XNNPACK reduce.h shim: the rsum/rmax/rminmax RVV kernels only #include this
// for the param structs (xnn_f32_scale_params for rsum, xnn_f32_default_params for
// rmax). The real header pulls in every reduction .inc family; we build one kernel at
// a time from its own source, so only the params are needed.
#ifndef MERLIN_CEILING_XNN_REDUCE_H
#define MERLIN_CEILING_XNN_REDUCE_H

#include "src/xnnpack/microparams.h"

#endif  // MERLIN_CEILING_XNN_REDUCE_H
