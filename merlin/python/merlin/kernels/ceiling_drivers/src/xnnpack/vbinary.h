// Minimal XNNPACK vbinary.h shim: the vmul/vadd/... RVV kernels only #include this
// for the param struct (the prototype comes from the kernel source itself). The real
// header additionally pulls in every f16/qs8/... .inc kernel family, which we do not
// build. Pull in just the param definitions.
#ifndef MERLIN_CEILING_XNN_VBINARY_H
#define MERLIN_CEILING_XNN_VBINARY_H

#include "src/xnnpack/microparams.h"

#endif  // MERLIN_CEILING_XNN_VBINARY_H
