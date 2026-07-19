// Minimal XNNPACK transpose.h shim: the x32-transposec RVV kernel needs only common.h
// and XNN_UNREACHABLE (used in its column-tail switch default); it takes no param
// struct. The real header pulls in every transposec .inc family, which we do not build.
#ifndef MERLIN_CEILING_XNN_TRANSPOSE_H
#define MERLIN_CEILING_XNN_TRANSPOSE_H

#include "src/xnnpack/common.h"

#ifndef XNN_UNREACHABLE
#define XNN_UNREACHABLE __builtin_unreachable()
#endif

#endif  // MERLIN_CEILING_XNN_TRANSPOSE_H
