// Minimal XNNPACK common.h shim for the HOST xnnpack_host backend's GEMM shim.
// Provides only the macros the scalar f32-gemm microkernel references. This is NOT
// the real XNNPACK common.h. Mirrors ceiling_drivers/src/xnnpack/common.h.
#ifndef MERLIN_XNNHOST_COMMON_H
#define MERLIN_XNNHOST_COMMON_H

#include <stddef.h>
#include <stdint.h>

#ifndef XNN_UNLIKELY
#define XNN_UNLIKELY(condition) (__builtin_expect(!!(condition), 0))
#endif
#ifndef XNN_LIKELY
#define XNN_LIKELY(condition) (__builtin_expect(!!(condition), 1))
#endif
#ifndef XNN_UNPREDICTABLE
#define XNN_UNPREDICTABLE(condition) (__builtin_expect(!!(condition), 0))
#endif
#ifndef XNN_INLINE
#define XNN_INLINE inline
#endif
#ifndef XNN_LOG2_SIZEOF_FLOAT
#define XNN_LOG2_SIZEOF_FLOAT 2
#endif

#endif  // MERLIN_XNNHOST_COMMON_H
