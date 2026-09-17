// Minimal XNNPACK common.h shim for the ceiling drivers. Provides only the
// macros the RVV microkernels reference: XNN_UNLIKELY / XNN_UNPREDICTABLE,
// XNN_LOG2_SIZEOF_FLOAT, and the `restrict`/inline plumbing. This is NOT the
// real XNNPACK common.h (no arch detection, no XNN_INTERNAL machinery); it is
// the smallest surface the f32-v{gelu,sigmoid,tanh}, qd8-gemm and dwconv RVV
// kernels need to compile standalone.
#ifndef MERLIN_CEILING_XNN_COMMON_H
#define MERLIN_CEILING_XNN_COMMON_H

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

// log2(sizeof(float)) == 2; the vunary kernels use it to turn a byte `batch`
// into an element count (batch >>= XNN_LOG2_SIZEOF_FLOAT).
#ifndef XNN_LOG2_SIZEOF_FLOAT
#define XNN_LOG2_SIZEOF_FLOAT 2
#endif

/* XNN_OOB_READS: upstream (src/xnnpack/common.h) expands to the sanitizer-disable attributes. The
   harvest only needs the ukernel to COMPILE and disassemble, and the attributes affect neither the
   emitted instruction mix nor the symbol -- so an empty expansion is faithful for that purpose.
   Without it, `... params) XNN_OOB_READS {` parses as a declarator followed by garbage and clang
   reports "expected function body after function declarator" (measured on f32-vrsqrt-rvv-rsqrt-u4v.c,
   which is the RMSNorm normaliser teacher). */
#ifndef XNN_OOB_READS
#define XNN_OOB_READS
#endif

#endif  // MERLIN_CEILING_XNN_COMMON_H
