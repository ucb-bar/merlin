// Minimal XNNPACK intrinsics-polyfill.h shim. The real header polyfills RVV / x86
// intrinsics for OLD toolchains and pulls in unaligned.h + the full math.h. The RVV
// kernels that include it (f32-rmax, f32-vclamp) use only NATIVE RVV intrinsics
// (__riscv_vfmax_vv_f32m8_tu, __riscv_vfredmax_vs_..., __riscv_vfmin/vfmax) that the
// K1 SpacemiT clang provides directly via <riscv_vector.h>. So the shim just satisfies
// the include and defines XNN_INTRINSIC (used by the real header's helpers) in case a
// kernel references it.
#ifndef MERLIN_CEILING_XNN_INTRINSICS_POLYFILL_H
#define MERLIN_CEILING_XNN_INTRINSICS_POLYFILL_H

#include <riscv_vector.h>
#include "src/xnnpack/common.h"

#ifndef XNN_INTRINSIC
#define XNN_INTRINSIC static inline __attribute__((__always_inline__))
#endif

#endif  // MERLIN_CEILING_XNN_INTRINSICS_POLYFILL_H
