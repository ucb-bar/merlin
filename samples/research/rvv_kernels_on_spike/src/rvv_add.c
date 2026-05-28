// SPDX-License-Identifier: Apache-2.0
// Vector add (f32) using RVV 1.0 intrinsics. The same source compiles for
// every IREE-supported RISC-V target with `--march=rv64gcv` plus the
// per-target tuning flags (e.g. `-mcpu=spacemit-x60` for SpacemiT). Used as
// a self-contained test on Spike to validate the embedding+execution
// pipeline before wiring into IREE's hal.executable.objects.

#include <riscv_vector.h>
#include <stddef.h>

void rvv_add_f32(const float *a, const float *b, float *c, size_t n) {
	size_t vl;
	for (size_t i = 0; i < n; i += vl) {
		vl = __riscv_vsetvl_e32m4(n - i);
		vfloat32m4_t va = __riscv_vle32_v_f32m4(a + i, vl);
		vfloat32m4_t vb = __riscv_vle32_v_f32m4(b + i, vl);
		vfloat32m4_t vc = __riscv_vfadd_vv_f32m4(va, vb, vl);
		__riscv_vse32_v_f32m4(c + i, vc, vl);
	}
}
