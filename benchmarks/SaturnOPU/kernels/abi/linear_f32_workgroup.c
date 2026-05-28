// IREE custom-dispatch ABI wrapper around the existing RVV f32 matrix-vector
// / matmul kernel from `../rvv_linear_direct.c`. The intrinsics body is
// inlined here (precompile.py expects a single source file per kernel
// entry); keep this in sync with `kernel_linear` next door.
//
// Granularity choice: one workgroup per output element (count == M*N). The
// matched op is a 2D linalg.matmul (no bias); each invocation computes a
// single (m, n) output value. Coarser granularity (workgroup-per-row, etc.)
// is a transform-spec tweak — change the `count(...) -> (workload, 1, 1)`
// region and the wrapper's tid decode without touching this file.

#include <riscv_vector.h>
#include <stddef.h>

// In-file copy of `kernel_linear` from rvv_linear_direct.c, restricted to a
// single (m, n) compute (the wrapper picks the (m, n) pair from tid). Bias
// is not part of the matched DAG — separate dispatch.
static inline float dot_rvv_f32(
	const float *in_row, const float *w_row, int K) {
	size_t vlmax = __riscv_vsetvlmax_e32m4();
	vfloat32m4_t vacc0 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
	vfloat32m4_t vacc1 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
	size_t k = 0;
	for (; k + 2 * vlmax <= (size_t)K; k += 2 * vlmax) {
		vfloat32m4_t va0 = __riscv_vle32_v_f32m4(in_row + k, vlmax);
		vfloat32m4_t vb0 = __riscv_vle32_v_f32m4(w_row + k, vlmax);
		vacc0 = __riscv_vfmacc_vv_f32m4(vacc0, va0, vb0, vlmax);
		vfloat32m4_t va1 = __riscv_vle32_v_f32m4(in_row + k + vlmax, vlmax);
		vfloat32m4_t vb1 = __riscv_vle32_v_f32m4(w_row + k + vlmax, vlmax);
		vacc1 = __riscv_vfmacc_vv_f32m4(vacc1, va1, vb1, vlmax);
	}
	for (; k < (size_t)K; k += vlmax) {
		vlmax = __riscv_vsetvl_e32m4(K - k);
		vfloat32m4_t va = __riscv_vle32_v_f32m4(in_row + k, vlmax);
		vfloat32m4_t vb = __riscv_vle32_v_f32m4(w_row + k, vlmax);
		vacc0 = __riscv_vfmacc_vv_f32m4(vacc0, va, vb, vlmax);
	}
	vfloat32m4_t vacc = __riscv_vfadd_vv_f32m4(vacc0, vacc1, vlmax);
	vfloat32m1_t vsum = __riscv_vfredusum_vs_f32m4_f32m1(
		vacc, __riscv_vfmv_s_f_f32m1(0.0f, 1), __riscv_vsetvlmax_e32m4());
	return __riscv_vfmv_f_s_f32m1_f32(vsum);
}

// Static-shape contract: this wrapper assumes the matmul operates on a
// row-major M×K input × K×N weight, with M, K, N passed as push-constant-
// derived dims. The auto-generated transform spec doesn't yet emit push
// constants; for the standalone Spike driver the dims are hard-coded by the
// driver, and for the embedded path you must hand-edit the spec to wire
// hal.interface.constant.load — see the manifest.json comment.
__attribute__((visibility("default"))) void linear_f32_workgroup(
	const float *restrict binding0, size_t binding0_offset,
	const float *restrict binding1, size_t binding1_offset,
	float *restrict binding2, size_t binding2_offset, size_t M, size_t K,
	size_t N, size_t tid) {
	if (tid >= M * N)
		return;
	size_t m = tid / N;
	size_t n = tid % N;
	const float *in_row = binding0 + binding0_offset + m * K;
	const float *w_row = binding1 + binding1_offset + n * K;
	float acc = dot_rvv_f32(in_row, w_row, (int)K);
	binding2[binding2_offset + m * N + n] = acc;
}
