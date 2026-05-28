// IREE custom-dispatch ABI wrapper for the standard f32 matmul named op
// (untransposed RHS): out[m, n] = sum_k lhs[m, k] * rhs[k, n]. Companion to
// linear_f32_workgroup.c which assumes RHS-transposed layout. Kept as a
// separate file so each can land in its own .o without source-language
// preprocessor branching.
//
// Granularity: one workgroup per output element (count == M*N). Dispatch
// ordering: bindings (lhs, rhs, out) followed by push constants (M, K, N)
// followed by tid — matches the ordering tools/kernels/spec_gen.py emits.

// Note: scalar implementation by design (strided gather of a column from a
// (K, N) row-major buffer is faster scalar than a vluxei gather for small N).
// If profiling motivates a vectorized variant, include <riscv_vector.h> and
// switch to `__riscv_vlse32_v_f32m4` over `b_col` with stride `b_stride_n`.

#include <stddef.h>

static inline float dot_kn_rvv(const float *a_row, const float *b_col_strided,
	size_t b_stride_n, size_t K) {
	// dot product where rhs column is gathered from a (K, N) row-major buffer:
	// b_col_strided[k] is at offset k * b_stride_n. We use scalar fallback for
	// strided gather since a vectorized gather (vluxei) is heavier and dronet's
	// single matmul is a (1×K) · (K×1) dot.
	float acc = 0.0f;
	for (size_t k = 0; k < K; ++k) {
		acc += a_row[k] * b_col_strided[k * b_stride_n];
	}
	return acc;
}

__attribute__((visibility("default"))) void matmul_f32_workgroup(
	const float *restrict binding0, size_t binding0_offset,
	const float *restrict binding1, size_t binding1_offset,
	float *restrict binding2, size_t binding2_offset, size_t M, size_t K,
	size_t N, size_t tid) {
	if (tid >= M * N)
		return;
	size_t m = tid / N;
	size_t n = tid % N;
	const float *a_row = binding0 + binding0_offset + m * K;
	const float *b_col = binding1 + binding1_offset + n; // first elem of col n
	binding2[binding2_offset + m * N + n] = dot_kn_rvv(a_row, b_col, N, K);
}
