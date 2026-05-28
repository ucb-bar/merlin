// TILE-granularity reference kernel: computes a 64x64 sub-tile of a
// 256x256 matmul. Uses the IREE embedded-elf ABI (see
// samples/custom_dispatch/cpu/embedded/CMakeLists.txt for the build
// recipe; tools/kernels/precompile.py wires this up automatically).
//
// Performance is intentionally not optimized — the goal is byte-equal
// numerics against the un-embedded baseline so the embedding mechanism
// itself can be verified end-to-end.

#include <stdint.h>

__attribute__((visibility("hidden"))) int tile_matmul_64x64(
	const float *lhs, // 64xK strided slice of A
	const float *rhs, // Kx64 strided slice of B
	float *out, // 64x64 sub-tile of C
	int64_t K, // contracted dim
	int64_t lhs_row_stride, int64_t rhs_row_stride, int64_t out_row_stride) {
	for (int64_t m = 0; m < 64; ++m) {
		for (int64_t n = 0; n < 64; ++n) {
			float acc = out[m * out_row_stride + n]; // pick up bias if any
			for (int64_t k = 0; k < K; ++k) {
				acc +=
					lhs[m * lhs_row_stride + k] * rhs[k * rhs_row_stride + n];
			}
			out[m * out_row_stride + n] = acc;
		}
	}
	return 0;
}
