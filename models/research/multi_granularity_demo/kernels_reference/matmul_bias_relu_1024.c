// MEGAKERNEL-granularity reference kernel: fused
//   C = relu(A * B + bias)
// for the 1024x1024 case. Replaces the chain
//   linalg.matmul -> linalg.generic(bias_add) -> linalg.generic(relu)
// (4 dispatches without aggressive fusion) with a single dispatch.

#include <stdint.h>

__attribute__((visibility("hidden"))) int matmul_bias_relu_1024(
	const float *lhs, // 1024x1024 row-major
	const float *rhs, // 1024x1024 row-major
	const float *bias, // 1024
	float *out) { // 1024x1024 row-major
	const int64_t M = 1024;
	const int64_t N = 1024;
	const int64_t K = 1024;
	for (int64_t m = 0; m < M; ++m) {
		for (int64_t n = 0; n < N; ++n) {
			float acc = bias[n];
			for (int64_t k = 0; k < K; ++k) {
				acc += lhs[m * K + k] * rhs[k * N + n];
			}
			out[m * N + n] = acc > 0.0f ? acc : 0.0f;
		}
	}
	return 0;
}
