// LAYER-granularity reference kernel: elementwise multiply on a
// 1024-element f32 vector. Replaces a single linalg.generic dispatch.

#include <stdint.h>

__attribute__((visibility("hidden"))) int elementwise_mul_1024(
	const float *lhs, const float *rhs, float *out) {
	for (int64_t i = 0; i < 1024; ++i) {
		out[i] = lhs[i] * rhs[i];
	}
	return 0;
}
