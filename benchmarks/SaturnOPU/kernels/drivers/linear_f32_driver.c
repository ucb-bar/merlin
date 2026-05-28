// Standalone Spike driver for the f32 matmul (linear) kernel. Validates
// against a scalar reference with a small fp32 tolerance and prints
// "linear_f32: PASS (M,K,N=...)" on success.

#include <math.h>
#include <stddef.h>
#include <stdio.h>

#define M 8
#define K 64
#define N 16

static float A[M * K];
static float B[N * K]; // weight stored as (N, K) row-major; transpose-of-K
static float OUT[M * N];
static float REF[M * N];

void linear_f32_workgroup(const float *binding0, size_t binding0_offset,
	const float *binding1, size_t binding1_offset, float *binding2,
	size_t binding2_offset, size_t Md, size_t Kd, size_t Nd, size_t tid);

static int approx_eq(float x, float y) {
	float d = x - y;
	if (d < 0.f)
		d = -d;
	float ax = x < 0.f ? -x : x;
	float ay = y < 0.f ? -y : y;
	float scale = ax > ay ? ax : ay;
	return d <= 1e-3f * (scale + 1.f);
}

int main(void) {
	for (int i = 0; i < M * K; ++i)
		A[i] = (float)((i * 7) % 11) - 5.f;
	for (int i = 0; i < N * K; ++i)
		B[i] = (float)((i * 13) % 9) - 4.f;
	for (int m = 0; m < M; ++m) {
		for (int n = 0; n < N; ++n) {
			float s = 0.f;
			for (int k = 0; k < K; ++k)
				s += A[m * K + k] * B[n * K + k];
			REF[m * N + n] = s;
		}
	}
	for (size_t tid = 0; tid < (size_t)(M * N); ++tid) {
		linear_f32_workgroup(
			A, 0, B, 0, OUT, 0, (size_t)M, (size_t)K, (size_t)N, tid);
	}
	int errors = 0;
	for (int i = 0; i < M * N; ++i) {
		if (!approx_eq(OUT[i], REF[i]))
			++errors;
	}
	if (errors) {
		printf("linear_f32: FAIL (%d mismatches; first OUT=%f REF=%f)\n",
			errors, OUT[0], REF[0]);
		return 1;
	}
	printf("linear_f32: PASS (M=%d K=%d N=%d)\n", M, K, N);
	return 0;
}
