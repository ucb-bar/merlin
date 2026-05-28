// Standalone Spike driver for the f32 elementwise add kernel. Validates
// bit-exact correctness against a scalar reference, prints
// "add_f32: PASS (n=...)" on success and "FAIL (...)" otherwise. Linked
// with `abi/add_f32_workgroup.c` and run under spike+pk via
// tools/kernels/spike_runner.py.

#include <stddef.h>
#include <stdio.h>

#define N 1024

static float a[N];
static float b[N];
static float out[N];
static float ref[N];

void add_f32_workgroup(const float *binding0, size_t binding0_offset,
	const float *binding1, size_t binding1_offset, float *binding2,
	size_t binding2_offset, size_t dim, size_t tid);

int main(void) {
	for (int i = 0; i < N; ++i) {
		a[i] = (float)(i + 1);
		b[i] = (float)(2 * i - 3);
		ref[i] = a[i] + b[i];
		out[i] = 0.0f;
	}
	// Simulate the IREE workgroup grid: one workgroup per element.
	for (size_t tid = 0; tid < N; ++tid) {
		add_f32_workgroup(a, 0, b, 0, out, 0, (size_t)N, tid);
	}
	int errors = 0;
	for (int i = 0; i < N; ++i) {
		if (out[i] != ref[i]) {
			++errors;
		}
	}
	if (errors) {
		printf("add_f32: FAIL (%d mismatches out of %d)\n", errors, N);
		return 1;
	}
	printf("add_f32: PASS (n=%d)\n", N);
	return 0;
}
