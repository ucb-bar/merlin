// SPDX-License-Identifier: Apache-2.0
// Spike-friendly driver for the rvv_add_f32 kernel.
// - All buffers are static so we don't need a heap allocator under spike.
// - Output is via printf (newlib semihosting on the proxy kernel `pk`).
// - Exit status: 0 on success, 1 on first mismatch.
//
// Build:
//   riscv64-unknown-elf-gcc -O3 -march=rv64gcv -mabi=lp64d \
//     driver.c ../src/rvv_add.c -o rvv_add_test.elf
// Run on spike:
//   spike --isa=rv64gcv pk rvv_add_test.elf

#include <math.h>
#include <stdio.h>

void rvv_add_f32(const float *a, const float *b, float *c, size_t n);

#define N 1024
static float A[N];
static float B[N];
static float C[N];

int main(void) {
	for (int i = 0; i < N; i++) {
		A[i] = (float)i * 0.5f;
		B[i] = (float)(N - i) * 0.25f;
	}
	rvv_add_f32(A, B, C, N);

	int errors = 0;
	for (int i = 0; i < N; i++) {
		float expected = A[i] + B[i];
		if (fabsf(C[i] - expected) > 1e-4f) {
			if (errors < 5) {
				printf("MISMATCH [%d]: got %f, expected %f\n", i, (double)C[i],
					(double)expected);
			}
			errors++;
		}
	}
	if (errors == 0) {
		printf("rvv_add_f32: PASS (n=%d)\n", N);
		return 0;
	}
	printf("rvv_add_f32: FAIL (%d mismatches)\n", errors);
	return 1;
}
