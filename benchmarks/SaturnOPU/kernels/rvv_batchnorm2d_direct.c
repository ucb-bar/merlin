void kernel_batchnorm2d(const float *input, const float *scale,
	const float *bias, float *output, int N, int C, int H, int W) {
	size_t vl;
	for (int n = 0; n < N; n++) {
		for (int c = 0; c < C; c++) {
			float s = scale[c];
			float b = bias[c];
			for (int h = 0; h < H; h++) {
				int idx_base = ((n * C + c) * H + h) * W;
				for (int w = 0; w < W; w += vl) {
					vl = __riscv_vsetvl_e32m1(W - w);
					vfloat32m1_t vin =
						__riscv_vle32_v_f32m1(input + idx_base + w, vl);
					/* Correct calculation: (input * scale) + bias */
					vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(s, vl);
					vfloat32m1_t vb = __riscv_vfmv_v_f_f32m1(b, vl);
					vfloat32m1_t vout = __riscv_vfadd_vv_f32m1(
						__riscv_vfmul_vv_f32m1(vin, vs, vl), vb, vl);
					__riscv_vse32_v_f32m1(output + idx_base + w, vout, vl);
				}
			}
		}
	}
}
