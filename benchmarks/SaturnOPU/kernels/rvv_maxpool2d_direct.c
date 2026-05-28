void kernel_maxpool2d(const float *input, float *output, int N, int C, int IH,
	int IW, int KH, int KW, int SH, int SW) {
	int OH = (IH - KH) / SH + 1;
	int OW = (IW - KW) / SW + 1;
	size_t vl;
	for (int n = 0; n < N; n++) {
		for (int c = 0; c < C; c++) {
			for (int oh = 0; oh < OH; oh++) {
				for (int ow = 0; ow < OW; ow += vl) {
					vl = __riscv_vsetvl_e32m1(OW - ow);
					int ih0 = oh * SH;
					int iw0 = ow * SW;
					vfloat32m1_t v_max = __riscv_vfmv_v_f_f32m1(-FLT_MAX, vl);
					for (int kh = 0; kh < KH; kh++) {
						int ih = ih0 + kh;
						for (int kw = 0; kw < KW; kw++) {
							int iw = iw0 + kw;
							vfloat32m1_t v_input = __riscv_vlse32_v_f32m1(
								input + ((n * C + c) * IH + ih) * IW + iw,
								(ptrdiff_t)sizeof(float) * SW, vl);
							v_max = __riscv_vfmax_vv_f32m1(v_max, v_input, vl);
						}
					}
					__riscv_vse32_v_f32m1(
						output + ((n * C + c) * OH + oh) * OW + ow, v_max, vl);
				}
			}
		}
	}
}
