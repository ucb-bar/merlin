void kernel_conv2d(const float *input, const float *weight, const float *bias,
	float *output, int N, int IC, int IH, int IW, int OC, int KH, int KW,
	int SH, int SW, int PH, int PW) {
	int OH = (IH + 2 * PH - KH) / SH + 1;
	int OW = (IW + 2 * PW - KW) / SW + 1;
	int M = OH * OW;
	int K = IC * KH * KW;

	float im2col_buf[M * K];

	for (int n = 0; n < N; n++) {
		/* Stage 1: im2col gather, padding-aware. */
		for (int oh = 0; oh < OH; oh++) {
			for (int ow = 0; ow < OW; ow++) {
				int row = oh * OW + ow;
				for (int ic = 0; ic < IC; ic++) {
					for (int kh = 0; kh < KH; kh++) {
						int ih = oh * SH - PH + kh;
						for (int kw = 0; kw < KW; kw++) {
							int iw = ow * SW - PW + kw;
							int col = (ic * KH + kh) * KW + kw;
							float v = 0.0f;
							if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
								v = input[((n * IC + ic) * IH + ih) * IW + iw];
							}
							im2col_buf[row * K + col] = v;
						}
					}
				}
			}
		}

		/* Stage 2: GEMM. weight is [OC, K] (OIHW flattened), output is
		 * NCHW which equals [N, OC, M] when M = OH*OW. */
		for (int oc = 0; oc < OC; oc++) {
			size_t vl;
			size_t vlmax = __riscv_vsetvlmax_e32m1();
			float b = bias ? bias[oc] : 0.0f;
			for (int row = 0; row < M; row++) {
				vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
				for (int k = 0; k < K; k += vl) {
					vl = __riscv_vsetvl_e32m1(K - k);
					vfloat32m1_t va =
						__riscv_vle32_v_f32m1(&im2col_buf[row * K + k], vl);
					vfloat32m1_t vw =
						__riscv_vle32_v_f32m1(&weight[oc * K + k], vl);
					vacc = __riscv_vfmacc_vv_f32m1(vacc, va, vw, vl);
				}
				vfloat32m1_t vinit = __riscv_vfmv_s_f_f32m1(0.0f, 1);
				vfloat32m1_t vsum =
					__riscv_vfredusum_vs_f32m1_f32m1(vacc, vinit, vlmax);
				float acc = __riscv_vfmv_f_s_f32m1_f32(vsum) + b;
				output[(n * OC + oc) * M + row] = acc;
			}
		}
	}
}
