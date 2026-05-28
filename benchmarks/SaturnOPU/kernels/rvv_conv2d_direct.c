void kernel_conv2d(const float *input, const float *weight, const float *bias,
	float *output, int N, int IC, int IH, int IW, int OC, int KH, int KW,
	int SH, int SW, int PH, int PW) {
	/* Hand-written RVV direct conv2d, OC-vectorized.
	 * Pattern adapted from XNNPACK f32-gemm/MRxNRv-rvv (SiFive 2024):
	 * vectorize over OUTPUT channels, broadcast input scalar.
	 * - Bounds checks are scalar, hoisted outside the SIMD loop.
	 * - Weight loads are strided by IC*KH*KW (OIHW layout).
	 * - Output stores are strided by OH*OW (NCHW layout).
	 * - LMUL=4 amortizes vsetvl overhead.
	 */
	int OH = (IH + 2 * PH - KH) / SH + 1;
	int OW = (IW + 2 * PW - KW) / SW + 1;
	const ptrdiff_t oc_stride_bytes = (ptrdiff_t)IC * (ptrdiff_t)KH *
		(ptrdiff_t)KW * (ptrdiff_t)sizeof(float);
	const ptrdiff_t out_oc_stride_bytes =
		(ptrdiff_t)OH * (ptrdiff_t)OW * (ptrdiff_t)sizeof(float);

	for (int n = 0; n < N; n++) {
		for (int oh = 0; oh < OH; oh++) {
			for (int ow = 0; ow < OW; ow++) {
				int oc = 0;
				while (oc < OC) {
					size_t vl = __riscv_vsetvl_e32m4((size_t)(OC - oc));
					vfloat32m4_t vacc;
					if (bias != NULL) {
						vacc = __riscv_vle32_v_f32m4(bias + oc, vl);
					} else {
						vacc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
					}
					for (int ic = 0; ic < IC; ic++) {
						for (int kh = 0; kh < KH; kh++) {
							int ih = oh * SH - PH + kh;
							if (ih < 0 || ih >= IH)
								continue;
							for (int kw = 0; kw < KW; kw++) {
								int iw = ow * SW - PW + kw;
								if (iw < 0 || iw >= IW)
									continue;
								float v =
									input[((n * IC + ic) * IH + ih) * IW + iw];
								const float *w_ptr = weight +
									((oc * IC + ic) * KH + kh) * KW + kw;
								vfloat32m4_t vw = __riscv_vlse32_v_f32m4(
									w_ptr, oc_stride_bytes, vl);
								vacc = __riscv_vfmacc_vf_f32m4(vacc, v, vw, vl);
							}
						}
					}
					float *out_ptr =
						output + ((n * OC + oc) * OH + oh) * OW + ow;
					__riscv_vsse32_v_f32m4(
						out_ptr, out_oc_stride_bytes, vacc, vl);
					oc += (int)vl;
				}
			}
		}
	}
}
