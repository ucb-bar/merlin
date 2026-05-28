// Scalar f32 conv2d in NCHW layout: out[n, f, oh, ow] = sum_{c, kh, kw}
// in[n, c, oh*sH + kh*dH, ow*sW + kw*dW] * w[f, c, kh, kw]. The output
// buffer carries the bias (passed as `outs(%broadcasted)` in dronet's
// preprocessing-phase IR), so we ACCUMULATE on top of `binding2[...]`
// rather than overwriting — matches the matched op's `outs` semantics.
//
// One workgroup per output element. Strides and dilations are NOT push
// constants today; the wrapper hard-codes the dronet path (stride=2,
// dilation=1) — a scalar kernel anyway, the right move for sub-1% of
// dronet runtime. A real RVV-vectorized conv kernel would unroll the
// channel loop and use vfmacc; that's a follow-on.

#include <stddef.h>

__attribute__((visibility("default"))) void conv_2d_nchw_fchw_workgroup(
	const float *restrict binding0, size_t binding0_offset, // input
	const float *restrict binding1, size_t binding1_offset, // weight
	float *restrict binding2, size_t binding2_offset, // out (carries bias)
	size_t N, size_t C_in, size_t H_in, size_t W_in, size_t F, size_t KH,
	size_t KW, size_t H_out, size_t W_out, size_t tid) {
	size_t total = N * F * H_out * W_out;
	if (tid >= total)
		return;
	// Decode tid -> (n, f, oh, ow).
	size_t hw = H_out * W_out;
	size_t fhw = F * hw;
	size_t n = tid / fhw;
	size_t r = tid - n * fhw;
	size_t f = r / hw;
	size_t r2 = r - f * hw;
	size_t oh = r2 / W_out;
	size_t ow = r2 - oh * W_out;

	// Stride 2, dilation 1 — see comment above.
	const size_t sH = 2;
	const size_t sW = 2;
	float acc =
		binding2[binding2_offset + ((n * F + f) * H_out + oh) * W_out + ow];
	for (size_t c = 0; c < C_in; ++c) {
		for (size_t kh = 0; kh < KH; ++kh) {
			size_t ih = oh * sH + kh;
			if (ih >= H_in)
				continue;
			for (size_t kw = 0; kw < KW; ++kw) {
				size_t iw = ow * sW + kw;
				if (iw >= W_in)
					continue;
				float a = binding0[binding0_offset +
					((n * C_in + c) * H_in + ih) * W_in + iw];
				float w = binding1[binding1_offset +
					((f * C_in + c) * KH + kh) * KW + kw];
				acc += a * w;
			}
		}
	}
	binding2[binding2_offset + ((n * F + f) * H_out + oh) * W_out + ow] = acc;
}
