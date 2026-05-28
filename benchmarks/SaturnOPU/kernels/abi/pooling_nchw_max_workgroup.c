// 2D max pool in NCHW layout: out[n, c, oh, ow] = max over (kh, kw) of
// in[n, c, oh*sH + kh, ow*sW + kw]. Stride and dilation are baked in
// (stride=2, dilation=1) — matches dronet's only max-pool layer.
//
// We declare 3 input bindings (input, window, init) — the matcher binds the
// `outs(...)` of `linalg.pooling_nchw_max` to `init` so we have a runtime
// signal for H_out / W_out. Init buffer contents are ignored; the kernel
// seeds its own -FLT_MAX accumulator. Output goes to binding3.

#include <float.h>
#include <stddef.h>

__attribute__((visibility("default"))) void pooling_nchw_max_workgroup(
	const float *restrict binding0, size_t binding0_offset, // input
	const float *restrict binding1,
	size_t binding1_offset, // window (shape only)
	const float *restrict binding2, size_t binding2_offset, // init (shape only)
	float *restrict binding3, size_t binding3_offset, // output
	size_t N, size_t C, size_t H_in, size_t W_in, size_t KH, size_t KW,
	size_t H_out, size_t W_out, size_t tid) {
	size_t total = N * C * H_out * W_out;
	if (tid >= total)
		return;
	size_t hw = H_out * W_out;
	size_t chw = C * hw;
	size_t n = tid / chw;
	size_t r = tid - n * chw;
	size_t c = r / hw;
	size_t r2 = r - c * hw;
	size_t oh = r2 / W_out;
	size_t ow = r2 - oh * W_out;

	const size_t sH = 2;
	const size_t sW = 2;
	float best = -FLT_MAX;
	for (size_t kh = 0; kh < KH; ++kh) {
		size_t ih = oh * sH + kh;
		if (ih >= H_in)
			continue;
		for (size_t kw = 0; kw < KW; ++kw) {
			size_t iw = ow * sW + kw;
			if (iw >= W_in)
				continue;
			float v = binding0[binding0_offset +
				((n * C + c) * H_in + ih) * W_in + iw];
			if (v > best)
				best = v;
		}
	}
	binding3[binding3_offset + ((n * C + c) * H_out + oh) * W_out + ow] = best;
	(void)binding1; // window tensor's only role is to encode KH×KW shape.
	(void)binding2; // init tensor's only role is to encode the output shape.
}
