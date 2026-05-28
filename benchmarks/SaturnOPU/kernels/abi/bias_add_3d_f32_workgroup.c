// 3D + 1D-broadcast bias add: out[c, h, w] = in[c, h, w] + bias[c]. Matches
// the conv bias-add pattern dronet emits after each Conv2D: indexing maps
// (d0,d1,d2) -> (d0,d1,d2) for in/out and (d0,d1,d2) -> (d0) for the bias.
//
// One workgroup per output element. The wrapper passes (C, H, W) as push
// constants; tid decodes to (c, h, w) via integer divmod.

#include <stddef.h>

__attribute__((visibility("default"))) void bias_add_3d_f32_workgroup(
	const float *restrict binding0, size_t binding0_offset,
	const float *restrict binding1, size_t binding1_offset,
	float *restrict binding2, size_t binding2_offset, size_t C, size_t H,
	size_t W, size_t tid) {
	size_t total = C * H * W;
	if (tid >= total)
		return;
	size_t hw = H * W;
	size_t c = tid / hw;
	size_t hw_idx = tid - c * hw;
	binding2[binding2_offset + tid] =
		binding0[binding0_offset + c * hw + hw_idx] +
		binding1[binding1_offset + c];
}
