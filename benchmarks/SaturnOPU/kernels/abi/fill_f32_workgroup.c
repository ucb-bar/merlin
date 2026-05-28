// 3D f32 zero/constant fill — one workgroup per output element.
// Matches the `linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?x?xf32>)`
// pattern that dronet emits 11 times for conv accumulator zero-init.
//
// The fill value is materialized inside the kernel as 0.0f because the
// match.mlir constrains %cst to `arith.constant 0.000000e+00 : f32` — see
// match/fill_f32.match.mlir. To support nonzero fills, fold %cst into a
// push constant (would need a float push-constant type; not yet supported
// by the auto-spec).

#include <stddef.h>

__attribute__((visibility("default"))) void fill_f32_workgroup(
	float *restrict binding0, size_t binding0_offset, size_t C, size_t H,
	size_t W, size_t tid) {
	if (tid >= C * H * W)
		return;
	binding0[binding0_offset + tid] = 0.0f;
}
