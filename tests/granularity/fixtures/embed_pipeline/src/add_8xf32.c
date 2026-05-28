// Tiny f32 elementwise add — exercises the auto-embed pipeline end-to-end.
//
// Signature follows the IREE CPU custom-dispatch ABI for embedded ELF targets
// (one workgroup per element since the auto-generated transform spec uses
// `count(%workload) -> (%workload, 1, 1)` — see
// tools/kernels/spec_gen.py:_executable_block). Each binding lowers to
// (ptr, offset); after all bindings come (dim, tid). Reference:
// third_party/iree_bar/samples/custom_dispatch/cpu/embedded/functions.c.
//
// Scalar code by design; production SaturnOPU kernels under
// benchmarks/SaturnOPU/kernels/ use RVV intrinsics.

#include <stddef.h>
#include <stdint.h>

void add_8xf32_workgroup(const float *restrict binding0, size_t binding0_offset,
	const float *restrict binding1, size_t binding1_offset,
	float *restrict binding2, size_t binding2_offset, size_t dim, size_t tid) {
	if (tid >= dim)
		return;
	binding2[binding2_offset + tid] =
		binding0[binding0_offset + tid] + binding1[binding1_offset + tid];
}
