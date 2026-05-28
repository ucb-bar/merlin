// IREE custom-dispatch ABI wrapper around the existing RVV f32 elementwise
// add kernel from `../rvv_add_direct.c`. The intrinsics body is inlined here
// (precompile.py expects a single source file per kernel entry); keep this
// in sync with `kernel_add` next door — the paper-baseline file is the
// canonical one, this is the IREE-ABI-shaped sibling.
//
// ABI: per kernel binding lowers to (ptr, offset); after all bindings come
// (dim, tid). One workgroup per element since the auto-generated transform
// spec uses `count(%workload) -> (%workload, 1, 1)`. Reference:
// third_party/iree_bar/samples/custom_dispatch/cpu/embedded/functions.c.

#include <riscv_vector.h>
#include <stddef.h>

static void rvv_add_f32(const float *a, const float *b, float *out, int n) {
	size_t vl;
	for (int i = 0; i < n; i += vl) {
		vl = __riscv_vsetvl_e32m8(n - i);
		vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
		vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
		vfloat32m8_t vout = __riscv_vfadd_vv_f32m8(va, vb, vl);
		__riscv_vse32_v_f32m8(out + i, vout, vl);
	}
}

__attribute__((visibility("default"))) void add_f32_workgroup(
	const float *restrict binding0, size_t binding0_offset,
	const float *restrict binding1, size_t binding1_offset,
	float *restrict binding2, size_t binding2_offset, size_t dim, size_t tid) {
	// Per-workgroup slab: this workgroup processes one element at index `tid`.
	// The fast path is a vectorized add at (slab) granularity; with
	// count == workload that slab is a single element and we just do the
	// scalar fallback. Wider workgroups (e.g. count=ceildiv(workload, 64)) can
	// call rvv_add_f32 over a 64-element slice — that's a transform-spec
	// tweak, not a kernel-side change.
	if (tid >= dim)
		return;
	binding2[binding2_offset + tid] =
		binding0[binding0_offset + tid] + binding1[binding1_offset + tid];
	(void)rvv_add_f32; // referenced for clarity; not used in the per-element
					   // path.
}
