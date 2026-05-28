// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Per-thread Muon kernel body for the canonical vecadd.
//
// Extracted from radiance-kernels/kernels/vecadd/kernel.cpp (commit
// 56aad6e1620c452bd131c948f72352dab0754d6e). The function signature matches
// what mu_schedule expects: it's the per-thread function the runtime
// invokes for each (warp, thread) pair. Use `extern "C"` so the symbol is
// linkable from the Phase-2 wrapper template by exact name.
//
// This file is the manifest-driven analogue of Phase 1's inline kernel
// definition. Compile via tools/kernels/precompile.py (target=radiance-muon)
// to produce radiance_vecadd_body.<hash>.muon.o, then link into
// kernel.radiance.elf alongside the wrapper that calls mu_schedule on this
// symbol.

#include <mu_intrinsics.h>
#include <shared_mem.h>

#include <stdint.h>

// VecAddArgs is declared in the wrapper TU; the precompile-time view here
// only sees that it has A/B/C/n fields of the right layout. We forward-
// declare the struct as opaque + define the four field offsets via a
// helper struct so the layout matches the wrapper exactly. This lets the
// kernel body compile in isolation without pulling in the wrapper's
// includes.
//
// IMPORTANT: keep the layout in sync with the wrapper template
// (build_tools/radiance/templates/kernel_phase2.cpp.j2) and with
// models/radiance_muon/vecadd_v2.yaml's args_fields list.
struct VecAddArgs {
	__global float *A;
	__global float *B;
	__global float *C;
	uint32_t n;
};

extern "C" void radiance_vecadd_body(void *arg, uint32_t tid_in_threadblock,
	uint32_t threads_per_threadblock, uint32_t threadblock_id) {
	auto *args = reinterpret_cast<VecAddArgs *>(arg);
	(void)threadblock_id;

	for (uint32_t i = tid_in_threadblock; i < args->n;
		 i += threads_per_threadblock) {
		args->C[i] = args->A[i] + args->B[i];
	}
}
