// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Generic model benchmark harness for FireSim / bare-metal IREE.
// Supports models with one or two inputs of float32 or int64 type.
// Configure via compile-time defines:
//   MODEL_NAME    — human-readable name (string)
//   INPUT_NDIMS   — number of input dimensions (1-4)
//   INPUT_DIM0..3 — input dimension sizes
//   VARIANT_NAME  — "OPU" or "RVV" (string)
//   INPUT_TYPE_I64 — if defined, use int64 inputs instead of float32
//   NUM_INPUTS    — number of inputs (default 1, max 2)

#include <malloc.h> // memalign — GNU/newlib extension, not in <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/device_group.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

// --- Hardware Timer ---
static inline uint64_t read_cycles(void) {
	uint64_t cycles;
	asm volatile("rdcycle %0" : "=r"(cycles));
	return cycles;
}

// External declarations (from device_embedded_sync.c)
extern iree_status_t create_sample_device(
	iree_allocator_t host_allocator, iree_hal_device_t **out_device);
extern const iree_const_byte_span_t load_bytecode_module_data(void);

// --- Configuration ---
#ifndef MODEL_NAME
#define MODEL_NAME "UNKNOWN"
#endif
#ifndef VARIANT_NAME
#define VARIANT_NAME "UNKNOWN"
#endif
#ifndef INPUT_NDIMS
#define INPUT_NDIMS 4
#endif
#ifndef INPUT_DIM0
#define INPUT_DIM0 1
#endif
#ifndef INPUT_DIM1
#define INPUT_DIM1 3
#endif
#ifndef INPUT_DIM2
#define INPUT_DIM2 224
#endif
#ifndef INPUT_DIM3
#define INPUT_DIM3 224
#endif
#ifndef FUNC_NAME
#define FUNC_NAME "module.main"
#endif
#ifndef NUM_INPUTS
#define NUM_INPUTS 1
#endif

iree_status_t Run(void) {
	iree_vm_instance_t *instance = NULL;
	IREE_RETURN_IF_ERROR(iree_vm_instance_create(
		IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(), &instance));
	IREE_RETURN_IF_ERROR(iree_hal_module_register_all_types(instance));

	iree_hal_device_t *device = NULL;
	IREE_RETURN_IF_ERROR(
		create_sample_device(iree_allocator_system(), &device));

	iree_hal_device_group_t *device_group = NULL;
	IREE_RETURN_IF_ERROR(iree_hal_device_group_create_from_device(
		device, iree_allocator_system(), &device_group));

	iree_vm_module_t *hal_module = NULL;
	IREE_RETURN_IF_ERROR(iree_hal_module_create(instance,
		iree_hal_module_device_policy_default(), device_group,
		IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
		iree_hal_module_debug_sink_stdio(stderr), iree_allocator_system(),
		&hal_module));
	iree_hal_device_group_release(device_group);

	const iree_const_byte_span_t module_data = load_bytecode_module_data();
	iree_vm_module_t *bytecode_module = NULL;
	IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(instance,
		IREE_VM_BYTECODE_MODULE_FLAG_NONE, module_data, iree_allocator_null(),
		iree_allocator_system(), &bytecode_module));

	iree_vm_context_t *context = NULL;
	iree_vm_module_t *modules[] = {hal_module, bytecode_module};
	IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(instance,
		IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), &modules[0],
		iree_allocator_system(), &context));
	iree_vm_module_release(hal_module);
	iree_vm_module_release(bytecode_module);

	// Resolve function
	iree_vm_function_t main_function;
	IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
		context, iree_make_cstring_view(FUNC_NAME), &main_function));

	// --- Build input shape ---
	iree_hal_dim_t shape[4] = {INPUT_DIM0, INPUT_DIM1, INPUT_DIM2, INPUT_DIM3};
	const int ndims = INPUT_NDIMS;

	iree_host_size_t num_elements = 1;
	for (int d = 0; d < ndims; ++d) {
		num_elements *= (iree_host_size_t)shape[d];
	}

	fprintf(stdout, "Model: %s, Variant: %s\n", MODEL_NAME, VARIANT_NAME);
#ifdef INPUT_TYPE_I64
	const char *type_name = "i64";
	iree_hal_element_type_t elem_type = IREE_HAL_ELEMENT_TYPE_INT_64;
	iree_host_size_t elem_size = sizeof(int64_t);
#else
	const char *type_name = "f32";
	iree_hal_element_type_t elem_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
	iree_host_size_t elem_size = sizeof(float);
#endif
	fprintf(stdout, "Input shape: ");
	for (int d = 0; d < ndims; ++d) {
		fprintf(stdout, "%s%d", d > 0 ? "x" : "", (int)shape[d]);
	}
	fprintf(stdout, " (%lu elements, %s, %d inputs)\n",
		(unsigned long)num_elements, type_name, NUM_INPUTS);
	fflush(stdout);

	iree_hal_buffer_params_t params = {
		.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
		.usage = IREE_HAL_BUFFER_USAGE_DEFAULT};

	iree_vm_list_t *inputs = NULL;
	IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
		NUM_INPUTS, iree_allocator_system(), &inputs));

	// Create NUM_INPUTS identical inputs.
	// Phase-2 alignment hypothesis: force the host-side input buffer to be
	// 128-byte aligned so a misaligned source can't be the cause of the
	// downstream device buffer landing on a bad alignment. The runtime
	// copies from this into a fresh device buffer; if either copy step
	// loses alignment we'll see it in the [binding] prints inside the ELF
	// loader.
	for (int inp = 0; inp < NUM_INPUTS; ++inp) {
		void *input_data = NULL;
		size_t input_bytes = num_elements * elem_size;
		size_t input_bytes_aligned = (input_bytes + 127) & ~((size_t)127);
		// memalign (GNU/newlib) instead of posix_memalign because newlib
		// doesn't link the latter on bare-metal even though it declares it.
		input_data = memalign(128, input_bytes_aligned);
		if (!input_data) {
			return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
				"memalign(128, %zu) failed", input_bytes_aligned);
		}
		extern int iree_merlin_binding_debug_enabled;
		if (iree_merlin_binding_debug_enabled) {
			fprintf(stderr,
				"[align] input[%d] @ %p (mod128=%lu mod64=%lu mod16=%lu) "
				"bytes=%zu\n",
				inp, input_data, (unsigned long)((uintptr_t)input_data & 0x7f),
				(unsigned long)((uintptr_t)input_data & 0x3f),
				(unsigned long)((uintptr_t)input_data & 0x0f), input_bytes);
			fflush(stderr);
		}
		memset(input_data, 0, input_bytes);
#ifdef INPUT_TYPE_I64
		// Fill with token IDs 1..N for LLM inputs
		int64_t *i64_data = (int64_t *)input_data;
		for (iree_host_size_t i = 0; i < num_elements; ++i) {
			i64_data[i] = (inp == 0) ? (int64_t)(i + 1) : 1;
		}
#else
		float *f32_data = (float *)input_data;
		for (iree_host_size_t i = 0; i < num_elements; ++i) {
			f32_data[i] = (float)(i % 256) / 256.0f;
		}
#endif
		iree_hal_buffer_view_t *bv = NULL;
		IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(device,
			iree_hal_device_allocator(device), ndims, shape, elem_type,
			IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
			iree_make_const_byte_span(input_data, num_elements * elem_size),
			&bv));
		free(input_data);
		iree_vm_ref_t ref = iree_hal_buffer_view_move_ref(bv);
		iree_vm_list_push_ref_move(inputs, &ref);
	}

	iree_vm_list_t *outputs = NULL;
	IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
		1, iree_allocator_system(), &outputs));

	// --- Benchmark Loop ---
	// Paper-grade measurement: 2 warmup + 5 bench, all silent during the
	// hot loop so the UART syscall cost doesn't pollute iteration timing.
	// Per-iteration cycle counts get stashed in a local array and dumped
	// once at the end as a single CSV row:
	//   CSV, <model>, <variant>, <cyc1>, <cyc2>, <cyc3>, <cyc4>, <cyc5>, <avg>
#ifndef BENCH_WARMUP_ITERS
#define BENCH_WARMUP_ITERS 2
#endif
#ifndef BENCH_MEASURE_ITERS
#define BENCH_MEASURE_ITERS 5
#endif
	const int kWarmup = BENCH_WARMUP_ITERS;
	const int kIters = BENCH_MEASURE_ITERS;
	// Always allocate >= 5 slots so the hardcoded 5-slot CSV row + avg
	// calc below stays valid even when BENCH_MEASURE_ITERS < 5. Unused
	// slots remain zero, which sums correctly into the kIters-divided avg.
#define BENCH_ITER_SLOTS ((BENCH_MEASURE_ITERS) > 5 ? (BENCH_MEASURE_ITERS) : 5)
	uint64_t iter_cycles[BENCH_ITER_SLOTS] = {0};

	// Dispatch-level debug prints ([dn], [dc]) are gated on
	// MERLIN_DISPATCH_DEBUG. Microtests enable them; full-model benchmarks
	// leave them off so the timed loop stays silent.
#if defined(MERLIN_DISPATCH_DEBUG) && MERLIN_DISPATCH_DEBUG
	extern void iree_merlin_enable_dispatch_debug(int);
	iree_merlin_enable_dispatch_debug(1);
#endif

	// Warmup — silent.
	for (int i = 0; i < kWarmup; ++i) {
		iree_vm_list_resize(outputs, 0);
		iree_status_t s =
			iree_vm_invoke(context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
				NULL, inputs, outputs, iree_allocator_system());
		if (!iree_status_is_ok(s)) {
			fprintf(stderr, "INVOKE FAILED at warmup iter %d: ", i + 1);
			iree_status_fprint(stderr, s);
			fprintf(stderr, "\n");
			fflush(stderr);
			iree_status_free(s);
			return iree_make_status(IREE_STATUS_INTERNAL, "invoke failed");
		}
	}

	// Benchmark — silent, per-iter cycles captured to stack array.
	for (int i = 0; i < kIters; ++i) {
		uint64_t it_start = read_cycles();
		iree_vm_list_resize(outputs, 0);
		IREE_RETURN_IF_ERROR(
			iree_vm_invoke(context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
				NULL, inputs, outputs, iree_allocator_system()));
		iter_cycles[i] = read_cycles() - it_start;
	}

	uint64_t avg_cycles = (iter_cycles[0] + iter_cycles[1] + iter_cycles[2] +
							  iter_cycles[3] + iter_cycles[4]) /
		(uint64_t)kIters;

	// Single post-bench CSV row. Parser expects exactly this format.
	fprintf(stdout, "CSV, %s, %s, %lu, %lu, %lu, %lu, %lu, %lu\n", MODEL_NAME,
		VARIANT_NAME, (unsigned long)iter_cycles[0],
		(unsigned long)iter_cycles[1], (unsigned long)iter_cycles[2],
		(unsigned long)iter_cycles[3], (unsigned long)iter_cycles[4],
		(unsigned long)avg_cycles);

	// Per-dispatch runtime cycle summary — ONLY in profile builds. Clean
	// builds never call the dump so zero UART traffic happens after the
	// CSV line. This keeps the speedup measurement uncontaminated by
	// the cycle-accounting overhead.
#if defined(MERLIN_PROFILE_CYCLES) && MERLIN_PROFILE_CYCLES
	extern void iree_merlin_dump_cycles(void);
	iree_merlin_dump_cycles();
#endif

	// Cleanup
	iree_vm_list_release(inputs);
	iree_vm_list_release(outputs);
	iree_hal_device_release(device);
	iree_vm_context_release(context);
	iree_vm_instance_release(instance);

	return iree_ok_status();
}

#ifdef SATURN_RVV_SELFTEST
// Isolated RVV self-test: probes the suspected-hang ops emitted by vit_small's
// LayerNorm/softmax dispatches. Each cp=N is independently skippable via the
// SATURN_RVV_SELFTEST_SKIP bitmask (1<<N) so we can step past a known hang to
// reach the next probe across multiple FireSim runs.
//
// Each print is flushed immediately so the host UART sees it before the next
// instruction runs (a hang inside the asm block leaves the previous cp= as
// the last visible line).
//
// Build with: -DSATURN_RVV_SELFTEST_SKIP=0x18 to skip cps 3 and 4 (1<<3|1<<4).
#ifndef SATURN_RVV_SELFTEST_SKIP
#define SATURN_RVV_SELFTEST_SKIP 0
#endif
#define RVV_SELFTEST_RUN(N) (!((SATURN_RVV_SELFTEST_SKIP) & (1u << (N))))

static void rvv_selftest(void) {
	fprintf(stderr, "[rvv] SELFTEST START skip=0x%x (cps 1..9)\n",
		(unsigned)(SATURN_RVV_SELFTEST_SKIP));
	fflush(stderr);

	// (1) vlenb CSR — baseline sanity. Always runs (cannot hang).
	if (RVV_SELFTEST_RUN(1)) {
		uint64_t vlenb = 0;
		asm volatile(".option push\n\t"
					 ".option arch, +v\n\t"
					 "csrr %0, vlenb\n\t"
					 ".option pop\n\t"
					 : "=r"(vlenb));
		fprintf(stderr, "[rvv] cp=1 vlenb=%llu\n", (unsigned long long)vlenb);
		fflush(stderr);
	}

	// (2) Plain vadd.vv — if even this hangs, it's not about reductions.
	if (RVV_SELFTEST_RUN(2)) {
		int32_t a[4] = {1, 2, 3, 4}, b[4] = {10, 20, 30, 40}, c[4] = {0};
		asm volatile(".option push\n\t"
					 ".option arch, +v\n\t"
					 "vsetivli zero, 4, e32, m1, ta, ma\n\t"
					 "vle32.v v8, (%0)\n\t"
					 "vle32.v v9, (%1)\n\t"
					 "vadd.vv v10, v8, v9\n\t"
					 "vse32.v v10, (%2)\n\t"
					 ".option pop\n\t"
					 :
					 : "r"(a), "r"(b), "r"(c)
					 : "memory");
		fprintf(stderr, "[rvv] cp=2 vadd c=[%d,%d,%d,%d]\n", c[0], c[1], c[2],
			c[3]);
		fflush(stderr);
	}

	// (3) vfredusum.vs — PRIMARY suspect (CONFIRMED hangs Saturn OPU).
	//     Sum of 4 floats {1,2,3,4} reduced into a scalar → expect 10.0.
	if (RVV_SELFTEST_RUN(3)) {
		float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
		float init = 0.0f;
		float out = -1.0f;
		asm volatile(".option push\n\t"
					 ".option arch, +v\n\t"
					 "vsetivli zero, 4, e32, m1, ta, ma\n\t"
					 "vle32.v v8, (%1)\n\t" // v8 = {1,2,3,4}
					 "vfmv.s.f v9, %2\n\t" // v9[0] = 0.0
					 "vfredusum.vs v10, v8, v9\n\t" // v10[0] = sum(v8)+v9[0]
					 "vfmv.f.s %0, v10\n\t" // out = v10[0]
					 ".option pop\n\t"
					 : "=f"(out)
					 : "r"(a), "f"(init)
					 : "memory");
		fprintf(stderr, "[rvv] cp=3 vfredusum out=%f (expected 10.0)\n",
			(double)out);
		fflush(stderr);
	}

	// (4) vfsqrt.v — used in LayerNorm for inv_sqrt(var+eps).
	if (RVV_SELFTEST_RUN(4)) {
		float a[4] = {4.0f, 9.0f, 16.0f, 25.0f};
		float out[4] = {0};
		asm volatile(".option push\n\t"
					 ".option arch, +v\n\t"
					 "vsetivli zero, 4, e32, m1, ta, ma\n\t"
					 "vle32.v v8, (%0)\n\t"
					 "vfsqrt.v v9, v8\n\t"
					 "vse32.v v9, (%1)\n\t"
					 ".option pop\n\t"
					 :
					 : "r"(a), "r"(out)
					 : "memory");
		fprintf(stderr,
			"[rvv] cp=4 vfsqrt out=[%f,%f,%f,%f] (expected 2,3,4,5)\n",
			(double)out[0], (double)out[1], (double)out[2], (double)out[3]);
		fflush(stderr);
	}

	// (5) vrgather.vi — used 12x in LayerNorm setup.
	if (RVV_SELFTEST_RUN(5)) {
		float a[4] = {10.0f, 20.0f, 30.0f, 40.0f};
		float out[4] = {0};
		asm volatile(".option push\n\t"
					 ".option arch, +v\n\t"
					 "vsetivli zero, 4, e32, m1, ta, ma\n\t"
					 "vle32.v v8, (%0)\n\t"
					 "vrgather.vi v9, v8, 2\n\t" // broadcast lane 2 → all lanes
					 "vse32.v v9, (%1)\n\t"
					 ".option pop\n\t"
					 :
					 : "r"(a), "r"(out)
					 : "memory");
		fprintf(stderr,
			"[rvv] cp=5 vrgather out=[%f,%f,%f,%f] (expected 30,30,30,30)\n",
			(double)out[0], (double)out[1], (double)out[2], (double)out[3]);
		fflush(stderr);
	}

	// (6) vfredmin.vs — sister of vfredusum; emitted by softmax/argmin paths.
	//     min({1,2,3,4} ∪ {+inf}) → expect 1.0
	if (RVV_SELFTEST_RUN(6)) {
		float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
		float init = 1.0e30f;
		float out = -1.0f;
		asm volatile(".option push\n\t"
					 ".option arch, +v\n\t"
					 "vsetivli zero, 4, e32, m1, ta, ma\n\t"
					 "vle32.v v8, (%1)\n\t"
					 "vfmv.s.f v9, %2\n\t"
					 "vfredmin.vs v10, v8, v9\n\t"
					 "vfmv.f.s %0, v10\n\t"
					 ".option pop\n\t"
					 : "=f"(out)
					 : "r"(a), "f"(init)
					 : "memory");
		fprintf(
			stderr, "[rvv] cp=6 vfredmin out=%f (expected 1.0)\n", (double)out);
		fflush(stderr);
	}

	// (7) vfredmax.vs — sister of vfredusum; emitted by softmax (max-subtract).
	//     max({1,2,3,4} ∪ {-inf}) → expect 4.0
	if (RVV_SELFTEST_RUN(7)) {
		float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
		float init = -1.0e30f;
		float out = -1.0f;
		asm volatile(".option push\n\t"
					 ".option arch, +v\n\t"
					 "vsetivli zero, 4, e32, m1, ta, ma\n\t"
					 "vle32.v v8, (%1)\n\t"
					 "vfmv.s.f v9, %2\n\t"
					 "vfredmax.vs v10, v8, v9\n\t"
					 "vfmv.f.s %0, v10\n\t"
					 ".option pop\n\t"
					 : "=f"(out)
					 : "r"(a), "f"(init)
					 : "memory");
		fprintf(
			stderr, "[rvv] cp=7 vfredmax out=%f (expected 4.0)\n", (double)out);
		fflush(stderr);
	}

	// (8) vfwredusum.vs — widening reduction (f32→f64 accumulator).
	//     sum({1,2,3,4} as f32) → expect 10.0 (read back as f64).
	//     Two-stage vsetvli: e64 for the f64 init scalar, then e32 for the
	//     reduction itself (vfwredusum is keyed off SEW=src width).
	if (RVV_SELFTEST_RUN(8)) {
		float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
		double init = 0.0;
		double out = -1.0;
		asm volatile(
			".option push\n\t"
			".option arch, +v\n\t"
			"vsetivli zero, 1, e64, m1, ta, ma\n\t"
			"vfmv.s.f v9, %2\n\t" // v9[0] = 0.0 (f64)
			"vsetivli zero, 4, e32, m1, ta, ma\n\t"
			"vle32.v v8, (%1)\n\t" // v8 = {1,2,3,4} (f32)
			"vfwredusum.vs v10, v8, v9\n\t" // v10[0] = sum(v8) + v9[0] (f64)
			"vsetivli zero, 1, e64, m1, ta, ma\n\t"
			"vfmv.f.s %0, v10\n\t" // out = v10[0] (f64)
			".option pop\n\t"
			: "=f"(out)
			: "r"(a), "f"(init)
			: "memory");
		fprintf(stderr, "[rvv] cp=8 vfwredusum out=%f (expected 10.0)\n", out);
		fflush(stderr);
	}

	// (9) vfslide1down.vf — tree-reduction primitive that LLVM uses as a
	//     fallback when vfredusum is unavailable. If this works but
	//     vfredusum doesn't, our scalarization can lower to a tree of
	//     vfslide1down + vfadd.vv instead of pure scalar code.
	//     {1,2,3,4} slide-down with fill 99 → expect [2,3,4,99]
	if (RVV_SELFTEST_RUN(9)) {
		float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
		float fill = 99.0f;
		float out[4] = {0};
		asm volatile(".option push\n\t"
					 ".option arch, +v\n\t"
					 "vsetivli zero, 4, e32, m1, ta, ma\n\t"
					 "vle32.v v8, (%0)\n\t"
					 "vfslide1down.vf v9, v8, %2\n\t"
					 "vse32.v v9, (%1)\n\t"
					 ".option pop\n\t"
					 :
					 : "r"(a), "r"(out), "f"(fill)
					 : "memory");
		fprintf(stderr,
			"[rvv] cp=9 vfslide1down out=[%f,%f,%f,%f] (expected 2,3,4,99)\n",
			(double)out[0], (double)out[1], (double)out[2], (double)out[3]);
		fflush(stderr);
	}

	fprintf(stderr, "[rvv] SELFTEST DONE — all enabled checkpoints passed\n");
	fflush(stderr);
}
#endif

int main(void) {
#ifdef SATURN_RVV_SELFTEST
	rvv_selftest();
#endif
	const iree_status_t result = Run();
	if (!iree_status_is_ok(result)) {
		iree_status_fprint(stderr, result);
		iree_status_free(result);
		return 1;
	}
	return 0;
}
