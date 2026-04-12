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

	// Create NUM_INPUTS identical inputs
	for (int inp = 0; inp < NUM_INPUTS; ++inp) {
		void *input_data = malloc(num_elements * elem_size);
		if (!input_data) {
			return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "malloc");
		}
		memset(input_data, 0, num_elements * elem_size);
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
#ifdef FAST_BENCHMARK
	const int kWarmup = 1;
#else
	const int kWarmup = 2;
#endif
#ifdef FAST_BENCHMARK
	const int kIters = 3;
#else
	const int kIters = 10;
#endif

	fprintf(stdout, "Warmup START (%d iterations)\n", kWarmup);
	fflush(stdout);
	for (int i = 0; i < kWarmup; ++i) {
		uint64_t w_start = read_cycles();
		fprintf(stdout, "Warmup iter %d/%d enter (cycle=%lu)\n", i + 1, kWarmup,
			(unsigned long)w_start);
		fflush(stdout);
		iree_vm_list_resize(outputs, 0);
		{
			iree_status_t s = iree_vm_invoke(context, main_function,
				IREE_VM_INVOCATION_FLAG_NONE, NULL, inputs, outputs,
				iree_allocator_system());
			if (!iree_status_is_ok(s)) {
				uint64_t w_fail = read_cycles();
				fprintf(stderr,
					"INVOKE FAILED at warmup iter %d (cycle=%lu, delta=%lu): ",
					i + 1, (unsigned long)w_fail,
					(unsigned long)(w_fail - w_start));
				iree_status_fprint(stderr, s);
				fprintf(stderr, "\n");
				fflush(stderr);
				iree_status_free(s);
				return iree_make_status(IREE_STATUS_INTERNAL, "invoke failed");
			}
		}
		uint64_t w_end = read_cycles();
		fprintf(stdout, "Warmup iter %d/%d done (cycle=%lu, delta=%lu)\n",
			i + 1, kWarmup, (unsigned long)w_end,
			(unsigned long)(w_end - w_start));
		fflush(stdout);
	}

	fprintf(stdout, "Benchmark START (%d iterations)\n", kIters);
	fflush(stdout);
	uint64_t start = read_cycles();
	for (int i = 0; i < kIters; ++i) {
		fprintf(stdout, "Bench iter %d/%d\n", i + 1, kIters);
		fflush(stdout);
		iree_vm_list_resize(outputs, 0);
		IREE_RETURN_IF_ERROR(
			iree_vm_invoke(context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
				NULL, inputs, outputs, iree_allocator_system()));
	}
	uint64_t end = read_cycles();

	uint64_t total_cycles = end - start;
	uint64_t avg_cycles = total_cycles / kIters;

	// CSV Output: model, variant, avg_cycles
	fprintf(stdout, "CSV, %s, %s, %lu\n", MODEL_NAME, VARIANT_NAME,
		(unsigned long)avg_cycles);

	// Cleanup
	iree_vm_list_release(inputs);
	iree_vm_list_release(outputs);
	iree_hal_device_release(device);
	iree_vm_context_release(context);
	iree_vm_instance_release(instance);

	return iree_ok_status();
}

int main(void) {
	const iree_status_t result = Run();
	if (!iree_status_is_ok(result)) {
		iree_status_fprint(stderr, result);
		iree_status_free(result);
		return 1;
	}
	return 0;
}
