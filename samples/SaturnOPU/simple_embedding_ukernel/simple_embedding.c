// Copyright 2025 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

// --- Hardware Timer ---
static inline uint64_t read_cycles() {
	uint64_t cycles;
	asm volatile("rdcycle %0" : "=r"(cycles));
	return cycles;
}

// External declarations
extern iree_status_t create_sample_device(
	iree_allocator_t host_allocator, iree_hal_device_t **out_device);
extern const iree_const_byte_span_t load_bytecode_module_data();

// --- Configuration Macros ---
#ifndef MODEL_SIZE
#define MODEL_SIZE 64
#endif
#ifndef UKERNEL_NAME
#define UKERNEL_NAME "UNKNOWN"
#endif

iree_status_t Run() {
	iree_vm_instance_t *instance = NULL;
	IREE_RETURN_IF_ERROR(iree_vm_instance_create(
		IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(), &instance));
	IREE_RETURN_IF_ERROR(iree_hal_module_register_all_types(instance));

	iree_hal_device_t *device = NULL;
	IREE_RETURN_IF_ERROR(
		create_sample_device(iree_allocator_system(), &device));

	iree_vm_module_t *hal_module = NULL;
	IREE_RETURN_IF_ERROR(iree_hal_module_create(instance,
		iree_hal_module_device_policy_default(), 1, &device,
		IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
		iree_hal_module_debug_sink_stdio(stderr), iree_allocator_system(),
		&hal_module));

	const iree_const_byte_span_t module_data = load_bytecode_module_data();
	iree_vm_module_t *bytecode_module = NULL;
	IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(instance, module_data,
		iree_allocator_null(), iree_allocator_system(), &bytecode_module));

	iree_vm_context_t *context = NULL;
	iree_vm_module_t *modules[] = {hal_module, bytecode_module};
	IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(instance,
		IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), &modules[0],
		iree_allocator_system(), &context));
	iree_vm_module_release(hal_module);
	iree_vm_module_release(bytecode_module);

	// Standard function name
	iree_vm_function_t main_function;
	IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
		context, iree_make_cstring_view("module.main"), &main_function));

	// --- Prepare Data ---
	const int size = MODEL_SIZE;
	const iree_host_size_t kCount = (iree_host_size_t)size * size;

	fprintf(stdout, "Benchmarking: Size=%d, Ukernel=%s\n", size, UKERNEL_NAME);

	// Allocate host memory
	int8_t *valA = (int8_t *)malloc(kCount * sizeof(int8_t));
	int8_t *valB = (int8_t *)malloc(kCount * sizeof(int8_t));
	memset(valA, 4, kCount * sizeof(int8_t));
	memset(valB, 2, kCount * sizeof(int8_t));

	iree_hal_dim_t shape[2] = {size, size};
	iree_hal_buffer_view_t *bv0 = NULL, *bv1 = NULL;

	iree_hal_buffer_params_t params = {
		.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
		.usage = IREE_HAL_BUFFER_USAGE_DEFAULT};

	IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(device,
		iree_hal_device_allocator(device), 2, shape,
		IREE_HAL_ELEMENT_TYPE_SINT_8, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
		params, iree_make_const_byte_span(valA, kCount), &bv0));

	IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(device,
		iree_hal_device_allocator(device), 2, shape,
		IREE_HAL_ELEMENT_TYPE_SINT_8, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
		params, iree_make_const_byte_span(valB, kCount), &bv1));

	// FREE HOST MEMORY ASAP
	free(valA);
	free(valB);

	iree_vm_list_t *inputs = NULL;
	IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
		2, iree_allocator_system(), &inputs));

	iree_vm_ref_t ref0 = iree_hal_buffer_view_move_ref(bv0);
	iree_vm_ref_t ref1 = iree_hal_buffer_view_move_ref(bv1);
	iree_vm_list_push_ref_move(inputs, &ref0);
	iree_vm_list_push_ref_move(inputs, &ref1);

	iree_vm_list_t *outputs = NULL;
	IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
		1, iree_allocator_system(), &outputs));

	// --- Benchmark Loop ---
	const int kWarmup = 2;
	const int kIters = 10;

	for (int i = 0; i < kWarmup; ++i) {
		iree_vm_list_resize(outputs, 0);
		IREE_RETURN_IF_ERROR(
			iree_vm_invoke(context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
				NULL, inputs, outputs, iree_allocator_system()));
	}

	uint64_t start = read_cycles();
	for (int i = 0; i < kIters; ++i) {
		iree_vm_list_resize(outputs, 0);
		IREE_RETURN_IF_ERROR(
			iree_vm_invoke(context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
				NULL, inputs, outputs, iree_allocator_system()));
	}
	uint64_t end = read_cycles();

	// --- SAFE INTEGER MATH ---
	uint64_t total_cycles = end - start;
	uint64_t avg_cycles_int = total_cycles / kIters;

	// Calculate Total Ops = 2 * N^3
	// Use uint64_t to prevent overflow for large N
	uint64_t n = (uint64_t)size;
	uint64_t total_ops = 2 * n * n * n;

	// Calculate Efficiency (Ops/Cycle) with 2 decimal places manually
	// Formula: (Ops * 100) / Cycles
	// This gives us "Efficiency x 100"
	uint64_t efficiency_x100 = 0;
	if (avg_cycles_int > 0) {
		efficiency_x100 = (total_ops * 100) / avg_cycles_int;
	}

	uint64_t eff_whole = efficiency_x100 / 100;
	uint64_t eff_frac = efficiency_x100 % 100;

	// CSV Output using only integer formatting (%lu)
	fprintf(stdout, "CSV, %d, %s, %lu, %lu.%02lu\n", size, UKERNEL_NAME,
		avg_cycles_int, eff_whole, eff_frac);

	// --- Verification ---
	iree_hal_buffer_view_t *ret_bv =
		iree_vm_list_get_buffer_view_assign(outputs, 0);
	int32_t *res_data = (int32_t *)malloc(kCount * sizeof(int32_t));
	IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(device,
		iree_hal_buffer_view_buffer(ret_bv), 0, res_data, kCount * 4,
		IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

	int32_t expected = 8 * size;
	int errs = 0;
	for (int i = 0; i < kCount; i += 101) {
		if (res_data[i] != expected) {
			if (errs < 1)
				fprintf(stderr, "Mismatch: Expected %d, got %d\n", expected,
					res_data[i]);
			errs++;
		}
	}

	free(res_data);
	iree_vm_list_release(inputs);
	iree_vm_list_release(outputs);
	iree_hal_device_release(device);
	iree_vm_context_release(context);
	iree_vm_instance_release(instance);

	if (errs > 0)
		return iree_make_status(IREE_STATUS_UNKNOWN, "Verification Failed");
	return iree_ok_status();
}

int main() {
	const iree_status_t result = Run();
	if (!iree_status_is_ok(result)) {
		iree_status_fprint(stderr, result);
		iree_status_free(result);
		return 1;
	}
	return 0;
}
