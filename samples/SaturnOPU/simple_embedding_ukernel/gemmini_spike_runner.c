// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Bare-metal Gemmini-on-Spike runner for the matmul_8x8x8 fixture.
//
// Loads a pre-compiled .vmfb (embedded as .incbin in this binary), invokes
// `module.matmul_8x8x8` through the IREE local-sync HAL driver, and prints
// the i32 result row-by-row. The .vmfb's dispatch ELF contains custom-3
// (RoCC) Gemmini instructions emitted by Merlin's gemmini compiler plugin;
// `spike --extension=gemmini pk <elf>` interprets them via libgemmini.so.
//
// Test pattern: A[i][j] = (i+j) & 0x7F, B = identity → result == A as i32.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/api.h"
#include "iree/hal/device.h"
#include "iree/hal/device_group.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/loaders/embedded_elf_loader.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

// Local-sync device factory. Bare-metal: the proactor pool wraps a
// NULL platform proactor (see proactor_platform.c IREE_PLATFORM_GENERIC
// carve-out); local-sync never exercises the proactor.
static iree_status_t create_sample_device(
	iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
	iree_hal_sync_device_params_t params;
	iree_hal_sync_device_params_initialize(&params);

	iree_hal_executable_loader_t *loader = NULL;
	IREE_RETURN_IF_ERROR(iree_hal_embedded_elf_loader_create(
		/*plugin_manager=*/NULL, host_allocator, &loader));

	iree_string_view_t identifier = iree_make_cstring_view("local-sync");
	iree_hal_allocator_t *device_allocator = NULL;
	iree_status_t status = iree_hal_allocator_create_heap(
		identifier, host_allocator, host_allocator, &device_allocator);

	iree_async_proactor_pool_t *proactor_pool = NULL;
	if (iree_status_is_ok(status)) {
		// Bare-metal: no real proactor backend (proactor_platform.c returns
		// NULL under IREE_PLATFORM_GENERIC) and no thread runner
		// (IREE_ENABLE_THREADING=OFF). Suppress the default runner factory
		// so pool_get returns OK with a NULL proactor — local-sync never
		// dereferences it.
		iree_async_proactor_pool_options_t pool_options =
			iree_async_proactor_pool_options_default();
		memset(&pool_options.runner, 0, sizeof(pool_options.runner));
		status = iree_async_proactor_pool_create(
			/*node_count=*/1, /*node_ids=*/NULL, pool_options, host_allocator,
			&proactor_pool);
	}

	iree_hal_device_create_params_t create_params =
		iree_hal_device_create_params_default();
	create_params.proactor_pool = proactor_pool;
	if (iree_status_is_ok(status)) {
		status = iree_hal_sync_device_create(identifier, &params,
			&create_params, /*loader_count=*/1, &loader, device_allocator,
			host_allocator, out_device);
	}
	iree_async_proactor_pool_release(proactor_pool);
	iree_hal_allocator_release(device_allocator);
	iree_hal_executable_loader_release(loader);
	return status;
}

#if defined(IREE_PLATFORM_GENERIC)
iree_status_t iree_thread_create(iree_thread_entry_t entry, void *entry_arg,
	iree_thread_create_params_t params, iree_allocator_t allocator,
	iree_thread_t **out_thread) {
	(void)entry;
	(void)entry_arg;
	(void)params;
	(void)allocator;
	*out_thread = NULL;
	return iree_ok_status();
}
void iree_thread_release(iree_thread_t *thread) {
	(void)thread;
}
/* proactor_platform.c on bare-metal links these for its destroy/wait paths;
 * we never have a real proactor thread to yield/join, so they're no-ops. */
void iree_thread_yield(void) {}
void iree_thread_join(iree_thread_t *thread) {
	(void)thread;
}
#endif

// Symbols from gemmini_spike_vmfb_embed.S (.incbin)
extern const uint8_t _binary_gemmini_spike_vmfb_start[];
extern const uint8_t _binary_gemmini_spike_vmfb_end[];

#ifndef MATMUL_M
#define MATMUL_M 8
#endif
#ifndef MATMUL_N
#define MATMUL_N 8
#endif
#ifndef MATMUL_K
#define MATMUL_K 8
#endif
#ifndef MATMUL_FN
#define MATMUL_FN "module.matmul_8x8x8"
#endif
#define M MATMUL_M
#define N MATMUL_N
#define K MATMUL_K

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

	const iree_const_byte_span_t module_data =
		iree_make_const_byte_span(_binary_gemmini_spike_vmfb_start,
			(size_t)(_binary_gemmini_spike_vmfb_end -
				_binary_gemmini_spike_vmfb_start));
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

	iree_vm_function_t main_function;
	IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
		context, iree_make_cstring_view(MATMUL_FN), &main_function));

#ifdef MATMUL_SOAK_PRE
	/* 2026-05-19 STATE-POLLUTION PROBE: issue a few gemmini RoCC ops with
	 * non-default state (transposes, exotic scales, weird strides) BEFORE
	 * invoking the matmul. If the matmul's own CONFIG sequence fully
	 * resets all state, the result must still be correct. If state leaks
	 * across calls, this soak should reproduce the dronet-context bug
	 * even though the matmul itself is correct in isolation. */
	{
		/* CONFIG_EX with a_transpose=1, b_transpose=1, dataflow=WS,
		 * sys_act=RELU. funct7=0 (k_CONFIG), funct3=0, opcode=0x7b. */
		uint64_t rs1 = ((uint64_t)0x3F800000u << 32) /* acc_scale = 1.0f */
			| (1ULL << 16) /* a_stride = 1 */
			| (1ULL << 9) /* b_transpose = 1 */
			| (1ULL << 8) /* a_transpose = 1 */
			| (1ULL << 3) /* sys_act = RELU(1) */
			| (1ULL << 2) /* dataflow = WS */
			| 0ULL /* cmd_type = CONFIG_EX (00) */;
		uint64_t rs2 = (1ULL << 48); /* c_stride = 1, sys_shift = 0 */
		asm volatile(".insn r 0x7b, 0, 0, x0, %0, %1" : : "r"(rs1), "r"(rs2));
		/* CONFIG_LD slot 0: weird stride 0xDEAD, scale 1.0f. cmd_type
		 * = CONFIG_LD (01). */
		uint64_t ld_rs1 = 1ULL | (0ULL << 3); /* slot 0 */
		uint64_t ld_rs2 = 0xDEADULL | ((uint64_t)0x3F800000u << 32);
		asm volatile(".insn r 0x7b, 0, 0, x0, %0, %1"
					 :
					 : "r"(ld_rs1), "r"(ld_rs2));
		/* CONFIG_LD slot 1 with very different stride 0xBEEF. */
		ld_rs1 = 1ULL | (1ULL << 3);
		ld_rs2 = 0xBEEFULL;
		asm volatile(".insn r 0x7b, 0, 0, x0, %0, %1"
					 :
					 : "r"(ld_rs1), "r"(ld_rs2));
		/* CONFIG_LD slot 2 with stride 0xCAFE. */
		ld_rs1 = 1ULL | (2ULL << 3);
		ld_rs2 = 0xCAFEULL;
		asm volatile(".insn r 0x7b, 0, 0, x0, %0, %1"
					 :
					 : "r"(ld_rs1), "r"(ld_rs2));
		/* CONFIG_ST with weird stride 0xFEED and RELU activation. */
		uint64_t st_rs1 = 2ULL | (1ULL << 3); /* cmd=CONFIG_ST, sys_act=RELU */
		uint64_t st_rs2 = 0xFEEDULL;
		asm volatile(".insn r 0x7b, 0, 0, x0, %0, %1"
					 :
					 : "r"(st_rs1), "r"(st_rs2));
		fprintf(stdout,
			"[gemmini-spike] soak: issued 5 weird-state CONFIG_* ops\n");
		fflush(stdout);
	}
#endif

	// --- Build deterministic A and B ---
	// A = all 1, B = all 1 → matmul[i][j] = K = 8 for every cell. Simpler
	// reference than identity-B; libgemmini DIM=16 pads K=8→16 with zeros
	// which preserves the sum.
	int8_t A[M * K];
	int8_t B[K * N];
#ifdef MATMUL_INPUT_PATTERN
	/* 2026-05-19 input-dependent bug probe: instead of all-ones, fill A
	 * and B with a deterministic pattern that includes negatives, zeros,
	 * and sign-flips — the kind of values a real conv-stack activation
	 * produces. If gemmini produces a result that matches naive numpy
	 * matmul over this pattern, the matmul math is sign-correct and the
	 * dronet bug is somewhere ELSE (HW state pollution, alignment, etc.).
	 * If it diverges, we have a localized compiler/codegen bug.            */
	for (int i = 0; i < M * K; ++i) {
		int v = (i * 0x9E3779B1u) >> 24; /* deterministic byte stream */
		A[i] = (int8_t)v;
	}
	for (int i = 0; i < K * N; ++i) {
		int v = (i * 0xBB40E64Du + 0x9E37u) >> 24;
		B[i] = (int8_t)v;
	}
	/* Print first 8 bytes so we can compute the numpy reference offline. */
	fprintf(stdout, "[gemmini-spike] A[0..7]=");
	for (int i = 0; i < 8 && i < M * K; ++i)
		fprintf(stdout, " %d", (int)A[i]);
	fprintf(stdout, "\n[gemmini-spike] B[0..7]=");
	for (int i = 0; i < 8 && i < K * N; ++i)
		fprintf(stdout, " %d", (int)B[i]);
	fprintf(stdout, "\n");
#else
	for (int i = 0; i < M * K; ++i)
		A[i] = 1;
	for (int i = 0; i < K * N; ++i)
		B[i] = 1;
#endif

	iree_hal_dim_t shapeA[2] = {M, K};
#ifdef MATMUL_B_TRANSPOSED
	iree_hal_dim_t shapeB[2] = {N, K}; // N×K layout (transposed)
#else
	iree_hal_dim_t shapeB[2] = {K, N}; // K×N layout (standard)
#endif
	iree_hal_buffer_params_t params = {
		.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
		.usage = IREE_HAL_BUFFER_USAGE_DEFAULT};

	iree_hal_buffer_view_t *bvA = NULL;
	IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(device,
		iree_hal_device_allocator(device), 2, shapeA,
		IREE_HAL_ELEMENT_TYPE_SINT_8, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
		params, iree_make_const_byte_span(A, sizeof(A)), &bvA));
	iree_hal_buffer_view_t *bvB = NULL;
	IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(device,
		iree_hal_device_allocator(device), 2, shapeB,
		IREE_HAL_ELEMENT_TYPE_SINT_8, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
		params, iree_make_const_byte_span(B, sizeof(B)), &bvB));

	// async-external ABI: (wait_fence, signal_fence, ...inputs) → (...outputs).
	// Build an immediately-signalled wait fence and an empty signal fence we
	// can wait on after invocation.
	iree_hal_semaphore_t *sem_wait = NULL;
	IREE_RETURN_IF_ERROR(
		iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
			/*initial_value=*/1, IREE_HAL_SEMAPHORE_FLAG_NONE, &sem_wait));
	iree_hal_fence_t *wait_fence = NULL;
	IREE_RETURN_IF_ERROR(iree_hal_fence_create_at(
		sem_wait, /*value=*/1, iree_allocator_system(), &wait_fence));
	iree_hal_semaphore_release(sem_wait);

	iree_hal_semaphore_t *sem_signal = NULL;
	IREE_RETURN_IF_ERROR(
		iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
			/*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_NONE, &sem_signal));
	iree_hal_fence_t *signal_fence = NULL;
	IREE_RETURN_IF_ERROR(iree_hal_fence_create_at(
		sem_signal, /*value=*/1, iree_allocator_system(), &signal_fence));
	iree_hal_semaphore_release(sem_signal);

	// coarse-fences ABI order: (input0, input1, wait_fence, signal_fence).
	iree_vm_list_t *inputs = NULL;
	IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
		4, iree_allocator_system(), &inputs));
	iree_vm_ref_t refA = iree_hal_buffer_view_move_ref(bvA);
	iree_vm_ref_t refB = iree_hal_buffer_view_move_ref(bvB);
	iree_vm_ref_t refWait = iree_hal_fence_move_ref(wait_fence);
	iree_vm_ref_t refSignal = iree_hal_fence_move_ref(signal_fence);
	iree_vm_list_push_ref_move(inputs, &refA);
	iree_vm_list_push_ref_move(inputs, &refB);
	iree_vm_list_push_ref_move(inputs, &refWait);
	iree_vm_list_push_ref_move(inputs, &refSignal);

	iree_vm_list_t *outputs = NULL;
	IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
		1, iree_allocator_system(), &outputs));

	fprintf(stdout, "[gemmini-spike] invoking %s (M=%d N=%d K=%d)...\n",
		MATMUL_FN, M, N, K);
	fflush(stdout);

	IREE_RETURN_IF_ERROR(
		iree_vm_invoke(context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
			NULL, inputs, outputs, iree_allocator_system()));

	// Wait on the signal fence so the output buffer reflects the dispatch.
	IREE_RETURN_IF_ERROR(iree_hal_fence_wait(
		signal_fence, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
	iree_hal_fence_release(signal_fence);
	iree_hal_fence_release(wait_fence);

	iree_hal_buffer_view_t *ret_bv =
		iree_vm_list_get_buffer_view_assign(outputs, 0);
	int32_t res[M * N];
	IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(device,
		iree_hal_buffer_view_buffer(ret_bv), 0, res, sizeof(res),
		IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

	fprintf(stdout, "[gemmini-spike] result %dx%d (i32):\n", M, N);
	int errs = 0;
#ifdef MATMUL_INPUT_PATTERN
	/* Compute reference via the same A, B that the host filled. */
	int32_t *ref = (int32_t *)malloc(sizeof(int32_t) * M * N);
	if (!ref) {
		iree_status_free(
			iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "ref alloc"));
		return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "ref alloc");
	}
	for (int i = 0; i < M * N; ++i)
		ref[i] = 0;
	for (int m = 0; m < M; ++m) {
		for (int n = 0; n < N; ++n) {
			int32_t s = 0;
			for (int k = 0; k < K; ++k) {
				int32_t a = (int32_t)(int8_t)A[m * K + k];
				int32_t b = (int32_t)(int8_t)B[k * N + n];
				s += a * b;
			}
			ref[m * N + n] = s;
		}
	}
	fprintf(stdout,
		"[gemmini-spike] ref[0..3]= %ld %ld %ld %ld (first 4 cells of expected "
		"output)\n",
		(long)ref[0], M * N > 1 ? (long)ref[1] : 0L,
		M * N > 2 ? (long)ref[2] : 0L, M * N > 3 ? (long)ref[3] : 0L);
	fprintf(stdout, "[gemmini-spike] got[0..3]= %ld %ld %ld %ld\n",
		(long)res[0], M * N > 1 ? (long)res[1] : 0L,
		M * N > 2 ? (long)res[2] : 0L, M * N > 3 ? (long)res[3] : 0L);
	int mismatch_count = 0;
	for (int i = 0; i < M * N; ++i) {
		if (res[i] != ref[i])
			mismatch_count++;
	}
	free(ref);
	if (mismatch_count == 0) {
		fprintf(stdout, "[gemmini-spike] PASS (input-pattern variant)\n");
		errs = 0;
	} else {
		fprintf(stderr,
			"[gemmini-spike] FAIL: %d mismatches vs naive int8 matmul "
			"reference\n",
			mismatch_count);
		return iree_make_status(
			IREE_STATUS_UNKNOWN, "input-pattern matmul mismatch");
	}
	return iree_ok_status();
#endif
	// MATMUL_EXPECTED is K for the plain matmul fixture and K+1 for the
	// _bias variant (output prefilled with 1 before the matmul).
#ifndef MATMUL_EXPECTED
#define MATMUL_EXPECTED K
#endif
	const int32_t want = (int32_t)(MATMUL_EXPECTED);
	// 2026-05-18: per-row summary so we can see WHICH rows diverge.
	for (int i = 0; i < M; ++i) {
		int row_errs = 0;
		int32_t row_first = res[i * N + 0];
		int32_t row_min = row_first, row_max = row_first;
		int first_err_col = -1, last_err_col = -1;
		for (int j = 0; j < N; ++j) {
			int32_t got = res[i * N + j];
			if (got != want) {
				++row_errs;
				if (first_err_col < 0)
					first_err_col = j;
				last_err_col = j;
			}
			if (got < row_min)
				row_min = got;
			if (got > row_max)
				row_max = got;
		}
		errs += row_errs;
		if (row_errs > 0) {
			fprintf(stdout,
				"row %4d: errs_in_row=%d first_err_col=%d last_err_col=%d ", i,
				row_errs, first_err_col, last_err_col);
			// Print the wrong cells' values
			int printed = 0;
			for (int j = 0; j < N && printed < 8; ++j) {
				if (res[i * N + j] != want) {
					fprintf(stdout, "[%d]=%ld ", j, (long)res[i * N + j]);
					++printed;
				}
			}
			fprintf(stdout, "\n");
		} else {
			fprintf(stdout,
				"row %4d: first=%ld min=%ld max=%ld errs_in_row=0\n", i,
				(long)row_first, (long)row_min, (long)row_max);
		}
	}
	fflush(stdout);

	iree_vm_list_release(inputs);
	iree_vm_list_release(outputs);
	iree_hal_device_release(device);
	iree_vm_context_release(context);
	iree_vm_instance_release(instance);

	if (errs > 0) {
		fprintf(stderr, "[gemmini-spike] FAIL: %d mismatches\n", errs);
		return iree_make_status(IREE_STATUS_UNKNOWN, "verification failed");
	}
	fprintf(stdout, "[gemmini-spike] PASS\n");
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
