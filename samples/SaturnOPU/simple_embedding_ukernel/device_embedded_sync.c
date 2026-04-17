// Copyright 2025 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <stdio.h>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/device.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/loaders/embedded_elf_loader.h"

// --- Dynamic Include ---
// CMake passes -DMODULE_HEADER="path/to/header.h"
#include MODULE_HEADER

iree_status_t create_sample_device(
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
		status = iree_async_proactor_pool_create(
			/*node_count=*/1, /*node_ids=*/NULL,
			iree_async_proactor_pool_options_default(), host_allocator,
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

// Bare-metal stub: thread_create/release are transitively linked but never
// called on bare-metal (IREE_ENABLE_THREADING=OFF). The proactor is handled
// by the IREE_PLATFORM_GENERIC case in proactor_platform.c.
#if defined(IREE_PLATFORM_GENERIC)
#include "iree/base/threading/thread.h"
iree_status_t iree_thread_create(iree_thread_entry_t entry, void *entry_arg,
	iree_thread_create_params_t params, iree_allocator_t allocator,
	iree_thread_t **out_thread) {
	(void)entry;
	(void)entry_arg;
	(void)params;
	*out_thread = NULL;
	return iree_ok_status();
}
void iree_thread_release(iree_thread_t *thread) {
	(void)thread;
}
#endif

// Helper to Load the Single Module for this Binary
const iree_const_byte_span_t load_bytecode_module_data() {
	// CMake passes -DMODULE_CREATE_FN=iree_samples_mm_..._create
	const struct iree_file_toc_t *toc = MODULE_CREATE_FN();
	return iree_make_const_byte_span(toc->data, toc->size);
}
