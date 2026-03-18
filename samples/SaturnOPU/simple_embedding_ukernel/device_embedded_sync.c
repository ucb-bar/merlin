// Copyright 2025 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
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

	if (iree_status_is_ok(status)) {
		status =
			iree_hal_sync_device_create(identifier, &params, /*loader_count=*/1,
				&loader, device_allocator, host_allocator, out_device);
	}

	iree_hal_allocator_release(device_allocator);
	iree_hal_executable_loader_release(loader);
	return status;
}

// Helper to Load the Single Module for this Binary
const iree_const_byte_span_t load_bytecode_module_data() {
	// CMake passes -DMODULE_CREATE_FN=iree_samples_mm_..._create
	const struct iree_file_toc_t *toc = MODULE_CREATE_FN();
	return iree_make_const_byte_span(toc->data, toc->size);
}
