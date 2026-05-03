// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "physical_device.h"

#include <string.h>

iree_status_t iree_hal_cuda_new_physical_device_initialize(
	const iree_hal_cuda_new_dynamic_symbols_t *syms, int cuda_ordinal,
	iree_hal_cuda_new_physical_device_t *out_device) {
	(void)syms;
	(void)cuda_ordinal;
	(void)out_device;
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"cuda_new physical device query not yet implemented (spike 2)");
}

void iree_hal_cuda_new_physical_device_deinitialize(
	iree_hal_cuda_new_physical_device_t *device) {
	if (!device) return;
	memset(device, 0, sizeof(*device));
}
