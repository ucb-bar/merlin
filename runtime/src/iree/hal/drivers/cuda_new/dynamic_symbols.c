// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "dynamic_symbols.h"

#include <string.h>

iree_status_t iree_hal_cuda_new_dynamic_symbols_initialize(
	iree_allocator_t host_allocator,
	iree_hal_cuda_new_dynamic_symbols_t *out_syms) {
	(void)host_allocator;
	memset(out_syms, 0, sizeof(*out_syms));
	return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
		"cuda_new dynamic symbol loading not yet implemented");
}

void iree_hal_cuda_new_dynamic_symbols_deinitialize(
	iree_hal_cuda_new_dynamic_symbols_t *syms) {
	memset(syms, 0, sizeof(*syms));
}
