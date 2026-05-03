// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "status_util.h"

iree_status_t iree_hal_cuda_new_result_to_status(int result, const char *file,
	uint32_t line) {
	if (result == 0) return iree_ok_status();
	return iree_make_status_with_location(file, line, IREE_STATUS_INTERNAL,
		"CUDA error %d", result);
}
