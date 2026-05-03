// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_STATUS_UTIL_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_STATUS_UTIL_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

iree_status_t iree_hal_cuda_new_result_to_status(int result, const char *file,
	uint32_t line);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_NEW_STATUS_UTIL_H_
