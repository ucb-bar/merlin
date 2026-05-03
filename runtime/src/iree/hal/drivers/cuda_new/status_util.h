// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_STATUS_UTIL_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_STATUS_UTIL_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "dynamic_symbols.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define IREE_CURESULT_TO_STATUS_NEW(syms, expr, ...) \
	iree_hal_cuda_new_result_to_status(               \
		(syms), ((syms)->expr), __FILE__, __LINE__)

#define IREE_CUDA_NEW_RETURN_IF_ERROR(syms, expr, ...)                  \
	IREE_RETURN_IF_ERROR(iree_hal_cuda_new_result_to_status(            \
							 (syms), ((syms)->expr), __FILE__, __LINE__), \
		__VA_ARGS__)

#define IREE_CUDA_NEW_RETURN_AND_END_ZONE_IF_ERROR(                       \
	zone_id, syms, expr, ...)                                             \
	IREE_RETURN_AND_END_ZONE_IF_ERROR(                                    \
		zone_id,                                                          \
		iree_hal_cuda_new_result_to_status((syms), ((syms)->expr),        \
			__FILE__, __LINE__),                                          \
		__VA_ARGS__)

#define IREE_CUDA_NEW_IGNORE_ERROR(syms, expr)                         \
	IREE_IGNORE_ERROR(iree_hal_cuda_new_result_to_status(              \
		(syms), ((syms)->expr), __FILE__, __LINE__))

iree_status_t iree_hal_cuda_new_result_to_status(
	const iree_hal_cuda_new_dynamic_symbols_t *syms, CUresult result,
	const char *file, uint32_t line);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_NEW_STATUS_UTIL_H_
