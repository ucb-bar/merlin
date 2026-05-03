// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "status_util.h"

#include <stddef.h>
#include <string.h>

#define IREE_CUDA_NEW_ERROR_LIST(IREE_CUDA_MAP_ERROR)                          \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_INVALID_VALUE",                            \
		IREE_STATUS_INVALID_ARGUMENT)                                          \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_OUT_OF_MEMORY",                            \
		IREE_STATUS_RESOURCE_EXHAUSTED)                                        \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_NOT_INITIALIZED", IREE_STATUS_INTERNAL)    \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_DEINITIALIZED", IREE_STATUS_INTERNAL)      \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_NO_DEVICE",                                \
		IREE_STATUS_FAILED_PRECONDITION)                                       \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_INVALID_DEVICE",                           \
		IREE_STATUS_FAILED_PRECONDITION)                                       \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_INVALID_IMAGE",                            \
		IREE_STATUS_FAILED_PRECONDITION)                                       \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_INVALID_CONTEXT", IREE_STATUS_INTERNAL)    \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_NO_BINARY_FOR_GPU",                        \
		IREE_STATUS_FAILED_PRECONDITION)                                       \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_NOT_FOUND", IREE_STATUS_NOT_FOUND)         \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_NOT_READY", IREE_STATUS_UNAVAILABLE)       \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_ILLEGAL_ADDRESS", IREE_STATUS_ABORTED)     \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",                  \
		IREE_STATUS_RESOURCE_EXHAUSTED)                                        \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_LAUNCH_TIMEOUT",                           \
		IREE_STATUS_DEADLINE_EXCEEDED)                                         \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_LAUNCH_FAILED", IREE_STATUS_ABORTED)       \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_NOT_PERMITTED",                            \
		IREE_STATUS_PERMISSION_DENIED)                                         \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_NOT_SUPPORTED",                            \
		IREE_STATUS_FAILED_PRECONDITION)                                       \
	IREE_CUDA_MAP_ERROR("CUDA_ERROR_UNKNOWN", IREE_STATUS_UNKNOWN)

static iree_status_code_t iree_hal_cuda_new_error_name_to_status_code(
	const char *error_name) {
#define IREE_CUDA_ERROR_TO_IREE_STATUS(cuda_error, iree_status) \
	if (strncmp(error_name, cuda_error, strlen(cuda_error)) == 0) { \
		return iree_status;                                         \
	}
	IREE_CUDA_NEW_ERROR_LIST(IREE_CUDA_ERROR_TO_IREE_STATUS)
#undef IREE_CUDA_ERROR_TO_IREE_STATUS
	return IREE_STATUS_UNKNOWN;
}

#undef IREE_CUDA_NEW_ERROR_LIST

iree_status_t iree_hal_cuda_new_result_to_status(
	const iree_hal_cuda_new_dynamic_symbols_t *syms, CUresult result,
	const char *file, uint32_t line) {
	if (IREE_LIKELY(result == CUDA_SUCCESS)) return iree_ok_status();

	const char *error_name = NULL;
	if (!syms || !syms->cuGetErrorName ||
		syms->cuGetErrorName(result, &error_name) != CUDA_SUCCESS) {
		error_name = "CUDA_ERROR_UNKNOWN";
	}

	const char *error_string = NULL;
	if (!syms || !syms->cuGetErrorString ||
		syms->cuGetErrorString(result, &error_string) != CUDA_SUCCESS) {
		error_string = "unknown error";
	}

	return iree_make_status_with_location(file, line,
		iree_hal_cuda_new_error_name_to_status_code(error_name),
		"CUDA error '%s' (%d): %s", error_name, result, error_string);
}
