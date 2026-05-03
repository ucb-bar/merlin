// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_DYNAMIC_SYMBOLS_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_DYNAMIC_SYMBOLS_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct iree_hal_cuda_new_dynamic_symbols_t {
	int reserved;
} iree_hal_cuda_new_dynamic_symbols_t;

iree_status_t iree_hal_cuda_new_dynamic_symbols_initialize(
	iree_allocator_t host_allocator,
	iree_hal_cuda_new_dynamic_symbols_t *out_syms);

void iree_hal_cuda_new_dynamic_symbols_deinitialize(
	iree_hal_cuda_new_dynamic_symbols_t *syms);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_NEW_DYNAMIC_SYMBOLS_H_
