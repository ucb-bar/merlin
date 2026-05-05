// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_NCCL_CHANNEL_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_NCCL_CHANNEL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/collective_batch.h"
#include "dynamic_symbols.h"
#include "nccl_dynamic_symbols.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// ncclUniqueId exposed without exporting the NCCL headers.
typedef struct {
	char data[128];
} iree_hal_cuda_new_nccl_id_t;

// Returns true if |id| is all zeros indicating an empty ID.
static inline bool iree_hal_cuda_new_nccl_id_is_empty(
	const iree_hal_cuda_new_nccl_id_t *id) {
	for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(id->data); ++i) {
		if (id->data[i] != 0) return false;
	}
	return true;
}

// Gets a unique ID to bootstrap a new communicator.
iree_status_t iree_hal_cuda_new_nccl_get_unique_id(
	const iree_hal_cuda_new_nccl_dynamic_symbols_t *symbols,
	iree_hal_cuda_new_nccl_id_t *out_id);

// Creates an IREE HAL channel using the given NCCL |id|, |rank|, and |count|.
iree_status_t iree_hal_cuda_new_nccl_channel_create(
	const iree_hal_cuda_new_dynamic_symbols_t *cuda_symbols,
	const iree_hal_cuda_new_nccl_dynamic_symbols_t *nccl_symbols,
	const iree_hal_cuda_new_nccl_id_t *id, int rank, int count,
	iree_allocator_t host_allocator, iree_hal_channel_t **out_channel);

// Performs a non-blocking submission of |batch| to |stream|.
// The backing storage of |batch| is dropped immediately but all resources
// referenced will be retained by the parent command buffer for its lifetime.
iree_status_t iree_hal_cuda_new_nccl_submit_batch(
	const iree_hal_cuda_new_nccl_dynamic_symbols_t *nccl_symbols,
	const iree_hal_collective_batch_t *batch, CUstream stream);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_NEW_NCCL_CHANNEL_H_
