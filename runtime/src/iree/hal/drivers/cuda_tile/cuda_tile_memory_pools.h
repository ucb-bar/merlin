// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Stub header for cuda_tile memory pools.
// The cuda_tile driver does not use async memory pools (simple cuMemAlloc
// path only). This header exists to satisfy the allocator's type references.
// See FUTURE.md for instructions on adding full memory pool support.

#ifndef IREE_HAL_DRIVERS_CUDA_TILE_MEMORY_POOLS_H_
#define IREE_HAL_DRIVERS_CUDA_TILE_MEMORY_POOLS_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Opaque forward declaration — pools are not used in the cuda_tile driver.
// The allocator accepts a NULL pools pointer and falls back to cuMemAlloc.
typedef struct iree_hal_cuda_tile_memory_pools_t
    iree_hal_cuda_tile_memory_pools_t;

// Stub: merges pool statistics into the allocator stats.
// No-op when pools is NULL.
static inline void iree_hal_cuda_tile_memory_pools_merge_statistics(
    iree_hal_cuda_tile_memory_pools_t* pools,
    iree_hal_allocator_statistics_t* out_statistics) {
  (void)pools;
  (void)out_statistics;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA_TILE_MEMORY_POOLS_H_
