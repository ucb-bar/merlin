// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_RADIANCE_EXECUTABLE_CACHE_H_
#define IREE_HAL_DRIVERS_RADIANCE_EXECUTABLE_CACHE_H_

#include <stdbool.h>
#include <stdint.h>

#include "executable.h"

typedef struct iree_hal_radiance_executable_cache_t {
	uint64_t last_hash;
	iree_hal_radiance_executable_t last_executable;
	bool has_last_executable;
} iree_hal_radiance_executable_cache_t;

void iree_hal_radiance_executable_cache_initialize(
	iree_hal_radiance_executable_cache_t *out_cache);

void iree_hal_radiance_executable_cache_store(
	iree_hal_radiance_executable_cache_t *cache, uint64_t executable_hash,
	const iree_hal_radiance_executable_t *executable);

bool iree_hal_radiance_executable_cache_lookup(
	const iree_hal_radiance_executable_cache_t *cache, uint64_t executable_hash,
	iree_hal_radiance_executable_t *out_executable);

#endif // IREE_HAL_DRIVERS_RADIANCE_EXECUTABLE_CACHE_H_
