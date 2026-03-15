// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable_cache.h"

void iree_hal_radiance_executable_cache_initialize(
	iree_hal_radiance_executable_cache_t *out_cache) {
	memset(out_cache, 0, sizeof(*out_cache));
}

void iree_hal_radiance_executable_cache_store(
	iree_hal_radiance_executable_cache_t *cache, uint64_t executable_hash,
	const iree_hal_radiance_executable_t *executable) {
	IREE_ASSERT_ARGUMENT(cache);
	IREE_ASSERT_ARGUMENT(executable);
	cache->last_hash = executable_hash;
	cache->last_executable = *executable;
	cache->has_last_executable = true;
}

bool iree_hal_radiance_executable_cache_lookup(
	const iree_hal_radiance_executable_cache_t *cache, uint64_t executable_hash,
	iree_hal_radiance_executable_t *out_executable) {
	IREE_ASSERT_ARGUMENT(cache);
	IREE_ASSERT_ARGUMENT(out_executable);
	if (!cache->has_last_executable || cache->last_hash != executable_hash) {
		return false;
	}
	*out_executable = cache->last_executable;
	return true;
}
