// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_NEW_NCCL_DYNAMIC_SYMBOLS_H_
#define IREE_HAL_DRIVERS_CUDA_NEW_NCCL_DYNAMIC_SYMBOLS_H_

#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"
#include "dynamic_symbols.h"
#include "nccl_headers.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// NCCL API dynamic symbols for the cuda_new HAL driver.
typedef struct iree_hal_cuda_new_nccl_dynamic_symbols_t {
	iree_dynamic_library_t *dylib;

#define IREE_NCCL_PFN_DECL(ncclSymbolName, ...) \
	ncclResult_t (*ncclSymbolName)(__VA_ARGS__);
#define IREE_NCCL_PFN_DECL_STR_RETURN(ncclSymbolName, ...) \
	const char *(*ncclSymbolName)(__VA_ARGS__);
#include "nccl_dynamic_symbol_table.h" // IWYU pragma: export
#undef IREE_NCCL_PFN_DECL
#undef IREE_NCCL_PFN_DECL_STR_RETURN
} iree_hal_cuda_new_nccl_dynamic_symbols_t;

// Initializes |out_syms| in-place with dynamically loaded NCCL symbols.
// Returns UNAVAILABLE if libnccl.so cannot be found (non-fatal).
iree_status_t iree_hal_cuda_new_nccl_dynamic_symbols_initialize(
	iree_allocator_t host_allocator,
	const iree_hal_cuda_new_dynamic_symbols_t *cuda_library,
	iree_hal_cuda_new_nccl_dynamic_symbols_t *out_syms);

// Deinitializes |syms| by unloading the backing library.
void iree_hal_cuda_new_nccl_dynamic_symbols_deinitialize(
	iree_hal_cuda_new_nccl_dynamic_symbols_t *syms);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_CUDA_NEW_NCCL_DYNAMIC_SYMBOLS_H_
