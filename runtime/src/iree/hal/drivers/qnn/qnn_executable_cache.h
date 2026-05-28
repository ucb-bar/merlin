// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_QNN_QNN_EXECUTABLE_CACHE_H_
#define IREE_HAL_DRIVERS_QNN_QNN_EXECUTABLE_CACHE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#include "qnn_driver.h"
#include "qnn_executable.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// No-op cache pattern — QNN context binaries are produced offline by
// qnn-context-binary-generator and we don't perform any on-device
// JIT/recompile. Each prepare_executable simply forwards the binary blob to
// iree_hal_qnn_executable_create. Mirrors amdgpu/executable_cache.{h,c}.
//
// |qnn_interface|, |backend_handle|, |device_handle| are borrowed from the
// device — must outlive the cache.
iree_status_t iree_hal_qnn_executable_cache_create(
	iree_string_view_t identifier, iree_hal_qnn_backend_t backend,
	iree_hal_qnn_interface_t qnn_interface,
	iree_hal_qnn_backend_handle_t backend_handle,
	iree_hal_qnn_device_handle_t device_handle, const void *system_interface,
	iree_allocator_t host_allocator,
	iree_hal_executable_cache_t **out_executable_cache);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_QNN_QNN_EXECUTABLE_CACHE_H_
