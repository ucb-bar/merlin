// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_QNN_QNN_DEVICE_H_
#define IREE_HAL_DRIVERS_QNN_QNN_DEVICE_H_

#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/hal/api.h"

#include "qnn_driver.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Per-backend QNN HAL device. Wraps a dlopen'd libQnn{Backend}.so + the
// QnnInterface providers vtable that backend exports. The device owns:
//
//   - the dynamic-library handle (dlclose on destroy)
//   - the device_allocator (host heap; QNN buffers are CPU-accessible)
//   - the deferred command-buffer block pool
//   - the queue serialisation mutex
//
// Per-context QnnContextHandle_t lifetime is owned by iree_hal_qnn_executable_t
// (one context per loaded .qnn-ctx blob), not by the device.
//
// Created by iree_hal_qnn_driver_create_device_by_id once the dlopen + provider
// query have succeeded. |create_params| must carry a non-NULL
// proactor_pool — the device acquires one proactor from it for the device's
// semaphore creation path.
IREE_API_EXPORT iree_status_t iree_hal_qnn_device_create(
	iree_string_view_t identifier, iree_hal_qnn_backend_t backend,
	iree_dynamic_library_t *backend_lib, iree_hal_driver_t *parent_driver,
	const iree_hal_device_create_params_t *create_params,
	iree_allocator_t host_allocator, iree_hal_device_t **out_device);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // IREE_HAL_DRIVERS_QNN_QNN_DEVICE_H_
