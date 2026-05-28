// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_QNN_QNN_DRIVER_H_
#define IREE_HAL_DRIVERS_QNN_QNN_DRIVER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Backends exposed by the QNN driver. Each maps to one Qnn{Backend}.so
// dlopen'd at runtime + one iree_hal_device_t.
//
// HTP and HTA are both Hexagon NPUs but cover different chip generations:
//   HTP — Snapdragon 8 Gen 2+ (libQnnHtp.so)
//   HTA — Snapdragon 865 / QRB5165 (libQnnHta.so)
// On a given board only one of the two is supported by the SDK; both are
// exposed as separate device URIs ("qnn://htp" vs "qnn://hta") so the
// scheduler can route to whichever is present.
typedef enum {
	IREE_HAL_QNN_BACKEND_CPU = 0,
	IREE_HAL_QNN_BACKEND_GPU = 1,
	IREE_HAL_QNN_BACKEND_HTP = 2,
	IREE_HAL_QNN_BACKEND_DSP = 3,
	IREE_HAL_QNN_BACKEND_HTA = 4,
	IREE_HAL_QNN_BACKEND_COUNT = 5,
} iree_hal_qnn_backend_t;

// Driver options. The QNN SDK root is auto-detected from the
// QNN_SDK_ROOT env var (or the build-time default in CMake) but can be
// overridden per-driver-instance.
typedef struct iree_hal_qnn_driver_options_t {
	// Bitmask of enabled backends. Each bit corresponds to one
	// iree_hal_qnn_backend_t. When 0, all backends are probed.
	uint32_t enabled_backends_mask;
} iree_hal_qnn_driver_options_t;

IREE_API_EXPORT void iree_hal_qnn_driver_options_initialize(
	iree_hal_qnn_driver_options_t *out_options);

// Creates a QNN driver registered as `identifier` (e.g. "qnn"). Each
// device exposed by the driver wraps a single QNN backend; the device id
// encodes the backend choice (matching `iree_hal_qnn_backend_t`).
IREE_API_EXPORT iree_status_t iree_hal_qnn_driver_create(
	iree_string_view_t identifier, const iree_hal_qnn_driver_options_t *options,
	iree_allocator_t host_allocator, iree_hal_driver_t **out_driver);

// Lazily loads libQnnSystem.so once per driver and returns the cached
// QnnSystemInterface_t pointer (cast to void* to keep the public header
// QNN-SDK-free). Used by the executable-create path to enumerate graphs
// in a .qnn-ctx blob without re-dlopen'ing libQnnSystem each time. Thread
// safe; idempotent. |*out_interface| is NULL on error.
iree_status_t iree_hal_qnn_driver_get_system_interface(
	iree_hal_driver_t *driver, const void **out_interface);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // IREE_HAL_DRIVERS_QNN_QNN_DRIVER_H_
