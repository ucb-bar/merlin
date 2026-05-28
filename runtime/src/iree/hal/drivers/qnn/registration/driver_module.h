// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_QNN_REGISTRATION_DRIVER_MODULE_H_
#define IREE_HAL_DRIVERS_QNN_REGISTRATION_DRIVER_MODULE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers the QNN HAL driver under the identifier "qnn" with the given
// driver registry. Driver options are read from the QNN_BACKEND_LIB_DIR
// env var (or the build-time default) on first device-creation request.
IREE_API_EXPORT iree_status_t iree_hal_qnn_driver_module_register(
	iree_hal_driver_registry_t *registry);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // IREE_HAL_DRIVERS_QNN_REGISTRATION_DRIVER_MODULE_H_
