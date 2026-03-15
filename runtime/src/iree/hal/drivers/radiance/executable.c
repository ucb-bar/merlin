// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "executable.h"

void iree_hal_radiance_executable_initialize(iree_const_byte_span_t image_data,
	uint64_t uploaded_device_address,
	iree_hal_radiance_executable_t *out_executable) {
	IREE_ASSERT_ARGUMENT(out_executable);
	memset(out_executable, 0, sizeof(*out_executable));
	out_executable->image_data = image_data;
	out_executable->uploaded_device_address = uploaded_device_address;
}
