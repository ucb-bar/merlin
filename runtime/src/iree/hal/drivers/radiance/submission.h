// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_RADIANCE_SUBMISSION_H_
#define IREE_HAL_DRIVERS_RADIANCE_SUBMISSION_H_

#include "command_buffer.h"
#include "dispatch_builder.h"
#include "transport/transport.h"

iree_status_t iree_hal_radiance_submission_replay(
	const iree_hal_radiance_command_buffer_t *command_buffer,
	iree_hal_radiance_transport_t *transport);

#endif // IREE_HAL_DRIVERS_RADIANCE_SUBMISSION_H_
