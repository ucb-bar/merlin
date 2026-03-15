// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_RADIANCE_PACKET_LAYOUT_H_
#define IREE_HAL_DRIVERS_RADIANCE_PACKET_LAYOUT_H_

#include <stdint.h>

// Gluon command stream format (24-byte packet consumed by gluon-sim).
enum iree_hal_radiance_packet_offsets_e {
	IREE_HAL_RADIANCE_CMD_STREAM_ID_OFFSET = 0,
	IREE_HAL_RADIANCE_CMD_TYPE_OFFSET = 1,
	IREE_HAL_RADIANCE_CMD_PAYLOAD_OFFSET = 2,
	IREE_HAL_RADIANCE_CMD_PACKET_SIZE = 24,
};

typedef enum iree_hal_radiance_cmd_type_e {
	IREE_HAL_RADIANCE_CMD_TYPE_KERNEL = 0,
	IREE_HAL_RADIANCE_CMD_TYPE_MEM = 1,
	IREE_HAL_RADIANCE_CMD_TYPE_CSR = 2,
	IREE_HAL_RADIANCE_CMD_TYPE_WAIT = 3,
} iree_hal_radiance_cmd_type_t;

typedef enum iree_hal_radiance_mem_cmd_type_e {
	IREE_HAL_RADIANCE_MEM_CMD_COPY = 0,
	IREE_HAL_RADIANCE_MEM_CMD_SET = 1,
} iree_hal_radiance_mem_cmd_type_t;

// Kernel command payload (starts at packet byte offset 2).
enum iree_hal_radiance_kernel_packet_offsets_e {
	IREE_HAL_RADIANCE_KERNEL_HOST_ADDR_OFFSET =
		IREE_HAL_RADIANCE_CMD_PAYLOAD_OFFSET + 0,
	IREE_HAL_RADIANCE_KERNEL_SIZE_OFFSET =
		IREE_HAL_RADIANCE_CMD_PAYLOAD_OFFSET + 8,
	IREE_HAL_RADIANCE_KERNEL_GPU_ADDR_OFFSET =
		IREE_HAL_RADIANCE_CMD_PAYLOAD_OFFSET + 12,
};

// Mem command payload (starts at packet byte offset 2).
enum iree_hal_radiance_mem_packet_offsets_e {
	IREE_HAL_RADIANCE_MEM_SUBTYPE_OFFSET = IREE_HAL_RADIANCE_CMD_PAYLOAD_OFFSET,
	IREE_HAL_RADIANCE_MEM_COPY_SRC_OFFSET =
		IREE_HAL_RADIANCE_CMD_PAYLOAD_OFFSET + 1,
	IREE_HAL_RADIANCE_MEM_COPY_DST_OFFSET =
		IREE_HAL_RADIANCE_CMD_PAYLOAD_OFFSET + 9,
	IREE_HAL_RADIANCE_MEM_COPY_LEN_OFFSET =
		IREE_HAL_RADIANCE_CMD_PAYLOAD_OFFSET + 17,
	IREE_HAL_RADIANCE_MEM_COPY_FLAGS_OFFSET =
		IREE_HAL_RADIANCE_CMD_PAYLOAD_OFFSET + 21,
};

#endif // IREE_HAL_DRIVERS_RADIANCE_PACKET_LAYOUT_H_
