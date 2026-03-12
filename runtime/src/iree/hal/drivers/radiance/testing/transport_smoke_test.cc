// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../api.h"
#include "fake_transport.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(RadianceTransportTest, DirectSubmitSmoke) {
	iree_hal_radiance_device_options_t options;
	iree_hal_radiance_device_options_initialize(&options);
	options.backend = IREE_HAL_RADIANCE_TRANSPORT_BACKEND_DIRECT_SUBMIT;

	iree_hal_radiance_transport_t *transport = nullptr;
	IREE_ASSERT_OK(iree_hal_radiance_transport_create(
		&options, iree_allocator_system(), &transport));

	uint64_t device_addr = 0;
	IREE_ASSERT_OK(
		iree_hal_radiance_transport_alloc_device(transport, 128, &device_addr));
	EXPECT_NE(device_addr, 0u);

	IREE_ASSERT_OK(iree_hal_radiance_transport_submit_fill(transport,
		/*stream_id=*/0, device_addr, /*value=*/0xAB,
		/*length=*/128));
	IREE_ASSERT_OK(iree_hal_radiance_transport_submit_copy(transport,
		/*stream_id=*/0, device_addr, device_addr + 64, /*length=*/64,
		IREE_HAL_RADIANCE_COPY_DIRECTION_H2D));

	const uint32_t packed_params[] = {1, 2, 3, 4};
	iree_hal_radiance_launch_params_t launch_params = {};
	launch_params.stream_id = 0;
	launch_params.start_pc = 0x100;
	launch_params.kernel_pc = 0x140;
	launch_params.grid_x = 1;
	launch_params.grid_y = 1;
	launch_params.grid_z = 1;
	launch_params.block_x = 16;
	launch_params.block_y = 1;
	launch_params.block_z = 1;
	launch_params.regs_per_thread = 32;
	launch_params.shmem_per_block = 0;
	launch_params.params_data =
		iree_make_const_byte_span(packed_params, sizeof(packed_params));
	IREE_ASSERT_OK(
		iree_hal_radiance_transport_submit_dispatch(transport, &launch_params));
	IREE_ASSERT_OK(iree_hal_radiance_transport_synchronize(transport,
		/*stream_id=*/0));
	iree_hal_radiance_transport_destroy(transport);
}

TEST(RadianceTransportTest, FakeTransportRecordsCalls) {
	iree_hal_radiance_fake_transport_stats_t stats = {};
	iree_hal_radiance_transport_t *transport = nullptr;
	IREE_ASSERT_OK(iree_hal_radiance_fake_transport_create(
		&stats, iree_allocator_system(), &transport));

	uint64_t device_addr = 0;
	IREE_ASSERT_OK(
		iree_hal_radiance_transport_alloc_device(transport, 256, &device_addr));

	const uint8_t params[] = {9, 8, 7, 6};
	iree_hal_radiance_launch_params_t launch_params = {};
	launch_params.grid_x = 2;
	launch_params.grid_y = 1;
	launch_params.grid_z = 1;
	launch_params.block_x = 16;
	launch_params.block_y = 1;
	launch_params.block_z = 1;
	launch_params.params_data =
		iree_make_const_byte_span(params, sizeof(params));

	IREE_ASSERT_OK(iree_hal_radiance_transport_submit_copy(transport,
		/*stream_id=*/1, /*src=*/0x10, /*dst=*/device_addr,
		/*length=*/16, IREE_HAL_RADIANCE_COPY_DIRECTION_H2D));
	IREE_ASSERT_OK(iree_hal_radiance_transport_submit_fill(transport,
		/*stream_id=*/1, device_addr, /*value=*/0x11, /*length=*/32));
	IREE_ASSERT_OK(
		iree_hal_radiance_transport_submit_dispatch(transport, &launch_params));
	IREE_ASSERT_OK(iree_hal_radiance_transport_synchronize(transport,
		/*stream_id=*/1));

	EXPECT_EQ(stats.alloc_count, 1u);
	EXPECT_EQ(stats.copy_count, 1u);
	EXPECT_EQ(stats.fill_count, 1u);
	EXPECT_EQ(stats.dispatch_count, 1u);
	EXPECT_EQ(stats.sync_count, 1u);
	EXPECT_EQ(stats.last_dispatch_grid_x, 2u);
	EXPECT_EQ(stats.last_dispatch_block_x, 16u);
	EXPECT_EQ(stats.last_dispatch_param_bytes, sizeof(params));

	iree_hal_radiance_transport_destroy(transport);
}

TEST(RadianceTransportTest, DeviceCreateSmoke) {
	iree_hal_radiance_device_options_t options;
	iree_hal_radiance_device_options_initialize(&options);
	options.backend = IREE_HAL_RADIANCE_TRANSPORT_BACKEND_DIRECT_SUBMIT;

	iree_hal_device_t *device = nullptr;
	IREE_ASSERT_OK(iree_hal_radiance_device_create(
		IREE_SV("radiance"), &options, iree_allocator_system(), &device));
	ASSERT_NE(device, nullptr);
	iree_hal_device_release(device);
}

} // namespace
