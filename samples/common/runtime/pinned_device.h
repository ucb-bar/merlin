/** @file pinned_device.h
 *  @brief Pinned local-task IREE device creation with per-device core affinity.
 *
 *  Creates a local-task device with a dedicated iree_task_executor_t pinned to
 *  specific CPU cores.  Each device gets its own worker thread pool, providing
 *  true core isolation between devices.
 *
 *  @note This bypasses iree_hal_driver_create_device_by_path because the
 *  local-task driver ignores the params argument (task_driver.c:157-169).
 *  Instead we build the driver directly via iree_hal_task_driver_create with
 *  a dedicated executor per device.
 *
 *  Requires IREE task-API internal headers.  When those headers are absent,
 *  CreatePinnedLocalTaskDevice() returns IREE_STATUS_UNAVAILABLE.
 */

#ifndef MERLIN_RUNTIME_PINNED_DEVICE_H_
#define MERLIN_RUNTIME_PINNED_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#if defined(__has_include)
#if __has_include("iree/task/api.h") &&                                        \
	__has_include("iree/task/topology.h") &&                                   \
		__has_include("iree/hal/drivers/local_task/task_driver.h") &&          \
			__has_include("iree/hal/local/loaders/registration/init.h")
#define MERLIN_HAS_PINNED_DEVICE 1
#include "iree/hal/drivers/local_task/task_driver.h"
#include "iree/hal/local/loaders/registration/init.h"
#include "iree/task/api.h"
#include "iree/task/topology.h"
#endif
#endif

#ifndef MERLIN_HAS_PINNED_DEVICE
#define MERLIN_HAS_PINNED_DEVICE 0
#endif

namespace merlin_bench {

/** @brief Create a local-task HAL device pinned to specific CPU cores.
 *
 *  The device identifier is set to "local" to match compiled VMFBs targeting
 *  `#hal.device.target<"local", ...>`.
 *
 *  @param host_allocator  Host allocator for all internal allocations.
 *  @param cpu_ids_csv     Comma-separated logical CPU IDs (e.g. "0,1,2,3").
 *  @param[out] out_device Receives the created device; caller owns it.
 *  @return OK on success, UNAVAILABLE when task-API headers are missing.
 */
inline iree_status_t CreatePinnedLocalTaskDevice(
	iree_allocator_t host_allocator, const char *cpu_ids_csv,
	iree_hal_device_t **out_device) {
#if MERLIN_HAS_PINNED_DEVICE
	*out_device = nullptr;

	// 1. Build topology from comma-separated CPU IDs.
	iree_task_topology_t topology;
	IREE_RETURN_IF_ERROR(
		iree_task_topology_initialize_from_logical_cpu_set_string(
			iree_make_cstring_view(cpu_ids_csv), &topology));

	// 2. Create task executor pinned to those cores.
	iree_task_executor_options_t exec_opts;
	iree_task_executor_options_initialize(&exec_opts);
	exec_opts.worker_local_memory_size = 64 * 1024;

	iree_task_executor_t *executor = nullptr;
	iree_status_t st = iree_task_executor_create(
		exec_opts, &topology, host_allocator, &executor);
	iree_task_topology_deinitialize(&topology);
	if (!iree_status_is_ok(st))
		return st;

	// 3. Create executable loaders.
	iree_hal_executable_loader_t *loaders[8] = {NULL};
	iree_host_size_t loader_count = 0;
	st = iree_hal_create_all_available_executable_loaders(
		/*plugin_manager=*/NULL, IREE_ARRAYSIZE(loaders), &loader_count,
		loaders, host_allocator);
	if (!iree_status_is_ok(st)) {
		iree_task_executor_release(executor);
		return st;
	}

	// 4. Create heap allocator for device buffers.
	iree_hal_allocator_t *device_allocator = NULL;
	st = iree_hal_allocator_create_heap(iree_make_cstring_view("local"),
		host_allocator, host_allocator, &device_allocator);
	if (!iree_status_is_ok(st)) {
		for (iree_host_size_t i = 0; i < loader_count; ++i)
			iree_hal_executable_loader_release(loaders[i]);
		iree_task_executor_release(executor);
		return st;
	}

	// 5. Create driver + device. Identifier "local" matches VMFB targets.
	iree_hal_task_device_params_t params;
	iree_hal_task_device_params_initialize(&params);

	iree_task_executor_t *executors[1] = {executor};
	iree_hal_driver_t *driver = nullptr;
	st = iree_hal_task_driver_create(iree_make_cstring_view("local"), &params,
		/*queue_count=*/1, executors, loader_count, loaders, device_allocator,
		host_allocator, &driver);

	iree_hal_allocator_release(device_allocator);
	for (iree_host_size_t i = 0; i < loader_count; ++i)
		iree_hal_executable_loader_release(loaders[i]);
	iree_task_executor_release(executor);

	if (!iree_status_is_ok(st))
		return st;

	st = iree_hal_driver_create_device_by_id(driver, IREE_HAL_DEVICE_ID_DEFAULT,
		/*param_count=*/0, /*params=*/nullptr, host_allocator, out_device);
	iree_hal_driver_release(driver);
	return st;
#else
	(void)host_allocator;
	(void)cpu_ids_csv;
	*out_device = nullptr;
	return iree_make_status(IREE_STATUS_UNAVAILABLE,
		"pinned device creation requires IREE task API headers");
#endif
}

} // namespace merlin_bench

#endif // MERLIN_RUNTIME_PINNED_DEVICE_H_
