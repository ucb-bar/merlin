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
#if __has_include("iree/async/util/proactor_pool.h")
#include "iree/async/util/proactor_pool.h"
#define MERLIN_HAS_PROACTOR_POOL 1
#endif
#endif
#if __has_include("iree/hal/drivers/local_sync/sync_driver.h") &&              \
	__has_include("iree/hal/local/loaders/registration/init.h")
#define MERLIN_HAS_SYNC_DEVICE 1
#include "iree/hal/drivers/local_sync/sync_driver.h"
#endif
#endif

#ifndef MERLIN_HAS_PINNED_DEVICE
#define MERLIN_HAS_PINNED_DEVICE 0
#endif

#ifndef MERLIN_HAS_SYNC_DEVICE
#define MERLIN_HAS_SYNC_DEVICE 0
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

#if MERLIN_HAS_PROACTOR_POOL
	// Newer IREE (post 2026 refactor) requires a proactor pool fed via
	// iree_hal_device_create_params_t. Mirrors the pattern in
	// iree/runtime/instance.c. The device retains its own ref to the
	// proactor pool, so we can release ours immediately after device
	// creation.
	iree_async_proactor_pool_t *proactor_pool = nullptr;
	st = iree_async_proactor_pool_create(iree_numa_node_count(),
		/*node_ids=*/nullptr, iree_async_proactor_pool_options_default(),
		host_allocator, &proactor_pool);
	if (!iree_status_is_ok(st)) {
		iree_hal_driver_release(driver);
		return st;
	}
	iree_hal_device_create_params_t create_params =
		iree_hal_device_create_params_default();
	create_params.proactor_pool = proactor_pool;
	st = iree_hal_driver_create_device_by_id(driver, IREE_HAL_DEVICE_ID_DEFAULT,
		/*param_count=*/0, /*params=*/nullptr, &create_params, host_allocator,
		out_device);
	iree_async_proactor_pool_release(proactor_pool);
#else
	// Legacy IREE pre-`iree_hal_device_create_params_t`: no params block.
	st = iree_hal_driver_create_device_by_id(driver, IREE_HAL_DEVICE_ID_DEFAULT,
		/*param_count=*/0, /*params=*/nullptr, host_allocator, out_device);
#endif
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

/** @brief Create a local-sync HAL device — executes dispatches inline
 *         on the calling thread, no task-pool round-trip per call.
 *
 *  For tiny per-dispatch work (mobilenet's 100-300us layers), the
 *  task-driver's submit + worker-pickup + fence-signal round-trip
 *  (~300-500us) dominates the dispatch wall. local-sync removes that
 *  overhead entirely — the iree_runtime_call_invoke executes the
 *  dispatch on the calling thread (the scheduler worker) and returns.
 *  Worker affinity to specific CPU cores happens in the worker thread
 *  itself (BestEffortPinCurrentThreadToCpuIds) — the device doesn't
 *  spawn its own threads.
 *
 *  Tradeoff: a single dispatch can't use multiple cores. For mobilenet
 *  layer compute that's fine — the per-layer work is small enough
 *  that single-threaded execution beats multi-thread setup.
 */
inline iree_status_t CreateSyncLocalDevice(
	iree_allocator_t host_allocator, iree_hal_device_t **out_device) {
#if MERLIN_HAS_SYNC_DEVICE
	*out_device = nullptr;

	iree_hal_executable_loader_t *loaders[8] = {nullptr};
	iree_host_size_t loader_count = 0;
	IREE_RETURN_IF_ERROR(iree_hal_create_all_available_executable_loaders(
		/*plugin_manager=*/nullptr, IREE_ARRAYSIZE(loaders), &loader_count,
		loaders, host_allocator));

	iree_hal_allocator_t *device_allocator = nullptr;
	iree_status_t st =
		iree_hal_allocator_create_heap(iree_make_cstring_view("local"),
			host_allocator, host_allocator, &device_allocator);
	if (!iree_status_is_ok(st)) {
		for (iree_host_size_t i = 0; i < loader_count; ++i)
			iree_hal_executable_loader_release(loaders[i]);
		return st;
	}

	iree_hal_sync_device_params_t params;
	iree_hal_sync_device_params_initialize(&params);

#if MERLIN_HAS_PROACTOR_POOL
	iree_async_proactor_pool_t *proactor_pool = nullptr;
	st = iree_async_proactor_pool_create(iree_numa_node_count(),
		/*node_ids=*/nullptr, iree_async_proactor_pool_options_default(),
		host_allocator, &proactor_pool);
	if (iree_status_is_ok(st)) {
		iree_hal_device_create_params_t create_params =
			iree_hal_device_create_params_default();
		create_params.proactor_pool = proactor_pool;
		st = iree_hal_sync_device_create(iree_make_cstring_view("local"),
			&params, &create_params, loader_count, loaders, device_allocator,
			host_allocator, out_device);
		iree_async_proactor_pool_release(proactor_pool);
	}
#else
	st = iree_hal_sync_device_create(iree_make_cstring_view("local"), &params,
		/*create_params=*/nullptr, loader_count, loaders, device_allocator,
		host_allocator, out_device);
#endif
	iree_hal_allocator_release(device_allocator);
	for (iree_host_size_t i = 0; i < loader_count; ++i)
		iree_hal_executable_loader_release(loaders[i]);
	return st;
#else
	(void)host_allocator;
	*out_device = nullptr;
	return iree_make_status(IREE_STATUS_UNAVAILABLE,
		"local-sync device creation requires IREE local_sync driver headers");
#endif
}

} // namespace merlin_bench

#endif // MERLIN_RUNTIME_PINNED_DEVICE_H_
