/** @file module_cache.h
 *  @brief VMFB session caching and entry-function invocation helpers.
 *
 *  Supports both synchronous benchmark exports and async coarse-fence
 *  exports.  When the loaded entry function uses the "coarse-fences"
 *  ABI model, CallModuleAsync() submits work non-blockingly and returns
 *  a signal fence the caller can poll or wait on.
 */

#ifndef MERLIN_RUNTIME_MODULE_CACHE_H_
#define MERLIN_RUNTIME_MODULE_CACHE_H_

#include <mutex>
#include <string>

#include "iree/hal/api.h"
#include "iree/runtime/api.h"
#include "iree/tooling/function_util.h"

#include "runtime/iree_module_utils.h"

namespace merlin_bench {

/** @brief Cached IREE runtime session for a single VMFB.
 *
 *  Holds the session handle, resolved entry function, and metadata about
 *  the export's calling convention.
 */
struct CachedModule {
	std::string vmfb_path; /**< Path to the loaded VMFB. */
	iree_runtime_session_t *session = nullptr; /**< Owning session handle. */
	iree_vm_function_t entry_fn = {0}; /**< Resolved entry function. */
	int arity = 0; /**< Number of entry arguments. */
	bool first_is_i32 = false; /**< True if arg 0 is i32. */
	bool is_async = false; /**< True if entry uses coarse-fences model. */
	std::mutex mu; /**< Guards session calls. */
};

/** @brief Release the IREE session owned by a CachedModule.
 *  @param m Module to release (may be NULL).
 */
inline void CachedModuleRelease(CachedModule *m) {
	if (!m)
		return;
	if (m->session) {
		iree_runtime_session_release(m->session);
		m->session = nullptr;
	}
}

/** @brief Check whether a function uses the coarse-fences ABI model.
 *  @param fn Function to inspect.
 *  @return True if the function has iree.abi.model = "coarse-fences".
 */
inline bool IsCoarseFencesModel(const iree_vm_function_t &fn) {
	iree_string_view_t model =
		iree_vm_function_lookup_attr_by_name(&fn, IREE_SV("iree.abi.model"));
	return iree_string_view_equal(model, IREE_SV("coarse-fences"));
}

/** @brief Load a VMFB into a new session and resolve its entry function.
 *
 *  Auto-detects whether the export uses the coarse-fences ABI model
 *  (async dispatch with fences) or a synchronous calling convention.
 *
 *  @param instance  IREE runtime instance.
 *  @param device    HAL device to bind the session to.
 *  @param vmfb_path Filesystem path to the VMFB bytecode.
 *  @param[out] out  Populated on success; caller owns the session.
 *  @return OK on success.
 */
inline iree_status_t LoadModule(iree_runtime_instance_t *instance,
	iree_hal_device_t *device, const std::string &vmfb_path,
	CachedModule *out) {
	out->vmfb_path = vmfb_path;

	iree_runtime_session_options_t session_opts;
	iree_runtime_session_options_initialize(&session_opts);

	IREE_RETURN_IF_ERROR(
		iree_runtime_session_create_with_device(instance, &session_opts, device,
			iree_runtime_instance_host_allocator(instance), &out->session));

	IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_file(
		out->session, vmfb_path.c_str()));

	iree_vm_context_t *ctx = iree_runtime_session_context(out->session);
	const iree_host_size_t module_count = iree_vm_context_module_count(ctx);
	if (module_count == 0) {
		return iree_make_status(
			IREE_STATUS_FAILED_PRECONDITION, "session context had 0 modules");
	}
	iree_vm_module_t *module = iree_vm_context_module_at(ctx, module_count - 1);

	IREE_RETURN_IF_ERROR(PickEntryFunction(
		module, &out->entry_fn, &out->arity, &out->first_is_i32));

	out->is_async = IsCoarseFencesModel(out->entry_fn);

	// For sync exports: only 0-arg or 1-i32-arg is supported.
	// For async exports: the fence args are appended dynamically, so we
	// only check the "base" args (before fences).
	if (!out->is_async) {
		if (!(out->arity == 0 || (out->arity == 1 && out->first_is_i32))) {
			return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
				"sync entry function arity=%d (first_i32=%d) not supported",
				out->arity, out->first_is_i32 ? 1 : 0);
		}
	}

	return iree_ok_status();
}

/** @brief Invoke a cached module synchronously (no locking).
 *
 *  Only valid for modules where is_async == false.
 *
 *  @param m              Cached module to invoke.
 *  @param dispatch_iters Value passed as the i32 argument (ignored if
 * arity==0).
 *  @param host_alloc     Allocator for the transient input list.
 *  @return OK on success.
 */
inline iree_status_t CallModuleUnlocked(
	CachedModule *m, int32_t dispatch_iters, iree_allocator_t host_alloc) {
	iree_vm_list_t *inputs = nullptr;
	iree_status_t st = iree_vm_list_create(iree_vm_make_undefined_type_def(),
		static_cast<iree_host_size_t>(m->arity), host_alloc, &inputs);
	if (!iree_status_is_ok(st))
		return st;

	if (m->arity == 1) {
		iree_vm_value_t v = iree_vm_value_make_i32(dispatch_iters);
		st = iree_vm_list_push_value(inputs, &v);
		if (!iree_status_is_ok(st)) {
			iree_vm_list_release(inputs);
			return st;
		}
	}

	st = iree_runtime_session_call(m->session, &m->entry_fn, inputs,
		/*output_list=*/nullptr);
	iree_vm_list_release(inputs);
	return st;
}

/** @brief Invoke a cached module's entry function (thread-safe).
 *
 *  Serializes access through CachedModule::mu.
 *
 *  @param m              Cached module to invoke.
 *  @param dispatch_iters Value passed as the i32 argument (ignored if
 * arity==0).
 *  @param host_alloc     Allocator for the transient input list.
 *  @return OK on success.
 */
inline iree_status_t CallModule(
	CachedModule *m, int32_t dispatch_iters, iree_allocator_t host_alloc) {
	std::lock_guard<std::mutex> lock(m->mu);
	return CallModuleUnlocked(m, dispatch_iters, host_alloc);
}

/** @brief Submit a cached module asynchronously using coarse fences.
 *
 *  The call returns after submitting work to the device queue.  The
 *  caller must wait on *out_signal_fence before the results are ready.
 *
 *  Only valid for modules where is_async == true.
 *
 *  @param m                 Cached module to invoke.
 *  @param dispatch_iters    Value passed as the i32 argument (if applicable).
 *  @param device            HAL device for fence/semaphore creation.
 *  @param wait_fence        Fence to wait on before starting (NULL = start
 *                           immediately).
 *  @param host_alloc        Allocator.
 *  @param[out] out_signal_fence  Receives the signal fence; caller owns it.
 *  @return OK on success.
 */
inline iree_status_t CallModuleAsync(CachedModule *m, int32_t dispatch_iters,
	iree_hal_device_t *device, iree_hal_fence_t *wait_fence,
	iree_allocator_t host_alloc, iree_hal_fence_t **out_signal_fence) {
	*out_signal_fence = nullptr;

	// Build input list: optional i32 dispatch_iters, then fences appended.
	const iree_host_size_t base_arity =
		(m->first_is_i32 && m->arity >= 1) ? 1 : 0;
	iree_vm_list_t *inputs = nullptr;
	iree_status_t st = iree_vm_list_create(
		iree_vm_make_undefined_type_def(), base_arity + 2, host_alloc, &inputs);
	if (!iree_status_is_ok(st))
		return st;

	if (base_arity == 1) {
		iree_vm_value_t v = iree_vm_value_make_i32(dispatch_iters);
		st = iree_vm_list_push_value(inputs, &v);
		if (!iree_status_is_ok(st)) {
			iree_vm_list_release(inputs);
			return st;
		}
	}

	// Append (wait_fence, signal_fence) using IREE's tooling helper.
	st = iree_tooling_append_async_fences(
		inputs, m->entry_fn, device, wait_fence, out_signal_fence);
	if (!iree_status_is_ok(st)) {
		iree_vm_list_release(inputs);
		return st;
	}

	// Submit — returns after queueing, actual execution is async.
	st = iree_runtime_session_call(m->session, &m->entry_fn, inputs,
		/*output_list=*/nullptr);
	iree_vm_list_release(inputs);

	if (!iree_status_is_ok(st) && *out_signal_fence) {
		iree_hal_fence_release(*out_signal_fence);
		*out_signal_fence = nullptr;
	}
	return st;
}

} // namespace merlin_bench

#endif // MERLIN_RUNTIME_MODULE_CACHE_H_
