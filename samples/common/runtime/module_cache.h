/** @file module_cache.h
 *  @brief VMFB session caching and entry-function invocation helpers.
 */

#ifndef MERLIN_RUNTIME_MODULE_CACHE_H_
#define MERLIN_RUNTIME_MODULE_CACHE_H_

#include <mutex>
#include <string>

#include "iree/hal/api.h"
#include "iree/runtime/api.h"

#include "runtime/iree_module_utils.h"

namespace merlin_bench {

/** @brief Cached IREE runtime session for a single VMFB.
 *
 *  Holds the session handle, resolved entry function, and a mutex for
 *  thread-safe invocation via CallModule().
 */
struct CachedModule {
	std::string vmfb_path; /**< Path to the loaded VMFB. */
	iree_runtime_session_t *session = nullptr; /**< Owning session handle. */
	iree_vm_function_t entry_fn = {0}; /**< Resolved entry function. */
	int arity = 0; /**< Number of entry arguments. */
	bool first_is_i32 = false; /**< True if arg 0 is i32. */
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

/** @brief Load a VMFB into a new session and resolve its entry function.
 *
 *  Supports entry functions with 0 arguments or 1 i32 argument.
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

	if (!(out->arity == 0 || (out->arity == 1 && out->first_is_i32))) {
		return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
			"entry function arity=%d (first_i32=%d) not supported; "
			"supported: 0 args or 1 i32 arg",
			out->arity, out->first_is_i32 ? 1 : 0);
	}

	return iree_ok_status();
}

/** @brief Invoke a cached module's entry function (thread-safe).
 *
 *  Serializes access through CachedModule::mu.  Use CallModuleUnlocked()
 *  when the module is accessed from a single thread only.
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

/** @brief Invoke a cached module's entry function (no locking).
 *
 *  Same as CallModule() but without acquiring the mutex.  Only safe when
 *  the caller guarantees exclusive access.
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

} // namespace merlin_bench

#endif // MERLIN_RUNTIME_MODULE_CACHE_H_
