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
 *
 *  Supported entry signatures:
 *    arity 0                              — push nothing.
 *    arity 1, first arg is i32            — push the dispatch_iters i32.
 *    arity 1, first arg is hal.buffer_view — allocate zero-filled
 *        buffer view from `iree.abi.declaration`, push that. Used by
 *        QNN proxy modules whose entry signature is
 *        `run(tensor<1x299x299x3xf32>) -> tensor<...>`. The declaration
 *        attribute is parsed at LoadModule time; the cached buffer
 *        view is reused across invocations.
 */
struct CachedModule {
	std::string vmfb_path; /**< Path to the loaded VMFB. */
	iree_runtime_session_t *session = nullptr; /**< Owning session handle. */
	iree_vm_function_t entry_fn = {0}; /**< Resolved entry function. */
	int arity = 0; /**< Number of entry arguments. */
	bool first_is_i32 = false; /**< True if arg 0 is i32. */
	bool first_is_buffer_view = false; /**< True if arg 0 is a buffer view. */
	bool all_args_buffer_view =
		false; /**< True if all args are buffer_view (arity > 1). */
	bool coarse_fences =
		false; /**< True when ABI has hidden wait/signal fence refs. */
	int abi_input_arity =
		-1; /**< Logical input count from iree.abi.declaration. */
	iree_hal_buffer_view_t *input_view =
		nullptr; /**< Pre-allocated zero input for buffer-view entry. */
	iree_runtime_call_t call =
		{}; /**< Reused call object (input pushed once). */
	bool call_initialized = false; /**< True when |call| holds resources. */
	int32_t cached_i32_arg = 0; /**< Last pushed i32 value (avoid re-push). */
	std::mutex mu; /**< Guards session calls. */
};

/** @brief Release the IREE session owned by a CachedModule.
 *  @param m Module to release (may be NULL).
 */
inline void CachedModuleRelease(CachedModule *m) {
	if (!m)
		return;
	if (m->call_initialized) {
		iree_runtime_call_deinitialize(&m->call);
		m->call_initialized = false;
	}
	if (m->input_view) {
		iree_hal_buffer_view_release(m->input_view);
		m->input_view = nullptr;
	}
	if (m->session) {
		iree_runtime_session_release(m->session);
		m->session = nullptr;
	}
}

/** @brief Parse "tensor<NxMxKxdtype>" — shape + element type.
 *
 *  Reads from the start of `s` (which is expected to begin with
 *  "tensor<") and consumes through the matching '>'. Returns the
 *  number of dims read into `out_shape` (up to `max_rank`), the
 *  element type, and element size in bytes.
 *
 *  Returns true on success.
 */
inline bool ParseTensorTypeStr(iree_string_view_t s, iree_hal_dim_t *out_shape,
	int max_rank, int *out_rank, iree_hal_element_type_t *out_etype,
	int *out_elem_size) {
	const char *p = s.data;
	const char *end = s.data + s.size;
	const char *kPrefix = "tensor<";
	const size_t kPrefixLen = 7;
	if (s.size < kPrefixLen || strncmp(p, kPrefix, kPrefixLen) != 0)
		return false;
	p += kPrefixLen;
	int rank = 0;
	while (p < end && *p != '>') {
		// Read integer dim.
		char *digit_end = nullptr;
		long v = strtol(p, &digit_end, 10);
		if (digit_end == p)
			break;
		if (rank >= max_rank)
			return false;
		out_shape[rank++] = (iree_hal_dim_t)v;
		p = digit_end;
		if (p < end && *p == 'x')
			++p;
		else
			break;
	}
	// |p| now points at the dtype suffix, terminated by '>'.
	const char *dt_start = p;
	while (p < end && *p != '>')
		++p;
	std::string dtype(dt_start, (size_t)(p - dt_start));
	if (dtype == "f32") {
		*out_etype = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
		*out_elem_size = 4;
	} else if (dtype == "f16") {
		*out_etype = IREE_HAL_ELEMENT_TYPE_FLOAT_16;
		*out_elem_size = 2;
	} else if (dtype == "i32") {
		*out_etype = IREE_HAL_ELEMENT_TYPE_INT_32;
		*out_elem_size = 4;
	} else if (dtype == "i8") {
		*out_etype = IREE_HAL_ELEMENT_TYPE_INT_8;
		*out_elem_size = 1;
	} else if (dtype == "ui8") {
		*out_etype = IREE_HAL_ELEMENT_TYPE_UINT_8;
		*out_elem_size = 1;
	} else {
		return false;
	}
	*out_rank = rank;
	return true;
}

/** @brief Allocate a zero-filled buffer view for a tensor type string.
 *
 *  Used to synthesize an input for a buffer-view-arity entry function
 *  whose shape is read from `iree.abi.declaration`.
 */
inline iree_status_t MakeZeroInputBufferView(iree_hal_device_t *device,
	iree_string_view_t tensor_type_str, iree_hal_buffer_view_t **out_view) {
	enum { kMaxRank = 8 };
	iree_hal_dim_t shape[kMaxRank];
	int rank = 0;
	iree_hal_element_type_t etype = IREE_HAL_ELEMENT_TYPE_NONE;
	int elem_size = 0;
	if (!ParseTensorTypeStr(
			tensor_type_str, shape, kMaxRank, &rank, &etype, &elem_size)) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"could not parse tensor type from '%.*s'",
			(int)tensor_type_str.size, tensor_type_str.data);
	}
	iree_host_size_t num_elements = 1;
	for (int i = 0; i < rank; ++i)
		num_elements *= (iree_host_size_t)shape[i];
	const iree_host_size_t num_bytes =
		num_elements * (iree_host_size_t)elem_size;
	iree_hal_buffer_params_t buf_params = {};
	buf_params.type =
		IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
	buf_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
	buf_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
	buf_params.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
	iree_hal_buffer_t *buf = nullptr;
	IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
		iree_hal_device_allocator(device), buf_params, num_bytes, &buf));
	iree_hal_buffer_mapping_t mapping;
	iree_status_t st = iree_hal_buffer_map_range(buf,
		IREE_HAL_MAPPING_MODE_PERSISTENT, IREE_HAL_MEMORY_ACCESS_WRITE,
		/*offset=*/0, num_bytes, &mapping);
	if (iree_status_is_ok(st)) {
		memset(mapping.contents.data, 0, mapping.contents.data_length);
		iree_hal_buffer_unmap_range(&mapping);
	}
	if (iree_status_is_ok(st)) {
		iree_hal_dim_t hal_shape[kMaxRank];
		for (int i = 0; i < rank; ++i)
			hal_shape[i] = shape[i];
		st = iree_hal_buffer_view_create(buf, (iree_host_size_t)rank, hal_shape,
			etype, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
			iree_allocator_system(), out_view);
	}
	iree_hal_buffer_release(buf);
	return st;
}

/** @brief Extract the first tensor<...> substring from an
 *         iree.abi.declaration string.
 *
 *  The declaration looks like
 *      "sync func @run(%input0: tensor<1x299x299x3xf32>) -> ..."
 *  Returns the string view "tensor<1x299x299x3xf32>" or empty on
 *  failure.
 */
inline iree_string_view_t FindFirstTensorTypeStr(iree_string_view_t s) {
	const char *p = s.data;
	const char *end = s.data + s.size;
	const char *kPrefix = "tensor<";
	const size_t kPrefixLen = 7;
	while (p + kPrefixLen <= end) {
		if (strncmp(p, kPrefix, kPrefixLen) == 0) {
			// Find matching '>' at depth 1.
			const char *q = p + kPrefixLen;
			int depth = 1;
			while (q < end && depth > 0) {
				if (*q == '<')
					++depth;
				else if (*q == '>')
					--depth;
				++q;
			}
			if (depth == 0) {
				return iree_string_view_t{p, (iree_host_size_t)(q - p)};
			}
			return iree_string_view_t{nullptr, 0};
		}
		++p;
	}
	return iree_string_view_t{nullptr, 0};
}

/** @brief Count logical function inputs in an iree.abi.declaration string.
 *
 *  Coarse-fence ABI exports append hidden wait/signal fence refs to the raw VM
 *  function signature. The declaration keeps the user-visible input list:
 *      async func @foo(%input0: !hal.buffer_view, %input1: !hal.buffer_view) ->
 * () This parser counts top-level '%' argument markers inside the first
 *  parenthesized argument list without depending on symbol names.
 */
inline int CountAbiDeclarationInputs(iree_string_view_t s) {
	const char *p = s.data;
	const char *end = s.data + s.size;
	while (p < end && *p != '(')
		++p;
	if (p == end)
		return -1;
	++p;
	int depth = 1;
	int count = 0;
	bool at_arg_start = true;
	for (; p < end && depth > 0; ++p) {
		const char c = *p;
		if (c == '(') {
			++depth;
			at_arg_start = false;
		} else if (c == ')') {
			--depth;
			at_arg_start = false;
		} else if (depth == 1 && c == '%') {
			if (at_arg_start)
				++count;
			at_arg_start = false;
		} else if (depth == 1 && c == ',') {
			at_arg_start = true;
		} else if (depth == 1 && (c == ' ' || c == '\t' || c == '\n')) {
			// keep current at_arg_start state
		} else {
			at_arg_start = false;
		}
	}
	return depth == 0 ? count : -1;
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

	iree_string_view_t abi_model = iree_vm_function_lookup_attr_by_name(
		&out->entry_fn, iree_make_cstring_view("iree.abi.model"));
	out->coarse_fences = iree_string_view_equal(
		abi_model, iree_make_cstring_view("coarse-fences"));
	iree_string_view_t abi_decl = iree_vm_function_lookup_attr_by_name(
		&out->entry_fn, iree_make_cstring_view("iree.abi.declaration"));
	if (abi_decl.size != 0) {
		out->abi_input_arity = CountAbiDeclarationInputs(abi_decl);
	}

	// Detect arity-1 buffer-view input (e.g. QNN proxy modules whose
	// entry is `run(tensor<...>) -> tensor<...>`). Only the first arg
	// is supported; functions with a second argument are rejected.
	out->first_is_buffer_view = false;
	if (out->arity == 1 && !out->first_is_i32) {
		const iree_vm_function_signature_t fsig =
			iree_vm_function_signature(&out->entry_fn);
		const iree_string_view_t cc = fsig.calling_convention;
		// cconv example for buffer view: "0r_r" — first input is ref.
		if (cc.size >= 2 && cc.data[0] == '0' && cc.data[1] == 'r')
			out->first_is_buffer_view = true;
	}

	// Multi-input buffer_view detection: when arity > 1, check that every
	// arg is a buffer_view by inspecting the calling-convention string
	// (each input arg yields one char in cc.data[arity_start_idx ..]; 'r'
	// = ref/buffer_view, 'i'/'I' = primitive). Used by the data-flow
	// scheduler to drive per-dispatch wrappers emitted by
	// DumpExecutableDispatchModulesPass (each with N buffer_view args).
	out->all_args_buffer_view = false;
	if (out->arity > 1) {
		const iree_vm_function_signature_t fsig =
			iree_vm_function_signature(&out->entry_fn);
		const iree_string_view_t cc = fsig.calling_convention;
		// Format: "0<arg-types>_<ret-types>" where each char is 'r'
		// (ref), 'i'/'I' (i32/i64), 'f'/'F' (f32/f64), etc.
		bool all_ref = true;
		int arg_seen = 0;
		for (iree_host_size_t i = 1 /*skip leading '0'*/;
			 i < cc.size && cc.data[i] != '_'; ++i) {
			const char c = cc.data[i];
			if (c >= '0' && c <= '9')
				continue; // tuple-arity prefix
			if (c != 'r') {
				all_ref = false;
				break;
			}
			++arg_seen;
		}
		if (all_ref && arg_seen == out->arity) {
			out->all_args_buffer_view = true;
		}
	}

	if (out->arity == 0 || (out->arity == 1 && out->first_is_i32)) {
		// OK as-is.
	} else if (out->arity > 1 && out->all_args_buffer_view) {
		// Multi-input mode: caller (e.g., scheduler_runner.cc in
		// --data-flow-mode) is responsible for pushing each input
		// buffer_view before invoke. We skip auto-allocation here.
	} else if (out->coarse_fences && out->abi_input_arity >= 0 &&
		out->arity == out->abi_input_arity + 2) {
		// HAL wrapper exports compiled with coarse-fences have raw VM
		// signatures `(logical_inputs..., wait_fence, signal_fence)`.
		// Callers push the real inputs, then two null fence refs.
	} else if (out->arity == 1 && out->first_is_buffer_view) {
		// Read iree.abi.declaration to determine input shape, then
		// pre-allocate a zero-filled buffer view to push at call time.
		iree_string_view_t decl = iree_vm_function_lookup_attr_by_name(
			&out->entry_fn, iree_make_cstring_view("iree.abi.declaration"));
		if (decl.size == 0) {
			return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
				"buffer-view-arity entry function has no "
				"iree.abi.declaration metadata; cannot synthesize input "
				"(vmfb=%s)",
				vmfb_path.c_str());
		}
		iree_string_view_t tt = FindFirstTensorTypeStr(decl);
		if (tt.size == 0) {
			return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
				"could not extract tensor<...> from "
				"iree.abi.declaration='%.*s' (vmfb=%s)",
				(int)decl.size, decl.data, vmfb_path.c_str());
		}
		IREE_RETURN_IF_ERROR(
			MakeZeroInputBufferView(device, tt, &out->input_view));
	} else {
		return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
			"entry function arity=%d (first_i32=%d, first_bufview=%d) "
			"not supported; supported: 0 args, 1 i32 arg, 1 buffer-view "
			"arg",
			out->arity, out->first_is_i32 ? 1 : 0,
			out->first_is_buffer_view ? 1 : 0);
	}

	// Pre-initialize the call object once. Inputs/outputs lists are
	// allocated by initialize and reused across invocations, avoiding
	// per-call malloc on the dispatch hot path. Push the input now so
	// CallModule only resets outputs + invokes.
	IREE_RETURN_IF_ERROR(
		iree_runtime_call_initialize(out->session, out->entry_fn, &out->call));
	out->call_initialized = true;

	if (out->arity == 1 && out->first_is_i32) {
		// Push a default 1; CallModule overwrites only when changed.
		out->cached_i32_arg = 1;
		iree_vm_value_t v = iree_vm_value_make_i32(1);
		iree_status_t st =
			iree_vm_list_push_value(iree_runtime_call_inputs(&out->call), &v);
		if (!iree_status_is_ok(st)) {
			iree_runtime_call_deinitialize(&out->call);
			out->call_initialized = false;
			return st;
		}
	} else if (out->arity == 1 && out->first_is_buffer_view) {
		IREE_RETURN_IF_ERROR(iree_runtime_call_inputs_push_back_buffer_view(
			&out->call, out->input_view));
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
/** Invoke the cached call. Reuses the pre-initialized inputs (pushed
 *  once at LoadModule); only resets outputs + bumps the cached i32
 *  value if it changed. No malloc on the dispatch hot path.
 */
inline iree_status_t CallModule(
	CachedModule *m, int32_t dispatch_iters, iree_allocator_t host_alloc) {
	(void)host_alloc;
	std::lock_guard<std::mutex> lock(m->mu);

	if (m->arity == 1 && m->first_is_i32 &&
		m->cached_i32_arg != dispatch_iters) {
		iree_vm_value_t v = iree_vm_value_make_i32(dispatch_iters);
		IREE_RETURN_IF_ERROR(
			iree_vm_list_set_value(iree_runtime_call_inputs(&m->call), 0, &v));
		m->cached_i32_arg = dispatch_iters;
	}
	iree_vm_list_resize(iree_runtime_call_outputs(&m->call), 0);
	return iree_runtime_call_invoke(&m->call, /*flags=*/0);
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
	(void)host_alloc;
	if (m->arity == 1 && m->first_is_i32 &&
		m->cached_i32_arg != dispatch_iters) {
		iree_vm_value_t v = iree_vm_value_make_i32(dispatch_iters);
		IREE_RETURN_IF_ERROR(
			iree_vm_list_set_value(iree_runtime_call_inputs(&m->call), 0, &v));
		m->cached_i32_arg = dispatch_iters;
	}
	iree_vm_list_resize(iree_runtime_call_outputs(&m->call), 0);
	return iree_runtime_call_invoke(&m->call, /*flags=*/0);
}

} // namespace merlin_bench

#endif // MERLIN_RUNTIME_MODULE_CACHE_H_
