#ifndef MERLIN_IREE_BENCH_IREE_MODULE_UTILS_H_
#define MERLIN_IREE_BENCH_IREE_MODULE_UTILS_H_

#include <cstring>

#include "iree/runtime/api.h"

namespace merlin_bench {

// Pick the best entry function from a VMFB module.
//
// Preference order:
//   1. If there is exactly one exported function, use it.
//   2. Otherwise, prefer an export with signature (i32)->...
//   3. Fallback to ordinal 0.
//
// Returns arity (0 or 1) and whether the first argument is i32.
inline iree_status_t PickEntryFunction(iree_vm_module_t *module,
	iree_vm_function_t *out_fn, int *out_arity, bool *out_first_is_i32) {
	*out_fn = (iree_vm_function_t){0};
	*out_arity = 0;
	*out_first_is_i32 = false;

	const iree_vm_module_signature_t sig = iree_vm_module_signature(module);
	if (sig.export_function_count == 0) {
		return iree_make_status(IREE_STATUS_NOT_FOUND, "module has no exports");
	}

	auto compute_arity = [&](const iree_vm_function_t &fn, int *arity,
							 bool *first_i32) {
		const iree_vm_function_signature_t fsig =
			iree_vm_function_signature(&fn);
		const iree_string_view_t cc = fsig.calling_convention;
		if (cc.size == 0 || cc.data[0] != '0') {
			*arity = 0;
			*first_i32 = false;
			return;
		}
		const void *u = memchr(cc.data, '_', cc.size);
		const size_t upos = u
			? static_cast<size_t>(static_cast<const char *>(u) - cc.data)
			: static_cast<size_t>(cc.size);
		const size_t n_in = upos >= 1 ? (upos - 1) : 0;
		*arity = static_cast<int>(n_in);
		*first_i32 = (n_in >= 1 && cc.data[1] == 'i');
	};

	// Single export: use it directly.
	if (sig.export_function_count == 1) {
		IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
			module, IREE_VM_FUNCTION_LINKAGE_EXPORT, 0, out_fn));
		compute_arity(*out_fn, out_arity, out_first_is_i32);
		return iree_ok_status();
	}

	// Prefer: (i32)->... (arity==1 && first arg i32).
	for (iree_host_size_t i = 0; i < sig.export_function_count; ++i) {
		iree_vm_function_t fn = {0};
		IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
			module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &fn));
		int arity = 0;
		bool first_i32 = false;
		compute_arity(fn, &arity, &first_i32);
		if (arity == 1 && first_i32) {
			*out_fn = fn;
			*out_arity = arity;
			*out_first_is_i32 = first_i32;
			return iree_ok_status();
		}
	}

	// Fallback to ordinal 0.
	IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
		module, IREE_VM_FUNCTION_LINKAGE_EXPORT, 0, out_fn));
	compute_arity(*out_fn, out_arity, out_first_is_i32);
	return iree_ok_status();
}

} // namespace merlin_bench

#endif // MERLIN_IREE_BENCH_IREE_MODULE_UTILS_H_
