// samples/common/xpu-rt/multi_device_runner.cc
//
// Runs a single VMFB across N pinned local-task devices in one process.
//
// `iree-run-module --device=local-task:// --device=local-task://` cannot pin
// each device to a distinct CPU set: the local-task driver ignores per-URI
// params/topology (task_driver.c:158-171) and the global flag
// --task_topology_cpu_ids feeds a single executor pool. Multi-device with real
// per-cluster affinity therefore needs a custom runner.
//
// Each --cluster=name:cpu_ids flag instantiates one pinned device via
// CreatePinnedLocalTaskDevice (its own iree_task_executor_t pinned to the
// listed cores). Devices are added to an iree_hal_device_group_t in flag
// order; the schedule-applied VMFB's util.global !hal.device declarations
// resolve against this ordered group.
//
// I/O parsing mirrors iree-run-module by reusing iree_tooling_parse_variants
// / iree_tooling_print_variants.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/device_group.h"
#include "iree/modules/hal/module.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/function_io.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"

#include "runtime/pinned_device.h"

namespace {

struct ClusterSpec {
	std::string name;
	std::string cpu_ids;
};

struct Args {
	std::string module_path;
	std::string function_name;
	std::vector<ClusterSpec> clusters;
	std::vector<std::string> inputs;
	std::vector<std::string> outputs;
	std::string output_dump_prefix; // dump each output to <prefix>.<i>.bin
	int max_print_elements = 1024;
	int repeat = 1; // invoke N times back-to-back; report per-iter wall.
	int warmup = 0; // invoke W times before timing.
	bool trace_execution = false;
};

void PrintUsage(const char *argv0) {
	std::fprintf(stderr,
		"Usage: %s --module=<vmfb> --function=<name> "
		"--cluster=<name>:<cpu_ids_csv> [--cluster=...] "
		"[--input=<spec>...] [--output=<spec>...] [--output_dump=<prefix>] "
		"[--trace_execution]\n"
		"Example:\n"
		"  %s --module=dronet.q.int8.vmfb --function=main \\\n"
		"      --cluster=device_a:4,5,6,7 --cluster=device_b:0,1 \\\n"
		"      --input=1x200x200x1xi8=@input.bin --output_dump=out\n",
		argv0, argv0);
}

bool ParseArgs(int argc, char **argv, Args *out) {
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		auto starts_with = [&](const char *pfx) {
			return arg.compare(0, std::strlen(pfx), pfx) == 0;
		};
		if (starts_with("--module=")) {
			out->module_path = arg.substr(9);
		} else if (starts_with("--function=")) {
			out->function_name = arg.substr(11);
		} else if (starts_with("--cluster=")) {
			std::string body = arg.substr(10);
			auto colon = body.find(':');
			if (colon == std::string::npos) {
				std::fprintf(stderr, "--cluster=<name>:<cpu_ids> required\n");
				return false;
			}
			out->clusters.push_back(
				{body.substr(0, colon), body.substr(colon + 1)});
		} else if (starts_with("--input=")) {
			out->inputs.push_back(arg.substr(8));
		} else if (starts_with("--output=")) {
			out->outputs.push_back(arg.substr(9));
		} else if (starts_with("--output_dump=")) {
			out->output_dump_prefix = arg.substr(14);
		} else if (starts_with("--max_elements=")) {
			out->max_print_elements = std::atoi(arg.substr(15).c_str());
		} else if (starts_with("--repeat=")) {
			out->repeat = std::max(1, std::atoi(arg.substr(9).c_str()));
		} else if (starts_with("--warmup=")) {
			out->warmup = std::max(0, std::atoi(arg.substr(9).c_str()));
		} else if (arg == "--trace_execution") {
			out->trace_execution = true;
		} else if (arg == "--help" || arg == "-h") {
			PrintUsage(argv[0]);
			return false;
		} else {
			std::fprintf(stderr, "Unknown arg: %s\n", arg.c_str());
			PrintUsage(argv[0]);
			return false;
		}
	}
	if (out->module_path.empty() || out->function_name.empty() ||
		out->clusters.empty()) {
		PrintUsage(argv[0]);
		return false;
	}
	return true;
}

iree_status_t LoadVmfbModule(iree_vm_instance_t *instance,
	const std::string &path, iree_allocator_t host_allocator,
	iree_vm_module_t **out_module) {
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f) {
		return iree_make_status(
			IREE_STATUS_NOT_FOUND, "failed to open module: %s", path.c_str());
	}
	std::streamsize size = f.tellg();
	f.seekg(0, std::ios::beg);
	std::vector<uint8_t> bytes(size);
	if (!f.read(reinterpret_cast<char *>(bytes.data()), size)) {
		return iree_make_status(
			IREE_STATUS_DATA_LOSS, "failed to read module: %s", path.c_str());
	}
	// Copy into an allocator-owned buffer so the bytecode module owns it.
	void *owned = nullptr;
	IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, size, &owned));
	std::memcpy(owned, bytes.data(), size);
	iree_const_byte_span_t flatbuffer = iree_make_const_byte_span(owned, size);
	iree_allocator_t deleter = host_allocator;
	return iree_vm_bytecode_module_create(instance,
		static_cast<iree_vm_bytecode_module_flags_t>(0), flatbuffer, deleter,
		host_allocator, out_module);
}

iree_status_t DumpBufferView(iree_hal_buffer_view_t *view,
	iree_hal_device_t *transfer_device, const std::string &path,
	iree_allocator_t host_allocator) {
	iree_hal_buffer_t *buf = iree_hal_buffer_view_buffer(view);
	iree_device_size_t bytes = iree_hal_buffer_byte_length(buf);
	std::vector<uint8_t> host(bytes);
	IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(transfer_device, buf,
		/*source_offset=*/0, host.data(), bytes,
		IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
	std::ofstream f(path, std::ios::binary);
	if (!f) {
		return iree_make_status(
			IREE_STATUS_PERMISSION_DENIED, "cannot write %s", path.c_str());
	}
	f.write(reinterpret_cast<const char *>(host.data()), bytes);
	(void)host_allocator;
	return iree_ok_status();
}

iree_status_t Run(const Args &args, iree_allocator_t host_allocator) {
	// 1. VM instance + HAL types.
	iree_vm_instance_t *instance = nullptr;
	IREE_RETURN_IF_ERROR(
		iree_tooling_create_instance(host_allocator, &instance));

	// 2. Create one pinned device per cluster, in order.
	std::vector<iree_hal_device_t *> devices;
	devices.reserve(args.clusters.size());
	iree_status_t st = iree_ok_status();
	for (const auto &c : args.clusters) {
		iree_hal_device_t *d = nullptr;
		st = merlin_bench::CreatePinnedLocalTaskDevice(
			host_allocator, c.cpu_ids.c_str(), &d);
		if (!iree_status_is_ok(st)) {
			st =
				iree_status_annotate_f(st, "creating cluster '%s' on cpus '%s'",
					c.name.c_str(), c.cpu_ids.c_str());
			break;
		}
		std::fprintf(stderr, "[multi_device_runner] cluster %s pinned to %s\n",
			c.name.c_str(), c.cpu_ids.c_str());
		devices.push_back(d);
	}

	// 3. Build the device group.
	iree_hal_device_group_t *group = nullptr;
	if (iree_status_is_ok(st)) {
		iree_hal_device_group_builder_t builder;
		iree_hal_device_group_builder_initialize(&builder);
		for (auto *d : devices) {
			st = iree_hal_device_group_builder_add_device(&builder, d);
			if (!iree_status_is_ok(st))
				break;
		}
		if (iree_status_is_ok(st)) {
			st = iree_hal_device_group_builder_finalize(
				&builder, host_allocator, &group);
		}
	}

	// 4. HAL module wrapping the group.
	iree_vm_module_t *hal_module = nullptr;
	if (iree_status_is_ok(st)) {
		st = iree_hal_module_create(instance,
			iree_hal_module_device_policy_default(), group,
			IREE_HAL_MODULE_FLAG_NONE, iree_hal_module_debug_sink_stdio(stderr),
			host_allocator, &hal_module);
	}

	// 5. User VMFB module.
	iree_vm_module_t *user_module = nullptr;
	if (iree_status_is_ok(st)) {
		st = LoadVmfbModule(
			instance, args.module_path, host_allocator, &user_module);
	}

	// 6. VM context [hal_module, user_module].
	iree_vm_context_t *context = nullptr;
	if (iree_status_is_ok(st)) {
		iree_vm_module_t *mods[2] = {hal_module, user_module};
		st = iree_vm_context_create_with_modules(instance,
			IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(mods), mods,
			host_allocator, &context);
	}

	// 7. Locate the entry function and pull its argument-side cconv (used
	// by iree_tooling_parse_variants to decode --input specs).
	iree_vm_function_t function;
	if (iree_status_is_ok(st)) {
		st = iree_vm_module_lookup_function_by_name(user_module,
			IREE_VM_FUNCTION_LINKAGE_EXPORT,
			iree_make_cstring_view(args.function_name.c_str()), &function);
	}
	iree_string_view_t cconv = iree_string_view_empty();
	if (iree_status_is_ok(st)) {
		iree_vm_function_signature_t sig =
			iree_vm_function_signature(&function);
		// Both out_arguments and out_results must be non-null; out_results is
		// discarded but the impl dereferences it unconditionally.
		iree_string_view_t results_cconv = iree_string_view_empty();
		st = iree_vm_function_call_get_cconv_fragments(
			&sig, &cconv, &results_cconv);
	}

	// 8. Parse inputs onto the lead device.
	iree_hal_device_t *lead = devices.empty() ? nullptr : devices.front();
	iree_hal_allocator_t *lead_alloc =
		lead ? iree_hal_device_allocator(lead) : nullptr;
	iree_vm_list_t *inputs_list = nullptr;
	if (iree_status_is_ok(st)) {
		std::vector<iree_string_view_t> input_views;
		input_views.reserve(args.inputs.size());
		for (const auto &s : args.inputs) {
			input_views.push_back(iree_make_cstring_view(s.c_str()));
		}
		iree_string_view_list_t specs = {
			input_views.size(), input_views.data()};
		st = iree_tooling_parse_variants(
			cconv, specs, lead, lead_alloc, host_allocator, &inputs_list);
	}

	// 9. Invoke. With --warmup=W --repeat=N we run W warmup iterations
	// then time N iterations, reporting per-iteration wall clock.
	iree_vm_list_t *outputs_list = nullptr;
	if (iree_status_is_ok(st)) {
		st = iree_vm_list_create(iree_vm_make_undefined_type_def(),
			/*initial_capacity=*/8, host_allocator, &outputs_list);
	}
	for (int i = 0; iree_status_is_ok(st) && i < args.warmup; ++i) {
		iree_vm_list_clear(outputs_list);
		st = iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
			/*policy=*/nullptr, inputs_list, outputs_list, host_allocator);
	}
	using Clock = std::chrono::steady_clock;
	auto t0 = Clock::now();
	for (int i = 0; iree_status_is_ok(st) && i < args.repeat; ++i) {
		if (i > 0)
			iree_vm_list_clear(outputs_list);
		st = iree_vm_invoke(context, function, IREE_VM_INVOCATION_FLAG_NONE,
			/*policy=*/nullptr, inputs_list, outputs_list, host_allocator);
	}
	if (iree_status_is_ok(st) && args.repeat > 0) {
		auto dur_us = std::chrono::duration_cast<std::chrono::microseconds>(
			Clock::now() - t0)
						  .count();
		double per_iter_ms = static_cast<double>(dur_us) / 1000.0 / args.repeat;
		std::fprintf(stderr,
			"[multi_device_runner] %s: %d warmup + %d timed iters; "
			"per-iter %.3f ms (total %.3f ms)\n",
			args.function_name.c_str(), args.warmup, args.repeat, per_iter_ms,
			dur_us / 1000.0);
	}

	// 10. Print + (optionally) dump outputs.
	if (iree_status_is_ok(st) && outputs_list) {
		iree_string_builder_t sb;
		iree_string_builder_initialize(host_allocator, &sb);
		iree_status_t pst = iree_tooling_format_variants(
			IREE_SV("output"), outputs_list, args.max_print_elements, &sb);
		if (iree_status_is_ok(pst)) {
			std::fwrite(iree_string_builder_buffer(&sb), 1,
				iree_string_builder_size(&sb), stdout);
			std::fputc('\n', stdout);
		}
		iree_string_builder_deinitialize(&sb);

		if (!args.output_dump_prefix.empty()) {
			iree_host_size_t n = iree_vm_list_size(outputs_list);
			for (iree_host_size_t i = 0; i < n; ++i) {
				iree_vm_ref_t ref = iree_vm_ref_null();
				if (!iree_status_is_ok(
						iree_vm_list_get_ref_assign(outputs_list, i, &ref))) {
					continue;
				}
				if (iree_hal_buffer_view_isa(ref)) {
					iree_hal_buffer_view_t *bv =
						iree_hal_buffer_view_deref(ref);
					char path[512];
					std::snprintf(path, sizeof(path), "%s.%zu.bin",
						args.output_dump_prefix.c_str(), (size_t)i);
					iree_status_t dst =
						DumpBufferView(bv, lead, path, host_allocator);
					if (!iree_status_is_ok(dst)) {
						std::fprintf(stderr,
							"[multi_device_runner] dump %s failed\n", path);
						iree_status_free(dst);
					} else {
						std::fprintf(
							stderr, "[multi_device_runner] wrote %s\n", path);
					}
				}
			}
		}
	}

	// 11. Cleanup.
	iree_vm_list_release(outputs_list);
	iree_vm_list_release(inputs_list);
	iree_vm_context_release(context);
	iree_vm_module_release(user_module);
	iree_vm_module_release(hal_module);
	iree_hal_device_group_release(group);
	for (auto *d : devices)
		iree_hal_device_release(d);
	iree_vm_instance_release(instance);
	return st;
}

} // namespace

int main(int argc, char **argv) {
	Args args;
	if (!ParseArgs(argc, argv, &args))
		return 2;

	iree_allocator_t host_allocator = iree_allocator_system();
	iree_status_t st = Run(args, host_allocator);
	if (!iree_status_is_ok(st)) {
		iree_status_fprint(stderr, st);
		iree_status_free(st);
		return 1;
	}
	return 0;
}
