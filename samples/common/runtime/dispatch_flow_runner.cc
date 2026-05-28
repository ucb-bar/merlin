// dispatch_flow_runner: runs a heterogeneous schedule with REAL data
// flowing dispatch-to-dispatch across CPU, QNN_GPU, and QNN_HTA devices.
//
// To avoid pulling a JSON dependency into the runtime, the orchestrator
// (XPU-RT/scripts/heterogeneous_loop.py) preprocesses schedule.json +
// profiled_manifest.json into a single flat sidecar file consumed here:
//
//   FLAT FORMAT (--plan=<file>): one record per line, TAB-separated:
//     <op_name>\t<machine>\t<func>\t<vmfb>\t<size_0,size_1,...>\t<pred_0,pred_1,...>
//   * machine ∈ {CPU, GPU, HTA}
//   * vmfb is an absolute path on the BOARD (this binary runs on board)
//   * sizes is a CSV of per-binding byte counts
//   * preds is a CSV of dispatch_names (the predecessors); empty = source op
//
// Inputs:
//   --plan=<path>         flat schedule (above)
//   --input-from=<file>   raw bytes for the chain's first op's binding[0]
//   --output-to=<file>    write the LAST op's first output as raw bytes
//   --trace-csv=<file>    per-op timing trace (header included)
//   --iterations=N        timed iterations after warmup (default 1)
//   --warmup=N            warmup iterations (default 0)

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"

#include "runtime/module_cache.h"
#include "xpu-rt/buffer_registry.h"

using Clock = std::chrono::steady_clock;
using merlin_bench::CachedModule;
using merlin_bench::CachedModuleRelease;
using merlin_bench::LoadModule;

namespace {

struct PlanOp {
	std::string name;
	std::string machine; // CPU | GPU | HTA
	std::string func; // module.<func> (entry function name)
	std::string vmfb; // path on this machine (the board)
	std::vector<int64_t> binding_byte_sizes;
	std::vector<std::string> predecessors;
	std::vector<std::string> binding_sources;
};

struct Options {
	std::string plan_path;
	std::string input_from;
	std::string output_to;
	std::string capture_dispatch_io_dir;
	std::vector<std::string> capture_dispatches;
	std::string trace_csv;
	int iterations = 1;
	int warmup = 0;
	bool verbose = false;
	bool strict_binding_sources = false;
};

const std::map<std::string, std::string> kMachineToUri = {
	{"CPU", "local-task"},
	{"GPU", "qnn://gpu"},
	{"HTA", "qnn://hta"},
};

void Usage(const char *argv0) {
	fprintf(stderr,
		"Usage: %s --plan=<flat-schedule> [--input-from=<file>] "
		"[--output-to=<file>] [--capture-dispatch-io-dir=<dir>] "
		"[--capture-dispatches=a,b,c] [--trace-csv=<file>] "
		"[--iterations=N] [--warmup=N] "
		"[--strict-binding-sources] [--verbose]\n",
		argv0);
}

std::vector<std::string> SplitCsv(const std::string &s);

bool ParseArgs(int argc, char **argv, Options *o) {
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		auto eq = a.find('=');
		auto key = (eq == std::string::npos) ? a : a.substr(0, eq);
		auto val = (eq == std::string::npos) ? std::string{} : a.substr(eq + 1);
		if (key == "--plan")
			o->plan_path = val;
		else if (key == "--input-from")
			o->input_from = val;
		else if (key == "--output-to")
			o->output_to = val;
		else if (key == "--capture-dispatch-io-dir") {
			o->capture_dispatch_io_dir = val;
		} else if (key == "--capture-dispatches") {
			o->capture_dispatches = SplitCsv(val);
		} else if (key == "--trace-csv")
			o->trace_csv = val;
		else if (key == "--iterations")
			o->iterations = std::atoi(val.c_str());
		else if (key == "--warmup")
			o->warmup = std::atoi(val.c_str());
		else if (a == "--strict-binding-sources") {
			o->strict_binding_sources = true;
		} else if (a == "--verbose")
			o->verbose = true;
		else if (a == "--help" || a == "-h") {
			Usage(argv[0]);
			return false;
		} else {
			fprintf(stderr, "unknown arg: %s\n", argv[i]);
			Usage(argv[0]);
			return false;
		}
	}
	if (o->plan_path.empty()) {
		Usage(argv[0]);
		return false;
	}
	return true;
}

std::vector<std::string> SplitCsv(const std::string &s) {
	std::vector<std::string> out;
	std::string cur;
	for (char c : s) {
		if (c == ',') {
			if (!cur.empty())
				out.push_back(cur);
			cur.clear();
		} else
			cur.push_back(c);
	}
	if (!cur.empty())
		out.push_back(cur);
	return out;
}

std::vector<std::string> SplitSemi(const std::string &s) {
	std::vector<std::string> out;
	std::string cur;
	for (char c : s) {
		if (c == ';') {
			out.push_back(cur);
			cur.clear();
		} else
			cur.push_back(c);
	}
	out.push_back(cur);
	return out;
}

bool ShouldCaptureDispatch(const Options &opts, const std::string &name) {
	if (opts.capture_dispatches.empty())
		return true;
	return std::find(opts.capture_dispatches.begin(),
			   opts.capture_dispatches.end(),
			   name) != opts.capture_dispatches.end();
}

std::vector<PlanOp> ParsePlan(const std::string &path) {
	std::vector<PlanOp> out;
	std::ifstream f(path);
	std::string line;
	while (std::getline(f, line)) {
		if (line.empty() || line[0] == '#')
			continue;
		std::vector<std::string> fields;
		std::string cur;
		for (char c : line) {
			if (c == '\t') {
				fields.push_back(cur);
				cur.clear();
			} else
				cur.push_back(c);
		}
		fields.push_back(cur);
		if (fields.size() < 5)
			continue;
		PlanOp op;
		op.name = fields[0];
		op.machine = fields[1];
		op.func = fields[2];
		op.vmfb = fields[3];
		for (auto &s : SplitCsv(fields[4])) {
			op.binding_byte_sizes.push_back(std::atoll(s.c_str()));
		}
		if (fields.size() >= 6)
			op.predecessors = SplitCsv(fields[5]);
		if (fields.size() >= 7)
			op.binding_sources = SplitSemi(fields[6]);
		out.push_back(std::move(op));
	}
	return out;
}

iree_status_t MakeDevice(iree_runtime_instance_t *instance,
	const std::string &uri, iree_hal_device_t **out_device) {
	iree_string_view_t uri_v = iree_make_cstring_view(uri.c_str());
	iree_string_view_t driver_name, rest;
	iree_string_view_split(uri_v, ':', &driver_name, &rest);
	if (iree_string_view_is_empty(rest)) {
		return iree_runtime_instance_try_create_default_device(
			instance, uri_v, out_device);
	}
	iree_string_view_t device_path = rest;
	if (device_path.size >= 2 && device_path.data[0] == '/' &&
		device_path.data[1] == '/') {
		device_path =
			iree_make_string_view(device_path.data + 2, device_path.size - 2);
	}
	iree_hal_driver_registry_t *reg =
		iree_runtime_instance_driver_registry(instance);
	iree_hal_driver_t *driver = NULL;
	IREE_RETURN_IF_ERROR(iree_hal_driver_registry_try_create(
		reg, driver_name, iree_allocator_system(), &driver));
	iree_async_proactor_pool_t *proactor_pool = NULL;
	iree_status_t s = iree_async_proactor_pool_create(iree_numa_node_count(),
		NULL, iree_async_proactor_pool_options_default(),
		iree_allocator_system(), &proactor_pool);
	if (iree_status_is_ok(s)) {
		iree_hal_device_create_params_t cp =
			iree_hal_device_create_params_default();
		cp.proactor_pool = proactor_pool;
		s = iree_hal_driver_create_device_by_path(driver, driver_name,
			device_path, 0, NULL, &cp, iree_allocator_system(), out_device);
	}
	iree_async_proactor_pool_release(proactor_pool);
	iree_hal_driver_release(driver);
	return s;
}

iree_status_t MakeBufferView(iree_hal_device_t *device, int64_t bytes,
	const uint8_t *fill_data, iree_hal_buffer_view_t **out_view) {
	iree_hal_buffer_params_t bp = {};
	bp.type =
		IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
	bp.access = IREE_HAL_MEMORY_ACCESS_ALL;
	bp.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
	bp.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
	iree_hal_buffer_t *buf = NULL;
	IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
		iree_hal_device_allocator(device), bp, bytes, &buf));
	iree_hal_buffer_mapping_t mapping;
	iree_status_t s =
		iree_hal_buffer_map_range(buf, IREE_HAL_MAPPING_MODE_PERSISTENT,
			IREE_HAL_MEMORY_ACCESS_WRITE, 0, bytes, &mapping);
	if (iree_status_is_ok(s)) {
		if (fill_data)
			std::memcpy(mapping.contents.data, fill_data, (size_t)bytes);
		else
			std::memset(mapping.contents.data, 0, (size_t)bytes);
		iree_hal_buffer_unmap_range(&mapping);
	}
	if (iree_status_is_ok(s)) {
		iree_hal_dim_t dim = (iree_hal_dim_t)bytes;
		s = iree_hal_buffer_view_create(buf, /*rank=*/1, &dim,
			IREE_HAL_ELEMENT_TYPE_INT_8, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
			iree_allocator_system(), out_view);
	}
	iree_hal_buffer_release(buf);
	return s;
}

iree_status_t ReadExactFile(
	const std::string &path, int64_t want_bytes, std::vector<uint8_t> *out) {
	std::string file_path = path;
	int64_t offset = 0;
	size_t at = path.rfind('@');
	if (at != std::string::npos) {
		file_path = path.substr(0, at);
		offset =
			(int64_t)std::strtoll(path.substr(at + 1).c_str(), nullptr, 10);
		if (offset < 0) {
			return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
				"negative source file offset in %s", path.c_str());
		}
	}
	std::ifstream f(file_path, std::ios::binary | std::ios::ate);
	if (!f) {
		return iree_make_status(IREE_STATUS_NOT_FOUND,
			"cannot open source file %s", file_path.c_str());
	}
	std::streamsize n = f.tellg();
	if (n < offset + want_bytes) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"source file %s has %lld bytes; need [%lld, %lld)",
			file_path.c_str(), (long long)n, (long long)offset,
			(long long)(offset + want_bytes));
	}
	f.seekg(offset);
	out->resize((size_t)want_bytes);
	f.read(reinterpret_cast<char *>(out->data()), want_bytes);
	if (!f) {
		return iree_make_status(IREE_STATUS_DATA_LOSS,
			"short read from source file %s", file_path.c_str());
	}
	return iree_ok_status();
}

iree_status_t BufferViewToHost(
	iree_hal_buffer_view_t *view, std::vector<uint8_t> *out) {
	iree_hal_buffer_t *buf = iree_hal_buffer_view_buffer(view);
	iree_device_size_t bytes = iree_hal_buffer_byte_length(buf);
	out->resize((size_t)bytes);
	iree_hal_buffer_mapping_t mapping;
	IREE_RETURN_IF_ERROR(
		iree_hal_buffer_map_range(buf, IREE_HAL_MAPPING_MODE_SCOPED,
			IREE_HAL_MEMORY_ACCESS_READ, 0, bytes, &mapping));
	std::memcpy(out->data(), mapping.contents.data, (size_t)bytes);
	iree_hal_buffer_unmap_range(&mapping);
	return iree_ok_status();
}

// Cross-device: read predecessor view to host bytes, allocate on dst, write.
iree_status_t TransferAcrossDevices(iree_hal_buffer_view_t *src_view,
	iree_hal_device_t *dst_device, iree_hal_buffer_view_t **out_view,
	double *out_us) {
	const auto t0 = Clock::now();
	std::vector<uint8_t> staging;
	IREE_RETURN_IF_ERROR(BufferViewToHost(src_view, &staging));

	iree_status_t s = MakeBufferView(
		dst_device, (int64_t)staging.size(), staging.data(), out_view);
	const auto t1 = Clock::now();
	*out_us =
		std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() /
		1e3;
	return s;
}

iree_status_t ResolvePredSourceToHost(const BufferRegistry &registry,
	const std::string &source, std::vector<uint8_t> *out) {
	if (source.rfind("pred:", 0) != 0) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"concat source must be pred:<dispatch>:<index>, got %s",
			source.c_str());
	}
	std::string rest = source.substr(5);
	size_t colon = rest.rfind(':');
	std::string pred_name = rest;
	size_t out_index = 0;
	if (colon != std::string::npos) {
		pred_name = rest.substr(0, colon);
		out_index =
			(size_t)std::strtoull(rest.substr(colon + 1).c_str(), nullptr, 10);
	}
	const DispatchOutputs *po = registry.Get(pred_name);
	if (!po || out_index >= po->views.size()) {
		return iree_make_status(IREE_STATUS_NOT_FOUND,
			"missing predecessor output for concat source %s", source.c_str());
	}
	return BufferViewToHost(po->views[out_index], out);
}

std::string SanitizePathComponent(const std::string &name) {
	std::string out;
	out.reserve(name.size());
	for (unsigned char c : name) {
		if (std::isalnum(c) || c == '.' || c == '_' || c == '-' || c == '$') {
			out.push_back(static_cast<char>(c));
		} else {
			out.push_back('_');
		}
	}
	return out.empty() ? "unnamed_dispatch" : out;
}

iree_status_t DumpBufferViewToFile(iree_hal_device_t *device,
	iree_hal_buffer_view_t *view, const std::string &path) {
	iree_hal_buffer_t *buf = iree_hal_buffer_view_buffer(view);
	iree_device_size_t bytes = iree_hal_buffer_byte_length(buf);
	std::vector<uint8_t> host((size_t)bytes);
	IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(device, buf,
		/*source_offset=*/0, host.data(), bytes,
		IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
	std::ofstream f(path, std::ios::binary);
	if (!f) {
		return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
			"cannot open %s for write", path.c_str());
	}
	f.write(reinterpret_cast<const char *>(host.data()),
		static_cast<std::streamsize>(host.size()));
	if (!f) {
		return iree_make_status(
			IREE_STATUS_DATA_LOSS, "failed writing %s", path.c_str());
	}
	return iree_ok_status();
}

iree_status_t CaptureDispatchIo(const std::string &root, const PlanOp &op,
	iree_hal_device_t *device,
	const std::vector<iree_hal_buffer_view_t *> &inputs,
	const std::vector<iree_hal_buffer_view_t *> &outputs) {
	namespace fs = std::filesystem;
	const fs::path op_dir = fs::path(root) / SanitizePathComponent(op.name);
	std::error_code ec;
	fs::create_directories(op_dir, ec);
	if (ec) {
		return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
			"failed creating capture dir %s: %s", op_dir.c_str(),
			ec.message().c_str());
	}

	for (size_t i = 0; i < inputs.size(); ++i) {
		IREE_RETURN_IF_ERROR(DumpBufferViewToFile(device, inputs[i],
			(op_dir / ("input_" + std::to_string(i) + ".bin")).string()));
	}
	for (size_t i = 0; i < outputs.size(); ++i) {
		IREE_RETURN_IF_ERROR(DumpBufferViewToFile(device, outputs[i],
			(op_dir / ("output_" + std::to_string(i) + ".bin")).string()));
	}

	std::ofstream meta((op_dir / "meta.tsv").string(), std::ios::trunc);
	if (!meta) {
		return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
			"cannot open capture metadata for %s", op.name.c_str());
	}
	meta << "name\t" << op.name << "\n";
	meta << "machine\t" << op.machine << "\n";
	meta << "func\t" << op.func << "\n";
	meta << "vmfb\t" << op.vmfb << "\n";
	meta << "inputs\t" << inputs.size() << "\n";
	meta << "outputs\t" << outputs.size() << "\n";
	return iree_ok_status();
}

} // namespace

int main(int argc, char **argv) {
	Options opts;
	if (!ParseArgs(argc, argv, &opts))
		return 1;

	std::vector<PlanOp> plan = ParsePlan(opts.plan_path);
	if (plan.empty()) {
		fprintf(stderr, "plan empty: %s\n", opts.plan_path.c_str());
		return 1;
	}
	if (opts.verbose)
		fprintf(stderr, "plan: %zu ops\n", plan.size());

	iree_runtime_instance_options_t opts_inst;
	iree_runtime_instance_options_initialize(&opts_inst);
	iree_runtime_instance_options_use_all_available_drivers(&opts_inst);
	iree_runtime_instance_t *instance = NULL;
	iree_status_t st = iree_runtime_instance_create(
		&opts_inst, iree_allocator_system(), &instance);
	if (!iree_status_is_ok(st)) {
		iree_status_fprint(stderr, st);
		return 1;
	}

	std::map<std::string, iree_hal_device_t *> device_by_machine;
	for (const auto &op : plan) {
		if (device_by_machine.count(op.machine))
			continue;
		auto it = kMachineToUri.find(op.machine);
		if (it == kMachineToUri.end()) {
			fprintf(stderr, "unknown machine %s\n", op.machine.c_str());
			return 1;
		}
		iree_hal_device_t *dev = NULL;
		st = MakeDevice(instance, it->second, &dev);
		if (!iree_status_is_ok(st)) {
			fprintf(stderr, "make_device(%s) failed:\n  ", it->second.c_str());
			iree_status_fprint(stderr, st);
			return 1;
		}
		device_by_machine[op.machine] = dev;
	}

	std::map<std::string, CachedModule> module_by_op;
	for (const auto &op : plan) {
		CachedModule &cm = module_by_op[op.name];
		st = LoadModule(instance, device_by_machine[op.machine], op.vmfb, &cm);
		if (!iree_status_is_ok(st)) {
			fprintf(stderr, "LoadModule(%s, %s) failed:\n  ", op.name.c_str(),
				op.vmfb.c_str());
			iree_status_fprint(stderr, st);
			return 1;
		}
	}

	std::vector<uint8_t> input_bytes;
	if (!opts.input_from.empty()) {
		std::ifstream f(opts.input_from, std::ios::binary | std::ios::ate);
		std::streamsize n = f.tellg();
		f.seekg(0);
		input_bytes.resize((size_t)n);
		f.read(reinterpret_cast<char *>(input_bytes.data()), n);
	}

	FILE *trace = NULL;
	if (!opts.trace_csv.empty()) {
		trace = std::fopen(opts.trace_csv.c_str(), "w");
		fprintf(trace,
			"iter,op,machine,inputs_us,invoke_us,outputs_us,transfer_us\n");
	}

	const int total_iters = opts.warmup + opts.iterations;
	const bool capture_last_timed_iter =
		!opts.capture_dispatch_io_dir.empty() && opts.iterations > 0;
	for (int iter = 0; iter < total_iters; ++iter) {
		const bool timed = (iter >= opts.warmup);
		BufferRegistry registry;
		for (const auto &op : plan) {
			CachedModule &cm = module_by_op.at(op.name);
			iree_hal_device_t *dev = device_by_machine.at(op.machine);
			iree_vm_list_clear(iree_runtime_call_inputs(&cm.call));
			std::vector<iree_hal_buffer_view_t *> retained_inputs;

			const auto t_in0 = Clock::now();
			double transfer_us = 0.0;

			for (size_t bi = 0; bi < op.binding_byte_sizes.size(); ++bi) {
				int64_t want_bytes = op.binding_byte_sizes[bi];
				iree_hal_buffer_view_t *bv = NULL;
				bool sourced = false;
				const std::string source = (bi < op.binding_sources.size())
					? op.binding_sources[bi]
					: "";

				if (!source.empty() && source != "auto") {
					if (source == "zero") {
						st = MakeBufferView(dev, want_bytes, nullptr, &bv);
						if (!iree_status_is_ok(st)) {
							iree_status_fprint(stderr, st);
							return 1;
						}
						sourced = true;
					} else if (source == "input") {
						if (input_bytes.empty() ||
							(int64_t)input_bytes.size() != want_bytes) {
							fprintf(stderr,
								"input source for %s binding %zu needs %lld "
								"bytes, "
								"but --input-from has %zu bytes\n",
								op.name.c_str(), bi, (long long)want_bytes,
								input_bytes.size());
							return 1;
						}
						st = MakeBufferView(
							dev, want_bytes, input_bytes.data(), &bv);
						if (!iree_status_is_ok(st)) {
							iree_status_fprint(stderr, st);
							return 1;
						}
						sourced = true;
					} else if (source.rfind("file:", 0) == 0) {
						std::vector<uint8_t> bytes;
						st =
							ReadExactFile(source.substr(5), want_bytes, &bytes);
						if (!iree_status_is_ok(st)) {
							iree_status_fprint(stderr, st);
							return 1;
						}
						st = MakeBufferView(dev, want_bytes, bytes.data(), &bv);
						if (!iree_status_is_ok(st)) {
							iree_status_fprint(stderr, st);
							return 1;
						}
						sourced = true;
					} else if (source.rfind("pred:", 0) == 0) {
						std::string rest = source.substr(5);
						size_t colon = rest.rfind(':');
						std::string pred_name = rest;
						size_t out_index = 0;
						if (colon != std::string::npos) {
							pred_name = rest.substr(0, colon);
							out_index = (size_t)std::strtoull(
								rest.substr(colon + 1).c_str(), nullptr, 10);
						}
						const DispatchOutputs *po = registry.Get(pred_name);
						if (!po || out_index >= po->views.size()) {
							fprintf(stderr,
								"missing predecessor output for %s binding "
								"%zu: %s\n",
								op.name.c_str(), bi, source.c_str());
							return 1;
						}
						iree_hal_buffer_view_t *pv = po->views[out_index];
						if (po->device_uri == kMachineToUri.at(op.machine)) {
							iree_hal_buffer_view_retain(pv);
							bv = pv;
						} else {
							st = TransferAcrossDevices(
								pv, dev, &bv, &transfer_us);
							if (!iree_status_is_ok(st)) {
								iree_status_fprint(stderr, st);
								return 1;
							}
						}
						sourced = true;
					} else if (source.rfind("concat:", 0) == 0) {
						const auto t_cat0 = Clock::now();
						std::vector<uint8_t> joined;
						const std::vector<std::string> parts =
							SplitCsv(source.substr(7));
						for (const auto &part : parts) {
							std::vector<uint8_t> bytes;
							st =
								ResolvePredSourceToHost(registry, part, &bytes);
							if (!iree_status_is_ok(st)) {
								iree_status_fprint(stderr, st);
								return 1;
							}
							joined.insert(
								joined.end(), bytes.begin(), bytes.end());
						}
						if ((int64_t)joined.size() != want_bytes) {
							fprintf(stderr,
								"concat source for %s binding %zu produced %zu "
								"bytes, "
								"expected %lld\n",
								op.name.c_str(), bi, joined.size(),
								(long long)want_bytes);
							return 1;
						}
						st =
							MakeBufferView(dev, want_bytes, joined.data(), &bv);
						if (!iree_status_is_ok(st)) {
							iree_status_fprint(stderr, st);
							return 1;
						}
						const auto t_cat1 = Clock::now();
						transfer_us +=
							std::chrono::duration_cast<
								std::chrono::nanoseconds>(t_cat1 - t_cat0)
								.count() /
							1e3;
						sourced = true;
					} else {
						fprintf(stderr,
							"unknown binding source for %s binding %zu: %s\n",
							op.name.c_str(), bi, source.c_str());
						return 1;
					}
				}

				if (!sourced && !op.predecessors.empty()) {
					size_t seen = 0;
					for (const auto &pn : op.predecessors) {
						const DispatchOutputs *po = registry.Get(pn);
						if (!po)
							continue;
						for (auto *pv : po->views) {
							if (seen == bi) {
								if (po->device_uri ==
									kMachineToUri.at(op.machine)) {
									iree_hal_buffer_view_retain(pv);
									bv = pv;
								} else {
									st = TransferAcrossDevices(
										pv, dev, &bv, &transfer_us);
									if (!iree_status_is_ok(st)) {
										iree_status_fprint(stderr, st);
										return 1;
									}
								}
								sourced = true;
								break;
							}
							++seen;
						}
						if (sourced)
							break;
					}
				}
				if (!sourced) {
					if (opts.strict_binding_sources) {
						fprintf(stderr,
							"no explicit source for %s binding %zu (%lld "
							"bytes); "
							"refusing zero fallback in strict mode\n",
							op.name.c_str(), bi, (long long)want_bytes);
						return 1;
					}
					const uint8_t *fill = nullptr;
					if (bi == 0 && !input_bytes.empty() &&
						(int64_t)input_bytes.size() == want_bytes) {
						fill = input_bytes.data();
					}
					st = MakeBufferView(dev, want_bytes, fill, &bv);
					if (!iree_status_is_ok(st)) {
						iree_status_fprint(stderr, st);
						return 1;
					}
				}
				retained_inputs.push_back(bv);
				st = iree_runtime_call_inputs_push_back_buffer_view(
					&cm.call, bv);
				if (!iree_status_is_ok(st)) {
					iree_status_fprint(stderr, st);
					return 1;
				}
			}
			if (cm.coarse_fences &&
				cm.arity == static_cast<int>(retained_inputs.size()) + 2) {
				// Coarse-fence ABI appends (wait_fence, signal_fence). The
				// wait fence may be null; the signal fence is signaled by the
				// bytecode after the dispatch completes, so it must be a real
				// fence object.
				iree_vm_ref_t wait_ref = iree_vm_ref_null();
				st = iree_vm_list_push_ref_retain(
					iree_runtime_call_inputs(&cm.call), &wait_ref);
				if (!iree_status_is_ok(st)) {
					iree_status_fprint(stderr, st);
					return 1;
				}
				iree_hal_fence_t *signal_fence = NULL;
				st = iree_hal_fence_create(
					/*capacity=*/0, iree_allocator_system(), &signal_fence);
				if (!iree_status_is_ok(st)) {
					iree_status_fprint(stderr, st);
					return 1;
				}
				iree_vm_ref_t signal_ref =
					iree_hal_fence_move_ref(signal_fence);
				st = iree_vm_list_push_ref_move(
					iree_runtime_call_inputs(&cm.call), &signal_ref);
				if (!iree_status_is_ok(st)) {
					iree_vm_ref_release(&signal_ref);
					iree_status_fprint(stderr, st);
					return 1;
				}
			}
			const auto t_in1 = Clock::now();

			iree_vm_list_clear(iree_runtime_call_outputs(&cm.call));
			const auto t_inv0 = Clock::now();
			st = iree_runtime_call_invoke(&cm.call, /*flags=*/0);
			const auto t_inv1 = Clock::now();
			if (!iree_status_is_ok(st)) {
				fprintf(stderr, "invoke %s failed:\n  ", op.name.c_str());
				iree_status_fprint(stderr, st);
				return 1;
			}

			const auto t_out0 = Clock::now();
			std::vector<iree_hal_buffer_view_t *> outputs;
			iree_vm_list_t *outputs_list = iree_runtime_call_outputs(&cm.call);
			const iree_host_size_t output_count =
				iree_vm_list_size(outputs_list);
			for (iree_host_size_t oi = 0; oi < output_count; ++oi) {
				if (auto *out_view =
						iree_vm_list_get_buffer_view_retain(outputs_list, oi)) {
					outputs.push_back(out_view);
				}
			}
			if (outputs.empty() && !retained_inputs.empty()) {
				// Legacy fallback for late HAL wrappers that model outputs as
				// in-place buffer-view args and return `()`. Canonical wrappers
				// will return real tensor outputs and bypass this path.
				auto *last = retained_inputs.back();
				iree_hal_buffer_view_retain(last);
				outputs.push_back(last);
			}
			const bool should_capture = capture_last_timed_iter && timed &&
				iter == total_iters - 1 && ShouldCaptureDispatch(opts, op.name);
			if (should_capture) {
				st = CaptureDispatchIo(opts.capture_dispatch_io_dir, op, dev,
					retained_inputs, outputs);
				if (!iree_status_is_ok(st)) {
					fprintf(stderr, "capture %s failed:\n  ", op.name.c_str());
					iree_status_fprint(stderr, st);
					return 1;
				}
			}
			registry.Set(op.name, outputs, kMachineToUri.at(op.machine));
			for (auto *v : outputs) {
				if (v)
					iree_hal_buffer_view_release(v);
			}
			for (auto *v : retained_inputs)
				iree_hal_buffer_view_release(v);
			const auto t_out1 = Clock::now();

			double inputs_us =
				std::chrono::duration_cast<std::chrono::nanoseconds>(
					t_in1 - t_in0)
					.count() /
				1e3;
			double invoke_us =
				std::chrono::duration_cast<std::chrono::nanoseconds>(
					t_inv1 - t_inv0)
					.count() /
				1e3;
			double outputs_us =
				std::chrono::duration_cast<std::chrono::nanoseconds>(
					t_out1 - t_out0)
					.count() /
				1e3;
			if (timed && trace) {
				fprintf(trace, "%d,%s,%s,%.3f,%.3f,%.3f,%.3f\n",
					iter - opts.warmup, op.name.c_str(), op.machine.c_str(),
					inputs_us, invoke_us, outputs_us, transfer_us);
			}
			if (opts.verbose) {
				fprintf(stderr,
					"[iter %d/%d] %s on %s invoke=%.3fms transfer=%.3fms\n",
					iter, total_iters, op.name.c_str(), op.machine.c_str(),
					invoke_us / 1e3, transfer_us / 1e3);
			}
		}

		if (timed && iter == total_iters - 1 && !opts.output_to.empty() &&
			!plan.empty()) {
			const auto &last_op = plan.back();
			const DispatchOutputs *po = registry.Get(last_op.name);
			if (po && !po->views.empty()) {
				iree_hal_device_t *out_dev = nullptr;
				for (const auto &kv : kMachineToUri) {
					if (kv.second == po->device_uri) {
						out_dev = device_by_machine.at(kv.first);
						break;
					}
				}
				if (!out_dev) {
					fprintf(stderr,
						"could not find device for final output uri %s\n",
						po->device_uri.c_str());
					return 1;
				}
				st = DumpBufferViewToFile(
					out_dev, po->views.front(), opts.output_to);
				if (!iree_status_is_ok(st)) {
					fprintf(stderr, "write final output failed:\n  ");
					iree_status_fprint(stderr, st);
					return 1;
				}
				iree_hal_buffer_t *buf =
					iree_hal_buffer_view_buffer(po->views.front());
				fprintf(stderr, "wrote %lld bytes of final output to %s\n",
					(long long)iree_hal_buffer_byte_length(buf),
					opts.output_to.c_str());
			}
		}
	}

	if (trace)
		std::fclose(trace);
	for (auto &kv : module_by_op)
		CachedModuleRelease(&kv.second);
	for (auto &kv : device_by_machine)
		iree_hal_device_release(kv.second);
	iree_runtime_instance_release(instance);
	return 0;
}
