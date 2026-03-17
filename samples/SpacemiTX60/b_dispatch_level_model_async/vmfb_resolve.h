// vmfb_resolve.h — VMFB path resolution for the dispatch-level scheduler.

#ifndef VMFB_RESOLVE_H_
#define VMFB_RESOLVE_H_

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "iree_bench/path_utils.h"

#include "dispatch_types.h"
#include "runtime_scheduler.h"

inline const char *VariantDirFromHardwareTarget(HardwareTarget target) {
	switch (target) {
		case HardwareTarget::kCpuP:
			return "RVV";
		case HardwareTarget::kCpuE:
			return "scalar";
		default:
			return "";
	}
}

inline std::string NormalizeModelFamilyFromModuleName(
	const std::string &module_name, const std::string &job_name) {
	if (!module_name.empty()) {
		const size_t pos = module_name.find("$async_dispatch_");
		if (pos != std::string::npos) {
			return module_name.substr(0, pos);
		}
	}

	// Fallback: mlp0..mlp16 -> mlp
	if (job_name.size() >= 3 && job_name.compare(0, 3, "mlp") == 0) {
		bool rest_is_digits = true;
		for (size_t i = 3; i < job_name.size(); ++i) {
			if (job_name[i] < '0' || job_name[i] > '9') {
				rest_is_digits = false;
				break;
			}
		}
		if (rest_is_digits)
			return "mlp";
	}

	return job_name;
}

inline std::string AddBenchmarkSuffix(const std::string &stem) {
	using merlin_bench::EndsWith;
	if (stem.empty())
		return stem;
	if (EndsWith(stem, "_benchmark.vmfb"))
		return stem;
	if (EndsWith(stem, ".vmfb")) {
		return stem.substr(0, stem.size() - 5) + "_benchmark.vmfb";
	}
	return stem + "_benchmark.vmfb";
}

inline void AppendUnique(std::vector<std::string> *out, const std::string &s) {
	if (s.empty())
		return;
	if (std::find(out->begin(), out->end(), s) == out->end()) {
		out->push_back(s);
	}
}

inline std::string BenchmarkStemFromModuleName(const std::string &module_name) {
	if (module_name.empty())
		return module_name;

	static const char kMarker[] = "_embedded_elf_riscv_64";
	const size_t pos = module_name.find(kMarker);
	if (pos == std::string::npos)
		return module_name;

	return module_name.substr(0, pos + strlen(kMarker));
}

inline std::string ResolveVmfbPath(const dispatch_graph_config_t *cfg,
	const std::string &json_dir, const DispatchNode &node) {
	using merlin_bench::FileReadable;
	using merlin_bench::PathJoin2;

	// 1. Exact vmfb_path from JSON wins.
	if (!node.vmfb_path_json.empty()) {
		if (node.vmfb_path_json[0] == '/')
			return node.vmfb_path_json;

		const std::string root =
			(cfg && cfg->vmfb_root_dir && cfg->vmfb_root_dir[0])
			? std::string(cfg->vmfb_root_dir)
			: json_dir;
		return PathJoin2(root, node.vmfb_path_json);
	}

	const std::string root =
		(cfg && cfg->vmfb_root_dir && cfg->vmfb_root_dir[0])
		? std::string(cfg->vmfb_root_dir)
		: json_dir;

	const std::string model_family =
		NormalizeModelFamilyFromModuleName(node.module_name, node.job_name);
	const std::string variant_dir =
		VariantDirFromHardwareTarget(node.hardware_target);
	const std::string model_dir = model_family + ".q.int8";

	std::vector<std::string> candidate_names;
	if (!node.module_name.empty()) {
		const std::string full_name = node.module_name;
		const std::string short_name =
			BenchmarkStemFromModuleName(node.module_name);

		AppendUnique(
			&candidate_names, "module_" + full_name + "_benchmark.vmfb");
		AppendUnique(&candidate_names, full_name + "_benchmark.vmfb");
		AppendUnique(&candidate_names, "module_" + full_name + ".vmfb");
		AppendUnique(&candidate_names, full_name + ".vmfb");

		AppendUnique(
			&candidate_names, "module_" + short_name + "_benchmark.vmfb");
		AppendUnique(&candidate_names, short_name + "_benchmark.vmfb");
		AppendUnique(&candidate_names, "module_" + short_name + ".vmfb");
		AppendUnique(&candidate_names, short_name + ".vmfb");
	}

	std::vector<std::string> candidate_dirs;

	// Exact Merlin layout:
	// <root>/gen/vmfb/<family>/spacemit_x60/<RVV|scalar>/<family>.q.int8/benchmarks/vmfb/
	candidate_dirs.push_back(PathJoin2(
		PathJoin2(
			PathJoin2(
				PathJoin2(PathJoin2(PathJoin2(PathJoin2(root, "gen"), "vmfb"),
							  model_family),
					"spacemit_x60"),
				variant_dir),
			model_dir),
		"benchmarks/vmfb"));

	// Fallbacks.
	candidate_dirs.push_back(PathJoin2(
		PathJoin2(
			PathJoin2(
				PathJoin2(PathJoin2(PathJoin2(PathJoin2(root, "gen"), "vmfb"),
							  model_family),
					"spacemit_x60"),
				variant_dir),
			model_dir),
		"benchmarks"));

	candidate_dirs.push_back(root);
	candidate_dirs.push_back(json_dir);

	for (const auto &dir : candidate_dirs) {
		for (const auto &name : candidate_names) {
			const std::string path = PathJoin2(dir, name);
			if (FileReadable(path))
				return path;
		}
	}

	if (!candidate_dirs.empty() && !candidate_names.empty()) {
		return PathJoin2(candidate_dirs.front(), candidate_names.front());
	}
	return std::string();
}

#endif // VMFB_RESOLVE_H_
