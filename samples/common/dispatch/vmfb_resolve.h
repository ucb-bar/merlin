/** @file vmfb_resolve.h
 *  @brief VMFB path resolution for dispatch schedulers.
 *
 *  Resolves the absolute VMFB path for a dispatch node using a multi-step
 *  strategy: exact JSON path, then constructed Merlin layout paths, then
 *  fallback directories. The target platform and ISA variant directories
 *  are configurable, making this work for any hardware target.
 */

#ifndef MERLIN_DISPATCH_VMFB_RESOLVE_H_
#define MERLIN_DISPATCH_VMFB_RESOLVE_H_

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "core/path_utils.h"
#include "dispatch/dispatch_types.h"

namespace merlin_bench {

/** @brief Map a HardwareTarget to an ISA variant directory name.
 *  @param target Hardware target enum.
 *  @return Directory name (e.g. "RVV" for kCpuP, "scalar" for kCpuE).
 */
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

/** @brief Extract the model family from a module_name.
 *
 *  Looks for the "$async_dispatch_" marker in module_name (e.g. returns
 *  "mlp" from "mlp$async_dispatch_0..."). Falls back to stripping trailing
 *  digits from job_name for "mlp0"-style names.
 *
 *  @param module_name IREE module name (may be empty).
 *  @param job_name    Logical job name used as fallback.
 *  @return Normalized model family string.
 */
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

/** @brief Append "_benchmark.vmfb" suffix to a VMFB stem if not already
 * present.
 *  @param stem Input filename or stem.
 *  @return Filename with "_benchmark.vmfb" suffix.
 */
inline std::string AddBenchmarkSuffix(const std::string &stem) {
	if (stem.empty())
		return stem;
	if (EndsWith(stem, "_benchmark.vmfb"))
		return stem;
	if (EndsWith(stem, ".vmfb")) {
		return stem.substr(0, stem.size() - 5) + "_benchmark.vmfb";
	}
	return stem + "_benchmark.vmfb";
}

/** @brief Append a string to a vector if not already present (and non-empty).
 *  @param out Target vector.
 *  @param s   String to add.
 */
inline void AppendUnique(std::vector<std::string> *out, const std::string &s) {
	if (s.empty())
		return;
	if (std::find(out->begin(), out->end(), s) == out->end()) {
		out->push_back(s);
	}
}

/** @brief Extract the benchmark stem from a module name.
 *
 *  Finds the "_embedded_elf_riscv_64" marker and returns everything up to
 *  and including it.
 *
 *  @param module_name Full IREE module name.
 *  @return Truncated stem, or the full module_name if the marker is absent.
 */
inline std::string BenchmarkStemFromModuleName(const std::string &module_name) {
	if (module_name.empty())
		return module_name;

	static const char kMarker[] = "_embedded_elf_riscv_64";
	const size_t pos = module_name.find(kMarker);
	if (pos == std::string::npos)
		return module_name;

	return module_name.substr(0, pos + strlen(kMarker));
}

/** @brief Resolve the absolute VMFB path for a dispatch node.
 *
 *  Resolution strategy (in order):
 *    1. Exact vmfb_path_json from the schedule (absolute or relative to root).
 *    2. Construct from model family + module name + ISA variant, using the
 *       standard Merlin layout:
 *         <root>/gen/vmfb/<family>/<platform>/<variant>/<model>/benchmarks/vmfb/
 *    3. Fallback directories (root, json_dir).
 *
 *  @param vmfb_root_dir   Root directory for VMFB resolution (nullable).
 *  @param target_platform Platform subdirectory (e.g. "spacemit_x60").
 *                         Defaults to "spacemit_x60" if null or empty.
 *  @param json_dir        Directory containing the schedule JSON.
 *  @param node            Dispatch node whose VMFB path is being resolved.
 *  @return Resolved absolute path, or best-effort candidate if not found.
 */
inline std::string ResolveVmfbPath(const char *vmfb_root_dir,
	const char *target_platform, const std::string &json_dir,
	const DispatchNode &node) {

	// 1. Exact vmfb_path from JSON wins.
	if (!node.vmfb_path_json.empty()) {
		if (node.vmfb_path_json[0] == '/')
			return node.vmfb_path_json;

		const std::string root = (vmfb_root_dir && vmfb_root_dir[0])
			? std::string(vmfb_root_dir)
			: json_dir;
		return PathJoin2(root, node.vmfb_path_json);
	}

	const std::string root = (vmfb_root_dir && vmfb_root_dir[0])
		? std::string(vmfb_root_dir)
		: json_dir;

	const std::string platform = (target_platform && target_platform[0])
		? std::string(target_platform)
		: "spacemit_x60";

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
	// <root>/gen/vmfb/<family>/<platform>/<variant>/<model>/benchmarks/vmfb/
	candidate_dirs.push_back(PathJoin2(
		PathJoin2(
			PathJoin2(
				PathJoin2(PathJoin2(PathJoin2(PathJoin2(root, "gen"), "vmfb"),
							  model_family),
					platform),
				variant_dir),
			model_dir),
		"benchmarks/vmfb"));

	// Fallbacks.
	candidate_dirs.push_back(PathJoin2(
		PathJoin2(
			PathJoin2(
				PathJoin2(PathJoin2(PathJoin2(PathJoin2(root, "gen"), "vmfb"),
							  model_family),
					platform),
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

/** @brief Resolve VMFB paths for nodes that only have vmfb_path_json set.
 *
 *  Simple resolution: joins vmfb_path_json relative to json_dir. Suitable
 *  for baseline dispatch graphs that do not use module-name-based lookup.
 *
 *  @param json_dir Directory containing the schedule JSON.
 *  @param nodes    Mutable vector of dispatch nodes to update in-place.
 */
inline void ResolveSimpleVmfbPaths(
	const std::string &json_dir, std::vector<DispatchNode> *nodes) {
	for (auto &n : *nodes) {
		if (!n.vmfb_path_json.empty() && n.vmfb_path_resolved.empty()) {
			n.vmfb_path_resolved = PathJoin2(json_dir, n.vmfb_path_json);
		}
	}
}

} // namespace merlin_bench

#endif // MERLIN_DISPATCH_VMFB_RESOLVE_H_
