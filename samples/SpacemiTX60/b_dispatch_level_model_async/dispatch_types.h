// dispatch_types.h — Private types for the dispatch-level scheduler sample.

#ifndef DISPATCH_TYPES_H_
#define DISPATCH_TYPES_H_

#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "iree_bench/stats.h"

enum class HardwareTarget {
	kCpuP = 0,
	kCpuE = 1,
};

inline const char *HardwareTargetName(HardwareTarget t) {
	switch (t) {
		case HardwareTarget::kCpuP:
			return "CPU_P";
		case HardwareTarget::kCpuE:
			return "CPU_E";
		default:
			return "UNKNOWN";
	}
}

inline bool ParseHardwareTarget(const std::string &s, HardwareTarget *out) {
	if (s == "CPU_P") {
		*out = HardwareTarget::kCpuP;
		return true;
	}
	if (s == "CPU_E") {
		*out = HardwareTarget::kCpuE;
		return true;
	}
	return false;
}

inline uint64_t MsToUs(double ms) {
	if (ms < 0.0)
		ms = 0.0;
	return static_cast<uint64_t>(llround(ms * 1000.0));
}

inline uint64_t UsSince(const std::chrono::steady_clock::time_point &base,
	std::chrono::steady_clock::time_point now) {
	return static_cast<uint64_t>(
		std::chrono::duration_cast<std::chrono::microseconds>(now - base)
			.count());
}

struct DispatchNode {
	std::string key;
	int id = -1;
	int ordinal = 0;
	int total = 0;

	HardwareTarget hardware_target = HardwareTarget::kCpuP;
	double start_time_ms = 0.0;
	double planned_duration_ms = 0.0;

	std::string job_name;
	std::string module_name;
	std::string time_dependency;

	std::vector<std::string> deps;
	std::vector<std::string> all_predecessors;

	std::string vmfb_path_json;
	std::string vmfb_path_resolved;

	merlin_bench::RunningStats run_stats;
};

struct GraphModel {
	std::vector<DispatchNode> nodes;
	std::string dispatch_vmfb_dir_from_json;
	double makespan_ms = 0.0;
};

inline bool IsMlpJob(const DispatchNode &node) {
	return node.job_name.size() >= 3 && node.job_name.compare(0, 3, "mlp") == 0;
}

inline bool IsMlpFirstDispatch(const DispatchNode &node) {
	return IsMlpJob(node) && node.id == 0;
}

#endif // DISPATCH_TYPES_H_
