/** @file dispatch_types.h
 *  @brief Core types and scheduling policies for dispatch graph execution.
 *
 *  Defines the data structures shared by all dispatch runners. Scheduling
 *  policies are data-driven: each node's release_policy and time_dep_mode
 *  control execution behavior, making the scheduler network-agnostic.
 *  When absent from JSON, InferSchedulingPolicies() applies
 *  backwards-compatible heuristics based on job_name.
 */

#ifndef MERLIN_DISPATCH_TYPES_H_
#define MERLIN_DISPATCH_TYPES_H_

#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "core/stats.h"

namespace merlin_bench {

//------------------------------------------------------------------------------
// Hardware targets
//------------------------------------------------------------------------------

/** @brief Execution target for a dispatch node. */
enum class HardwareTarget {
	kCpuP = 0, /**< Performance CPU cluster. */
	kCpuE = 1, /**< Efficiency CPU cluster. */
};

/** @brief Return a human-readable name for a HardwareTarget.
 *  @param t Target enum value.
 *  @return Static string such as "CPU_P" or "CPU_E".
 */
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

/** @brief Parse a HardwareTarget from its string representation.
 *  @param s   Input string ("CPU_P" or "CPU_E").
 *  @param out Receives the parsed value on success.
 *  @return True if parsing succeeded.
 */
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

//------------------------------------------------------------------------------
// Scheduling policies
//------------------------------------------------------------------------------

/** @brief How a dispatch node's release time is determined when eligible.
 *
 *  - kImmediate:   Released as soon as all predecessors complete.
 *  - kPhaseLocked: Held until planned_start_us before becoming eligible.
 */
enum class ReleasePolicy {
	kImmediate = 0, /**< Release immediately after predecessors. */
	kPhaseLocked = 1, /**< Hold until planned start time. */
};

/** @brief How the time_dependency field is treated during predecessor
 * expansion.
 *
 *  - kHard: time_dependency is added to all_predecessors (blocks execution).
 *  - kSoft: time_dependency is a scheduling hint only (not a hard gate).
 */
enum class TimeDependencyMode {
	kHard = 0, /**< Blocks execution until dependency completes. */
	kSoft = 1, /**< Advisory hint; does not block. */
};

/** @brief Parse a ReleasePolicy from its JSON string representation.
 *  @param s   Input string ("immediate" or "phase_locked").
 *  @param out Receives the parsed value on success.
 *  @return True if parsing succeeded.
 */
inline bool ParseReleasePolicy(const std::string &s, ReleasePolicy *out) {
	if (s == "immediate") {
		*out = ReleasePolicy::kImmediate;
		return true;
	}
	if (s == "phase_locked") {
		*out = ReleasePolicy::kPhaseLocked;
		return true;
	}
	return false;
}

/** @brief Parse a TimeDependencyMode from its JSON string representation.
 *  @param s   Input string ("hard" or "soft").
 *  @param out Receives the parsed value on success.
 *  @return True if parsing succeeded.
 */
inline bool ParseTimeDependencyMode(
	const std::string &s, TimeDependencyMode *out) {
	if (s == "hard") {
		*out = TimeDependencyMode::kHard;
		return true;
	}
	if (s == "soft") {
		*out = TimeDependencyMode::kSoft;
		return true;
	}
	return false;
}

//------------------------------------------------------------------------------
// Dispatch node
//------------------------------------------------------------------------------

/** @brief A single dispatch unit in the execution graph.
 *
 *  Holds identity, scheduling metadata, dependency edges, VMFB paths, and
 *  per-node runtime statistics. Populated from JSON and optionally enriched
 *  by InferSchedulingPolicies() and ExpandAllPredecessors().
 */
struct DispatchNode {
	std::string key; /**< Unique string key (JSON object key). */
	int id = -1; /**< Numeric dispatch ID within its job. */
	int ordinal = 0; /**< Ordinal index within the dispatch sequence. */
	int total = 0; /**< Total dispatches in the owning job. */

	HardwareTarget hardware_target =
		HardwareTarget::kCpuP; /**< Execution target. */
	double start_time_ms = 0.0; /**< Planned start time (ms). */
	double planned_duration_ms = 0.0; /**< Planned duration (ms). */

	std::string job_name; /**< Logical job name (e.g. "mlp0"). */
	std::string module_name; /**< IREE module name for VMFB lookup. */
	std::string time_dependency; /**< Key of the time-dependency predecessor. */

	/** Scheduling policies. Set from JSON or inferred by
	 *  InferSchedulingPolicies(). */
	ReleasePolicy release_policy = ReleasePolicy::kImmediate;
	TimeDependencyMode time_dep_mode = TimeDependencyMode::kHard;
	bool policies_from_json = false; /**< True if policies came from JSON. */

	/** Explicit dependency keys from JSON. */
	std::vector<std::string> deps;
	/** Expanded predecessor set (deps + hard time_dependency). */
	std::vector<std::string> all_predecessors;

	std::string vmfb_path_json; /**< Raw VMFB path from JSON. */
	std::string vmfb_path_resolved; /**< Resolved absolute VMFB path. */

	RunningStats run_stats; /**< Per-node runtime statistics. */
};

/** @brief Top-level model holding all dispatch nodes and graph metadata. */
struct GraphModel {
	std::vector<DispatchNode> nodes; /**< All dispatch nodes. */
	std::string
		dispatch_vmfb_dir_from_json; /**< VMFB dir override from JSON. */
	double makespan_ms = 0.0; /**< Planned total makespan (ms). */
};

//------------------------------------------------------------------------------
// Time helpers
//------------------------------------------------------------------------------

/** @brief Convert milliseconds to microseconds, clamping negatives to zero.
 *  @param ms Time in milliseconds.
 *  @return Equivalent time in microseconds.
 */
inline uint64_t MsToUs(double ms) {
	if (ms < 0.0)
		ms = 0.0;
	return static_cast<uint64_t>(llround(ms * 1000.0));
}

/** @brief Compute elapsed microseconds from a base time point.
 *  @param base Reference time point.
 *  @param now  Current time point.
 *  @return Microseconds elapsed (now - base).
 */
inline uint64_t UsSince(const std::chrono::steady_clock::time_point &base,
	std::chrono::steady_clock::time_point now) {
	return static_cast<uint64_t>(
		std::chrono::duration_cast<std::chrono::microseconds>(now - base)
			.count());
}

//------------------------------------------------------------------------------
// Scheduling policy inference (backwards compatibility)
//------------------------------------------------------------------------------

/** @brief Infer scheduling policies for nodes that lack explicit JSON fields.
 *
 *  Preserves the original MLP-vs-dronet behavior:
 *  - Jobs starting with "mlp": first dispatch (id==0) is phase-locked,
 *    time_dependency is soft.
 *  - All other jobs: immediate release, time_dependency is hard.
 *
 *  Nodes with policies_from_json == true are left unchanged.
 *
 *  @param nodes Mutable vector of dispatch nodes to update in-place.
 */
inline void InferSchedulingPolicies(std::vector<DispatchNode> *nodes) {
	for (auto &n : *nodes) {
		if (n.policies_from_json)
			continue;

		const bool is_periodic =
			n.job_name.size() >= 3 && n.job_name.compare(0, 3, "mlp") == 0;

		if (is_periodic && n.id == 0) {
			n.release_policy = ReleasePolicy::kPhaseLocked;
		} else {
			n.release_policy = ReleasePolicy::kImmediate;
		}

		n.time_dep_mode =
			is_periodic ? TimeDependencyMode::kSoft : TimeDependencyMode::kHard;
	}
}

} // namespace merlin_bench

#endif // MERLIN_DISPATCH_TYPES_H_
