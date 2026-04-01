/** @file dispatch_types.h
 *  @brief Core types and scheduling policies for dispatch graph execution.
 *
 *  Defines the data structures shared by all dispatch runners. Scheduling
 *  policies are data-driven: each node's release_policy and time_dep_mode
 *  control execution behavior, making the scheduler network-agnostic.
 *  When absent from JSON, InferSchedulingPolicies() applies
 *  backwards-compatible heuristics based on job_name.
 *
 *  Hardware targets are dynamic and registry-based: callers populate a
 *  TargetRegistry at startup from JSON or CLI flags. Well-known constants
 *  kTargetCpuP (0) and kTargetCpuE (1) exist for backward compatibility.
 */

#ifndef MERLIN_DISPATCH_TYPES_H_
#define MERLIN_DISPATCH_TYPES_H_

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "core/stats.h"

namespace merlin_bench {

//------------------------------------------------------------------------------
// Hardware targets — registry-based, N-target
//------------------------------------------------------------------------------

/** @brief Numeric target identifier. Indexes into a TargetRegistry. */
using TargetId = uint8_t;

/** Well-known target IDs for backward compatibility. */
static constexpr TargetId kTargetCpuP = 0;
static constexpr TargetId kTargetCpuE = 1;
static constexpr TargetId kTargetInvalid = 255;

/** Maximum number of simultaneously registered targets. */
static constexpr int kMaxTargets = 16;

/** @brief Per-target metadata stored in the registry. */
struct TargetInfo {
	std::string name; /**< Human-readable name (e.g. "CPU_P"). */
	std::string cpu_ids; /**< Comma-separated logical CPU IDs. */
	std::string variant_dir; /**< ISA variant directory (e.g. "RVV"). */
};

/** @brief Dynamic registry mapping TargetId <-> name + metadata.
 *
 *  Targets are assigned sequential IDs starting from 0. The registry
 *  supports up to kMaxTargets entries.
 *
 *  Typical usage:
 *    TargetRegistry reg;
 *    reg.Register("CPU_P", "0,1,2,3", "RVV");   // -> id 0
 *    reg.Register("CPU_E", "4,5", "scalar");     // -> id 1
 *    reg.Register("CPU_AUX", "6,7", "scalar");   // -> id 2
 */
struct TargetRegistry {
	std::vector<TargetInfo> targets_;

	/** @brief Register a new target.
	 *  @return The assigned TargetId, or kTargetInvalid if full. */
	TargetId Register(const std::string &name, const std::string &cpu_ids = "",
		const std::string &variant_dir = "") {
		if ((int)targets_.size() >= kMaxTargets) {
			fprintf(stderr, "TargetRegistry: too many targets (max %d)\n",
				kMaxTargets);
			return kTargetInvalid;
		}
		TargetId id = static_cast<TargetId>(targets_.size());
		targets_.push_back({name, cpu_ids, variant_dir});
		return id;
	}

	/** @brief Number of registered targets. */
	int Size() const {
		return static_cast<int>(targets_.size());
	}

	/** @brief Look up a target by name.
	 *  @return The TargetId, or kTargetInvalid if not found. */
	TargetId Parse(const std::string &name) const {
		for (size_t i = 0; i < targets_.size(); ++i) {
			if (targets_[i].name == name)
				return static_cast<TargetId>(i);
		}
		return kTargetInvalid;
	}

	/** @brief Return the human-readable name for a TargetId. */
	const char *Name(TargetId id) const {
		if (id < targets_.size())
			return targets_[id].name.c_str();
		return "UNKNOWN";
	}

	/** @brief Return the CPU IDs string for a TargetId. */
	const char *CpuIds(TargetId id) const {
		if (id < targets_.size())
			return targets_[id].cpu_ids.c_str();
		return "";
	}

	/** @brief Return the variant directory for a TargetId. */
	const char *VariantDir(TargetId id) const {
		if (id < targets_.size())
			return targets_[id].variant_dir.c_str();
		return "";
	}

	/** @brief Check whether a TargetId is valid in this registry. */
	bool Valid(TargetId id) const {
		return id != kTargetInvalid && id < targets_.size();
	}

	/** @brief Create a default 2-target registry (CPU_P + CPU_E).
	 *
	 *  For backward compatibility with code that assumes exactly two
	 *  targets. The caller can override cpu_ids and variant_dirs after. */
	static TargetRegistry Default2Target() {
		TargetRegistry reg;
		reg.Register("CPU_P", "0,1,2,3", "RVV");
		reg.Register("CPU_E", "4,5", "scalar");
		return reg;
	}
};

//------------------------------------------------------------------------------
// Backward-compatible free functions (delegate to a registry)
//------------------------------------------------------------------------------

/** @brief Return a human-readable name for a TargetId using a registry.
 *  @param id  Target identifier.
 *  @param reg Registry to look up in.
 *  @return Static-ish string such as "CPU_P" or "CPU_E".
 */
inline const char *TargetName(TargetId id, const TargetRegistry &reg) {
	return reg.Name(id);
}

/** @brief Parse a target name to a TargetId using a registry.
 *  @param s   Input string (e.g. "CPU_P").
 *  @param reg Registry to look up in.
 *  @param out Receives the parsed TargetId on success.
 *  @return True if parsing succeeded.
 */
inline bool ParseTarget(
	const std::string &s, const TargetRegistry &reg, TargetId *out) {
	TargetId id = reg.Parse(s);
	if (id == kTargetInvalid)
		return false;
	*out = id;
	return true;
}

//------------------------------------------------------------------------------
// Legacy HardwareTarget compatibility shim
//------------------------------------------------------------------------------

/** @brief Legacy alias — kept so existing code that names the type still
 *  compiles. New code should use TargetId directly. */
using HardwareTarget = TargetId;

/** @brief Legacy constant aliases. */
static constexpr HardwareTarget kCpuP = kTargetCpuP;
static constexpr HardwareTarget kCpuE = kTargetCpuE;

/** @brief Legacy name lookup using the default 2-target convention.
 *
 *  Prefer TargetName(id, registry) in new code. This overload exists for
 *  call sites that have not yet been plumbed with a registry reference. */
inline const char *HardwareTargetName(HardwareTarget t) {
	switch (t) {
		case kTargetCpuP:
			return "CPU_P";
		case kTargetCpuE:
			return "CPU_E";
		default: {
			// Fallback for ids > 1 when no registry is available.
			static thread_local char buf[32];
			snprintf(buf, sizeof(buf), "TARGET_%u", (unsigned)t);
			return buf;
		}
	}
}

/** @brief Legacy parse using the default 2-target convention.
 *
 *  Prefer ParseTarget(s, registry, out) in new code. */
inline bool ParseHardwareTarget(const std::string &s, HardwareTarget *out) {
	if (s == "CPU_P") {
		*out = kTargetCpuP;
		return true;
	}
	if (s == "CPU_E") {
		*out = kTargetCpuE;
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

	TargetId hardware_target = kTargetCpuP; /**< Execution target ID. */
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
