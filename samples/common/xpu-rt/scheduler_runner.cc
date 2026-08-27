// samples/common/xpu-rt/scheduler_runner.cc
//
// Generic two-cluster dispatch scheduler (CPU_P + CPU_E).
//
// - Two long-lived worker threads, one per hardware target.
// - Each worker is pinned to its CPU set via pthread_setaffinity_np.
// - Pinned local-task devices with dedicated task executors (one per core set).
// - One cached runtime session per (target, vmfb_path).
// - Release-time scheduling with phase-locked roots and dependency-driven
//   chains.
// - Spin-wait for short delays (<5ms) to avoid condvar timer overshoot.
//
// Target-agnostic: hardware-specific parameters (core layout, ISA variants,
// platform name) are supplied via the scheduler_runner_config_t struct.

#include "xpu-rt/scheduler_runner.h"

#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <deque>
#include <map>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"

#include "core/path_utils.h"
#include "core/stats.h"
#include "dispatch/dispatch_graph.h"
#include "dispatch/dispatch_output.h"
#include "dispatch/dispatch_types.h"
#include "dispatch/vmfb_resolve.h"
#include "runtime/fatal_state.h"
#include "runtime/module_cache.h"
#include "runtime/pinned_device.h"

namespace {

using namespace merlin_bench;
using Clock = std::chrono::steady_clock;

//------------------------------------------------------------------------------
// CPU set parsing / validation / affinity
//------------------------------------------------------------------------------

static bool SplitCpuIds(const char *text, std::vector<int> *out_ids) {
	out_ids->clear();
	if (!text || !text[0])
		return false;
	const char *cur = text;
	while (*cur) {
		char *endptr = nullptr;
		long v = strtol(cur, &endptr, 10);
		if (endptr == cur)
			return false;
		out_ids->push_back(static_cast<int>(v));
		if (*endptr == '\0')
			break;
		if (*endptr != ',')
			return false;
		cur = endptr + 1;
	}
	return !out_ids->empty();
}

static bool ValidateCorePartition(const scheduler_runner_config_t *cfg) {
	const int visible_cores = cfg->visible_cores > 0 ? cfg->visible_cores : 64;

	std::vector<int> p_ids;
	std::vector<int> e_ids;
	if (!cfg->cpu_p_cpu_ids || !cfg->cpu_p_cpu_ids[0]) {
		fprintf(stderr, "cpu_p_cpu_ids is required\n");
		return false;
	}
	if (!cfg->cpu_e_cpu_ids || !cfg->cpu_e_cpu_ids[0]) {
		fprintf(stderr, "cpu_e_cpu_ids is required\n");
		return false;
	}
	if (!SplitCpuIds(cfg->cpu_p_cpu_ids, &p_ids)) {
		fprintf(stderr, "Invalid --cpu_p_cpu_ids\n");
		return false;
	}
	if (!SplitCpuIds(cfg->cpu_e_cpu_ids, &e_ids)) {
		fprintf(stderr, "Invalid --cpu_e_cpu_ids\n");
		return false;
	}

	std::unordered_set<int> seen;
	for (int v : p_ids) {
		if (v < 0 || v >= visible_cores) {
			fprintf(stderr, "CPU_P core %d out of range [0,%d)\n", v,
				visible_cores);
			return false;
		}
		if (!seen.insert(v).second) {
			fprintf(stderr, "Duplicate logical core %d in CPU_P set\n", v);
			return false;
		}
	}
	for (int v : e_ids) {
		if (v < 0 || v >= visible_cores) {
			fprintf(stderr, "CPU_E core %d out of range [0,%d)\n", v,
				visible_cores);
			return false;
		}
		if (!seen.insert(v).second) {
			fprintf(
				stderr, "CPU_E core %d overlaps CPU_P or is duplicated\n", v);
			return false;
		}
	}
	return true;
}

static void BestEffortPinCurrentThreadToCpuIds(const char *cpu_ids_csv) {
#if defined(__linux__)
	std::vector<int> ids;
	if (!SplitCpuIds(cpu_ids_csv, &ids))
		return;

	cpu_set_t set;
	CPU_ZERO(&set);
	for (int id : ids)
		CPU_SET(id, &set);

	// Best effort only.
	(void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
	(void)cpu_ids_csv;
#endif
}

//------------------------------------------------------------------------------
// Scheduler runtime state
//------------------------------------------------------------------------------

struct NodeExecState {
	uint64_t planned_start_us = 0;
	uint64_t release_us = 0;
	uint64_t ready_us = 0;
	uint64_t start_us = 0;
	uint64_t end_us = 0;
	bool enqueued = false;
	bool running = false;
	bool done = false;
};

struct SchedulerShared {
	std::mutex mu;
	std::condition_variable cv;

	bool shutdown = false;
	bool active = false;

	int current_graph_iter = 0;
	Clock::time_point iter_t0{};

	std::vector<int> remaining_preds;
	std::vector<NodeExecState> exec;

	std::vector<int> ready_p;
	std::vector<int> ready_e;

	std::vector<int> future_p;
	std::vector<int> future_e;

	size_t completed = 0;
	size_t total_nodes = 0;

	// Which physical cores are occupied right now, one bitmask per cluster
	// (index 0 = CPU_P, 1 = CPU_E), bit k = that cluster's core k.
	//
	// A one-core node takes one bit; a shard -- one dispatch spread over
	// several cores, written "CPU_P#0+CPU_P#1+..." in the schedule -- takes all
	// of them. Without this the runner would happily start three more
	// dispatches on cores 1-3 while a four-core shard was using them, and the
	// resulting trace would describe a machine with more cores than the board
	// has. That is the same class of mistake as declaring eight machines
	// against two cores, which already produced one retracted conclusion.
	uint64_t busy_cores[2] = {0, 0};
	// Cores a *waiting* shard has claimed. Single-core work refuses to start on
	// a claimed core, so a stream of short dispatches cannot starve a shard
	// that needs all of them free at once. Shards ignore it, which keeps two
	// shards from deadlocking on each other's claims; they are still mutually
	// excluded by busy_cores.
	uint64_t wanted_cores[2] = {0, 0};
};

// Cluster index used by the core masks: CPU_P is 0, CPU_E is 1.
static inline int ClusterIndex(HardwareTarget t) {
	return t == HardwareTarget::kCpuP ? 0 : 1;
}

// Bits for every core a node occupies. An unplaced node (core_index < 0) holds
// no cores: it is not pinned, so there is nothing to exclude it from.
static inline uint64_t CoreMaskFor(const DispatchNode &n) {
	uint64_t m = 0;
	if (!n.core_indices.empty()) {
		for (int c : n.core_indices) {
			if (c >= 0 && c < 64)
				m |= (uint64_t)1 << c;
		}
	} else if (n.core_index >= 0 && n.core_index < 64) {
		m |= (uint64_t)1 << n.core_index;
	}
	return m;
}

static inline bool IsShard(const DispatchNode &n) {
	return n.core_indices.size() > 1;
}

static std::vector<int> &ReadyQueueFor(SchedulerShared *s, HardwareTarget t) {
	return (t == HardwareTarget::kCpuP) ? s->ready_p : s->ready_e;
}

static std::vector<int> &FutureQueueFor(SchedulerShared *s, HardwareTarget t) {
	return (t == HardwareTarget::kCpuP) ? s->future_p : s->future_e;
}

static void InsertFutureSorted(std::vector<int> *q,
	const std::vector<DispatchNode> &nodes,
	const std::vector<NodeExecState> &exec, int node_idx) {
	auto it = q->begin();
	for (; it != q->end(); ++it) {
		const uint64_t a = exec[(size_t)node_idx].release_us;
		const uint64_t b = exec[(size_t)*it].release_us;
		if (a < b)
			break;
		if (a == b) {
			const auto &A = nodes[(size_t)node_idx];
			const auto &B = nodes[(size_t)*it];
			if (A.start_time_ms < B.start_time_ms)
				break;
			if (A.start_time_ms == B.start_time_ms && A.key < B.key)
				break;
		}
	}
	q->insert(it, node_idx);
}

static void PromoteReleasedNodesLocked(SchedulerShared *sched,
	const std::vector<DispatchNode> &nodes, HardwareTarget target,
	uint64_t now_us) {
	std::vector<int> &future = FutureQueueFor(sched, target);
	std::vector<int> &ready = ReadyQueueFor(sched, target);

	size_t i = 0;
	while (i < future.size()) {
		const int node_idx = future[i];
		if (sched->exec[(size_t)node_idx].release_us > now_us)
			break;
		ready.push_back(node_idx);
		future.erase(future.begin() + i);
	}
}

static uint64_t NextReleaseUsLocked(
	SchedulerShared *sched, HardwareTarget target) {
	const std::vector<int> &future = FutureQueueFor(sched, target);
	if (future.empty())
		return UINT64_MAX;
	return sched->exec[(size_t)future.front()].release_us;
}

// Pick the best ready node this worker may run.
//
// worker_core is the physical core index this worker owns, or -1 when the
// worker serves a whole target (the historical one-thread-per-cluster mode).
// A core-owning worker takes only nodes placed on its core, plus -- if it owns
// the target's first core -- nodes the schedule left unplaced (core_index < 0),
// so an unplaced node can never strand with every worker refusing it.
//
// Returns an index into `ready`, or -1 when nothing here belongs to us.
static int PickBestReadyIndex(const std::vector<int> &ready,
	const std::vector<DispatchNode> &nodes, int worker_core = -1,
	bool accepts_unplaced = true, const SchedulerShared *sched = nullptr) {
	auto eligible = [&](int node_idx) {
		const DispatchNode &n = nodes[(size_t)node_idx];
		if (worker_core >= 0) {
			// A shard is owned by the worker that owns its FIRST core, so
			// exactly one worker will ever try to start it.
			const int c = n.core_index;
			if (c < 0) {
				if (!accepts_unplaced)
					return false;
			} else if (c != worker_core) {
				return false;
			}
		}
		if (!sched)
			return true;
		const uint64_t mask = CoreMaskFor(n);
		if (!mask)
			return true;  // unplaced: occupies no named core
		const int cl = ClusterIndex(n.hardware_target);
		if (sched->busy_cores[cl] & mask)
			return false;
		if (!IsShard(n) && (sched->wanted_cores[cl] & mask))
			return false;  // yield to a shard already waiting on this core
		return true;
	};
	int best_i = -1;
	for (int i = 0; i < (int)ready.size(); ++i) {
		if (!eligible(ready[i]))
			continue;
		if (best_i < 0) {
			best_i = i;
			continue;
		}
		const auto &A = nodes[(size_t)ready[i]];
		const auto &B = nodes[(size_t)ready[best_i]];

		if (A.start_time_ms != B.start_time_ms) {
			if (A.start_time_ms < B.start_time_ms)
				best_i = i;
			continue;
		}
		if (A.id != B.id) {
			if (A.id < B.id)
				best_i = i;
			continue;
		}
		if (A.ordinal != B.ordinal) {
			if (A.ordinal < B.ordinal)
				best_i = i;
			continue;
		}
		if (A.key < B.key)
			best_i = i;
	}
	return best_i;
}

static void SeedReadyNodes(
	const std::vector<DispatchNode> &nodes, SchedulerShared *sched) {
	sched->ready_p.clear();
	sched->ready_e.clear();
	sched->future_p.clear();
	sched->future_e.clear();
	sched->completed = 0;

	for (size_t i = 0; i < nodes.size(); ++i) {
		if (sched->remaining_preds[i] != 0)
			continue;

		NodeExecState &xs = sched->exec[i];
		xs.enqueued = true;

		if (nodes[i].release_policy == ReleasePolicy::kPhaseLocked) {
			xs.release_us = xs.planned_start_us;
		} else {
			xs.release_us = 0;
		}
		xs.ready_us = xs.release_us;

		if (xs.release_us == 0) {
			ReadyQueueFor(sched, nodes[i].hardware_target).push_back((int)i);
		} else {
			InsertFutureSorted(&FutureQueueFor(sched, nodes[i].hardware_target),
				nodes, sched->exec, (int)i);
		}
	}
}

// Let a worker reserve the cores its own shard is waiting for.
//
// Without this, a four-core shard on cores 0-3 can wait indefinitely: cores 1-3
// keep accepting 60 us MLP dispatches, so all four are never simultaneously
// free. The shard's owning worker therefore marks them wanted, and single-core
// work declines to start on a wanted core. The claim is advisory and is dropped
// the moment the shard actually starts, or the worker leaves.
static void ClaimForWaitingShardLocked(SchedulerShared *sched,
	const std::vector<int> &ready, const std::vector<DispatchNode> &nodes,
	int worker_core, bool accepts_unplaced, uint64_t *claimed_mask,
	int *claimed_cluster) {
	if (*claimed_mask) {
		sched->wanted_cores[*claimed_cluster] &= ~*claimed_mask;
		*claimed_mask = 0;
	}
	if (worker_core < 0)
		return;
	for (int node_idx : ready) {
		const DispatchNode &n = nodes[(size_t)node_idx];
		if (!IsShard(n))
			continue;
		if (n.core_index != worker_core) {
			if (n.core_index >= 0 || !accepts_unplaced)
				continue;
		}
		const uint64_t mask = CoreMaskFor(n);
		if (!mask)
			continue;
		*claimed_cluster = ClusterIndex(n.hardware_target);
		*claimed_mask = mask;
		sched->wanted_cores[*claimed_cluster] |= mask;
		return;
	}
}

//------------------------------------------------------------------------------
// Worker thread
//------------------------------------------------------------------------------

static void WorkerMain(HardwareTarget target, const char *cpu_ids_csv,
	int worker_core, bool accepts_unplaced,
	std::vector<DispatchNode> *nodes,
	const std::vector<std::vector<int>> *dependents,
	const std::vector<CachedModule *> *node_modules, int dispatch_iters,
	iree_allocator_t host_alloc, SharedState *fatal, SchedulerShared *sched,
	TraceWriter *trace) {
	BestEffortPinCurrentThreadToCpuIds(cpu_ids_csv);

	uint64_t claimed_mask = 0;
	int claimed_cluster = 0;

	while (true) {
		int node_idx = -1;
		int graph_iter = 0;
		Clock::time_point iter_t0;
		uint64_t held_mask = 0;
		int held_cluster = 0;

		{
			std::unique_lock<std::mutex> lock(sched->mu);

			while (true) {
				if (sched->shutdown || HasFatal(fatal)) {
					if (claimed_mask) {
						sched->wanted_cores[claimed_cluster] &= ~claimed_mask;
						sched->cv.notify_all();
					}
					return;
				}
				if (!sched->active) {
					sched->cv.wait(lock);
					continue;
				}

				const uint64_t now_us = UsSince(sched->iter_t0, Clock::now());
				PromoteReleasedNodesLocked(sched, *nodes, target, now_us);

				std::vector<int> &ready = ReadyQueueFor(sched, target);
				const int best_i = PickBestReadyIndex(
					ready, *nodes, worker_core, accepts_unplaced, sched);
				if (best_i >= 0) {
					node_idx = ready[(size_t)best_i];
					ready.erase(ready.begin() + best_i);

					// Take the cores before releasing the lock; from here on
					// no other worker can start anything that overlaps them.
					const DispatchNode &picked = (*nodes)[(size_t)node_idx];
					held_mask = CoreMaskFor(picked);
					held_cluster = ClusterIndex(picked.hardware_target);
					sched->busy_cores[held_cluster] |= held_mask;
					sched->wanted_cores[held_cluster] &= ~held_mask;
					claimed_mask = 0;

					graph_iter = sched->current_graph_iter;
					iter_t0 = sched->iter_t0;

					NodeExecState &xs = sched->exec[(size_t)node_idx];
					xs.running = true;
					xs.start_us = UsSince(iter_t0, Clock::now());
					break;
				}

				// Nothing runnable. If the only thing keeping this worker idle
				// is that a shard of ours is waiting for busy cores, claim
				// those cores so short work stops jumping the queue.
				ClaimForWaitingShardLocked(sched, ready, *nodes, worker_core,
					accepts_unplaced, &claimed_mask, &claimed_cluster);

				const uint64_t next_release_us =
					NextReleaseUsLocked(sched, target);
				if (next_release_us == UINT64_MAX) {
					sched->cv.wait(lock);
				} else {
					// Spin-wait for short sleeps to avoid condvar
					// timer overshoot (~2ms on RISC-V kernel).
					const uint64_t now2 = UsSince(sched->iter_t0, Clock::now());
					if (next_release_us > now2 + 5000) {
						// Long wait: condvar, but wake 2ms early to spin.
						const auto wake_tp = sched->iter_t0 +
							std::chrono::microseconds(next_release_us - 2000);
						sched->cv.wait_until(lock, wake_tp);
					} else {
						// Short wait: drop lock, spin-yield, re-acquire.
						lock.unlock();
						while (UsSince(sched->iter_t0, Clock::now()) <
							next_release_us) {
							sched_yield();
						}
						lock.lock();
					}
				}
			}
		}

		iree_status_t st = CallModuleUnlocked((*node_modules)[(size_t)node_idx],
			(int32_t)dispatch_iters, host_alloc);

		const uint64_t end_us = UsSince(iter_t0, Clock::now());

		if (!iree_status_is_ok(st)) {
			{
				// Give the cores back even on the failure path, so the other
				// workers wind down instead of blocking on a dead shard.
				std::lock_guard<std::mutex> lock(sched->mu);
				sched->busy_cores[held_cluster] &= ~held_mask;
			}
			SetFatalOnce(
				fatal, st, "[dispatch] sync benchmark module call failed");
			sched->cv.notify_all();
			return;
		}

		uint64_t planned_start_us = 0;
		uint64_t ready_us = 0;
		uint64_t start_us = 0;

		{
			std::lock_guard<std::mutex> lock(sched->mu);
			NodeExecState &xs = sched->exec[(size_t)node_idx];
			xs.running = false;
			xs.done = true;
			xs.end_us = end_us;

			planned_start_us = xs.planned_start_us;
			ready_us = xs.ready_us;
			start_us = xs.start_us;

			sched->busy_cores[held_cluster] &= ~held_mask;

			const uint64_t run_us =
				end_us >= start_us ? (end_us - start_us) : 0;
			(*nodes)[(size_t)node_idx].run_stats.Add(run_us);

			for (int child : (*dependents)[(size_t)node_idx]) {
				int &rem = sched->remaining_preds[(size_t)child];
				rem--;
				if (rem == 0) {
					NodeExecState &cs = sched->exec[(size_t)child];
					cs.enqueued = true;
					cs.release_us = end_us;
					cs.ready_us = cs.release_us;

					if (cs.release_us <= UsSince(iter_t0, Clock::now())) {
						ReadyQueueFor(
							sched, (*nodes)[(size_t)child].hardware_target)
							.push_back(child);
					} else {
						InsertFutureSorted(
							&FutureQueueFor(
								sched, (*nodes)[(size_t)child].hardware_target),
							*nodes, sched->exec, child);
					}
				}
			}

			sched->completed++;
			if (sched->completed == sched->total_nodes) {
				sched->active = false;
			}
		}

		trace->WriteRow(graph_iter, (*nodes)[(size_t)node_idx],
			planned_start_us, ready_us, start_us, end_us);

		sched->cv.notify_all();
	}
}

//------------------------------------------------------------------------------
// Summary JSON output (sample-specific config section)
//------------------------------------------------------------------------------

static bool WriteSummaryJson(const char *path,
	const scheduler_runner_config_t *cfg, const GraphModel &model,
	const std::vector<int> &topo_order) {
	if (!path || !path[0])
		return true;
	FILE *f = fopen(path, "wb");
	if (!f) {
		fprintf(stderr, "Failed to open out_json: %s\n", path);
		return false;
	}

	fprintf(f, "{\n");
	fprintf(f, "  \"config\": {\n");
	fprintf(f, "    \"graph_json_path\": ");
	JsonWriteEscaped(f, cfg->graph_json_path ? cfg->graph_json_path : "");
	fprintf(f, ",\n");
	fprintf(f, "    \"driver\": ");
	JsonWriteEscaped(f, cfg->driver_name ? cfg->driver_name : "");
	fprintf(f, ",\n");
	fprintf(f, "    \"graph_iters\": %d,\n", cfg->graph_iters);
	fprintf(f, "    \"dispatch_iters\": %d,\n", cfg->dispatch_iters);
	fprintf(f, "    \"report_every\": %d,\n", cfg->report_every);
	fprintf(f, "    \"vmfb_root_dir\": ");
	JsonWriteEscaped(f, cfg->vmfb_root_dir ? cfg->vmfb_root_dir : "");
	fprintf(f, ",\n");
	fprintf(f, "    \"cpu_p_cpu_ids\": ");
	JsonWriteEscaped(f, cfg->cpu_p_cpu_ids ? cfg->cpu_p_cpu_ids : "");
	fprintf(f, ",\n");
	fprintf(f, "    \"cpu_e_cpu_ids\": ");
	JsonWriteEscaped(f, cfg->cpu_e_cpu_ids ? cfg->cpu_e_cpu_ids : "");
	fprintf(f, ",\n");
	fprintf(f, "    \"visible_cores\": %d,\n", cfg->visible_cores);
	fprintf(f, "    \"schedule_makespan_ms\": %.6f\n", model.makespan_ms);
	fprintf(f, "  },\n");

	WriteNodesJson(f, model.nodes);
	fprintf(f, ",\n");
	WriteTopoOrderJson(f, model.nodes, topo_order);
	fprintf(f, "\n}\n");

	fclose(f);
	return true;
}

} // namespace

//------------------------------------------------------------------------------
// Entry point
//------------------------------------------------------------------------------

extern "C" int scheduler_runner_run(const scheduler_runner_config_t *cfg) {
	using namespace merlin_bench;

	if (!cfg || !cfg->graph_json_path || !cfg->graph_json_path[0]) {
		fprintf(stderr, "scheduler_runner_run: missing graph_json_path\n");
		return 1;
	}

	const char *driver = (cfg->driver_name && cfg->driver_name[0])
		? cfg->driver_name
		: "local-task";
	const int graph_iters = (cfg->graph_iters > 0) ? cfg->graph_iters : 1;
	const int dispatch_iters =
		(cfg->dispatch_iters > 0) ? cfg->dispatch_iters : 1;
	const int report_every = (cfg->report_every >= 0) ? cfg->report_every : 0;

	if (strcmp(driver, "local-task") != 0) {
		fprintf(stderr, "This scheduler requires driver=local-task; got '%s'\n",
			driver);
		return 1;
	}
	if (!ValidateCorePartition(cfg))
		return 1;

	fprintf(stdout,
		"Dispatch scheduler (sync benchmark VMFBs):\n"
		"  json          = %s\n"
		"  driver        = %s\n"
		"  graph_iters   = %d\n"
		"  dispatch_iters= %d\n"
		"  report_every  = %d\n"
		"  vmfb_root_dir = %s\n"
		"  CPU_P cores   = %s\n"
		"  CPU_E cores   = %s\n"
		"  visible_cores = %d\n"
		"  out_json      = %s\n"
		"  out_dot       = %s\n"
		"  trace_csv     = %s\n",
		cfg->graph_json_path, driver, graph_iters, dispatch_iters, report_every,
		cfg->vmfb_root_dir ? cfg->vmfb_root_dir : "",
		cfg->cpu_p_cpu_ids ? cfg->cpu_p_cpu_ids : "",
		cfg->cpu_e_cpu_ids ? cfg->cpu_e_cpu_ids : "", cfg->visible_cores,
		cfg->out_json_path ? cfg->out_json_path : "",
		cfg->out_dot_path ? cfg->out_dot_path : "",
		cfg->trace_csv_path ? cfg->trace_csv_path : "");
	fflush(stdout);

	GraphModel model;
	if (!ParseDispatchScheduleJson(cfg->graph_json_path, &model)) {
		fprintf(stderr, "Failed to parse schedule JSON: %s\n",
			cfg->graph_json_path);
		return 1;
	}

	InferSchedulingPolicies(&model.nodes);
	ExpandAllPredecessors(&model.nodes);

	if (model.makespan_ms <= 0.0) {
		double max_end_ms = 0.0;
		for (const auto &n : model.nodes) {
			const double end_ms = n.start_time_ms + n.planned_duration_ms;
			if (end_ms > max_end_ms)
				max_end_ms = end_ms;
		}
		model.makespan_ms = max_end_ms;
	}

	const std::string json_dir = PathDirname(cfg->graph_json_path);
	for (auto &n : model.nodes) {
		n.vmfb_path_resolved =
			ResolveVmfbPath(cfg->vmfb_root_dir, cfg->target_platform, json_dir,
				n, cfg->variant_p_dir, cfg->variant_e_dir, cfg->elf_marker);
		if (n.vmfb_path_resolved.empty()) {
			fprintf(
				stderr, "Unable to resolve VMFB for node %s\n", n.key.c_str());
			return 1;
		}
		if (!FileReadable(n.vmfb_path_resolved)) {
			fprintf(stderr,
				"VMFB not readable for node %s:\n"
				"  module_name        = %s\n"
				"  vmfb_path_json     = %s\n"
				"  vmfb_path_resolved = %s\n",
				n.key.c_str(), n.module_name.c_str(), n.vmfb_path_json.c_str(),
				n.vmfb_path_resolved.c_str());
			return 1;
		}
	}

	std::vector<int> topo_order;
	std::vector<std::vector<int>> dependents;
	if (!TopoSort(model.nodes, &topo_order, &dependents))
		return 1;

	fprintf(stdout, "Submit priority order (%zu nodes):\n", topo_order.size());
	for (size_t i = 0; i < topo_order.size(); ++i) {
		const auto &n = model.nodes[static_cast<size_t>(topo_order[i])];
		fprintf(stdout, "  %zu) %s target=%s start=%.3fms dur=%.3fms\n", i + 1,
			n.key.c_str(), HardwareTargetName(n.hardware_target),
			n.start_time_ms, n.planned_duration_ms);
	}
	fflush(stdout);

	SharedState shared;
	iree_allocator_t host_alloc = iree_allocator_system();
	iree_runtime_instance_t *instance = nullptr;
	iree_hal_device_t *device_p = nullptr;
	iree_hal_device_t *device_e = nullptr;
	// (target, core_index) -> device pinned to that one physical core.
	// Populated only when cfg->pin_per_core is set.
	std::map<std::pair<int, int>, iree_hal_device_t *> per_core_devices;
	// (target, core bitmask) -> device pinned to every core of a shard, so a
	// dispatch placed on "CPU_P#0+CPU_P#1+CPU_P#2+CPU_P#3" gets a local-task
	// topology of four harts and IREE really does distribute its workgroups.
	// Measured on the K1: DroNet's 22.8 ms convolution becomes 6.1 ms.
	std::map<std::pair<int, uint64_t>, iree_hal_device_t *> shard_devices;

	TraceWriter trace;
	if (cfg->trace_csv_path && cfg->trace_csv_path[0]) {
		if (!trace.Open(cfg->trace_csv_path)) {
			fprintf(
				stderr, "Failed to open trace_csv: %s\n", cfg->trace_csv_path);
			return 1;
		}
	}

	{
		iree_runtime_instance_options_t opts;
		iree_runtime_instance_options_initialize(&opts);
		iree_runtime_instance_options_use_all_available_drivers(&opts);

		iree_status_t st =
			iree_runtime_instance_create(&opts, host_alloc, &instance);
		if (!iree_status_is_ok(st)) {
			iree_status_fprint(stderr, st);
			iree_status_ignore(st);
			trace.Close();
			return 1;
		}
	}

	{
		iree_status_t st = CreatePinnedLocalTaskDevice(
			host_alloc, cfg->cpu_p_cpu_ids, &device_p);
		if (!iree_status_is_ok(st)) {
			fprintf(stderr, "Failed creating pinned CPU_P device\n");
			iree_status_fprint(stderr, st);
			iree_status_ignore(st);
			iree_runtime_instance_release(instance);
			trace.Close();
			return 1;
		}
	}

	{
		iree_status_t st = CreatePinnedLocalTaskDevice(
			host_alloc, cfg->cpu_e_cpu_ids, &device_e);
		if (!iree_status_is_ok(st)) {
			fprintf(stderr, "Failed creating pinned CPU_E device\n");
			iree_status_fprint(stderr, st);
			iree_status_ignore(st);
			iree_hal_device_release(device_p);
			iree_runtime_instance_release(instance);
			trace.Close();
			return 1;
		}
	}

	if (cfg->pin_per_core) {
		// One device per physical core, so a dispatch placed on CPU_P#2 runs
		// on CPU_P's third core and nowhere else. Without this the core index
		// is parsed and discarded, and IREE's local-task picks any core in the
		// cluster pool -- which silently invalidates a schedule built from
		// single-core profiles.
		std::vector<int> ids_p, ids_e;
		SplitCpuIds(cfg->cpu_p_cpu_ids, &ids_p);
		SplitCpuIds(cfg->cpu_e_cpu_ids, &ids_e);
		const std::vector<int> *sets[2] = {&ids_p, &ids_e};
		for (int t = 0; t < 2; ++t) {
			for (size_t k = 0; k < sets[t]->size(); ++k) {
				char one[16];
				snprintf(one, sizeof(one), "%d", (*sets[t])[k]);
				iree_hal_device_t *dev = nullptr;
				iree_status_t st =
					CreatePinnedLocalTaskDevice(host_alloc, one, &dev);
				if (!iree_status_is_ok(st)) {
					fprintf(stderr,
						"Failed creating per-core device for cpu %s\n", one);
					iree_status_fprint(stderr, st);
					iree_status_ignore(st);
					continue;
				}
				per_core_devices[std::make_pair(t, (int)k)] = dev;
			}
		}
		fprintf(stdout, "[dispatch] per-core pinning ON (%zu devices)\n",
			per_core_devices.size());

		// One device per distinct shard placement actually present in the
		// schedule. Built from the graph rather than from every possible core
		// subset: a device costs threads, and only the placements the
		// scheduler chose can ever be used.
		const std::vector<int> *phys[2] = {&ids_p, &ids_e};
		std::map<std::pair<int, uint64_t>, std::string> shard_csv;
		for (const DispatchNode &n : model.nodes) {
			if (!IsShard(n))
				continue;
			const int cl = ClusterIndex(n.hardware_target);
			const uint64_t mask = CoreMaskFor(n);
			if (!mask || shard_csv.count(std::make_pair(cl, mask)))
				continue;
			std::string csv;
			bool ok = true;
			for (int idx : n.core_indices) {
				if (idx < 0 || (size_t)idx >= phys[cl]->size()) {
					ok = false;
					break;
				}
				if (!csv.empty())
					csv += ",";
				csv += std::to_string((*phys[cl])[(size_t)idx]);
			}
			if (!ok) {
				fprintf(stderr,
					"[dispatch] shard %s names a core outside %s's cpu id "
					"list; refusing to guess an affinity for it\n",
					n.key.c_str(), HardwareTargetName(n.hardware_target));
				trace.Close();
				return 1;
			}
			shard_csv[std::make_pair(cl, mask)] = csv;
		}
		for (const auto &kv : shard_csv) {
			iree_hal_device_t *dev = nullptr;
			iree_status_t st = CreatePinnedLocalTaskDevice(
				host_alloc, kv.second.c_str(), &dev);
			if (!iree_status_is_ok(st)) {
				fprintf(stderr,
					"Failed creating shard device for cpus %s\n",
					kv.second.c_str());
				iree_status_fprint(stderr, st);
				iree_status_ignore(st);
				trace.Close();
				return 1;
			}
			shard_devices[kv.first] = dev;
			fprintf(stdout, "[dispatch] shard device cpus={%s}\n",
				kv.second.c_str());
		}
	}

	fprintf(stdout,
		"[dispatch] CPU_P local-task topology = {%s}\n"
		"[dispatch] CPU_E local-task topology = {%s}\n",
		cfg->cpu_p_cpu_ids, cfg->cpu_e_cpu_ids);
	fflush(stdout);

	// Cache one session per (target, vmfb_path).
	std::unordered_map<std::string, std::unique_ptr<CachedModule>> cache;
	cache.reserve(model.nodes.size() * 2);

	auto device_for = [&](const DispatchNode &n) -> iree_hal_device_t * {
		iree_hal_device_t *fallback =
			(n.hardware_target == HardwareTarget::kCpuP) ? device_p : device_e;
		if (!cfg->pin_per_core || n.core_index < 0)
			return fallback;
		const int cl = ClusterIndex(n.hardware_target);
		if (IsShard(n)) {
			auto sit = shard_devices.find(
				std::make_pair(cl, CoreMaskFor(n)));
			// Never silently fall back to the cluster device here: that would
			// run the shard on 4 harts by accident on a 4-core cluster and on
			// the wrong number anywhere else, and the trace would look fine.
			return sit == shard_devices.end() ? nullptr : sit->second;
		}
		auto it = per_core_devices.find(std::make_pair(cl, n.core_index));
		return it == per_core_devices.end() ? fallback : it->second;
	};

	// The core is part of the key: a module is loaded against a device, so two
	// cores running the same VMFB need two cached modules.
	auto cache_key_for = [&](const DispatchNode &n, const std::string &path) {
		std::string key = std::string(HardwareTargetName(n.hardware_target)) +
			"#" + std::to_string(cfg->pin_per_core ? n.core_index : -1);
		// A shard is loaded against a differently-pinned device than the single
		// core of the same index, so it needs its own cache entry -- sharing
		// one would run the shard on a one-hart device and quietly delete the
		// speedup the whole rung is measuring.
		if (cfg->pin_per_core && IsShard(n))
			key += "x" + std::to_string(n.core_indices.size());
		return key + "|" + path;
	};

	std::vector<CachedModule *> node_modules(model.nodes.size(), nullptr);

	for (size_t i = 0; i < model.nodes.size(); ++i) {
		DispatchNode &node = model.nodes[i];
		const std::string cache_key =
			cache_key_for(node, node.vmfb_path_resolved);

		auto it = cache.find(cache_key);
		if (it == cache.end()) {
			auto cm = std::make_unique<CachedModule>();
			iree_hal_device_t *target_device = device_for(node);
			if (!target_device) {
				fprintf(stderr,
					"No device for node %s: it is placed on a shard with no "
					"matching pinned device (was --pin_per_core set?)\n",
					node.key.c_str());
				trace.Close();
				return 1;
			}

			iree_status_t st = LoadModule(
				instance, target_device, node.vmfb_path_resolved, cm.get());
			if (!iree_status_is_ok(st)) {
				fprintf(stderr, "Failed loading VMFB for node %s\n",
					node.key.c_str());
				iree_status_fprint(stderr, st);
				iree_status_ignore(st);

				trace.Close();
				if (device_e)
					iree_hal_device_release(device_e);
				if (device_p)
					iree_hal_device_release(device_p);
				if (instance)
					iree_runtime_instance_release(instance);
				return 1;
			}

			node_modules[i] = cm.get();
			cache.emplace(cache_key, std::move(cm));
		} else {
			node_modules[i] = it->second.get();
		}
	}

	SchedulerShared sched;
	sched.total_nodes = model.nodes.size();

	// Pre-warm every unique cached module.
	fprintf(stdout, "[dispatch] Pre-warming %zu unique cached modules...\n",
		cache.size());
	fflush(stdout);

	for (auto &kv : cache) {
		CachedModule *cm = kv.second.get();

		const int32_t warm_iters = (cm->arity == 1 && cm->first_is_i32)
			? 0
			: static_cast<int32_t>(dispatch_iters);

		iree_status_t st = CallModuleUnlocked(cm, warm_iters, host_alloc);
		if (!iree_status_is_ok(st)) {
			fprintf(
				stderr, "Warmup failed for VMFB: %s\n", cm->vmfb_path.c_str());
			iree_status_fprint(stderr, st);
			iree_status_ignore(st);

			trace.Close();
			for (auto &cache_kv : cache)
				CachedModuleRelease(cache_kv.second.get());
			if (device_e)
				iree_hal_device_release(device_e);
			if (device_p)
				iree_hal_device_release(device_p);
			if (instance)
				iree_runtime_instance_release(instance);
			return 1;
		}
	}

	fprintf(stdout, "[dispatch] Warmup complete.\n");
	fflush(stdout);

	// One worker per core when per-core pinning is on, otherwise one per
	// target -- the historical shape.
	//
	// N-way is only safe together with per-core devices, and that is not a
	// coincidence: a CachedModule wraps an IREE session, and two threads
	// calling into one session concurrently is a data race. With
	// --pin_per_core the module cache is keyed by core, so each module belongs
	// to exactly one core, and giving each core exactly one worker means each
	// session is touched by exactly one thread. Spawning N workers per target
	// WITHOUT per-core devices would share sessions across threads, which is
	// why that combination is deliberately not offered.
	std::vector<std::thread> workers;
	// Backing storage for the per-worker cpu-id strings. WorkerMain takes a
	// const char*, so these must outlive the threads; a deque never
	// reallocates its elements, so pointers taken from it stay valid.
	std::deque<std::string> worker_cpu_csv;
	{
		std::vector<int> ids_p, ids_e;
		SplitCpuIds(cfg->cpu_p_cpu_ids, &ids_p);
		SplitCpuIds(cfg->cpu_e_cpu_ids, &ids_e);
		struct TargetSpec {
			HardwareTarget target;
			const char *csv;
			const std::vector<int> *ids;
		};
		const TargetSpec specs[2] = {
			{HardwareTarget::kCpuP, cfg->cpu_p_cpu_ids, &ids_p},
			{HardwareTarget::kCpuE, cfg->cpu_e_cpu_ids, &ids_e},
		};
		for (const TargetSpec &ts : specs) {
			if (!cfg->pin_per_core || ts.ids->empty()) {
				workers.emplace_back(WorkerMain, ts.target, ts.csv,
					/*worker_core=*/-1, /*accepts_unplaced=*/true,
					&model.nodes, &dependents, &node_modules, dispatch_iters,
					host_alloc, &shared, &sched, &trace);
				continue;
			}
			for (size_t k = 0; k < ts.ids->size(); ++k) {
				// Pin this worker to the one physical core it serves, so the
				// thread that submits also runs on the core the schedule named.
				worker_cpu_csv.emplace_back(std::to_string((*ts.ids)[k]));
				const char *csv = worker_cpu_csv.back().c_str();
				workers.emplace_back(WorkerMain, ts.target, csv,
					/*worker_core=*/(int)k,
					/*accepts_unplaced=*/(k == 0),
					&model.nodes, &dependents, &node_modules, dispatch_iters,
					host_alloc, &shared, &sched, &trace);
			}
		}
		fprintf(stdout, "[dispatch] %zu worker thread(s)%s\n", workers.size(),
			cfg->pin_per_core ? " (one per core)" : " (one per target)");
		fflush(stdout);
	}

	const auto run_t0 = Clock::now();

	for (int gi = 0; gi < graph_iters && !HasFatal(&shared); ++gi) {
		{
			std::lock_guard<std::mutex> lock(sched.mu);
			sched.current_graph_iter = gi;
			sched.iter_t0 = Clock::now();
			sched.active = true;
			sched.completed = 0;

			sched.remaining_preds.assign(model.nodes.size(), 0);
			sched.exec.assign(model.nodes.size(), NodeExecState{});

			for (size_t i = 0; i < model.nodes.size(); ++i) {
				sched.remaining_preds[i] =
					static_cast<int>(model.nodes[i].all_predecessors.size());
				sched.exec[i].planned_start_us =
					MsToUs(model.nodes[i].start_time_ms);
			}

			SeedReadyNodes(model.nodes, &sched);
		}
		sched.cv.notify_all();

		{
			std::unique_lock<std::mutex> lock(sched.mu);
			sched.cv.wait(lock, [&]() {
				return HasFatal(&shared) ||
					sched.completed == sched.total_nodes;
			});
			sched.active = false;
			sched.ready_p.clear();
			sched.ready_e.clear();
		}
		sched.cv.notify_all();

		if (report_every > 0 && ((gi + 1) % report_every) == 0 &&
			!HasFatal(&shared)) {
			fprintf(stdout, "[graph_iter %d/%d]\n", gi + 1, graph_iters);
			for (int idx : topo_order) {
				const auto &n = model.nodes[static_cast<size_t>(idx)];
				fprintf(stdout,
					"  %s target=%s plan=%.3fms run_avg=%.3fms p90=%.3fms "
					"max=%.3fms\n",
					n.key.c_str(), HardwareTargetName(n.hardware_target),
					n.planned_duration_ms, n.run_stats.AvgMs(),
					n.run_stats.P90Ms(), n.run_stats.MaxMs());
			}
			fflush(stdout);
		}
	}

	{
		std::lock_guard<std::mutex> lock(sched.mu);
		sched.shutdown = true;
		sched.active = false;
		sched.ready_p.clear();
		sched.ready_e.clear();
	}
	sched.cv.notify_all();

	for (auto &w : workers)
		w.join();

	const auto run_t1 = Clock::now();
	const double total_s =
		std::chrono::duration_cast<std::chrono::duration<double>>(
			run_t1 - run_t0)
			.count();

	if (!HasFatal(&shared)) {
		fprintf(stdout,
			"Run complete:\n"
			"  total_wall_ms=%.3f\n"
			"  schedule_makespan_ms=%.3f\n"
			"  completed_graph_iters=%d\n",
			total_s * 1000.0, model.makespan_ms, graph_iters);

		for (int idx : topo_order) {
			const auto &n = model.nodes[static_cast<size_t>(idx)];
			fprintf(stdout,
				"  %s target=%s plan=%.3fms run_avg=%.3fms p50=%.3fms "
				"p90=%.3fms "
				"p99=%.3fms min=%.3fms max=%.3fms\n",
				n.key.c_str(), HardwareTargetName(n.hardware_target),
				n.planned_duration_ms, n.run_stats.AvgMs(), n.run_stats.P50Ms(),
				n.run_stats.P90Ms(), n.run_stats.P99Ms(), n.run_stats.MinMs(),
				n.run_stats.MaxMs());
		}
		fprintf(stdout, "Done.\n");
		fflush(stdout);
	}

	bool ok_write = true;
	ok_write = ok_write &&
		WriteSummaryJson(cfg->out_json_path, cfg, model, topo_order);
	ok_write = ok_write && WriteDotGraph(cfg->out_dot_path, model);
	if (!ok_write) {
		fprintf(stderr, "Warning: failed writing one or more outputs\n");
	}

	trace.Close();

	for (auto &kv : cache)
		CachedModuleRelease(kv.second.get());
	for (auto &kv : per_core_devices) {
		if (kv.second)
			iree_hal_device_release(kv.second);
	}
	for (auto &kv : shard_devices) {
		if (kv.second)
			iree_hal_device_release(kv.second);
	}
	if (device_e)
		iree_hal_device_release(device_e);
	if (device_p)
		iree_hal_device_release(device_p);
	if (instance)
		iree_runtime_instance_release(instance);

	return HasFatal(&shared) ? 1 : 0;
}
