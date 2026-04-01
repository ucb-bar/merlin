// samples/common/xpu-rt/scheduler_runner.cc
//
// Generic N-target dispatch scheduler.
//
// - N long-lived worker threads, one per hardware target.
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

static bool ValidateCorePartition(
	const TargetRegistry &reg, int visible_cores) {
	const int vc = visible_cores > 0 ? visible_cores : 64;
	std::unordered_set<int> seen;

	for (int t = 0; t < reg.Size(); ++t) {
		const char *cpu_ids = reg.CpuIds(static_cast<TargetId>(t));
		if (!cpu_ids || !cpu_ids[0]) {
			fprintf(stderr, "Target %s: cpu_ids is required\n",
				reg.Name(static_cast<TargetId>(t)));
			return false;
		}
		std::vector<int> ids;
		if (!SplitCpuIds(cpu_ids, &ids)) {
			fprintf(stderr, "Target %s: invalid cpu_ids '%s'\n",
				reg.Name(static_cast<TargetId>(t)), cpu_ids);
			return false;
		}
		for (int v : ids) {
			if (v < 0 || v >= vc) {
				fprintf(stderr, "Target %s: core %d out of range [0,%d)\n",
					reg.Name(static_cast<TargetId>(t)), v, vc);
				return false;
			}
			if (!seen.insert(v).second) {
				fprintf(stderr,
					"Target %s: core %d overlaps another target or is "
					"duplicated\n",
					reg.Name(static_cast<TargetId>(t)), v);
				return false;
			}
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
	// Per-target condvars so workers only wake when their target has work.
	// This avoids spurious cross-target wakeups that cause context switches
	// on executor cores.
	std::vector<std::unique_ptr<std::condition_variable>> target_cv;
	std::condition_variable main_cv; // For main thread waiting on completion.

	bool shutdown = false;
	bool active = false;

	int current_graph_iter = 0;
	Clock::time_point iter_t0{};

	std::vector<int> remaining_preds;
	std::vector<NodeExecState> exec;

	// Per-target ready and future queues indexed by TargetId.
	std::vector<std::vector<int>> ready_queues;
	std::vector<std::vector<int>> future_queues;

	size_t completed = 0;
	size_t total_nodes = 0;

	void InitQueues(int num_targets) {
		ready_queues.assign(num_targets, {});
		future_queues.assign(num_targets, {});
		target_cv.resize(num_targets);
		for (int i = 0; i < num_targets; ++i) {
			if (!target_cv[i])
				target_cv[i] = std::make_unique<std::condition_variable>();
		}
	}

	void ClearQueues() {
		for (auto &q : ready_queues)
			q.clear();
		for (auto &q : future_queues)
			q.clear();
	}

	void NotifyTarget(TargetId t) {
		target_cv[t]->notify_one();
	}

	void NotifyAll() {
		for (auto &cv : target_cv)
			cv->notify_one();
		main_cv.notify_all();
	}
};

static std::vector<int> &ReadyQueueFor(SchedulerShared *s, TargetId target) {
	return s->ready_queues[target];
}

static std::vector<int> &FutureQueueFor(SchedulerShared *s, TargetId target) {
	return s->future_queues[target];
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
	const std::vector<DispatchNode> &nodes, TargetId target, uint64_t now_us) {
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

static uint64_t NextReleaseUsLocked(SchedulerShared *sched, TargetId target) {
	const std::vector<int> &future = FutureQueueFor(sched, target);
	if (future.empty())
		return UINT64_MAX;
	return sched->exec[(size_t)future.front()].release_us;
}

static int PickBestReadyIndex(
	const std::vector<int> &ready, const std::vector<DispatchNode> &nodes) {
	int best_i = 0;
	for (int i = 1; i < (int)ready.size(); ++i) {
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
	sched->ClearQueues();
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

		TargetId tid = nodes[i].hardware_target;
		if (xs.release_us == 0) {
			ReadyQueueFor(sched, tid).push_back((int)i);
		} else {
			InsertFutureSorted(
				&FutureQueueFor(sched, tid), nodes, sched->exec, (int)i);
		}
	}
}

//------------------------------------------------------------------------------
// Worker thread
//------------------------------------------------------------------------------

static void WorkerMain(TargetId target_id, const char *sched_core_csv,
	const char *target_name, iree_hal_device_t *device,
	std::vector<DispatchNode> *nodes,
	const std::vector<std::vector<int>> *dependents,
	const std::vector<CachedModule *> *node_modules, int dispatch_iters,
	iree_allocator_t host_alloc, SharedState *fatal, SchedulerShared *sched,
	TraceWriter *trace) {
	BestEffortPinCurrentThreadToCpuIds(sched_core_csv);

	while (true) {
		int node_idx = -1;
		int graph_iter = 0;
		Clock::time_point iter_t0;

		{
			std::unique_lock<std::mutex> lock(sched->mu);

			while (true) {
				if (sched->shutdown || HasFatal(fatal))
					return;
				if (!sched->active) {
					sched->target_cv[target_id]->wait(lock);
					continue;
				}

				const uint64_t now_us = UsSince(sched->iter_t0, Clock::now());
				PromoteReleasedNodesLocked(sched, *nodes, target_id, now_us);

				std::vector<int> &ready = ReadyQueueFor(sched, target_id);
				if (!ready.empty()) {
					const int best_i = PickBestReadyIndex(ready, *nodes);
					node_idx = ready[(size_t)best_i];
					ready.erase(ready.begin() + best_i);

					graph_iter = sched->current_graph_iter;
					iter_t0 = sched->iter_t0;

					NodeExecState &xs = sched->exec[(size_t)node_idx];
					xs.running = true;
					xs.start_us = UsSince(iter_t0, Clock::now());
					break;
				}

				const uint64_t next_release_us =
					NextReleaseUsLocked(sched, target_id);
				if (next_release_us == UINT64_MAX) {
					sched->target_cv[target_id]->wait(lock);
				} else {
					// Spin-wait for short sleeps to avoid condvar
					// timer overshoot (~2ms on RISC-V kernel).
					const uint64_t now2 = UsSince(sched->iter_t0, Clock::now());
					if (next_release_us > now2 + 5000) {
						// Long wait: condvar, but wake 2ms early to spin.
						const auto wake_tp = sched->iter_t0 +
							std::chrono::microseconds(next_release_us - 2000);
						sched->target_cv[target_id]->wait_until(lock, wake_tp);
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

		CachedModule *cm = (*node_modules)[(size_t)node_idx];
		iree_status_t st;
		if (cm->is_async) {
			// Async dispatch: submit with fences, then wait for completion.
			iree_hal_fence_t *signal_fence = nullptr;
			st = CallModuleAsync(cm, (int32_t)dispatch_iters, device,
				/*wait_fence=*/nullptr, host_alloc, &signal_fence);
			if (iree_status_is_ok(st) && signal_fence) {
				st = iree_hal_fence_wait(signal_fence, iree_infinite_timeout(),
					IREE_ASYNC_WAIT_FLAG_NONE);
				iree_hal_fence_release(signal_fence);
			}
		} else {
			// Sync dispatch: blocks until complete.
			st = CallModuleUnlocked(cm, (int32_t)dispatch_iters, host_alloc);
		}

		const uint64_t end_us = UsSince(iter_t0, Clock::now());

		if (!iree_status_is_ok(st)) {
			SetFatalOnce(fatal, st, "[dispatch] module call failed");
			sched->NotifyAll();
			return;
		}

		uint64_t planned_start_us = 0;
		uint64_t ready_us = 0;
		uint64_t start_us = 0;

		// Track which targets got new work so we only wake those.
		uint32_t targets_with_new_work = 0;
		bool all_done = false;

		{
			std::lock_guard<std::mutex> lock(sched->mu);
			NodeExecState &xs = sched->exec[(size_t)node_idx];
			xs.running = false;
			xs.done = true;
			xs.end_us = end_us;

			planned_start_us = xs.planned_start_us;
			ready_us = xs.ready_us;
			start_us = xs.start_us;

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

					TargetId child_tid =
						(*nodes)[(size_t)child].hardware_target;
					targets_with_new_work |= (1u << child_tid);
					if (cs.release_us <= UsSince(iter_t0, Clock::now())) {
						ReadyQueueFor(sched, child_tid).push_back(child);
					} else {
						InsertFutureSorted(&FutureQueueFor(sched, child_tid),
							*nodes, sched->exec, child);
					}
				}
			}

			sched->completed++;
			if (sched->completed == sched->total_nodes) {
				sched->active = false;
				all_done = true;
			}
		}

		trace->WriteRow(graph_iter, (*nodes)[(size_t)node_idx],
			planned_start_us, ready_us, start_us, end_us, target_name);

		// Only wake targets that received new work, plus self.
		targets_with_new_work |= (1u << target_id);
		for (size_t t = 0; t < sched->target_cv.size(); ++t) {
			if (targets_with_new_work & (1u << t))
				sched->target_cv[t]->notify_one();
		}
		if (all_done)
			sched->main_cv.notify_all();
	}
}

//------------------------------------------------------------------------------
// Build the TargetRegistry from config
//------------------------------------------------------------------------------

static TargetRegistry BuildRegistryFromConfig(
	const scheduler_runner_config_t *cfg) {
	TargetRegistry reg;

	if (cfg->num_targets > 0 && cfg->target_names && cfg->target_cpu_ids &&
		cfg->target_variant_dirs) {
		for (int i = 0; i < cfg->num_targets; ++i) {
			reg.Register(cfg->target_names[i] ? cfg->target_names[i] : "",
				cfg->target_cpu_ids[i] ? cfg->target_cpu_ids[i] : "",
				cfg->target_variant_dirs[i] ? cfg->target_variant_dirs[i] : "");
		}
	} else if (cfg->cpu_p_cpu_ids && cfg->cpu_p_cpu_ids[0]) {
		// Legacy 2-target mode.
		reg.Register("CPU_P", cfg->cpu_p_cpu_ids ? cfg->cpu_p_cpu_ids : "",
			cfg->variant_p_dir ? cfg->variant_p_dir : "");
		reg.Register("CPU_E", cfg->cpu_e_cpu_ids ? cfg->cpu_e_cpu_ids : "",
			cfg->variant_e_dir ? cfg->variant_e_dir : "");
	}

	return reg;
}

//------------------------------------------------------------------------------
// Summary JSON output (sample-specific config section)
//------------------------------------------------------------------------------

static bool WriteSummaryJson(const char *path,
	const scheduler_runner_config_t *cfg, const TargetRegistry &reg,
	const GraphModel &model, const std::vector<int> &topo_order) {
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

	fprintf(f, "    \"targets\": [\n");
	for (int t = 0; t < reg.Size(); ++t) {
		TargetId tid = static_cast<TargetId>(t);
		fprintf(f, "      {\"name\": ");
		JsonWriteEscaped(f, reg.Name(tid));
		fprintf(f, ", \"cpu_ids\": ");
		JsonWriteEscaped(f, reg.CpuIds(tid));
		fprintf(f, ", \"variant_dir\": ");
		JsonWriteEscaped(f, reg.VariantDir(tid));
		fprintf(f, "}%s\n", (t + 1 < reg.Size()) ? "," : "");
	}
	fprintf(f, "    ],\n");

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
	const int warmup_iters = (cfg->warmup_iters > 0) ? cfg->warmup_iters : 0;
	const int dispatch_iters =
		(cfg->dispatch_iters > 0) ? cfg->dispatch_iters : 1;
	const int report_every = (cfg->report_every >= 0) ? cfg->report_every : 0;

	if (strcmp(driver, "local-task") != 0) {
		fprintf(stderr, "This scheduler requires driver=local-task; got '%s'\n",
			driver);
		return 1;
	}

	// Build target registry from config.
	TargetRegistry reg = BuildRegistryFromConfig(cfg);
	if (reg.Size() == 0) {
		fprintf(stderr,
			"No targets configured. Use --target=NAME:CPU_IDS:VARIANT or "
			"legacy --cpu_p_cpu_ids/--cpu_e_cpu_ids flags.\n");
		return 1;
	}

	if (!ValidateCorePartition(reg, cfg->visible_cores))
		return 1;

	const int num_targets = reg.Size();

	fprintf(stdout,
		"Dispatch scheduler (sync benchmark VMFBs):\n"
		"  json          = %s\n"
		"  driver        = %s\n"
		"  graph_iters   = %d\n"
		"  warmup_iters  = %d\n"
		"  dispatch_iters= %d\n"
		"  report_every  = %d\n"
		"  vmfb_root_dir = %s\n"
		"  num_targets   = %d\n",
		cfg->graph_json_path, driver, graph_iters, warmup_iters, dispatch_iters,
		report_every, cfg->vmfb_root_dir ? cfg->vmfb_root_dir : "",
		num_targets);
	for (int t = 0; t < num_targets; ++t) {
		TargetId tid = static_cast<TargetId>(t);
		fprintf(stdout, "  target[%d]     = %s cores={%s} variant=%s\n", t,
			reg.Name(tid), reg.CpuIds(tid), reg.VariantDir(tid));
	}
	fflush(stdout);

	// Parse schedule JSON with registry-aware target lookup.
	GraphModel model;
	if (!ParseDispatchScheduleJson(cfg->graph_json_path, &model, &reg)) {
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

	// Resolve VMFB paths using registry-based variant lookup.
	const std::string json_dir = PathDirname(cfg->graph_json_path);
	for (auto &n : model.nodes) {
		const char *variant_dir = reg.VariantDir(n.hardware_target);
		n.vmfb_path_resolved = ResolveVmfbPathWithVariant(cfg->vmfb_root_dir,
			cfg->target_platform, json_dir, n, variant_dir, cfg->elf_marker);
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
			n.key.c_str(), reg.Name(n.hardware_target), n.start_time_ms,
			n.planned_duration_ms);
	}
	fflush(stdout);

	SharedState shared;
	iree_allocator_t host_alloc = iree_allocator_system();
	iree_runtime_instance_t *instance = nullptr;

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

	// Create one pinned device per target.  Each target gets its own driver
	// with a dedicated executor pinned to its core set.  This provides true
	// core isolation — the key finding from dev blog §16-17.
	std::vector<iree_hal_device_t *> devices(num_targets, nullptr);
	for (int t = 0; t < num_targets; ++t) {
		TargetId tid = static_cast<TargetId>(t);
		iree_status_t st = CreatePinnedLocalTaskDevice(
			host_alloc, reg.CpuIds(tid), &devices[t]);
		if (!iree_status_is_ok(st)) {
			fprintf(stderr, "Failed creating device for target %s\n",
				reg.Name(tid));
			iree_status_fprint(stderr, st);
			iree_status_ignore(st);
			for (int j = 0; j < t; ++j) {
				if (devices[j])
					iree_hal_device_release(devices[j]);
			}
			iree_runtime_instance_release(instance);
			trace.Close();
			return 1;
		}
	}

	for (int t = 0; t < num_targets; ++t) {
		TargetId tid = static_cast<TargetId>(t);
		fprintf(stdout, "[dispatch] %s local-task topology = {%s}\n",
			reg.Name(tid), reg.CpuIds(tid));
	}
	fflush(stdout);

	// Cache one session per (target, vmfb_path).
	std::unordered_map<std::string, std::unique_ptr<CachedModule>> cache;
	cache.reserve(model.nodes.size() * 2);

	auto cache_key_for = [&reg](TargetId target, const std::string &path) {
		return std::string(reg.Name(target)) + "|" + path;
	};

	std::vector<CachedModule *> node_modules(model.nodes.size(), nullptr);

	for (size_t i = 0; i < model.nodes.size(); ++i) {
		DispatchNode &node = model.nodes[i];
		const std::string cache_key =
			cache_key_for(node.hardware_target, node.vmfb_path_resolved);

		auto it = cache.find(cache_key);
		if (it == cache.end()) {
			auto cm = std::make_unique<CachedModule>();
			iree_hal_device_t *target_device = devices[node.hardware_target];

			iree_status_t st = LoadModule(
				instance, target_device, node.vmfb_path_resolved, cm.get());
			if (!iree_status_is_ok(st)) {
				fprintf(stderr, "Failed loading VMFB for node %s\n",
					node.key.c_str());
				iree_status_fprint(stderr, st);
				iree_status_ignore(st);

				trace.Close();
				for (auto &cache_kv : cache)
					CachedModuleRelease(cache_kv.second.get());
				for (int t = 0; t < num_targets; ++t) {
					if (devices[t])
						iree_hal_device_release(devices[t]);
				}
				if (instance)
					iree_runtime_instance_release(instance);
				return 1;
			}

			fprintf(stdout, "[dispatch] Loaded %-35s -> %s [%s]\n",
				node.key.c_str(), node.vmfb_path_resolved.c_str(),
				cm->is_async ? "async" : "sync");
			node_modules[i] = cm.get();
			cache.emplace(cache_key, std::move(cm));
		} else {
			node_modules[i] = it->second.get();
		}
	}

	SchedulerShared sched;
	sched.total_nodes = model.nodes.size();
	sched.InitQueues(num_targets);

	// Pre-warm every unique cached module.
	fprintf(stdout, "[dispatch] Pre-warming %zu unique cached modules...\n",
		cache.size());
	fflush(stdout);

	for (auto &kv : cache) {
		CachedModule *cm = kv.second.get();

		const int32_t warm_iters = (cm->arity >= 1 && cm->first_is_i32)
			? 0
			: static_cast<int32_t>(dispatch_iters);

		iree_status_t st;
		if (cm->is_async) {
			// For async modules, find the device from the cache key and
			// do an async warmup with fence wait.
			// Extract target name from cache key "TARGET_NAME|vmfb_path".
			const std::string &key = kv.first;
			size_t sep = key.find('|');
			std::string tname =
				(sep != std::string::npos) ? key.substr(0, sep) : "";
			TargetId tid = reg.Parse(tname);
			iree_hal_device_t *dev = reg.Valid(tid) ? devices[tid] : nullptr;
			if (!dev) {
				fprintf(stderr, "Warmup: no device for async module %s\n",
					cm->vmfb_path.c_str());
				st = iree_make_status(
					IREE_STATUS_INTERNAL, "no device for async warmup");
			} else {
				iree_hal_fence_t *signal = nullptr;
				st = CallModuleAsync(
					cm, warm_iters, dev, nullptr, host_alloc, &signal);
				if (iree_status_is_ok(st) && signal) {
					st = iree_hal_fence_wait(signal, iree_infinite_timeout(),
						IREE_ASYNC_WAIT_FLAG_NONE);
					iree_hal_fence_release(signal);
				}
			}
		} else {
			st = CallModuleUnlocked(cm, warm_iters, host_alloc);
		}
		if (!iree_status_is_ok(st)) {
			fprintf(
				stderr, "Warmup failed for VMFB: %s\n", cm->vmfb_path.c_str());
			iree_status_fprint(stderr, st);
			iree_status_ignore(st);

			trace.Close();
			for (auto &cache_kv : cache)
				CachedModuleRelease(cache_kv.second.get());
			for (int t = 0; t < num_targets; ++t) {
				if (devices[t])
					iree_hal_device_release(devices[t]);
			}
			if (instance)
				iree_runtime_instance_release(instance);
			return 1;
		}
	}

	fprintf(stdout, "[dispatch] Module warmup complete.\n");
	fflush(stdout);

	// Launch one worker thread per target.
	// Workers run continuously, driven by the main thread's iteration loop.
	// Trace writer is toggled on/off to separate warmup from real capture.
	TraceWriter warmup_sink; // Dummy sink — discards rows during warmup.

	// Compute free cores (not used by any executor) for scheduler workers.
	// Pinning workers to non-executor cores prevents scheduler threads from
	// competing with IREE executor worker threads for CPU time.
	std::unordered_set<int> executor_cores;
	for (int t = 0; t < num_targets; ++t) {
		std::vector<int> ids;
		SplitCpuIds(reg.CpuIds(static_cast<TargetId>(t)), &ids);
		for (int id : ids)
			executor_cores.insert(id);
	}
	int vc = cfg->visible_cores > 0 ? cfg->visible_cores : 8;
	std::vector<int> free_cores;
	for (int c = 0; c < vc; ++c) {
		if (executor_cores.find(c) == executor_cores.end())
			free_cores.push_back(c);
	}

	// Build per-worker core CSV strings.
	std::vector<std::string> sched_core_strs(num_targets);
	for (int t = 0; t < num_targets; ++t) {
		if (!free_cores.empty()) {
			// Assign each worker a distinct free core (round-robin).
			sched_core_strs[t] =
				std::to_string(free_cores[(size_t)t % free_cores.size()]);
		} else {
			// No free cores — fall back to executor cores.
			sched_core_strs[t] = reg.CpuIds(static_cast<TargetId>(t));
		}
	}

	std::vector<std::thread> workers;
	workers.reserve(num_targets);
	for (int t = 0; t < num_targets; ++t) {
		TargetId tid = static_cast<TargetId>(t);
		fprintf(stdout, "[dispatch] %s scheduler worker -> core {%s}\n",
			reg.Name(tid), sched_core_strs[t].c_str());
		workers.emplace_back(WorkerMain, tid, sched_core_strs[t].c_str(),
			reg.Name(tid), devices[t], &model.nodes, &dependents, &node_modules,
			dispatch_iters, host_alloc, &shared, &sched,
			warmup_iters > 0 ? &warmup_sink : &trace);
	}

	// --- Warmup iterations (untraced) ---
	for (int gi = 0; gi < warmup_iters && !HasFatal(&shared); ++gi) {
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
		sched.NotifyAll();

		{
			std::unique_lock<std::mutex> lock(sched.mu);
			sched.main_cv.wait(lock, [&]() {
				return HasFatal(&shared) ||
					sched.completed == sched.total_nodes;
			});
			sched.active = false;
			sched.ClearQueues();
		}
		sched.NotifyAll();
	}

	if (warmup_iters > 0 && !HasFatal(&shared)) {
		fprintf(stdout, "[dispatch] %d warmup graph iterations complete.\n",
			warmup_iters);
		fflush(stdout);

		// Reset per-node stats so only traced iterations contribute.
		for (auto &n : model.nodes)
			n.run_stats = RunningStats{};

		// Swap workers to the real trace writer.  We do this by shutting
		// down the warmup workers and relaunching with the trace writer.
		{
			std::lock_guard<std::mutex> lock(sched.mu);
			sched.shutdown = true;
			sched.active = false;
			sched.ClearQueues();
		}
		sched.NotifyAll();
		for (auto &w : workers)
			w.join();
		workers.clear();

		// Reset scheduler state for the real run.
		sched.shutdown = false;
		sched.InitQueues(num_targets);

		for (int t = 0; t < num_targets; ++t) {
			TargetId tid = static_cast<TargetId>(t);
			workers.emplace_back(WorkerMain, tid, sched_core_strs[t].c_str(),
				reg.Name(tid), devices[t], &model.nodes, &dependents,
				&node_modules, dispatch_iters, host_alloc, &shared, &sched,
				&trace);
		}
	}

	// --- Traced iterations ---
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
		sched.NotifyAll();

		{
			std::unique_lock<std::mutex> lock(sched.mu);
			sched.main_cv.wait(lock, [&]() {
				return HasFatal(&shared) ||
					sched.completed == sched.total_nodes;
			});
			sched.active = false;
			sched.ClearQueues();
		}
		sched.NotifyAll();

		if (report_every > 0 && ((gi + 1) % report_every) == 0 &&
			!HasFatal(&shared)) {
			fprintf(stdout, "[graph_iter %d/%d]\n", gi + 1, graph_iters);
			for (int idx : topo_order) {
				const auto &n = model.nodes[static_cast<size_t>(idx)];
				fprintf(stdout,
					"  %s target=%s plan=%.3fms run_avg=%.3fms p90=%.3fms "
					"max=%.3fms\n",
					n.key.c_str(), reg.Name(n.hardware_target),
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
		sched.ClearQueues();
	}
	sched.NotifyAll();

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
				n.key.c_str(), reg.Name(n.hardware_target),
				n.planned_duration_ms, n.run_stats.AvgMs(), n.run_stats.P50Ms(),
				n.run_stats.P90Ms(), n.run_stats.P99Ms(), n.run_stats.MinMs(),
				n.run_stats.MaxMs());
		}
		fprintf(stdout, "Done.\n");
		fflush(stdout);
	}

	bool ok_write = true;
	ok_write = ok_write &&
		WriteSummaryJson(cfg->out_json_path, cfg, reg, model, topo_order);
	ok_write = ok_write && WriteDotGraph(cfg->out_dot_path, model);
	if (!ok_write) {
		fprintf(stderr, "Warning: failed writing one or more outputs\n");
	}

	trace.Close();

	for (auto &kv : cache)
		CachedModuleRelease(kv.second.get());
	for (int t = 0; t < num_targets; ++t) {
		if (devices[t])
			iree_hal_device_release(devices[t]);
	}
	if (instance)
		iree_runtime_instance_release(instance);

	return HasFatal(&shared) ? 1 : 0;
}
