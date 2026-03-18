// samples/SpacemiTX60/dispatch_scheduler/runtime_scheduler.cc
//
// Dispatch-level scheduler for two-cluster systems (CPU_P + CPU_E).
//
// - Two long-lived worker threads, one per hardware target.
// - Each worker is pinned to its CPU set via pthread_setaffinity_np.
// - Pinned local-task devices with dedicated task executors (one per core set).
// - One cached runtime session per (target, vmfb_path).
// - Release-time scheduling with phase-locked roots and dependency-driven
//   chains.
// - Spin-wait for short delays (<5ms) to avoid condvar timer overshoot on
//   RISC-V.

#include "runtime_scheduler.h"

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

static bool ValidateCorePartition(const dispatch_graph_config_t *cfg) {
	const int visible_cores = cfg->visible_cores > 0 ? cfg->visible_cores : 8;

	std::vector<int> p_ids;
	std::vector<int> e_ids;
	if (!SplitCpuIds(
			cfg->cpu_p_cpu_ids ? cfg->cpu_p_cpu_ids : "0,1,2,3", &p_ids)) {
		fprintf(stderr, "Invalid --cpu_p_cpu_ids\n");
		return false;
	}
	if (!SplitCpuIds(cfg->cpu_e_cpu_ids ? cfg->cpu_e_cpu_ids : "4,5", &e_ids)) {
		fprintf(stderr, "Invalid --cpu_e_cpu_ids\n");
		return false;
	}
	if (p_ids.size() != 4) {
		fprintf(
			stderr, "CPU_P must have exactly 4 cores; got %zu\n", p_ids.size());
		return false;
	}
	if (e_ids.size() != 2) {
		fprintf(
			stderr, "CPU_E must have exactly 2 cores; got %zu\n", e_ids.size());
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
};

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

//------------------------------------------------------------------------------
// Worker thread
//------------------------------------------------------------------------------

static void WorkerMain(HardwareTarget target, const char *cpu_ids_csv,
	std::vector<DispatchNode> *nodes,
	const std::vector<std::vector<int>> *dependents,
	const std::vector<CachedModule *> *node_modules, int dispatch_iters,
	iree_allocator_t host_alloc, SharedState *fatal, SchedulerShared *sched,
	TraceWriter *trace) {
	BestEffortPinCurrentThreadToCpuIds(cpu_ids_csv);

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
					sched->cv.wait(lock);
					continue;
				}

				const uint64_t now_us = UsSince(sched->iter_t0, Clock::now());
				PromoteReleasedNodesLocked(sched, *nodes, target, now_us);

				std::vector<int> &ready = ReadyQueueFor(sched, target);
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
	const dispatch_graph_config_t *cfg, const GraphModel &model,
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

extern "C" int dispatch_graph_run(const dispatch_graph_config_t *cfg) {
	using namespace merlin_bench;

	if (!cfg || !cfg->graph_json_path || !cfg->graph_json_path[0]) {
		fprintf(stderr, "dispatch_graph_run: missing graph_json_path\n");
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
			ResolveVmfbPath(cfg->vmfb_root_dir, "spacemit_x60", json_dir, n);
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

	fprintf(stdout,
		"[dispatch] CPU_P local-task topology = {%s}\n"
		"[dispatch] CPU_E local-task topology = {%s}\n",
		cfg->cpu_p_cpu_ids, cfg->cpu_e_cpu_ids);
	fflush(stdout);

	// Cache one session per (target, vmfb_path).
	std::unordered_map<std::string, std::unique_ptr<CachedModule>> cache;
	cache.reserve(model.nodes.size() * 2);

	auto cache_key_for = [](HardwareTarget target, const std::string &path) {
		return std::string(HardwareTargetName(target)) + "|" + path;
	};

	std::vector<CachedModule *> node_modules(model.nodes.size(), nullptr);

	for (size_t i = 0; i < model.nodes.size(); ++i) {
		DispatchNode &node = model.nodes[i];
		const std::string cache_key =
			cache_key_for(node.hardware_target, node.vmfb_path_resolved);

		auto it = cache.find(cache_key);
		if (it == cache.end()) {
			auto cm = std::make_unique<CachedModule>();
			iree_hal_device_t *target_device =
				(node.hardware_target == HardwareTarget::kCpuP) ? device_p
																: device_e;

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

	std::thread worker_p(WorkerMain, HardwareTarget::kCpuP,
		cfg->cpu_p_cpu_ids ? cfg->cpu_p_cpu_ids : "0,1,2,3", &model.nodes,
		&dependents, &node_modules, dispatch_iters, host_alloc, &shared, &sched,
		&trace);

	std::thread worker_e(WorkerMain, HardwareTarget::kCpuE,
		cfg->cpu_e_cpu_ids ? cfg->cpu_e_cpu_ids : "4,5", &model.nodes,
		&dependents, &node_modules, dispatch_iters, host_alloc, &shared, &sched,
		&trace);

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

	worker_p.join();
	worker_e.join();

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
	if (device_e)
		iree_hal_device_release(device_e);
	if (device_p)
		iree_hal_device_release(device_p);
	if (instance)
		iree_runtime_instance_release(instance);

	return HasFatal(&shared) ? 1 : 0;
}
