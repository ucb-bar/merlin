#include "xpu-rt/baseline_runner.h"

#include <inttypes.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"

#include "core/path_utils.h"
#include "core/stats.h"
#include "dispatch/dispatch_graph.h"
#include "dispatch/dispatch_types.h"
#include "dispatch/vmfb_resolve.h"
#include "runtime/fatal_state.h"
#include "runtime/module_cache.h"
#include "runtime/pinned_device.h"

namespace {

using namespace merlin_bench;
using Clock = std::chrono::steady_clock;

// Convert a core_mask bitmask to a comma-separated CPU IDs string.
static std::string MaskToCpuIds(uint64_t mask) {
	std::string csv;
	for (int i = 0; i < 64; ++i) {
		if ((mask >> i) & 1ull) {
			if (!csv.empty())
				csv += ",";
			csv += std::to_string(i);
		}
	}
	return csv;
}

// Parallel executor for one graph iteration.
struct WorkQueue {
	std::mutex mu;
	std::condition_variable cv;
	std::deque<int> ready;
	int remaining = 0;
	bool stop = false;
};

} // namespace

extern "C" int baseline_runner_run(const baseline_runner_config_t *cfg) {
	IREE_TRACE_ZONE_BEGIN_NAMED(z_run, "baseline_runner_run");
	using namespace merlin_bench;

	if (!cfg || !cfg->graph_json_path || !cfg->graph_json_path[0]) {
		fprintf(stderr, "baseline_runner_run: missing graph_json_path\n");
		return 1;
	}

	const char *driver = (cfg->driver_name && cfg->driver_name[0])
		? cfg->driver_name
		: "local-task";
	const int graph_iters = (cfg->graph_iters > 0) ? cfg->graph_iters : 1;
	const int dispatch_iters =
		(cfg->dispatch_iters > 0) ? cfg->dispatch_iters : 1;
	const int report_every = (cfg->report_every >= 0) ? cfg->report_every : 0;
	const uint64_t core_mask = cfg->core_mask;
	const int parallelism = (cfg->parallelism > 0) ? cfg->parallelism : 1;

	fprintf(stdout,
		"Dispatch-graph runner:\n"
		"  json          = %s\n"
		"  driver        = %s\n"
		"  graph_iters   = %d\n"
		"  dispatch_iters= %d\n"
		"  report_every  = %d\n"
		"  core_mask     = 0x%016" PRIx64 "\n"
		"  parallelism   = %d\n",
		cfg->graph_json_path, driver, graph_iters, dispatch_iters, report_every,
		core_mask, parallelism);
	fflush(stdout);

	// Parse graph (using unified parser).
	GraphModel model;
	if (!ParseDispatchScheduleJson(cfg->graph_json_path, &model)) {
		fprintf(stderr, "Failed to parse dispatch graph JSON: %s\n",
			cfg->graph_json_path);
		return 1;
	}

	// Resolve VMFB paths relative to JSON directory.
	const std::string json_dir = PathDirname(cfg->graph_json_path);
	ResolveSimpleVmfbPaths(json_dir, &model.nodes);

	// Topo sort uses deps (all_predecessors is empty for simple graphs).
	std::vector<int> order;
	std::vector<std::vector<int>> dependents;
	if (!TopoSort(model.nodes, &order, &dependents)) {
		fprintf(stderr, "Failed to topo-sort dispatch graph\n");
		return 1;
	}

	fprintf(
		stdout, "Dispatch execution topo order (%zu nodes):\n", order.size());
	for (size_t i = 0; i < order.size(); ++i) {
		const auto &n = model.nodes[order[i]];
		fprintf(stdout, "  %zu) %s (id=%d ord=%d/%d)\n", i + 1, n.key.c_str(),
			n.id, n.ordinal, n.total);
	}
	fflush(stdout);

	SharedState shared;

	iree_allocator_t host_alloc = iree_allocator_system();
	iree_status_t st = iree_ok_status();

	iree_runtime_instance_t *instance = nullptr;
	iree_hal_device_t *device = nullptr;

	// Create instance.
	{
		iree_runtime_instance_options_t opts;
		iree_runtime_instance_options_initialize(&opts);
		iree_runtime_instance_options_use_all_available_drivers(&opts);
		st = iree_runtime_instance_create(&opts, host_alloc, &instance);
		if (!iree_status_is_ok(st)) {
			iree_status_fprint(stderr, st);
			iree_status_ignore(st);
			return 1;
		}
	}

	// Create device (pinned local-task if requested).
	if (core_mask != 0) {
		std::string cpu_ids = MaskToCpuIds(core_mask);
		iree_status_t pin_st =
			CreatePinnedLocalTaskDevice(host_alloc, cpu_ids.c_str(), &device);
		if (iree_status_is_ok(pin_st) && device) {
			fprintf(stdout,
				"[dispatch] Using pinned local-task device "
				"(core_mask=0x%016" PRIx64 ", cpus=%s)\n",
				core_mask, cpu_ids.c_str());
			fflush(stdout);
		} else {
			if (!iree_status_is_ok(pin_st)) {
				fprintf(stderr,
					"[dispatch] Pinning requested but failed; falling back.\n");
				iree_status_fprint(stderr, pin_st);
				iree_status_ignore(pin_st);
			}
#if !MERLIN_HAS_PINNED_DEVICE
			fprintf(stderr,
				"[dispatch] Pinning requested but task headers "
				"unavailable; falling back.\n");
#endif
			device = nullptr;
		}
	}

	if (!device) {
		st = iree_runtime_instance_try_create_default_device(
			instance, iree_make_cstring_view(driver), &device);
		if (!iree_status_is_ok(st)) {
			iree_status_fprint(stderr, st);
			iree_status_ignore(st);
			iree_runtime_instance_release(instance);
			return 1;
		}
	}

	// Cache modules by vmfb_path (loads once, reused many times).
	std::unordered_map<std::string, std::unique_ptr<CachedModule>> cache;
	cache.reserve(model.nodes.size() * 2);

	auto get_cached = [&](const std::string &vmfb_path) -> CachedModule * {
		auto it = cache.find(vmfb_path);
		if (it != cache.end())
			return it->second.get();

		auto cm = std::make_unique<CachedModule>();
		iree_status_t load_st =
			LoadModule(instance, device, vmfb_path, cm.get());
		if (!iree_status_is_ok(load_st)) {
			SetFatalOnce(&shared, load_st, "[dispatch] load module failed");
			return nullptr;
		}
		auto *ptr = cm.get();
		cache.emplace(vmfb_path, std::move(cm));
		return ptr;
	};

	// Pre-resolve cached module pointer per node.
	std::vector<CachedModule *> node_module(model.nodes.size(), nullptr);
	for (size_t i = 0; i < model.nodes.size(); ++i) {
		node_module[i] = get_cached(model.nodes[i].vmfb_path_resolved);
		if (HasFatal(&shared) || !node_module[i])
			break;
	}
	if (HasFatal(&shared)) {
		iree_hal_device_release(device);
		iree_runtime_instance_release(instance);
		return 1;
	}

	// Sequential topo-order runner (baseline).
	auto run_sequential_iter = [&](int graph_iter) {
		(void)graph_iter;
		IREE_TRACE_ZONE_BEGIN_NAMED(z_seq, "graph_iter_sequential");
		for (int idx : order) {
			if (HasFatal(&shared))
				break;

			const auto &key [[maybe_unused]] = model.nodes[idx].key;
			IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(
				z_call, key.c_str(), key.size());

			const auto t0 = Clock::now();
			iree_status_t call_st = CallModule(
				node_module[idx], (int32_t)dispatch_iters, host_alloc);
			const auto t1 = Clock::now();

			IREE_TRACE_ZONE_END(z_call);

			if (!iree_status_is_ok(call_st)) {
				SetFatalOnce(&shared, call_st, "[dispatch] call failed");
				break;
			}

			const uint64_t us =
				(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
					t1 - t0)
					.count();
			model.nodes[idx].run_stats.Add(us);
		}
		IREE_TRACE_ZONE_END(z_seq);
	};

	// Parallel runner using dependency counts (per iteration).
	auto run_parallel_iter = [&](int graph_iter) {
		(void)graph_iter;

		// Use deps for parallel indegree (all_predecessors not populated).
		std::vector<int> indeg(model.nodes.size(), 0);
		for (size_t i = 0; i < model.nodes.size(); ++i)
			indeg[i] = (int)model.nodes[i].deps.size();

		WorkQueue wq;
		wq.remaining = (int)model.nodes.size();

		for (int i = 0; i < (int)model.nodes.size(); ++i) {
			if (indeg[i] == 0)
				wq.ready.push_back(i);
		}

		auto worker = [&]() {
			while (true) {
				int node_idx = -1;
				{
					std::unique_lock<std::mutex> lock(wq.mu);
					wq.cv.wait(lock, [&]() {
						return wq.stop || HasFatal(&shared) ||
							!wq.ready.empty() || wq.remaining == 0;
					});

					if (wq.stop || HasFatal(&shared) || wq.remaining == 0)
						return;

					node_idx = wq.ready.front();
					wq.ready.pop_front();
				}

				// Execute node.
				const auto &key [[maybe_unused]] = model.nodes[node_idx].key;
				IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(
					z_call, key.c_str(), key.size());
				const auto t0 = Clock::now();
				iree_status_t call_st = CallModule(
					node_module[node_idx], (int32_t)dispatch_iters, host_alloc);
				const auto t1 = Clock::now();
				IREE_TRACE_ZONE_END(z_call);

				if (!iree_status_is_ok(call_st)) {
					SetFatalOnce(&shared, call_st, "[dispatch] call failed");
					std::lock_guard<std::mutex> lk(wq.mu);
					wq.stop = true;
					wq.cv.notify_all();
					return;
				}

				const uint64_t us =
					(uint64_t)
						std::chrono::duration_cast<std::chrono::microseconds>(
							t1 - t0)
							.count();
				model.nodes[node_idx].run_stats.Add(us);

				// Release dependents.
				{
					std::lock_guard<std::mutex> lock(wq.mu);
					for (int child : dependents[node_idx]) {
						indeg[child]--;
						if (indeg[child] == 0)
							wq.ready.push_back(child);
					}
					wq.remaining--;
				}
				wq.cv.notify_all();
			}
		};

		std::vector<std::thread> threads;
		threads.reserve((size_t)parallelism);
		for (int i = 0; i < parallelism; ++i)
			threads.emplace_back(worker);

		{
			std::lock_guard<std::mutex> lk(wq.mu);
			wq.cv.notify_all();
		}

		for (auto &t : threads)
			t.join();
	};

	// Run loop.
	const auto t_run0 = Clock::now();

	for (int gi = 0; gi < graph_iters && !HasFatal(&shared); ++gi) {
		if (parallelism <= 1) {
			run_sequential_iter(gi);
		} else {
			run_parallel_iter(gi);
		}

		if (report_every > 0 && ((gi + 1) % report_every) == 0 &&
			!HasFatal(&shared)) {
			fprintf(stdout, "[graph_iter %d/%d]\n", gi + 1, graph_iters);
			for (int idx : order) {
				const auto &n = model.nodes[idx];
				fprintf(stdout,
					"  %s: count=%" PRIu64 " avg=%.3fms p50=%.3fms p90=%.3fms "
					"p99=%.3fms min=%.3fms max=%.3fms\n",
					n.key.c_str(), n.run_stats.count, n.run_stats.AvgMs(),
					n.run_stats.P50Ms(), n.run_stats.P90Ms(),
					n.run_stats.P99Ms(), n.run_stats.MinMs(),
					n.run_stats.MaxMs());
			}
			fflush(stdout);
		}
	}

	const auto t_run1 = Clock::now();
	const double total_s =
		std::chrono::duration_cast<std::chrono::duration<double>>(
			t_run1 - t_run0)
			.count();
	const double graphs_per_s =
		(total_s > 0.0) ? ((double)graph_iters / total_s) : 0.0;

	// Final report.
	if (!HasFatal(&shared)) {
		fprintf(stdout,
			"Run complete:\n"
			"  total_wall_ms=%.3f\n"
			"  graph_iters_per_s=%.3f\n",
			total_s * 1000.0, graphs_per_s);
		for (int idx : order) {
			const auto &n = model.nodes[idx];
			fprintf(stdout,
				"  %s: count=%" PRIu64 " avg=%.3fms p50=%.3fms p90=%.3fms "
				"p99=%.3fms min=%.3fms max=%.3fms\n",
				n.key.c_str(), n.run_stats.count, n.run_stats.AvgMs(),
				n.run_stats.P50Ms(), n.run_stats.P90Ms(), n.run_stats.P99Ms(),
				n.run_stats.MinMs(), n.run_stats.MaxMs());
		}
		fprintf(stdout, "Done.\n");
		fflush(stdout);
	}

	// Cleanup.
	for (auto &kv : cache) {
		CachedModuleRelease(kv.second.get());
	}
	iree_hal_device_release(device);
	iree_runtime_instance_release(instance);

	IREE_TRACE_ZONE_END(z_run);
	return HasFatal(&shared) ? 1 : 0;
}
