#include "runtime_dispatch_graph.h"

#include <inttypes.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"

#include "iree_bench/fatal_state.h"
#include "iree_bench/iree_module_utils.h"
#include "iree_bench/json_parser.h"
#include "iree_bench/path_utils.h"
#include "iree_bench/stats.h"

// --- Optional local-task topology pinning (best-effort) ---
#if defined(__has_include)
#if __has_include("iree/task/api.h") &&                                        \
	__has_include("iree/task/topology.h") &&                                   \
		__has_include("iree/hal/drivers/local_task/driver.h") &&               \
			__has_include("iree/base/internal/threading.h")
#define DISPATCH_GRAPH_HAS_LOCAL_TASK_PINNING 1
#include "iree/base/internal/threading.h"
#include "iree/hal/drivers/local_task/driver.h"
#include "iree/task/api.h"
#include "iree/task/topology.h"
#else
#define DISPATCH_GRAPH_HAS_LOCAL_TASK_PINNING 0
#endif
#else
#define DISPATCH_GRAPH_HAS_LOCAL_TASK_PINNING 0
#endif

namespace {

using namespace merlin_bench;
using Clock = std::chrono::steady_clock;

// Dispatch node parsed from JSON
struct DispatchNode {
	std::string key; // "dispatch_16_2"
	int id = -1; // 16
	int ordinal = 0; // 1/2 (optional)
	int total = 0; // 2 (optional)
	std::string vmfb_path; // resolved absolute-ish path
	std::vector<std::string> deps; // keys

	RunningStats stats;
};

static bool ParseDispatchesObject(JsonParser *jp, const std::string &json_dir,
	std::vector<DispatchNode> *out_nodes) {
	out_nodes->clear();
	if (!jp->Consume('{'))
		return false;
	jp->SkipWs();
	if (jp->Consume('}'))
		return true;

	while (true) {
		std::string dispatch_key;
		if (!jp->ParseString(&dispatch_key))
			return false;
		if (!jp->Consume(':'))
			return false;
		if (!jp->Consume('{'))
			return false;

		DispatchNode node;
		node.key = std::move(dispatch_key);

		jp->SkipWs();
		if (!jp->Consume('}')) {
			while (true) {
				std::string field;
				if (!jp->ParseString(&field))
					return false;
				if (!jp->Consume(':'))
					return false;

				if (field == "id") {
					int v = 0;
					if (!jp->ParseInt(&v))
						return false;
					node.id = v;
				} else if (field == "ordinal") {
					int v = 0;
					if (!jp->ParseInt(&v))
						return false;
					node.ordinal = v;
				} else if (field == "total") {
					int v = 0;
					if (!jp->ParseInt(&v))
						return false;
					node.total = v;
				} else if (field == "vmfb_path") {
					std::string rel;
					if (!jp->ParseString(&rel))
						return false;
					node.vmfb_path = PathJoin2(json_dir, rel);
				} else if (field == "dependencies") {
					if (!ParseDependenciesArray(jp, &node.deps))
						return false;
				} else {
					if (!jp->SkipValue())
						return false;
				}

				jp->SkipWs();
				if (jp->Consume('}'))
					break;
				if (!jp->Consume(','))
					return false;
			}
		}

		out_nodes->push_back(std::move(node));

		jp->SkipWs();
		if (jp->Consume('}'))
			break;
		if (!jp->Consume(','))
			return false;
	}
	return true;
}

static bool ParseDispatchGraphJson(
	const std::string &json_path, std::vector<DispatchNode> *out_nodes) {
	std::ifstream f(json_path, std::ios::binary);
	if (!f)
		return false;
	std::string buf(
		(std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
	if (buf.empty())
		return false;

	const std::string json_dir = PathDirname(json_path);

	JsonParser jp;
	jp.p = buf.data();
	jp.end = buf.data() + buf.size();

	// Root object; scan for "dispatches": { ... }.
	bool ok = false;
	if (!jp.Consume('{'))
		return false;
	jp.SkipWs();
	if (jp.Consume('}'))
		return false;

	while (true) {
		std::string key;
		if (!jp.ParseString(&key))
			break;
		if (!jp.Consume(':'))
			break;

		if (key == "dispatches") {
			ok = ParseDispatchesObject(&jp, json_dir, out_nodes);
			break;
		} else {
			if (!jp.SkipValue())
				break;
		}

		jp.SkipWs();
		if (jp.Consume('}'))
			break;
		if (!jp.Consume(','))
			break;
	}

	return ok && !out_nodes->empty();
}

// ------------------------------
// Topo sort
// Deterministic tie-break: smaller (id, ordinal, key) first.
// ------------------------------
static bool TopoSort(const std::vector<DispatchNode> &nodes,
	std::vector<int> *out_order,
	std::vector<std::vector<int>> *out_dependents) {
	out_order->clear();
	out_dependents->assign(nodes.size(), {});

	std::unordered_map<std::string, int> index_of;
	index_of.reserve(nodes.size() * 2);
	for (int i = 0; i < (int)nodes.size(); ++i) {
		index_of[nodes[i].key] = i;
	}

	std::vector<int> indeg(nodes.size(), 0);
	for (int i = 0; i < (int)nodes.size(); ++i) {
		indeg[i] = (int)nodes[i].deps.size();
		for (const auto &dep_key : nodes[i].deps) {
			auto it = index_of.find(dep_key);
			if (it == index_of.end()) {
				fprintf(stderr, "Missing dependency '%s' for node '%s'\n",
					dep_key.c_str(), nodes[i].key.c_str());
				return false;
			}
			(*out_dependents)[it->second].push_back(i);
		}
	}

	auto better = [&](int a, int b) {
		const auto &A = nodes[a];
		const auto &B = nodes[b];
		if (A.id != B.id)
			return A.id < B.id;
		if (A.ordinal != B.ordinal)
			return A.ordinal < B.ordinal;
		return A.key < B.key;
	};

	std::vector<int> ready;
	ready.reserve(nodes.size());
	for (int i = 0; i < (int)nodes.size(); ++i) {
		if (indeg[i] == 0)
			ready.push_back(i);
	}

	while (!ready.empty()) {
		// pick best
		int best_i = 0;
		for (int i = 1; i < (int)ready.size(); ++i) {
			if (better(ready[i], ready[best_i]))
				best_i = i;
		}
		int pick = ready[best_i];
		ready.erase(ready.begin() + best_i);

		out_order->push_back(pick);

		for (int child : (*out_dependents)[pick]) {
			indeg[child]--;
			if (indeg[child] == 0)
				ready.push_back(child);
		}
	}

	if (out_order->size() != nodes.size()) {
		fprintf(stderr, "Cycle detected in dispatch dependency graph\n");
		return false;
	}

	return true;
}

// ------------------------------
// Optional local-task pinned device
// ------------------------------
static iree_hal_device_t *CreatePinnedLocalTaskDeviceIfAvailable(
	iree_allocator_t host_allocator, uint64_t core_mask,
	iree_status_t *out_status) {
	*out_status = iree_ok_status();

#if DISPATCH_GRAPH_HAS_LOCAL_TASK_PINNING
	if (core_mask == 0)
		return nullptr;

	iree_task_topology_t topology;
	iree_task_topology_initialize(&topology);

	for (int core_id = 0; core_id < 64; ++core_id) {
		if (((core_mask >> core_id) & 1ull) == 0)
			continue;

		iree_task_topology_group_t group;
		iree_task_topology_group_initialize(topology.group_count, &group);

		group.processor_index = core_id;
		memset(&group.ideal_thread_affinity, 0,
			sizeof(group.ideal_thread_affinity));
		iree_thread_affinity_set_bit(&group.ideal_thread_affinity, core_id);

		iree_status_t st = iree_task_topology_push_group(&topology, &group);
		if (!iree_status_is_ok(st)) {
			iree_task_topology_deinitialize(&topology);
			*out_status = st;
			return nullptr;
		}
	}

	iree_task_executor_options_t exec_opts =
		iree_task_executor_options_default();
	exec_opts.worker_local_memory_size = 64 * 1024;

	iree_task_executor_t *executor = nullptr;
	iree_status_t st = iree_task_executor_create(
		exec_opts, &topology, host_allocator, &executor);
	iree_task_topology_deinitialize(&topology);
	if (!iree_status_is_ok(st)) {
		*out_status = st;
		return nullptr;
	}

	iree_hal_task_device_params_t params =
		iree_hal_task_device_params_default();

	iree_hal_device_t *device = nullptr;
	st = iree_hal_task_device_create(
		iree_make_cstring_view("pinned_local_task"), &params, executor,
		/*queue_count=*/1,
		/*queues=*/nullptr, host_allocator, &device);

	iree_task_executor_release(executor);

	*out_status = st;
	if (!iree_status_is_ok(st))
		return nullptr;
	return device;
#else
	(void)host_allocator;
	(void)core_mask;
	return nullptr;
#endif
}

// ------------------------------
// Module/session cache
// ------------------------------
struct CachedModule {
	std::string vmfb_path;
	iree_runtime_session_t *session = nullptr;
	iree_vm_function_t entry_fn = {0};
	int arity = 0;
	bool first_is_i32 = false;

	std::mutex mu; // serialize calls on this session
};

static void CachedModuleRelease(CachedModule *m) {
	if (!m)
		return;
	iree_runtime_session_release(m->session);
	m->session = nullptr;
}

static iree_status_t LoadModuleCached(iree_runtime_instance_t *instance,
	iree_hal_device_t *device, iree_allocator_t host_alloc,
	const std::string &vmfb_path, CachedModule *out) {
	out->vmfb_path = vmfb_path;

	iree_runtime_session_options_t session_opts;
	iree_runtime_session_options_initialize(&session_opts);

	IREE_RETURN_IF_ERROR(
		iree_runtime_session_create_with_device(instance, &session_opts, device,
			iree_runtime_instance_host_allocator(instance), &out->session));

	IREE_RETURN_IF_ERROR(iree_runtime_session_append_bytecode_module_from_file(
		out->session, vmfb_path.c_str()));

	iree_vm_context_t *ctx = iree_runtime_session_context(out->session);
	const iree_host_size_t module_count = iree_vm_context_module_count(ctx);
	if (module_count == 0) {
		return iree_make_status(
			IREE_STATUS_FAILED_PRECONDITION, "session context had 0 modules");
	}
	iree_vm_module_t *module = iree_vm_context_module_at(ctx, module_count - 1);

	IREE_RETURN_IF_ERROR(PickEntryFunction(
		module, &out->entry_fn, &out->arity, &out->first_is_i32));

	if (!(out->arity == 0 || (out->arity == 1 && out->first_is_i32))) {
		return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
			"entry function arity=%d (first_i32=%d) not supported; "
			"supported: 0 args or 1 i32 arg",
			out->arity, out->first_is_i32 ? 1 : 0);
	}

	(void)host_alloc;
	return iree_ok_status();
}

static iree_status_t CallCachedModule(
	CachedModule *m, int32_t dispatch_iters, iree_allocator_t host_alloc) {
	std::lock_guard<std::mutex> lock(m->mu);

	iree_status_t st = iree_ok_status();
	iree_vm_list_t *inputs = nullptr;

	// Build input list matching signature.
	st = iree_vm_list_create(iree_vm_make_undefined_type_def(),
		/*initial_capacity=*/(iree_host_size_t)m->arity, host_alloc, &inputs);
	if (!iree_status_is_ok(st))
		goto cleanup;

	if (m->arity == 1) {
		iree_vm_value_t v = iree_vm_value_make_i32(dispatch_iters);
		st = iree_vm_list_push_value(inputs, &v);
		if (!iree_status_is_ok(st))
			goto cleanup;
	}

	st = iree_runtime_session_call(m->session, &m->entry_fn, inputs,
		/*output_list=*/nullptr);

cleanup:
	iree_vm_list_release(inputs);
	return st;
}

// ------------------------------
// Parallel executor for one graph iteration
// ------------------------------
struct WorkQueue {
	std::mutex mu;
	std::condition_variable cv;
	std::deque<int> ready;
	int remaining = 0;
	bool stop = false;
};

} // namespace

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

	// Parse graph
	std::vector<DispatchNode> nodes;
	if (!ParseDispatchGraphJson(cfg->graph_json_path, &nodes)) {
		fprintf(stderr, "Failed to parse dispatch graph JSON: %s\n",
			cfg->graph_json_path);
		return 1;
	}

	std::vector<int> order;
	std::vector<std::vector<int>> dependents;
	if (!TopoSort(nodes, &order, &dependents)) {
		fprintf(stderr, "Failed to topo-sort dispatch graph\n");
		return 1;
	}

	fprintf(
		stdout, "Dispatch execution topo order (%zu nodes):\n", order.size());
	for (size_t i = 0; i < order.size(); ++i) {
		const auto &n = nodes[order[i]];
		fprintf(stdout, "  %zu) %s (id=%d ord=%d/%d)\n", i + 1, n.key.c_str(),
			n.id, n.ordinal, n.total);
	}
	fflush(stdout);

	SharedState shared;

	iree_allocator_t host_alloc = iree_allocator_system();
	iree_status_t st = iree_ok_status();

	iree_runtime_instance_t *instance = nullptr;
	iree_hal_device_t *device = nullptr;

	// Create instance
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

	// Create device (pinned local-task if requested)
	if (core_mask != 0) {
		iree_status_t pin_st = iree_ok_status();
		iree_hal_device_t *pinned = CreatePinnedLocalTaskDeviceIfAvailable(
			host_alloc, core_mask, &pin_st);
		if (pinned) {
			device = pinned;
			fprintf(stdout,
				"[dispatch] Using pinned local-task device "
				"(core_mask=0x%016" PRIx64 ")\n",
				core_mask);
			fflush(stdout);
		} else {
			if (!iree_status_is_ok(pin_st)) {
				fprintf(stderr,
					"[dispatch] Pinning requested but failed; falling back.\n");
				iree_status_fprint(stderr, pin_st);
				iree_status_ignore(pin_st);
			} else {
#if !DISPATCH_GRAPH_HAS_LOCAL_TASK_PINNING
				fprintf(stderr,
					"[dispatch] Pinning requested but task headers "
					"unavailable; falling back.\n");
#endif
			}
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

	// Cache modules by vmfb_path (loads once, reused many times)
	std::unordered_map<std::string, std::unique_ptr<CachedModule>> cache;
	cache.reserve(nodes.size() * 2);

	auto get_cached = [&](const std::string &vmfb_path) -> CachedModule * {
		auto it = cache.find(vmfb_path);
		if (it != cache.end())
			return it->second.get();

		auto cm = std::make_unique<CachedModule>();
		iree_status_t load_st =
			LoadModuleCached(instance, device, host_alloc, vmfb_path, cm.get());
		if (!iree_status_is_ok(load_st)) {
			SetFatalOnce(&shared, load_st, "[dispatch] load module failed");
			return nullptr;
		}
		auto *ptr = cm.get();
		cache.emplace(vmfb_path, std::move(cm));
		return ptr;
	};

	// Pre-resolve cached module pointer per node
	std::vector<CachedModule *> node_module(nodes.size(), nullptr);
	for (size_t i = 0; i < nodes.size(); ++i) {
		node_module[i] = get_cached(nodes[i].vmfb_path);
		if (HasFatal(&shared) || !node_module[i])
			break;
	}
	if (HasFatal(&shared)) {
		iree_hal_device_release(device);
		iree_runtime_instance_release(instance);
		return 1;
	}

	// Sequential topo-order runner (baseline)
	auto run_sequential_iter = [&](int graph_iter) {
		(void)graph_iter;
		for (int idx : order) {
			if (HasFatal(&shared))
				return;

			const auto t0 = Clock::now();
			iree_status_t call_st = CallCachedModule(
				node_module[idx], (int32_t)dispatch_iters, host_alloc);
			const auto t1 = Clock::now();

			if (!iree_status_is_ok(call_st)) {
				SetFatalOnce(&shared, call_st, "[dispatch] call failed");
				return;
			}

			const uint64_t us =
				(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
					t1 - t0)
					.count();
			nodes[idx].stats.Add(us);
		}
	};

	// Parallel runner using dependency counts (per iteration)
	auto run_parallel_iter = [&](int graph_iter) {
		(void)graph_iter;

		// Recompute indegrees for this iteration.
		std::vector<int> indeg(nodes.size(), 0);
		for (size_t i = 0; i < nodes.size(); ++i)
			indeg[i] = (int)nodes[i].deps.size();

		WorkQueue wq;
		wq.remaining = (int)nodes.size();

		for (int i = 0; i < (int)nodes.size(); ++i) {
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

				// Execute node
				const auto t0 = Clock::now();
				iree_status_t call_st = CallCachedModule(
					node_module[node_idx], (int32_t)dispatch_iters, host_alloc);
				const auto t1 = Clock::now();

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
				nodes[node_idx].stats.Add(us);

				// Release dependents
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

	// Run loop
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
				const auto &n = nodes[idx];
				fprintf(stdout,
					"  %s: count=%" PRIu64 " avg=%.3fms p50=%.3fms p90=%.3fms "
					"p99=%.3fms min=%.3fms max=%.3fms\n",
					n.key.c_str(), n.stats.count, n.stats.AvgMs(),
					n.stats.P50Ms(), n.stats.P90Ms(), n.stats.P99Ms(),
					n.stats.MinMs(), n.stats.MaxMs());
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

	// Final report
	if (!HasFatal(&shared)) {
		fprintf(stdout,
			"Run complete:\n"
			"  total_wall_ms=%.3f\n"
			"  graph_iters_per_s=%.3f\n",
			total_s * 1000.0, graphs_per_s);
		for (int idx : order) {
			const auto &n = nodes[idx];
			fprintf(stdout,
				"  %s: count=%" PRIu64 " avg=%.3fms p50=%.3fms p90=%.3fms "
				"p99=%.3fms min=%.3fms max=%.3fms\n",
				n.key.c_str(), n.stats.count, n.stats.AvgMs(), n.stats.P50Ms(),
				n.stats.P90Ms(), n.stats.P99Ms(), n.stats.MinMs(),
				n.stats.MaxMs());
		}
		fprintf(stdout, "Done.\n");
		fflush(stdout);
	}

	// Cleanup
	for (auto &kv : cache) {
		CachedModuleRelease(kv.second.get());
	}
	iree_hal_device_release(device);
	iree_runtime_instance_release(instance);

	return HasFatal(&shared) ? 1 : 0;
}
