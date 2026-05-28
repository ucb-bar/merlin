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
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>

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

// XPU-RT telemetry sink: per-dispatch JSON-Lines emission for hardware-in-
// the-loop feedback. The host-side daemon (xpu-rt/streaming_feedback.py)
// tails this stream, windows epochs, and posts incremental hints back to
// the targetgen_mcp ingest_xpurt_feedback tool.
//
// Lifetime: opened in scheduler_runner_run() before the worker threads
// start, closed at the end. When neither path nor fd is configured, all
// emit calls fast-path to a no-op (zero overhead vs today's behavior —
// preserves the additive-only invariant).
class TelemetrySink {
  public:
	~TelemetrySink() {
		Close();
	}

	// Open the sink. Returns 0 on success, -1 on error. When |path| is
	// NULL/empty and |fd| is <= 0, this is a no-op and the sink stays
	// inactive. fd <= 0 is treated as "unused" (callers that memset()
	// the config struct to zero must not see fd=0 hijack stdout/stdin —
	// telemetry to stdin is never intended).
	int Open(const char *path, int caller_fd) {
		Close();
		if (caller_fd > 0) {
			fd_ = caller_fd;
			owns_fd_ = false;
			active_ = true;
			return 0;
		}
		if (!path || !path[0]) {
			return 0; // disabled, not an error
		}
		const int flags = O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC;
		const mode_t mode = 0644;
		const int opened = open(path, flags, mode);
		if (opened < 0) {
			fprintf(stderr,
				"[telemetry] WARNING: open(%s) failed (errno=%d); "
				"telemetry disabled\n",
				path, errno);
			return -1;
		}
		fd_ = opened;
		owns_fd_ = true;
		active_ = true;
		return 0;
	}

	void Close() {
		if (active_ && owns_fd_ && fd_ >= 0) {
			close(fd_);
		}
		fd_ = -1;
		owns_fd_ = false;
		active_ = false;
	}

	bool active() const {
		return active_;
	}

	// Emit a single JSON-Lines record for one dispatch end. Lock-free
	// per-record format + a single ::write() keeps contention low across
	// the two worker threads (kernel-level append is atomic for writes
	// up to PIPE_BUF, which JSON lines easily fit under).
	void EmitDispatchEnd(int graph_iter, const DispatchNode &n,
		uint64_t planned_start_us, uint64_t start_us, uint64_t end_us,
		bool deadline_violated, bool skip_fired) {
		if (!active_)
			return;
		// JSON-escape the dispatch_id minimally (covers " and \).
		char id_buf[256];
		EscapeJsonString(n.key.c_str(), id_buf, sizeof(id_buf));
		char line[512];
		const uint64_t planned_dur_us =
			n.planned_duration_ms > 0 ? MsToUs(n.planned_duration_ms) : 0;
		const uint64_t run_us = end_us >= start_us ? (end_us - start_us) : 0;
		const int len = snprintf(line, sizeof(line),
			"{\"epoch\":%d,"
			"\"dispatch_id\":\"%s\","
			"\"target\":\"%s\","
			"\"planned_start_us\":%" PRIu64 ","
			"\"start_us\":%" PRIu64 ","
			"\"end_us\":%" PRIu64 ","
			"\"run_us\":%" PRIu64 ","
			"\"planned_duration_us\":%" PRIu64 ","
			"\"deadline_miss\":%s,"
			"\"skip_fired\":%s}\n",
			graph_iter, id_buf, HardwareTargetName(n.hardware_target),
			planned_start_us, start_us, end_us, run_us, planned_dur_us,
			deadline_violated ? "true" : "false",
			skip_fired ? "true" : "false");
		if (len <= 0 || len >= (int)sizeof(line))
			return; // drop oversized
		ssize_t off = 0;
		while (off < len) {
			ssize_t w = ::write(fd_, line + off, (size_t)(len - off));
			if (w < 0) {
				if (errno == EINTR)
					continue;
				return; // drop on persistent error; don't block worker
			}
			off += w;
		}
	}

  private:
	static void EscapeJsonString(const char *src, char *dst, size_t cap) {
		size_t out = 0;
		while (*src && out + 1 < cap) {
			char c = *src++;
			if (c == '"' || c == '\\') {
				if (out + 2 >= cap)
					break;
				dst[out++] = '\\';
				dst[out++] = c;
			} else if ((unsigned char)c < 0x20) {
				// drop control chars; dispatch ids should never contain them
				continue;
			} else {
				dst[out++] = c;
			}
		}
		dst[out] = 0;
	}

	int fd_ = -1;
	bool owns_fd_ = false;
	bool active_ = false;
};

// XPU-RT schedule hot-swap watcher. Polls a path's mtime between graph
// iterations and atomically swaps in the new release-time / target /
// deadline / skip configuration when the file changes. Bounded to the
// fields the runtime can swap safely without rebuilding the dependency
// graph or module cache:
//   - DispatchNode::start_time_ms      (planned start, drives kPhaseLocked)
//   - DispatchNode::hardware_target    (which queue receives the node)
//   - DispatchNode::deadline_ms        (real-time deadline)
//   - DispatchNode::skipped            (drop directive)
//
// Adding/removing dispatches or moving them to a different VMFB requires
// a recompile, so those changes are NOT applied — the watcher logs the
// mismatch and keeps the live graph intact.
class ScheduleHotSwap {
  public:
	void SetPath(const char *path) {
		path_ = (path && path[0]) ? path : "";
		last_mtime_ns_ = 0;
		seen_first_check_ = false;
	}

	bool active() const {
		return !path_.empty();
	}

	// Returns true if the file changed and the swap was applied. Should
	// only be called between graph iterations (when no worker is reading
	// node state). |live_mu| is held only for the brief field copy.
	bool MaybeApply(GraphModel *live, std::mutex *live_mu) {
		if (!active())
			return false;
		struct stat st;
		if (::stat(path_.c_str(), &st) != 0) {
			return false; // not yet present; not an error
		}
		const uint64_t mtime_ns = (uint64_t)st.st_mtim.tv_sec * 1000000000ull +
			(uint64_t)st.st_mtim.tv_nsec;
		if (!seen_first_check_) {
			// First call: record the baseline mtime without swapping. An
			// old schedule_next.json sitting on disk should not trigger
			// a spurious swap on the very first iteration.
			last_mtime_ns_ = mtime_ns;
			seen_first_check_ = true;
			return false;
		}
		if (mtime_ns == last_mtime_ns_)
			return false;

		GraphModel incoming;
		if (!ParseDispatchScheduleJson(path_, &incoming)) {
			fprintf(stderr, "[hot-swap] failed to parse %s; ignoring\n",
				path_.c_str());
			last_mtime_ns_ = mtime_ns;
			return false;
		}

		std::unordered_map<std::string, const DispatchNode *> by_key;
		by_key.reserve(incoming.nodes.size());
		for (const auto &n : incoming.nodes) {
			by_key.emplace(n.key, &n);
		}

		int swapped = 0, missing = 0;
		{
			std::lock_guard<std::mutex> lk(*live_mu);
			for (auto &live_n : live->nodes) {
				auto it = by_key.find(live_n.key);
				if (it == by_key.end()) {
					missing++;
					continue;
				}
				const DispatchNode &src = *it->second;
				live_n.start_time_ms = src.start_time_ms;
				live_n.hardware_target = src.hardware_target;
				live_n.deadline_ms = src.deadline_ms;
				live_n.skipped = src.skipped;
				swapped++;
				by_key.erase(it);
			}
		}
		const int new_unsupported = (int)by_key.size();

		fprintf(stdout,
			"[hot-swap] applied %s: swapped=%d missing=%d "
			"new_unsupported=%d\n",
			path_.c_str(), swapped, missing, new_unsupported);
		fflush(stdout);
		last_mtime_ns_ = mtime_ns;
		return true;
	}

  private:
	std::string path_;
	uint64_t last_mtime_ns_ = 0;
	bool seen_first_check_ = false;
};

// Create a QNN HAL device for the given backend path ("gpu" or "htp").
// Pulls the QNN driver out of the runtime instance's driver registry, which
// is populated by `use_all_available_drivers` when the QNN HAL plugin is
// linked in. Returns IREE_STATUS_UNAVAILABLE if the QNN driver isn't
// registered (e.g. host build without libQnn{Gpu,Hta}.so).
static iree_status_t CreateQnnDevice(iree_runtime_instance_t *instance,
	const char *backend_path, iree_allocator_t host_allocator,
	iree_hal_device_t **out_device) {
#if MERLIN_HAS_PROACTOR_POOL
	*out_device = nullptr;
	iree_hal_driver_registry_t *registry =
		iree_runtime_instance_driver_registry(instance);
	if (!registry) {
		return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
			"runtime instance has no driver registry; cannot create QNN "
			"device");
	}

	iree_hal_driver_t *driver = nullptr;
	IREE_RETURN_IF_ERROR(iree_hal_driver_registry_try_create(
		registry, IREE_SV("qnn"), host_allocator, &driver));

	iree_async_proactor_pool_t *proactor_pool = nullptr;
	iree_status_t st = iree_async_proactor_pool_create(iree_numa_node_count(),
		/*node_ids=*/nullptr, iree_async_proactor_pool_options_default(),
		host_allocator, &proactor_pool);
	if (iree_status_is_ok(st)) {
		iree_hal_device_create_params_t create_params =
			iree_hal_device_create_params_default();
		create_params.proactor_pool = proactor_pool;
		st = iree_hal_driver_create_device_by_path(driver, IREE_SV("qnn"),
			iree_make_cstring_view(backend_path),
			/*param_count=*/0, /*params=*/nullptr, &create_params,
			host_allocator, out_device);
	}
	iree_async_proactor_pool_release(proactor_pool);
	iree_hal_driver_release(driver);
	return st;
#else
	(void)instance;
	(void)backend_path;
	(void)host_allocator;
	*out_device = nullptr;
	return iree_make_status(IREE_STATUS_UNAVAILABLE,
		"QNN device creation requires IREE proactor pool API");
#endif
}

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
	// Per-target cvs eliminate wakeup churn: a dispatch completing on
	// CPU_P only wakes the CPU_P worker whose ready queue gained the
	// child, not every worker. Saves ~30-60 us per dispatch on
	// non-amortized runs (4 workers × ~15 us context-switch back to
	// re-wait). The legacy `cv` field is kept for global broadcasts
	// (shutdown, graph-iter completion) where every waiter needs to
	// wake.
	std::condition_variable cv;
	std::condition_variable cv_cpu_p;
	std::condition_variable cv_cpu_e;
	std::condition_variable cv_qnn_gpu;
	std::condition_variable cv_qnn_hta;
	std::condition_variable cv_cpu;

	bool shutdown = false;
	bool active = false;

	int current_graph_iter = 0;
	Clock::time_point iter_t0{};

	std::vector<int> remaining_preds;
	std::vector<NodeExecState> exec;

	std::vector<int> ready_p;
	std::vector<int> ready_e;
	std::vector<int> ready_qnn_gpu;
	std::vector<int> ready_qnn_hta;
	std::vector<int> ready_cpu;

	std::vector<int> future_p;
	std::vector<int> future_e;
	std::vector<int> future_qnn_gpu;
	std::vector<int> future_qnn_hta;
	std::vector<int> future_cpu;

	size_t completed = 0;
	size_t total_nodes = 0;
};

static std::vector<int> &ReadyQueueFor(SchedulerShared *s, HardwareTarget t) {
	switch (t) {
		case HardwareTarget::kCpuP:
			return s->ready_p;
		case HardwareTarget::kCpuE:
			return s->ready_e;
		case HardwareTarget::kQnnGpu:
			return s->ready_qnn_gpu;
		case HardwareTarget::kQnnHta:
			return s->ready_qnn_hta;
		case HardwareTarget::kCpu:
			return s->ready_cpu;
	}
	return s->ready_p;
}

static std::vector<int> &FutureQueueFor(SchedulerShared *s, HardwareTarget t) {
	switch (t) {
		case HardwareTarget::kCpuP:
			return s->future_p;
		case HardwareTarget::kCpuE:
			return s->future_e;
		case HardwareTarget::kQnnGpu:
			return s->future_qnn_gpu;
		case HardwareTarget::kQnnHta:
			return s->future_qnn_hta;
		case HardwareTarget::kCpu:
			return s->future_cpu;
	}
	return s->future_p;
}

// Wake every waiter — used by main-thread events (graph-iter start,
// shutdown) + fatal-error fan-out. Workers listen on per-target cvs
// during normal dispatch flow but must also wake on these.
static void WakeAllWorkers(SchedulerShared *s) {
	s->cv.notify_all();
	s->cv_cpu_p.notify_all();
	s->cv_cpu_e.notify_all();
	s->cv_qnn_gpu.notify_all();
	s->cv_qnn_hta.notify_all();
	s->cv_cpu.notify_all();
}

static std::condition_variable &CvForTarget(
	SchedulerShared *s, HardwareTarget t) {
	switch (t) {
		case HardwareTarget::kCpuP:
			return s->cv_cpu_p;
		case HardwareTarget::kCpuE:
			return s->cv_cpu_e;
		case HardwareTarget::kQnnGpu:
			return s->cv_qnn_gpu;
		case HardwareTarget::kQnnHta:
			return s->cv_qnn_hta;
		case HardwareTarget::kCpu:
			return s->cv_cpu;
	}
	return s->cv_cpu_p;
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
	sched->ready_qnn_gpu.clear();
	sched->ready_qnn_hta.clear();
	sched->ready_cpu.clear();
	sched->future_p.clear();
	sched->future_e.clear();
	sched->future_qnn_gpu.clear();
	sched->future_qnn_hta.clear();
	sched->future_cpu.clear();
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
	TraceWriter *trace, TelemetrySink *telemetry) {
	BestEffortPinCurrentThreadToCpuIds(cpu_ids_csv);

	while (true) {
		int node_idx = -1;
		int graph_iter = 0;
		Clock::time_point iter_t0;

		{
			std::unique_lock<std::mutex> lock(sched->mu);

			std::condition_variable &my_cv = CvForTarget(sched, target);
			while (true) {
				if (sched->shutdown || HasFatal(fatal))
					return;
				if (!sched->active) {
					// Inactive (between graph iters) — wait on the
					// global cv since the main thread broadcasts there.
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
					// Wait on OUR target's cv only — eliminates spurious
					// wakeups when other targets get new dispatches.
					my_cv.wait(lock);
				} else {
					// Spin-wait for short sleeps to avoid condvar
					// timer overshoot (~2ms on RISC-V kernel).
					const uint64_t now2 = UsSince(sched->iter_t0, Clock::now());
					if (next_release_us > now2 + 5000) {
						// Long wait: per-target cv until 2ms before release.
						const auto wake_tp = sched->iter_t0 +
							std::chrono::microseconds(next_release_us - 2000);
						my_cv.wait_until(lock, wake_tp);
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

		// Robotics-deadline support (PR 6 of the rosy-sundae plan):
		//
		//  - skipped: the schedule explicitly told the runtime to drop
		//    this node. Don't run the VMFB, record run_us=0, mark
		//    descendants ready immediately. The node's outputs are
		//    undefined for downstream chunks but that's the schedule's
		//    contract.
		//  - deadline_ms: when set, refuse to start this node if doing
		//    so would push past the planned deadline. Trace records
		//    deadline_violation=1 (we synthesise a zero-duration row
		//    but mark it so the plot can render it as such).
		const DispatchNode &cur_node = (*nodes)[(size_t)node_idx];
		const uint64_t now_for_check = UsSince(iter_t0, Clock::now());
		bool skip_run = cur_node.skipped;
		bool deadline_violated = false;
		if (!skip_run && cur_node.deadline_ms > 0.0) {
			const uint64_t deadline_us = MsToUs(cur_node.deadline_ms);
			// Estimate worst-case finish: wall time now + planned dur.
			// If even the optimistic case would miss, drop the node.
			const uint64_t opt_finish =
				now_for_check + MsToUs(cur_node.planned_duration_ms);
			if (opt_finish > deadline_us) {
				deadline_violated = true;
				skip_run = true;
			}
		}

		iree_status_t st = iree_ok_status();
		uint64_t end_us;
		uint64_t invoke_only_us = 0;
		if (skip_run) {
			end_us = now_for_check; // zero-duration record
		} else {
			const auto invoke_t0 = Clock::now();
			st = CallModuleUnlocked((*node_modules)[(size_t)node_idx],
				(int32_t)dispatch_iters, host_alloc);
			const auto invoke_t1 = Clock::now();
			end_us = UsSince(iter_t0, invoke_t1);
			invoke_only_us =
				(uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
					invoke_t1 - invoke_t0)
					.count();
		}
		(void)invoke_only_us;

		if (!iree_status_is_ok(st)) {
			SetFatalOnce(
				fatal, st, "[dispatch] sync benchmark module call failed");
			WakeAllWorkers(sched);
			return;
		}

		uint64_t planned_start_us = 0;
		uint64_t ready_us = 0;
		uint64_t start_us = 0;
		// Bitmask of HardwareTarget values whose ready/future queues
		// gained a node; we wake only those. Saves the wakeup churn
		// across workers that have no new work.
		uint8_t targets_woken = 0;
		bool graph_done = false;

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

					HardwareTarget child_t =
						(*nodes)[(size_t)child].hardware_target;
					if (cs.release_us <= UsSince(iter_t0, Clock::now())) {
						ReadyQueueFor(sched, child_t).push_back(child);
					} else {
						InsertFutureSorted(&FutureQueueFor(sched, child_t),
							*nodes, sched->exec, child);
					}
					targets_woken |= (uint8_t)(1u << (uint8_t)child_t);
				}
			}

			sched->completed++;
			if (sched->completed == sched->total_nodes) {
				sched->active = false;
				graph_done = true;
			}
		}

		trace->WriteRow(graph_iter, (*nodes)[(size_t)node_idx],
			planned_start_us, ready_us, start_us, end_us);

		// XPU-RT telemetry: per-dispatch JSON-Lines for hardware-in-the-loop
		// feedback. Inert when neither path nor fd was configured (sink
		// stays inactive across the full lifetime). This is what lets the
		// host-side daemon stream feedback while the workload runs.
		if (telemetry && telemetry->active()) {
			telemetry->EmitDispatchEnd(graph_iter, (*nodes)[(size_t)node_idx],
				planned_start_us, start_us, end_us, deadline_violated,
				skip_run);
		}

		// Wake only the targets whose ready queues gained a node.
		// graph_done additionally wakes the main thread (global cv).
		if (targets_woken & (1u << (uint8_t)HardwareTarget::kCpuP))
			sched->cv_cpu_p.notify_one();
		if (targets_woken & (1u << (uint8_t)HardwareTarget::kCpuE))
			sched->cv_cpu_e.notify_one();
		if (targets_woken & (1u << (uint8_t)HardwareTarget::kQnnGpu))
			sched->cv_qnn_gpu.notify_one();
		if (targets_woken & (1u << (uint8_t)HardwareTarget::kQnnHta))
			sched->cv_qnn_hta.notify_one();
		if (targets_woken & (1u << (uint8_t)HardwareTarget::kCpu))
			sched->cv_cpu.notify_one();
		if (graph_done) {
			// Graph complete — wake main thread + any inactive workers.
			WakeAllWorkers(sched);
		}
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
	fprintf(f, "    \"qnn_gpu_enabled\": %d,\n", cfg->qnn_gpu_enabled);
	fprintf(f, "    \"qnn_hta_enabled\": %d,\n", cfg->qnn_hta_enabled);
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
	iree_hal_device_t *device_qnn_gpu = nullptr;
	iree_hal_device_t *device_qnn_hta = nullptr;
	iree_hal_device_t *device_cpu = nullptr;

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
		iree_status_t st = cfg->cpu_use_local_sync
			? CreateSyncLocalDevice(host_alloc, &device_p)
			: CreatePinnedLocalTaskDevice(
				  host_alloc, cfg->cpu_p_cpu_ids, &device_p);
		if (!iree_status_is_ok(st)) {
			fprintf(stderr, "Failed creating %s CPU_P device\n",
				cfg->cpu_use_local_sync ? "local-sync" : "pinned local-task");
			iree_status_fprint(stderr, st);
			iree_status_ignore(st);
			iree_runtime_instance_release(instance);
			trace.Close();
			return 1;
		}
	}

	{
		iree_status_t st = cfg->cpu_use_local_sync
			? CreateSyncLocalDevice(host_alloc, &device_e)
			: CreatePinnedLocalTaskDevice(
				  host_alloc, cfg->cpu_e_cpu_ids, &device_e);
		if (!iree_status_is_ok(st)) {
			fprintf(stderr, "Failed creating %s CPU_E device\n",
				cfg->cpu_use_local_sync ? "local-sync" : "pinned local-task");
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

	// Detect what the schedule needs so we don't fail outright on pure-CPU
	// schedules just because the user forgot a flag.
	bool schedule_needs_qnn_gpu = false;
	bool schedule_needs_qnn_hta = false;
	bool schedule_needs_cpu = false;
	for (const auto &n : model.nodes) {
		if (n.hardware_target == HardwareTarget::kQnnGpu)
			schedule_needs_qnn_gpu = true;
		if (n.hardware_target == HardwareTarget::kQnnHta)
			schedule_needs_qnn_hta = true;
		if (n.hardware_target == HardwareTarget::kCpu)
			schedule_needs_cpu = true;
	}

	// Unified CPU device (all 8 cores, single device). When the schedule has
	// any kCpu nodes, build it now. Uses the same pinned local-task path as
	// CPU_P / CPU_E with whatever cpu_cpu_ids the caller supplied (default
	// "0,1,2,3,4,5,6,7" if missing — covers QRB5165 and similar 8-core SoCs).
	if (schedule_needs_cpu) {
		const char *cpu_ids = (cfg->cpu_cpu_ids && cfg->cpu_cpu_ids[0])
			? cfg->cpu_cpu_ids
			: "0,1,2,3,4,5,6,7";
		iree_status_t st =
			CreatePinnedLocalTaskDevice(host_alloc, cpu_ids, &device_cpu);
		if (!iree_status_is_ok(st)) {
			fprintf(stderr, "Failed creating unified CPU device\n");
			iree_status_fprint(stderr, st);
			iree_status_ignore(st);
			if (device_e)
				iree_hal_device_release(device_e);
			if (device_p)
				iree_hal_device_release(device_p);
			iree_runtime_instance_release(instance);
			trace.Close();
			return 1;
		}
		fprintf(stdout, "[dispatch] CPU local-task topology = {%s}\n", cpu_ids);
		fflush(stdout);
	}

	if ((cfg->qnn_gpu_enabled || schedule_needs_qnn_gpu)) {
		iree_status_t st =
			CreateQnnDevice(instance, "gpu", host_alloc, &device_qnn_gpu);
		if (!iree_status_is_ok(st)) {
			fprintf(stderr, "Failed creating QNN_GPU device\n");
			iree_status_fprint(stderr, st);
			iree_status_ignore(st);
			if (device_e)
				iree_hal_device_release(device_e);
			if (device_p)
				iree_hal_device_release(device_p);
			iree_runtime_instance_release(instance);
			trace.Close();
			return 1;
		}
		fprintf(stdout, "[dispatch] QNN_GPU device created (libQnnGpu.so)\n");
		fflush(stdout);
	}

	if ((cfg->qnn_hta_enabled || schedule_needs_qnn_hta)) {
		// QRB5165 (Snapdragon 865) uses libQnnHta.so for the Hexagon
		// 698 NPU. libQnnHtp.so requires Snapdragon 8 Gen 1+ silicon
		// and rejects this board with "Unsupported SnapdragonModel".
		iree_status_t st =
			CreateQnnDevice(instance, "hta", host_alloc, &device_qnn_hta);
		if (!iree_status_is_ok(st)) {
			fprintf(stderr, "Failed creating QNN_HTA (htp) device\n");
			iree_status_fprint(stderr, st);
			iree_status_ignore(st);
			if (device_qnn_gpu)
				iree_hal_device_release(device_qnn_gpu);
			if (device_e)
				iree_hal_device_release(device_e);
			if (device_p)
				iree_hal_device_release(device_p);
			iree_runtime_instance_release(instance);
			trace.Close();
			return 1;
		}
		fprintf(
			stdout, "[dispatch] QNN_HTA device created (libQnnHtp.so/HTA)\n");
		fflush(stdout);
	}

	auto device_for_target = [&](HardwareTarget t) -> iree_hal_device_t * {
		switch (t) {
			case HardwareTarget::kCpuP:
				return device_p;
			case HardwareTarget::kCpuE:
				return device_e;
			case HardwareTarget::kQnnGpu:
				return device_qnn_gpu;
			case HardwareTarget::kQnnHta:
				return device_qnn_hta;
			case HardwareTarget::kCpu:
				return device_cpu;
		}
		return device_p;
	};

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
				device_for_target(node.hardware_target);
			if (!target_device) {
				fprintf(stderr,
					"No device available for node %s target=%s "
					"(missing --qnn_gpu_enabled / --qnn_hta_enabled?)\n",
					node.key.c_str(), HardwareTargetName(node.hardware_target));
				trace.Close();
				if (device_qnn_hta)
					iree_hal_device_release(device_qnn_hta);
				if (device_qnn_gpu)
					iree_hal_device_release(device_qnn_gpu);
				if (device_e)
					iree_hal_device_release(device_e);
				if (device_p)
					iree_hal_device_release(device_p);
				if (instance)
					iree_runtime_instance_release(instance);
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
				if (device_qnn_hta)
					iree_hal_device_release(device_qnn_hta);
				if (device_qnn_gpu)
					iree_hal_device_release(device_qnn_gpu);
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

		// Run each module ONCE at warmup so the underlying compute
		// (not just the IREE call path) is JIT'd, the executable
		// loaded into the device's executable cache, and per-target
		// driver state (libQnnGpu/Hta context binaries, embedded ELF
		// images, etc.) is in steady state. Passing 0 to an i32-arg
		// wrapper would skip the embedded function body — defeating
		// the warmup. Passing 1 runs the body exactly once.
		const int32_t warm_iters = (cm->arity == 1 && cm->first_is_i32)
			? 1
			: static_cast<int32_t>(dispatch_iters);

		// Multiple warm calls amortize first-call cold-cache cost
		// further. Two passes balances startup time vs first-iter
		// jitter on the trace.
		iree_status_t st = iree_ok_status();
		for (int w = 0; w < 2 && iree_status_is_ok(st); ++w) {
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
			if (device_qnn_hta)
				iree_hal_device_release(device_qnn_hta);
			if (device_qnn_gpu)
				iree_hal_device_release(device_qnn_gpu);
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

	// Open the XPU-RT telemetry sink. Inert when neither cfg field is set
	// (the sink stays in its default-inactive state and EmitDispatchEnd
	// fast-paths to a no-op). Stays open for the full run.
	TelemetrySink telemetry;
	if (telemetry.Open(cfg->telemetry_jsonl_path, cfg->telemetry_fd) != 0) {
		fprintf(stderr,
			"[telemetry] sink failed to open; continuing without telemetry\n");
	} else if (telemetry.active()) {
		fprintf(stdout, "[telemetry] active (path=%s, fd=%d)\n",
			cfg->telemetry_jsonl_path ? cfg->telemetry_jsonl_path : "(none)",
			cfg->telemetry_fd);
	}

	// XPU-RT schedule hot-swap watcher. Inert when cfg->schedule_next_path
	// is unset. When active, the watcher is consulted at every graph
	// iteration boundary (between epochs); a new schedule_next.json
	// triggers an atomic field-level swap on the existing model.nodes.
	ScheduleHotSwap hot_swap;
	hot_swap.SetPath(cfg->schedule_next_path);
	if (hot_swap.active()) {
		fprintf(stdout, "[hot-swap] watching %s\n", cfg->schedule_next_path);
	}

	std::thread worker_p(WorkerMain, HardwareTarget::kCpuP, cfg->cpu_p_cpu_ids,
		&model.nodes, &dependents, &node_modules, dispatch_iters, host_alloc,
		&shared, &sched, &trace, &telemetry);

	std::thread worker_e(WorkerMain, HardwareTarget::kCpuE, cfg->cpu_e_cpu_ids,
		&model.nodes, &dependents, &node_modules, dispatch_iters, host_alloc,
		&shared, &sched, &trace, &telemetry);

	// QNN worker threads are unpinned — the QNN HAL spawns its own worker
	// inside the SDK runtime and our dispatch loop just submits + waits.
	std::unique_ptr<std::thread> worker_qnn_gpu;
	if (device_qnn_gpu) {
		worker_qnn_gpu = std::make_unique<std::thread>(WorkerMain,
			HardwareTarget::kQnnGpu, /*cpu_ids_csv=*/"", &model.nodes,
			&dependents, &node_modules, dispatch_iters, host_alloc, &shared,
			&sched, &trace, &telemetry);
	}
	std::unique_ptr<std::thread> worker_qnn_hta;
	if (device_qnn_hta) {
		worker_qnn_hta = std::make_unique<std::thread>(WorkerMain,
			HardwareTarget::kQnnHta, /*cpu_ids_csv=*/"", &model.nodes,
			&dependents, &node_modules, dispatch_iters, host_alloc, &shared,
			&sched, &trace, &telemetry);
	}

	// Unified CPU worker (8 cores) — pinned to whichever cpu_cpu_ids the
	// caller supplied (default 0..7). Only spawned when device_cpu exists.
	std::unique_ptr<std::thread> worker_cpu;
	if (device_cpu) {
		const char *cpu_ids = (cfg->cpu_cpu_ids && cfg->cpu_cpu_ids[0])
			? cfg->cpu_cpu_ids
			: "0,1,2,3,4,5,6,7";
		worker_cpu = std::make_unique<std::thread>(WorkerMain,
			HardwareTarget::kCpu, cpu_ids, &model.nodes, &dependents,
			&node_modules, dispatch_iters, host_alloc, &shared, &sched, &trace,
			&telemetry);
	}

	const auto run_t0 = Clock::now();

	for (int gi = 0; gi < graph_iters && !HasFatal(&shared); ++gi) {
		// XPU-RT schedule hot-swap. Runs at the epoch boundary, before
		// the per-iteration setup re-reads model.nodes[] start times into
		// sched.exec[].planned_start_us — that re-read picks up any
		// swapped values automatically. Workers from the prior iteration
		// are quiescent here (sched.active was set to false at the end of
		// the previous iteration), so the swap takes sched.mu only as a
		// formality to make the synchronization contract explicit.
		hot_swap.MaybeApply(&model, &sched.mu);

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
		// Iter starts: wake every worker so they re-check their queues.
		WakeAllWorkers(&sched);

		{
			std::unique_lock<std::mutex> lock(sched.mu);
			sched.cv.wait(lock, [&]() {
				return HasFatal(&shared) ||
					sched.completed == sched.total_nodes;
			});
			sched.active = false;
			sched.ready_p.clear();
			sched.ready_e.clear();
			sched.ready_qnn_gpu.clear();
			sched.ready_qnn_hta.clear();
			sched.ready_cpu.clear();
		}
		WakeAllWorkers(&sched);

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
		sched.ready_qnn_gpu.clear();
		sched.ready_qnn_hta.clear();
		sched.ready_cpu.clear();
	}
	// Final shutdown — every worker must exit.
	WakeAllWorkers(&sched);

	worker_p.join();
	worker_e.join();
	if (worker_qnn_gpu)
		worker_qnn_gpu->join();
	if (worker_qnn_hta)
		worker_qnn_hta->join();
	if (worker_cpu)
		worker_cpu->join();

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
	if (device_cpu)
		iree_hal_device_release(device_cpu);
	if (device_qnn_hta)
		iree_hal_device_release(device_qnn_hta);
	if (device_qnn_gpu)
		iree_hal_device_release(device_qnn_gpu);
	if (device_e)
		iree_hal_device_release(device_e);
	if (device_p)
		iree_hal_device_release(device_p);
	if (instance)
		iree_runtime_instance_release(instance);

	return HasFatal(&shared) ? 1 : 0;
}
