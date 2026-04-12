# 2026-03-27: Runtime Scheduling Baseline and PMCS v2 Direction

This dev blog is the follow-up to the original dispatch-scheduler tuning log in
[2026-03-16 Dispatch Scheduler + Tracy](2026-03-16-dispatch-level-async.md).

That earlier entry records how the current scheduler was debugged and fixed.
This entry answers a different question:

**what does Merlin's scheduling stack actually look like today, what does IREE
expose as the baseline, and what exactly should Merlin build next for dynamic
multi-workload scheduling on heterogeneous SoCs?**

It intentionally combines:

- the current Merlin/XPU-RT/IREE scheduling baseline,
- the scheduling-related compiler and runtime surfaces that IREE exposes,
- a file-by-file inventory of `iree/task` and `iree/async`,
- and the concrete design and implementation direction for
  **Merlin Predictive Mixed-Criticality Scheduler (PMCS) v2**.

## 0. Scope

This document covers the scheduling stack used by:

- `samples/common/xpu-rt/`
- `samples/SpacemiTX60/dispatch_scheduler/`
- `/scratch2/agustin/XPU-RT`
- `third_party/iree_bar/runtime/src/iree/task`
- `third_party/iree_bar/runtime/src/iree/async`

It starts from the current CPU-cluster path and extends that baseline to the
heterogeneous SoC direction Merlin should move toward.

## 1. Current Merlin Scheduling Baseline

The current dispatch scheduler is a layered stack, not one scheduler.

| Layer | Current owner | What it decides | Static vs dynamic |
| --- | --- | --- | --- |
| Compile-time execution model | IREE compiler | HAL execution model, dispatch shaping, initialization mode, compiler scheduling stats | Static |
| Offline workload planning | XPU-RT | Periodic expansion, precedence, placement windows, non-overlap, makespan-oriented planning | Static ahead of time |
| Runtime graph semantics | Merlin dispatch parser and types | `hardware_target`, release policy, hard vs soft timing rules | Static metadata interpreted at runtime |
| Host-side schedule enactment | Merlin `scheduler_runner` | Release timing, ready/future movement, target queue order, traces | Dynamic, but at host dispatch-node granularity |
| Per-target execution | IREE `local-task` + `iree/task` | Worker scheduling, affinity, dispatch sharding, queue draining, stealing | Dynamic |
| Cross-device async substrate | IREE `iree/async` | Semaphores, notifications, timers, frontiers, proactor polling | Dynamic substrate, not the current top-level scheduler |

The current SpacemiT path works like this:

1. models are compiled with an async-capable IREE execution model,
2. XPU-RT expands periodic workloads and computes an offline schedule,
3. Merlin parses that schedule JSON into dispatch nodes,
4. `scheduler_runner.cc` builds separate pinned `local-task` devices for
   `CPU_P` and `CPU_E`,
5. the host runtime tracks predecessor state plus per-target future and ready
   queues,
6. each runnable node is invoked through a cached IREE runtime session.

In practice:

- XPU-RT computes the plan,
- Merlin enforces release and dependency rules in host code,
- IREE executes the chosen work inside each target executor.

That means Merlin already owns the global multi-model policy today, even though
it relies on IREE as the per-target executor.

## 2. Static Versus Dynamic Scheduling In The Current Stack

### Static today

- IREE compile-time scheduling chooses the execution model and lowerings.
- XPU-RT computes a schedule offline and encodes planned starts, precedence,
  durations, and placements.
- Merlin's dispatch graph carries scheduling metadata such as target tags,
  release policy, and hard versus soft time-dependency semantics.
- the `CPU_P` versus `CPU_E` partition is fixed before runtime execution.

### Dynamic today

- Merlin dynamically releases nodes when predecessors finish or planned release
  time arrives,
- Merlin dynamically chooses the next runnable node inside each target queue,
- IREE dynamically balances work inside each `local-task` executor using ready
  queues, mailbox flushes, affinity constraints, dispatch sharding, and work
  stealing.

### Not dynamic yet

- there is no unified multi-model admission controller beyond the precomputed
  schedule and simple ready-queue rules,
- observed runtime does not yet reshape the global policy in a rich way,
- no shared runtime abstraction spans CPU, accelerator, DMA, and network
  readiness,
- Merlin does not currently use `iree/async` semaphores or frontiers as the
  top-level scheduling language.

## 3. Merlin Versus IREE Responsibilities

| Concern | Merlin today | IREE baseline |
| --- | --- | --- |
| Cross-model precedence and release timing | Yes | No, not at Merlin's graph granularity |
| Cluster partitioning for `CPU_P` / `CPU_E` | Yes | Yes, via topology and affinity once embedded |
| Worker scheduling inside a cluster | No | Yes, in `iree/task` |
| Dispatch sharding within a dispatch | No | Yes, in `iree/task` |
| Per-device ready queues and stealing | No | Yes, in `iree/task` |
| Async device, network, or file coordination | No | Yes, in `iree/async` |
| Timeline semaphores and causal frontiers | No | Yes, in `iree/async` |

This is the most important baseline conclusion:

**Merlin should keep IREE as the per-target executor and make the host
scheduler more dynamic above it before trying to redesign IREE internals.**

## 4. Compiler-Side Scheduling Surfaces In IREE

The vendored compiler exposes several scheduling-related controls that Merlin
can rely on.

| Surface | What it controls | Merlin relevance |
| --- | --- | --- |
| `--iree-execution-model=host-only` | No host/device async execution model | Too limited for the current scheduler sample |
| `--iree-execution-model=async-internal` | Async internally, synchronous external ABI | Useful in some embeddings, not the current XPU-RT path |
| `--iree-execution-model=async-external` | Async internally and externally | Current Merlin/XPU-RT path |
| `--iree-execution-model=inline-static` | Inline host-local execution with statically linked executables | Constrained embedded path |
| `--iree-execution-model=inline-dynamic` | Inline host-local execution with dynamic executables | Similar niche to `inline-static` |
| `--iree-scheduling-dump-statistics-file` | Emit scheduling statistics | Useful for offline analysis |
| `--iree-scheduling-dump-statistics-format` | Pretty, verbose, csv, or json output | Useful for tooling |
| `--iree-scheduling-initialization-mode=sync|async` | Module initializer behavior | Relevant for startup and bring-up |
| `--iree-scheduling-optimize-bindings` | Binding fusion and specialization | Can change the dispatch structure exposed to runtime |

Current Merlin targets such as `spacemit_x60`, `npu_ucb`, and `gemmini_mx`
already lean on async execution models.

## 5. Runtime Scheduling Surfaces In `iree/task`

### High-value public or embeddable APIs

- `iree_task_executor_create`
- `iree_task_executor_options_initialize`
- `iree_task_executor_options_initialize_from_flags`
- `iree_task_topology_initialize_from_logical_cpu_set_string`
- `iree_task_topology_initialize_from_flags`
- `iree_task_submission_*`
- `iree_task_scope_*`
- `iree_task_affinity_set_t`
- `iree_task_t` and the built-in task kinds

### Runtime flags that matter in experiments

- `--task_worker_spin_us`
- `--task_worker_stack_size`
- `--task_worker_local_memory`
- `--task_topology_mode`
- `--task_topology_group_count`
- `--task_topology_cpu_ids`
- `--task_topology_nodes`
- `--task_topology_max_group_count`
- `--task_topology_performance_level`
- `--task_topology_distribution`
- `--task_topology_favor`
- `--dump_task_topologies`

### Behavior that matters for Merlin

- the executor schedules ready work dynamically after submission,
- workers own local FIFO queues and can steal work,
- dispatch tasks are sharded dynamically by default,
- topology and affinity act as constraints on dynamic scheduling, not as a full
  scheduler-policy plug-in interface.

### Important limitation

`iree/task` does **not** expose a mature public interface for plugging in a
custom global fairness, deadline, or mixed-criticality scheduler across models.
The practical knobs today are:

- topology,
- affinity,
- worker tuning,
- task-graph structure,
- dispatch sharding behavior,
- executor count and partitioning.

## 6. Local-Task HAL Surfaces

Merlin's current runtime path goes through the `local-task` HAL driver and its
embedding seams:

- `iree_hal_task_driver_create`
- `iree_hal_driver_create_device_by_id`
- `CreatePinnedLocalTaskDevice` in `samples/common/runtime/pinned_device.h`

That pinned-device seam matters because Merlin already uses it to impose
per-cluster topology and affinity before work reaches IREE's executor.

## 7. Async Surfaces In `iree/async`

`iree/async` is not the CPU task scheduler. It is a proactor-based async I/O
and synchronization substrate.

The key surfaces for Merlin are:

- `iree_async_proactor_t`
- `iree_async_semaphore_t`
- `iree_async_frontier_t`
- `iree_async_frontier_tracker_t`
- `iree_async_notification_t`
- `iree_async_sequence_operation_t`
- timer, event-wait, and semaphore operations

For Merlin:

- `iree/task` is the per-target CPU-style executor,
- `iree/async` is the future baseline for device readiness, semaphore wiring,
  DMA or transfer sequencing, and causal ordering across queues or devices.

## 8. Inventory: `iree/task`

The list below covers the scheduling-relevant files under
`third_party/iree_bar/runtime/src/iree/task`.

### 8.1 Core Sources

- `BUILD.bazel`, `CMakeLists.txt`: build wiring for the task runtime and its
  tests and benchmarks
- `affinity_set.h`: worker-affinity bitset helpers and the low-level placement
  constraint type
- `api.c`, `api.h`: task runtime flags, topology parsing from flags, executor
  factory helpers
- `executor.c`, `executor.h`, `executor_impl.h`: executor design, ready-task
  scheduling, coordinator path, worker and executor data structures
- `list.c`, `list.h`: intrusive task-list utilities used across submissions and
  queues
- `pool.c`, `pool.h`: task-object pooling and allocator helpers
- `post_batch.c`, `post_batch.h`: ready-task fan-out from coordinator logic to
  worker mailboxes
- `queue.c`, `queue.h`: worker-local FIFO queue plus stealing path
- `scope.c`, `scope.h`: task scopes, failure propagation, idle waiting, and
  scope statistics
- `submission.c`, `submission.h`: submission containers for DAG roots and ready
  tasks
- `task.c`, `task.h`, `task_impl.h`: task header, task kinds, dependency
  mechanics, dispatch issue and retire support
- `topology.c`, `topology.h`: topology representation, formatting, parsing, and
  helper constructors
- `topology_cpuinfo.c`: topology discovery through `cpuinfo`
- `topology_darwin.c`: Darwin-specific topology discovery
- `topology_emscripten.c`: Emscripten topology fallback
- `topology_fallback.c`: conservative fallback topology implementation
- `topology_sysfs.c`: Linux sysfs-based topology discovery
- `topology_win32.c`: Windows topology discovery
- `tuning.h`: compile-time executor tuning constants and hard limits
- `worker.c`, `worker.h`: worker-thread lifecycle, mailbox handling, local
  queue ownership, and stealing

### 8.2 Demos, Tests, And Benchmarks

- `executor_demo.cc`: minimal example of building and submitting a task DAG
- `executor_test.cc`: executor behavior tests
- `list_test.cc`: intrusive-list tests
- `pool_test.cc`: pool tests
- `queue_test.cc`: queue and stealing tests
- `scope_test.cc`: scope, failure, and idle tests
- `task_test_barrier.cc`, `task_test_call.cc`, `task_test_dispatch.cc`,
  `task_test_fence.cc`, `task_test_nop.cc`: unit tests for the main task kinds
- `topology_test.cc`: topology parsing and discovery tests

### 8.3 Benchmark Support

- `benchmarks/BUILD.bazel`, `benchmarks/CMakeLists.txt`: benchmark build rules
- `benchmarks/benchmark_base.h`, `benchmarks/benchmark_main.cc`: shared
  benchmark harness
- `benchmarks/dispatch_benchmark.cc`: dispatch overhead and throughput benchmark
- `benchmarks/wake_benchmark.cc`: wake and wait cost benchmark
- `benchmarks/workload_benchmark.cc`: workload-shape benchmark useful for
  imbalance and dynamic scheduling studies

### 8.4 Testing Utilities And Data

- `testing/BUILD.bazel`, `testing/CMakeLists.txt`: testing support build rules
- `testing/task_test.h`: task test scaffolding
- `testing/test_util.h`: shared testing helpers
- `testdata/sysfs/.gitignore`: keeps captured topology fixtures out of source
  control noise
- `testdata/sysfs/README.md`: how to capture and replay sysfs topology data
- `testdata/sysfs/arm64_pixel6_tensor.tar.gz`: heterogeneous mobile fixture
- `testdata/sysfs/capture_sysfs.sh`: helper for capturing topology data from a
  real system

These tests matter because they show the actual executor behavior Merlin must
build on.

## 9. Inventory: `iree/async`

The list below covers the scheduling-relevant files under
`third_party/iree_bar/runtime/src/iree/async`.

### 9.1 Top-Level Build And Core API

- `BUILD.bazel`, `CMakeLists.txt`: build wiring for async core, backends, tests,
  and benchmarks
- `README.md`: architectural overview of the proactor-based async system
- `affinity.h`: async-side thread-affinity declarations for backend helpers
- `api.h`: umbrella header for the public async surface
- `types.h`: common async type definitions
- `primitive.c`, `primitive.h`, `primitive_test.cc`: portable waitable
  primitive abstraction and tests
- `region.c`, `region.h`: memory-region registration and ownership helpers
- `span.h`: scatter-gather span definitions
- `slab.c`, `slab.h`: slab allocation support used by async operations

### 9.2 Resource And Core-Object Types

- `address.c`, `address.h`, `address_generic.c`, `address_impl.h`,
  `address_posix.c`, `address_win32.c`, `address_test.cc`, `address_fuzz.cc`:
  address parsing and platform encoding
- `buffer_pool.c`, `buffer_pool.h`: pooled registered-buffer support
- `event.c`, `event.h`: async event resource
- `event_pool.c`, `event_pool.h`, `event_pool_test.cc`: event pooling helpers
- `file.c`, `file.h`: async file wrapper
- `socket.c`, `socket.h`: async socket wrapper
- `operation.c`, `operation.h`: base async operation lifecycle
- `notification.c`, `notification.h`: lightweight epoch-based wakeup primitive
- `relay.h`: notification and semaphore relays
- `proactor.c`, `proactor.h`: central proactor abstraction and capabilities
- `proactor_platform.c`, `proactor_platform.h`: backend selection
- `semaphore.c`, `semaphore.h`, `semaphore_test.cc`: timeline semaphores,
  timepoints, links, and failure propagation
- `frontier.c`, `frontier.h`, `frontier_test.cc`, `frontier_benchmark.cc`:
  causal frontier representation and merge logic
- `frontier_tracker.c`, `frontier_tracker.h`,
  `frontier_tracker_test.cc`, `frontier_tracker_benchmark.cc`: frontier
  aggregation and tracking

### 9.3 Operation Subtypes

- `operations/file.h`: file I/O operation types
- `operations/futex.h`: futex-related operations
- `operations/message.h`: message-oriented operations
- `operations/net.h`: socket and network operations
- `operations/scheduling.h`: nop, timer, event-wait, and sequence operations
- `operations/semaphore.h`: semaphore wait and signal operations

### 9.4 Utility Layer

- `util/BUILD.bazel`, `util/CMakeLists.txt`: build wiring for async utilities
- `util/completion_pool.c`, `util/completion_pool.h`: reusable completion pool
- `util/continuation.c`, `util/continuation.h`,
  `util/continuation_test.cc`: continuation helpers for completion flow
- `util/intrusive_list.h`: intrusive-list utilities for async internals
- `util/message_pool.c`, `util/message_pool.h`: pooled message buffers
- `util/operation_pool.c`, `util/operation_pool.h`,
  `util/operation_pool_test.cc`: operation-object pooling
- `util/proactor_pool.c`, `util/proactor_pool.h`,
  `util/proactor_pool_types.h`, `util/proactor_pool_test.cc`: multi-proactor
  pooling and fan-out support
- `util/proactor_thread.c`, `util/proactor_thread.h`,
  `util/proactor_thread_runner.c`, `util/proactor_thread_runner.h`: optional
  dedicated-thread wrappers for a caller-driven proactor
- `util/ready_pool.c`, `util/ready_pool.h`: ready-object pooling helpers
- `util/sequence_emulation.c`, `util/sequence_emulation.h`: user-space
  emulation for backends without native operation chaining
- `util/signal.c`, `util/signal.h`: signal-handling helpers
- `util/test_base.h`: shared async utility test harness

### 9.5 Platform Backends

- `platform/BUILD.bazel`, `platform/CMakeLists.txt`: backend selection wiring
- `platform/linux/*`: Linux-specific signal helpers
- `platform/posix/*`: POSIX backend support
- `platform/win32/*`: Win32 backend support

The main takeaway is that `iree/async` already contains the semaphore,
notification, timer, sequencing, and causal-frontier tools Merlin will need for
real heterogeneous coordination. Merlin just does not use that path as its
top-level scheduler yet.

## 10. Why Replay Is No Longer Enough

The original scheduler changes improved the `dronet + mlp` path by adding:

- explicit release-time tracking,
- future and ready queues,
- different timing semantics for MLP versus `dronet`,
- better trace capture.

Those were useful steps, but they did not solve the deeper problem:

- runtime duration still drifts away from offline duration,
- multiple workloads still compete dynamically,
- transfer and memory cost still sit mostly outside the online policy,
- one fixed replay order is too rigid for heterogeneous SoCs.

That is why the next feature should not be framed as "better replay." It should
be framed as:

**replace exact replay in `scheduler_runner` with a runtime admission
controller.**

## 11. PMCS v2: The Concrete Feature

The concrete feature to build is:

**Merlin Predictive Mixed-Criticality Scheduler (PMCS) v2**

PMCS v2 should:

1. treat XPU-RT output as **windows, hints, constraints, and policy bundles**
   instead of a literal schedule,
2. use **prediction-aware scoring** for `(stage, implementation, target)`
   selection,
3. enforce a **hard safety shield** for critical chains with deterministic
   fallback,
4. support **runtime objective switching** among latency, energy, fairness, and
   thermal or risk modes,
5. keep **IREE as the per-target executor**, not the global scheduler.

## 12. Design Lineage

PMCS v2 should be informed by four main idea sources.

### XAUTO / XNODE

This is the closest match to Merlin's current abstraction gap:

- schedule **stages**, not whole modules,
- allow multiple hardware implementations per stage,
- solve assignment and priority jointly offline,
- keep per-target coordinators below one global scheduler.

This changes Merlin's primary runtime abstraction.

### Stream

This is the strongest planner-side reference:

- optional fine-grained graph expansion below whole-dispatch level,
- memory-aware and communication-aware cost modeling,
- steady-state optimization for repeated workloads,
- constrained offline planning over heterogeneous targets.

This changes how XPU-RT should think about cost and planning.

### Mixed-criticality adaptive-control work

This matters most for:

- two-timescale control,
- mixed-criticality constraints,
- verification plus deterministic fallback.

This changes the safety and control architecture.

### Objective-switching and priority-aware scheduling work

These matter for:

- runtime-switchable objectives,
- separating metrics, weights, and policy,
- anti-starvation behavior instead of naive strict priority.

This changes the runtime objective engine and fairness policy.

## 13. Scope, Non-Goals, And Invariants

### Scope

PMCS v2 is the new global scheduler for Merlin workloads described by
compiler-generated stage graphs and executed through XPU-RT plus IREE.

### Non-goals

- do not rewrite `iree/task`
- do not move global scheduling into IREE
- do not start with end-to-end RL in the hot path
- do not start with multi-host or distributed scheduling
- do not require actual in-flight interruption support on every target in v1
- do not force tile-level expansion for every workload before PMCS is usable

### Invariants

- the fast path stays in C++ inside `scheduler_runner`
- PMCS remains correct when all learned components are disabled
- learned logic may influence thresholds, margins, or weights, but may not
  bypass the safety guard
- IREE remains the per-target execution substrate
- legacy replay remains available during bring-up and A/B testing

## 14. PMCS v2 Architecture At A Glance

| Layer | Owner | Role in PMCS v2 |
| --- | --- | --- |
| Compiler lowering and execution model | Merlin + IREE compiler | Produce dispatches and target-specific runtime artifacts |
| Stage extraction and profiling | XPU-RT | Build stage graph, profile implementations, collect priors |
| Offline policy planner | XPU-RT | Solve assignment plus priority and emit `PolicyBundle`s |
| Global runtime scheduler | Merlin PMCS | Score ready stages and choose `(stage, impl, target)` |
| Per-target coordinators | Merlin PMCS | Own target-local pending and running state |
| Per-target execution | IREE runtime | Execute work on pinned executors or device queues |
| Cross-device readiness | IREE async / HAL | Surface semaphores, fences, and device completion |

## 15. Exact Delivery Order

PMCS v2 should be delivered in this order:

1. stage-graph schema and compatibility lifting
2. XPU-RT multi-implementation output
3. policy bundles
4. runtime estimator
5. online scorer for `(stage, impl, target)`
6. safety shield and deterministic fallback
7. objective engine with runtime mode switching
8. predictor sidecar and slow-loop integration
9. optional fine-grained offline tile planner
10. optional learned slow-loop controller

Do not invert this order. The system must be deterministic and safe before it
becomes predictive and adaptive.

## 16. Primary Scheduling Abstraction

PMCS v2 should replace "dispatch node already bound to one target" with "stage
plus execution alternatives."

### 16.1 Core Types

These are the runtime-facing types PMCS should introduce in
`samples/common/dispatch/dispatch_types.h`:

```cpp
enum class MerlinCriticality : uint8_t {
  kHardRealTime = 0,
  kSoftRealTime = 1,
  kBestEffort = 2,
};

enum class MerlinObjectiveMode : uint8_t {
  kLatencyFirst = 0,
  kEnergyFirst = 1,
  kBalanced = 2,
  kDeadlineRecovery = 3,
  kFairnessRecovery = 4,
  kThermalProtect = 5,
};

enum class MerlinStageKind : uint8_t {
  kModelExec = 0,
  kTransfer = 1,
  kMemoryLookup = 2,
  kHostCompute = 3,
  kControl = 4,
};

struct ResourceVector {
  double compute = 0.0;
  double memory_bw = 0.0;
  double memory_cap = 0.0;
  double network_bw = 0.0;
  double disk_bw = 0.0;
  double host_cpu = 0.0;
};

struct TargetCostEnvelope {
  uint64_t compute_us = 0;
  uint64_t input_transfer_us = 0;
  uint64_t output_transfer_us = 0;
  uint64_t queue_wait_us = 0;
  uint64_t launch_overhead_us = 0;
  uint64_t scratch_bytes = 0;
  uint64_t resident_bytes = 0;
  uint64_t offchip_bytes = 0;
  uint64_t interconnect_bytes = 0;
  uint64_t interconnect_us = 0;
  double bandwidth_bytes_per_s = 0.0;
};

struct StageImplementation {
  std::string impl_id;
  std::string target_id;
  std::string vmfb_path;
  std::string entry_name;

  uint64_t wcet_us = 0;
  uint64_t runtime_q50_us = 0;
  uint64_t runtime_q90_us = 0;
  uint64_t runtime_q99_us = 0;

  uint64_t launch_overhead_us = 0;
  uint64_t input_transfer_us = 0;
  uint64_t output_transfer_us = 0;
  uint64_t migration_penalty_us = 0;

  uint64_t scratch_bytes = 0;
  uint64_t working_set_bytes = 0;
  bool preemptible = false;
  bool checkpointable = false;
  bool stateful = false;

  ResourceVector demand;
  TargetCostEnvelope cost;
};

struct DispatchStage {
  std::string stage_id;
  std::string workload_id;
  std::string chain_id;
  MerlinStageKind kind = MerlinStageKind::kModelExec;

  std::vector<std::string> hard_predecessors;
  std::vector<std::string> soft_predecessors;
  std::vector<StageImplementation> implementations;

  uint64_t earliest_release_us = 0;
  uint64_t preferred_start_us = 0;
  uint64_t latest_useful_start_us = 0;
  uint64_t absolute_deadline_us = 0;
  uint64_t chain_deadline_us = 0;
  uint64_t chain_slack_budget_us = 0;

  MerlinCriticality criticality = MerlinCriticality::kBestEffort;
  int static_priority = 0;
  double soft_dep_penalty = 0.0;
  double fairness_floor = 0.0;
};

struct PolicyBundle {
  std::string bundle_id;
  std::unordered_map<std::string, std::string> preferred_impl_by_stage;
  std::unordered_map<std::string, int> priority_by_stage;
  std::unordered_map<std::string, double> reserve_fraction_by_target;
  std::unordered_map<std::string, double> risk_margin_by_target;
  std::unordered_map<std::string, double> affinity_bias_by_stage_target;
};
```

### 16.2 Compatibility rule

Legacy `DispatchNode` inputs must remain readable during transition, but PMCS
should treat them as a compatibility layer, not the final abstraction.

Lift them as:

- one `DispatchNode` -> one `DispatchStage`
- one `hardware_target` -> one `StageImplementation`
- `start_time_ms` and `planned_duration_ms` -> initial window hints
- `deps` -> `hard_predecessors`
- soft `time_dependency` -> `soft_predecessors`

## 17. Runtime Configuration Surface

Extend `samples/common/xpu-rt/scheduler_runner.h` with PMCS-specific runtime
configuration:

```cpp
enum class SchedulerMode : uint8_t {
  kOfflineReplay = 0,
  kPmcs = 1,
};

typedef struct scheduler_runner_config_t {
  const char *graph_json_path;
  const char *driver_name;
  int graph_iters;
  int dispatch_iters;
  int report_every;
  const char *vmfb_root_dir;
  const char *cpu_p_cpu_ids;
  const char *cpu_e_cpu_ids;
  int visible_cores;
  const char *out_json_path;
  const char *out_dot_path;
  const char *trace_csv_path;
  const char *target_platform;
  const char *variant_p_dir;
  const char *variant_e_dir;
  const char *elf_marker;

  const char *scheduler_mode;
  const char *objective_mode;
  const char *predictor_onnx_path;
  const char *pmcs_trace_jsonl_path;
  int slow_loop_period_ms;
  int enable_preemption;
  int enable_thermal_guard;

  double hard_reserve_cpu_p;
  double hard_reserve_cpu_e;
} scheduler_runner_config_t;
```

`kOfflineReplay` remains the baseline. `kPmcs` becomes the default only after
the acceptance criteria in Section 28 pass.

## 18. Runtime State And Per-Target State

Add runtime-only PMCS state in `scheduler_runner.cc`:

```cpp
enum class StageRunState : uint8_t {
  kFuture,
  kReleased,
  kReady,
  kRunning,
  kDone,
  kBlocked,
};

struct RuntimeStageState {
  StageRunState state = StageRunState::kFuture;
  uint64_t released_at_us = 0;
  uint64_t ready_at_us = 0;
  uint64_t started_at_us = 0;
  uint64_t ended_at_us = 0;
  uint64_t last_considered_at_us = 0;
  uint32_t dispatch_count = 0;
  uint32_t starvation_ticks = 0;
  int remaining_hard_preds = 0;
  int remaining_soft_preds = 0;
  std::string assigned_impl;
  std::string assigned_target;
};

struct TargetState {
  std::string target_id;
  uint32_t queue_depth = 0;
  uint32_t inflight = 0;
  double util = 0.0;
  double miss_rate = 0.0;
  double predicted_temp_c = 0.0;
  double temp_limit_c = 0.0;
  double headroom_c = 0.0;
  double available_memory_bytes = 0.0;
  double available_bandwidth_bytes_per_s = 0.0;
  uint64_t next_available_us = 0;
};
```

## 19. XPU-RT Output And Offline Planner Changes

### 19.1 New output contract

XPU-RT should stop emitting only a timestamped replay script. It must emit:

- a stage graph,
- multiple implementations per stage,
- release windows and deadlines,
- chain metadata,
- cost priors covering compute, transfer, and memory behavior,
- one or more policy bundles describing operating regimes.

Required stage record shape:

```json
{
  "stage_id": "dispatch_042",
  "workload_id": "dronet",
  "chain_id": "fg_chain_01",
  "kind": "model_exec",
  "earliest_release_us": 180000,
  "preferred_start_us": 194000,
  "latest_useful_start_us": 230000,
  "absolute_deadline_us": 250000,
  "chain_deadline_us": 250000,
  "chain_slack_budget_us": 15000,
  "criticality": "soft_rt",
  "static_priority": 2,
  "hard_predecessors": ["dispatch_017"],
  "soft_predecessors": ["dispatch_009"],
  "soft_dep_penalty": 0.25,
  "implementations": [
    {
      "impl_id": "dispatch_042_cpu_p",
      "target_id": "CPU_P",
      "vmfb_path": "dronet/cpu_p.vmfb",
      "entry_name": "main",
      "wcet_us": 3300,
      "runtime_q50_us": 1800,
      "runtime_q90_us": 2400,
      "runtime_q99_us": 3200,
      "launch_overhead_us": 30,
      "input_transfer_us": 0,
      "output_transfer_us": 0,
      "migration_penalty_us": 0,
      "scratch_bytes": 8388608,
      "working_set_bytes": 8388608,
      "preemptible": false,
      "checkpointable": false
    },
    {
      "impl_id": "dispatch_042_cpu_e",
      "target_id": "CPU_E",
      "vmfb_path": "dronet/cpu_e.vmfb",
      "entry_name": "main",
      "wcet_us": 5400,
      "runtime_q50_us": 2900,
      "runtime_q90_us": 3900,
      "runtime_q99_us": 5200,
      "launch_overhead_us": 30,
      "input_transfer_us": 0,
      "output_transfer_us": 0,
      "migration_penalty_us": 250,
      "scratch_bytes": 8388608,
      "working_set_bytes": 8388608,
      "preemptible": true,
      "checkpointable": false
    }
  ]
}
```

Required bundle record shape:

```json
{
  "bundle_id": "nominal",
  "preferred_impl_by_stage": {
    "dispatch_042": "dispatch_042_cpu_p"
  },
  "priority_by_stage": {
    "dispatch_042": 2
  },
  "reserve_fraction_by_target": {
    "CPU_P": 0.25,
    "CPU_E": 0.10
  },
  "risk_margin_by_target": {
    "CPU_P": 0.10,
    "CPU_E": 0.15
  }
}
```

Recommended initial bundle set:

- `nominal`
- `burst`
- `thermal`
- `cpu_hot`
- `accelerator_busy`

### 19.2 Offline planner responsibilities

XPU-RT should become a three-stage planner:

1. **Profiler**
   - run each stage implementation in isolation,
   - capture WCET-style timing plus q50, q90, and q99,
   - measure launch, transfer, and memory footprints.
2. **Assignment and priority solver**
   - jointly choose preferred implementation and static priority,
   - honor precedence, per-target capacity, criticality, and deadlines.
3. **Bundle emitter**
   - emit multiple operating-regime bundles instead of one irreversible plan.

This work should start in:

- `/scratch2/agustin/XPU-RT/xpu-rt/workload.py`
- `/scratch2/agustin/XPU-RT/xpu-rt/workload_factory.py`
- `/scratch2/agustin/XPU-RT/xpu-rt/scheduler.py`

### 19.3 Legacy compatibility

During transition, if PMCS mode is enabled and the JSON lacks stage-level PMCS
fields, Merlin may synthesize a conservative policy from legacy fields:

- `preferred_start_us = MsToUs(start_time_ms)`
- `earliest_release_us = 0` for immediate nodes, otherwise `preferred_start_us`
- `latest_useful_start_us = preferred_start_us + max(1000, MsToUs(planned_duration_ms))`
- `absolute_deadline_us = latest_useful_start_us + max(1000, MsToUs(planned_duration_ms))`
- `hard_predecessors = deps`
- `soft_predecessors` gets `time_dependency` when the edge is soft

This path is only for bring-up. Production PMCS evaluation should use upgraded
XPU-RT output.

## 20. Stream-Style Planning Discipline

The offline planner should adopt the key modeling discipline from Stream:

- model communication, not just compute,
- model memory fit and off-chip traffic,
- model repeated steady-state regions separately from startup and drain,
- optionally expand a stage into a finer-grained tile graph when whole-dispatch
  granularity is too coarse.

Use a cost equation shaped like:

`predicted_finish = queue_wait + compute + transfer + launch + sync + migration`

not just "runtime on target."

Optional tile expansion belongs **offline**, not in the runtime hot path.

## 21. Runtime Architecture

### 21.1 Fast path

The fast path remains inside `samples/common/xpu-rt/scheduler_runner.cc` and is
event-driven on:

- stage release,
- predecessor completion,
- target availability,
- async fence or semaphore completion,
- slow-loop parameter refresh.

The fast path owns:

- candidate discovery,
- scoring across alternatives,
- admission,
- queue insertion,
- migration or defer decisions,
- trace emission.

### 21.2 Slow path

The slow path runs every `50-250 ms` and owns:

- active `PolicyBundle` selection,
- objective mode updates,
- reserve and risk-margin adjustments,
- preemption-threshold tuning,
- fairness boosts,
- predictor polling.

It may become learned later, but must start deterministic and rule-based.

### 21.3 Per-target coordinators

PMCS should introduce one coordinator per target or queue:

- `CPU_P`
- `CPU_E`
- future `NPU0`
- future `GPU0`
- future `DMA0`

Each coordinator owns:

- current running stage or stages,
- target-local pending queue ordered by effective priority,
- target-specific stats,
- preempt and restore hooks where supported,
- interface glue to the underlying IREE session or device queue.

This keeps the split clear:

- PMCS global policy decides what to run and where,
- target coordinator owns target-local queueing and execution state,
- IREE performs actual execution once work reaches the target.

## 22. Runtime Modules To Add

Create or extend these modules under `samples/common/xpu-rt/`:

- `runtime_estimator.h/.cc`
- `objective_engine.h/.cc`
- `safety_guard.h/.cc`
- `preemption.h/.cc`
- `predictor_client.h/.cc`
- `scheduler_trace.h/.cc`

The main scheduler stays in `scheduler_runner.h/.cc`.

## 23. Runtime Estimator

`RuntimeEstimator` should track high-side behavior per implementation and
target, not just mean runtime.

Recommended state:

```cpp
struct RuntimeEstimate {
  double ewma_us = 0.0;
  double ewvar_us2 = 0.0;
  P2Quantile q50;
  P2Quantile q90;
  P2Quantile q99;
  double last_update_us = 0.0;
  uint64_t count = 0;
  std::array<double, 8> residual_ring = {};
};
```

Key rules:

- key by `(workload_id, stage_id, impl_id, target_id, shape_signature,
  criticality)`,
- update on every completion,
- blend online estimates with offline priors until enough samples exist,
- use `q99` for hard RT, `q90` for soft RT, `max(q50, ewma)` for best effort,
- keep residual history for the predictor.

## 24. Deterministic Online Scoring

PMCS should replace literal queue replay with score-based admission over stage
alternatives.

At every scheduling decision:

1. build the candidate set of stages whose hard predecessors are satisfied and
   whose `earliest_release_us` has arrived,
2. enumerate every implementation and eligible target coordinator,
3. estimate finish time using queue, compute, transfer, launch, and migration
   cost,
4. reject unsafe combinations through the safety guard,
5. choose the best feasible `(stage, implementation, target)`.

Recommended initial score:

```cpp
score =
  + 8.0 * CriticalityWeight(stage.criticality)
  + 4.0 * PriorityWeight(stage, bundle)
  + 5.0 * DeadlinePressure(stage, est_finish_us)
  + 2.0 * StarvationBonus(stage_state)
  + 2.0 * BundlePreferenceBonus(stage, impl, bundle)
  + 1.5 * PreferredTargetBonus(impl, bundle)
  - 3.0 * QueuePressurePenalty(target_state)
  - 2.0 * InterferencePenalty(target_state, predictor)
  - 2.0 * TransferPenalty(impl)
  - 1.5 * MigrationPenalty(stage_state, impl)
  - 1.0 * EnergyPenalty(impl, objective_mode)
  - 1.0 * ThermalPenalty(target_state, objective_mode)
  - 0.5 * SoftDependencyPenalty(stage_state, stage);
```

Recommended helper behavior:

- hard RT criticality -> `3.0`
- soft RT criticality -> `1.5`
- best-effort criticality -> `0.5`
- deadline pressure should rise sharply once slack becomes negative,
- starvation bonus should be clamped to avoid runaway weight,
- bundle preference should bias but not force implementation choice.

The scorer must remain deterministic with the predictor disabled.

## 25. Safety Guard, Fallback, And Preemption

### 25.1 Safety guard

`SafetyGuard` is mandatory. It should reject any candidate that violates:

- hard deadline feasibility with a guard band,
- reserved capacity for critical work,
- memory availability,
- bandwidth availability,
- thermal headroom,
- safe preemption rules.

Recommended reject surface:

```cpp
enum class SafetyRejectReason {
  kDeadlineMissRisk,
  kHardReserveViolation,
  kThermalViolation,
  kMemoryViolation,
  kBandwidthViolation,
  kUnsafePreemption,
};

struct SafetyDecision {
  bool allowed = false;
  SafetyRejectReason reason = SafetyRejectReason::kDeadlineMissRisk;
  double margin_us = 0.0;
};
```

Default guard bands:

- hard RT: `max(200 us, 0.1 * runtime_q99_us)`
- soft RT: `max(100 us, 0.05 * runtime_q90_us)`

Default critical reserves:

- `CPU_P`: `25%`
- `CPU_E`: `10%`
- accelerator queues: configurable

### 25.2 Deterministic fallback

When PMCS cannot admit safely, fall back to:

- hard RT: EDF inside reserved capacity,
- soft RT: fixed-priority with aging,
- best effort: deficit round-robin by workload or chain id.

### 25.3 Preemption and migration

Preemption must remain bounded and conservative:

- never preempt a running hard RT stage,
- only preempt stages whose implementation declares `preemptible`,
- prefer checkpointable victims,
- migrate only if another eligible implementation is feasible and beneficial.

Recommended threshold policy:

- initial high-util threshold: `0.85`
- initial low-util threshold: `0.65`
- lower threshold when miss rate rises above `5%`
- raise threshold when miss rate stays below `1%`
- clamp to `[0.70, 0.92]`

## 26. Objective Engine And Predictor

### 26.1 Objective engine

The objective engine should keep metrics, weights, and algorithms separate so
Merlin can switch runtime intent without recompiling.

Recommended modes:

- `LatencyFirst`
- `EnergyFirst`
- `Balanced`
- `DeadlineRecovery`
- `FairnessRecovery`
- `ThermalProtect`

Recommended initial weights:

| Mode | `(latency, energy, reliability, thermal, fairness)` |
| --- | --- |
| `LatencyFirst` | `(0.55, 0.10, 0.20, 0.05, 0.10)` |
| `EnergyFirst` | `(0.20, 0.45, 0.15, 0.10, 0.10)` |
| `Balanced` | `(0.30, 0.20, 0.20, 0.15, 0.15)` |
| `DeadlineRecovery` | `(0.60, 0.05, 0.25, 0.05, 0.05)` |
| `FairnessRecovery` | `(0.20, 0.10, 0.15, 0.05, 0.50)` |
| `ThermalProtect` | `(0.20, 0.20, 0.15, 0.35, 0.10)` |

Recommended switching rules:

- if hard miss rate exceeds `1%` -> `DeadlineRecovery`
- else if starvation index exceeds threshold -> `FairnessRecovery`
- else if thermal headroom is low -> `ThermalProtect`
- else if a power cap is active -> `EnergyFirst`
- else -> `Balanced`

### 26.2 Predictor sidecar

The predictor is a slow-loop hint source, not an action generator in v1.

Create the offline and runtime bridge in:

- `tools/scheduler_train/build_dataset.py`
- `tools/scheduler_train/model.py`
- `tools/scheduler_train/train_predictor.py`
- `tools/scheduler_train/eval_predictor.py`
- `tools/scheduler_train/export_onnx.py`
- `samples/common/xpu-rt/predictor_client.h/.cc`

Recommended inputs:

- arrivals, releases, starts, completions,
- runtime residuals,
- miss rate by workload,
- queue depth and inflight count per target,
- utilization per target,
- transfer volume,
- migration and preemption counts.

Recommended outputs:

- `contention_multiplier[target]`
- `risk_quantile[target]`
- `expected_arrivals_next_window[workload]`
- `preemption_bias[target]`

The scheduler must remain fully functional when all predictor outputs are
missing.

## 27. Hardware And Workload Plan

### Phase 1: `spacemit_x60`

This remains the first deployment target because Merlin already has:

- pinned `CPU_P` and `CPU_E` executors,
- the dispatch-scheduler sample,
- runtime traces and plots,
- XPU-RT schedule integration.

Primary workload:

- `dronet + mlp`

This is the best correctness, fairness, and mixed-policy bring-up case.

### Phase 2: `npu_ucb`

This is the first CPU-plus-accelerator extension target.

Primary workload:

- `smolVLA.q.fp8`

This is the best heavier workload for estimator calibration, contention, and
alternative-implementation selection.

### Later targets

Keep these as later extensions:

- `gemmini_mx`
- `saturn_opu`

## 28. Testing And Evaluation

### 28.1 Unit tests

Add coverage for:

- stage-graph parsing,
- legacy graph lifting,
- bundle parsing and selection,
- runtime-estimator prior blending and quantiles,
- score determinism over multiple implementations,
- safety rejection reasons,
- preemption victim selection,
- objective-mode switching.

### 28.2 Integration tests

Run first on `spacemit_x60`:

- nominal `dronet + mlp`,
- burst arrivals,
- injected slowdown on `CPU_P` or `CPU_E`,
- mixed-criticality overload,
- anti-starvation scenarios.

Then run on `npu_ucb` with `smolVLA.q.fp8`:

- preferred-implementation selection,
- fallback when the preferred target slows down,
- transfer-cost sensitivity,
- bundle switching under accelerator contention.

### 28.3 Metrics

Track at least:

- p50, p95, p99 latency,
- miss rate by criticality,
- throughput,
- utilization by target,
- idle-hole reduction versus replay,
- migration count,
- preemption count,
- starvation time or fairness index,
- hot-path scheduler overhead,
- slow-loop predictor overhead.

### 28.4 Acceptance criteria

PMCS v2 is ready to replace replay as the default only when:

- hard-critical chains do not regress on miss rate,
- fairness protection prevents starvation in stress tests,
- utilization or idle-hole behavior improves under contention,
- scheduler overhead stays within the accepted runtime budget,
- disabling the predictor does not break correctness.

## 29. Main Implementation Worklist

The main implementation work should land in:

- `samples/common/dispatch/dispatch_types.h`
- `samples/common/dispatch/dispatch_graph.h/.cc`
- `samples/common/xpu-rt/scheduler_runner.h/.cc`
- new PMCS helper modules under `samples/common/xpu-rt/`
- `tools/scheduler_train/`
- `/scratch2/agustin/XPU-RT/xpu-rt/`

The implementation order should be:

1. stage-graph schema and legacy lifting,
2. XPU-RT multi-implementation output,
3. policy bundles,
4. runtime estimator,
5. online scorer,
6. safety shield,
7. objective engine,
8. predictor sidecar,
9. optional fine-grained offline tile planner,
10. optional learned slow-loop controller.

## 30. Main Conclusion

The important architectural conclusion is:

**Merlin should not keep treating scheduling as replay of pre-bound dispatch
nodes. It should move to stage graphs with multiple execution alternatives,
offline policy bundles, and runtime-adaptive but safety-bounded selection above
IREE.**

That is the strongest common lesson from:

- the current `dronet + mlp` scheduler tuning work,
- IREE's actual task and async runtime surfaces,
- XAUTO's stage-level scheduling model,
- Stream's communication-aware and memory-aware planning discipline,
- the mixed-criticality PMCS direction.
