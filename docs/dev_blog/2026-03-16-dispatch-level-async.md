# 2026-03-17: SpacemiTX60 Dispatch Scheduler Tuning for Periodic MLP + dronet

> **Repro pin:** merlin@[`e18fc562`](https://github.com/ucb-bar/merlin/commit/e18fc562c5c9a9601fc3e34a6d990a0427ddc255) · iree_bar@[`dd293bb513`](https://github.com/ucb-bar/iree_bar/commit/dd293bb513)
> **Status:** Active

## 0. Scope

This devlog covers the ongoing tuning work for the dispatch-level runtime
scheduler used by:

* `samples/SpacemiTX60/b_dispatch_level_model_async/runtime_scheduler.cc`

The target workload is a combined schedule containing:

* periodic `mlp0..mlp16` work on `CPU_E`
* `dronet` work spanning mostly `CPU_P`, with a late `CPU_E` tail

The main purpose of this workstream has been to make runtime behavior match the
JSON schedule more closely, especially for:

* MLP start phase / frequency
* MLP intra-chain behavior
* dronet continuity
* reducing unexplained idle gaps (“bubbles”)

---

## 1. Goal

The scheduler should implement the following intent:

### For MLP

* the **first dispatch** of each `mlpN` should start as close as possible to the
  JSON `start_time`
* after that, the rest of the dispatches for that same MLP should run
  immediately after each other
* we care more about preserving the requested MLP phase than preserving all
  JSON timing edges as hard dependencies

### For dronet

* dronet should remain mostly dependency-driven
* dronet should keep good continuity on `CPU_P`
* dronet should not be distorted by MLP timing rules

---

## 2. Files Changed

## 2.1 Main runtime file

Primary file under active modification:

* `samples/SpacemiTX60/b_dispatch_level_model_async/runtime_scheduler.cc`

This is where nearly all scheduler behavior changes were made.

### Main areas changed inside this file

* graph predecessor expansion
* MLP-vs-dronet timing semantics
* queueing model (`ready_*` vs `future_*`)
* release-time handling
* MLP root seeding logic
* worker dispatch policy
* trace generation for timing analysis

---

## 2.2 Plotting / analysis script

A plotting script was also used and iterated on to visualize planned vs observed
execution. The exact local path may vary depending on your workspace, but the
script shape used in this workstream does the following:

* reads the runtime trace CSV
* renders:

  * `CPU_E observed`
  * `CPU_E planned`
  * `CPU_P observed`
  * `CPU_P planned`
* assigns lanes per overlap region
* colorizes by job (`dronet`, `mlp0..mlp16`)
* supports full and zoomed schedule plots

This script was important for validating whether scheduler changes were actually
improving runtime behavior.

---

## 3. Baseline Runtime Model

The original scheduler design was intentionally lightweight:

* two long-lived worker threads:

  * one for `CPU_P`
  * one for `CPU_E`
* two local-task devices pinned to separate CPU sets
* one cached runtime session per `(target, vmfb_path)`
* host-side dependency scheduling
* synchronous benchmark VMFB exports
* `start_time` treated mostly as a **hint / ordering preference**

This model was good for low overhead, but it turned out to be too weak for
periodic MLP timing.

---

## 4. Problems Seen in Early Runs

From the early plots and trace CSVs, the main issues were:

### 4.1 MLP phase drift

* MLP roots were not starting at their scheduled JSON times
* periodic frequency was not being respected
* later MLPs could drift several milliseconds to the right

### 4.2 MLP shape distortion

* some MLP segments took visibly longer than neighboring equivalent segments
* the MLP chain was not always compact

### 4.3 dronet bubbles

* dronet still showed short idle gaps on `CPU_P`
* some bubbles on `CPU_P` visually aligned with gaps on `CPU_E`
* this suggested a combination of:

  * dependency-frontier stalls
  * worker wakeup overhead
  * queueing effects

### 4.4 Strange ordering artifacts

At different points during tuning, plots showed:

* dronet chunks visually appearing “out of place”
* MLPs bunched incorrectly
* dronet on `CPU_P` looking okay while MLP timing was still clearly wrong

This indicated that the scheduler needed different policies for the two
workload families.

---

## 5. Runtime Changes Made

## 5.1 Added explicit release-time scheduling

A major change was moving away from pure “ready now” scheduling and introducing
explicit release tracking per node.

### Added runtime state

Each node now tracks timing-related execution state, including:

* `planned_start_us`
* `release_us`
* `ready_us`
* `start_us`
* `end_us`

### Added queue split

Per target, the scheduler keeps:

* ready queues
* future queues

So a node can be:

* dependency-satisfied but not yet released
* released and runnable
* running
* done

### Effect

This was the first step that gave the scheduler any real chance of honoring
scheduled time rather than just ordering nodes by priority.

---

## 5.2 Split semantics: MLP vs dronet

The next big change was stopping the assumption that all jobs should be handled
with the same policy.

### Intended policy after this change

#### MLP

* root dispatch should be phase-driven
* subsequent dispatches should follow immediately after predecessor completion

#### dronet

* stays dependency-driven
* schedule edges remain meaningful hard constraints

### Effect

This improved the overall timeline shape:

* MLP chains became tighter
* dronet stayed more natural
* plots became more interpretable

But MLP roots were still often late.

---

## 5.3 Treat MLP `time_dependency` as soft

A very important semantic correction was made in predecessor expansion.

Originally, `time_dependency` was effectively being treated as another hard
predecessor. That works for dronet, but it was hurting MLP timing.

### Updated rule

During predecessor expansion:

* explicit `dependencies` always remain hard
* for MLP:

  * `time_dependency` is **not** added to `all_predecessors`
* for non-MLP:

  * `time_dependency` remains part of the hard predecessor set

### Effect

This removed artificial delays on MLP roots.

It aligned runtime behavior more closely with the actual requirement:

* “start the MLP when the schedule says to start it”

instead of:

* “force the MLP to wait for a schedule-edge predecessor even if that destroys
  the periodic phase”

---

## 5.4 Phase-lock only the first dispatch of each MLP

Another key refinement was deciding that the scheduler should only phase-lock the
**first** dispatch of each MLP, not every dispatch in the chain.

### Helper introduced

* `IsMlpFirstDispatch(const DispatchNode& node)`

Semantics:

* if `node.job_name` looks like `mlp*` and `node.id == 0`, it is treated as the
  phase-locked root

### Release behavior

* `mlpN_dispatch_0`

  * `release_us = planned_start_us`
* later `mlpN_dispatch_k`

  * `release_us = end_us` of the predecessor

### Effect

This improved the MLP shape substantially.

Instead of trying to force every layer to match a micro-schedule, the runtime
now only phase-locks startup and then lets the chain flow naturally.

That better matched the actual requirement.

---

## 6. Trace-Based Findings

The trace CSV made the remaining problems very clear.

Representative rows showed cases like:

* `mlp1_dispatch_0`

  * planned at `2134 us`
  * actually started at `5683 us`
* `mlp5_dispatch_0`

  * planned at `10000 us`
  * actually started at `12196 us`
* `mlp7_dispatch_0`

  * planned at `14000 us`
  * actually started at `16769 us`

But later in the same run:

* `mlp10_dispatch_0`

  * planned at `20000 us`
  * started at `20008 us`

This means the current code is already capable of getting close in some cases.

### Interpretation

The remaining issue is no longer just wrong eligibility logic.

Instead, the runtime is now closer to this situation:

* MLP roots become **eligible** at the right time
* but they do not always **start** at that time
* because the shared `CPU_E` lane is already busy

That is a capacity/isolation problem, not just a dependency problem.

---

## 7. Effects of Specific Changes

## 7.1 Pure priority-hint model

### Behavior

* simplest runtime
* lowest conceptual overhead
* poorest MLP phase fidelity

### Effect

* MLP timing drifted badly
* frequency not respected

---

## 7.2 Release-time + future-queue model

### Behavior

* nodes can be held until a real release time
* more faithful than just sorting ready work

### Effect

* clear improvement over pure priority ordering
* still insufficient for start-time fidelity under shared-lane contention

---

## 7.3 Softening MLP `time_dependency`

### Behavior

* MLP no longer blocked by schedule edges that are not really intended as hard
  runtime dependencies

### Effect

* reduced artificial MLP startup delay
* important semantic correction
* one of the most valuable fixes so far

---

## 7.4 Phase-locking only MLP roots

### Behavior

* MLP startup is schedule-driven
* later layers are dependency-driven within the same chain

### Effect

* improved MLP compactness
* better fit to user intent
* did not fully solve root-start latency under contention

---

## 7.5 Keeping dronet dependency-driven

### Behavior

* dronet preserved as a natural chain / DAG on its own target assignments

### Effect

* dronet generally looked better than when MLP and dronet were forced through a
  single shared policy
* still some short bubbles remain

---

## 8. Why Timing Still Does Not Match the JSON

The most important current limitation is structural:

### There is still only one `CPU_E` execution lane

Right now, all `CPU_E` work still shares:

* one worker thread
* one local-task device
* one ready queue
* one future queue

So even with improved release logic:

* an MLP root can become eligible exactly on time
* but still wait behind other `CPU_E` work
* and therefore miss its desired start time

### This is the key distinction

The code now does a better job of:

* deciding **when a node is allowed to run**

But it still cannot guarantee:

* **when it will actually begin running**

unless that execution resource is reserved.

---

## 9. Understanding the Remaining dronet Bubbles

The dronet bubbles seem to come from multiple causes.

## 9.1 Tiny runtime overheads are visible

Because many dispatches are very short, even small overheads show up in the
plot:

* condition-variable wakeup
* queue handoff
* bookkeeping under lock
* synchronous runtime call overhead

## 9.2 Some gaps are real graph-frontier stalls

Some visually aligned `CPU_P` and `CPU_E` idle regions likely come from true
dependency frontier effects.

This is especially plausible near the late mixed-target dronet tail where:

* dronet has CPU_E work
* that CPU_E work interacts with other graph structure
* both sides may briefly wait on the same frontier

So not every aligned bubble is a bug.

---

## 10. Commands Used During This Workstream

## 10.1 Build command

Typical rebuild command used:

```bash
/usr/bin/cmake --build /scratch2/agustin/merlin/build/spacemit-merlin-release \
  --target merlin_benchmark_dispatch_level_model_async
```

---

## 10.2 Example configure/build invocation context

Typical build flow used the Spacemit cross-compile setup with:

* `local-task`
* pinned CPU partitions
* runtime samples enabled
* compiler off in some runs
* benchmark target build only

The important practical point is that runtime changes were validated quickly by
rebuilding only the sample target rather than the full tree.

---

## 10.3 Plotting command

The plotting script was used to generate:

* full schedule view
* optional zoomed view

Typical usage pattern:

```bash
python plot_cluster_schedule.py \
  --trace-csv /path/to/run_trace.csv \
  --out-dir analysis/plots \
  --zoom-ms 12
```

Adjust the script path to match your local workspace.

---

## 11. Code-Level Problems Encountered

## 11.1 Duplicate helper definition

At one point, `ReadyQueueFor(...)` ended up defined twice, causing a compile
failure.

### Effect

Hard build failure until helper definitions were deduplicated.

---

## 11.2 Incomplete helper refactor

A later iteration referred to:

* `IsMlpNode`
* `IsMlpRootNode`

without defining them.

### Effect

Build failure with undeclared identifier errors.

### Resolution

The simpler helper set that actually exists now is:

* `IsMlpJob(...)`
* `IsMlpFirstDispatch(...)`

That is enough for the current policy.

---

## 11.3 Semantic confusion around `time_dependency`

This was not a compile problem, but it was one of the biggest runtime issues.

The schedule JSON contains `time_dependency`, but that did **not** mean the same
thing for:

* dronet
* MLP

Treating both the same was wrong.

### Resolution

* dronet: keep as hard predecessor
* MLP: treat as soft scheduling hint

---

## 12. Current Code Semantics

With the current version, the intended runtime semantics are:

### MLP

* `mlpN_dispatch_0`

  * seeded into the future queue at `planned_start_us`
* `mlpN_dispatch_1..4`

  * released immediately when their predecessor completes
* MLP `time_dependency`

  * not part of hard predecessor expansion

### dronet

* hard dependencies preserved
* `time_dependency` preserved in predecessor expansion
* release is dependency-driven

This is the correct semantic direction.

---

## 13. What This Version Does *Not* Yet Guarantee

Even with the current changes, the scheduler still does **not** guarantee that:

* every MLP root starts exactly at the JSON time
* dronet has zero bubbles
* the observed plot exactly overlays the planned plot

The current scheduler improves **eligibility semantics**, but it does not yet
solve **resource contention** on `CPU_E`.

That is why behavior improved, but still did not become perfect.

---

## 14. Recommended Next Step

The most promising next step is to split `CPU_E` into separate execution lanes.

### Proposed lane split

* `CPU_P`
* `CPU_E_MLP`
* `CPU_E_OTHER`

### Implementation sketch

This would require:

* separate worker threads
* separate local-task devices
* separate ready/future queues
* cache keys that include lane identity

### Expected effect

This should allow:

* MLP roots to start much closer to the requested schedule phase
* reduced queueing delays for MLP
* clearer separation between periodic and dependency-driven work

This is the missing architectural piece.

---

## 15. Known Regressions / Open Questions

## 15.1 Open question: how strict should MLP timing be?

We now know that “release on time” is not enough.

Open question:

* do we want:

  * best-effort start-time alignment?
  * or strict dedicated-lane phase alignment?

The answer likely determines whether the lane split is optional or required.

---

## 15.2 Open question: should dronet get its own sub-policy too?

Right now dronet is just “dependency-driven.” That is likely correct, but there
may still be value in:

* more aggressive continuation bias
* reducing small bubbles between consecutive dronet nodes on the same target

---

## 15.3 Open question: should MLP roots preempt other CPU_E work?

If exact start phase matters enough, another possible future direction is:

* root-dispatch priority boosting
* or reserved MLP capacity

This would be a stronger scheduler policy than the current FIFO/ordered queue
selection.

---

## 15.4 Known limitation: one worker per target is still too coarse

This remains the central limitation of the current design.

---

## 16. How the IREE Local-Task Runtime Actually Works (2026-03-17)

Understanding the IREE local-task runtime internals was essential to diagnosing
the scheduling problems. This section documents the key findings.

### Architecture: driver -> executor -> device -> session

The IREE local-task backend has this layered structure:

```
iree_runtime_instance_t
  └─ driver registry
       └─ iree_hal_task_driver_t (created by driver_module.c factory)
            ├─ iree_task_executor_t[]  (worker thread pools)
            ├─ iree_hal_executable_loader_t[]  (ELF/VMVX loaders)
            └─ iree_hal_allocator_t  (heap allocator)
                 └─ iree_hal_device_t  (created by driver)
                      └─ iree_runtime_session_t  (VM context + HAL module)
                           └─ bytecode modules (VMFBs)
```

Key IREE source files:
- `third_party/iree_bar/runtime/src/iree/hal/drivers/local_task/registration/driver_module.c` — factory that creates the driver
- `third_party/iree_bar/runtime/src/iree/hal/drivers/local_task/task_driver.c` — driver implementation
- `third_party/iree_bar/runtime/src/iree/hal/drivers/local_task/task_device.c` — device implementation
- `third_party/iree_bar/runtime/src/iree/task/executor.h` — task executor (worker thread pool)
- `third_party/iree_bar/runtime/src/iree/task/topology.h` — CPU topology configuration

### Critical finding 1: `create_device_by_path` ignores params

The local-task driver's `create_device_by_path` in `task_driver.c:157-169`
delegates directly to `create_device_by_id`, **ignoring the params array**.
Passing `task_topology_cpu_ids` as a device parameter has no effect:

```c
// task_driver.c:157 — params are NOT consulted
static iree_status_t iree_hal_task_driver_create_device_by_path(
    iree_hal_driver_t* base_driver, iree_string_view_t driver_name,
    iree_string_view_t device_path, iree_host_size_t param_count,
    const iree_string_pair_t* params, ...) {
  // Ignores param_count and params entirely:
  return iree_hal_task_driver_create_device_by_id(
      base_driver, IREE_HAL_DEVICE_ID_DEFAULT, param_count, params, ...);
}
```

The topology is set at **driver creation time** via `iree_task_executor_t`,
not at device creation time.

### Critical finding 2: device identifier must match VMFB target

Compiled VMFBs encode their required device target:
```
#hal.device.target<"local", [#hal.executable.target<"llvm-cpu", ...>]>
```

The `"local"` string is matched against the device identifier via
`iree_string_view_match_pattern` in `task_device.c:234`. If you create a
driver with identifier `"local-task"`, the device reports `"local-task"`,
which does NOT match `"local"`. This causes VMFB loading to fail with:
```
INCOMPATIBLE; HAL device `__device_0` not found or unavailable
```

**Fix**: pass `"local"` (not `"local-task"`) as the identifier to
`iree_hal_task_driver_create`.

### The correct way to create pinned local-task devices

To get truly separate, pinned devices, bypass the driver registry and create
each driver+device pair directly:

```cpp
// From runtime_scheduler.cc — CreatePinnedLocalTaskDevice()

// 1. Build topology from "0,1,2,3" style CPU IDs
iree_task_topology_t topology;
iree_task_topology_initialize_from_logical_cpu_set_string(
    iree_make_cstring_view(cpu_ids_csv), &topology);

// 2. Create dedicated task executor (worker thread pool)
iree_task_executor_options_t opts;
iree_task_executor_options_initialize(&opts);
opts.worker_local_memory_size = 64 * 1024;
iree_task_executor_create(opts, &topology, alloc, &executor);

// 3. Create executable loaders
iree_hal_create_all_available_executable_loaders(
    NULL, 8, loaders, &loader_count, alloc);

// 4. Create heap allocator
iree_hal_allocator_create_heap("local", alloc, alloc, &device_allocator);

// 5. Create driver (identifier MUST be "local" to match VMFBs)
iree_hal_task_driver_create(
    iree_make_cstring_view("local"), &params,
    1, &executor, loader_count, loaders, device_allocator,
    alloc, &driver);

// 6. Create device from driver
iree_hal_driver_create_device_by_id(
    driver, IREE_HAL_DEVICE_ID_DEFAULT, 0, NULL, alloc, &device);
```

This is the same pattern used internally by IREE's own `driver_module.c`
factory, except we control the topology per device.

---

## 17. Bug Fix 1: Device Pinning (2026-03-17)

### Problem

The old `CreateConfiguredLocalTaskDeviceFromCpuIds` passed CPU IDs as params
to `iree_hal_driver_create_device_by_path`. As described above, these params
are ignored. Both CPU_P and CPU_E devices shared the same default executor
(worker thread pool). No core isolation existed.

### Evidence

MLP dispatches (expected ~80us) took 3-5ms during heavy dronet convolutions:

| Dispatch | Expected | Actual | Slowdown |
|----------|----------|--------|----------|
| mlp1_dispatch_2 | 84us | 3313us | 39x |
| mlp6_dispatch_1 | 78us | 2251us | 29x |
| mlp9_dispatch_2 | 84us | 4837us | 58x |

### Fix

Replaced with `CreatePinnedLocalTaskDevice` (see section 16 for full code).
Each device now gets its own `iree_task_executor_t` with workers pinned to
dedicated cores via `iree_task_topology_initialize_from_logical_cpu_set_string`.

### Result

MLP dispatch times returned to ~50-100us regardless of concurrent dronet work.
Individual dispatches no longer contend with dronet for worker threads.

---

## 18. Bug Fix 2: Condvar Timer Overshoot (2026-03-17)

### Problem

After fixing device pinning, MLP roots still started ~2ms late. The CPU_E
worker was idle between MLP chains (no dispatches in the gap), but
`std::condition_variable::wait_until` overshot by ~2000us consistently:

```
mlp1: ready=2134us, start=2755us  (delay=621us)
mlp2: ready=4134us, start=6149us  (delay=2015us)  <-- 2ms overshoot
mlp4: ready=8000us, start=10320us (delay=2320us)  <-- 2ms overshoot
```

Root cause: the Linux futex-based condvar on the SpacemiT X60 RISC-V kernel
(6.1.15) has coarse timer resolution (~2ms). The `wait_until` sleeps for
too long, missing the target wakeup time.

### Fix

In `WorkerMain` (`runtime_scheduler.cc`), replaced the condvar `wait_until`
with a hybrid approach:

```cpp
if (next_release_us > now + 5000) {
    // Long wait (>5ms): condvar sleep, but wake 2ms early
    sched->cv.wait_until(lock, iter_t0 + us(next_release_us - 2000));
} else {
    // Short wait (<=5ms): drop lock, spin-yield, re-acquire
    lock.unlock();
    while (UsSince(iter_t0, Clock::now()) < next_release_us) {
        sched_yield();
    }
    lock.lock();
}
```

The CPU_E worker now spins with `sched_yield()` for short waits, giving
microsecond-accurate wakeup. For long waits (>5ms), it still uses the condvar
but wakes 2ms early to spin the remainder.

### Result

MLP root dispatch timing is now within 1-4 microseconds of planned:

| MLP | Planned (us) | Actual start (us) | Delay |
|-----|-------------|-------------------|-------|
| mlp1 | 2134 | 2138 | **4us** |
| mlp2 | 4134 | 4136 | **2us** |
| mlp6 | 12000 | 12002 | **2us** |
| mlp10 | 20000 | 20001 | **1us** |
| mlp16 | 32000 | 32001 | **1us** |

---

## 19. Code Refactor: Shared Library Extracted (2026-03-17, updated 2026-03-18)

Duplicated code between the two samples was extracted into a layered shared
library under `samples/common/`. The library is split into three layers by
dependency:

#### `samples/common/core/` — Generic utilities (no IREE dependency)

| Header | Contents |
|--------|----------|
| `stats.h` | Log2Histogram, RunningStats |
| `json_parser.h` | JsonParser, ParseDependenciesArray |
| `path_utils.h` | PathDirname, PathJoin2, FileReadable, StartsWith, EndsWith |
| `cli_utils.h` | parse_int_or_default, get_flag_value (C-compatible) |

#### `samples/common/runtime/` — IREE runtime utilities

| Header | Contents |
|--------|----------|
| `fatal_state.h` | SharedState, HasFatal, SetFatalOnce |
| `iree_module_utils.h` | PickEntryFunction |
| `module_cache.h` | CachedModule, LoadModule, CallModule, CallModuleUnlocked |
| `pinned_device.h` | CreatePinnedLocalTaskDevice (correct driver-level approach) |

#### `samples/common/dispatch/` — Dispatch scheduling library

| Header | Contents |
|--------|----------|
| `dispatch_types.h` | HardwareTarget, DispatchNode, GraphModel, ReleasePolicy, TimeDependencyMode, InferSchedulingPolicies |
| `dispatch_graph.h` | JSON parsing, ExpandAllPredecessors, TopoSort |
| `vmfb_resolve.h` | VMFB path resolution chain (configurable target platform) |
| `dispatch_output.h` | TraceWriter, WriteDotGraph, WriteNodesJson, JSON/DOT helpers |

#### Scheduling policies are now data-driven

The scheduler no longer checks network names (`IsMlpJob`, `IsMlpFirstDispatch`).
Instead, each dispatch node carries two policy fields:

* `release_policy`: `"immediate"` (default) or `"phase_locked"` — controls
  whether a root dispatch waits until its planned start time
* `time_dep_mode`: `"hard"` (default) or `"soft"` — controls whether
  `time_dependency` is a real predecessor or just a hint

These can be set explicitly in the schedule JSON. When absent,
`InferSchedulingPolicies()` applies backwards-compatible heuristics based on
`job_name`, preserving the original MLP-vs-dronet behavior.

The old samples (`baseline_dual_model_async`, `simple_dual_model_async`,
`simple_2_model_sync`, `simple_2_model_async`, `dispatch_level_model_async`,
`v2_2_model_async`) were removed. Research prototypes were moved to
`samples/research/`.

---

## 20. Reproducibility

### File layout

```
samples/common/
  core/                        Generic utilities (no IREE dependency)
    cli_utils.h                CLI parsing helpers
    json_parser.h              Minimal JSON parser
    path_utils.h               Path manipulation
    stats.h                    Log2Histogram, RunningStats
  runtime/                     IREE runtime utilities
    fatal_state.h              Atomic fatal state tracking
    iree_module_utils.h        VMFB export introspection
    module_cache.h             Session caching and dispatch calling
    pinned_device.h            Pinned local-task device creation
  dispatch/                    Dispatch scheduling library
    dispatch_types.h           Shared types and scheduling policies
    dispatch_graph.h           JSON parsing, topo sort, predecessor expansion
    dispatch_output.h          Trace CSV, DOT graph, JSON output helpers
    vmfb_resolve.h             VMFB path resolution

samples/SpacemiTX60/dispatch_scheduler/
  main.c                       Entry point (parses IREE flags + app flags)
  runtime_scheduler.h          C config struct
  runtime_scheduler.cc         Scheduler: workers, main loop, summary JSON
  CMakeLists.txt               Build target
  analysis/
    run_on_board.sh            End-to-end: build, deploy, run, plot
    plot_dispatch_trace.py     Matplotlib visualization
    reference_trace.csv        Known-good trace (2026-03-17)
    reference_plot.png         Known-good plot
    scheduled_networks_periodic_profile_profiled.json
```

### Build and run

```bash
# From repo root:
bash samples/SpacemiTX60/dispatch_scheduler/analysis/run_on_board.sh

# Or step by step:
conda run -n merlin-dev uv run tools/merlin.py build \
  --profile spacemit \
  --cmake-target merlin_dispatch_scheduler

scp build/spacemit-merlin-release/.../merlin-dispatch-scheduler \
  root@10.44.86.251:/home/baseline/

ssh root@10.44.86.251 '/home/baseline/merlin-dispatch-scheduler \
  /home/baseline/scheduled_networks_periodic_profile_profiled.json \
  local-task 1 1 1 \
  --vmfb_dir=/home/baseline/dispatches \
  --cpu_p_cpu_ids=0,1,2,3 --cpu_e_cpu_ids=4,5 --visible_cores=8 \
  --trace_csv=/home/baseline/run/out/run_trace.csv \
  --out_json=/home/baseline/run/out/run_summary.json \
  --out_dot=/home/baseline/run/out/run_graph.dot'
```

### Board layout (k1 at 10.44.86.251)

```
/home/baseline/
  merlin-dispatch-scheduler   Binary
  scheduled_networks_periodic_profile_profiled.json
  dispatches/gen/vmfb/
    dronet/spacemit_x60/{RVV,scalar}/dronet.q.int8/benchmarks/vmfb/*.vmfb
    mlp/spacemit_x60/{RVV,scalar}/mlp.q.int8/benchmarks/vmfb/*.vmfb
  run/out/
    run_trace.csv      Trace output
    run_summary.json   Summary output
    run_graph.dot      DOT output
```

---

## 21. Bug Fix 3: Schedule JSON Module Name Mismatch (2026-03-17)

### Problem

Dronet dispatches 0-4 in the schedule JSON had `module_name` starting with
`mlp$async_dispatch_*` instead of `dronet$async_dispatch_*`. Since dronet's
first layers are MLP-shaped operations (matmuls, elementwise), the schedule
generator incorrectly tagged them with MLP module names.

The VMFB resolver used `module_name` to find benchmark VMFBs, so dronet's
first 5 dispatches resolved to the **MLP network's** tiny benchmark VMFBs
instead of **dronet's** actual first-layer VMFBs.

### Evidence

| Dispatch | module_name (wrong) | Planned dur | Actual run |
|----------|-------------------|-------------|------------|
| dronet_dispatch_1 | `mlp$async_dispatch_1...` | 6210us | 197us (31x fast) |
| dronet_dispatch_4 | `mlp$async_dispatch_4...` | 3990us | 83us (48x fast) |

### Fix

Replaced `mlp$` with `dronet$` in the schedule JSON for `dronet_dispatch_0`
through `dronet_dispatch_4`. After the fix:

| Dispatch | module_name (correct) | Planned dur | Actual run |
|----------|---------------------|-------------|------------|
| dronet_dispatch_1 | `dronet$async_dispatch_1...` | 6210us | 6754us (1.1x) |
| dronet_dispatch_4 | `dronet$async_dispatch_4...` | 3990us | 4042us (1.0x) |

---

## 22. Tracy Profiling Workflow (2026-03-17)

### Tooling additions

Two flags were added to the build/compile scripts for Tracy support:

**`tools/build.py --enable-tracy`** overlays runtime tracing onto any config:

* `IREE_ENABLE_RUNTIME_TRACING=ON`
* `IREE_TRACING_MODE=1` (instrumentation only; higher modes crash on RISC-V
  due to pointer compression and callstack issues in Tracy's server code)
* `TRACY_NO_POINTER_COMPRESSION=ON` for RISC-V targets

**`tools/compile.py --tracy`** adds debug info to compiled VMFBs:

* `--iree-hal-executable-debug-level=3`
* `--iree-llvmcpu-link-embedded=false`
* `--iree-llvmcpu-debug-symbols=true`

### Build commands

```bash
# Compile models with Tracy debug info
conda run -n merlin-dev uv run tools/compile.py \
  models/dronet/dronet.q.int8.mlir \
  --target spacemit_x60 --hw RVV --quantized --tracy

conda run -n merlin-dev uv run tools/compile.py \
  models/mlp/mlp.q.int8.mlir \
  --target spacemit_x60 --hw RVV --quantized --tracy

# Build runtime with Tracy instrumentation
conda run -n merlin-dev uv run tools/merlin.py build \
  --profile spacemit --config release --enable-tracy \
  --cmake-target merlin_baseline_async

# Build iree-tracy-capture for on-board trace collection
conda run -n merlin-dev uv run tools/merlin.py build \
  --profile spacemit --config release --enable-tracy \
  --cmake-arg=-DIREE_BUILD_TRACY=ON \
  --cmake-target iree-tracy-capture
```

### On-board trace capture

```bash
# On the board (k1):
# 1. Start the binary (waits for Tracy connection)
TRACY_NO_EXIT=1 ./merlin-baseline-async \
  dronet.q.int8_dispatch_graph.json local-task 3 1 1 --parallelism=6 &
sleep 1

# 2. Capture the trace
./iree-tracy-capture -f -o trace_baseline.tracy

# 3. Copy trace back to host and open with tracy-profiler
```

### RISC-V Tracy limitations

* `IREE_TRACING_MODE >= 2` crashes due to `PackPointer` assertion
  (`TracyWorker.cpp:3931`) — RISC-V 64-bit virtual address ranges are
  incompatible with Tracy's pointer compression
* `IREE_TRACING_MODE >= 3` crashes due to `ProcessZoneBeginCallstack`
  assertion — callstack collection is broken on RISC-V
* Mode 1 (instrumentation + log messages only) works reliably

---

## 23. Summary

### Three bugs caused dispatch timing to fail

1. **Device pinning was silently broken.** `iree_hal_driver_create_device_by_path`
   ignores the params argument in the local-task driver. Both devices shared the
   same default worker pool. Fixed by creating drivers directly via
   `iree_hal_task_driver_create` with dedicated `iree_task_executor_t` per device.
   Device identifier must be `"local"` (not `"local-task"`) to match VMFB targets.

2. **Condvar `wait_until` has ~2ms overshoot on RISC-V.** The Linux futex timer
   on SpacemiT X60 (kernel 6.1.15) has coarse resolution. Fixed by spin-waiting
   with `sched_yield()` for short sleeps (<=5ms).

3. **Schedule JSON had wrong module names for dronet dispatches 0-4.** The
   schedule generator tagged dronet's MLP-shaped first layers with `mlp$` module
   names, causing the VMFB resolver to load the wrong (standalone MLP) VMFBs.
   Fixed by correcting to `dronet$async_dispatch_*` in the JSON.

### Before vs after

| Metric | Before | After |
|--------|--------|-------|
| MLP root start delay | 2000-6000us | 1-4us |
| MLP dispatch runtime during dronet conv | 3000-5000us | 50-100us |
| Dronet dispatch 0-4 runtime | 83-197us (wrong VMFBs) | Matches planned (correct VMFBs) |
| Total wall time (1 graph iter) | ~35ms | ~35ms (matches schedule) |
| Schedule fidelity | Poor | Excellent |

### What remains

* Re-profile dronet dispatch 0-4 durations in the schedule JSON now that
  the correct VMFBs are being used (current profiled durations may be stale)
* Multi-iteration runs for statistical confidence
* Tracy-based analysis comparing baseline (single device, all cores) vs
  dispatch-level (pinned devices, core partitioning) approaches

---

*Dev-blog written by:* Agustin Coppari Hollmann

*Project Members:* Kris Dong, Dima Nikiforov and Agustin N. Coppari Hollmann
