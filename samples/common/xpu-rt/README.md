# xpu-rt — Runtime Library for XPU-RT

This directory contains the generic, target-agnostic runtime library used by
[XPU-RT](https://github.com/ucb-bar/XPU-RT) and other downstream consumers
(e.g. Zephyr RTOS applications).

## What is XPU-RT?

XPU-RT is an adaptable full-stack E2E compilation and scheduling flow for
efficient mapping of robotic multi-model workloads onto heterogeneous
shared-memory SoCs. It lives in its own repository and consumes merlin as a
git submodule. See the [XPU-RT repo](https://github.com/ucb-bar/XPU-RT) for
the Python scheduler, profiling scripts, and end-to-end workflow.

## What this folder provides

Two dispatch-graph runners built on top of merlin's IREE runtime integration:

| File | Description |
|---|---|
| `baseline_runner.h/cc` | Topo-order runner: sequential or parallel execution with optional core-mask pinning, multi-iteration, per-node statistics. |
| `scheduler_runner.h/cc` | Two-cluster scheduler: CPU_P + CPU_E worker threads, pinned local-task devices, phase-locked release timing, dependency-driven chains, trace CSV / DOT / JSON output. |

Both runners are **target-agnostic**. Hardware-specific parameters (platform
name, ISA variant directories, core layout, ELF marker) are supplied by the
caller through the config struct. See `samples/SpacemiTX60/` for an example.

### Supporting libraries (in sibling directories)

| Directory | Description |
|---|---|
| `core/` | JSON parser, path utils, CLI helpers, running statistics (no IREE dependency) |
| `dispatch/` | Dispatch graph parsing, types, VMFB resolution, topo sort, output writers |
| `runtime/` | IREE module caching, pinned device creation, fatal-state tracking |

## Build targets

When built as part of merlin's IREE plugin infrastructure:

| CMake target | Type | Description |
|---|---|---|
| `xpurt_objs` | OBJECT | Compiled runner objects (for bundling into archives) |
| `xpurt` | STATIC | Static library for in-tree linking (samples, tests) |
| `xpurt_standalone` | STATIC | **Standalone archive**: xpu-rt objects + full IREE runtime in one `.a` (see below) |

## How XPU-RT uses this

XPU-RT includes merlin as a git submodule. After `merlin build --profile spacemit`,
the SpacemiT sample binaries are ready to use:

```
merlin/build/spacemit-merlin-perf/runtime/plugins/merlin-samples/
  merlin-baseline-async
  merlin-dispatch-scheduler
```

XPU-RT's profiling and scheduling scripts use these binaries directly. For
custom tools, XPU-RT can link against `libxpurt_standalone.a` out-of-tree
(see `XPU-RT/runtime/CMakeLists.txt`).

## Zephyr integration

The standalone archive (`libxpurt_standalone.a`) is the integration point for
Zephyr and other RTOS targets. It is fully self-contained: xpu-rt runner
code + the complete IREE runtime, with no build-tree dependencies at link time.

### Workflow

1. Cross-compile merlin with a Zephyr-compatible toolchain profile:
   ```bash
   cd merlin
   uv run tools/merlin.py build --profile <zephyr-profile> --config perf
   ```

2. Locate the standalone archive in the merlin build tree:
   ```
   merlin/build/<profile>/runtime/src/iree/runtime/libxpurt_standalone.a
   ```

3. Link into your Zephyr application:
   ```cmake
   # Option A: Zephyr helper
   zephyr_library_import_from_static(xpurt /path/to/libxpurt_standalone.a)

   # Option B: direct CMake
   target_link_libraries(app PRIVATE
     -Wl,--whole-archive /path/to/libxpurt_standalone.a -Wl,--no-whole-archive
     Threads::Threads m)
   ```

4. Include headers from `merlin/samples/common/`:
   ```c
   #include "xpu-rt/baseline_runner.h"
   #include "xpu-rt/scheduler_runner.h"
   ```

### Writing a target-specific main

Create a `main.c` that populates the config struct with your platform values:

```c
#include "xpu-rt/scheduler_runner.h"

int main(int argc, char **argv) {
    scheduler_runner_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.graph_json_path = argv[1];
    cfg.driver_name     = "local-task";
    cfg.graph_iters     = 10;
    cfg.dispatch_iters  = 1;

    // Your platform-specific values:
    cfg.cpu_p_cpu_ids   = "0,1";
    cfg.cpu_e_cpu_ids   = "2,3";
    cfg.visible_cores   = 4;
    cfg.target_platform = "my_platform";
    cfg.variant_p_dir   = "vector";
    cfg.variant_e_dir   = "scalar";
    cfg.elf_marker      = "_embedded_elf_riscv_64";

    return scheduler_runner_run(&cfg);
}
```

## Adding a new hardware target

1. Create a directory under `samples/<TargetName>/` with a thin `main.c` that
   sets your target defaults in the config struct.
2. Add a `CMakeLists.txt` that links `xpurt`:
   ```cmake
   add_executable(my_runner main.c)
   target_link_libraries(my_runner PRIVATE xpurt Threads::Threads)
   ```
3. No changes needed in this directory or in `common/` -- the runners are
   fully parameterized.
