# Add A Sample Application

This guide shows how to add new runtime/sample apps and how to locate built
binaries.

## 1) Choose Placement

Use one of:

- `samples/common/<app_name>/` for target-agnostic samples
- `samples/<Platform>/<app_name>/` for platform-specific samples

Current platform examples:

- `samples/SpacemiTX60/baseline_dual_model_async`
- `samples/SaturnOPU/custom_dispatch_ukernels`

## 2) Add CMake For Your App

Create `CMakeLists.txt` in your app dir and define executable target(s).

Reference:

- `samples/SpacemiTX60/baseline_dual_model_async/CMakeLists.txt`

Key points:

- define a clear target name
- set output name if you want a stable binary filename
- link against required IREE runtime libs

## 3) Ensure Directory Is Included

Top-level sample inclusion is controlled by:

- `samples/CMakeLists.txt`
- platform subdirs, for example `samples/SpacemiTX60/CMakeLists.txt`

Runtime-plugin integration comes from:

- `iree_runtime_plugin.cmake` with `MERLIN_RUNTIME_ENABLE_SAMPLES`

## 4) Build A Specific Sample

For SpacemiTX60 async baseline sample:

```bash
conda run -n merlin-dev python tools/build.py \
  --profile spacemit \
  --config perf \
  --cmake-target merlin_baseline_dual_model_async_run
```

## 5) Locate Output Binaries

Paths follow:

- `build/<target>-<variant>-<config>/...`

For the async baseline sample, typical outputs are:

- runtime sample binary:
  - `build/spacemit-merlin-perf/runtime/plugins/merlin-samples/SpacemiTX60/baseline_dual_model_async/baseline-dual-model-async-run`
- benchmark wrapper binary (if generated in your build):
  - `build/spacemit-merlin-perf/runtime/plugins/merlin-samples/SpacemiTX60/b_baseline_dual_model_async/benchmark-baseline-dual-model-async-run`

Use `find` when in doubt:

```bash
find build -type f -name '*baseline-dual-model-async-run*'
```

## 6) Async Sample Design Pattern

For asynchronous deployment style, structure app code by concerns:

- sensor/input producer(s)
- scheduling policy/runtime orchestration
- model invocation + result handling

Reference split:

- `main.c`
- `runtime_scheduler.cc/.h`
- `sensor_generator.cc/.h`

from `samples/SpacemiTX60/baseline_dual_model_async`.

## 7) Where To Look In IREE For More Sample Patterns

Useful upstream sample references in this repo checkout:

- `third_party/iree_bar/samples/custom_module/async`
- `third_party/iree_bar/samples/simple_embedding`
- `third_party/iree_bar/samples/multiple_modules`
- `third_party/iree_bar/samples/promise_devices_layer`

These help for runtime wiring, async flows, and multi-module usage patterns.
