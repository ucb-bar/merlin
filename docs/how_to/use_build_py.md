# Use `tools/build.py` Effectively

This page is a practical guide for common build workflows.

## 1) Preferred Entry

Use:

```bash
conda run -n merlin-dev python tools/build.py --profile <profile>
```

Available profiles include:

- `vanilla`
- `full-plugin`
- `radiance`
- `gemmini`
- `npu`
- `spacemit`
- `firesim`

## 2) Build Directory Naming

`tools/build.py` uses:

- `build/<target>-<variant>-<config>/`

where:

- `target`: `host`, `spacemit`, `firesim`
- `variant`: `vanilla` or `merlin` (depends on plugin enablement)
- `config`: `debug`, `release`, `asan`, `trace`, `perf`

Examples:

- `build/host-merlin-release`
- `build/spacemit-merlin-perf`

## 3) Common Commands

Host compiler with NPU plugin scope:

```bash
conda run -n merlin-dev python tools/build.py --profile npu --config release
```

Host runtime Radiance smoke target:

```bash
conda run -n merlin-dev python tools/build.py \
  --profile radiance \
  --cmake-target iree_hal_drivers_radiance_testing_transport_smoke_test
```

Cross-target sample build:

```bash
conda run -n merlin-dev python tools/build.py \
  --profile spacemit \
  --config perf \
  --cmake-target merlin_baseline_dual_model_async_run
```

## 4) Where Outputs Go

Common output locations:

- compiler tools:
  - `build/<...>/install/bin/iree-compile`
  - `build/<...>/install/bin/iree-opt`
- runtime sample binaries:
  - `build/<...>/runtime/plugins/merlin-samples/...`
- radiance driver tests:
  - `build/<...>/runtime/plugins/merlin/runtime/iree/hal/drivers/radiance/testing/...`

## 5) Useful Flags Beyond Profiles

- `--compiler-scope {all,gemmini,npu,saturn,spacemit,none}`
- `--plugin-compiler` / `--no-plugin-compiler`
- `--plugin-runtime` / `--no-plugin-runtime`
- `--plugin-runtime-radiance*` toggles
- `--build-compiler`, `--build-tests`, `--build-python-bindings`, etc.
- `--cmake-target <target>`

## 6) Verify Build Result Quickly

Check compiler plugin load:

```bash
build/host-merlin-release/install/bin/iree-compile --iree-list-plugins
```

Find a sample binary:

```bash
find build -type f -name '*baseline-dual-model-async-run*'
```

Find Radiance smoke test binary:

```bash
find build -type f -name 'transport_smoke_test'
```
