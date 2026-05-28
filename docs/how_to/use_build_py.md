# Use `./merlin build` effectively

A practical guide for common build workflows. (Previously titled "Use
`tools/build.py`" — the entry point has been consolidated; see
`tools/build/cli.py`.)

> Per `AGENTS.md`, never invoke `cmake` / `ninja` directly. Always go
> through `./merlin build`.

## 1. Preferred entry

```bash
./merlin build --profile <profile>
```

Available profiles (canonical list lives in
`tools/build/presets.py:PROFILE_PRESETS`):

- `vanilla` — host compiler tools only
- `full-plugin` — host compiler + every merlin plugin enabled
- `radiance` — host runtime smoke target for the Radiance HAL
- `gemmini` — Gemmini accelerator bare-metal runtime
- `npu` — host compiler scoped to the NPU dialect
- `spacemit` — SpacemiT X60 RISC-V cross-compile
- `firesim` — FireSim bare-metal runtime
- `zephyr` — Zephyr-on-IREE workload
- `qrb5165` — Qualcomm QRB5165 board runtime
- `qnn-compiler` — host compiler with QNN HAL wiring

## 2. Build directory naming

Outputs land at `build/<target>-<variant>-<config>/`, where:

- `target`: `host`, `spacemit`, `firesim`, `gemmini`, `zephyr`, `qrb5165`
- `variant`: `vanilla` or `merlin` (plugins on/off)
- `config`: `release` (default), `debug`, `asan`, `perf`, `trace`

Examples:

- `build/host-merlin-release`
- `build/spacemit-merlin-perf`

A few presets pick non-default suffixes — see
`tools/build/AGENTS.md` for the special cases (e.g. `qnn-compiler` →
`host-merlin-release-qrb`).

## 3. Common commands

Host compiler with NPU plugin scope:

```bash
./merlin build --profile npu --config release
```

Host runtime Radiance smoke target:

```bash
./merlin build --profile radiance \
  --cmake-target iree_hal_drivers_radiance_testing_transport_smoke_test
```

Cross-target sample build:

```bash
./merlin build --profile spacemit --config perf \
  --cmake-target merlin_baseline_dual_model_async_run
```

## 4. Where outputs go

- compiler tools: `build/<...>/install/bin/iree-compile`,
  `iree-opt`, `iree-run-module`
- runtime sample binaries:
  `build/<...>/runtime/plugins/merlin-samples/...`
- Radiance driver tests:
  `build/<...>/runtime/plugins/merlin/runtime/iree/hal/drivers/radiance/testing/...`

## 5. Useful flags beyond profiles

- `--compiler-scope {all,gemmini,npu,saturn,spacemit,none}` — limit which
  plugins are wired into the compiler build.
- `--config {release,debug,asan,perf,trace}` — build type.
- `--cmake-target <name>` — build only the named target.
- `--enable-tracy` — add Tracy instrumentation (compiler + runtime); see
  [`build_tracy_ubuntu.md`](../build_tracy_ubuntu.md).
- `--package` — emit a packaged install tree.

`./merlin build --help` is the canonical reference.

## 6. Freshness check before compile/run

Before any `./merlin compile` / `./merlin run` / `./merlin verify-output`
invocation, the build for the relevant target must be up to date. The MCP
tool `build_check_freshness` (exposed by `tools/mcp_servers/build.py`) hashes
source roots and reports staleness; Claude Code calls it automatically
per `AGENTS.md` Golden Rule #6.

If you're invoking manually, the rule of thumb: any C++ / CMake / preset
change → rebuild the affected profile before testing.
