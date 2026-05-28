# Build configurations

The merlin build wrapper (`./merlin build`) supports several configurations
beyond the default release build. This page summarizes when to reach for
each one and the matching `./merlin build` invocation.

> Per `AGENTS.md` Golden Rule #1, never invoke `cmake` / `ninja` directly.
> All builds go through `./merlin build`. The CMake flag composition lives
> in `tools/build/presets.py` and `tools/build/cmake.py`.

## Standard release

Best for: fast iteration, deployment, regression tests.

- Speed: high (optimized)
- Assertions: enabled
- Tracing: off

```bash
./merlin build --profile vanilla
# or for the plugin-enabled compiler:
./merlin build --profile full-plugin
```

Outputs land in `build/<target>-<variant>-release/` (see
`docs/how_to/use_build_py.md`).

## Release + Tracy profiling

Best for: profiling host-side compilation time or runtime hot paths with
Tracy.

```bash
./merlin build --profile vanilla --config release --enable-tracy
```

See [`build_tracy_ubuntu.md`](build_tracy_ubuntu.md) for the full Tracy
profiling workflow (compile-side `--tracy` flag + viewer setup).

## Debug

Best for: stepping with `gdb` / `lldb`, asserts-heavy testing.

```bash
./merlin build --profile vanilla --config debug
```

> Disk usage warning: debug builds can balloon to 150 GB+. See AGENTS.md
> Golden Rule #2 (default release config).

## AddressSanitizer (ASan)

Best for: memory-bug investigation.

```bash
./merlin build --profile vanilla --config asan
```

## Cross-compile profiles

Targets a specific board's toolchain:

```bash
./merlin build --profile spacemit       # SpacemiT X60 RISC-V
./merlin build --profile firesim        # FireSim bare-metal
./merlin build --profile gemmini        # Gemmini accelerator
./merlin build --profile zephyr         # Zephyr-on-IREE
./merlin build --profile qrb5165        # Qualcomm QRB5165
```

See `tools/build/presets.py:PROFILE_PRESETS` for the canonical list and
[`use_build_py.md`](how_to/use_build_py.md) for end-to-end examples.

## Common modifiers

| Flag | Effect |
|---|---|
| `--config {release,debug,asan,perf,trace}` | Build type (default: release). |
| `--cmake-target <name>` | Build only the named CMake target. |
| `--enable-tracy` | Add Tracy instrumentation (runtime + compiler). |
| `--compiler-scope {all,gemmini,npu,saturn,spacemit,none}` | Limit which compiler plugins are wired into `iree-compile`. |
| `--package` | Emit a packaged install tree under `build/<...>/install/`. |

`./merlin build --help` is the full reference.
