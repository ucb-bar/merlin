# CLI Reference

This page is generated from the real argparse parsers behind `./merlin <subcommand>`.

Each command is shown with argument introspection and raw `--help` output.

## `./merlin`

Unified Merlin developer command dispatcher.

### Usage

```text
usage: ./merlin [-h]
                {build,compile,quantize,verify-output,perf-decompose,coverage-check,setup,ci,patches,benchmark,chipyard,ray,targetgen,spike,sim,run,mcp}
                ...
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `benchmark`

```text
usage: ./merlin benchmark [-h] [--dry-run]
                          target {compile-dual-vmfb,run-dual-remote} ...
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--dry-run` | no | `False` | - | Print commands without executing |
| `target` | yes | - | - | Target name from config/targets.json |
| `action` | yes | - | `compile-dual-vmfb, run-dual-remote` | Benchmark action |
| `extra_args` | yes | - | - |  |

#### Subcommand `build`

```text
usage: ./merlin build [-h] [--dry-run]
                      [--profile {firesim,full-plugin,gemmini,npu,package-firesim,package-host,package-spacemit,qnn-compiler,qrb5165,radiance,radiance_muon,spacemit,vanilla,zephyr,zephyr-task}]
                      [--target {host,spacemit,qrb5165,firesim,zephyr,radiance_muon}]
                      [--kernel-dir KERNEL_DIR] [--kernel-name KERNEL_NAME]
                      [--kernel-body-obj KERNEL_BODY_OBJ]
                      [--config {debug,release,asan,trace,perf}]
                      [--cmake-target CMAKE_TARGET] [--with-plugin]
                      [--plugin-compiler | --no-plugin-compiler]
                      [--plugin-runtime | --no-plugin-runtime]
                      [--plugin-runtime-radiance | --no-plugin-runtime-radiance]
                      [--plugin-runtime-qnn | --no-plugin-runtime-qnn]
                      [--plugin-runtime-samples | --no-plugin-runtime-samples]
                      [--plugin-runtime-benchmarks | --no-plugin-runtime-benchmarks]
                      [--plugin-runtime-radiance-tests | --no-plugin-runtime-radiance-tests]
                      [--plugin-runtime-radiance-rpc | --no-plugin-runtime-radiance-rpc]
                      [--plugin-runtime-radiance-direct | --no-plugin-runtime-radiance-direct]
                      [--plugin-runtime-radiance-kmod | --no-plugin-runtime-radiance-kmod]
                      [--compiler-scope {all,gemmini,npu,saturn,spacemit,radiance,none}]
                      [--build-compiler | --no-build-compiler]
                      [--build-python-bindings | --no-build-python-bindings]
                      [--build-samples | --no-build-samples]
                      [--build-tests | --no-build-tests]
                      [--enable-libbacktrace | --no-enable-libbacktrace]
                      [--enable-tracy] [--offline-friendly]
                      [--cmake-bin CMAKE_BIN] [--use-system-cmake]
                      [--use-ccache | --no-use-ccache] [--cmake-arg CMAKE_ARG]
                      [--cmake-build-arg CMAKE_BUILD_ARG]
                      [--native-build-arg NATIVE_BUILD_ARG] [--clean]
                      [--verbose]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--dry-run` | no | `False` | - | Print commands without executing |
| `--profile` | no | - | `firesim, full-plugin, gemmini, npu, package-firesim, package-host, package-spacemit, qnn-compiler, qrb5165, radiance, radiance_muon, spacemit, vanilla, zephyr, zephyr-task` | High-level user profile preset. Use this for normal workflows; advanced flags may still override details. |
| `--target` | no | - | `host, spacemit, qrb5165, firesim, zephyr, radiance_muon` | Target platform. |
| `--kernel-dir` | no | - | - | For --target radiance_muon: absolute path to a directory containing kernel.cpp (and optionally host.cpp). Defaults to $RADIANCE_KERNELS_ROOT/kernels/vecadd. |
| `--kernel-name` | no | - | - | For --target radiance_muon: basename of the produced ELF (<name>.radiance.elf). Default: derived from --kernel-dir. |
| `--kernel-body-obj` | no | - | - | For --target radiance_muon (manifest mode): path to a precompiled Muon kernel-body .o file (typically produced by kernels/core/precompile.py from the Radiance manifest). When set, the wrapper template declares the kernel as `extern "C"` and the body .o is linked into kernel.radiance.elf at link time. |
| `--config` | no | - | `debug, release, asan, trace, perf` | Build configuration type |
| `--cmake-target` | no | - | - | Build specific CMake target (default: install) |
| `--with-plugin` | no | `False` | - | Enable Merlin compiler+runtime plugins (legacy umbrella switch). |
| `--plugin-compiler`, `--no-plugin-compiler` | no | - | - | Enable/disable Merlin compiler plugin targets (default follows --with-plugin). |
| `--plugin-runtime`, `--no-plugin-runtime` | no | - | - | Enable/disable Merlin runtime plugin integration (default follows --with-plugin). |
| `--plugin-runtime-radiance`, `--no-plugin-runtime-radiance` | no | - | - | Enable/disable Radiance HAL runtime plugin path (default: host+plugin only). |
| `--plugin-runtime-qnn`, `--no-plugin-runtime-qnn` | no | - | - | Enable/disable QNN HAL runtime plugin path for QRB5165 profiling. |
| `--plugin-runtime-samples`, `--no-plugin-runtime-samples` | no | - | - | Enable/disable runtime plugin samples subdir. |
| `--plugin-runtime-benchmarks`, `--no-plugin-runtime-benchmarks` | no | - | - | Enable/disable runtime plugin benchmarks subdir. |
| `--plugin-runtime-radiance-tests`, `--no-plugin-runtime-radiance-tests` | no | - | - | Enable/disable Radiance runtime plugin tests. |
| `--plugin-runtime-radiance-rpc`, `--no-plugin-runtime-radiance-rpc` | no | - | - | Enable/disable Radiance RPC-compat transport backend. |
| `--plugin-runtime-radiance-direct`, `--no-plugin-runtime-radiance-direct` | no | - | - | Enable/disable Radiance direct-submit transport backend. |
| `--plugin-runtime-radiance-kmod`, `--no-plugin-runtime-radiance-kmod` | no | - | - | Enable/disable Radiance kmod transport backend. |
| `--compiler-scope` | no | - | `all, gemmini, npu, saturn, spacemit, radiance, none` | Limit compiler-plugin target registration scope. Only used when compiler plugin + compiler build are enabled. |
| `--build-compiler`, `--no-build-compiler` | no | - | - | Override IREE_BUILD_COMPILER for this build. |
| `--build-python-bindings`, `--no-build-python-bindings` | no | - | - | Override IREE_BUILD_PYTHON_BINDINGS for this build. |
| `--build-samples`, `--no-build-samples` | no | - | - | Override IREE_BUILD_SAMPLES for this build. |
| `--build-tests`, `--no-build-tests` | no | - | - | Override IREE_BUILD_TESTS for this build. |
| `--enable-libbacktrace`, `--no-enable-libbacktrace` | no | - | - | Override IREE_ENABLE_LIBBACKTRACE for this build. |
| `--enable-tracy` | no | `False` | - | Enable Tracy runtime tracing (IREE_ENABLE_RUNTIME_TRACING=ON, IREE_TRACING_MODE=4). Compatible with any --config. |
| `--offline-friendly` | no | `False` | - | Prefer settings that avoid network fetches in CMake (equivalent to --no-build-compiler --no-build-python-bindings --no-enable-libbacktrace unless explicitly overridden). |
| `--cmake-bin` | no | `cmake` | - | CMake executable to use (default: cmake). |
| `--use-system-cmake` | no | `False` | - | Use /usr/bin/cmake instead of cmake from PATH. |
| `--use-ccache`, `--no-use-ccache` | no | `True` | - | Enable/disable ccache compiler launchers (default: enabled). |
| `--cmake-arg`, `--configure-custom-arg` | no | `[]` | - | Extra argument forwarded to CMake configure (repeatable). |
| `--cmake-build-arg`, `--build-custom-arg` | no | `[]` | - | Extra argument forwarded to CMake build command (repeatable). |
| `--native-build-arg` | no | `[]` | - | Extra argument forwarded to the native build tool after '--' (repeatable). |
| `--clean` | no | `False` | - | Delete build directory before building |
| `--verbose` | no | `False` | - | Enable verbose build output |

#### Subcommand `chipyard`

```text
usage: ./merlin chipyard [-h] [--dry-run] [--chipyard-root CHIPYARD_ROOT]
                         {set-path,info,validate,checkout,build-sim,run,configure-firesim,build-bitstream,register-hwdb,stage-workload,stage-zephyr-workload,run-zephyr,run-radiance-muon,build-firemarshal,status}
                         ...
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--dry-run` | no | `False` | - | Print commands without executing |
| `--chipyard-root` | no | - | - | Override chipyard root for this invocation |

#### Subcommand `ci`

```text
usage: ./merlin ci [-h] [--dry-run]
                   {lint,cli-docs-drift,patch-gate,release-status} ...
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--dry-run` | no | `False` | - | Print commands without executing |

#### Subcommand `compile`

```text
usage: ./merlin compile [-h] [--dry-run] --target TARGET [--hw HW]
                        [--quantized] [--output-dir OUTPUT_DIR]
                        [--build-dir BUILD_DIR] [--compile-to COMPILE_TO]
                        [--dump-compilation-phases-to DUMP_COMPILATION_PHASES_TO]
                        [--iree-compile-arg IREE_COMPILE_ARG]
                        [--reuse-imported-mlir] [--tracy] [--dump-artifacts]
                        [--dump-phases] [--dump-graph] [--qnn-partition]
                        [--build-benchmarks] [--qnn-preprocess-nhwc]
                        [--with-schedule WITH_SCHEDULE]
                        [--with-feedback WITH_FEEDBACK]
                        [--kernels-dir KERNELS_DIR]
                        [--kernel-manifest KERNEL_MANIFEST]
                        [--kernel-cache-dir KERNEL_CACHE_DIR]
                        [--no-kernel-embedding] [--kernels-strict-coverage]
                        input_path
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--dry-run` | no | `False` | - | Print commands without executing |
| `input_path` | yes | - | - | Path to the model directory OR specific .mlir/.onnx file |
| `--target` | yes | - | - | Target YAML config file name (e.g., spacemit_x60) |
| `--hw` | no | - | - | Hardware sub-target from YAML (e.g., RVV, OPU). If omitted, uses default_hw from YAML. |
| `--quantized` | no | `False` | - | Force quantized mode (auto-detected if .q. in filename) |
| `--output-dir` | no | - | - | Override output directory (default: build/compiled_models/<model>/<target>_<basename>/). If set, all generated files/artifacts are written under this directory. |
| `--build-dir` | no | `host-vanilla-release` | - | Which build directory to use for compiler tools (default: host-vanilla-release). If omitted and target YAML uses plugin_flags, compile.py auto-selects host-merlin-release. |
| `--compile-to` | no | - | - | Stop compilation at the given phase (for example: global-optimization). When set, output is emitted as an intermediate MLIR file. |
| `--dump-compilation-phases-to` | no | - | - | Directory for --dump-compilation-phases-to. If omitted and --dump-phases is set, defaults to <output_dir>/phases/. |
| `--iree-compile-arg`, `--compilation-custom-arg` | no | `[]` | - | Extra flag forwarded directly to iree-compile. Repeat for multiple flags. |
| `--reuse-imported-mlir` | no | `False` | - | Reuse an existing output MLIR instead of refreshing from explicit input files. By default, explicit input files are re-imported/re-copied. |
| `--tracy` | no | `False` | - | Enable Tracy profiling flags: embed debug info, use system linking, and enable debug symbols in generated code. Equivalent to --iree-hal-executable-debug-level=3 --iree-llvmcpu-link-embedded=false --iree-llvmcpu-debug-symbols=true |
| `--dump-artifacts` | no | `False` | - | Dump executable sources, binaries, and configs |
| `--dump-phases` | no | `False` | - | Dump MLIR compilation phases |
| `--dump-graph` | no | `False` | - | Dump the flow dispatch graph (.dot) |
| `--qnn-partition` | no | `False` | - | Run the QNN subgraph partitioner on the imported MLIR and emit a JSON dump of the per-island partition decision to <output_dir>/qnn_partition.json. Inspectable artifact for Phase 3b debugging; does not (yet) drive the final compile. |
| `--build-benchmarks` | no | `False` | - | Recompile individual dispatch benchmarks and zip them |
| `--qnn-preprocess-nhwc` | no | `False` | - | Insert iree-preprocessing-convert-conv-to-channels-last before the input-conversion phase so NCHW convs become NHWC. Required for the nhwc_int8_conv recognizer to match YOLOv8's stem/trunk/head convs (NCHW-anchored convs only get the Transpose-wrapped recognizer which HTA + Adreno reject on QAIRT 2.45). Auto-on when --with-schedule references HTA or GPU. |
| `--with-schedule` | no | - | - | Path to an XPU-RT schedule.json. Compiled with --iree-merlin-schedule-spec=<path> so DispatchCreation stamps stream.affinity (and split/grow/shard, when those land) per dispatch id. |
| `--with-feedback` | no | - | - | Path to an XPU-RT feedback.json (the persisted form written by targetgen_mcp.ingest_xpurt_feedback). When set, compile.py logs the overlay summary, derives a model-level granularity disposition, and writes <output_dir>/feedback_applied.json so downstream tooling (target-specific compile scripts, tools/run_full_loop.py) can read it. Inert when omitted — compile behavior is unchanged. See docs/merlin_integration.md. |
| `--kernels-dir` | no | - | - | Directory containing a kernels manifest.json (e.g. models/compiled_models/<model>/<target>/kernels/). When set, compile.py precompiles each kernel to its target object, auto-generates a transform-dialect spec, and threads --iree-preprocessing-transform-spec-filename + --iree-hal-executable-object-search-path into iree-compile. |
| `--kernel-manifest` | no | - | - | Explicit manifest path; overrides --kernels-dir/manifest.json. |
| `--kernel-cache-dir` | no | - | - | Where to write precompiled kernel objects + the generated transform spec. Defaults to <output_dir>/kernels_cache/. |
| `--no-kernel-embedding` | no | `False` | - | Disable the kernel embedding pipeline even if a manifest is discoverable from --kernels-dir / YAML custom_kernels. |
| `--kernels-strict-coverage` | no | `False` | - | After a kernel-embedded compile, fail with a non-zero exit if ANY linalg op in the input survived past the rewrite (i.e. fell through to IREE codegen). Use to verify that the manifest covers every op in the model. Implies --dump-phases. |

#### Subcommand `coverage-check`

```text
usage: ./merlin coverage-check [-h] [--csv CSV] vmfb
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `vmfb` | yes | - | - | .vmfb file to inspect |
| `--csv` | no | - | - | Write per-function CSV |

#### Subcommand `mcp`

```text
usage: ./merlin mcp [-h] {build,compile,run,perf,verify,targetgen}
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `name` | yes | - | `build, compile, run, perf, verify, targetgen` | Which MCP server to start. Each name N corresponds to tools/mcp_servers/N.py. |

#### Subcommand `patches`

```text
usage: ./merlin patches [-h] {verify,log,drift,export-upstream} ...
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `perf-decompose`

```text
usage: ./merlin perf-decompose [-h] [--topk TOPK] [--csv CSV] uartlog
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `uartlog` | yes | - | - | FireSim uartlog file |
| `--topk` | no | `20` | - | Print top-K hot dispatches (default 20) |
| `--csv` | no | - | - | Also write a CSV summary |

#### Subcommand `quantize`

```text
usage: ./merlin quantize [-h] --shape SHAPE [--output OUTPUT]
                         [--calibration-samples CALIBRATION_SAMPLES]
                         input_onnx
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `input_onnx` | yes | - | - | Source .onnx file |
| `--shape` | yes | - | - | Input tensor shape as comma-separated integers (e.g. 1,3,224,224). Repeat once per input for multi-input models. |
| `--output` | no | - | - | Output path (default: <input>.q.int8.onnx alongside the input). |
| `--calibration-samples` | no | `50` | - | Number of random calibration samples (default 50). |

#### Subcommand `ray`

```text
usage: ./merlin ray [-h] [--dry-run] [--state-root STATE_ROOT]
                    {cluster,jobs,resources,artifacts} ...
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--dry-run` | no | `False` | - | Print commands without executing |
| `--state-root` | no | `build/generated/ray` | - | Directory for Merlin-owned Ray cluster, run, artifact, and lease metadata. |

#### Subcommand `run`

```text
usage: ./merlin run [-h]
                    {full-loop,het-e2e,het-matrix,multi-device,roundtrip,schedule}
                    ...
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `mode` | yes | - | `full-loop, het-e2e, het-matrix, multi-device, roundtrip, schedule` | Which board-execution flow to run. See module docstrings under tools/run/ for per-mode flags. |
| `passthrough` | yes | - | - | Arguments forwarded verbatim to the underlying script. |

#### Subcommand `setup`

```text
usage: ./merlin setup [-h] [--env-name ENV_NAME] [--env-file ENV_FILE]
                      [--offline] [--skip-conda] [--skip-pip]
                      [--python-deps {auto,uv,pip}]
                      [--conda-no-plugins | --no-conda-no-plugins]
                      [--submodules-profile {core,npu,smolvla,full}]
                      [--submodule-path SUBMODULE_PATH]
                      [--submodule-paths-recursive | --no-submodule-paths-recursive]
                      [--submodule-depth SUBMODULE_DEPTH]
                      [--submodule-jobs SUBMODULE_JOBS] [--submodule-sync]
                      [--toolchain-target {spacemit,firesim,all}]
                      [--with-qemu] [--toolchain-force]
                      [--prebuilt-artifact {host-linux-x86_64,host-macos,runtime-saturnopu,runtime-spacemit}]
                      [--prebuilt-tag PREBUILT_TAG]
                      [--prebuilt-repo PREBUILT_REPO] [--prebuilt-force]
                      [{all,env,toolchain,submodules,prebuilt}]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `component` | no | `all` | `all, env, toolchain, submodules, prebuilt` |  |
| `--env-name` | no | `merlin-dev` | - | Conda environment name to update/install packages into (default: merlin-dev). |
| `--env-file` | no | `<repo>/env_linux.yml` | - | Conda environment file to use. Default is platform-specific: env_linux.yml |
| `--offline` | no | `False` | - | Run setup in offline mode when possible. |
| `--skip-conda` | no | `False` | - | Skip conda environment sync. |
| `--skip-pip` | no | `False` | - | Skip Python dependency sync (uv/pip). |
| `--python-deps` | no | `auto` | `auto, uv, pip` | Python dependency installer. 'auto' prefers uv sync with uv.lock and falls back to pip. |
| `--conda-no-plugins`, `--no-conda-no-plugins` | no | - | - | Force CONDA_NO_PLUGINS for conda env update. If unset, setup retries with CONDA_NO_PLUGINS=true on failure. |
| `--submodules-profile` | no | `core` | `core, npu, smolvla, full` | Which submodule profile to initialize (default: core). |
| `--submodule-path` | no | `[]` | - | Additional top-level submodule path to initialize (repeatable). |
| `--submodule-paths-recursive`, `--no-submodule-paths-recursive` | no | `False` | - | Whether extra --submodule-path entries should be initialized recursively. |
| `--submodule-depth` | no | `1` | - | Shallow depth for submodule fetches (default: 1). Use 0 for full history. |
| `--submodule-jobs` | no | `8` | - | Parallel submodule fetch jobs (default: 8). |
| `--submodule-sync` | no | `False` | - | Run `git submodule sync --recursive` before updating. |
| `--toolchain-target` | no | `spacemit` | `spacemit, firesim, all` | Which toolchain target to install (default: spacemit). |
| `--with-qemu` | no | `False` | - | For firesim toolchain setup, also install QEMU. |
| `--toolchain-force` | no | `False` | - | Reinstall toolchains even if the destination already exists. |
| `--prebuilt-artifact` | no | `host-linux-x86_64` | `host-linux-x86_64, host-macos, runtime-saturnopu, runtime-spacemit` | Which published Merlin prebuilt artifact to install. |
| `--prebuilt-tag` | no | `latest` | - | GitHub release tag to download from, or 'latest' (default: latest). |
| `--prebuilt-repo` | no | `ucb-bar/merlin` | - | GitHub repository containing release assets (default: ucb-bar/merlin). |
| `--prebuilt-force` | no | `False` | - | Replace an existing destination build tree when installing a prebuilt artifact. |

#### Subcommand `sim`

```text
usage: ./merlin sim [-h] [--target TARGET] [--bench-target BENCH_TARGET]
                    [--simulator {vcs,verilator}] [--config CONFIG]
                    [--reference REFERENCE] [--output-dir OUTPUT_DIR] [--keep]
                    [--build-dir BUILD_DIR]
                    [--firesim-build-dir FIRESIM_BUILD_DIR] [--skip-build]
                    [--skip-compile] [--timeout TIMEOUT] [-v]
                    input
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `input` | yes | - | - | Input .mlir fixture (any model — see tests/integration/<target>/fixtures/ for examples). |
| `--target` | no | `gemmini_mx_vcs` | - | Model YAML target (default: gemmini_mx_vcs). Any models/<target>.yaml is accepted; pass --bench-target if the cmake bench target name does not match the default mapping. |
| `--bench-target` | no | - | - | Explicit cmake bench target name. Overrides the default mapping in _TARGET_TO_BENCH; required when --target is not a key in the default mapping. |
| `--simulator` | no | `vcs` | `vcs, verilator` | Chipyard simulator backend (default: vcs) |
| `--config` | no | `RadianceGemminiOnlyConfig` | - | Chipyard CONFIG (default: RadianceGemminiOnlyConfig) |
| `--reference` | no | - | - | Path to expected output (one i32 per line). If set, run outputs are diffed against this and an exit code of 0/1 is returned. |
| `--output-dir` | no | - | - | Directory for produced artifacts (default: /scratch2/agustin/merlin/build/sim/<fixture>) |
| `--keep` | no | `False` | - | Keep simulator working dir + log on success |
| `--build-dir` | no | `host-merlin-release` | - | Host build dir for iree-compile (default: host-merlin-release) |
| `--firesim-build-dir` | no | `firesim-merlin-release` | - | Firesim build dir holding the bench ELF (default: firesim-merlin-release) |
| `--skip-build` | no | `False` | - | Skip ./merlin build step (use a pre-built ELF) |
| `--skip-compile` | no | `False` | - | Skip ./merlin compile step (use a pre-built VMFB) |
| `--timeout` | no | `900` | - | Simulator wallclock timeout in seconds (default 900) |
| `-v`, `--verbose` | no | `False` | - | Verbose logging |

#### Subcommand `spike`

```text
usage: ./merlin spike [-h] [--output-dir OUTPUT_DIR] [--target TARGET]
                      [--build-dir BUILD_DIR] [-v]
                      input
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `input` | yes | - | - | Input .mlir fixture (tensor-domain) |
| `--output-dir` | no | - | - | Directory for the produced .vmfb (default: build/spike/<basename>) |
| `--target` | no | `gemmini_spike` | - | Model YAML name (default: gemmini_spike) |
| `--build-dir` | no | `host-merlin-debug` | - | Which build dir to use for iree-compile (default: host-merlin-debug) |
| `-v`, `--verbose` | no | `False` | - | Verbose logging |

#### Subcommand `targetgen`

```text
usage: ./merlin targetgen [-h] [--dry-run]
                          {validate,plan,generate,explain,orchestrate,execute,stage-mutation,answer,status,ingest,classify,modification-map,mcp}
                          ...
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--dry-run` | no | `False` | - | Print commands without executing |

#### Subcommand `verify-output`

```text
usage: ./merlin verify-output [-h] --shape SHAPE [--observed OBSERVED]
                              [--uartlog UARTLOG] [--seed SEED]
                              [--random-input] [--skip-golden]
                              model
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `model` | yes | - | - | Quantized .q.int8.onnx model |
| `--shape` | yes | - | - | Input shape comma-separated (repeat per input) |
| `--observed` | no | `[]` | - | Backend hash to verify: <hex_hash>:<label> (e.g. 0x498...:gemmini) |
| `--uartlog` | no | `[]` | - | FireSim uartlog file to extract hashes from |
| `--seed` | no | `51966` | - | RNG seed for x86 reference input when --random-input (default 0xCAFE) |
| `--random-input` | no | `False` | - | Use random input instead of all-zero (the runner uses zeros via ZERO_FILL buffer alloc). Use this only for sanity checks. |
| `--skip-golden` | no | `False` | - | Skip the onnxruntime baseline (just cross-check observed hashes) |

### `--help` Output

```text
usage: ./merlin [-h]
                {build,compile,quantize,verify-output,perf-decompose,coverage-check,setup,ci,patches,benchmark,chipyard,ray,targetgen,spike,sim,run,mcp}
                ...

Unified Merlin developer command dispatcher.

positional arguments:
  {build,compile,quantize,verify-output,perf-decompose,coverage-check,setup,ci,patches,benchmark,chipyard,ray,targetgen,spike,sim,run,mcp}
    build               Configure and build Merlin and target runtimes
    compile             Compile MLIR/ONNX models to target artifacts
    quantize            INT8-quantize any .onnx model (QDQ, symmetric, per-
                        tensor)
    verify-output       Cross-hash check backend outputs vs onnxruntime golden
    perf-decompose      Per-dispatch performance decomposition from FireSim
                        uartlog
    coverage-check      Per-dispatch accelerator coverage report of a VMFB
    setup               Bootstrap developer environment and toolchains
    ci                  Run repository CI/lint/patch workflows
    patches             Verify submodule state and manage upstream patches
    benchmark           Run benchmark helper scripts
    chipyard            Manage Chipyard hardware backend interactions
    ray                 Manage Merlin's Ray control plane, jobs, resources,
                        and artifacts
    targetgen           Plan and orchestrate hardware-spec-driven target
                        enablement
    spike               Run a Gemmini kernel fixture under spike+pk for
                        functional validation
    sim                 Run an mxGemmini fixture through the IREE pipeline +
                        chipyard VCS simulator
    run                 Execute a compiled model on a target board
                        (schedule/multi-device/het-e2e/...)
    mcp                 Start one of the Merlin MCP servers
                        (build/compile/run/perf/verify)

options:
  -h, --help            show this help message and exit
```

## `./merlin build`

Configure and build Merlin and target runtimes.

### Usage

```text
usage: ./merlin build [-h]
                      [--profile {firesim,full-plugin,gemmini,npu,package-firesim,package-host,package-spacemit,qnn-compiler,qrb5165,radiance,radiance_muon,spacemit,vanilla,zephyr,zephyr-task}]
                      [--target {host,spacemit,qrb5165,firesim,zephyr,radiance_muon}]
                      [--kernel-dir KERNEL_DIR] [--kernel-name KERNEL_NAME]
                      [--kernel-body-obj KERNEL_BODY_OBJ]
                      [--config {debug,release,asan,trace,perf}]
                      [--cmake-target CMAKE_TARGET] [--with-plugin]
                      [--plugin-compiler | --no-plugin-compiler]
                      [--plugin-runtime | --no-plugin-runtime]
                      [--plugin-runtime-radiance | --no-plugin-runtime-radiance]
                      [--plugin-runtime-qnn | --no-plugin-runtime-qnn]
                      [--plugin-runtime-samples | --no-plugin-runtime-samples]
                      [--plugin-runtime-benchmarks | --no-plugin-runtime-benchmarks]
                      [--plugin-runtime-radiance-tests | --no-plugin-runtime-radiance-tests]
                      [--plugin-runtime-radiance-rpc | --no-plugin-runtime-radiance-rpc]
                      [--plugin-runtime-radiance-direct | --no-plugin-runtime-radiance-direct]
                      [--plugin-runtime-radiance-kmod | --no-plugin-runtime-radiance-kmod]
                      [--compiler-scope {all,gemmini,npu,saturn,spacemit,radiance,none}]
                      [--build-compiler | --no-build-compiler]
                      [--build-python-bindings | --no-build-python-bindings]
                      [--build-samples | --no-build-samples]
                      [--build-tests | --no-build-tests]
                      [--enable-libbacktrace | --no-enable-libbacktrace]
                      [--enable-tracy] [--offline-friendly]
                      [--cmake-bin CMAKE_BIN] [--use-system-cmake]
                      [--use-ccache | --no-use-ccache] [--cmake-arg CMAKE_ARG]
                      [--cmake-build-arg CMAKE_BUILD_ARG]
                      [--native-build-arg NATIVE_BUILD_ARG] [--clean]
                      [--verbose]
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--profile` | no | - | `firesim, full-plugin, gemmini, npu, package-firesim, package-host, package-spacemit, qnn-compiler, qrb5165, radiance, radiance_muon, spacemit, vanilla, zephyr, zephyr-task` | High-level user profile preset. Use this for normal workflows; advanced flags may still override details. |
| `--target` | no | - | `host, spacemit, qrb5165, firesim, zephyr, radiance_muon` | Target platform. |
| `--kernel-dir` | no | - | - | For --target radiance_muon: absolute path to a directory containing kernel.cpp (and optionally host.cpp). Defaults to $RADIANCE_KERNELS_ROOT/kernels/vecadd. |
| `--kernel-name` | no | - | - | For --target radiance_muon: basename of the produced ELF (<name>.radiance.elf). Default: derived from --kernel-dir. |
| `--kernel-body-obj` | no | - | - | For --target radiance_muon (manifest mode): path to a precompiled Muon kernel-body .o file (typically produced by kernels/core/precompile.py from the Radiance manifest). When set, the wrapper template declares the kernel as `extern "C"` and the body .o is linked into kernel.radiance.elf at link time. |
| `--config` | no | - | `debug, release, asan, trace, perf` | Build configuration type |
| `--cmake-target` | no | - | - | Build specific CMake target (default: install) |
| `--with-plugin` | no | `False` | - | Enable Merlin compiler+runtime plugins (legacy umbrella switch). |
| `--plugin-compiler`, `--no-plugin-compiler` | no | - | - | Enable/disable Merlin compiler plugin targets (default follows --with-plugin). |
| `--plugin-runtime`, `--no-plugin-runtime` | no | - | - | Enable/disable Merlin runtime plugin integration (default follows --with-plugin). |
| `--plugin-runtime-radiance`, `--no-plugin-runtime-radiance` | no | - | - | Enable/disable Radiance HAL runtime plugin path (default: host+plugin only). |
| `--plugin-runtime-qnn`, `--no-plugin-runtime-qnn` | no | - | - | Enable/disable QNN HAL runtime plugin path for QRB5165 profiling. |
| `--plugin-runtime-samples`, `--no-plugin-runtime-samples` | no | - | - | Enable/disable runtime plugin samples subdir. |
| `--plugin-runtime-benchmarks`, `--no-plugin-runtime-benchmarks` | no | - | - | Enable/disable runtime plugin benchmarks subdir. |
| `--plugin-runtime-radiance-tests`, `--no-plugin-runtime-radiance-tests` | no | - | - | Enable/disable Radiance runtime plugin tests. |
| `--plugin-runtime-radiance-rpc`, `--no-plugin-runtime-radiance-rpc` | no | - | - | Enable/disable Radiance RPC-compat transport backend. |
| `--plugin-runtime-radiance-direct`, `--no-plugin-runtime-radiance-direct` | no | - | - | Enable/disable Radiance direct-submit transport backend. |
| `--plugin-runtime-radiance-kmod`, `--no-plugin-runtime-radiance-kmod` | no | - | - | Enable/disable Radiance kmod transport backend. |
| `--compiler-scope` | no | - | `all, gemmini, npu, saturn, spacemit, radiance, none` | Limit compiler-plugin target registration scope. Only used when compiler plugin + compiler build are enabled. |
| `--build-compiler`, `--no-build-compiler` | no | - | - | Override IREE_BUILD_COMPILER for this build. |
| `--build-python-bindings`, `--no-build-python-bindings` | no | - | - | Override IREE_BUILD_PYTHON_BINDINGS for this build. |
| `--build-samples`, `--no-build-samples` | no | - | - | Override IREE_BUILD_SAMPLES for this build. |
| `--build-tests`, `--no-build-tests` | no | - | - | Override IREE_BUILD_TESTS for this build. |
| `--enable-libbacktrace`, `--no-enable-libbacktrace` | no | - | - | Override IREE_ENABLE_LIBBACKTRACE for this build. |
| `--enable-tracy` | no | `False` | - | Enable Tracy runtime tracing (IREE_ENABLE_RUNTIME_TRACING=ON, IREE_TRACING_MODE=4). Compatible with any --config. |
| `--offline-friendly` | no | `False` | - | Prefer settings that avoid network fetches in CMake (equivalent to --no-build-compiler --no-build-python-bindings --no-enable-libbacktrace unless explicitly overridden). |
| `--cmake-bin` | no | `cmake` | - | CMake executable to use (default: cmake). |
| `--use-system-cmake` | no | `False` | - | Use /usr/bin/cmake instead of cmake from PATH. |
| `--use-ccache`, `--no-use-ccache` | no | `True` | - | Enable/disable ccache compiler launchers (default: enabled). |
| `--cmake-arg`, `--configure-custom-arg` | no | `[]` | - | Extra argument forwarded to CMake configure (repeatable). |
| `--cmake-build-arg`, `--build-custom-arg` | no | `[]` | - | Extra argument forwarded to CMake build command (repeatable). |
| `--native-build-arg` | no | `[]` | - | Extra argument forwarded to the native build tool after '--' (repeatable). |
| `--clean` | no | `False` | - | Delete build directory before building |
| `--verbose` | no | `False` | - | Enable verbose build output |

### `--help` Output

```text
usage: ./merlin build [-h]
                      [--profile {firesim,full-plugin,gemmini,npu,package-firesim,package-host,package-spacemit,qnn-compiler,qrb5165,radiance,radiance_muon,spacemit,vanilla,zephyr,zephyr-task}]
                      [--target {host,spacemit,qrb5165,firesim,zephyr,radiance_muon}]
                      [--kernel-dir KERNEL_DIR] [--kernel-name KERNEL_NAME]
                      [--kernel-body-obj KERNEL_BODY_OBJ]
                      [--config {debug,release,asan,trace,perf}]
                      [--cmake-target CMAKE_TARGET] [--with-plugin]
                      [--plugin-compiler | --no-plugin-compiler]
                      [--plugin-runtime | --no-plugin-runtime]
                      [--plugin-runtime-radiance | --no-plugin-runtime-radiance]
                      [--plugin-runtime-qnn | --no-plugin-runtime-qnn]
                      [--plugin-runtime-samples | --no-plugin-runtime-samples]
                      [--plugin-runtime-benchmarks | --no-plugin-runtime-benchmarks]
                      [--plugin-runtime-radiance-tests | --no-plugin-runtime-radiance-tests]
                      [--plugin-runtime-radiance-rpc | --no-plugin-runtime-radiance-rpc]
                      [--plugin-runtime-radiance-direct | --no-plugin-runtime-radiance-direct]
                      [--plugin-runtime-radiance-kmod | --no-plugin-runtime-radiance-kmod]
                      [--compiler-scope {all,gemmini,npu,saturn,spacemit,radiance,none}]
                      [--build-compiler | --no-build-compiler]
                      [--build-python-bindings | --no-build-python-bindings]
                      [--build-samples | --no-build-samples]
                      [--build-tests | --no-build-tests]
                      [--enable-libbacktrace | --no-enable-libbacktrace]
                      [--enable-tracy] [--offline-friendly]
                      [--cmake-bin CMAKE_BIN] [--use-system-cmake]
                      [--use-ccache | --no-use-ccache] [--cmake-arg CMAKE_ARG]
                      [--cmake-build-arg CMAKE_BUILD_ARG]
                      [--native-build-arg NATIVE_BUILD_ARG] [--clean]
                      [--verbose]

Configure and build Merlin and target runtimes.

options:
  -h, --help            show this help message and exit
  --profile {firesim,full-plugin,gemmini,npu,package-firesim,package-host,package-spacemit,qnn-compiler,qrb5165,radiance,radiance_muon,spacemit,vanilla,zephyr,zephyr-task}
                        High-level user profile preset. Use this for normal
                        workflows; advanced flags may still override details.
  --target {host,spacemit,qrb5165,firesim,zephyr,radiance_muon}
                        Target platform.
  --kernel-dir KERNEL_DIR
                        For --target radiance_muon: absolute path to a
                        directory containing kernel.cpp (and optionally
                        host.cpp). Defaults to
                        $RADIANCE_KERNELS_ROOT/kernels/vecadd.
  --kernel-name KERNEL_NAME
                        For --target radiance_muon: basename of the produced
                        ELF (<name>.radiance.elf). Default: derived from
                        --kernel-dir.
  --kernel-body-obj KERNEL_BODY_OBJ
                        For --target radiance_muon (manifest mode): path to a
                        precompiled Muon kernel-body .o file (typically
                        produced by kernels/core/precompile.py from the
                        Radiance manifest). When set, the wrapper template
                        declares the kernel as `extern "C"` and the body .o is
                        linked into kernel.radiance.elf at link time.
  --config {debug,release,asan,trace,perf}
                        Build configuration type
  --cmake-target CMAKE_TARGET
                        Build specific CMake target (default: install)
  --with-plugin         Enable Merlin compiler+runtime plugins (legacy
                        umbrella switch).
  --plugin-compiler, --no-plugin-compiler
                        Enable/disable Merlin compiler plugin targets (default
                        follows --with-plugin).
  --plugin-runtime, --no-plugin-runtime
                        Enable/disable Merlin runtime plugin integration
                        (default follows --with-plugin).
  --plugin-runtime-radiance, --no-plugin-runtime-radiance
                        Enable/disable Radiance HAL runtime plugin path
                        (default: host+plugin only).
  --plugin-runtime-qnn, --no-plugin-runtime-qnn
                        Enable/disable QNN HAL runtime plugin path for QRB5165
                        profiling.
  --plugin-runtime-samples, --no-plugin-runtime-samples
                        Enable/disable runtime plugin samples subdir.
  --plugin-runtime-benchmarks, --no-plugin-runtime-benchmarks
                        Enable/disable runtime plugin benchmarks subdir.
  --plugin-runtime-radiance-tests, --no-plugin-runtime-radiance-tests
                        Enable/disable Radiance runtime plugin tests.
  --plugin-runtime-radiance-rpc, --no-plugin-runtime-radiance-rpc
                        Enable/disable Radiance RPC-compat transport backend.
  --plugin-runtime-radiance-direct, --no-plugin-runtime-radiance-direct
                        Enable/disable Radiance direct-submit transport
                        backend.
  --plugin-runtime-radiance-kmod, --no-plugin-runtime-radiance-kmod
                        Enable/disable Radiance kmod transport backend.
  --compiler-scope {all,gemmini,npu,saturn,spacemit,radiance,none}
                        Limit compiler-plugin target registration scope. Only
                        used when compiler plugin + compiler build are
                        enabled.
  --build-compiler, --no-build-compiler
                        Override IREE_BUILD_COMPILER for this build.
  --build-python-bindings, --no-build-python-bindings
                        Override IREE_BUILD_PYTHON_BINDINGS for this build.
  --build-samples, --no-build-samples
                        Override IREE_BUILD_SAMPLES for this build.
  --build-tests, --no-build-tests
                        Override IREE_BUILD_TESTS for this build.
  --enable-libbacktrace, --no-enable-libbacktrace
                        Override IREE_ENABLE_LIBBACKTRACE for this build.
  --enable-tracy        Enable Tracy runtime tracing
                        (IREE_ENABLE_RUNTIME_TRACING=ON, IREE_TRACING_MODE=4).
                        Compatible with any --config.
  --offline-friendly    Prefer settings that avoid network fetches in CMake
                        (equivalent to --no-build-compiler --no-build-python-
                        bindings --no-enable-libbacktrace unless explicitly
                        overridden).
  --cmake-bin CMAKE_BIN
                        CMake executable to use (default: cmake).
  --use-system-cmake    Use /usr/bin/cmake instead of cmake from PATH.
  --use-ccache, --no-use-ccache
                        Enable/disable ccache compiler launchers (default:
                        enabled).
  --cmake-arg CMAKE_ARG, --configure-custom-arg CMAKE_ARG
                        Extra argument forwarded to CMake configure
                        (repeatable).
  --cmake-build-arg CMAKE_BUILD_ARG, --build-custom-arg CMAKE_BUILD_ARG
                        Extra argument forwarded to CMake build command
                        (repeatable).
  --native-build-arg NATIVE_BUILD_ARG
                        Extra argument forwarded to the native build tool
                        after '--' (repeatable).
  --clean               Delete build directory before building
  --verbose             Enable verbose build output
```

## `./merlin compile`

Compile MLIR/ONNX models to target artifacts.

### Usage

```text
usage: ./merlin compile [-h] --target TARGET [--hw HW] [--quantized]
                        [--output-dir OUTPUT_DIR] [--build-dir BUILD_DIR]
                        [--compile-to COMPILE_TO]
                        [--dump-compilation-phases-to DUMP_COMPILATION_PHASES_TO]
                        [--iree-compile-arg IREE_COMPILE_ARG]
                        [--reuse-imported-mlir] [--tracy] [--dump-artifacts]
                        [--dump-phases] [--dump-graph] [--qnn-partition]
                        [--build-benchmarks] [--qnn-preprocess-nhwc]
                        [--with-schedule WITH_SCHEDULE]
                        [--with-feedback WITH_FEEDBACK]
                        [--kernels-dir KERNELS_DIR]
                        [--kernel-manifest KERNEL_MANIFEST]
                        [--kernel-cache-dir KERNEL_CACHE_DIR]
                        [--no-kernel-embedding] [--kernels-strict-coverage]
                        input_path
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `input_path` | yes | - | - | Path to the model directory OR specific .mlir/.onnx file |
| `--target` | yes | - | - | Target YAML config file name (e.g., spacemit_x60) |
| `--hw` | no | - | - | Hardware sub-target from YAML (e.g., RVV, OPU). If omitted, uses default_hw from YAML. |
| `--quantized` | no | `False` | - | Force quantized mode (auto-detected if .q. in filename) |
| `--output-dir` | no | - | - | Override output directory (default: build/compiled_models/<model>/<target>_<basename>/). If set, all generated files/artifacts are written under this directory. |
| `--build-dir` | no | `host-vanilla-release` | - | Which build directory to use for compiler tools (default: host-vanilla-release). If omitted and target YAML uses plugin_flags, compile.py auto-selects host-merlin-release. |
| `--compile-to` | no | - | - | Stop compilation at the given phase (for example: global-optimization). When set, output is emitted as an intermediate MLIR file. |
| `--dump-compilation-phases-to` | no | - | - | Directory for --dump-compilation-phases-to. If omitted and --dump-phases is set, defaults to <output_dir>/phases/. |
| `--iree-compile-arg`, `--compilation-custom-arg` | no | `[]` | - | Extra flag forwarded directly to iree-compile. Repeat for multiple flags. |
| `--reuse-imported-mlir` | no | `False` | - | Reuse an existing output MLIR instead of refreshing from explicit input files. By default, explicit input files are re-imported/re-copied. |
| `--tracy` | no | `False` | - | Enable Tracy profiling flags: embed debug info, use system linking, and enable debug symbols in generated code. Equivalent to --iree-hal-executable-debug-level=3 --iree-llvmcpu-link-embedded=false --iree-llvmcpu-debug-symbols=true |
| `--dump-artifacts` | no | `False` | - | Dump executable sources, binaries, and configs |
| `--dump-phases` | no | `False` | - | Dump MLIR compilation phases |
| `--dump-graph` | no | `False` | - | Dump the flow dispatch graph (.dot) |
| `--qnn-partition` | no | `False` | - | Run the QNN subgraph partitioner on the imported MLIR and emit a JSON dump of the per-island partition decision to <output_dir>/qnn_partition.json. Inspectable artifact for Phase 3b debugging; does not (yet) drive the final compile. |
| `--build-benchmarks` | no | `False` | - | Recompile individual dispatch benchmarks and zip them |
| `--qnn-preprocess-nhwc` | no | `False` | - | Insert iree-preprocessing-convert-conv-to-channels-last before the input-conversion phase so NCHW convs become NHWC. Required for the nhwc_int8_conv recognizer to match YOLOv8's stem/trunk/head convs (NCHW-anchored convs only get the Transpose-wrapped recognizer which HTA + Adreno reject on QAIRT 2.45). Auto-on when --with-schedule references HTA or GPU. |
| `--with-schedule` | no | - | - | Path to an XPU-RT schedule.json. Compiled with --iree-merlin-schedule-spec=<path> so DispatchCreation stamps stream.affinity (and split/grow/shard, when those land) per dispatch id. |
| `--with-feedback` | no | - | - | Path to an XPU-RT feedback.json (the persisted form written by targetgen_mcp.ingest_xpurt_feedback). When set, compile.py logs the overlay summary, derives a model-level granularity disposition, and writes <output_dir>/feedback_applied.json so downstream tooling (target-specific compile scripts, tools/run_full_loop.py) can read it. Inert when omitted — compile behavior is unchanged. See docs/merlin_integration.md. |
| `--kernels-dir` | no | - | - | Directory containing a kernels manifest.json (e.g. models/compiled_models/<model>/<target>/kernels/). When set, compile.py precompiles each kernel to its target object, auto-generates a transform-dialect spec, and threads --iree-preprocessing-transform-spec-filename + --iree-hal-executable-object-search-path into iree-compile. |
| `--kernel-manifest` | no | - | - | Explicit manifest path; overrides --kernels-dir/manifest.json. |
| `--kernel-cache-dir` | no | - | - | Where to write precompiled kernel objects + the generated transform spec. Defaults to <output_dir>/kernels_cache/. |
| `--no-kernel-embedding` | no | `False` | - | Disable the kernel embedding pipeline even if a manifest is discoverable from --kernels-dir / YAML custom_kernels. |
| `--kernels-strict-coverage` | no | `False` | - | After a kernel-embedded compile, fail with a non-zero exit if ANY linalg op in the input survived past the rewrite (i.e. fell through to IREE codegen). Use to verify that the manifest covers every op in the model. Implies --dump-phases. |

### `--help` Output

```text
usage: ./merlin compile [-h] --target TARGET [--hw HW] [--quantized]
                        [--output-dir OUTPUT_DIR] [--build-dir BUILD_DIR]
                        [--compile-to COMPILE_TO]
                        [--dump-compilation-phases-to DUMP_COMPILATION_PHASES_TO]
                        [--iree-compile-arg IREE_COMPILE_ARG]
                        [--reuse-imported-mlir] [--tracy] [--dump-artifacts]
                        [--dump-phases] [--dump-graph] [--qnn-partition]
                        [--build-benchmarks] [--qnn-preprocess-nhwc]
                        [--with-schedule WITH_SCHEDULE]
                        [--with-feedback WITH_FEEDBACK]
                        [--kernels-dir KERNELS_DIR]
                        [--kernel-manifest KERNEL_MANIFEST]
                        [--kernel-cache-dir KERNEL_CACHE_DIR]
                        [--no-kernel-embedding] [--kernels-strict-coverage]
                        input_path

Compile MLIR/ONNX models to target artifacts.

positional arguments:
  input_path            Path to the model directory OR specific .mlir/.onnx
                        file

options:
  -h, --help            show this help message and exit
  --target TARGET       Target YAML config file name (e.g., spacemit_x60)
  --hw HW               Hardware sub-target from YAML (e.g., RVV, OPU). If
                        omitted, uses default_hw from YAML.
  --quantized           Force quantized mode (auto-detected if .q. in
                        filename)
  --output-dir OUTPUT_DIR
                        Override output directory (default:
                        build/compiled_models/<model>/<target>_<basename>/).
                        If set, all generated files/artifacts are written
                        under this directory.
  --build-dir BUILD_DIR
                        Which build directory to use for compiler tools
                        (default: host-vanilla-release). If omitted and target
                        YAML uses plugin_flags, compile.py auto-selects host-
                        merlin-release.
  --compile-to COMPILE_TO
                        Stop compilation at the given phase (for example:
                        global-optimization). When set, output is emitted as
                        an intermediate MLIR file.
  --dump-compilation-phases-to DUMP_COMPILATION_PHASES_TO
                        Directory for --dump-compilation-phases-to. If omitted
                        and --dump-phases is set, defaults to
                        <output_dir>/phases/.
  --iree-compile-arg IREE_COMPILE_ARG, --compilation-custom-arg IREE_COMPILE_ARG
                        Extra flag forwarded directly to iree-compile. Repeat
                        for multiple flags.
  --reuse-imported-mlir
                        Reuse an existing output MLIR instead of refreshing
                        from explicit input files. By default, explicit input
                        files are re-imported/re-copied.
  --tracy               Enable Tracy profiling flags: embed debug info, use
                        system linking, and enable debug symbols in generated
                        code. Equivalent to --iree-hal-executable-debug-
                        level=3 --iree-llvmcpu-link-embedded=false --iree-
                        llvmcpu-debug-symbols=true
  --dump-artifacts      Dump executable sources, binaries, and configs
  --dump-phases         Dump MLIR compilation phases
  --dump-graph          Dump the flow dispatch graph (.dot)
  --qnn-partition       Run the QNN subgraph partitioner on the imported MLIR
                        and emit a JSON dump of the per-island partition
                        decision to <output_dir>/qnn_partition.json.
                        Inspectable artifact for Phase 3b debugging; does not
                        (yet) drive the final compile.
  --build-benchmarks    Recompile individual dispatch benchmarks and zip them
  --qnn-preprocess-nhwc
                        Insert iree-preprocessing-convert-conv-to-channels-
                        last before the input-conversion phase so NCHW convs
                        become NHWC. Required for the nhwc_int8_conv
                        recognizer to match YOLOv8's stem/trunk/head convs
                        (NCHW-anchored convs only get the Transpose-wrapped
                        recognizer which HTA + Adreno reject on QAIRT 2.45).
                        Auto-on when --with-schedule references HTA or GPU.
  --with-schedule WITH_SCHEDULE
                        Path to an XPU-RT schedule.json. Compiled with --iree-
                        merlin-schedule-spec=<path> so DispatchCreation stamps
                        stream.affinity (and split/grow/shard, when those
                        land) per dispatch id.
  --with-feedback WITH_FEEDBACK
                        Path to an XPU-RT feedback.json (the persisted form
                        written by targetgen_mcp.ingest_xpurt_feedback). When
                        set, compile.py logs the overlay summary, derives a
                        model-level granularity disposition, and writes
                        <output_dir>/feedback_applied.json so downstream
                        tooling (target-specific compile scripts,
                        tools/run_full_loop.py) can read it. Inert when
                        omitted — compile behavior is unchanged. See
                        docs/merlin_integration.md.
  --kernels-dir KERNELS_DIR
                        Directory containing a kernels manifest.json (e.g.
                        models/compiled_models/<model>/<target>/kernels/).
                        When set, compile.py precompiles each kernel to its
                        target object, auto-generates a transform-dialect
                        spec, and threads --iree-preprocessing-transform-spec-
                        filename + --iree-hal-executable-object-search-path
                        into iree-compile.
  --kernel-manifest KERNEL_MANIFEST
                        Explicit manifest path; overrides --kernels-
                        dir/manifest.json.
  --kernel-cache-dir KERNEL_CACHE_DIR
                        Where to write precompiled kernel objects + the
                        generated transform spec. Defaults to
                        <output_dir>/kernels_cache/.
  --no-kernel-embedding
                        Disable the kernel embedding pipeline even if a
                        manifest is discoverable from --kernels-dir / YAML
                        custom_kernels.
  --kernels-strict-coverage
                        After a kernel-embedded compile, fail with a non-zero
                        exit if ANY linalg op in the input survived past the
                        rewrite (i.e. fell through to IREE codegen). Use to
                        verify that the manifest covers every op in the model.
                        Implies --dump-phases.
```

## `./merlin setup`

Bootstrap developer environment and toolchains.

### Usage

```text
usage: ./merlin setup [-h] [--env-name ENV_NAME] [--env-file ENV_FILE]
                      [--offline] [--skip-conda] [--skip-pip]
                      [--python-deps {auto,uv,pip}]
                      [--conda-no-plugins | --no-conda-no-plugins]
                      [--submodules-profile {core,npu,smolvla,full}]
                      [--submodule-path SUBMODULE_PATH]
                      [--submodule-paths-recursive | --no-submodule-paths-recursive]
                      [--submodule-depth SUBMODULE_DEPTH]
                      [--submodule-jobs SUBMODULE_JOBS] [--submodule-sync]
                      [--toolchain-target {spacemit,firesim,all}]
                      [--with-qemu] [--toolchain-force]
                      [--prebuilt-artifact {host-linux-x86_64,host-macos,runtime-saturnopu,runtime-spacemit}]
                      [--prebuilt-tag PREBUILT_TAG]
                      [--prebuilt-repo PREBUILT_REPO] [--prebuilt-force]
                      [{all,env,toolchain,submodules,prebuilt}]
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `component` | no | `all` | `all, env, toolchain, submodules, prebuilt` |  |
| `--env-name` | no | `merlin-dev` | - | Conda environment name to update/install packages into (default: merlin-dev). |
| `--env-file` | no | `<repo>/env_linux.yml` | - | Conda environment file to use. Default is platform-specific: env_linux.yml |
| `--offline` | no | `False` | - | Run setup in offline mode when possible. |
| `--skip-conda` | no | `False` | - | Skip conda environment sync. |
| `--skip-pip` | no | `False` | - | Skip Python dependency sync (uv/pip). |
| `--python-deps` | no | `auto` | `auto, uv, pip` | Python dependency installer. 'auto' prefers uv sync with uv.lock and falls back to pip. |
| `--conda-no-plugins`, `--no-conda-no-plugins` | no | - | - | Force CONDA_NO_PLUGINS for conda env update. If unset, setup retries with CONDA_NO_PLUGINS=true on failure. |
| `--submodules-profile` | no | `core` | `core, npu, smolvla, full` | Which submodule profile to initialize (default: core). |
| `--submodule-path` | no | `[]` | - | Additional top-level submodule path to initialize (repeatable). |
| `--submodule-paths-recursive`, `--no-submodule-paths-recursive` | no | `False` | - | Whether extra --submodule-path entries should be initialized recursively. |
| `--submodule-depth` | no | `1` | - | Shallow depth for submodule fetches (default: 1). Use 0 for full history. |
| `--submodule-jobs` | no | `8` | - | Parallel submodule fetch jobs (default: 8). |
| `--submodule-sync` | no | `False` | - | Run `git submodule sync --recursive` before updating. |
| `--toolchain-target` | no | `spacemit` | `spacemit, firesim, all` | Which toolchain target to install (default: spacemit). |
| `--with-qemu` | no | `False` | - | For firesim toolchain setup, also install QEMU. |
| `--toolchain-force` | no | `False` | - | Reinstall toolchains even if the destination already exists. |
| `--prebuilt-artifact` | no | `host-linux-x86_64` | `host-linux-x86_64, host-macos, runtime-saturnopu, runtime-spacemit` | Which published Merlin prebuilt artifact to install. |
| `--prebuilt-tag` | no | `latest` | - | GitHub release tag to download from, or 'latest' (default: latest). |
| `--prebuilt-repo` | no | `ucb-bar/merlin` | - | GitHub repository containing release assets (default: ucb-bar/merlin). |
| `--prebuilt-force` | no | `False` | - | Replace an existing destination build tree when installing a prebuilt artifact. |

### `--help` Output

```text
usage: ./merlin setup [-h] [--env-name ENV_NAME] [--env-file ENV_FILE]
                      [--offline] [--skip-conda] [--skip-pip]
                      [--python-deps {auto,uv,pip}]
                      [--conda-no-plugins | --no-conda-no-plugins]
                      [--submodules-profile {core,npu,smolvla,full}]
                      [--submodule-path SUBMODULE_PATH]
                      [--submodule-paths-recursive | --no-submodule-paths-recursive]
                      [--submodule-depth SUBMODULE_DEPTH]
                      [--submodule-jobs SUBMODULE_JOBS] [--submodule-sync]
                      [--toolchain-target {spacemit,firesim,all}]
                      [--with-qemu] [--toolchain-force]
                      [--prebuilt-artifact {host-linux-x86_64,host-macos,runtime-saturnopu,runtime-spacemit}]
                      [--prebuilt-tag PREBUILT_TAG]
                      [--prebuilt-repo PREBUILT_REPO] [--prebuilt-force]
                      [{all,env,toolchain,submodules,prebuilt}]

Bootstrap developer environment and toolchains.

positional arguments:
  {all,env,toolchain,submodules,prebuilt}

options:
  -h, --help            show this help message and exit
  --env-name ENV_NAME   Conda environment name to update/install packages into
                        (default: merlin-dev).
  --env-file ENV_FILE   Conda environment file to use. Default is platform-
                        specific: env_linux.yml
  --offline             Run setup in offline mode when possible.
  --skip-conda          Skip conda environment sync.
  --skip-pip            Skip Python dependency sync (uv/pip).
  --python-deps {auto,uv,pip}
                        Python dependency installer. 'auto' prefers uv sync
                        with uv.lock and falls back to pip.
  --conda-no-plugins, --no-conda-no-plugins
                        Force CONDA_NO_PLUGINS for conda env update. If unset,
                        setup retries with CONDA_NO_PLUGINS=true on failure.
  --submodules-profile {core,npu,smolvla,full}
                        Which submodule profile to initialize (default: core).
  --submodule-path SUBMODULE_PATH
                        Additional top-level submodule path to initialize
                        (repeatable).
  --submodule-paths-recursive, --no-submodule-paths-recursive
                        Whether extra --submodule-path entries should be
                        initialized recursively.
  --submodule-depth SUBMODULE_DEPTH
                        Shallow depth for submodule fetches (default: 1). Use
                        0 for full history.
  --submodule-jobs SUBMODULE_JOBS
                        Parallel submodule fetch jobs (default: 8).
  --submodule-sync      Run `git submodule sync --recursive` before updating.
  --toolchain-target {spacemit,firesim,all}
                        Which toolchain target to install (default: spacemit).
  --with-qemu           For firesim toolchain setup, also install QEMU.
  --toolchain-force     Reinstall toolchains even if the destination already
                        exists.
  --prebuilt-artifact {host-linux-x86_64,host-macos,runtime-saturnopu,runtime-spacemit}
                        Which published Merlin prebuilt artifact to install.
  --prebuilt-tag PREBUILT_TAG
                        GitHub release tag to download from, or 'latest'
                        (default: latest).
  --prebuilt-repo PREBUILT_REPO
                        GitHub repository containing release assets (default:
                        ucb-bar/merlin).
  --prebuilt-force      Replace an existing destination build tree when
                        installing a prebuilt artifact.
```

## `./merlin ci`

Run repository CI / lint / patch workflows.

### Usage

```text
usage: ./merlin ci [-h] {lint,cli-docs-drift,patch-gate,release-status} ...
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `cli-docs-drift`

```text
usage: ./merlin ci cli-docs-drift [-h]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `lint`

```text
usage: ./merlin ci lint [-h]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `patch-gate`

```text
usage: ./merlin ci patch-gate [-h]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `release-status`

```text
usage: ./merlin ci release-status [-h] [--tracking-file TRACKING_FILE]
                                  [--offline] [--json]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--tracking-file` | no | `<repo>/.github/upstream_tracking.yaml` | - |  |
| `--offline` | no | `False` | - |  |
| `--json` | no | `False` | - |  |

### `--help` Output

```text
usage: ./merlin ci [-h] {lint,cli-docs-drift,patch-gate,release-status} ...

Run repository CI / lint / patch workflows.

positional arguments:
  {lint,cli-docs-drift,patch-gate,release-status}
    lint                Run linters (shellcheck, python)
    cli-docs-drift      Regenerate docs/reference/cli.md and fail on drift
    patch-gate          CI gate: apply, verify, drift check
    release-status      Check upstream IREE versions

options:
  -h, --help            show this help message and exit
```

## `./merlin patches`

Apply / verify / refresh / drift patch stack.

### Usage

```text
usage: ./merlin patches [-h] {verify,log,drift,export-upstream} ...
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `drift`

```text
usage: ./merlin patches drift [-h]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `export-upstream`

```text
usage: ./merlin patches export-upstream [-h] commit
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `commit` | yes | - | - | Commit hash to export |

#### Subcommand `log`

```text
usage: ./merlin patches log [-h]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `verify`

```text
usage: ./merlin patches verify [-h]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

### `--help` Output

```text
usage: ./merlin patches [-h] {verify,log,drift,export-upstream} ...

Apply / verify / refresh / drift patch stack.

positional arguments:
  {verify,log,drift,export-upstream}
    verify              Verify submodule is a clean rebase of pinned upstream
    log                 Show Merlin commits on top of upstream base
    drift               Check how far behind upstream the base is
    export-upstream     Export a commit as format-patch for upstream PR

options:
  -h, --help            show this help message and exit
```

## `./merlin benchmark`

Run benchmark helper scripts.

### Usage

```text
usage: ./merlin benchmark [-h] target {compile-dual-vmfb,run-dual-remote} ...
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `target` | yes | - | - | Target name from config/targets.json |
| `action` | yes | - | `compile-dual-vmfb, run-dual-remote` | Benchmark action |
| `extra_args` | yes | - | - |  |

### `--help` Output

```text
usage: ./merlin benchmark [-h] target {compile-dual-vmfb,run-dual-remote} ...

Run benchmark helper scripts.

positional arguments:
  target                Target name from config/targets.json
  {compile-dual-vmfb,run-dual-remote}
                        Benchmark action
  extra_args

options:
  -h, --help            show this help message and exit
```

## `./merlin chipyard`

Manage Chipyard hardware backend interactions.

### Usage

```text
usage: ./merlin chipyard [-h] [--chipyard-root CHIPYARD_ROOT]
                         {set-path,info,validate,checkout,build-sim,run,configure-firesim,build-bitstream,register-hwdb,stage-workload,stage-zephyr-workload,run-zephyr,run-radiance-muon,build-firemarshal,status}
                         ...
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--chipyard-root` | no | - | - | Override chipyard root for this invocation |

#### Subcommand `build-bitstream`

```text
usage: ./merlin chipyard build-bitstream [-h] recipe
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `recipe` | yes | - | - | Recipe name |

#### Subcommand `build-firemarshal`

```text
usage: ./merlin chipyard build-firemarshal [-h]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `build-sim`

```text
usage: ./merlin chipyard build-sim [-h] recipe
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `recipe` | yes | - | - | Recipe name |

#### Subcommand `checkout`

```text
usage: ./merlin chipyard checkout [-h] recipe
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `recipe` | yes | - | - | Recipe name (e.g., gemmini_mx, saturn_opu_u250) |

#### Subcommand `configure-firesim`

```text
usage: ./merlin chipyard configure-firesim [-h] recipe
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `recipe` | yes | - | - | Recipe name |

#### Subcommand `info`

```text
usage: ./merlin chipyard info [-h]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `register-hwdb`

```text
usage: ./merlin chipyard register-hwdb [-h] recipe
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `recipe` | yes | - | - | Recipe name |

#### Subcommand `run`

```text
usage: ./merlin chipyard run [-h] recipe binary
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `recipe` | yes | - | - | Recipe name |
| `binary` | yes | - | - | Path to bare-metal ELF |

#### Subcommand `run-radiance-muon`

```text
usage: ./merlin chipyard run-radiance-muon [-h] [--kernel KERNEL] recipe
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `recipe` | yes | - | - | Recipe name (bare_metal mode, e.g. radiance_muon) |
| `--kernel` | no | - | - | Path to a Muon kernel ELF (kernel.radiance.elf). Defaults to the single .radiance.elf in build/radiance_muon-vanilla-release/, then to the recipe's bare_metal.reference_binary. |

#### Subcommand `run-zephyr`

```text
usage: ./merlin chipyard run-zephyr [-h] [--elf ELF] recipe
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `recipe` | yes | - | - | Recipe name (workload.kind must be bare-metal-zephyr) |
| `--elf` | no | - | - | Override Zephyr ELF path (see stage-zephyr-workload) |

#### Subcommand `set-path`

```text
usage: ./merlin chipyard set-path [-h] path
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `path` | yes | - | - | Path to chipyard repository root |

#### Subcommand `stage-workload`

```text
usage: ./merlin chipyard stage-workload [-h] recipe [overlay_dir]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `recipe` | yes | - | - | Recipe name |
| `overlay_dir` | no | - | - | Directory to overlay into workload (default: build/firesim-merlin-release/install) |

#### Subcommand `stage-zephyr-workload`

```text
usage: ./merlin chipyard stage-zephyr-workload [-h] [--elf ELF] recipe
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `recipe` | yes | - | - | Recipe name (workload.kind must be bare-metal-zephyr) |
| `--elf` | no | - | - | Path to the Zephyr ELF to stage. Falls back to the recipe's firesim.workload.elf field, then to $ZEPHYR_BUILD_DIR/zephyr/zephyr.elf. |

#### Subcommand `status`

```text
usage: ./merlin chipyard status [-h] recipe
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `recipe` | yes | - | - | Recipe name |

#### Subcommand `validate`

```text
usage: ./merlin chipyard validate [-h] recipe
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `recipe` | yes | - | - | Recipe name (e.g., gemmini_mx, saturn_opu_u250) |

### `--help` Output

```text
usage: ./merlin chipyard [-h] [--chipyard-root CHIPYARD_ROOT]
                         {set-path,info,validate,checkout,build-sim,run,configure-firesim,build-bitstream,register-hwdb,stage-workload,stage-zephyr-workload,run-zephyr,run-radiance-muon,build-firemarshal,status}
                         ...

Manage Chipyard hardware backend interactions.

positional arguments:
  {set-path,info,validate,checkout,build-sim,run,configure-firesim,build-bitstream,register-hwdb,stage-workload,stage-zephyr-workload,run-zephyr,run-radiance-muon,build-firemarshal,status}
    set-path            Save chipyard workspace path
    info                Show chipyard state and available recipes
    validate            Validate chipyard checkout matches a recipe
    checkout            Switch chipyard branch and submodules to match a
                        recipe
    build-sim           Build VCS/Verilator RTL simulator
    run                 Run bare-metal ELF on simulator
    configure-firesim   Write FireSim deploy configs for a recipe
    build-bitstream     Build FireSim FPGA bitstream
    register-hwdb       Register built bitstream in FireSim HWDB
    stage-workload      Stage Merlin workload for FireSim
    stage-zephyr-workload
                        Stage a Zephyr ELF as a bare-metal FireSim workload
    run-zephyr          Stage Zephyr ELF + firesim infrasetup + firesim
                        runworkload
    run-radiance-muon   Run a Muon kernel ELF on a Radiance bare-metal sim
                        (RadianceMuonConfig etc.)
    build-firemarshal   Build FireMarshal base Linux image
    status              Check build/simulation status

options:
  -h, --help            show this help message and exit
  --chipyard-root CHIPYARD_ROOT
                        Override chipyard root for this invocation
```

## `./merlin targetgen`

Planner-first hardware target integration framework.

### Usage

```text
usage: ./merlin targetgen [-h]
                          {validate,plan,generate,explain,orchestrate,execute,stage-mutation,answer,status,ingest,classify,modification-map,mcp}
                          ...
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `answer`

```text
usage: ./merlin targetgen answer [-h] --target-dir TARGET_DIR --question-id
                                 QUESTION_ID --choice CHOICE
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--target-dir` | yes | - | - | Absolute or repo-relative path to a generated TargetGen target directory |
| `--question-id` | yes | - | - | Stable operator request id to answer |
| `--choice` | yes | - | - | Chosen option id from the operator request |

#### Subcommand `classify`

```text
usage: ./merlin targetgen classify [-h] --target-name TARGET_NAME
                                   [--from-dir FROM_DIR] [--out-dir OUT_DIR]
                                   [--dry-run]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--target-name` | yes | - | - | Logical target name; selects subdirectory under --from-dir / --out-dir |
| `--from-dir` | no | `build/generated/targetgen` | - | Directory containing source_inventory.json (default: build/generated/targetgen) |
| `--out-dir` | no | - | - | Override output directory (defaults to --from-dir) |
| `--dry-run` | no | `False` | - | Print summary without writing artifacts |

#### Subcommand `execute`

```text
usage: ./merlin targetgen execute [-h] [--overlay OVERLAY]
                                  [--from-dir FROM_DIR] [--out-dir OUT_DIR]
                                  [--prompt-backend {none,manualllm,provider}]
                                  [--agent AGENT] [--prompts-dir PROMPTS_DIR]
                                  [--resume] [--engine {local,ray}]
                                  [--ray-state-root RAY_STATE_ROOT]
                                  [capability]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `capability` | no | - | - | Path to a canonical TargetGen capability spec |
| `--overlay` | no | - | - | Optional deployment overlay that augments the capability spec |
| `--from-dir` | no | - | - | Existing target output directory to resume from |
| `--out-dir` | no | `build/generated/targetgen` | - | Base output directory for execution artifacts |
| `--prompt-backend` | no | `manualllm` | `none, manualllm, provider` | Prompt packet backend to use for execution |
| `--agent` | no | - | - | Optional mlirAgent provider config name to attach to prompt packets |
| `--prompts-dir` | no | - | - | Optional output directory for prompt_NNN.md packets; defaults to <out-dir>/<target>/prompts |
| `--resume` | no | `False` | - | Resume from existing execution_state.json when present |
| `--engine` | no | `local` | `local, ray` | Execution backend. `local` runs the in-process executor, `ray` submits the existing local executor as a Ray job. |
| `--ray-state-root` | no | `build/generated/ray` | - | Merlin-owned state directory for Ray cluster and run metadata when --engine ray is used. |

#### Subcommand `explain`

```text
usage: ./merlin targetgen explain [-h] [--overlay OVERLAY] capability
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `capability` | yes | - | - | Path to a canonical TargetGen capability spec |
| `--overlay` | no | - | - | Optional deployment overlay that augments the capability spec |

#### Subcommand `generate`

```text
usage: ./merlin targetgen generate [-h] [--overlay OVERLAY]
                                   [--out-dir OUT_DIR]
                                   capability
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `capability` | yes | - | - | Path to a canonical TargetGen capability spec |
| `--overlay` | no | - | - | Optional deployment overlay that augments the capability spec |
| `--out-dir` | no | `build/generated/targetgen` | - | Output directory for generated scaffold artifacts |

#### Subcommand `ingest`

```text
usage: ./merlin targetgen ingest [-h] --target-name TARGET_NAME --source
                                 SOURCE [--scanner SCANNER]
                                 [--out-dir OUT_DIR] [--dry-run]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--target-name` | yes | - | - | Logical target name used for the output subdirectory |
| `--source` | yes | - | - | Path to a target source tree (may be repeated) |
| `--scanner` | no | - | - | Restrict to a specific scanner (may be repeated). Defaults to all scanners. |
| `--out-dir` | no | `build/generated/targetgen` | - | Output directory for ingestion artifacts |
| `--dry-run` | no | `False` | - | Print summary without writing artifacts |

#### Subcommand `mcp`

```text
usage: ./merlin targetgen mcp [-h]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `modification-map`

```text
usage: ./merlin targetgen modification-map [-h] [--overlay OVERLAY]
                                           [--out-dir OUT_DIR] [--dry-run]
                                           capability
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `capability` | yes | - | - | Path to a canonical TargetGen capability spec |
| `--overlay` | no | - | - | Optional deployment overlay that augments the capability spec |
| `--out-dir` | no | `build/generated/targetgen` | - | Output directory for modification-map artifacts |
| `--dry-run` | no | `False` | - | Print summary without writing artifacts |

#### Subcommand `orchestrate`

```text
usage: ./merlin targetgen orchestrate [-h] [--overlay OVERLAY]
                                      [--out-dir OUT_DIR]
                                      [--prompt-backend {none,manualllm,provider}]
                                      [--agent AGENT]
                                      [--prompts-dir PROMPTS_DIR]
                                      capability
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `capability` | yes | - | - | Path to a canonical TargetGen capability spec |
| `--overlay` | no | - | - | Optional deployment overlay that augments the capability spec |
| `--out-dir` | no | `build/generated/targetgen` | - | Output directory for execution-bundle artifacts |
| `--prompt-backend` | no | `manualllm` | `none, manualllm, provider` | Prompt packet backend to prepare for orchestration output |
| `--agent` | no | - | - | Optional mlirAgent provider config name to attach to prompt packets |
| `--prompts-dir` | no | - | - | Optional output directory for prompt_NNN.md packets; defaults to <out-dir>/<target>/prompts |

#### Subcommand `plan`

```text
usage: ./merlin targetgen plan [-h] [--overlay OVERLAY] [--out-dir OUT_DIR]
                               capability
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `capability` | yes | - | - | Path to a canonical TargetGen capability spec |
| `--overlay` | no | - | - | Optional deployment overlay that augments the capability spec |
| `--out-dir` | no | `build/generated/targetgen` | - | Output directory for generated planner artifacts |

#### Subcommand `stage-mutation`

```text
usage: ./merlin targetgen stage-mutation [-h] [--overlay OVERLAY]
                                         [--from-dir FROM_DIR]
                                         [--out-dir OUT_DIR]
                                         [capability]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `capability` | no | - | - | Path to a canonical TargetGen capability spec |
| `--overlay` | no | - | - | Optional deployment overlay that augments the capability spec |
| `--from-dir` | no | - | - | Existing generated TargetGen target directory with inputs snapshots |
| `--out-dir` | no | `build/generated/targetgen` | - | Base output directory for generation and mutation artifacts |

#### Subcommand `status`

```text
usage: ./merlin targetgen status [-h] --target-dir TARGET_DIR
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--target-dir` | yes | - | - | Absolute or repo-relative path to a generated TargetGen target directory |

#### Subcommand `validate`

```text
usage: ./merlin targetgen validate [-h] [--overlay OVERLAY] capability
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `capability` | yes | - | - | Path to a canonical TargetGen capability spec |
| `--overlay` | no | - | - | Optional deployment overlay that augments the capability spec |

### `--help` Output

```text
usage: ./merlin targetgen [-h]
                          {validate,plan,generate,explain,orchestrate,execute,stage-mutation,answer,status,ingest,classify,modification-map,mcp}
                          ...

Planner-first hardware target integration framework.

positional arguments:
  {validate,plan,generate,explain,orchestrate,execute,stage-mutation,answer,status,ingest,classify,modification-map,mcp}
    validate            Validate TargetGen capability specs and overlays
    plan                Emit support-plan and task-graph artifacts
    generate            Emit non-live scaffold files under
                        build/generated/targetgen without touching repo-
                        tracked sources
    explain             Print a human-readable TargetGen explanation
    orchestrate         Emit an execution bundle and LLM-oriented task briefs
                        from the task graph
    execute             Advance execution state, emit prompts, ingest
                        responses, and stop on operator gates
    stage-mutation      Stage a proposed mutation tree under
                        build/generated/targetgen without applying repo-
                        tracked edits
    answer              Record an operator choice for an open executor request
    status              Show executor task states and open operator requests
    ingest              Scan an external target source tree and emit a
                        SourceInventory
    classify            Classify a previously ingested SourceInventory into
                        integration styles
    modification-map    Emit a per-stage patch-surface modification map for a
                        capability spec
    mcp                 Launch the TargetGen MCP server over stdio for Claude
                        Code

options:
  -h, --help            show this help message and exit
```

## `./merlin ray`

Ray-based execution engine for targetgen workflows.

### Usage

```text
usage: ./merlin ray [-h] [--state-root STATE_ROOT]
                    {cluster,jobs,resources,artifacts} ...
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--state-root` | no | `build/generated/ray` | - | Directory for Merlin-owned Ray cluster, run, artifact, and lease metadata. |

#### Subcommand `artifacts`

```text
usage: ./merlin ray artifacts [-h] {list,fetch} ...
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `cluster`

```text
usage: ./merlin ray cluster [-h] {start-local,status,stop} ...
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `jobs`

```text
usage: ./merlin ray jobs [-h] {submit,status,logs,cancel} ...
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `resources`

```text
usage: ./merlin ray resources [-h] {list,reserve,release} ...
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

### `--help` Output

```text
usage: ./merlin ray [-h] [--state-root STATE_ROOT]
                    {cluster,jobs,resources,artifacts} ...

Ray-based execution engine for targetgen workflows.

positional arguments:
  {cluster,jobs,resources,artifacts}
    cluster             Manage the local Ray cluster bootstrap for Merlin
    jobs                Submit and inspect Ray-backed Merlin jobs
    resources           Manage Merlin resource leases
    artifacts           Inspect artifacts captured for Merlin Ray runs

options:
  -h, --help            show this help message and exit
  --state-root STATE_ROOT
                        Directory for Merlin-owned Ray cluster, run, artifact,
                        and lease metadata.
```

## `./merlin run`

Run compiled models on local or board devices.

### Usage

```text
usage: ./merlin run [-h]
                    {full-loop,het-e2e,het-matrix,multi-device,roundtrip,schedule}
                    ...
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `mode` | yes | - | `full-loop, het-e2e, het-matrix, multi-device, roundtrip, schedule` | Which board-execution flow to run. See module docstrings under tools/run/ for per-mode flags. |
| `passthrough` | yes | - | - | Arguments forwarded verbatim to the underlying script. |

### `--help` Output

```text
usage: ./merlin run [-h]
                    {full-loop,het-e2e,het-matrix,multi-device,roundtrip,schedule}
                    ...

Run compiled models on local or board devices.

positional arguments:
  {full-loop,het-e2e,het-matrix,multi-device,roundtrip,schedule}
                        Which board-execution flow to run. See module
                        docstrings under tools/run/ for per-mode flags.
  passthrough           Arguments forwarded verbatim to the underlying script.

options:
  -h, --help            show this help message and exit
```

## `./merlin perf-decompose`

Per-dispatch performance decomposition.

### Usage

```text
usage: ./merlin perf-decompose [-h] [--topk TOPK] [--csv CSV] uartlog
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `uartlog` | yes | - | - | FireSim uartlog file |
| `--topk` | no | `20` | - | Print top-K hot dispatches (default 20) |
| `--csv` | no | - | - | Also write a CSV summary |

### `--help` Output

```text
usage: ./merlin perf-decompose [-h] [--topk TOPK] [--csv CSV] uartlog

Per-dispatch performance decomposition.

positional arguments:
  uartlog      FireSim uartlog file

options:
  -h, --help   show this help message and exit
  --topk TOPK  Print top-K hot dispatches (default 20)
  --csv CSV    Also write a CSV summary
```

## `./merlin verify-output`

Verify model outputs against golden references.

### Usage

```text
usage: ./merlin verify-output [-h] --shape SHAPE [--observed OBSERVED]
                              [--uartlog UARTLOG] [--seed SEED]
                              [--random-input] [--skip-golden]
                              model
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `model` | yes | - | - | Quantized .q.int8.onnx model |
| `--shape` | yes | - | - | Input shape comma-separated (repeat per input) |
| `--observed` | no | `[]` | - | Backend hash to verify: <hex_hash>:<label> (e.g. 0x498...:gemmini) |
| `--uartlog` | no | `[]` | - | FireSim uartlog file to extract hashes from |
| `--seed` | no | `51966` | - | RNG seed for x86 reference input when --random-input (default 0xCAFE) |
| `--random-input` | no | `False` | - | Use random input instead of all-zero (the runner uses zeros via ZERO_FILL buffer alloc). Use this only for sanity checks. |
| `--skip-golden` | no | `False` | - | Skip the onnxruntime baseline (just cross-check observed hashes) |

### `--help` Output

```text
usage: ./merlin verify-output [-h] --shape SHAPE [--observed OBSERVED]
                              [--uartlog UARTLOG] [--seed SEED]
                              [--random-input] [--skip-golden]
                              model

Verify model outputs against golden references.

positional arguments:
  model                Quantized .q.int8.onnx model

options:
  -h, --help           show this help message and exit
  --shape SHAPE        Input shape comma-separated (repeat per input)
  --observed OBSERVED  Backend hash to verify: <hex_hash>:<label> (e.g.
                       0x498...:gemmini)
  --uartlog UARTLOG    FireSim uartlog file to extract hashes from
  --seed SEED          RNG seed for x86 reference input when --random-input
                       (default 0xCAFE)
  --random-input       Use random input instead of all-zero (the runner uses
                       zeros via ZERO_FILL buffer alloc). Use this only for
                       sanity checks.
  --skip-golden        Skip the onnxruntime baseline (just cross-check
                       observed hashes)
```

## `./merlin coverage-check`

Kernel-embedding coverage check.

### Usage

```text
usage: ./merlin coverage-check [-h] [--csv CSV] vmfb
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `vmfb` | yes | - | - | .vmfb file to inspect |
| `--csv` | no | - | - | Write per-function CSV |

### `--help` Output

```text
usage: ./merlin coverage-check [-h] [--csv CSV] vmfb

Kernel-embedding coverage check.

positional arguments:
  vmfb        .vmfb file to inspect

options:
  -h, --help  show this help message and exit
  --csv CSV   Write per-function CSV
```

## `./merlin quantize`

Quantize models to int8 and analyze.

### Usage

```text
usage: ./merlin quantize [-h] --shape SHAPE [--output OUTPUT]
                         [--calibration-samples CALIBRATION_SAMPLES]
                         input_onnx
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `input_onnx` | yes | - | - | Source .onnx file |
| `--shape` | yes | - | - | Input tensor shape as comma-separated integers (e.g. 1,3,224,224). Repeat once per input for multi-input models. |
| `--output` | no | - | - | Output path (default: <input>.q.int8.onnx alongside the input). |
| `--calibration-samples` | no | `50` | - | Number of random calibration samples (default 50). |

### `--help` Output

```text
usage: ./merlin quantize [-h] --shape SHAPE [--output OUTPUT]
                         [--calibration-samples CALIBRATION_SAMPLES]
                         input_onnx

Quantize models to int8 and analyze.

positional arguments:
  input_onnx            Source .onnx file

options:
  -h, --help            show this help message and exit
  --shape SHAPE         Input tensor shape as comma-separated integers (e.g.
                        1,3,224,224). Repeat once per input for multi-input
                        models.
  --output OUTPUT       Output path (default: <input>.q.int8.onnx alongside
                        the input).
  --calibration-samples CALIBRATION_SAMPLES
                        Number of random calibration samples (default 50).
```

## `./merlin sim`

RTL simulator orchestration.

### Usage

```text
usage: ./merlin sim [-h] [--target TARGET] [--bench-target BENCH_TARGET]
                    [--simulator {vcs,verilator}] [--config CONFIG]
                    [--reference REFERENCE] [--output-dir OUTPUT_DIR] [--keep]
                    [--build-dir BUILD_DIR]
                    [--firesim-build-dir FIRESIM_BUILD_DIR] [--skip-build]
                    [--skip-compile] [--timeout TIMEOUT] [-v]
                    input
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `input` | yes | - | - | Input .mlir fixture (any model — see tests/integration/<target>/fixtures/ for examples). |
| `--target` | no | `gemmini_mx_vcs` | - | Model YAML target (default: gemmini_mx_vcs). Any models/<target>.yaml is accepted; pass --bench-target if the cmake bench target name does not match the default mapping. |
| `--bench-target` | no | - | - | Explicit cmake bench target name. Overrides the default mapping in _TARGET_TO_BENCH; required when --target is not a key in the default mapping. |
| `--simulator` | no | `vcs` | `vcs, verilator` | Chipyard simulator backend (default: vcs) |
| `--config` | no | `RadianceGemminiOnlyConfig` | - | Chipyard CONFIG (default: RadianceGemminiOnlyConfig) |
| `--reference` | no | - | - | Path to expected output (one i32 per line). If set, run outputs are diffed against this and an exit code of 0/1 is returned. |
| `--output-dir` | no | - | - | Directory for produced artifacts (default: /scratch2/agustin/merlin/build/sim/<fixture>) |
| `--keep` | no | `False` | - | Keep simulator working dir + log on success |
| `--build-dir` | no | `host-merlin-release` | - | Host build dir for iree-compile (default: host-merlin-release) |
| `--firesim-build-dir` | no | `firesim-merlin-release` | - | Firesim build dir holding the bench ELF (default: firesim-merlin-release) |
| `--skip-build` | no | `False` | - | Skip ./merlin build step (use a pre-built ELF) |
| `--skip-compile` | no | `False` | - | Skip ./merlin compile step (use a pre-built VMFB) |
| `--timeout` | no | `900` | - | Simulator wallclock timeout in seconds (default 900) |
| `-v`, `--verbose` | no | `False` | - | Verbose logging |

### `--help` Output

```text
usage: ./merlin sim [-h] [--target TARGET] [--bench-target BENCH_TARGET]
                    [--simulator {vcs,verilator}] [--config CONFIG]
                    [--reference REFERENCE] [--output-dir OUTPUT_DIR] [--keep]
                    [--build-dir BUILD_DIR]
                    [--firesim-build-dir FIRESIM_BUILD_DIR] [--skip-build]
                    [--skip-compile] [--timeout TIMEOUT] [-v]
                    input

RTL simulator orchestration.

positional arguments:
  input                 Input .mlir fixture (any model — see
                        tests/integration/<target>/fixtures/ for examples).

options:
  -h, --help            show this help message and exit
  --target TARGET       Model YAML target (default: gemmini_mx_vcs). Any
                        models/<target>.yaml is accepted; pass --bench-target
                        if the cmake bench target name does not match the
                        default mapping.
  --bench-target BENCH_TARGET
                        Explicit cmake bench target name. Overrides the
                        default mapping in _TARGET_TO_BENCH; required when
                        --target is not a key in the default mapping.
  --simulator {vcs,verilator}
                        Chipyard simulator backend (default: vcs)
  --config CONFIG       Chipyard CONFIG (default: RadianceGemminiOnlyConfig)
  --reference REFERENCE
                        Path to expected output (one i32 per line). If set,
                        run outputs are diffed against this and an exit code
                        of 0/1 is returned.
  --output-dir OUTPUT_DIR
                        Directory for produced artifacts (default:
                        /scratch2/agustin/merlin/build/sim/<fixture>)
  --keep                Keep simulator working dir + log on success
  --build-dir BUILD_DIR
                        Host build dir for iree-compile (default: host-merlin-
                        release)
  --firesim-build-dir FIRESIM_BUILD_DIR
                        Firesim build dir holding the bench ELF (default:
                        firesim-merlin-release)
  --skip-build          Skip ./merlin build step (use a pre-built ELF)
  --skip-compile        Skip ./merlin compile step (use a pre-built VMFB)
  --timeout TIMEOUT     Simulator wallclock timeout in seconds (default 900)
  -v, --verbose         Verbose logging
```

## `./merlin spike`

Spike RISC-V ISA simulator runner.

### Usage

```text
usage: ./merlin spike [-h] [--output-dir OUTPUT_DIR] [--target TARGET]
                      [--build-dir BUILD_DIR] [-v]
                      input
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `input` | yes | - | - | Input .mlir fixture (tensor-domain) |
| `--output-dir` | no | - | - | Directory for the produced .vmfb (default: build/spike/<basename>) |
| `--target` | no | `gemmini_spike` | - | Model YAML name (default: gemmini_spike) |
| `--build-dir` | no | `host-merlin-debug` | - | Which build dir to use for iree-compile (default: host-merlin-debug) |
| `-v`, `--verbose` | no | `False` | - | Verbose logging |

### `--help` Output

```text
usage: ./merlin spike [-h] [--output-dir OUTPUT_DIR] [--target TARGET]
                      [--build-dir BUILD_DIR] [-v]
                      input

Spike RISC-V ISA simulator runner.

positional arguments:
  input                 Input .mlir fixture (tensor-domain)

options:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
                        Directory for the produced .vmfb (default:
                        build/spike/<basename>)
  --target TARGET       Model YAML name (default: gemmini_spike)
  --build-dir BUILD_DIR
                        Which build dir to use for iree-compile (default:
                        host-merlin-debug)
  -v, --verbose         Verbose logging
```

## `./merlin mcp`

MCP (Model Context Protocol) server dispatcher.

### Usage

```text
usage: ./merlin mcp [-h] {build,compile,run,perf,verify,targetgen}
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `name` | yes | - | `build, compile, run, perf, verify, targetgen` | Which MCP server to start. Each name N corresponds to tools/mcp_servers/N.py. |

### `--help` Output

```text
usage: ./merlin mcp [-h] {build,compile,run,perf,verify,targetgen}

MCP (Model Context Protocol) server dispatcher.

positional arguments:
  {build,compile,run,perf,verify,targetgen}
                        Which MCP server to start. Each name N corresponds to
                        tools/mcp_servers/N.py.

options:
  -h, --help            show this help message and exit
```
