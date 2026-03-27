# CLI Reference

This page is generated from real argparse parsers in `tools/*.py`.

Each command is shown with argument introspection and raw `--help` output.

## `tools/merlin.py`

Unified Merlin developer command reference parser.

### Usage

```text
usage: uv run tools/merlin.py [-h]
                              {build,compile,setup,ci,patches,benchmark,chipyard,ray,targetgen}
                              ...
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `targetgen`

`targetgen` is the planner and orchestration entrypoint for capability-spec
driven target enablement.

Key actions:

- `validate`: schema-check a capability spec and optional deployment overlay
- `plan`: emit `support_plan.json`, `task_graph.json`, and derived compile or
  deployment views
- `generate`: emit non-live scaffold files under `build/generated/targetgen`
  at prospective repo paths without touching repo-tracked sources
- `explain`: print the normalized classification and verification ladder
- `orchestrate`: emit `execution_bundle.json`, `execution_state.json`,
  per-task state files, sectioned `briefs/*.md`, and optional prompt packets
- `execute`: advance executor state, run safe preflight checks, emit prompts,
  ingest `.response.md` files, and stop on operator or mutation gates
- `stage-mutation`: assemble the mutating subset into `mutation/proposed_tree/`
  and emit a branch/worktree plan without applying any repo edits
- `answer`: record an explicit choice for an open operator request
- `status`: print current task state plus any open or resolved operator requests

Useful `orchestrate` flags:

- `--prompt-backend none|manualllm|provider`
- `--agent <config-name>`
- `--prompts-dir <path>`

Useful `execute` flags:

- `--from-dir <target-output-dir>`
- `--resume`
- `--engine local|ray`
- `--ray-state-root <path>`
- `--prompt-backend none|manualllm|provider`
- `--agent <config-name>`

Examples:

```bash
uv run tools/merlin.py targetgen plan target_specs/examples/saturn_opu_v128/capability.yaml \
  --overlay target_specs/examples/saturn_opu_v128/overlays/firesim_u250.yaml

uv run tools/merlin.py targetgen generate \
  target_specs/examples/gemmini_mx/capability.yaml \
  --overlay target_specs/examples/gemmini_mx/overlays/baremetal_local.yaml

uv run tools/merlin.py targetgen orchestrate \
  target_specs/examples/npu_ucb/capability.yaml \
  --overlay target_specs/examples/npu_ucb/overlays/simulator_local.yaml \
  --prompt-backend manualllm

uv run tools/merlin.py targetgen execute \
  target_specs/examples/nvidia_vulkan_ada/capability.yaml \
  --overlay target_specs/examples/nvidia_vulkan_ada/overlays/desktop_local.yaml \
  --engine ray

uv run tools/merlin.py targetgen answer \
  --target-dir build/generated/targetgen/nvidia_vulkan_ada \
  --question-id implement_runtime_hal-device \
  --choice device_available

uv run tools/merlin.py targetgen stage-mutation \
  --from-dir build/generated/targetgen/gemmini_mx

uv run tools/merlin.py targetgen status \
  --target-dir build/generated/targetgen/nvidia_vulkan_ada
```

#### Subcommand `ray`

`ray` is Merlin's fixed-cluster control plane for Ray-backed runs, leases, and
artifact discovery.

Key actions:

- `cluster start-local|status|stop`
- `jobs submit|status|logs|cancel`
- `resources list|reserve|release`
- `artifacts list|fetch`

Examples:

```bash
uv run tools/merlin.py ray cluster start-local

uv run tools/merlin.py ray jobs status <run-id>

uv run tools/merlin.py ray resources reserve firesim_u250 --owner nightly-smoke
```

#### Subcommand `benchmark`

```text
usage: uv run tools/merlin.py benchmark [-h] [--dry-run]
                                        target
                                        {compile-dual-vmfb,run-dual-remote}
                                        ...
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--dry-run` | no | `False` | - | Print commands without executing |
| `target` | yes | - | - | Target name from config/targets.json |
| `action` | yes | - | `compile-dual-vmfb, run-dual-remote` | Benchmark action |
| `extra_args` | yes | - | - |  |

#### Subcommand `build`

```text
usage: uv run tools/merlin.py build [-h] [--dry-run]
                                    [--profile {firesim,full-plugin,gemmini,npu,package-firesim,package-host,package-spacemit,radiance,spacemit,vanilla}]
                                    [--target {host,spacemit,firesim}]
                                    [--config {debug,release,asan,trace,perf}]
                                    [--cmake-target CMAKE_TARGET]
                                    [--with-plugin]
                                    [--plugin-compiler | --no-plugin-compiler]
                                    [--plugin-runtime | --no-plugin-runtime]
                                    [--plugin-runtime-radiance | --no-plugin-runtime-radiance]
                                    [--plugin-runtime-samples | --no-plugin-runtime-samples]
                                    [--plugin-runtime-benchmarks | --no-plugin-runtime-benchmarks]
                                    [--plugin-runtime-radiance-tests | --no-plugin-runtime-radiance-tests]
                                    [--plugin-runtime-radiance-rpc | --no-plugin-runtime-radiance-rpc]
                                    [--plugin-runtime-radiance-direct | --no-plugin-runtime-radiance-direct]
                                    [--plugin-runtime-radiance-kmod | --no-plugin-runtime-radiance-kmod]
                                    [--compiler-scope {all,gemmini,npu,saturn,spacemit,none}]
                                    [--build-compiler | --no-build-compiler]
                                    [--build-python-bindings | --no-build-python-bindings]
                                    [--build-samples | --no-build-samples]
                                    [--build-tests | --no-build-tests]
                                    [--enable-libbacktrace | --no-enable-libbacktrace]
                                    [--enable-tracy] [--offline-friendly]
                                    [--cmake-bin CMAKE_BIN]
                                    [--use-system-cmake]
                                    [--use-ccache | --no-use-ccache]
                                    [--cmake-arg CMAKE_ARG]
                                    [--cmake-build-arg CMAKE_BUILD_ARG]
                                    [--native-build-arg NATIVE_BUILD_ARG]
                                    [--clean] [--verbose]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--dry-run` | no | `False` | - | Print commands without executing |
| `--profile` | no | - | `firesim, full-plugin, gemmini, npu, package-firesim, package-host, package-spacemit, radiance, spacemit, vanilla` | High-level user profile preset. Use this for normal workflows; advanced flags may still override details. |
| `--target` | no | - | `host, spacemit, firesim` | Target platform. |
| `--config` | no | - | `debug, release, asan, trace, perf` | Build configuration type |
| `--cmake-target` | no | - | - | Build specific CMake target (default: install) |
| `--with-plugin` | no | `False` | - | Enable Merlin compiler+runtime plugins (legacy umbrella switch). |
| `--plugin-compiler`, `--no-plugin-compiler` | no | - | - | Enable/disable Merlin compiler plugin targets (default follows --with-plugin). |
| `--plugin-runtime`, `--no-plugin-runtime` | no | - | - | Enable/disable Merlin runtime plugin integration (default follows --with-plugin). |
| `--plugin-runtime-radiance`, `--no-plugin-runtime-radiance` | no | - | - | Enable/disable Radiance HAL runtime plugin path (default: host+plugin only). |
| `--plugin-runtime-samples`, `--no-plugin-runtime-samples` | no | - | - | Enable/disable runtime plugin samples subdir. |
| `--plugin-runtime-benchmarks`, `--no-plugin-runtime-benchmarks` | no | - | - | Enable/disable runtime plugin benchmarks subdir. |
| `--plugin-runtime-radiance-tests`, `--no-plugin-runtime-radiance-tests` | no | - | - | Enable/disable Radiance runtime plugin tests. |
| `--plugin-runtime-radiance-rpc`, `--no-plugin-runtime-radiance-rpc` | no | - | - | Enable/disable Radiance RPC-compat transport backend. |
| `--plugin-runtime-radiance-direct`, `--no-plugin-runtime-radiance-direct` | no | - | - | Enable/disable Radiance direct-submit transport backend. |
| `--plugin-runtime-radiance-kmod`, `--no-plugin-runtime-radiance-kmod` | no | - | - | Enable/disable Radiance kmod transport backend. |
| `--compiler-scope` | no | - | `all, gemmini, npu, saturn, spacemit, none` | Limit compiler-plugin target registration scope. Only used when compiler plugin + compiler build are enabled. |
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
usage: uv run tools/merlin.py chipyard [-h] [--dry-run]
                                       [--chipyard-root CHIPYARD_ROOT]
                                       {set-path,info,validate,build-sim,run,configure-firesim,build-bitstream,register-hwdb,stage-workload,build-firemarshal,status}
                                       ...
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--dry-run` | no | `False` | - | Print commands without executing |
| `--chipyard-root` | no | - | - | Override chipyard root for this invocation |

#### Subcommand `ci`

```text
usage: uv run tools/merlin.py ci [-h] [--dry-run]
                                 {lint,cli-docs-drift,patch-gate,release-status}
                                 ...
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--dry-run` | no | `False` | - | Print commands without executing |

#### Subcommand `compile`

```text
usage: uv run tools/merlin.py compile [-h] [--dry-run] --target TARGET
                                      [--hw HW] [--quantized]
                                      [--output-dir OUTPUT_DIR]
                                      [--build-dir BUILD_DIR]
                                      [--compile-to COMPILE_TO]
                                      [--dump-compilation-phases-to DUMP_COMPILATION_PHASES_TO]
                                      [--iree-compile-arg IREE_COMPILE_ARG]
                                      [--reuse-imported-mlir] [--tracy]
                                      [--dump-artifacts] [--dump-phases]
                                      [--dump-graph] [--build-benchmarks]
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
| `--build-benchmarks` | no | `False` | - | Recompile individual dispatch benchmarks and zip them |

#### Subcommand `patches`

```text
usage: uv run tools/merlin.py patches [-h]
                                      {verify,log,drift,export-upstream} ...
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `setup`

```text
usage: uv run tools/merlin.py setup [-h] [--env-name ENV_NAME]
                                    [--env-file ENV_FILE] [--offline]
                                    [--skip-conda] [--skip-pip]
                                    [--python-deps {auto,uv,pip}]
                                    [--conda-no-plugins | --no-conda-no-plugins]
                                    [--submodules-profile {core,npu,smolvla,full}]
                                    [--submodule-path SUBMODULE_PATH]
                                    [--submodule-paths-recursive | --no-submodule-paths-recursive]
                                    [--submodule-depth SUBMODULE_DEPTH]
                                    [--submodule-jobs SUBMODULE_JOBS]
                                    [--submodule-sync]
                                    [--toolchain-target {spacemit,firesim,all}]
                                    [--with-qemu] [--toolchain-force]
                                    [--prebuilt-artifact {host-linux-x86_64,host-macos,runtime-spacemit,runtime-saturnopu}]
                                    [--prebuilt-tag PREBUILT_TAG]
                                    [--prebuilt-repo PREBUILT_REPO]
                                    [--prebuilt-force]
                                    [{all,env,toolchain,submodules,prebuilt}]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `component` | no | `all` | `all, env, toolchain, submodules, prebuilt` |  |
| `--env-name` | no | `merlin-dev` | - | Conda environment name to update/install packages into (default: merlin-dev). |
| `--env-file` | no | `/scratch2/agustin/merlin/env_linux.yml` | - | Conda environment file to use. Default is platform-specific: env_linux.yml |
| `--offline` | no | `False` | - | Run setup in offline mode when possible. |
| `--skip-conda` | no | `False` | - | Skip conda environment sync. |
| `--skip-pip` | no | `False` | - | Skip Python dependency sync (uv/pip). |
| `--python-deps` | no | `auto` | `auto, uv, pip` | Python dependency installer. 'auto' prefers uv sync with uv.lock and falls back to pip requirements. |
| `--conda-no-plugins`, `--no-conda-no-plugins` | no | - | - | Force CONDA_NO_PLUGINS for conda env update. If unset, setup.py retries with CONDA_NO_PLUGINS=true on failure. |
| `--submodules-profile` | no | `core` | `core, npu, smolvla, full` | Which submodule profile to initialize (default: core). |
| `--submodule-path` | no | `[]` | - | Additional top-level submodule path to initialize (repeatable). |
| `--submodule-paths-recursive`, `--no-submodule-paths-recursive` | no | `False` | - | Whether extra --submodule-path entries should be initialized recursively. |
| `--submodule-depth` | no | `1` | - | Shallow depth for submodule fetches (default: 1). Use 0 for full history. |
| `--submodule-jobs` | no | `8` | - | Parallel submodule fetch jobs (default: 8). |
| `--submodule-sync` | no | `False` | - | Run `git submodule sync --recursive` before updating. |
| `--toolchain-target` | no | `spacemit` | `spacemit, firesim, all` | Which toolchain target to install (default: spacemit). |
| `--with-qemu` | no | `False` | - | For firesim toolchain setup, also install QEMU. |
| `--toolchain-force` | no | `False` | - | Reinstall toolchains even if the destination already exists. |
| `--prebuilt-artifact` | no | `host-linux-x86_64` | `host-linux-x86_64, host-macos, runtime-spacemit, runtime-saturnopu` | Which published Merlin prebuilt artifact to install. |
| `--prebuilt-tag` | no | `latest` | - | GitHub release tag to download from, or 'latest' (default: latest). |
| `--prebuilt-repo` | no | `ucb-bar/merlin` | - | GitHub repository containing release assets (default: ucb-bar/merlin). |
| `--prebuilt-force` | no | `False` | - | Replace an existing destination build tree when installing a prebuilt artifact. |

### `--help` Output

```text
usage: uv run tools/merlin.py [-h]
                              {build,compile,setup,ci,patches,benchmark,chipyard,ray,targetgen}
                              ...

Unified Merlin developer command reference parser.

positional arguments:
  {build,compile,setup,ci,patches,benchmark,chipyard,ray,targetgen}
    build               Configure and build Merlin and target runtimes
    compile             Compile MLIR/ONNX models to target artifacts
    setup               Bootstrap developer environment and toolchains
    ci                  Run repository CI/lint/patch workflows
    patches             Verify submodule state and manage upstream patches
    benchmark           Run benchmark helper scripts
    chipyard            Manage Chipyard hardware backend interactions

options:
  -h, --help            show this help message and exit
```

## `tools/build.py`

Configure and build Merlin and target runtimes

### Usage

```text
usage: uv run tools/build.py [-h]
                             [--profile {firesim,full-plugin,gemmini,npu,package-firesim,package-host,package-spacemit,radiance,spacemit,vanilla}]
                             [--target {host,spacemit,firesim}]
                             [--config {debug,release,asan,trace,perf}]
                             [--cmake-target CMAKE_TARGET] [--with-plugin]
                             [--plugin-compiler | --no-plugin-compiler]
                             [--plugin-runtime | --no-plugin-runtime]
                             [--plugin-runtime-radiance | --no-plugin-runtime-radiance]
                             [--plugin-runtime-samples | --no-plugin-runtime-samples]
                             [--plugin-runtime-benchmarks | --no-plugin-runtime-benchmarks]
                             [--plugin-runtime-radiance-tests | --no-plugin-runtime-radiance-tests]
                             [--plugin-runtime-radiance-rpc | --no-plugin-runtime-radiance-rpc]
                             [--plugin-runtime-radiance-direct | --no-plugin-runtime-radiance-direct]
                             [--plugin-runtime-radiance-kmod | --no-plugin-runtime-radiance-kmod]
                             [--compiler-scope {all,gemmini,npu,saturn,spacemit,none}]
                             [--build-compiler | --no-build-compiler]
                             [--build-python-bindings | --no-build-python-bindings]
                             [--build-samples | --no-build-samples]
                             [--build-tests | --no-build-tests]
                             [--enable-libbacktrace | --no-enable-libbacktrace]
                             [--enable-tracy] [--offline-friendly]
                             [--cmake-bin CMAKE_BIN] [--use-system-cmake]
                             [--use-ccache | --no-use-ccache]
                             [--cmake-arg CMAKE_ARG]
                             [--cmake-build-arg CMAKE_BUILD_ARG]
                             [--native-build-arg NATIVE_BUILD_ARG] [--clean]
                             [--verbose]
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--profile` | no | - | `firesim, full-plugin, gemmini, npu, package-firesim, package-host, package-spacemit, radiance, spacemit, vanilla` | High-level user profile preset. Use this for normal workflows; advanced flags may still override details. |
| `--target` | no | - | `host, spacemit, firesim` | Target platform. |
| `--config` | no | - | `debug, release, asan, trace, perf` | Build configuration type |
| `--cmake-target` | no | - | - | Build specific CMake target (default: install) |
| `--with-plugin` | no | `False` | - | Enable Merlin compiler+runtime plugins (legacy umbrella switch). |
| `--plugin-compiler`, `--no-plugin-compiler` | no | - | - | Enable/disable Merlin compiler plugin targets (default follows --with-plugin). |
| `--plugin-runtime`, `--no-plugin-runtime` | no | - | - | Enable/disable Merlin runtime plugin integration (default follows --with-plugin). |
| `--plugin-runtime-radiance`, `--no-plugin-runtime-radiance` | no | - | - | Enable/disable Radiance HAL runtime plugin path (default: host+plugin only). |
| `--plugin-runtime-samples`, `--no-plugin-runtime-samples` | no | - | - | Enable/disable runtime plugin samples subdir. |
| `--plugin-runtime-benchmarks`, `--no-plugin-runtime-benchmarks` | no | - | - | Enable/disable runtime plugin benchmarks subdir. |
| `--plugin-runtime-radiance-tests`, `--no-plugin-runtime-radiance-tests` | no | - | - | Enable/disable Radiance runtime plugin tests. |
| `--plugin-runtime-radiance-rpc`, `--no-plugin-runtime-radiance-rpc` | no | - | - | Enable/disable Radiance RPC-compat transport backend. |
| `--plugin-runtime-radiance-direct`, `--no-plugin-runtime-radiance-direct` | no | - | - | Enable/disable Radiance direct-submit transport backend. |
| `--plugin-runtime-radiance-kmod`, `--no-plugin-runtime-radiance-kmod` | no | - | - | Enable/disable Radiance kmod transport backend. |
| `--compiler-scope` | no | - | `all, gemmini, npu, saturn, spacemit, none` | Limit compiler-plugin target registration scope. Only used when compiler plugin + compiler build are enabled. |
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
usage: uv run tools/build.py [-h]
                             [--profile {firesim,full-plugin,gemmini,npu,package-firesim,package-host,package-spacemit,radiance,spacemit,vanilla}]
                             [--target {host,spacemit,firesim}]
                             [--config {debug,release,asan,trace,perf}]
                             [--cmake-target CMAKE_TARGET] [--with-plugin]
                             [--plugin-compiler | --no-plugin-compiler]
                             [--plugin-runtime | --no-plugin-runtime]
                             [--plugin-runtime-radiance | --no-plugin-runtime-radiance]
                             [--plugin-runtime-samples | --no-plugin-runtime-samples]
                             [--plugin-runtime-benchmarks | --no-plugin-runtime-benchmarks]
                             [--plugin-runtime-radiance-tests | --no-plugin-runtime-radiance-tests]
                             [--plugin-runtime-radiance-rpc | --no-plugin-runtime-radiance-rpc]
                             [--plugin-runtime-radiance-direct | --no-plugin-runtime-radiance-direct]
                             [--plugin-runtime-radiance-kmod | --no-plugin-runtime-radiance-kmod]
                             [--compiler-scope {all,gemmini,npu,saturn,spacemit,none}]
                             [--build-compiler | --no-build-compiler]
                             [--build-python-bindings | --no-build-python-bindings]
                             [--build-samples | --no-build-samples]
                             [--build-tests | --no-build-tests]
                             [--enable-libbacktrace | --no-enable-libbacktrace]
                             [--enable-tracy] [--offline-friendly]
                             [--cmake-bin CMAKE_BIN] [--use-system-cmake]
                             [--use-ccache | --no-use-ccache]
                             [--cmake-arg CMAKE_ARG]
                             [--cmake-build-arg CMAKE_BUILD_ARG]
                             [--native-build-arg NATIVE_BUILD_ARG] [--clean]
                             [--verbose]

Configure and build Merlin and target runtimes

options:
  -h, --help            show this help message and exit
  --profile {firesim,full-plugin,gemmini,npu,package-firesim,package-host,package-spacemit,radiance,spacemit,vanilla}
                        High-level user profile preset. Use this for normal
                        workflows; advanced flags may still override details.
  --target {host,spacemit,firesim}
                        Target platform.
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
  --compiler-scope {all,gemmini,npu,saturn,spacemit,none}
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

## `tools/compile.py`

Compile MLIR/ONNX models to target artifacts

### Usage

```text
usage: uv run tools/compile.py [-h] --target TARGET [--hw HW] [--quantized]
                               [--output-dir OUTPUT_DIR]
                               [--build-dir BUILD_DIR]
                               [--compile-to COMPILE_TO]
                               [--dump-compilation-phases-to DUMP_COMPILATION_PHASES_TO]
                               [--iree-compile-arg IREE_COMPILE_ARG]
                               [--reuse-imported-mlir] [--tracy]
                               [--dump-artifacts] [--dump-phases]
                               [--dump-graph] [--build-benchmarks]
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
| `--build-benchmarks` | no | `False` | - | Recompile individual dispatch benchmarks and zip them |

### `--help` Output

```text
usage: uv run tools/compile.py [-h] --target TARGET [--hw HW] [--quantized]
                               [--output-dir OUTPUT_DIR]
                               [--build-dir BUILD_DIR]
                               [--compile-to COMPILE_TO]
                               [--dump-compilation-phases-to DUMP_COMPILATION_PHASES_TO]
                               [--iree-compile-arg IREE_COMPILE_ARG]
                               [--reuse-imported-mlir] [--tracy]
                               [--dump-artifacts] [--dump-phases]
                               [--dump-graph] [--build-benchmarks]
                               input_path

Compile MLIR/ONNX models to target artifacts

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
  --build-benchmarks    Recompile individual dispatch benchmarks and zip them
```

## `tools/setup.py`

Bootstrap developer environment and toolchains

### Usage

```text
usage: uv run tools/setup.py [-h] [--env-name ENV_NAME] [--env-file ENV_FILE]
                             [--offline] [--skip-conda] [--skip-pip]
                             [--python-deps {auto,uv,pip}]
                             [--conda-no-plugins | --no-conda-no-plugins]
                             [--submodules-profile {core,npu,smolvla,full}]
                             [--submodule-path SUBMODULE_PATH]
                             [--submodule-paths-recursive | --no-submodule-paths-recursive]
                             [--submodule-depth SUBMODULE_DEPTH]
                             [--submodule-jobs SUBMODULE_JOBS]
                             [--submodule-sync]
                             [--toolchain-target {spacemit,firesim,all}]
                             [--with-qemu] [--toolchain-force]
                             [--prebuilt-artifact {host-linux-x86_64,host-macos,runtime-spacemit,runtime-saturnopu}]
                             [--prebuilt-tag PREBUILT_TAG]
                             [--prebuilt-repo PREBUILT_REPO]
                             [--prebuilt-force]
                             [{all,env,toolchain,submodules,prebuilt}]
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `component` | no | `all` | `all, env, toolchain, submodules, prebuilt` |  |
| `--env-name` | no | `merlin-dev` | - | Conda environment name to update/install packages into (default: merlin-dev). |
| `--env-file` | no | `/scratch2/agustin/merlin/env_linux.yml` | - | Conda environment file to use. Default is platform-specific: env_linux.yml |
| `--offline` | no | `False` | - | Run setup in offline mode when possible. |
| `--skip-conda` | no | `False` | - | Skip conda environment sync. |
| `--skip-pip` | no | `False` | - | Skip Python dependency sync (uv/pip). |
| `--python-deps` | no | `auto` | `auto, uv, pip` | Python dependency installer. 'auto' prefers uv sync with uv.lock and falls back to pip requirements. |
| `--conda-no-plugins`, `--no-conda-no-plugins` | no | - | - | Force CONDA_NO_PLUGINS for conda env update. If unset, setup.py retries with CONDA_NO_PLUGINS=true on failure. |
| `--submodules-profile` | no | `core` | `core, npu, smolvla, full` | Which submodule profile to initialize (default: core). |
| `--submodule-path` | no | `[]` | - | Additional top-level submodule path to initialize (repeatable). |
| `--submodule-paths-recursive`, `--no-submodule-paths-recursive` | no | `False` | - | Whether extra --submodule-path entries should be initialized recursively. |
| `--submodule-depth` | no | `1` | - | Shallow depth for submodule fetches (default: 1). Use 0 for full history. |
| `--submodule-jobs` | no | `8` | - | Parallel submodule fetch jobs (default: 8). |
| `--submodule-sync` | no | `False` | - | Run `git submodule sync --recursive` before updating. |
| `--toolchain-target` | no | `spacemit` | `spacemit, firesim, all` | Which toolchain target to install (default: spacemit). |
| `--with-qemu` | no | `False` | - | For firesim toolchain setup, also install QEMU. |
| `--toolchain-force` | no | `False` | - | Reinstall toolchains even if the destination already exists. |
| `--prebuilt-artifact` | no | `host-linux-x86_64` | `host-linux-x86_64, host-macos, runtime-spacemit, runtime-saturnopu` | Which published Merlin prebuilt artifact to install. |
| `--prebuilt-tag` | no | `latest` | - | GitHub release tag to download from, or 'latest' (default: latest). |
| `--prebuilt-repo` | no | `ucb-bar/merlin` | - | GitHub repository containing release assets (default: ucb-bar/merlin). |
| `--prebuilt-force` | no | `False` | - | Replace an existing destination build tree when installing a prebuilt artifact. |

### `--help` Output

```text
usage: uv run tools/setup.py [-h] [--env-name ENV_NAME] [--env-file ENV_FILE]
                             [--offline] [--skip-conda] [--skip-pip]
                             [--python-deps {auto,uv,pip}]
                             [--conda-no-plugins | --no-conda-no-plugins]
                             [--submodules-profile {core,npu,smolvla,full}]
                             [--submodule-path SUBMODULE_PATH]
                             [--submodule-paths-recursive | --no-submodule-paths-recursive]
                             [--submodule-depth SUBMODULE_DEPTH]
                             [--submodule-jobs SUBMODULE_JOBS]
                             [--submodule-sync]
                             [--toolchain-target {spacemit,firesim,all}]
                             [--with-qemu] [--toolchain-force]
                             [--prebuilt-artifact {host-linux-x86_64,host-macos,runtime-spacemit,runtime-saturnopu}]
                             [--prebuilt-tag PREBUILT_TAG]
                             [--prebuilt-repo PREBUILT_REPO]
                             [--prebuilt-force]
                             [{all,env,toolchain,submodules,prebuilt}]

Bootstrap developer environment and toolchains

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
                        with uv.lock and falls back to pip requirements.
  --conda-no-plugins, --no-conda-no-plugins
                        Force CONDA_NO_PLUGINS for conda env update. If unset,
                        setup.py retries with CONDA_NO_PLUGINS=true on
                        failure.
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
  --prebuilt-artifact {host-linux-x86_64,host-macos,runtime-spacemit,runtime-saturnopu}
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

## `tools/ci.py`

Run repository CI/lint/patch workflows

### Usage

```text
usage: uv run tools/ci.py [-h]
                          {lint,cli-docs-drift,patch-gate,release-status} ...
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `cli-docs-drift`

```text
usage: uv run tools/ci.py cli-docs-drift [-h]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `lint`

```text
usage: uv run tools/ci.py lint [-h]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `patch-gate`

```text
usage: uv run tools/ci.py patch-gate [-h]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `release-status`

```text
usage: uv run tools/ci.py release-status [-h] [--tracking-file TRACKING_FILE]
                                         [--offline] [--json]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--tracking-file` | no | `/scratch2/agustin/merlin/.github/upstream_tracking.yaml` | - |  |
| `--offline` | no | `False` | - |  |
| `--json` | no | `False` | - |  |

### `--help` Output

```text
usage: uv run tools/ci.py [-h]
                          {lint,cli-docs-drift,patch-gate,release-status} ...

Run repository CI/lint/patch workflows

positional arguments:
  {lint,cli-docs-drift,patch-gate,release-status}
    lint                Run linters (shellcheck, python)
    cli-docs-drift      Regenerate docs/reference/cli.md and fail on drift
    patch-gate          CI gate: apply, verify, drift check
    release-status      Check upstream IREE versions

options:
  -h, --help            show this help message and exit
```

## `tools/patches.py`

Apply/verify/refresh/drift patch stack

### Usage

```text
usage: uv run tools/patches.py [-h] {verify,log,drift,export-upstream} ...
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `drift`

```text
usage: uv run tools/patches.py drift [-h]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `export-upstream`

```text
usage: uv run tools/patches.py export-upstream [-h] commit
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `commit` | yes | - | - | Commit hash to export |

#### Subcommand `log`

```text
usage: uv run tools/patches.py log [-h]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

#### Subcommand `verify`

```text
usage: uv run tools/patches.py verify [-h]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

### `--help` Output

```text
usage: uv run tools/patches.py [-h] {verify,log,drift,export-upstream} ...

Apply/verify/refresh/drift patch stack

positional arguments:
  {verify,log,drift,export-upstream}
    verify              Verify submodule is a clean rebase of pinned upstream
    log                 Show Merlin commits on top of upstream base
    drift               Check how far behind upstream the base is
    export-upstream     Export a commit as format-patch for upstream PR

options:
  -h, --help            show this help message and exit
```

## `tools/benchmark.py`

Run benchmark helper scripts

### Usage

```text
usage: uv run tools/benchmark.py [-h]
                                 target {compile-dual-vmfb,run-dual-remote}
                                 ...
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `target` | yes | - | - | Target name from config/targets.json |
| `action` | yes | - | `compile-dual-vmfb, run-dual-remote` | Benchmark action |
| `extra_args` | yes | - | - |  |

### `--help` Output

```text
usage: uv run tools/benchmark.py [-h]
                                 target {compile-dual-vmfb,run-dual-remote}
                                 ...

Run benchmark helper scripts

positional arguments:
  target                Target name from config/targets.json
  {compile-dual-vmfb,run-dual-remote}
                        Benchmark action
  extra_args

options:
  -h, --help            show this help message and exit
```
