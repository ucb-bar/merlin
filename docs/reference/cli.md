# CLI Reference

This page is generated from real argparse parsers in `tools/*.py`.

Each command is shown with argument introspection and raw `--help` output.

## `tools/merlin.py`

Unified Merlin developer command reference parser.

### Usage

```text
usage: uv run tools/merlin.py [-h]
                              {build,compile,setup,ci,patches,benchmark} ...
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |

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
                                    [--profile {firesim,full-plugin,gemmini,npu,radiance,spacemit,vanilla}]
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
                                    [--offline-friendly]
                                    [--cmake-bin CMAKE_BIN]
                                    [--use-system-cmake]
                                    [--use-ccache | --no-use-ccache] [--clean]
                                    [--verbose]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--dry-run` | no | `False` | - | Print commands without executing |
| `--profile` | no | - | `firesim, full-plugin, gemmini, npu, radiance, spacemit, vanilla` | High-level user profile preset. Use this for normal workflows; advanced flags may still override details. |
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
| `--offline-friendly` | no | `False` | - | Prefer settings that avoid network fetches in CMake (equivalent to --no-build-compiler --no-build-python-bindings --no-enable-libbacktrace unless explicitly overridden). |
| `--cmake-bin` | no | `cmake` | - | CMake executable to use (default: cmake). |
| `--use-system-cmake` | no | `False` | - | Use /usr/bin/cmake instead of cmake from PATH. |
| `--use-ccache`, `--no-use-ccache` | no | `True` | - | Enable/disable ccache compiler launchers (default: enabled). |
| `--clean` | no | `False` | - | Delete build directory before building |
| `--verbose` | no | `False` | - | Enable verbose build output |

#### Subcommand `ci`

```text
usage: uv run tools/merlin.py ci [-h] [--dry-run]
                                 {lint,patch-gate,release-status} ...
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--dry-run` | no | `False` | - | Print commands without executing |

#### Subcommand `compile`

```text
usage: uv run tools/merlin.py compile [-h] [--dry-run] --target TARGET
                                      [--hw HW] [--quantized]
                                      [--build-dir BUILD_DIR]
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
| `--build-dir` | no | `host-vanilla-release` | - | Which build directory to use for compiler tools (default: host-vanilla-release) |
| `--dump-artifacts` | no | `False` | - | Dump executable sources, binaries, and configs |
| `--dump-phases` | no | `False` | - | Dump MLIR compilation phases |
| `--dump-graph` | no | `False` | - | Dump the flow dispatch graph (.dot) |
| `--build-benchmarks` | no | `False` | - | Recompile individual dispatch benchmarks and zip them |

#### Subcommand `patches`

```text
usage: uv run tools/merlin.py patches [-h] [--dry-run]
                                      {apply,verify,refresh,drift}
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--dry-run` | no | `False` | - | Print commands without executing |
| `action` | yes | - | `apply, verify, refresh, drift` | Patch workflow action |

#### Subcommand `setup`

```text
usage: uv run tools/merlin.py setup [-h] [--env-name ENV_NAME] [--offline]
                                    [--skip-conda] [--skip-pip]
                                    [--python-deps {auto,uv,pip}]
                                    [--conda-no-plugins | --no-conda-no-plugins]
                                    [{all,env,toolchain}]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `component` | no | `all` | `all, env, toolchain` |  |
| `--env-name` | no | `merlin-dev` | - | Conda environment name to update/install packages into (default: merlin-dev). |
| `--offline` | no | `False` | - | Run setup in offline mode when possible. |
| `--skip-conda` | no | `False` | - | Skip conda environment sync. |
| `--skip-pip` | no | `False` | - | Skip Python dependency sync (uv/pip). |
| `--python-deps` | no | `auto` | `auto, uv, pip` | Python dependency installer. 'auto' prefers uv sync with uv.lock and falls back to pip requirements. |
| `--conda-no-plugins`, `--no-conda-no-plugins` | no | - | - | Force CONDA_NO_PLUGINS for conda env update. If unset, setup.py retries with CONDA_NO_PLUGINS=true on failure. |

### `--help` Output

```text
usage: uv run tools/merlin.py [-h]
                              {build,compile,setup,ci,patches,benchmark} ...

Unified Merlin developer command reference parser.

positional arguments:
  {build,compile,setup,ci,patches,benchmark}
    build               Configure and build Merlin and target runtimes
    compile             Compile MLIR/ONNX models to target artifacts
    setup               Bootstrap developer environment and toolchains
    ci                  Run repository CI/lint/patch workflows
    patches             Apply/verify/refresh/drift patch stack
    benchmark           Run benchmark helper scripts

options:
  -h, --help            show this help message and exit
```

## `tools/build.py`

Configure and build Merlin and target runtimes

### Usage

```text
usage: uv run tools/build.py [-h]
                             [--profile {firesim,full-plugin,gemmini,npu,radiance,spacemit,vanilla}]
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
                             [--offline-friendly] [--cmake-bin CMAKE_BIN]
                             [--use-system-cmake]
                             [--use-ccache | --no-use-ccache] [--clean]
                             [--verbose]
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--profile` | no | - | `firesim, full-plugin, gemmini, npu, radiance, spacemit, vanilla` | High-level user profile preset. Use this for normal workflows; advanced flags may still override details. |
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
| `--offline-friendly` | no | `False` | - | Prefer settings that avoid network fetches in CMake (equivalent to --no-build-compiler --no-build-python-bindings --no-enable-libbacktrace unless explicitly overridden). |
| `--cmake-bin` | no | `cmake` | - | CMake executable to use (default: cmake). |
| `--use-system-cmake` | no | `False` | - | Use /usr/bin/cmake instead of cmake from PATH. |
| `--use-ccache`, `--no-use-ccache` | no | `True` | - | Enable/disable ccache compiler launchers (default: enabled). |
| `--clean` | no | `False` | - | Delete build directory before building |
| `--verbose` | no | `False` | - | Enable verbose build output |

### `--help` Output

```text
usage: uv run tools/build.py [-h]
                             [--profile {firesim,full-plugin,gemmini,npu,radiance,spacemit,vanilla}]
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
                             [--offline-friendly] [--cmake-bin CMAKE_BIN]
                             [--use-system-cmake]
                             [--use-ccache | --no-use-ccache] [--clean]
                             [--verbose]

Configure and build Merlin and target runtimes

options:
  -h, --help            show this help message and exit
  --profile {firesim,full-plugin,gemmini,npu,radiance,spacemit,vanilla}
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
  --clean               Delete build directory before building
  --verbose             Enable verbose build output
```

## `tools/compile.py`

Compile MLIR/ONNX models to target artifacts

### Usage

```text
usage: uv run tools/compile.py [-h] --target TARGET [--hw HW] [--quantized]
                               [--build-dir BUILD_DIR] [--dump-artifacts]
                               [--dump-phases] [--dump-graph]
                               [--build-benchmarks]
                               input_path
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `input_path` | yes | - | - | Path to the model directory OR specific .mlir/.onnx file |
| `--target` | yes | - | - | Target YAML config file name (e.g., spacemit_x60) |
| `--hw` | no | - | - | Hardware sub-target from YAML (e.g., RVV, OPU). If omitted, uses default_hw from YAML. |
| `--quantized` | no | `False` | - | Force quantized mode (auto-detected if .q. in filename) |
| `--build-dir` | no | `host-vanilla-release` | - | Which build directory to use for compiler tools (default: host-vanilla-release) |
| `--dump-artifacts` | no | `False` | - | Dump executable sources, binaries, and configs |
| `--dump-phases` | no | `False` | - | Dump MLIR compilation phases |
| `--dump-graph` | no | `False` | - | Dump the flow dispatch graph (.dot) |
| `--build-benchmarks` | no | `False` | - | Recompile individual dispatch benchmarks and zip them |

### `--help` Output

```text
usage: uv run tools/compile.py [-h] --target TARGET [--hw HW] [--quantized]
                               [--build-dir BUILD_DIR] [--dump-artifacts]
                               [--dump-phases] [--dump-graph]
                               [--build-benchmarks]
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
  --build-dir BUILD_DIR
                        Which build directory to use for compiler tools
                        (default: host-vanilla-release)
  --dump-artifacts      Dump executable sources, binaries, and configs
  --dump-phases         Dump MLIR compilation phases
  --dump-graph          Dump the flow dispatch graph (.dot)
  --build-benchmarks    Recompile individual dispatch benchmarks and zip them
```

## `tools/setup.py`

Bootstrap developer environment and toolchains

### Usage

```text
usage: uv run tools/setup.py [-h] [--env-name ENV_NAME] [--offline]
                             [--skip-conda] [--skip-pip]
                             [--python-deps {auto,uv,pip}]
                             [--conda-no-plugins | --no-conda-no-plugins]
                             [{all,env,toolchain}]
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `component` | no | `all` | `all, env, toolchain` |  |
| `--env-name` | no | `merlin-dev` | - | Conda environment name to update/install packages into (default: merlin-dev). |
| `--offline` | no | `False` | - | Run setup in offline mode when possible. |
| `--skip-conda` | no | `False` | - | Skip conda environment sync. |
| `--skip-pip` | no | `False` | - | Skip Python dependency sync (uv/pip). |
| `--python-deps` | no | `auto` | `auto, uv, pip` | Python dependency installer. 'auto' prefers uv sync with uv.lock and falls back to pip requirements. |
| `--conda-no-plugins`, `--no-conda-no-plugins` | no | - | - | Force CONDA_NO_PLUGINS for conda env update. If unset, setup.py retries with CONDA_NO_PLUGINS=true on failure. |

### `--help` Output

```text
usage: uv run tools/setup.py [-h] [--env-name ENV_NAME] [--offline]
                             [--skip-conda] [--skip-pip]
                             [--python-deps {auto,uv,pip}]
                             [--conda-no-plugins | --no-conda-no-plugins]
                             [{all,env,toolchain}]

Bootstrap developer environment and toolchains

positional arguments:
  {all,env,toolchain}

options:
  -h, --help            show this help message and exit
  --env-name ENV_NAME   Conda environment name to update/install packages into
                        (default: merlin-dev).
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
```

## `tools/ci.py`

Run repository CI/lint/patch workflows

### Usage

```text
usage: uv run tools/ci.py [-h] {lint,patch-gate,release-status} ...
```

### Arguments

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
| `--tracking-file` | no | `/scratch2/agustin/merlin/config/upstream_tracking.yaml` | - |  |
| `--offline` | no | `False` | - |  |
| `--json` | no | `False` | - |  |

### `--help` Output

```text
usage: uv run tools/ci.py [-h] {lint,patch-gate,release-status} ...

Run repository CI/lint/patch workflows

positional arguments:
  {lint,patch-gate,release-status}
    lint                Run linters (shellcheck, python)
    patch-gate          CI gate: apply, verify, drift check
    release-status      Check upstream IREE versions

options:
  -h, --help            show this help message and exit
```

## `tools/patches.py`

Apply/verify/refresh/drift patch stack

### Usage

```text
usage: uv run tools/patches.py [-h] {apply,verify,refresh,drift}
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `action` | yes | - | `apply, verify, refresh, drift` | Patch workflow action |

### `--help` Output

```text
usage: uv run tools/patches.py [-h] {apply,verify,refresh,drift}

Apply/verify/refresh/drift patch stack

positional arguments:
  {apply,verify,refresh,drift}
                        Patch workflow action

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
