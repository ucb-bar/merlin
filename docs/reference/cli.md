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
                                    [--target {host,spacemit,firesim}]
                                    [--config {debug,release,asan,trace,perf}]
                                    [--cmake-target CMAKE_TARGET]
                                    [--with-plugin] [--clean] [--verbose]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--dry-run` | no | `False` | - | Print commands without executing |
| `--target` | no | `host` | `host, spacemit, firesim` | Target platform (default: host) |
| `--config` | no | `debug` | `debug, release, asan, trace, perf` | Build configuration type |
| `--cmake-target` | no | - | - | Build specific CMake target (default: install) |
| `--with-plugin` | no | `False` | - | Enable Merlin compiler plugin |
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
usage: uv run tools/merlin.py setup [-h] [{all,env,toolchain}]
```

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `component` | no | `all` | `all, env, toolchain` |  |

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
usage: uv run tools/build.py [-h] [--target {host,spacemit,firesim}]
                             [--config {debug,release,asan,trace,perf}]
                             [--cmake-target CMAKE_TARGET] [--with-plugin]
                             [--clean] [--verbose]
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `--target` | no | `host` | `host, spacemit, firesim` | Target platform (default: host) |
| `--config` | no | `debug` | `debug, release, asan, trace, perf` | Build configuration type |
| `--cmake-target` | no | - | - | Build specific CMake target (default: install) |
| `--with-plugin` | no | `False` | - | Enable Merlin compiler plugin |
| `--clean` | no | `False` | - | Delete build directory before building |
| `--verbose` | no | `False` | - | Enable verbose build output |

### `--help` Output

```text
usage: uv run tools/build.py [-h] [--target {host,spacemit,firesim}]
                             [--config {debug,release,asan,trace,perf}]
                             [--cmake-target CMAKE_TARGET] [--with-plugin]
                             [--clean] [--verbose]

Configure and build Merlin and target runtimes

options:
  -h, --help            show this help message and exit
  --target {host,spacemit,firesim}
                        Target platform (default: host)
  --config {debug,release,asan,trace,perf}
                        Build configuration type
  --cmake-target CMAKE_TARGET
                        Build specific CMake target (default: install)
  --with-plugin         Enable Merlin compiler plugin
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
usage: uv run tools/setup.py [-h] [{all,env,toolchain}]
```

### Arguments

| Argument | Required | Default | Choices | Help |
| --- | --- | --- | --- | --- |
| `component` | no | `all` | `all, env, toolchain` |  |

### `--help` Output

```text
usage: uv run tools/setup.py [-h] [{all,env,toolchain}]

Bootstrap developer environment and toolchains

positional arguments:
  {all,env,toolchain}

options:
  -h, --help           show this help message and exit
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
