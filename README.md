# Merlin Compiler Infrastructure

Merlin is an MLIR/IREE-based compiler stack for compiling models to CPU and custom RISC-V targets (including SpacemiT and Saturn OPU flows). Ideally here is where he handle the ***Compiler Magic*** that happens in the background so that you dont have to.

<p align="center">
  <img src="docs/assets/merlin_transparent.png" width="400">
</p>

We recommend navigating the Merlin documentation using the website:

- [**Merlin Documentation**](https://ucb-bar.github.io/merlin/)

Merlin can be used either from published prebuilt binaries or by building from source.
For most users, the prebuilt release artifacts are the fastest way to get started.

## Choose Your Path

Most first-time users do not need to understand the whole repository.

- If you want to compile or run models, start with:
  - `tools/`
  - `models/`
  - `docs/getting_started.md`
  - `build/` outputs
- If you are bringing up a new hardware target, add:
  - `target_specs/`
  - `models/*.yaml`
  - `build_tools/hardware/`
  - `docs/architecture/target_generator.md`
- If you are modifying compiler or runtime internals, you will eventually work in:
  - `compiler/`
  - `runtime/`
  - `third_party/iree_bar`

Most first users can ignore `third_party/`, `projects/`, most of `benchmarks/`,
and the deeper `docs/dev_blog/` entries until they are debugging or extending
Merlin.

For a more opinionated first-user repo walkthrough, see
[docs/user_paths.md](docs/user_paths.md).

## Quick Start

There are two supported ways to use Merlin:

- **Recommended:** install a prebuilt release artifact
- **Developer path:** build Merlin from source

If you only want to compile models or run released runtimes, use the prebuilt binaries first.
If you are actively developing Merlin, changing compiler passes, or working on unreleased targets, build from source.

### Option A) Use prebuilt binaries (recommended)

Release artifacts are published on the GitHub Releases page.

Current release artifact families:

- `merlin-host-linux-x86_64.tar.gz`
- `merlin-host-macos.tar.gz`
- `merlin-runtime-spacemit.tar.gz`
- `merlin-runtime-saturnopu.tar.gz`

These artifacts are meant to be installed into the same `build/...` layout that Merlin expects locally, so the rest of the scripts can continue to work normally.

Typical installed layouts:

- host tools:
  - `build/host-merlin-perf/install/bin/`
- SpacemiT runtime package:
  - `build/spacemit-merlin-perf/install/`
  - `build/spacemit-merlin-perf/runtime/plugins/merlin-samples/`
- Saturn OPU / FireSim runtime package:
  - `build/firesim-merlin-perf/install/`
  - `build/firesim-merlin-perf/runtime/plugins/merlin-samples/`

To install a prebuilt release, use:

```bash
conda run -n merlin-dev uv run tools/merlin.py setup prebuilt --help
```

Then install the artifact you want from a tagged release.

Recommended artifacts by use case:

- **Linux/macOS host compiler tools:** use the corresponding `merlin-host-*` artifact
- **SpacemiT runtime + samples:** use `merlin-runtime-spacemit`
- **Saturn OPU / FireSim runtime + samples:** use `merlin-runtime-saturnopu`

Once installed, Merlin commands should work against those prebuilt tools using the normal `build/...` locations.

### Option B) Build from source (developer path)

Use this path if you are:

- developing Merlin itself
- changing compiler/runtime code
- working on targets or flows that do not yet have released binaries

#### 0) Git setup

Choose the Merlin branch you want to work on before initializing submodules.
Submodule SHAs follow the currently checked out Merlin commit, so if you switch
branches later you must rerun the submodule setup step.

```bash
git checkout dev/main
```

#### 1) Environment

```bash
conda env create -f env_linux.yml
conda activate merlin-dev
uv sync
pre-commit install
```

Optional convenience alias for your shell session:

```bash
alias merlin='uv run tools/merlin.py'
```

#### 2) Submodules

Initialize the core submodules for the branch you checked out above:

```bash
conda activate merlin-dev
uv run tools/merlin.py setup submodules --submodules-profile core --submodule-sync
```

If you need additional development flows later, use the appropriate submodule
profile. If you switch Merlin branches later, rerun the same setup command so
`third_party/` matches the new branch pins.

#### 3) Build host compiler tools

```bash
conda activate merlin-dev
uv run tools/merlin.py build --profile full-plugin --config release
```

This creates host tools under:

- `build/host-merlin-release/install/bin/`

If you ever need a strictly upstream IREE build without Merlin plugins, use:

```bash
uv run tools/merlin.py build --profile vanilla --config release
```

which outputs to:

- `build/host-vanilla-release/install/bin/`

#### 4) Compile one model with `compile.py`

```bash
conda activate merlin-dev
uv run tools/merlin.py compile models/dronet/dronet.mlir --target spacemit_x60
```

Expected output artifact:

- `build/compiled_models/dronet/spacemit_x60_RVV_dronet/dronet.vmfb`

#### 5) Build one runtime/sample binary

```bash
conda activate merlin-dev
uv run tools/merlin.py build --target spacemit --config release --with-plugin --cmake-target merlin_baseline_dual_model_async_run
find build/spacemit-merlin-release -name baseline-dual-model-async-run
```

Typical binary location:

- `build/spacemit-merlin-release/samples/SpacemiTX60/baseline_dual_model_async/baseline-dual-model-async-run`

## Build Profiles (Recommended)

For common workflows, prefer `--profile` instead of combining many low-level flags:

```bash
# Host vanilla developer build
uv run tools/merlin.py build --profile vanilla

# Host full plugin build (compiler + runtime plugin paths)
uv run tools/merlin.py build --profile full-plugin

# Host packaged release-style build
uv run tools/merlin.py build --profile package-host

# Host Radiance runtime bring-up path (minimal compiler dependencies)
uv run tools/merlin.py build --profile radiance --cmake-target iree_hal_drivers_radiance_testing_transport_smoke_test

# Cross-target runtime/sample builds
uv run tools/merlin.py build --profile spacemit
uv run tools/merlin.py build --profile firesim

# Cross-target packaged runtime builds
uv run tools/merlin.py build --profile package-spacemit
uv run tools/merlin.py build --profile package-firesim

# Host Gemmini-focused compiler plugin flow
uv run tools/merlin.py build --profile gemmini
```

If needed, you can still override profile defaults with specific flags for advanced use.

## Prebuilt Release Artifacts

Official release artifacts are published on GitHub Releases.

Artifact naming:

- `merlin-host-linux-x86_64.tar.gz`: Linux host compiler/runtime tools
- `merlin-host-macos.tar.gz`: macOS host compiler/runtime tools
- `merlin-runtime-spacemit.tar.gz`: SpacemiT runtime package and samples
- `merlin-runtime-saturnopu.tar.gz`: Saturn OPU / FireSim runtime package and samples

Use `tools/install_prebuilt.py` to place these into the expected local `build/...` layout.

### Creating a release

See the [packaging and release builds](docs/how_to/use_build_py.md#6-package-profiles-and-release-builds)
guide for the full walkthrough. In short: Linux artifacts are built locally
with `./build_tools/docker/build_release.sh v<VERSION>`, the macOS artifact
is built by CI on tag push, and tarballs are uploaded to the GitHub release.

## Where Build Outputs Go

- Host tool binaries: `build/host-*/install/bin/`
- Packaged release archives: `dist/*.tar.gz`
- Cross-target build trees: `build/spacemit-*`, `build/firesim-*`
- Compiled model artifacts (`.mlir`, `.vmfb`, optional dumps): `build/compiled_models/<model>/...`
- Generated docs site (local): `site/`

## Repository Map (User View First)

```text
merlin/
├── tools/              # The main CLI entrypoint (`tools/merlin.py`) and developer helpers
├── models/             # Models to compile and current target YAML views used by `compile.py`
├── docs/               # Start here for getting started, how-to guides, and repo navigation
├── build/              # Generated outputs, compiled artifacts, local build trees
├── target_specs/       # Canonical TargetGen capability specs and deployment overlays
├── compiler/           # MLIR dialects, passes, plugins, and compiler-side target work
├── runtime/            # HAL drivers and runtime-side target work
├── build_tools/        # Toolchains, recipes, packaging, and deployment helpers
├── samples/            # End-to-end runtime examples and target-facing app flows
├── benchmarks/         # Benchmark and profiling workflows
├── third_party/        # Forks and submodules, including IREE and LLVM
└── projects/           # Sidecar research or tooling projects such as `mlirAgent`
```

If you are new here, the first four entries are the important ones.

Detailed folder guide with contributor-facing placement rules:

- [docs/repository_guide.md](docs/repository_guide.md)

## If You Want To Extend Merlin

- New compiler pass/dialect: add under `compiler/src/merlin/...`
- New plugin/target wiring: add under `compiler/plugins/...`
- New model path/config: add model files in `models/<model>/` and flags in `models/<target>.yaml`
- New sample runtime app: add under `samples/<platform>/...`
- New benchmark flow: add under `benchmarks/<target>/...`

## Documentation

Primary docs:

- [docs/index.md](docs/index.md)
- [docs/getting_started.md](docs/getting_started.md)
- [docs/how_to/index.md](docs/how_to/index.md)
- [docs/iree_setup.md](docs/iree_setup.md)
- [docs/dev_blog/index.md](docs/dev_blog/index.md)
- [docs/architecture/plugin_and_patch_model.md](docs/architecture/plugin_and_patch_model.md)
- [docs/architecture/cmake_presets.md](docs/architecture/cmake_presets.md)

Build docs locally:

```bash
conda activate merlin-dev
MLIR_TBLGEN=build/host-vanilla-release/llvm-project/bin/mlir-tblgen   uv run --with-requirements docs/requirements.txt python docs/hooks.py
uv run --with-requirements docs/requirements.txt zensical serve
```

Unified CLI help:

```bash
conda activate merlin-dev
uv run tools/merlin.py --help
```

Published docs URL (after GitHub Pages is enabled): `https://ucb-bar.github.io/merlin/`

Then open `http://127.0.0.1:8000`.

## Formatting and Checks

```bash
conda activate merlin-dev
pre-commit run --all-files
uv run tools/merlin.py ci lint
```

## Contributing

- [CONTRIBUTING.md](CONTRIBUTING.md)
