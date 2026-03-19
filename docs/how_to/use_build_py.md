# Use `tools/build.py` Effectively

This page is a practical guide for common build workflows.

## 1) Preferred Entry

Use:

```bash
conda run -n merlin-dev uv run tools/build.py --profile <profile>
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
conda run -n merlin-dev uv run tools/build.py --profile npu --config release
```

Host runtime Radiance smoke target:

```bash
conda run -n merlin-dev uv run tools/build.py \
  --profile radiance \
  --cmake-target iree_hal_drivers_radiance_testing_transport_smoke_test
```

Cross-target sample build:

```bash
conda run -n merlin-dev uv run tools/build.py \
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
- `--cmake-arg <arg>` / `--configure-custom-arg <arg>` (repeatable, configure passthrough)
- `--cmake-build-arg <arg>` / `--build-custom-arg <arg>` (repeatable, `cmake --build` passthrough)
- `--native-build-arg <arg>` (repeatable, native tool passthrough after `--`)

Examples:

```bash
conda run -n merlin-dev uv run tools/build.py \
  --profile full-plugin \
  --cmake-arg=-DIREE_ENABLE_CPUINFO=OFF \
  --cmake-build-arg=--parallel \
  --cmake-build-arg=16
```

## 6) Package Profiles and Release Builds

Three profiles produce stripped, tarball-packaged artifacts in `dist/`:

- `package-host` — Linux host compiler and runtime tools
- `package-spacemit` — SpacemiT cross-compiled runtime and samples
- `package-firesim` — FireSim / Saturn OPU cross-compiled runtime and samples

These are the profiles used to create official release binaries.

### Building release artifacts locally with Docker

The recommended way to produce release tarballs is with the Docker-based
builder, which provides a reproducible environment:

```bash
# Check out the tag you want to release
git checkout v<VERSION>

# Build all three Linux artifacts inside Docker
./build_tools/docker/build_release.sh v<VERSION>
```

This produces three tarballs in `dist/`:

- `merlin-host-linux-x86_64.tar.gz`
- `merlin-runtime-spacemit.tar.gz`
- `merlin-runtime-saturnopu.tar.gz`

The Docker image bakes in the conda environment from `env_linux.yml` so
builds are hermetic. The repo is bind-mounted into the container — no
source is copied into the image.

Prerequisites: Docker must be installed on the build machine.

### macOS artifact

The macOS host artifact (`merlin-host-macos.tar.gz`) is built by the
`release-binaries` GitHub Actions workflow, which triggers on `v*` tag
pushes. It creates a draft GitHub release with the macOS tarball attached.

### Publishing a release

```bash
# 1. Tag and build Linux artifacts
git tag v<VERSION>
./build_tools/docker/build_release.sh v<VERSION>

# 2. Push the tag (triggers CI for macOS)
git push origin v<VERSION>

# 3. Upload Linux artifacts to the draft release
gh release upload v<VERSION> dist/*.tar.gz
```

If `gh` is not installed, upload through the GitHub web UI on the Releases
page. Edit the draft to add release notes, then publish.

## 7) Verify Build Result Quickly

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
