# Getting Started

This is the fastest path to a working Merlin checkout. After these six steps
you will have host compiler tools built, one model compiled, one runtime
binary built, and you will know where outputs land.

If this is your first time in the repo, you can ignore most of the tree.
Start with `tools/`, `models/`, `docs/`, and the generated `build/` outputs.
Use [user_paths.md](user_paths.md) for a workflow-first map of the repo.

## Quick Path

The commands below use `./merlin` for brevity. These three forms are all
equivalent — pick whichever you prefer:

```bash
./merlin <subcommand>                                          # wrapper
conda activate merlin-dev && uv run tools/merlin.py <subcommand>   # direct
conda run -n merlin-dev uv run tools/merlin.py <subcommand>        # direct, no activation
```

Prefer a one-command setup? Open the repo in VSCode with the **Dev Containers**
extension and accept the "Reopen in Container" prompt — or, from the command
line, `docker compose run --rm merlin-dev ./merlin --help`. See
[`.devcontainer/devcontainer.json`](https://github.com/ucb-bar/merlin/blob/main/.devcontainer/devcontainer.json).
The image is several GB because it bakes the conda env; first build takes
~10 minutes.

### 1) Environment

```bash
git checkout dev/main
conda env create -f env_linux.yml   # macOS: env_macOS.yml
conda activate merlin-dev
uv sync
pre-commit install
```

### 2) Sync core submodules

```bash
./merlin setup submodules --submodules-profile core --submodule-sync
```

Submodule SHAs follow the current Merlin commit. If you switch branches later,
rerun the same command before building.

### 3) Build host compiler tools

```bash
./merlin build --profile vanilla
```

Expected install location: `build/host-vanilla-release/install/bin/`.
Use `--profile full-plugin` if you need the Merlin compiler plugins enabled.

### 4) Compile one model

```bash
./merlin compile models/dronet/dronet.mlir --target spacemit_x60
```

Expected output: `build/compiled_models/dronet/spacemit_x60_RVV_dronet/dronet.vmfb`.

### 5) Build one sample binary

```bash
./merlin build --profile spacemit --cmake-target merlin_baseline_dual_model_async_run
find build/spacemit-merlin-release -name baseline-dual-model-async-run
```

Typical binary path:
`build/spacemit-merlin-release/runtime/plugins/merlin-samples/SpacemiTX60/baseline_dual_model_async/baseline-dual-model-async-run`

### 6) Where outputs go

- Host tools: `build/host-*/install/bin/`
- Cross builds: `build/spacemit-*`, `build/firesim-*`
- Compiled models/artifacts: `build/compiled_models/<model>/...`
- Local rendered docs: `site/`

## Detailed Prerequisites

You only need this section if `./merlin setup` or `./merlin build` fails on
your machine, or if you are bringing up a target whose toolchain is not yet
managed by `./merlin`.

### Cross-toolchains

The conda env handles host clang/cmake/ninja. Cross-toolchains are installed
by helper scripts under `build_tools/`:

- SpacemiT (RISC-V Linux board): `build_tools/SpacemiT/setup_toolchain.sh`
- FireSim / Saturn OPU (bare-metal RISC-V): `build_tools/firesim/setup_toolchain.sh`

Run the script that matches your target before invoking `./merlin build --profile <target>`.

### Without `./merlin` (raw cmake)

If you specifically need to drive cmake yourself (e.g., debugging a build
issue), the IREE host build expects:

- `IREE_SRC = third_party/iree_bar`
- A conda env with `clang`, `clangxx`, `cmake`, `ninja`, and `lld`
- Configure with `-DIREE_TARGET_BACKEND_LLVM_CPU=ON`,
  `-DIREE_HAL_DRIVER_LOCAL_TASK=ON`, `-DIREE_BUILD_PYTHON_BINDINGS=OFF`

This is exactly what `./merlin build --profile vanilla` does for you. Prefer
the wrapper unless you have a reason not to.

## Next

- Practical extension guides (dialects/HAL/samples/targets): `how_to/index.md`
- User-first repo navigation: `user_paths.md`
- Repository layout and placement conventions: `repository_guide.md`
- Active workstream notes: `dev_blog/index.md`
- Full command and API reference: `reference/`
