# Hardware Backends

Merlin compiles ML models into IREE artifacts for RISC-V targets. Hardware
backends define how those artifacts execute on real or simulated hardware.

## Backend categories

There are three categories of backend:

1. **Physical boards** -- SpacemiT X60 (SpacemiT K1 SoC). Run compiled
   artifacts directly on a Linux-capable RISC-V board over SSH.
2. **FPGA simulation** -- Saturn OPU on FireSim U250. Full-system simulation on
   Xilinx Alveo U250 FPGAs using the FireSim infrastructure.
3. **Bare-metal RTL simulation** -- Gemmini MX on Chipyard VCS. Cycle-accurate
   RTL simulation of bare-metal ELFs using Synopsys VCS (or Verilator).

## What defines a backend

Each backend is defined by a **recipe** in `build_tools/hardware/*.yaml`. A
recipe pins:

- The **Chipyard branch and SHA** the hardware was validated on.
- The **Chipyard config class** (Scala) that defines the SoC.
- The **execution mode** (`firesim`, `bare-metal`, or `board`).
- The **Merlin build profile** and target YAML for compilation.
- For FireSim recipes: the full build recipe, runtime config, and workload
  definition — everything the `merlin chipyard` tool needs to configure
  FireSim automatically.

## The `merlin chipyard` tool

The `merlin chipyard` tool handles all Chipyard interactions so you never need
to manually edit Chipyard config files. It reads the recipes from
`build_tools/hardware/` and drives everything.

### First-time setup

```bash
# 1. Initialize the IREE submodule (already contains all Merlin changes)
git submodule update --init third_party/iree_bar

# 2. Tell Merlin where Chipyard lives (saved persistently)
conda run -n merlin-dev uv run tools/merlin.py chipyard set-path /path/to/chipyard

# 3. Validate your Chipyard checkout matches a recipe
conda run -n merlin-dev uv run tools/merlin.py chipyard validate gemmini_mx
```

### Bare-metal flow (Gemmini MX on VCS)

```bash
# Build the RTL simulator
merlin chipyard build-sim gemmini_mx

# Build the bare-metal IREE runtime
conda run -n merlin-dev uv run tools/build.py --profile firesim --config release

# Compile your model
conda run -n merlin-dev uv run tools/compile.py \
  models/mlp/mlp.q.int8.mlir --target gemmini_mx --quantized

# Run on VCS
merlin chipyard run gemmini_mx path/to/iree_sample.elf
```

### FireSim flow (Saturn OPU on U250)

```bash
# Build FireMarshal base Linux image (once per workspace)
merlin chipyard build-firemarshal

# Configure FireSim deploy YAMLs (config_build.yaml, config_runtime.yaml)
merlin chipyard configure-firesim saturn_opu_u250

# Build the FPGA bitstream (takes hours — use tmux)
merlin chipyard build-bitstream saturn_opu_u250

# Register the built bitstream in config_hwdb.yaml
merlin chipyard register-hwdb saturn_opu_u250

# Stage Merlin workload (creates workload JSON + overlay)
merlin chipyard stage-workload saturn_opu_u250

# Check status at any time
merlin chipyard status saturn_opu_u250

# Run (from chipyard deploy directory)
cd $CHIPYARD_ROOT/sims/firesim/deploy
firesim infrasetup && firesim runworkload
```

### Available subcommands

| Command | Description |
|---|---|
| `set-path <path>` | Save Chipyard workspace path (persisted in `.chipyard_config.json`) |
| `info` | Show Chipyard state and list available recipes |
| `validate <recipe>` | Check branch, SHA, submodules, sysroot against a recipe |
| `build-sim <recipe>` | Build VCS/Verilator RTL simulator |
| `run <recipe> <elf>` | Run bare-metal ELF on simulator |
| `configure-firesim <recipe>` | Write `config_build.yaml` and `config_runtime.yaml` |
| `build-bitstream <recipe>` | Run `firesim buildbitstream` |
| `register-hwdb <recipe>` | Find built bitstream and register in `config_hwdb.yaml` |
| `stage-workload <recipe> [overlay]` | Create workload JSON and overlay directory |
| `build-firemarshal` | Build FireMarshal base Linux image |
| `status <recipe>` | Check build processes, bitstream, HWDB registration |

## How the bare-metal runtime works

The bare-metal IREE runtime patches are **compile-time** — they are activated
automatically when you build with the `firesim` profile. There is no runtime
flag to set.

The `firesim` profile selects `build_tools/firesim/riscv_firesim.toolchain.cmake`,
which sets two key C defines:

- `-DIREE_PLATFORM_GENERIC=1` — activates explicit status-check code paths
  instead of macro-based error propagation that breaks on bare-metal newlib.
- `-DIREE_MEMORY_ACCESS_ALIGNMENT_REQUIRED` — activates byte-by-byte memory
  access in place of `memcpy`, which can emit unaligned loads on bare-metal
  RISC-V.

These defines gate `#ifdef` blocks in the IREE runtime source files (applied
via patch `0002-merlin-bare-metal-unaligned-access-and-status.patch`). When
building for Linux targets (`spacemit`, `host`), neither define is set, so
the original optimized code paths are used.

## Chipyard path configuration

The `merlin chipyard set-path` command saves the path persistently in
`.chipyard_config.json` (gitignored). The path is also available via:

- `--chipyard-root` flag on any `merlin chipyard` subcommand
- `CHIPYARD_ROOT` environment variable
- The toolchain cmake reads `CHIPYARD_ROOT` to find the newlib sysroot at
  `$CHIPYARD_ROOT/.conda-env/riscv-tools/riscv64-unknown-elf`

## How downstream IREE changes are managed

Merlin maintains downstream changes to IREE as **commits on the `ucb-bar/main`
branch** of our fork (`github.com/ucb-bar/iree`). The `third_party/iree_bar`
submodule points at this branch.

**There are no patch files to apply.** The submodule already contains all
Merlin changes. Just initialize the submodule and build:

```bash
git submodule update --init third_party/iree_bar
```

### Verifying the submodule

```bash
# Check that the submodule is a clean rebase of the pinned upstream base
conda run -n merlin-dev uv run tools/merlin.py patches verify

# See the Merlin-specific commits on top of upstream
conda run -n merlin-dev uv run tools/merlin.py patches log

# Check how far behind upstream the current base is
conda run -n merlin-dev uv run tools/merlin.py patches drift
```

### Bumping IREE upstream

When a new upstream IREE version is available:

```bash
cd third_party/iree_bar
git fetch https://github.com/iree-org/iree main
git rebase FETCH_HEAD
# Resolve per-commit conflicts if any, then:
git push origin ucb-bar/main --force-with-lease
cd ../..
git add third_party/iree_bar
```

Then update `IREE_UPSTREAM_BASE` in `build_tools/patches/manifest.env`.

### Upstream PR preparation

Curated `git format-patch` exports for upstream PRs live in
`build_tools/patches/upstream/` with READMEs explaining what to include. To export a
commit for a PR:

```bash
conda run -n merlin-dev uv run tools/merlin.py patches export-upstream <commit-hash>
```

For the full model, see [Plugin & Patch Model](../architecture/plugin_and_patch_model.md).

## Further reading

- [Compatibility Matrix](compatibility_matrix.md) -- pinned versions for every
  backend.
- [Chipyard Concepts](chipyard_concepts.md) -- quick primer on Chipyard,
  FireSim, and VCS for Merlin users.
- Recipes:
  - [Saturn OPU on FireSim U250](recipes/saturn_opu_u250.md)
  - [Gemmini MX (Bare-Metal)](recipes/gemmini_mx.md)
  - [SpacemiT X60 (Linux Board)](recipes/spacemit_x60.md)
