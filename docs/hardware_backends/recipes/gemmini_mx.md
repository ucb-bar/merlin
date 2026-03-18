# Recipe: Gemmini MX (Bare-Metal)

End-to-end steps to compile a model with Merlin and run it on the Gemmini MX
accelerator via bare-metal RTL simulation with VCS.

**Recipe file:** `build_tools/hardware/gemmini_mx.yaml`

## Prerequisites

- Chipyard `graphics` branch checked out at the pinned SHA (see
  [Compatibility Matrix](../compatibility_matrix.md)).
- Synopsys VCS simulator license available (or use Verilator as a free
  alternative).
- Merlin patches applied.

## Steps

### 0. One-time setup

```bash
# Initialize IREE submodule (already contains all Merlin changes)
git submodule update --init third_party/iree_bar

# Save your Chipyard path (persisted — only needed once)
conda run -n merlin-dev uv run tools/merlin.py chipyard set-path /path/to/chipyard

# Validate the Chipyard checkout matches this recipe
conda run -n merlin-dev uv run tools/merlin.py chipyard validate gemmini_mx
```

### 1. Build the RTL simulator

```bash
conda run -n merlin-dev uv run tools/merlin.py chipyard build-sim gemmini_mx
```

This runs `make CONFIG=RadianceGemminiOnlyConfig` in `$CHIPYARD_ROOT/sims/vcs`.
The first build takes significant time (RTL elaboration + VCS compilation).

### 2. (Optional) Validate the simulator with a reference kernel

Build and run the gemmini-rocc-tests suite to confirm the simulator is working
before attempting Merlin workloads:

```bash
git clone -b dev https://github.com/Rakanic/gemmini-rocc-tests
cd gemmini-rocc-tests
bash build.sh
```

Run the reference MX matmul kernel:

```bash
conda run -n merlin-dev uv run tools/merlin.py chipyard run gemmini_mx \
    /path/to/gemmini-rocc-tests/build/bareMetalC/matmul_ws_mx_generic
```

### 3. Build the bare-metal IREE runtime

```bash
conda run -n merlin-dev uv run tools/build.py --profile firesim --config release
```

This cross-compiles the IREE runtime for bare-metal RISC-V using the
`riscv_firesim.toolchain.cmake` toolchain. The resulting libraries are linked
into bare-metal ELFs that run on VCS via HTIF.

### 4. Compile the model with Merlin

```bash
conda run -n merlin-dev uv run tools/compile.py \
    models/mlp/mlp.q.int8.mlir \
    --target gemmini_mx \
    --quantized
```

### 5. Run on VCS

```bash
conda run -n merlin-dev uv run tools/merlin.py chipyard run gemmini_mx \
    /path/to/iree_sample.elf
```

### 6. Check status

```bash
conda run -n merlin-dev uv run tools/merlin.py chipyard status gemmini_mx
```

This checks for active build processes and whether the simulator binary exists.

## RadianceGemminiOnlyConfig reference

The Scala config class that defines this backend's SoC
(from `chipyard/RadianceConfigs.scala`):

```scala
class RadianceGemminiOnlyConfig extends Config(
  new WithRadianceMxGemmini(location = InCluster(0), dim = 16, accSizeInKB = 32, tileSize = (8, 8, 8)) ++
  new WithMuonCores(1, location = InCluster(0), ..., disabled = true) ++
  new WithRadianceCluster(0, ...) ++
  new WithExtGPUMem() ++
  new freechips.rocketchip.rocket.WithNSmallCores(1) ++
  new RadianceBaseConfig
)
```

## What happens under the hood

The `merlin chipyard run` command expands to:

```bash
cd $CHIPYARD_ROOT/sims/vcs
make CONFIG=RadianceGemminiOnlyConfig BINARY=/path/to/elf LOADMEM=1 run-binary
```

- `LOADMEM=1` bypasses DRAM initialization, loading the ELF directly into
  simulated memory.
- The IREE runtime communicates via HTIF (`tohost`/`fromhost` memory-mapped
  registers) for standard output and program termination.
