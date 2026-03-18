# Recipe: Saturn OPU on FireSim U250

End-to-end steps to compile a model with Merlin and run it on the Saturn OPU
accelerator via FireSim on a Xilinx Alveo U250 FPGA.

**Recipe file:** `build_tools/hardware/saturn_opu_u250.yaml`

## Prerequisites

- Chipyard checked out at the pinned SHA (see
  [Compatibility Matrix](../compatibility_matrix.md)).
- Xilinx Alveo U250 FPGA available on the host machine.
- FireSim installed and configured within the Chipyard workspace.
- Merlin patches applied.

## Steps

### 0. One-time setup

```bash
# Initialize IREE submodule (already contains all Merlin changes)
git submodule update --init third_party/iree_bar

# Save your Chipyard path (persisted — only needed once)
conda run -n merlin-dev uv run tools/merlin.py chipyard set-path /path/to/chipyard

# Validate the Chipyard checkout matches this recipe
conda run -n merlin-dev uv run tools/merlin.py chipyard validate saturn_opu_u250
```

### 1. Build the FireMarshal base image (once per workspace)

```bash
conda run -n merlin-dev uv run tools/merlin.py chipyard build-firemarshal
```

This builds the `br-base` Linux image that all Merlin FireSim workloads use as
their base rootfs. Only needs to run once.

### 2. Configure FireSim

```bash
conda run -n merlin-dev uv run tools/merlin.py chipyard configure-firesim saturn_opu_u250
```

This automatically writes:

- `config_build.yaml` — sets `builds_to_run` to the Saturn OPU build recipe.
- `config_runtime.yaml` — sets `default_hw_config` and `workload_name`.

You do **not** need to manually edit any FireSim YAML files.

### 3. Build the FPGA bitstream

```bash
conda run -n merlin-dev uv run tools/merlin.py chipyard build-bitstream saturn_opu_u250
```

This runs `firesim buildbitstream` from the deploy directory. The build takes
several hours. Use `tmux` to avoid losing progress if your terminal disconnects.

Check progress at any time:

```bash
conda run -n merlin-dev uv run tools/merlin.py chipyard status saturn_opu_u250
```

### 4. Register the built bitstream

```bash
conda run -n merlin-dev uv run tools/merlin.py chipyard register-hwdb saturn_opu_u250
```

This finds the built `firesim.tar.gz` in `results-build/` and adds it to
`config_hwdb.yaml` automatically.

### 5. Compile the model with Merlin

```bash
conda run -n merlin-dev uv run tools/compile.py \
    models/mlp/mlp.q.int8.mlir \
    --target saturn_opu \
    --hw OPU \
    --quantized
```

### 6. Build the IREE runtime

```bash
conda run -n merlin-dev uv run tools/build.py --profile firesim --config release
```

### 7. Stage the workload

```bash
conda run -n merlin-dev uv run tools/merlin.py chipyard stage-workload saturn_opu_u250
```

This creates:

- A workload JSON file in `$CHIPYARD_ROOT/sims/firesim/deploy/workloads/`.
- An overlay directory containing the Merlin IREE runtime and compiled model
  artifacts, which get baked into the FireSim Linux image.

To stage a custom overlay directory:

```bash
merlin chipyard stage-workload saturn_opu_u250 /path/to/custom/overlay
```

### 8. Run on FireSim

```bash
cd $CHIPYARD_ROOT/sims/firesim/deploy
firesim infrasetup
firesim runworkload
```

### 9. Collect results

FireSim stores output logs in the configured simulation directory. Look for
`uartlog` files in the results. Extract benchmark timings from the IREE
runtime output.

## What the tool configures

The `merlin chipyard` subcommands modify these Chipyard files automatically:

| File | Modified by | What changes |
|---|---|---|
| `config_build.yaml` | `configure-firesim` | `builds_to_run` list |
| `config_runtime.yaml` | `configure-firesim` | `default_hw_config`, `workload_name` |
| `config_hwdb.yaml` | `register-hwdb` | Adds bitstream entry with `bitstream_tar` path |
| `workloads/<name>.json` | `stage-workload` | Creates workload JSON with overlay files |
| `workloads/<name>/overlay/` | `stage-workload` | Creates overlay with Merlin binaries |

The tool does **not** modify `config_build_recipes.yaml` — build recipes are
more static and the existing entries in Chipyard should already contain the
needed configs.

## Reference

See `docs/reproducibility/reproduce_ukernel_benchmark_firesim.md` for a
detailed A/B benchmarking workflow on this backend.
