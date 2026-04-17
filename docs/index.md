# Merlin

<div align="center">
  <img src="assets/merlin_transparent.png" alt="Merlin Compiler Logo" width="350">
  <p><em>MLIR/IREE-based compiler stack for CPU and custom RISC-V targets.</em></p>
</div>

Merlin compiles ML models down to host CPU, the SpacemiT X60 board, and the
Saturn OPU / FireSim flow, with first-class plugins for new dialects, HAL
drivers, and target generators.

## Pick your path

!!! tip "Use Merlin"
    Compile or run models on a supported target. Start here:

    - [Quick start](getting_started.md)
    - [User paths](user_paths.md)
    - [CLI reference](reference/cli.md)

!!! tip "Bring up a target"
    Wire a new accelerator into the codegen + runtime pipeline:

    - [Target Generator](architecture/target_generator.md)
    - [Hardware Backends overview](hardware_backends/overview.md)
    - [Add a Compile Target](how_to/add_compile_target.md)

!!! tip "Develop Merlin"
    Change compiler passes, HAL drivers, or plugins:

    - [Repository Guide](repository_guide.md)
    - [Plugin & Patch Model](architecture/plugin_and_patch_model.md)
    - [How-To index](how_to/index.md)
    - [Workstream logs](dev_blog/index.md)

## Build these docs locally

```bash
conda activate merlin-dev
MLIR_TBLGEN=build/host-vanilla-release/llvm-project/bin/mlir-tblgen \
  uv run --with-requirements docs/requirements.txt python docs/hooks.py
uv run --with-requirements docs/requirements.txt zensical serve
```

Then open <http://127.0.0.1:8000>.
