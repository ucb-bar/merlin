# Merlin Compiler Infrastructure

**Merlin** (**M**L**IR**-**in**) is an end-to-end compiler lowering funnel that connects high-level ML frameworks (PyTorch, ONNX) to custom RISC-V silicon and the [UCB-BAR](https://github.com/ucb-bar) ecosystem via [IREE](https://github.com/openxla/iree).

## Build the repo:

```bash
conda env create -f env_linux.yml
conda activate merlin-dev

python tools/setup.py env
python tools/build.py build --target host --config debug 
```




## ⚡ Overview

Merlin bridges the gap between software models and bare-metal hardware execution. It is designed to support:
1. **Custom Accelerators:** Targeted support for the **Saturn Outer Product Unit (OPU)** via custom microkernels and compiler intrinsics.
2. **End-to-End Lowering:** Compiles standard models (ResNet, MobileNet, Dronet) down to `.vmfb` artifacts for execution on RISC-V softcores (FireSim) and commercial chips.
3. **Benchmarking:** Automated suite (`KernelBench`) to profile performance across CPU, GPU (A100), and RISC-V targets.

## 🏗️ Architecture

- **Frontend:** PyTorch / ONNX / JAX
- **Middle-end:** IREE (Linalg, Flow, Stream dialects) + Custom Merlin Passes
- **Backend:** LLVM CPU / RISC-V (RVV 1.0 + Custom Extensions)
- **Runtime:** IREE HAL (Hardware Abstraction Layer) for Bare-metal

## 📦 Supported Models
Merlin includes pre-configured compilation flows for:
- **Robotics:** Dronet (Collision Avoidance), FastDepth
- **Vision:** MobileNetV2, GLPDepth
- **General:** MLPs, Diffusion Policies

## 🚀 Getting Started

### Prerequisites
- IREE Compiler v3.8.0+
- RISC-V Toolchain (for cross-compilation)
- Ninja Build System

### Compiling a Custom Kernel (Saturn OPU)
Merlin allows dispatching specific operations to custom hardware units:

```bash
# Configure for RISC-V Cross-Compilation
cmake -G Ninja -B build-riscv \
    -DCMAKE_TOOLCHAIN_FILE=../toolchain.cmake \
    -DIREE_HOST_BIN_DIR=../host/bin

# Build the custom dispatch sample
cmake --build build-riscv --target compile_custom_model
```

## Repository Workflows

For a small maintainer team, use the unified entrypoint:

```bash
python3 tools/merlin.py --help
python3 tools/merlin.py targets list
```

Core routines:

```bash
# Patch-stack lifecycle
python3 tools/merlin.py patches apply
python3 tools/merlin.py patches verify
python3 tools/merlin.py patches drift

# Lint + script sanity
python3 tools/merlin.py ci lint

# Build profile wrappers
python3 tools/merlin.py build host-release
python3 tools/merlin.py build riscv-spacemit-dual-model

# Upstream release tracking
python3 tools/merlin.py release-status
python3 tools/merlin.py release-status --json
```

## Project Structure (Maintained Paths)

- `compiler/`: Merlin-owned compiler/plugin logic.
- `patches/`: IREE/LLVM patch stack, manifests, and patch tooling.
- `scripts/`: build helpers (`scripts/legacy/` is archived reference-only).
- `tools/`: stable developer/CI entrypoints.
- `samples/`: runnable runtime examples and sample code.
- `benchmark/target/<board>/`: deployment + profiling flows per hardware target.
- `config/`: canonical small config files consumed by `tools/merlin.py` and CI.

Further maintenance/process docs:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [UPSTREAM_SYNC.md](UPSTREAM_SYNC.md)
- [docs/architecture/plugin_and_patch_model.md](docs/architecture/plugin_and_patch_model.md)
- [docs/architecture/cmake_presets.md](docs/architecture/cmake_presets.md)
