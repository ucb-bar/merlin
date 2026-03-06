# CMake Presets

Repository presets live in:

- `CMakePresets.json`

These presets target the IREE source tree under `third_party/iree_bar`.

## Example Commands

```bash
# Configure host debug samples build
cmake --preset iree-host-debug-samples

# Build dual-model async runtime sample target
cmake --build --preset iree-host-debug-dual-model
```

RISC-V presets require environment variables:

- `RISCV_TOOLCHAIN_ROOT`
- `IREE_HOST_BIN_DIR`

Example:

```bash
export RISCV_TOOLCHAIN_ROOT=/path/to/spacemit-toolchain
export IREE_HOST_BIN_DIR=/path/to/host/install/bin
cmake --preset iree-riscv-spacemit-debug
cmake --build --preset iree-riscv-spacemit-dual-model
```
