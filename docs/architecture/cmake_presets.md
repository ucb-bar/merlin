# CMake Presets

> **When to read this:** only if you are configuring IREE directly with
> `cmake --preset` (e.g. running upstream IREE workflows, debugging a
> preset definition, or porting flags into Merlin's build wrapper).
> **For day-to-day Merlin builds use `./merlin build`** — it composes the
> same flags via `tools/build/presets.py` and is the path documented in
> [`how_to/use_build_py.md`](../how_to/use_build_py.md). Per
> `AGENTS.md` Golden Rule #1, do not invoke `cmake` directly for routine
> builds.

The repo ships a `CMakePresets.json` at the root. Its presets target the
IREE source tree under `third_party/iree_bar` and are the source of
truth for what flags `./merlin build` ultimately resolves.

## Available presets

| Preset | Purpose | Toolchain |
|---|---|---|
| `iree-host-debug-samples` | Host x86 IREE compiler + samples, debug build | Native (Ubuntu Clang/GCC) |
| `iree-host-debug-dual-model` | Host x86, dual-model async runtime sample target | Native |
| `iree-riscv-spacemit-debug` | SpacemiT X60 cross-compile, debug | Requires `RISCV_TOOLCHAIN_ROOT` + `IREE_HOST_BIN_DIR` |
| `iree-riscv-spacemit-dual-model` | SpacemiT X60 dual-model sample target | Same as above |

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
