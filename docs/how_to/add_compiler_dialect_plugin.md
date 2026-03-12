# Add A Compiler Dialect Plugin

This guide shows how to add a new compiler target plugin in the same style as
Gemmini/NPU.

## 1) Create Dialect + Pass Structure

Under `compiler/src/merlin/Dialect/<YourTarget>/`, use this shape:

- `IR/` for dialect/ops/attrs definitions and C++ impl
- `Transforms/` for lowering/rewrite passes
- `Translation/` for target-specific translation (optional)
- `Register<YourTarget>.*` to register dialect/pass entry points

Use existing stacks as templates:

- `compiler/src/merlin/Dialect/Gemmini`
- `compiler/src/merlin/Dialect/NPU`

## 2) Add The Compiler Plugin Target

Create:

- `compiler/plugins/target/<YourTarget>/CMakeLists.txt`
- `compiler/plugins/target/<YourTarget>/<YourTarget>Options.h/.cpp`
- `compiler/plugins/target/<YourTarget>/PluginRegistration.cpp`

The plugin should:

1. register passes (`registerPasses()`)
2. register dialects (`onRegisterDialects`)
3. hook pipeline stages (usually `extendPostGlobalOptimizationPassPipeline`)

Reference implementations:

- `compiler/plugins/target/Gemmini/PluginRegistration.cpp`
- `compiler/plugins/target/NPU/PluginRegistration.cpp`

## 3) Wire Plugin Into Merlin/IREE CMake

Enable plugin subdir in:

- `iree_compiler_plugin.cmake`

Pattern already used:

- `MERLIN_ENABLE_TARGET_GEMMINI` / `MERLIN_BUILD_GEMMINI`
- `MERLIN_ENABLE_TARGET_NPU` / `MERLIN_BUILD_NPU`

Add your target with the same compatibility style.

## 4) Expose Build-Time Control In `tools/build.py`

Add:

- profile preset (optional but recommended)
- `compiler_scope` handling so users can build only your plugin target
- corresponding CMake flag mapping (`-DMERLIN_ENABLE_TARGET_<TARGET>=ON`)

Current examples:

- profile `gemmini`
- profile `npu`
- `--compiler-scope {all,gemmini,npu,saturn,spacemit,none}`

## 5) Design Integration With IREE Pipeline

Recommended pattern (used by Gemmini/NPU):

1. keep generic MLIR/IREE semantics as long as possible
2. hook at post-global-optimization for target recovery/lowering
3. keep target semantics explicit in target dialect ops
4. only lower to very backend-specific forms when needed

Practically:

- target pass pipeline should be explicit and staged
- keep plugin options for strict/fallback behavior
- add canonicalize/CSE between major lowering steps

## 6) Add Tests Early

At minimum:

- unit lit tests for each pass
- one post-global-opt hook test proving plugin integration

Examples:

- `compiler/src/merlin/Dialect/Gemmini/Transforms/tests/post-global-opt-hook.mlir`
- `compiler/src/merlin/Dialect/NPU/Transforms/tests/post-global-opt-hook.mlir`

## 7) Validate Plugin Visibility + Behavior

Build:

```bash
conda run -n merlin-dev python tools/build.py --profile npu --config release
```

Check loaded plugins:

```bash
build/host-merlin-release/install/bin/iree-compile --iree-list-plugins
```

Run pass-level checks with `iree-opt` / `iree-compile` test commands.

## 8) IREE Example Pointers

Useful upstream examples in this repo checkout:

- `third_party/iree_bar/samples/compiler_plugins/example`
- `third_party/iree_bar/samples/compiler_plugins/simple_io_sample`

Those are good references for plugin mechanics and test structure.
