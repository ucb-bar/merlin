# Add A Compile Target (`tools/compile.py`)

`tools/compile.py` uses YAML target files in `models/` to define compile flags.

## 1) Add A New Target YAML

Create:

- `models/<target_name>.yaml`

Current in-tree example:

- `models/spacemit_x60.yaml`
- `models/npu_ucb.yaml`
- `models/gemmini_mx.yaml`

## 2) YAML Schema Used By `tools/compile.py`

Common keys:

- `default_hw`: default hardware profile when `--hw` is omitted
- `generic`: base flags always applied
- `targets`: per-hardware flag lists (keyed by `--hw` values)
- `quantized` or `quantized_<type>`: extra flags for quantized mode
- `models`: per-model overrides
- `plugin_flags`: optional plugin-related flags appended with generic flags

`tools/compile.py` currently merges flags in this order:

1. `generic` + `plugin_flags`
2. `targets[--hw]` (if used)
3. quantized flags
4. model-specific overrides

When `plugin_flags` is non-empty and `--build-dir` is left at its default
(`host-vanilla-release`), `tools/compile.py` automatically uses
`host-merlin-release` so plugin-enabled targets work with short user commands.

## 3) Compile With Your New Target

Examples:

```bash
conda run -n merlin-dev uv run tools/compile.py models/dronet/dronet.mlir --target <target_name>
```

```bash
conda run -n merlin-dev uv run tools/compile.py models/dronet/dronet.mlir --target <target_name> --hw <hw_profile>
```

## 4) Output Layout

Artifacts are emitted under:

- `build/compiled_models/<model_name>/<target_and_hw>_<basename>/`

Typical outputs:

- `<basename>.mlir`
- `<basename>.vmfb`
- optional dumps/phases/graph/benchmarks if enabled

## 5) Tool Selection Behavior

`tools/compile.py` picks `iree-compile` from:

1. `build/<build_dir>/tools/`
2. `build/<build_dir>/install/bin/`
3. fallback `build/host-merlin-release/tools/`
4. fallback `build/host-merlin-release/install/bin/`
5. fallback current environment

Control primary location with:

- `--build-dir <build_dir_name>`
- `--compile-to <phase_name>`
- `--dump-compilation-phases-to <dir>`
- `--iree-compile-arg <flag>` / `--compilation-custom-arg <flag>` (repeatable passthrough)
- `--reuse-imported-mlir` (skip refresh of copied/imported MLIR)

## 6) When To Add A New Target vs. New `--hw`

Prefer:

- new YAML target when overall toolchain/backend family differs
- new `--hw` entry inside one YAML when backend family is same but micro-arch
  flag bundles differ

## 7) Optional: Add Convenience Build Profile

If the new compile target needs special compiler plugin/runtime configuration,
also add a `tools/build.py` profile so the build command is easy to reproduce.
