# SmolVLA Export Scripts

This directory provides a stable interface to export SmolVLA from
`third_party/Understanding-PI0` into three MLIR variants:

- `smolVLA.mlir` (fp32 export path)
- `smolVLA.q.int8.mlir` (forced int8 quantization path)
- `smolVLA.q.fp8.mlir` (mixed fp8/int8 quantization path)

## Quick Start

From repo root:

```bash
conda run -n merlin-dev uv run models/smolVLA/export_smolvla.py --mode all --device cuda
```

Outputs are written to `models/smolVLA/` by default.

If you do not see `smolVLA*.mlir` files yet, run the export command first.
These MLIR artifacts are generated outputs, not committed static files.
They are gitignored in this repo.

```bash
ls -lh models/smolVLA/smolVLA*.mlir
```

## Export Modes

- `--mode fp32` exports only `smolVLA.mlir`
- `--mode int8` exports only `smolVLA.q.int8.mlir`
- `--mode fp8` exports only `smolVLA.q.fp8.mlir`
- `--mode all` exports all three

## Notes

- The wrapper prefers `third_party/Understanding-PI0/.venv/bin/python` if it
  exists. Override with `--python /path/to/python`.
- Use `--dry-run` to print commands without executing.
- Common shape/model flags are forwarded to underlying export scripts
  (`--model-id`, `--batch-size`, `--image-h`, `--image-w`, `--prompt-len`,
  `--no-vision`, `--skip-patches`, and `--no-exportable-mx`).

## Compile via `tools/merlin.py`

All user-facing compilation in this repo should go through
`tools/merlin.py compile`.
Example commands from repo root:

Before building or compiling plugin targets on a fresh Merlin clone, check out
the intended Merlin branch and sync its pinned submodules:

```bash
git checkout dev/main
conda activate merlin-dev
uv run tools/merlin.py setup submodules --submodules-profile core --submodule-sync
```

If you switch Merlin branches later, rerun the same submodule setup command
before rebuilding `host-merlin-release`.

```bash
# Baseline target (spacemit settings)
conda run -n merlin-dev uv run tools/merlin.py compile \
  models/smolVLA/smolVLA.q.fp8.mlir \
  --target spacemit_x60 \
  --quantized \
  --compile-to global-optimization \
  --dump-phases
```

```bash
# NPU-targeted global-opt lowering path
conda run -n merlin-dev uv run tools/merlin.py compile \
  models/smolVLA/smolVLA.q.fp8.mlir \
  --target npu_ucb \
  --quantized \
  --compile-to global-optimization \
  --dump-phases
```

```bash
# Gemmini-targeted global-opt matching path
conda run -n merlin-dev uv run tools/merlin.py compile \
  models/smolVLA/smolVLA.q.fp8.mlir \
  --target gemmini_mx \
  --quantized \
  --compile-to global-optimization \
  --dump-phases
```

The SmolVLA-specific compile flags are configured in target YAML files, so these
commands stay short. For plugin targets (`npu_ucb`, `gemmini_mx`), the compile
wrapper auto-selects `host-merlin-release` when needed.

If compilation fails on `torch.prims.device_put`, that usually means the local
compiler build is stale or missing the patched
`third_party/iree_bar/third_party/torch-mlir` lowering, not that the model must
be exported with `--device cpu`. Rebuild the compiler and retry:

```bash
conda run -n merlin-dev uv run tools/merlin.py build \
  --profile npu \
  --config release \
  --no-build-python-bindings \
  --no-enable-libbacktrace
```

Compiled artifacts are written under `build/compiled_models/` (not in `models/`)
so generated outputs do not get committed accidentally.

Inspect the two required dump files each run (example for `npu_ucb`):

```bash
ls -lh \
  build/compiled_models/smolVLA/npu_ucb_RVV_smolVLA.q.fp8/phases/module.1.input.mlir \
  build/compiled_models/smolVLA/npu_ucb_RVV_smolVLA.q.fp8/phases/module.4.global-optimization.mlir
```
