# models/

Inputs to `./merlin compile` plus the per-target compile-flag bundles.

- `<model_name>/` (e.g. `dronet/`, `mlp/`, `tinyllama/`) — model assets:
  source `.py` exporters, exported `.mlir` / `.onnx`, sometimes test data.
- `*.yaml` (e.g. `spacemit_x60.yaml`, `saturn_opu.yaml`, `npu_ucb.yaml`,
  `gemmini_mx.yaml`) — target views consumed by `./merlin compile --target …`.
  These bundle the IREE flags, ukernel selection, and pipeline overrides for
  a given hardware target.
- `models_config.json` — meta-config used by helper scripts.
- `quantize_models.py` — shared INT8 / FP8 quantization helper invoked from
  per-model exporters.
- `compiled_models/` — generated outputs (gitignored). Build artifacts land
  under `build/compiled_models/<model>/<target>/` instead.

Compile a model: `./merlin compile models/dronet/dronet.mlir --target spacemit_x60`.
