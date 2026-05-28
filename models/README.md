# models/

Inputs to `./merlin compile` plus the per-target compile-flag bundles.

- `<model_name>/` (e.g. `dronet/`, `mlp/`, `tinyllama/`) — model assets:
  source `.py` exporters, exported `.mlir` / `.onnx`, sometimes test data.
- `*.yaml` (e.g. `spacemit_x60.yaml`, `saturn_opu.yaml`, `npu_ucb.yaml`,
  `gemmini_mx.yaml`) — target views consumed by `./merlin compile --target …`.
  These bundle the IREE flags, ukernel selection, and pipeline overrides for
  a given hardware target.
- `models_config.json` — registry of well-known models (source path +
  input shapes). Consumed by `./merlin quantize --model <name>`.
- `compiled_models/` — generated outputs (gitignored). Build artifacts land
  under `build/compiled_models/<model>/<target>/` instead.

Quick recipes:

```bash
# Compile to a hardware target
./merlin compile models/dronet/dronet.mlir --target spacemit_x60

# Quantize a registered model (looks up shapes from models_config.json)
./merlin quantize --model dronet

# Quantize an arbitrary .onnx by passing shapes directly
./merlin quantize path/to/model.onnx --shape 1,3,224,224
```
