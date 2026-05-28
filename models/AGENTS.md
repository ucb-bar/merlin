# `models/` — agent guide

## Mental model

Two distinct things share this folder:

| Subtree | Role |
|---|---|
| `<model_name>/` (e.g. `dronet/`, `tinyllama/`, `yolov8_nano/`, …) | Per-model assets: source `.py` / `.onnx` / `.mlir`, calibration data, sometimes test fixtures. One folder per model. |
| `*.yaml` (e.g. `spacemit_x60.yaml`, `gemmini_mx_vcs.yaml`, `qrb5165_qnn.yaml`) | **Target views** consumed by `./merlin compile --target <name>`. They bundle `iree-compile` flags + ukernel selection + per-model overrides for a hardware target. |
| `models_config.json` | Registry of well-known models (source path + input shapes). Consumed by `./merlin quantize --model <name>`. |

## YAML schema

```yaml
default_hw: <variant>            # default --hw value if user omits
generic:                         # base flags, always applied
  - --iree-...
plugin_flags:                    # merlin-plugin flags (optional)
  - --iree-plugin=<backend>
targets:                         # per-hw extra flags appended to generic
  VARIANT_A: [--iree-...]
  VARIANT_B: [--iree-...]
quantized:                       # appended when input is .q.int8.onnx
  - --iree-...
models:                          # appended when --model-name matches
  yolov8_nano:
    - --iree-preprocessing-...
```

Loader: `tools/compile/cli.py` (search for `cfg = yaml.safe_load`).

## Pitfalls

- **Flag order matters for last-wins semantics.** When `targets[hw]`
  needs to override a `plugin_flags` value, the per-hw entry appears
  AFTER `generic` + `plugin_flags` in the final flag list. Verify with
  `--dry-run` before assuming an override took effect.
- **`default_hw` must be a key in `targets:`** when `targets:` exists.
  Otherwise the loader errors out. Empty list (`HW_NAME: []`) is fine
  for "no extra flags, but valid hw name".
- **Don't duplicate flags across sister yamls.** Sister variants
  (e.g. `mx_vcs` vs `mx_vcs_fp4`) should share one YAML with multiple
  `targets:` entries — see [[project_opu_benchmark_suite]] for the
  consolidation rationale.
- **`models_config.json` paths are relative to `models/`.** Don't make
  them absolute — breaks for forks/checkouts elsewhere.

## Cross-references

- Consumed by `./merlin compile` (the YAML target loader at
  `tools/compile/cli.py:233`).
- Consumed by `./merlin quantize --model <name>` (registry lookup at
  `tools/quantize/cli.py:_resolve_registry_entry`).
- The CLI reference is auto-generated; the canonical surface lives in
  this folder, not in docs.
- Newer canonical hardware capability specs live in `target_specs/` —
  see that subtree's AGENTS.md for the distinction.

## Update triggers

Re-read this file and update it in the same turn if you:

- Add or rename `models/<target>.yaml` — update the layout table; touch
  `docs/different_build_types.md` and `docs/how_to/use_build_py.md` if
  the public-facing target name changes.
- Edit the YAML schema (a new top-level key consumed by the loader) —
  bump the "YAML schema" section and the loader at
  `tools/compile/cli.py`.
- Modify `models_config.json` schema — `./merlin quantize` and
  `tools/quantize/cli.py:_resolve_registry_entry` both consume it.
