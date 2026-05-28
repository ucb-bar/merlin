"""Backend-agnostic kernel-embedding pipeline.

Modules:

- `discover.py`    — Walk an MLIR module, classify linalg ops, emit a
  manifest skeleton.
- `manifest.py`    — Kernel manifest schema (YAML) + loader + validation.
- `precompile.py`  — Source-language → object-artifact dispatch. Knows
  about cl/glsl/spirv/c/cpp/ll/qnn-context-binary; delegates QNN-specific
  steps to `kernels.qnn.precompile_extras`.
- `spec_gen.py`    — Emit a transform-dialect spec MLIR from a kernel
  manifest, with per-target HAL attributes.
"""
