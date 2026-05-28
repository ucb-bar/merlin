"""Merlin kernel-embedding framework.

A Python library — *not* a `./merlin` tool — that the compile pipeline
imports to substitute MLIR dispatches with pre-built per-backend kernel
artifacts. Lives at the top of the repo (alongside `compiler/`, `runtime/`,
`models/`) because it's first-class infrastructure, not a developer CLI.

Layout:

- `kernels.core`   — backend-agnostic pipeline (`discover`, `manifest`,
  `precompile`, `spec_gen`). Every backend reuses these.
- `kernels.qnn`    — Qualcomm QNN kernel embedding (emitter, IR,
  partition, route, build, recognizers, headers, tests).
- `kernels.spike`  — Spike simulator runner for RISC-V kernels.

To add a new backend (e.g. Gemmini, SaturnOPU), create `kernels/<name>/`
with a `build.py` that produces object artifacts and an optional
`emit.py` that lowers MLIR. The core pipeline picks it up via the
`source_lang` dispatch in `core.precompile`.
"""
