---
title: Design note: integration adapters
kind: design
status: current
owner: core
last_verified: 2026-09-20
related: [integrations, architecture]
code_refs: [src/merlin/integrations/modelir.py, src/merlin/integrations/model2mlir.py, src/merlin/frontends/quant_ext.py, src/merlin/llvmlower/toolchain.py, src/merlin/kernels/ingest]
---

# Design note: external-tool integration adapters

## Ownership boundaries

An integration adapts an upstream contract; it does not create a second implementation of
that upstream's semantics. Optional frameworks must not be imported by ordinary compiler CLI
discovery. Missing tools fail at their point of use with a configuration error.

| Upstream | Upstream owns | Merlin owns |
| --- | --- | --- |
| AET | Agent accounting, trajectories, run storage | Phase definitions, grading policy, adapters to AET records |
| model2MLIR | Framework capture and model loading | Compiler-facing capture bundle, lowering and target compilation |
| PyTorch / torchao | Framework and quantization semantics | Reading captured contracts; no competing quantization implementation |
| ModeLIR (`mlc`) / CIRCT | RTL extraction and model execution | Consuming derived facts, provenance checks, target/compiler policy |
| SpecIR | Specification interpretation and reference generation | Capsule selection, packaging and independent certification |
| LLVM / MLIR / xDSL | Compiler infrastructure and bindings | Merlin dialects, schedules, passes and target-independent generation |

Compiler Python is configured separately from model capture using `MERLIN_COMPILER_PYTHON`
or `MERLIN_COMPILER_VENV`. The old model2MLIR environment remains a compatibility fallback,
not a requirement for a new configuration. `build_tools/toolchains/compiler-python.toml`
records the observed binding versions; it explicitly does not certify an independently rebuilt
environment. External tool versions and hardware pins remain part of result provenance.

`integrations.model2mlir` owns the checkout aliases shared by capture bundles and the opt-in
quant frontend. Its root resolver preserves capture's historical ordering: either process
alias before either `.env` alias, and `MERLIN_MODEL2MLIR` before `MERLIN_M2M_DIR` within each
source. An invalid preferred root does not select an alternate checkout. The public
`capture.bundle.model2mlir_root()` remains a delegator, including its historical unset placeholder.

Shared framework capture uses `MERLIN_M2M_PYTHON`, then `MERLIN_M2M_VENV`, then the checkout's
`.venv/bin/python`; each setting honors process environment before `.env`. Missing/nonexecutable
choices fail explicitly, and venv executable paths are not resolved to their system symlink.
GGUF capture uses this interpreter, never compiler Python. Per-workload
`capture.bundle.capture_python(model)` still reads that workload's `capture.toml` and retains
its distinct optional-result policy; the remaining native capture-worker defaults are unchanged.

Quant imports temporarily add the configured checkout to `sys.path` and restore the path on
success or failure. Absence is not cached. Loaded classes retain normal Python module identity:
a conflicting or unverifiable preloaded `m2m` graph yields optional unavailability, without
purging modules. This does not support switching framework checkouts inside one process or
provide a Python sandbox. The default unregistered quant parsing path is unchanged.

The concrete `src/merlin/integrations/modelir.py` adapter scopes fallback imports to the call
and restores `sys.path` even on failure. Existing RTL, specification and frontend adapters
retain their established interfaces during this migration. This is not a claim that all
upstream APIs are now normalized or that upstream semantic defects have been corrected.

## Adding an adapter

Implement an adapter in its owning distribution when a concrete need arises; do not keep
an empty adapter skeleton. Compiler-used adapters belong in core; study-only adapters belong
in their optional package. Keep independent verification oracles independent of the code
they check, even where algorithms appear similar.

The original design reserved `merlin/integrations/<tool>/` (XNNPACK, Autocomp, Exo, Triton, xDSL,
IREE, CUDA-Tile, Hexagon-MLIR, OpenBLAS) for lightweight **adapters** that parse/index/normalize an
external project (passed by path/env, never vendored) and emit merlin schema artifacts
(`kernel_record` / `abstraction_candidate` / `policy_rule`, per `merlin/schemas/`). Every dir was
intent-only (`README.md` + `manifest.yaml` + `AGENT.md`, zero `.py`), so it was removed.

**When implementing one:** use a `kernels/ingest/` source or a small `integrations`
subpackage, gated by explicit configuration (for example `MERLIN_<TOOL>_REPO`), emitting
the normalized schema artifacts. Kernel-mining already ingests several external kernel sources this
way under `src/merlin/kernels/`.
