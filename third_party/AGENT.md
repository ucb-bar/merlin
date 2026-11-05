# AGENT.md — third_party

## Purpose

Pinned hard build/test dependencies that merlin itself cannot build or test without (e.g. a vendored LLVM/MLIR or lit, if and when needed).

## What belongs here

- True build-time dependencies, pinned for reproducibility.
- Small header-only libs or test utilities required by the build.

## What does not belong here

- External analysis repos (XNNPACK, Autocomp, Exo, Triton, ...) — those are integrations.
- Anything merlin can run without. Prefer a `pyproject.toml` dependency or env-var path.

External baseline compilers and runtimes are selected from independent checkouts through explicit
environment paths. Their adapters remain in `packages/merlin-analysis`; no baseline source tree is
tracked below `third_party/`. The selected framework revision remains part of each measurement.

One exception: `third_party/tvm` is a submodule pin (no vendored source) of the TVM fork
`grammy117/tvm` at `f3cfd1a7d`, the exact compiler behind the published F2 FireSim results
(`examples/gemmini/comparisons/tvm/REPRODUCING_F2_RESULTS.md`). It records which TVM those
numbers belong to. Do not move it without new measurements; the comparison still selects its
TVM through `TVM_SOURCE`, and nothing in Merlin's build consumes it.

## Interfaces

Consumed by the build system (`CMakeLists.txt`, `build_tools/`).

## Invariants

- Only hard build/test dependencies belong here.
- Do not add external analysis repos here by default.
- Prefer 'bring your own LLVM build' over vendoring at this stage.

## Testing expectations

Build must remain green; document any pin and its rationale.

## Notes for future agents

xDSL is a `pyproject.toml` optional dependency by default; only vendor it here if a pinned local copy becomes necessary for CI or local patches.
