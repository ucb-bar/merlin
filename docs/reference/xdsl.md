---
title: xDSL prototyping plane
kind: reference
status: current
owner: ir
last_verified: 2026-07-14
related: [dialects, core_dialects]
code_refs: [merlin/python/merlin/xdsl_dialects]
---

# xDSL — the default prototyping plane

xDSL is merlin's **default way of handling MLIR-style IR** until a specific pipeline is chosen.
It lets us iterate on dialects and lowerings in Python without a full LLVM/MLIR build.

## Roles

- **Fast dialect prototyping.** TargetGen's dialect plane is xDSL: `xdsl_dialects/targets/factory.py::build_dialect`
  synthesizes an xDSL dialect (IRDL ops/types → parser/printer/verifier) from the target's `dialect_plan`
  before any C++ TableGen.
- **Research IR** for `contract / schedule / interface / runtime / dse`
  (`merlin/python/merlin/xdsl_dialects/`).
- **Standalone analysis/lowering playground** for design-pressure and DSE.

## What xDSL is NOT

The single production plane. External compatibility (Hexagon-MLIR, CUDA Tile, IREE, production
target plugins, long-term C++ pipelines) would live in a stable MLIR/C++ plane — **not yet built**;
see `docs/design/compiler_plane.md`.

## Two-plane summary

```
Python/xDSL plane:  fast prototypes, generated dialect experiments, design-pressure, DSE, kernel mining
MLIR/C++ plane:     stable dialects, real target plugins, external MLIR integration, production lowering
```

Promote a stable xDSL dialect to a future MLIR/C++ plane only when needed (see
`docs/design/compiler_plane.md` — create `merlin/compiler/` with real TableGen/CMake then); keep names aligned. xDSL is installed as an optional dependency: `uv sync --extra xdsl` (or
`pip install -e '.[xdsl]'`).
