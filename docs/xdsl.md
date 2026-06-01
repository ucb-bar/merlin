# xDSL — the default prototyping plane

xDSL is merlin's **default way of handling MLIR-style IR** until a specific pipeline is chosen.
It lets us iterate on dialects and lowerings in Python without a full LLVM/MLIR build.

## Roles

- **Fast dialect prototyping.** TargetGen's first backend is `xdsl_backend.py`; generate an xDSL
  dialect + parser/printer/verifier + small tests before any C++ TableGen.
- **Research IR** for `merlin.contract / schedule / interface / runtime`
  (`merlin/python/merlin/xdsl_dialects/`).
- **Standalone analysis/lowering playground** for design-pressure and DSE.

## What xDSL is NOT

The single production plane. External compatibility (Hexagon-MLIR, CUDA Tile, IREE, production
target plugins, long-term C++ pipelines) lives in the MLIR/C++ plane (`merlin/compiler/`).

## Two-plane summary

```
Python/xDSL plane:  fast prototypes, generated dialect experiments, design-pressure, DSE, kernel mining
MLIR/C++ plane:     stable dialects, real target plugins, external MLIR integration, production lowering
```

Promote a stable xDSL dialect to MLIR/C++ under `merlin/compiler/` only when needed; keep names
aligned. xDSL is installed as an optional dependency: `uv sync --extra xdsl` (or
`pip install -e '.[xdsl]'`).
