# AGENT.md — src/merlin/compiler/lower/llvm

## Purpose

Shared MLIR-to-LLVM finalization and ABI contracts.

## Modules

- `abi.py` — Compiler-owned pointer ABI compaction with an explicit byte-offset contract.
- `normalize.py` — Normalize MLIR-printer LLVM literals for LLVM text parsing.
- `passes.py` — Shared terminal MLIR-to-LLVM conversion for scalar and extension pipelines.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
