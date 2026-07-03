# AGENT.md — merlin/python/merlin/targetgen/contract

## Purpose

Experiment-ABI contract layer.

## Modules

- `compile.py` — Runner-owned compile + execute of a *package-produced* lowered LLVM/RoCC MLIR.
- `interface_emit.py` — ``merlin_iface`` interface-grammar: emit a Merlin command buffer as contract text, and
- `schemas.py` — Fail-closed JSON-Schema validation against the ``merlin/contract/schemas/`` bundle.
- `toolchain.py` — MLIR toolchain resolution for the experiment ABI (env-overridable).

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
