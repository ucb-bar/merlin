# AGENT.md — merlin/python/merlin/targetgen/contract

## Purpose

Experiment-ABI contract layer.

## Modules

- `compile.py` — Runner-owned compile + execute of a *package-produced* lowered LLVM/RoCC MLIR.
- `build_recipe.py` — Pure compile/link recipe; the runtime backend re-exports this same class.
- `build_service.py` — Typed host-only build service and isolated pure target-package loader; no backend discovery, execution or reference imports.
- `interface_emit.py` — ``merlin_iface`` interface-grammar: emit a Merlin command buffer as contract text, and
- `schemas.py` — Fail-closed JSON-Schema validation against the ``merlin/contract/schemas/`` bundle.
- `toolchain.py` — MLIR toolchain resolution for the experiment ABI (env-overridable).

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

`materialize.py` owns public copying, explicit-ceiling cohort publication, admission
validation, coverage and leases. It never selects evaluator adapters. Use core
`materialize_public_cohort` with an explicit ceiling; default evaluated selection is
`merlin_experiments.corpus.admission.public_capsules_for`, not a lazy core export.
