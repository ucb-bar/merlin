# AGENT.md — merlin/python/merlin/pipelines

## Purpose

xDSL pass registry + pipeline builder. Turns a compilation strategy's `lowering_pipeline` string into a runnable xDSL pipeline (MLIR --pass-pipeline style).

## What belongs here

- Named pass/rewrite registration (`registry.py`) and pipeline assembly (`builder.py`).

## What does not belong here

- Strategy/search logic (that is `dse/` and `search/`).
- Stable C++ passes (those live in `merlin/compiler/`).

## Invariants

- Passes are referenced by name so compilation approaches stay data, not code.
- Keep pass names aligned with the eventual MLIR/C++ pass names for clean promotion.
- No real passes yet — TODO stubs only.
