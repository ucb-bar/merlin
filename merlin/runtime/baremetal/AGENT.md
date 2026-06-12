# AGENT.md — merlin/runtime/baremetal

## Purpose

Merlin-owned bare-metal runtime backend. `spike/` holds the harness (crt, HTIF, linker script, RVV kernel library) used by `merlin/python/merlin/runtime/backends/spike.py` to run command buffers on spike as a multicore RVV CPU.

## What belongs here

- Per-execution-environment harness subdirectories (`spike/`; later real boards).
- C/assembly that is target-independent runtime substrate (Merlin owns the runtime).

## What does not belong here

- Generated per-command-buffer drivers (emitted into work dirs by `rvv_codegen.py`).
- Target-specific runtime models — targets implement adapters only.
- Generated outputs (use gitignored `build/`/`output/`).

## Interfaces

Consumed by `merlin/python/merlin/runtime/backends/` (paths resolved via `merlin.common.paths.repo_root()`).

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.
