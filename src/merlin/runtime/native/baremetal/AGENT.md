# AGENT.md — src/merlin/runtime/native/baremetal

## Purpose

Merlin-owned bare-metal runtime backend. `spike/` holds the harness (crt, HTIF, linker script, RVV kernel library) used by `merlin/python/merlin/runtime/backends/spike.py` to run command buffers on spike as a multicore RVV CPU.

## What belongs here

- Per-execution-environment harness subdirectories (`spike/`; later real boards).
- C/assembly that is target-independent runtime substrate (Merlin owns the runtime).
- `out_b64.h` provides opt-in lossless container-word framing through a caller's
  block text writer. The bounded staging buffer contains actual output words;
  range-derived narrowing preserves their values and never consults references.
  The caller owns layout traversal, frame declarations and terminal completion.
- `out_bin.h` is a separate opt-in binary container-word packer. Its bounded
  buffer passes explicit lengths to a caller-supplied coherent byte writer, so
  arbitrary payload bytes are retained. It changes no `out_b64.h` behavior and
  grants no numerical or target support.

## What does not belong here

- Generated per-command-buffer drivers (emitted into work dirs by `rvv_codegen.py`).
- Target-specific runtime models — targets implement adapters only.
- Generated outputs (write to `runs/`/`artifacts/`; compiled trees to `build/`).

## Interfaces

Consumed by `merlin/python/merlin/runtime/backends/` (paths resolved via `merlin.common.paths.repo_root()`).

## Invariants

- Keep this directory focused on its stated purpose.
- Every subdirectory must also contain an AGENT.md.
