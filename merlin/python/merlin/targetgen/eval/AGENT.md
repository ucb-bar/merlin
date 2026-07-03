# AGENT.md — merlin/python/merlin/targetgen/eval

## Purpose

Merlin evaluation/recording helpers (conformance batteries, aet suites).

## Modules

- `gemmini_conformance.py` — Gemmini conformance battery — command-buffer builders for the C-rungs.
- `gemmini_contract_sweep.py` — Migrate the Gemmini conformance battery THROUGH the experiment-ABI contract runner.
- `gemmini_dispatcher.py` — Resumable cartesian conformance dispatcher (abc-testing-style).
- `gemmini_suite.py` — aet recording substrate for Gemmini conformance runs.
- `saturn_vec_conformance.py` — Saturn-vectors conformance battery — a NON-matmul (vector/SIMD) family.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
