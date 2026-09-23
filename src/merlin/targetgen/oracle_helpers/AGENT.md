# AGENT.md — merlin/python/merlin/targetgen/oracle_helpers

## Purpose

Target-independent subprocess helpers for execution and ISA introspection.
Target-specific assembly and dtype layout live in the selected OOT support provider,
declared by `runner.program_emitter`, not in this package.

## Modules

- `program_cosim.py` — process-isolated Arc program runner; gives every self-hosted-ISA target a real wall timeout.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
