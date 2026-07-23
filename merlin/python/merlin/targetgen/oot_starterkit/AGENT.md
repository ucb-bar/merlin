# AGENT.md — merlin/python/merlin/targetgen/oot_starterkit

## Purpose

OOT starter kit — hw-agnostic, answer-free framework plumbing for authoring an MLIR OOT backend.

## Modules

- `cmdbuf.py` — Output plumbing: build a SCHEMA-VALID command_buffer.json (the frozen ABI).
- `dialect.py` — Expose the framework's TYPED merlin_iface input dialect — parse into VERIFIED xDSL IR (the C++ benefit).
- `iface.py` — Input plumbing: parse the fixed `merlin_iface` interface grammar into a plain model.
- `transforms.py` — Generic, target-AGNOSTIC compiler transforms the agent calls. NOT target-specific lowerings.
- `verify.py` — Structural verification for the Python/xDSL path — the C++-MLIR-verifier equivalent.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->
