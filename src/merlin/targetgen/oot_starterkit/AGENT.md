# AGENT.md — merlin/python/merlin/targetgen/oot_starterkit

## Purpose

OOT starter kit — hw-agnostic, answer-free framework plumbing for authoring an MLIR OOT backend.

## Modules

- `cmdbuf.py` — Output plumbing: build a SCHEMA-VALID command_buffer.json (the frozen ABI).
- `dialect.py` — Expose the framework's TYPED merlin_iface input dialect — parse into VERIFIED xDSL IR (the C++ benefit).
- `iface.py` — Input plumbing: parse the fixed `merlin_iface` interface grammar into a plain model.
- `llvm_context.py` — Public typed LLVM parsing that preserves standard compiler metadata and original verification checks.
- `plan.py` — Public exact-source operation inventory and structural mixed whole-program plan preflight; never execution or numerical proof.
- `transforms.py` — Generic, target-AGNOSTIC compiler transforms the agent calls. NOT target-specific lowerings.
- `verify.py` — Structural verification for the Python/xDSL path — the C++-MLIR-verifier equivalent.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

The public plan inventory verifies the actual terminating return against its
original declared function result types before exposing ordered value bindings.
Arguments, repeated returns and individual result indices retain their identity;
unsupported multi-block joins refuse. This boundary check is structural only and
does not establish body semantics, effects, lowering or runtime authority.

LLVM function `memory_effects` annotations retain their exact standard attribute
kind and payload through parsing, verification and printing. The stock compiler
owns payload legality; this parser neither infers body effects nor relaxes
pointer storage correspondence or the original operation checks.
