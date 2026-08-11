# AGENT.md — merlin/python/merlin/triton

## Purpose

Triton as a target-independent KERNEL FRONTEND to Merlin (not a per-target backend).

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Hard rules for this package

Design + the numbered invariants: `docs/design/triton_frontend.md`. Guarded by
`merlin/tests/infra/test_triton_target_independence.py` (run it before committing here).

- **Target-blind.** No target-name literal, no `merlin.targets.*` / `merlin.rvvgen.*` import. The
  target is a parameter threaded in from the contract. If making a second target work requires
  editing a file here, the abstraction failed — fix Merlin's shared lowering instead.
- **No `import re`.** TTIR is parsed structurally (tokenizer / xDSL IR). A regex line-matcher
  silently drops valid-but-differently-spelled input; that failure mode has already cost this repo
  real mis-measurements (see the RoCC trace decoder history).
- **linalg-on-tensors is the output contract.** Never lower TTIR toward a target dialect from here —
  that bypasses contract inference, scheduling and interface materialization. And the bridge must
  emit **tensor-typed func arguments feeding the matmul directly**: `lower_to_interface` maps
  operands through a `value_map` built only from `src_block.args`, and
  `contract_facts._trace_to_block_arg` walks only linalg/tensor view ops, so a memref- or
  bufferization-boundaried operand either raises `KeyError` or silently loses residency.
- **Fail closed.** An unsupported TTIR op, an unproven pointer pattern, or a dropped mask is a
  located error, never a best-effort lowering. Record `UNKNOWN` and surface it; do not substitute a
  default.
- **One module owns Triton internals.** Only `source.py` may import from `triton.compiler` /
  `triton.backends`; those APIs move fast, and the blast radius has to stay one file.
- **Artifacts** go through `artifacts.new_product("triton-kernel", target=…, version=0)` and
  `cache_dir("triton")` — never a bespoke output root.
