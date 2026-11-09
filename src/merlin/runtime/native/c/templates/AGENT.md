# AGENT.md — src/merlin/runtime/native/c/templates

## Purpose

Portable C runtime templates specialized from independently validated typed
source contracts. Resolve templates through `merlin.common.paths.data_path`
and declare their transitive resources in `build_tools/package_resources.json`.

## Invariants

- Dimensions and constants come from source operations and legality proofs.
  Model names, captured IDs and golden outputs never select an implementation.
- ISA operations and physical target ABI glue belong to the OOT provider.
- Mutable storage belongs to the caller. Require explicit lifetime, capacity,
  alignment and disjointness contracts; publish outputs only on success.
- Retain the actual original source computation for runtime refusal.
- Internal helper symbols must not collide when multiple plans are linked.
- Source rounding, arithmetic order, consumer observations and exception
  contracts remain explicit. Compiler capabilities are optional and qualified
  independently from numerical eligibility.

## Verification

Use native source oracles, different legal plans in one link, dirty workspace
reuse, strided inputs and refusal cases. Hardware performance and target ISA
audits are supplied by the corresponding provider.
