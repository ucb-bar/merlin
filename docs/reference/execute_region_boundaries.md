---
title: Execute-region boundaries and source attributes
kind: reference
status: current
owner: frontends
last_verified: 2026-11-11
related: [linalg_named_regions, loop_tensor_destinations, architecture]
code_refs: [src/merlin/xdsl_dialects/standard_scf.py, src/merlin/frontends/linalg_mlir.py, merlin/tests/ir/test_execute_region_attributes.py]
---

# Execute-region boundaries and source attributes

The standard SCF reader retains every `scf.execute_region` attribute through
custom and generic syntax. xDSL 0.68's original custom parser passes the parsed
dictionary to a constructor that discards it. A prepared model can consequently
lose both source provenance and `no_inline` before an otherwise unrelated host
rewrite prints generic IR again.

This affected byte-backed atomic recurrences in ordinary whole-model lowering.
Six protected snapshots lost their boundaries; destination preparation then
missed three legal carried-state reuses, and the unchanged native lifetime gate
rejected twelve premature/duplicate releases. Preserving the original source
fields lets the existing complete snapshot and ownership proofs run. No
lifetime, borrowing, effect or arithmetic requirement is relaxed.

`standard_scf.ExecuteRegionOp` retains the complete attribute dictionary,
supports the native `no_inline` keyword and existing dictionary spelling, and
prints explicit terminators for void regions. `no_inline` must be a unit
attribute. Empty regions and entry block arguments remain invalid; interior CFG
blocks, typed results and source attributes retain their original structure.

The owned Python canonicalization trait delegates ordinary unmarked
single-block inlining to xDSL's existing implementation. Marked regions remain
until an explicit lowering. Replacing that inherited inliner preserves the
other inherited traits. The source flag requests boundary preservation; it
supplies no proof of purity, storage independence or source equivalence.

Tests cover both syntax forms, provenance, native void printing, marked and
unmarked Python canonicalization, multiblock CFGs and malformed attributes or
regions. Actual model qualifications independently check every result/state
word, source parameter and readonly input through ordinary target execution.
