# Scoped configuration elision — reviewed Q3/F3 result

Integrated into feat/pass-verification by f6ab5cdc3, source commit bc2ba713f.
This validates complete supported straight-line modules using the existing Lean
known-state checker and the new complete-entry correspondence proof. It does not
register a new compiler gate or approve instruction effects.

`ActualConfigurationElision.complete_entries` preserves all modeled entry lookups
and complete successful/error/terminal states, under a common admitted
interpretation. Initial configuration is arbitrary. An omitted setter must be a
total exact pure overwrite; unknown/unordered effects clear knowledge, clobbers
invalidate fields, and all other instructions and SSA bindings remain retained.
The state must include caller/return/observer/SSA data, with a nonaliasing field
mapping. Actual Python/xDSL projection, source/ISA/effect decoding, native/hardware
correspondence and independent human review remain explicit trust boundaries.

Q3/F3 qualification covered thirteen actual supported pairs and one unsupported
dynamic-operand pair; twenty-one valid mutations refused by the actual Lean
closed obligation, plus one valid reserved-metadata mutation refused by host
admission. It retained 87 compiler-valid buffers with their 83+1+3 source lineage,
fifteen forged-request controls, eleven live artifact controls and a fresh
separate-Python replay. Each supported query audited seventeen roots and 247
transitive theorems, standard three axioms only, zero violations. Do not add
overlapping audit totals.

The actual multifield example has independently checked constructive nonzero
success and first-error witnesses: observer21, error73 and the same terminal
state, derived directly from the source execution. The witness audit covered
110 roots and 276 transitive theorems. In the separate read-only type comparison,
109 printed types match exactly; one differs only by a consistently renamed
compiler-generated bound variable. Its raw difference is preserved in the W
packet. No source-success or whole-run-equality premise replaces the generic
soundness argument.

Historical failures remain: Q1/F1 was unavailable before proof elaboration because
of an incorrect compiler working directory. Q2/F2 passed its selected tests but
is explicitly UNQUALIFIED: xDSL silently overwrote a reserved raw attribute.
The final structural lexer admission fix rejects bare, quoted, escaped and unit
`op_name__` keys before parsing, with the existing thirteen query/root bundles
unchanged. Neither earlier campaign was relabeled as the corrected qualification.

Primary integration passed 211 focused SDK-free tests and the structure gate.
The first primary test attempt had 22 missing-fixture setup errors; copying the
exact retained corpus/selection files resolved them without source/proof changes.
A fresh primary baseline check delivered exact final bytes; its separate-process
replay and reports are in COORDINATOR/primary-check. These repeat the baseline,
not thirteen new independent cases or a new-host qualification.

Start with [F's final status](F/FINAL_STATUS.json),
[the publication index](PUBLICATION.json), and
[the developer API guide](../../../../../../docs/guides/lean_configuration_elision_gate.md).
The archives retain exact source/query/proof/process text and the old failed
profiles. Compiled native objects and symlink attack fixtures remain in their
original campaigns and are explicitly excluded by member manifests. Rebuild
fresh objects for replay; copied reports give no live authority. Required
external producer/SDK/Python/xDSL identities are recorded, not proved faithful.

Full-pass/all-graph/model/native correctness, policy/human approval and adjacent
pass composition remain unestablished. Numerical work is deferred.

The same three workers now have separate large NEXT-HOIST tasks: Q actual Kernel
projection/gate, W partial-loop soundness, F independent fixtures/qualification.
They use the newer read-only Graham/Agustin producer e0484f488 and initially
exclude completion-token loops, zero-trip hoists, conflicting writes, nested
loops and nonpure setters. All implementation still ends on feat/pass-verification.
