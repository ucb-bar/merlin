# Reviewed scoped private-buffer result — 2026-10-28

Integrated into `feat/pass-verification` at **09fea17e6**. The source commits are
3cb865655 (proof/type repairs), 1fb8df215 (actual-IR gate and guide), and
2fcdae7b3 (qualification harness and tests). The coordinator reviewed the changes;
this is not independent human specification approval.

The direct developer API validates the complete **original → normalized → reused
→ segmented** boundary. It independently reconstructs the memory plan and actual
endpoints, checks twelve generated Lean modules against 111 project modules,
audits the roots/imports/axioms, and permits a live consumer to receive only the
checked final bytes. Saved reports cannot authorize delivery. See the
[usage and exact domain](../../../../../../docs/guides/lean_buffer_reuse_gate.md).

## Claim and assumptions

The all-entry theorem `GraphMemoryAfterEntries.every_definition` establishes
equality of sets of possible successful returned/public/caller-heap/user
observations, under its typed input/caller domain, common scalar/literal
interpretation, and **total, private, unobservable allocation** model. This is a
different relation from the scalar loop checker's return/error/terminal-world
equality. The supported initializer has static rank-two rectangular uniform
loops, disjoint checked sequential lifetimes, and complete initialization.

This does not prove finite-capacity behavior, failure/divergence equivalence,
arbitrary pointer escape or aliasing, asynchronous completion, hardware behavior,
native allocator fidelity, or whole-model coverage. The retained 8-to-16-byte
resource counterexample remains valid. Unchanged entries must have checked
call-closed bodies; a wrapper calling the changed kernel is refused by this basic
API. The separate one-call Context candidate is preserved but not integrated.
Python/xDSL projection and host transport remain trusted boundaries.

## Evidence and lineage

- F2 independently reexecuted five actual variants from producer revision
  `4499eebb959c0f92cefd92cc6d80646ba66474ed`, checking 116 exact compiler-verified
  MLIR buffers. All five live gate/delivery checks passed.
- 24 valid mutated quadruples were refused: 23 at host admission and one at the
  actual Lean `WholeMemory.GraphMemorySCFObligation.valid` obligation. These are
  different kinds of refusal, not 24 Lean counterproofs.
- The campaign covered 14 request controls, five forged/copied/saved-token
  controls, four allocator-mode controls, nine live artifact guards and a fresh
  separate-Python replay. Baseline audits covered 250 roots/2,338 theorems;
  three epochs covered 255 roots/2,343 theorems, with only the standard three
  axioms and zero violations. Overlapping audit counts must not be summed.
- That full campaign used the historical **186-source** fingerprint. A later
  correction explicitly binds five already imported scheduling files, making
  **191**. The corrected exact-source isolated baseline check and separate
  replay passed, each with 125 successful processes. This is additive evidence,
  not a relabeling of the five-case campaign as 191-source qualification.
- All 191 current primary sources and all five actual query/part/root bundles
  match the successor exactly: [primary comparison](BUFFER-PRIMARY-MATCH.json).
  No additional native run on the primary checkout is claimed.
- Constructive nonzero source executions are retained. W checked all five
  corpus witnesses; F independently rebound the baseline (326 roots, 2,422
  audited theorems). F's supplementary four-case repeat stopped at its first
  timeout, so it is explicitly partial, not four further independent successes.
- Primary integration passed 84 focused SDK-free tests and the structure gate.
  Seven preexisting long lines remain in reused emitters; other scoped lint
  rules passed. Earlier monolithic SIGSEGV, epoch timeout, shape-limit failure,
  collector failure and supplementary timeout records remain preserved.

## Review and reproduction packet

Start with [the publication index](buffer-publication-v2/MANIFEST.json),
[F's exact final status](buffer-publication-v2/F/FINAL_STATUS.json),
[the 191-source successor result](buffer-publication-v2/W/binding-successor-v1/RESULT.json)
and [the detailed IR/theorem review](buffer-publication-v2/W/FINAL-REVIEW/README.md).
The packet includes hashed text/source archives, original native reports,
proof witnesses, failed attempts and exact source manifests. Archive members were
checked for safe relative regular-file paths, UTF-8 source/text, and exact hashes.
No SDK or compiled native/Lean object is shipped. Original absolute evidence
paths are intentionally preserved; transfer does not grant live proof authority.
The selected external Lean/Python/xDSL environment is still required for replay.

`whole_pass_verified`, policy/human approval, primitive/native fidelity,
compiler-wide enforcement, and all-graph/whole-model correctness remain false.

## Current worker assignments

All three existing workers are released to bounded straight-line redundant
**persistent configuration-write elimination**, reusing the existing Lean
configuration-state infrastructure. Q owns actual IR projection and gate; W owns
Lean checker/module soundness; F owns independent actual-producer qualification
and adversarial evidence. Their [release and task packet](buffer-publication-v2/configuration-tasks/RELEASE.json)
supersedes historical HELD/buffer assignments. No numerics, barrier removal,
loop-hoisting, pipelines or hardware correctness is included in this next scope.
