# Shared-pass verification progress at the requested pause

Recorded at **2026-10-19 04:11:54 UTC** (2026-10-18 21:11:54 Pacific).
All three workers stopped; the coordinator stopped its remaining regression job. No jobs
remain active. Implementation stays paused until the user explicitly resumes it.

This publication contains progress documents. The newest implementation batches remain in
separate local clones; publishing these documents does not publish those code commits.
Private commit IDs below require the retained local clones and may not resolve on GitHub.

## Published implementation

The last published implementation commit at the pause was
`2ceacd3096258a3e09deb760d80f351a6c61447d` on `feat/pass-verification`.
The documentation commit containing this report is a later commit on that branch.

The published implementation includes bounded named-op generalization, integer arithmetic
and cast rules, shape and splat normalization, SSA-aware aliases and dead-code checks,
integer comparison ordering, scalar constant hoisting, bitcast cancellation, repeated-input
parallel fusion, and whole-return identity-loop refinement. These are accepted-boundary
proofs with explicit conditions, not proofs covering every transformation of each pass.
The earlier publication retained separate 1,284-case and 400-case qualification runs, both
with zero failures, errors or skips. They overlap and are not a unique coverage denominator.

## Committed local batches

| Batch | Implementation and example | Completed validation | Publication status |
| --- | --- | --- | --- |
| Cohort 13 | Current-binding AND/XOR factors, canonical splat casts, captured-producer fusion; e.g. `(x xor c) xor c -> x` requires a proved initialized local `c` | Audit: 6,573 theorems, zero violations. 484 binding and 198 MLIR/native tests passed, zero errors or skips; all 9,949 frozen files unchanged | Local commit `2b311f28ffeaef92cf63d840d79a8ba05d9498b2`; remaining 448-case regression suite interrupted at user request, so the batch is not fully qualified or published |
| Cohort 14 | Initialized add/sub cancellation, full static slices of canonical splats, captures in both fused bodies; e.g. `(x + c) - c -> x` and `(x + p) * q` fusion | Audit: 6,788 theorems, zero violations. Composition preflight: 20 binding and 10 native tests passed. Four actual consumer-capture chains accepted across both index widths and operand orders. All 260 prior lock entries unchanged | Local final commit `3f340725e3efbacfc148aa7a4126660e6f41cd81`; full integrated qualification not started |
| Integer constant combination | Two-node add/sub chains with two actual initialized constants; e.g. `(x + 5) + 7 -> x + 12`, preserving poison and escaping values | Build and 6,248-theorem audit passed with zero violations; 31 raw guards, including 4,096 exhaustive i8 comparisons, and 49 actual MLIR probes passed | Worker commit `edecf1dd703a54c356bf03873a57898213cb98ca`; 72 binding and 48 native cases collected but not executed; central integration pending |

Cohort 14's composition preflight used commit `a7590d2557f40dd515b92ed7542541e0cbc3f208`,
before the final Python fixture and documentation updates. Its successful preflight is not
full qualification of the final commit. The theorem audit and unchanged lock statements do
not replace receipt replay and native checks of the final integrated sources.

## Worker delivery checks

| Delivery | Completed private checks | Remaining integration requirement |
| --- | --- | --- |
| Full static slices | 62 binding checks: 32 new cases and 30 cast regressions; 24 native checks; 366 frozen files unchanged | Replay within the final central batch. Mathematical zero-extent guards are separate from actual MLIR validity controls |
| Consumer-capture fusion | 48 binding checks: 18 new cases and 30 legacy regressions; 13 native/gate checks; 230 frozen files unchanged | Final central replay. The private run deselected two obsolete gap expectations; central copies were strengthened into positive strict-chain tests after actual acceptance |

Consumer-fusion checks include same and distinct scalar captures, both operand positions,
actual changing two-loop-to-one boundaries, independent modular references, deliberately
wrong outputs, a nine-boundary strict gate chain and ordinary LLVM-byte parity. An additional
tensor-input case still refuses before acknowledgment and native suffix execution.

Counts across private and central runs overlap; do not add them as unique rule or model coverage.
Theorem counts describe the audit environment, not a count of implemented rewrite rules.

## Unfinished work and retained issues

- Cast/slice forwarding was planned but not implemented. Current exact facts recognize original
  splat producers; cast-to-slice and slice-to-cast chains remain recorded coverage gaps.
- Transitive add/sub chains have 16 actual MLIR probes and a proposed provenance interface;
  their general proof and checker implementation have not begun.
- Operand-taking CSE has a plan and actual pass probes. An uncommitted
  `MerlinVerify/Rules/CseOperands.lean` draft exists only in the splat worker clone, failed
  its first build and is unqualified. It has not entered either central batch.
- The first full-slice fixture run had 60 passes and two test-construction failures. That run
  remains unqualified. The corrected fixture replay passed; no semantic or compiler workaround
  was introduced.
- A separate Python 3.14 test-collection process segfaulted during garbage collection. The
  original crash is retained; later read-only collection completed. Its cause is unknown and
  the crashed process is not counted as passing qualification.
- Existing index32 signed-wrap disagreement and captured-destination downstream-lowering
  exclusions remain documented in the assumption ledger. They were not fixed by these batches.

## Resume checklist

1. Resume only after the user's instruction. Inspect the private heads and working-tree state
   in [checkpoint.json](checkpoint.json), including the uncommitted CSE draft; preserve all
   historical evidence and failed attempts.
2. Complete a fresh 448-case regression run for frozen cohort 13. Verify its source, checker
   and library inventory before and after the run. Use a new evidence directory; retain the
   interrupted run as unqualified. Do not edit or rebuild that checkout during receipt tests.
3. Review the full history to be published, reconcile it with the documentation commit, and
   preserve the qualified implementation identity. Publish cohort 13 only after its remaining
   checks pass. Do not overwrite frozen evidence to accommodate a new history.
4. Review and qualify cohort 14's final integrated source and tests: new families, legacy fusion
   routes, strict source-defined chains, valid mutations, native comparisons and retained
   fail-closed controls. Preserve old statement pins and frozen semantics.
5. Integrate the integer constant-combination helpers, kernel, checker and test drafts in a new
   isolated cohort. Execute its 120 collected checker/native cases before treating it as qualified.
6. Review the unfinished forwarding, transitive arithmetic and CSE work before continuing it.
   Never count a draft, build-only result or sampled native agreement as an accepted boundary proof.

The PI-selected model bundle and build configuration are still needed for a paper coverage
denominator. This work establishes targeted translation validation relative to the Lean model.
Universal fidelity of that model to MLIR, complete shared-pass coverage, default whole-model
gating, the OOT compiler and downstream LLVM execution remain outside the completed claim.

## Evidence and retention

[checkpoint.json](checkpoint.json) is a portable copy of the original local pause record:
repository-root prefixes were removed; the original remains unchanged.
[validation.json](validation.json) records checked XML counts, audit summaries, worker deliveries
and SHA-256 references to local evidence. [manifest.yaml](manifest.yaml) binds this report's files
and the original checkpoint hash. Paths in those files are relative to the retained repository.

Raw logs, receipts, frozen inventories, binaries and private implementation objects stay local
under the listed paths. Their hashes provide attribution; they do not make those files available
in a fresh clone. Preserve the local clones and evidence until the pending batches are qualified
and their reviewed implementations are published.
