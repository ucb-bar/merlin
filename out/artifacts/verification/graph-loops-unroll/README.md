# Ordered-unrolling batch — complete, sealed, awaiting central integration

Work location: `/home/jeremy/Projects/merlin`, branch `feat/pass-verification`.
Actual producer: detached read-only `merlin-graph-proof-intake@4499eebb959c0f92cefd92cc6d80646ba66474ed`.

The11-file additive batch is complete. Start with
[the qualified claim, domains, examples, failures, paper row and trust gaps](unroll-mode-v1/README.md),
[exact source delivery/dependency closure](unroll-mode-v1/DELIVERY.json),
[qualification seal](unroll-mode-v1/SEAL.json), and
[retained evidence inventory](unroll-mode-v1/EVIDENCE_SEAL.json).
The new source-only patch SHA256 is
`3b58da79ce9fbc41b96a4801f3fa2fee98c56817a220fe2cff33e0edb8245081` (104086bytes).

- 151 tests pass in qualification and151 in ONE fresh-process replay, zero skips.
  The population contains the old88 checks plus63 new unrolling checks.
- All27 public new theorem roots pass a transitive standard3-only axiom audit.
- Each run retains82 proof queries,82 checker records,17 finite runtime witnesses,
  one composed query and one composed checker record. The sealed inventory indexes2280 files.
- Eight original pinned producer tests pass separately. Ruff/format, no-regex,
  no-target-name and artifact-layout checks pass. The unrelated inherited structure
  failure remains `zephyr_model.py`3557>3537.
- All24 prior source bytes and seals remain unchanged. New+old source-only patches
  apply to an empty private destination and all35 owned files compare byte-for-byte.
  The four exact unchanged common intake definitions remain separately required.
- No stage, commit, push, publication, production/shared edit or numerical work occurred.

The theorem preserves complete ordered original-body occurrences, carry and modeled
state for inline (including empty) and bulk/optional-tail forms. Signed bounds are
derived. Complete actual SSA/capture/yield/context projection is checked. The actual
no-tail unroll→segment composition explicitly installs the middle body as a logged
factor grouping in the SAME existing SCF semantics. Python/raw-IR operational
bridges, arbitrary full-pipeline composition, Lean-MLIR/native/hardware fidelity and
all numerical gates remain outside the proved claim.

Documentation clarification: the qualified README's phrase
`GraphLoopsSegmentState.CarriedState` names the source file informally. The exact
Lean declaration is `Merlin.GraphLoopsSegment.CarriedState`, defined in
`GraphLoopsSegmentState.lean`. The compiled source and query use the correct
namespace. The sealed capsule is preserved byte-for-byte.

Central review/integration remains the next owner action. See
[reproduction instructions](REPRODUCE.md). This worker's assignment is complete;
no extra frozen-profile replay or unsupported next task has been launched.
