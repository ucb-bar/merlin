# Buffer-reuse batch — complete and sealed, conditional allocator model

PRIMARY `/home/jeremy/Projects/merlin`, `feat/pass-verification`. Actual producer:
read-only `merlin-graph-proof-intake@4499eebb959c0f92cefd92cc6d80646ba66474ed`.

Start with [the theorem/domain/paper row and trust gaps](reuse-mode-v1/README.md),
[exact source delivery and dependencies](reuse-mode-v1/DELIVERY.json),
[qualification seal](reuse-mode-v1/SEAL.json),
[retained evidence inventory](reuse-mode-v1/EVIDENCE_SEAL.json), and
[reproduction instructions](REPRODUCE.md).

- **214 tests pass in qualification and214 in ONE fresh-process replay**, zero
  skips. This is the old151 plus63 new checks. All40 public new theorem roots
  pass transitive standard3-only audits. Two original storage tests pass separately.
- The theorem relates every live initialized logical-epoch cell to its physical
  slot and preserves all complete possible caller/scalar/memory/effect outcomes
  in both directions, with no source-success or primitive-totality premise.
  Full actual SSA/use/initializer/free/alias/context bindings and signed relative
  geometry are checked. A nonzero17/29 Lean model and seven initialized finite
  actual xDSL witness families are retained.
- **Allocation is a declared TOTAL PRIVATE UNOBSERVABLE premise.** An actual
  produced case peaks at8bytes before reuse and16after; finite capacity8 can
  distinguish them. Allocation failure/identity/native conformance is not proved.
  The raw SCF/initializer/primitive operational bridge remains explicit trust.
- A real next segmentation boundary is captured and REFUSED as a composition
  gap. Allocation-changing states are not fixed-cell renaming. No numerical,
  whole-pipeline, native or hardware gate was discharged.
- The shared Std.Time.Format.Basic incident was checked once: NOT_SELECTED here.
  All actual selected imports/sidecars and host-library bindings were exact.
  The foreign cause remains unknown; no toolchain rewrite or hash promotion occurred.
- Each run retains104 queries,103 checker records,24 runtime witness files,
  one concrete stale-data counterexample and one prior supported composed record.
  New memory qualification itself does not claim neighboring composition.
  The final inventory indexes3063 files.
- New source-only patch:13files,133316bytes, SHA256
  `12a109bd29169874f3a7c3c8944f60a53aafa6eec5826559a6267e3616678839`.
  Previous35 source bytes/seals are unchanged. The four patch chain applies to
  an empty private destination and all48 owned sources compare byte-for-byte;
  unchanged common4-source intake remains separately required.

Ruff/format, no-regex, no-target-name and artifact-layout checks pass. The structure
gate retains only inherited `zephyr_model.py`3557>3537 size debt. Old numerical
failures/counterexamples and unobserved statuses remain untouched. Nothing was
staged, committed, pushed or published; central review/integration remains pending.
The current large assignment is complete. Frozen source and evidence are read-only.
