# Qualifying the exact fresh compiler after a framework repair

`qualify_phase1.py` is an operator-only route with no author/provider launch. It
does not resume or replace the original fresh campaign. Preserve that campaign's
transcript, conformance, native OOT history, usage, partial grades and source pins.

After the original author actually terminates, use `terminal_author`,
`native_snapshot` and `copy_snapshot` from `full_model_verification.py` to export
its exact native OOT compiler. Stop/reap the old campaign before the new run.
Create a pinned `merlin.operator.fresh_author_handoff.v1` JSON containing
`original_run_dir`, `native_oot_commit`, `original_source_revision`,
`original_hard_deadline_unix`, `frozen_submission` (the full `package_identity`
record), `qualification_only: true` and `authoring_converged: false`. Include the
original terminal-author, conformance and transcript evidence as extra fields.

When the original controller blocks after actual author termination on stale
grading bookkeeping, use the explicit alternative
`freeze_mode: terminal_author_operator_checkpoint`. Create a **separate** operator
OOT repository and supply `operator_oot_path`, `operator_oot_commit`,
`terminal_authored_source_path`, `actual_terminal_source_tree_sha256`,
`source_sha256_before` and `source_sha256_after`. The three source hashes, reopened
terminal source and operator checkpoint digest must all match the frozen compiler.
The source path must be the original run environment's workspace submission.
Retain the stale original native OOT as `original_native_oot`; do not call the
operator checkpoint a native author checkpoint or invent author convergence.
This mode requires `original_empty_origin: true` and the original source revision.
The original author's termination and owned-cleanup evidence remain independently pinned.

Bind the corrected source and operator runtime hashes in a separate reviewed
config, preserving the selected corpus, private host-selection inputs, numerical
policy and simulator. Choose the validation scope explicitly; the default for
historical configs remains `full`.

## Capsule-only requalification

Set `validation_scope: capsules` to grade the same frozen compiler against the
complete selected public and hidden capsule rosters. The native command retains
the private spec and instruction-selection inputs needed by public host/model
capsules. It starts no author, full-model source preflight, private full-model
execution, model-check job or all-output audit. The ordinary capsule preparation
still checks the current source/config/runtime pins, corpus seal and private
bundle/spec metadata; deferred model payload/readiness checks receive no pass
credit. Copy the selected private spec byte-for-byte into the new prep directory.

The capsule-only supervisor accepts at most **3600 seconds**, including preparation,
grading and independent certificate reopening. Use a new run ID, prep directory
and explicit absolute deadline. Its inherited simulator and per-capsule budgets
remain selected inputs; the outer deadline owns the whole process tree.

```sh
python examples/gemmini/phase1/launch/qualify_phase1.py \
  --config CAPSULE_QUALIFICATION_CONFIG --task-root TASK_ROOT \
  --prep NEW_OPERATOR_PREP --run-id NEW_CAPSULE_QUALIFICATION_ID \
  --submission FROZEN_FRESH_COMPILER \
  --author-proof HANDOFF_JSON --author-proof-sha256 HANDOFF_SHA256 \
  --deadline-unix NEW_BOUNDED_QUALIFICATION_DEADLINE \
  --budget-extension QUALIFICATION_BUDGET_JSON \
  --budget-extension-sha256 QUALIFICATION_BUDGET_SHA256 --run
```

A deadline after the original campaign cutoff requires the existing explicit,
hash-bound `merlin.operator.qualification_budget_extension.v1` receipt and its
operator authorization. Bind the original handoff SHA, original deadline, exact
submission SHA, authorization time and new finite deadline. This is a separately
accounted qualification budget, not an extension or successful completion of the
original three-hour authoring experiment. The original handoff stays unchanged.

On success, `submission_qualification.json` and `status.json` report
`validation_scope: capsules` and `capsule_complete: true`. The supervisor reopens
the actual technical `capsule_certification.json`, its exact frozen compiler,
selected descriptor, complete rosters and required evidence; exit zero alone
cannot grant completion. `authoring_converged`, `phase1_complete`,
`external_full_models_complete` and `overall_release_qualified` remain false.
The technical capsule certificate does not mint an authored-convergence or final
Phase 2 publication. External models stay pending under their own later owner.

A source change since an old full-model readiness receipt does not make that
receipt current. This capsule route does not consume it as qualification. Refresh
`SOURCE_REVISION` and every changed operator/runtime source pin, archive the
resolved config and keep historical model evidence under its original identity.
Full-model readiness must be rebound independently before selecting full scope.

## Complete external qualification

Set `validation_scope: full` to retain the original conjunction of native formal
grading and the selected complete model chains. Select
`full_model_verification.output_scope: all_declared` and its pinned
`all_output_plan`; this scope refuses their absence. Prepare the config's private
model declaration in the new prep directory before invoking the helper.

```sh
python examples/gemmini/phase1/launch/qualify_phase1.py \
  --config CURRENT_FULL_CONFIG --task-root TASK_ROOT --prep NEW_OPERATOR_PREP \
  --run-id NEW_FULL_QUALIFICATION_ID --submission FROZEN_FRESH_COMPILER \
  --author-proof HANDOFF_JSON --author-proof-sha256 HANDOFF_SHA256 \
  --deadline-unix SELECTED_QUALIFICATION_ABSOLUTE_DEADLINE --run
```

Use the same explicit budget-extension flags when a separately authorized full
qualification goes beyond the original deadline. Without `--run`, the helper
only prints the derived unpaid native command. Full scope performs ordinary
source/config/bundle/positive-readiness admission, then starts native
`--qualify-submission` grading and every selected full-model job concurrently.
Native grading, a judged-only comparison or a successful build alone cannot
complete full qualification.

Both scopes use process sessions, start ticks, boot identity and Linux subreaper
cleanup. The deadline alarm covers preparation and in-process verification;
cleanup may continue after cutoff to reap children, but incomplete work never
receives credit. `status.json` records the selected scope, separate outcomes,
qualification clock and cleanup. Both scopes explicitly retain
`qualification_only: true` and `authoring_converged: false`. Current-source
qualification never rewrites the original source's failed receipts.
