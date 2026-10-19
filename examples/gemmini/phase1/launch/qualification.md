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
Root's actual termination and owned-cleanup evidence remains independently pinned.

Bind the corrected source and operator runtime hashes in a separate reviewed
config, preserving the selected corpus, model inputs, numerical policy and normal
simulator. Select `full_model_verification.output_scope: all_declared` and its
pinned `all_output_plan`; the helper refuses their absence. Prepare the config's
private-model declaration in the new prep directory before invoking it.

```sh
python examples/gemmini/phase1/launch/qualify_phase1.py \
  --config CURRENT_CONFIG --task-root TASK_ROOT --prep NEW_OPERATOR_PREP \
  --run-id NEW_QUALIFICATION_ID --submission FROZEN_FRESH_COMPILER \
  --author-proof HANDOFF_JSON --author-proof-sha256 HANDOFF_SHA256 \
  --deadline-unix ORIGINAL_CAMPAIGN_ABSOLUTE_DEADLINE --run
```

Without `--run`, only the derived unpaid native command is printed. The actual
run performs ordinary source/config/bundle/positive-readiness admission, then
starts native `--qualify-submission` grading and every selected full-model job
concurrently. The dedicated supervisor uses process sessions, start ticks, boot
identity and Linux subreaper cleanup. Its deadline alarm also covers preparation
and in-process offline all-output audits; cleanup may continue after the cutoff
to reap children, but no incomplete execution receives credit.

`status.json` records the separate native and model outcomes, all-output receipts,
cleanup, and the conjunction `phase1_complete`. The existing native
`submission_qualification.json` remains authoritative for formal grading. Both
records explicitly state `qualification_only: true` and
`authoring_converged: false`. A current-source qualification never rewrites the
original source's failed receipts. Native grading alone, a judged-only output
comparison, or a successful build alone cannot complete this route.
