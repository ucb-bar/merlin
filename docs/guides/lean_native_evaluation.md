# Lean compiler numerical loop

Use this path for the Phase 2 optimization campaign. The user requested that
historical qualification and provenance preflight no longer gate exploration.
Compiler emission and complete-output checks still run for each candidate;
final qualification is deferred until a compiler has been selected and frozen.

`merlin_experiments.phase2.lean_authoring.run_lean_authoring` drives the existing
continuous Codex author and asynchronous evaluation broker. The author edits
compiler source, freezes its exact bytes, submits selected capsules, reads
feedback and continues without an operator managing individual iterations.
Its configuration supplies the saved member mappings and an evaluator callback,
an immutable `baseline`, `deadline_utc`, `admission_cutoff_utc`,
`frozen_model_sha256` and `target_fraction`. Keep the original campaign clock
across retries: a retry does not restart the three-to-four-hour budget.

The configured experimental stopping threshold is 0.30 for every capsule in the
nine-member roster under one retained compiler. Admission closes before the hard
deadline so outstanding jobs can drain. Comparable current-candidate estimates
take priority over original-control estimates when choosing the next capsule;
the lowest attainment is first. Missing counts stay explicitly unknown.
Reaching this threshold is an experimental result, because the comparison uses
an ideal compute floor and a frozen approximate timing model. It does not admit
an attainable whole-invocation roofline or establish measured hardware latency.
Freeze the chosen compiler before subsequent FireSim validation.

`merlin_experiments.phase2.lean_native_evaluation.evaluate_candidate` accepts
`candidate_source`, `baseline_source`, a tuple of `selected_members`, a fresh
`work_root`, `timeout_s` and persistent `baseline_cache_root`. Each member declares
`id`, `capsule_dir`, `interface_mlir`, `runtime_bundle`, `target_descriptor` and
`rtl_facts`; optional fields include `target`, `board_config`, `floor_cycles`
and `target_fraction`. All members in a worker process share one target selection.

The callback invokes the package's ordinary `emit_analysis_bundle`, then the
existing complete-output Spike numerical runner with its explicit L2 ceiling
and V2 storage transport. It constructs no historical qualification, pass
factory, publication or source-effect owner and starts no tier audit. Compile
errors, mismatches and incomplete output counts remain failures.

Results contain `members[{id,baseline,candidate}]`. Each arm records its compiler,
input/source identity, CB/LLVM/native-result/ELF pins and
`complete_outputs_checked`. Successful numerical execution has
`status="numeric_pass"`. Every result remains `qualification="UNQUALIFIED"`
and `source_effects="PENDING"`.

For count estimates, pass an explicit prepared `count_runtime` and its frozen
`count_identity`. The callback invokes `lean_count_observation.observe_evaluation`
after complete numerical success and retains its result as `count_observation`.
Unknown or failed counts do not erase a completed numerical fact and cannot
establish performance improvement. The unchanged baseline is cached by its
package, all saved source/input/reference bytes and model/runtime identity;
cache hits reopen the exact numerical receipt and compiled ELF. Final source
and hardware qualification remain separate from this search loop.

A candidate with exactly the control's relative filenames and file bytes reuses
that control observation, labelled
`observation_origin="identical_source_control_reuse"`. Its original numerical
and count artifacts remain visible as the observation's origin. Every changed
package receives its own compilation, full-output numerical run and count run.

While counts run, the arm's `evaluation.json` retains its completed numerical
checkpoint. The paired `progress.json` contains only member names, phases,
numerical completion flags and count status. It remains `status="running"`
until every selected member returns; pending counts supply no cycle estimate
or performance attainment.
