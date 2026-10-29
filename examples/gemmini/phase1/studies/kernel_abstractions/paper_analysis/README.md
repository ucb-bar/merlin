# Campaign results and paper analysis

The consolidated event table is in the original task:
`/work/merlin/tasks/el4-grader-throughput-20261024/.tmp/paper-results-20261028/data/all_results.csv`.
Start with [RESULTS_20261028.md](RESULTS_20261028.md) for findings and caveats,
[data-index.json](data-index.json) for artifact paths/hashes, and
[focused-fresh6-protocol.json](focused-fresh6-protocol.json) for the proposed
three-family continuation study. The protocol is a proposal, not a launch.

The table includes the completed EL1/EL2/EL3 studies, fresh10CCA/fresh10seed,
the three live tooling ablations, six completed 30-minute family studies, and
an expert comparison subject. The expert is not an independent author run.
Preflight jobs, failed preparation generations and unlaunched configurations
are excluded. The first EL launch was a disrupted pilot retired for documented ABI/schema visibility and restricted-feedback bugs; its preserved receipts/artifacts are indexed in `data/scope-exclusions.json`. It is not pooled with the clean r2 reruns. Each data release freezes an analysis cutoff; later run activity
does not silently update its CSV or figures.

## Read the CSV as an event table

Select `record_type` before aggregation. Nullable columns belong to different
observation types. Never sum derived `small_metric` token totals with
`provider_usage` rows, physical-cell medians with physical samples, or a grade
snapshot's count with its individual case rows.

| Record type | Meaning / aggregation key |
|---|---|
| `run` | Native author clock/configuration; one independent author per run, except the separately labeled expert subject |
| `provider_usage` | One owned, deduplicated completed provider response per `(run_id, thread_id, response_id)` |
| `grade_case`, `grade_snapshot` | Background case result / publication; preserve grade ID, request/source identity, selection and per-row attribution |
| `request`, `selfcheck_case`, `selfcheck_snapshot` | Author-issued request / its case rows / aggregate, separate from background checks |
| `source_snapshot` | Actual retained compiler tree and capture time, not an inferred implementation milestone |
| `checkpoint_published_view` | Historical per-case coverage view at a checkpoint; can mix compiler sources |
| `author_tool` | Sanitized EL command/edit outcomes; nonzero command exits are operational proxies |
| `full_model_observation` | One gate/program observation; supersession, pending/unavailable and shared infrastructure errors retained |
| `small_case_result` | Source-bound family case, during authoring or after freezing; inspect `subject`, `evaluation_phase`, `split` |
| `small_physical_cell`, `small_physical_sample` | Raw FireSim replay cell / repeated target-cycle sample; unqualified timing remains explicitly labeled |
| `small_broker_request`, `small_tool_observation` | Issued family request or CCA operation; unsuccessful/unavailable calls retained |
| `small_command_outcome`, `small_file_edit` | Sanitized family effort events, not proofs of implementation intent |
| `small_metric`, `small_case_metric` | Existing derived scalar observations; `metric`, `value`, `unit` preserve the original measurement names |

`run_id` is the native identity; `experiment` is a human-readable stable label.
`cohort`, `framework_revision`, initial/final source hashes and configuration
registry separate different seeds, tooling and budgets. See the adjacent
[configuration registry](../configurations/20261028/README.md) for exactly what
the eight main campaign environments received. The six small configurations
and tool overlays are pinned in `audits/fresh10/small-study-run-metadata.jsonl`
and `small-study-initial-source-comparability.json` beneath the data task.

`observed_at` preserves native case attribution; `publication_at` is the
containing grade's publication. Use the larger of `elapsed_s` and
`publication_elapsed_s` for an availability timeline. `within_first_4h`
requires both to be in the window. A result completed during the drain can
characterize a frozen four-hour source, but cannot improve the convergence
curve before four hours. Unknown timestamps remain blank; postfreeze family
results cannot be placed on an authoring timeline. Full-model timestamps use
file mtime where the native receipt lacks a clock, with that basis recorded.

Provider `tokens_input` already includes cache reads/writes. `tokens_total`
is input plus output; reasoning is a subset of output. Fresh input is input
minus cached input and cache writes. Lead and helper ownership excludes forked
parent history. All token totals are observed lower bounds, not billable dollar
cost estimates or accounting for unfinished responses. Historical resumed lead
stream counters are cumulative and must not be added together.

`numeric_status`, `l2_status`, `l3_status` and structural `ordering_status`
are separate observations. In these records, a numerical pass can coexist
with an L2 execution failure. Structural trace verification is not automatically
a dynamic race check. An expected refusal is not a numerical execution pass;
an unavailable or gated test is not a failure. Pending mandatory L3 can overlap
a passing L2, so snapshot counters are not disjoint stacked outcome classes.

Ever-observed case coverage spans revisions. `coherent_snapshot` checks
per-row sources and publication scope, but a coherent snapshot can still be
partial: check `count_scope`, case IDs and the fixed denominator. Neither a
cross-source coverage union nor a copied `latest_completed_grade.json` is a
final compiler grade. The common background roster has149 cases; the public
author roster has152, including three separately evaluated programs. One case
is an expected refusal and has no positive L2 execution obligation.

Timing-model estimates, fast estimates and physical cycles occupy separate
columns. Match case/capsule, numerical contract, model, domain, scope and
source/ELF before comparing. `model_engine` and `measured_engine` distinguish
the timing observation types. `hardware_status`, `timing_qualified` and
`details_json` preserve physical admission limits. Family case-name grouping
is a documented heuristic, not a proof of an algorithm's implementation.

## Reproduce

Collection uses the existing Python3.12 campaign environment and its native
telemetry/result readers. Plotting uses a separate task-local environment;
installed versions are recorded in `plot-requirements.txt` under the result
directory. No author, compiler, simulator, model fit or FireSim job is started.

```bash
export TASK_DIR=/work/merlin/tasks/el4-grader-throughput-20261024
cd "$TASK_DIR/merlin"
export TMPDIR="$TASK_DIR/.tmp" USER=ubuntu LOGNAME=ubuntu
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD/packages/merlin-experiments/src"
taskset -c 0-3 \
  /work/merlin/tasks/gemmini-shared-phase1-20261021/.tmp/phase1-runtime-v1/venv/bin/python3.12 \
  examples/gemmini/phase1/studies/kernel_abstractions/paper_analysis/collect_results.py \
  --task "$TASK_DIR" --out "$TASK_DIR/.tmp/paper-results-20261028/data-recollected"

export MPLCONFIGDIR="$TASK_DIR/.tmp/paper-results-20261028/mpl"
taskset -c 0-3 "$TASK_DIR/.tmp/paper-results-20261028/venv/bin/python" \
  examples/gemmini/phase1/studies/kernel_abstractions/paper_analysis/analyze_results.py \
  --data "$TASK_DIR/.tmp/paper-results-20261028/data" \
  --out "$TASK_DIR/.tmp/paper-results-20261028/figures"
python3 -m unittest discover \
  -s examples/gemmini/phase1/studies/kernel_abstractions/paper_analysis -p 'test_*.py'
```

Use a new output directory when collecting a later release. Audited token
JSONLs for completed/historical windows are immutable extraction inputs, with
their own capture times; live native streams are lower bounds at collection.
`sources.json` records hashes/lengths, including append-only prefix pins;
`extraction_issues.json` and `receipt.json` record coverage and failures.

The paper folder and running experiments are read-only inputs to this work.
Draft figures remain in the task, with PDF/SVG/PNG exports and captions in the
results report. Main figures compare matched four-hour windows; the live
tooling wave and 30-minute family studies have separate panels and censoring.
