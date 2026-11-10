# Campaign event analysis

These scripts collect recorded events and produce descriptive plots. They do not
start authors, compile kernels, fit timing models or run simulators. The original
configuration registry, dated reports, plot-selection notes and extraction index
are preserved in the historical study archive retained under ignored `out/` with its original source revision.
Their original bytes are evidence. Current plot scripts are source code.

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
historical configuration registry retained with those artifacts for exactly what
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

## Plot a supplied event-table release

Select an artifact directory containing `all_results.csv` and `runs.json`. Verify
its recorded hashes before use. Install the plotting dependencies recorded with
that data release, then run from the Merlin source root:

```sh
python3 examples/compiler-generation/gemmini/backend-generation/studies/kernel_abstractions/paper_analysis/analyze_results.py \
  --data "$CAMPAIGN_DATA" --out "$MERLIN_OUT_ROOT/artifacts/campaign-analysis"
```

Set `CAMPAIGN_DATA` to the selected data release and `MERLIN_OUT_ROOT` to your
generated-output root. `summary.json` and figures are written beneath `--out`.
A later live campaign is not an interchangeable source for a frozen data release.

## Recollect original run records

`collect_results.py` expects a selected task directory containing a `merlin/`
repository, its original `out/runs/` and workspace records, and the historical
configuration registry under its original source-relative path. Restore the
archive into a separate evidence tree first. The archive alone does not include
all author streams, run state or external referenced objects. A complete task
artifact and any required path mapping must be supplied independently. Do not
rewrite the archived records or use a newly launched campaign to fill missing
historical evidence.

```sh
python3 examples/compiler-generation/gemmini/backend-generation/studies/kernel_abstractions/paper_analysis/collect_results.py \
  --task "$RESTORED_TASK" --out "$RESTORED_TASK/merlin/out/artifacts/recollected-events" \
  --cutoff "$RECORDED_ANALYSIS_CUTOFF"
python3 -m unittest discover \
  -s examples/compiler-generation/gemmini/backend-generation/studies/kernel_abstractions/paper_analysis -p 'test_*.py'
```

Collection output must remain inside the selected task. Set the exact original
UTC cutoff to reproduce an earlier extraction. `sources.json` records file
hashes and lengths, including append-only prefix pins. `extraction_issues.json`
and `receipt.json` report missing evidence and coverage. Never interpret a
partial or failed extraction as reproduction of the complete table.
