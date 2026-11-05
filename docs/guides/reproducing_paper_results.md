---
title: Recompute reported compiler study results
kind: guide
status: current
owner: experiments
last_verified: 2026-11-05
related: [artifact_evaluation, reproducibility, ../reference/panel_tile_loads]
code_refs:
  - build_tools/scripts/recompute_paper_results.py
  - build_tools/scripts/restore_reference_archive.py
  - experiments/reference-data/paper-studies/registry.json
  - experiments/reference-data/paper-studies/supplemental-archive.json
---

# Recompute reported compiler study results

A result has three reproduction levels: recomputing its statistics from saved
records, rebuilding its frozen compiler and executable, and repeating execution
on the selected hardware. The registry and command below implement the first.
They do not replace the other two. A new authoring campaign is a separate
experiment, even if it uses the same prompt and time budget.

The [result registry](../../experiments/reference-data/paper-studies/registry.json)
uses stable recipe IDs instead of manuscript figure numbers. It lists original
run IDs, source producer commands, checksummed data files, calculation scope and
missing execution dependencies. The compiler repository's
[run index](https://github.com/ucb-bar/gemmini-mlir/tree/paper/index) maps those
run IDs to exact source commits and preserved source hashes. External IREE,
TVM, Voyager and iterative-agent measurements remain supplied comparison data.
These recipes do not independently validate those implementations.

## Restore the data inputs

Obtain the artifact archives selected by the registry and verify their full
SHA-256 values before extraction. Restore their original top-level member names
beneath a new `data/` directory:

```text
data/
  perf-studies_gemmini_v4_20261028T232951Z_744bcbb/
  paper-v1/
  fsm-rerun/
  fsm-counterfactual-execution/
  fsm-decode-strides/
  supplemental-studies/
```

The frozen-kernel archive contains hardlinks, including image `results.csv`
files linked to `collected/results.csv`. Extract the complete archive or retain
both members when selecting files. A partial extraction that omits the linked
member is incomplete. The archive may also reference external dependencies.
Restoring saved statistics does not supply every build or execution dependency.

The supplemental small-study archive has its own exact member inventory:

```sh
python3 build_tools/scripts/restore_reference_archive.py \
  --manifest experiments/reference-data/paper-studies/supplemental-archive.json \
  --archive "$ARTIFACT_ROOT/supplemental-study-records.tar.gz" \
  --destination "$ARTIFACT_DATA"
```

Original producer scripts and their recorded configurations are preserved as
provenance. Some contain historical machine paths and write beside their inputs.
Do not execute them in place on the immutable archive. The portable command uses
explicit artifact paths and creates a new output receipt:

```sh
python3 build_tools/scripts/recompute_paper_results.py \
  --registry experiments/reference-data/paper-studies/registry.json \
  --artifact-root "$ARTIFACT_DATA" \
  --output "$MERLIN_OUT_ROOT/artifacts/paper-recomputation/all-results.json"
```

The receipt records script and registry hashes, every consumed input hash,
computed metrics and limitations. Missing or changed selected inputs fail rather
than producing a partial success. Use `--recipe` to select one result family or
repeat it to select several. No model credentials, accelerator, compiler build,
plotting dependency or native simulator is needed for saved-data arithmetic.

## Quantities and comparisons

Utilization is `100 * useful_MACs / (peak_MACs_per_cycle * completed_call_cycles)`.
The frozen protocol supplies the 256-MAC/cycle peak. This is useful peak
attainment over the timed call, not a hardware processing-element busy counter.
The timed scope and cache profile come from the measurement records. The recipe
selects `warm_repeat` and keeps unrankable outcomes unmeasured. It checks stored
cycle medians against their samples before calculating utilization or speed.

Speed is reference cycles divided by compiler cycles for the same case.
Aggregate speed is the geometric mean of per-case ratios. The main sixteen
workloads and the seventy-three shape-sweep cases are reported separately.
The common FSM reference used in the original plots and the later same-image
FSM controls remain separate output fields. They must not be silently combined.
Current automatic-library measurements and manually adapted tiles or grouped
attention calls are separate cohorts in `fsm-controls`. A tested manual tile is
not proof of the globally best FSM schedule.

The frozen-performance receipt also reports both compiler snapshots against the
supplied iterative-agent cycle values over the complete sixteen-workload panel.
It refuses a missing panel entry or conflicting supplied values. This recomputes
the reported comparison arithmetic without qualifying the external implementation
or claiming matched development budgets or execution protocols.

E1-E3 calculations use the saved, extracted event tables. They reconstruct the
sixty-minute pass count and uncached input-token total, sum author-window failed
grade results excluding expected refusals, and compute median/interquartile
self-check latency. Token subtotal rows are selected once for all author roles,
not added to their helper/lead or hourly components. Historical pass curves can
combine successes across compiler versions. Latency depends on the requested
checks. Re-extracting these tables requires the original streams and pinned
native telemetry readers.

Scheduling ratios divide control feedback-availability time by checked-scheduler
time at the first ten-percent modeled improvement. All five plotted studies
remain separate. The curve's internal completion timestamp can precede the time
the result became available to the author. The thirty-minute anchor is checked
against its saved publication-aware milestone. Other repeat-study milestones
are preserved extracted observations. Re-extracting them requires their native
owner records. Executable identity is checked separately from a timing ratio.

The convolution recipe recomputes within-image ratios of recorded median cycles
for every plotted shape and takes their geometric mean over all four layouts.
A hanging baseline has no finite speedup. The controlled CCA recipe recomputes
medians from each cell's saved samples and retains all four same-image boot
contrasts. Its measured subset has eleven of fifteen cases per pair. Four boots
are hardware repeats, not additional independent author replicates.

## Limits retained from the original evidence

The single-source CCA comparison is not a matched ablation. CCA-on also receives
additional operator notes and a newer export. Family geometric means describe
those selected runs without isolating the cause. Static instruction-site counts
and credited CCA replies are mechanism evidence, not measured speedups or proof
that a tool caused an optimization.

The small convolution and controlled-CCA records retain strict physical
qualification as `UNKNOWN`, including unresolved consumed-image/domain evidence.
Recomputing their recorded raw contrasts does not promote that status. Original
unmeasured cases, hangs, refusals and missing dependencies remain visible.
Sealed-case identities used during the controlled study are separate inputs.
Their private identity file is unnecessary for recomputing already recorded
anonymous ratios but is required to repeat the original selected test panel.

For independent execution, retain the exact compiler and selected shared
library, original inputs/references, toolchain, hardware/runtime versions,
compiled objects, linked ELF and UART/check records. A current source checkout
cannot substitute for an executed framework export. For new authoring, additionally
select prompts, model/version, reasoning effort, helpers, time/token accounting
and sandbox-visible inputs. The [artifact setup guide](artifact_evaluation.md)
separates these requirements. Passing this saved-data command establishes
statistical consistency of the selected records only.
