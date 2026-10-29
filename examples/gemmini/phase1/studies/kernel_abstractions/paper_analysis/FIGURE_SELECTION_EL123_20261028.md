# Updated figure recommendation: the full four-hour EL123 comparison

**Correction:** The capability interpretation below is withdrawn. Six chain
verdicts depend on ABI-selected output-name comparison, and the seventh
baseline case was unrun. Read [COMPARATOR_ERRATUM_20261028.md](COMPARATOR_ERRATUM_20261028.md)
before using any figures or conclusions. The current PDF is a descriptive
two-panel draft with this issue visible; the previous layout is archived.

This supersedes the narrower EL2-versus-EL3 recommendation in
`FIGURE_SELECTION_20261028.md`. The user selected a fixed four-hour comparison
with EL1 retained after reviewing both its early-convergence advantage and
EL3's broader observed coverage. No run has been removed from the dataset.

Use the single double-column figure at
`/work/merlin/tasks/el4-grader-throughput-20261024/.tmp/paper-results-20261028/paper-figure-el123/el123_fixed_budget.pdf`.
The directory also contains editable SVG, PNG, the caption, source-bound
derived JSON, a verification record and an artifact manifest. The preceding
focused figure is retained as an exploratory alternative, not the recommended
main comparison.

## Supported message

**EL3 demonstrates broader compiler coverage within four hours, with fewer
recorded total tokens than EL1 and EL2.** Its best observed source passes all
128 cases demonstrated by both baselines, plus seven more. This is a positive
construction/coverage result for the xDSL-plus-authoring-support bundle. It
does not isolate the language alone or establish a hardware speedup.

| Full four-hour observation | EL1: direct C++ | EL2: C++ dialect scaffold | EL3: xDSL + authoring support |
|---|---:|---:|---:|
| Best single-source L2 passes, out of 149 | 128 | 128 | 135 |
| Recorded total tokens, including cache | 111,123,416 | 120,624,069 | 105,395,953 |
| Fresh input plus output | 3,006,168 | 2,440,005 | 2,849,137 |
| Failed author self-check case outcomes | 37 | 96 | 41 |
| Observed self-check case attempts | 361 | 300 | 653 |

EL3 uses 5.154% fewer total tokens than EL1 and 12.624% fewer than EL2. These
are fixed elapsed-budget totals at different achieved coverage. They do not
mean EL3 is cheaper at every matched coverage target. Fresh input plus output
is lower than EL1 but higher than EL2; total tokens are not dollar charges.
Failure counts are observed request/source/case outcomes; test mixes differ
and many self-checks lack explicit L2 outcomes. Do not interpret the attempted
denominator as a matched success rate or count every nonfailure as a pass.

## Which additional cases?

The six composed-chain cases comprise residual convolution, residual linear,
and sibling-feed chains, with both edge and scale/model variants. Both
baselines have explicit L2 failures on these six, whereas EL3 passes them.
The seventh is
`SY_independent_column_readout_matmul_int8_distinct_scales_v2`; EL1/EL2 have no
explicit L2 verdict on it, so call it additional demonstrated coverage rather
than a corrected failure.

One EL1 residual-linear failure reports an emitted hardware encoding that
declares a non-identity accumulator scale without the required non-identity
`CONFIG_ST` scale. Its command-buffer numeric check passes. This illustrates
why the figure uses native L2 execution evidence rather than numeric status
alone; it does not prove that one abstraction caused the difference.

Raw example evidence:

- EL1: `merlin/out/runs/gemmini/phase1/xs-el123-20261028-r2-el1/_qa_work/fruns_1793188373195682124/runs/gemmini-capsule-bench/CH_residual_linear_edge_m17k17n17/capsule_result.json`
- EL3: `merlin/out/runs/gemmini/phase1/xs-el123-20261028-r2-el3/_qa_work/fruns_1793190082623662964/runs/gemmini-capsule-bench/CH_residual_linear_edge_m17k17n17/capsule_result.json`

## Preserve the faster baseline and the performance limits

EL1 reaches the exact same shared 128-case set at 90.984 minutes and 50.26M
tokens, versus EL3 at 121.406 minutes and 71.46M tokens, and EL2 at 148.405
minutes and 98.54M tokens. It also has fewer failed self-check case outcomes
at that milestone: 30 versus EL3's 41. This early-convergence tradeoff is
visible in the curve and explicitly stated below the panels.

Generated-code performance is mixed. Of eight matching available EL1/EL3
frozen-source timing-model cases, EL3 is roughly 12–14% better on five GEMM
cases, but worse on KV heads and substantially worse on both convolutions
(approximately 11.1x and 12.4x EL1 cycles). These are provisional model
observations on a limited matched subset, not measured all-suite performance.
All three lack complete L3 qualification and evaluated full connected-program
correctness. This figure is deliberately about compiler construction and
execution coverage, with no runtime-superiority claim.

The full four-hour exact edited sources have incomplete grades (observed
passes 73/119/68). The figure instead asks what source-coherent coverage each
author demonstrated by the deadline. Each point can choose the best observed
candidate, but its passing cases must all belong to one exact source. The
source hashes and case evidence are saved in the derived JSON. Evaluation and
publication delay are included in the timeline.

## Reproduce

From the task folder:

```bash
export TMPDIR="$PWD/.tmp" PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/.tmp/paper-results-20261028/mpl"
taskset -c 0-3 .tmp/paper-results-20261028/venv/bin/python \
  merlin/examples/gemmini/phase1/studies/kernel_abstractions/paper_analysis/plot_el123_comparison.py \
  --csv .tmp/paper-results-20261028/data/all_results.csv \
  --out .tmp/paper-results-20261028/paper-figure-el123
```

The immutable CSV and earlier figures remain unchanged. The CSV cutoff is
`2026-10-28T19:06:46.946018+00:00`; its SHA256 is
`a13d23e55bdd7df49fc424bf35e4ab3dd510fff34d8c81f2d3197d6800b60454`.
No author, compiler, simulator or FireSim job was launched for this revision.
The existing source-coherence and extraction tests pass (15 tests). The PDF
is checked for one-page double-column dimensions and embedded metric labels,
and the rendered PNG is visually inspected. This remains a descriptive,
one-author-per-condition result rather than a replicated causal estimate.

Work remains on `cc/el4-grader-throughput-20261024`, base HEAD `388ee5ff2`.
No merge state is changed and nothing is pushed.
