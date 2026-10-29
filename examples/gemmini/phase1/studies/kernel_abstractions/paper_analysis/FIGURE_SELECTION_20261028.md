# One figure for the authoring-support result

**Later correction:** These are historical recorded-verdict comparisons under
ABI-dependent acceptance rules, not clean evidence of an abstraction effect.
Read [COMPARATOR_ERRATUM_20261028.md](COMPARATOR_ERRATUM_20261028.md). The later
EL123 figure retains all three runs and visibly discloses the checker issue.

Use the double-column `tooling_convergence_el2_el3.pdf` in
`/work/merlin/tasks/el4-grader-throughput-20261024/.tmp/paper-results-20261028/paper-figure/`.
It has one comparison and three aligned outcomes: time to identical tested
coverage, recorded total tokens at that goal, and failed self-check case
outcomes at that goal. Gray consistently denotes the C++ dialect scaffold
(EL2), blue the xDSL plus authoring-support environment (EL3). The absolute
values and percentage differences are visible without consulting a table.
The vector PDF/SVG and PNG contain the same figure. The adjacent `caption.md`,
derived JSON and `manifest.json` document the definition, evidence and limits.

The supported message is **EL3's authoring bundle reaches EL2's observed
compiler coverage sooner, with fewer cached-inclusive tokens and failed
self-check case outcomes**. This compares construction and Spike execution
validation. It does not show that more tooling always helps, that xDSL alone
causes the change, or that the generated compiler is faster on hardware.

## Preserved information

- Same public test IDs, starting state, references, model, framework and CPU
  allowance; the environment names identify the changed authoring bundle.
- The baseline goal is the 128 passing case IDs of EL2's best observed source
  in the fixed four-hour window, drawn from the 149-case background roster.
  This is a retrospective goal, not a preregistered target.
- One compiler source must pass all goal cases. Its published grading batches
  may be combined; different revisions cannot supply different parts of the
  goal. The plotted curve is the best observed source's coverage of the fixed
  case set, and ends at the first goal verification.
- EL2 reaches the goal at 148.404797 minutes, versus 121.406126 for EL3.
  Recorded total tokens are 98,537,893 versus 71,459,357. Failed author
  self-check case outcomes are 92 versus 41. The corresponding reductions
  are 18.1926%, 27.4803%, and 55.4348%.
- EL3 passes the same 128 cases plus seven others at that milestone, for 135.
- Fresh input plus output is 1,926,693 versus 1,981,213 tokens: EL3 is 2.83%
  higher on this subset. Lower total tokens therefore does not establish less
  newly generated work or a lower dollar cost. Observed usage is a lower bound
  because it excludes unfinished/unreported responses.
- The failed checks are request/source/case outcomes, not unique failed
  implementations or invalid schedule attempts. There are 186 observed
  self-check case attempts for EL2 versus 478 for EL3, with different mixes.
- One independent author run per condition; no confidence interval or
  population-level causal claim. Spike/L2 passes do not imply complete L3,
  dynamic race qualification, or measured hardware performance.

## Selection and results not shown

**EL1 remains a faster, cheaper direct-C++ comparison.** On this identical case
set it reaches the goal at 90.983706 minutes and 50,255,362 total tokens, with
30 failed self-check case outcomes. Removing it from this focused contrast
does not remove it from the data or the analysis. The caption discloses this
fact; a paper must not generalize this figure to every less-tooled baseline.

The five earlier plots remain under `figures/` in the result directory,
including `all_experiment_plots.pdf`. Their underlying runs remain in the
immutable consolidated CSV. Selection changes presentation, not the dataset.

- Earlier ever-passed case unions span compiler revisions. They are useful
  for exploration but can exaggerate how soon a single compiler reaches a
  coverage goal. This figure uses source-specific evidence instead.
- Comparing fresh10 seed and CCA campaigns also changes starting material and
  feedback/tooling. That is an overview rather than an isolated authoring
  ablation. The three live tooling arms were incomplete at the frozen cutoff;
  early partial coverage gaps are not final results.
- The short family studies suggest useful reliability/transfer effects:
  library treatment has 15/15 successful held-out physical replay cells versus
  2/15 control, and scheduling treatment 14/14 held-out cases versus 10/14.
  However, treatment token totals are higher, one control ends early, and the
  physical timings lack full timing qualification. These should not be folded
  into a universal convergence or performance improvement claim. The short
  CCA study does not show a uniform win either.
- Frozen-source modeled performance is mixed across shapes, including a large
  convolution regression. It cannot substantiate a universal runtime win and
  is not the message of this construction figure.

## Reproduce and audit

From the task folder, with the existing plot environment:

```bash
export TMPDIR="$PWD/.tmp"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/.tmp/paper-results-20261028/mpl"
taskset -c 0-3 .tmp/paper-results-20261028/venv/bin/python \
  merlin/examples/gemmini/phase1/studies/kernel_abstractions/paper_analysis/plot_focused_convergence.py \
  --csv .tmp/paper-results-20261028/data/all_results.csv \
  --out .tmp/paper-results-20261028/paper-figure
```

The consolidated CSV SHA256 is
`a13d23e55bdd7df49fc424bf35e4ab3dd510fff34d8c81f2d3197d6800b60454`.
Its native-data cutoff is `2026-10-28T19:06:46.946018+00:00`.
The result JSON pins all 128 goal IDs and the source-bound case evidence.
The independent evidence audit is in
`reviews/el2-el3-positive-figure-audit.json` in the result directory. It
checks raw numerical/L2 receipts, paired contracts, source hashes, and owned
token streams. Retained sources and verdicts exist; actual emitted ELFs were
cleaned by the runs, so this is not a retained executable replay bundle.
The existing figure review found no clipping or legibility blocker.

No campaign, compiler, simulator or FireSim run was launched for this analysis.
The original CSV and original figure release are unchanged. These outputs
remain on `cc/el4-grader-throughput-20261024`; nothing is pushed.
