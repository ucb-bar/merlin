# Merlin analysis

`merlin-analysis` owns baseline comparisons, measurement studies, and reporting.
Install it independently for `merlin-compare`, `merlin-bundle-pretranspose`, `merlin-recovery`, and
baseline tooling. Its direct AET dependency owns the study/provenance API; ordinary
analysis does not install the agent experiment harness.

Install `merlin-analysis[experiments]` for agent-study controllers and reports that
consume `merlin.benchharness` or the experiment evaluator. Add the `plots` extra for
Matplotlib figures, or use `merlin-analysis[experiments,plots]` for both.

Core capture and ISA audit APIs remain in `merlin.capture` and `merlin.runtime`.
Historical `merlin.baselines`, `merlin.compare`, `merlin.plotting`, `merlin.agentreport`, and
`merlin.verify.plots`, and `merlin.perf.recovery` imports keep
their identities through the shared namespace; this distribution does not own or
replace the core namespace initializers.

Verification figures are analysis-owned; compiler verification itself remains in core.
Use the `plots` extra to redraw saved records with `python -m merlin.verify.plots --from DIR`.

Historical replay is analysis-owned too, retaining `python -m merlin.verify.replay` and
`python -m merlin.verify.replay_layers`. Install `merlin-analysis[replay]` for pytest, core's verification
extra and the numeric evaluator. Running a replay still needs a Merlin source checkout with the
pinned Git history, tests and corpus, plus the declared external tools; importing the modules and
showing `--help` does not run a study. New records retain `executed_checks_v2` and explicitly report
lit as unqualified until its child commands provide isolated receipts. This is not a claim that all
five layers are usable, and old historical measurements are not rewritten.

Ceiling measurement controllers retain their `merlin.kernels.ceiling_drivers` imports here;
their shared native C/header resources remain core-owned. Generated workload comparisons need
the `mining` extra (`merlin-analysis[mining]`). Importing these controllers does not run a
measurement; invoking their main functions can compile and execute external tools.
Collecting fresh formal measurements also needs core's `verify` extra and the declared toolchains;
missing prerequisites remain unavailable measurements, not passing verification.
