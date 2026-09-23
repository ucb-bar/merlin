# Verification analysis

Owns verification figure generation (`merlin.verify.plots`) and its study collectors.
Owns historical replay (`merlin.verify.replay` and `merlin.verify.replay_layers`), including its trusted
qualification bootstrap. Install `merlin-analysis[replay]` for pytest, core verification dependencies
and the experiments-owned numeric evaluator. Actual replay also needs the source checkout's history,
tests, corpus and declared tools; an installed wheel alone cannot reproduce that research input.
The compiler verifiers and `merlin.verify` namespace initializer remain in core.
Do not add an initializer here or duplicate solver/compiler implementation.

Install the analysis `plots` extra to render figures. Measurements come from input records or
explicit collection, never literals; unavailable verification remains distinct from a passing
result. Retain the historical module entrypoint for existing analysis commands.

Preserve `executed_checks_v2`, frozen namespace isolation, exact historical population definitions,
and the explicit unqualified lit limitation. Missing dependencies, runtime/import failures and
all-skipped checks are unavailable, never detections. Never rewrite previous replay measurements.
