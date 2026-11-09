---
title: Reproduce a historical verification instrument
kind: guide
status: current
owner: verification
last_verified: 2026-11-09
related: [../design/compiler_verification, reproducing_paper_results]
code_refs:
  - src/merlin/verify/replay.py
  - src/merlin/verify/replay_layers.py
---

# Reproduce a historical verification instrument

Historical detection rates describe the exact instrument that produced them.
Reproducing one requires its recorded Git revision or frozen source snapshot,
original test and corpus inputs, selected dependencies and tools, population,
sample seed, layer commands and qualification policy. Restore those inputs into
a separate working tree. Keep the original evidence immutable.

The compact active suite retires three framework selftest modules referenced by
the optional live historical replay: `engines-agree`, `cb-semantics` and
`compilation-validation`. Their layer declarations remain historical identities.
After pruning, those layers are unavailable in the live checkout. A missing
module produces `error`, neither `green` nor `red`; it cannot count as a clean
comparison or a detected defect. The replay summary reports unusable baseline
layers. An entirely unavailable baseline disqualifies the measurement.

Do not remap these layer names to the compact suite, copy retired test bodies
into production qualification code, or interpret a run of fewer usable layers
as the original instrument. Use the original Git/frozen instrument for paper
reproduction. A run using different checks is a different experiment and cannot
replace a saved detection rate.

The `executed_checks_v2` policy still requires evidence that a comparison
actually executed. Import, setup and runtime failures, missing dependencies and
all-skipped checks remain unavailable. Lit's child commands remain unqualified
for replay until they supply the required execution receipts. Timeouts remain
distinct from rejections. Preserve all sampled commits, unreplayable outcomes
and the recorded denominator rather than discarding unavailable cases.

This retirement concerns a research instrument's framework selftests.
Production verification remains in `src/merlin/verification/`: actual source
and target artifacts, selected semantics, checker/tool identities and live
completion determine acceptance. The compact suite and historical replay do
not replace those algorithms or confer numerical or accelerator authority.
