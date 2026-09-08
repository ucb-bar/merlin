---
title: Radiance search, GSIM certification, and kernel-library comparison
kind: design
status: current
owner: targetgen
last_verified: 2026-09-08
code_refs:
  - merlin/experiments/capsule_bench/targets/radiance/target_experiment.yaml
  - merlin/python/merlin/targetgen/evaluation_cohort.py
---

# Radiance staged evaluation

Radiance compiler search and Radiance kernel-library comparison are different experiments.  Combining
them in one repeated 29-capsule loop both teaches the compiler the comparison suite and spends expensive
feedback on shapes that are not the declared applications.  The target descriptor now makes the order
explicit.

## Stage 1: application-derived convergence

The paid loop contains fifteen `_l2` capsules derived from the declared SmolVLA and LSTMNetVIT
captures: the original fourteen contraction-class members plus one exact-rank fp32 add slice at
`tensor<1x113x1xf32>`.  The add closes the highest-frequency missing captured family with semantics the
target-neutral PyTorch writer can reproduce; its source points to a byte-pinned capture region, not a
PR-kernel shape.  These retain application shapes and run through the fast functional oracle.  No
capsule selected because it resembles Radiance PR #1 is in this cohort.

The family census is deliberately broader than the cohort claim.  Across the six declared captures it
finds 13,186 regions: 5,780 elementwise maps, 3,469 movements, 1,457 contractions, 1,341 normalizations,
915 reductions, 84 attention regions, and 140 unclassified regions.  Before the add, the fourteen search
members represented only contraction (1,457/13,046 classified occurrences, 11.168%).  The single add
does not qualify the other elementwise operations or shapes.  Movement, normalization, reduction, and
attention also remain outside search because Radiance's effective capability map does not currently
admit those families for a `must_accelerate` capsule; the 140 unclassified regions remain evidence for
neither coverage nor a gap.  The machine-readable census and L2 receipt are under
`out/artifacts/capsule-bench/radiance/application_family_coverage_v1_20260908/`.

## Stage 2: frozen-candidate GSIM evaluation

After search converges, freeze the package and materialize `derived_gsim`.  It contains the exact same
fifteen full application-shape capsules—not their reduced `_l3` lookalikes.  Materialization removes
the search-time L2 ceiling and makes L3 mandatory in the isolated copy; the committed source capsules
remain unchanged.  The reduced siblings are suitable for GSIM smoke tests only and cannot contribute to
the proper derived-workload score.

```bash
PYTHONPATH=merlin/python .venv/bin/python -m merlin.targetgen.evaluation_cohort \
  --target radiance --stage derived_gsim \
  --candidate out/artifacts/targets/radiance/<frozen-package> \
  --dest out/artifacts/capsule-bench/radiance/<run>/derived_gsim
```

The command exits 2 if the selected GSIM executable is unavailable, is not the descriptor-requested
engine, or lacks a receipt binding its bytes to the elaborated FIRRTL.  A populated cohort is therefore
not automatically runnable or certified.  The cohort manifest also seals the complete candidate tree;
validation fails if any compiler byte changes between convergence and GSIM.  Once preflight is green,
grade that frozen package against the single materialized root with `merlin.targetgen.capsule_grade`.

## Stage 3: independent kernel-library comparison

Only a package that passes `derived_gsim` advances to `kernel_library_comparison`.  That cohort contains
the 24 public capsules corresponding to the external PR's GEMM, MX, attention, normalization, activation,
embedding, patch, and fused-operation workloads.  It is also materialized with mandatory L3 GSIM.

```bash
PYTHONPATH=merlin/python .venv/bin/python -m merlin.targetgen.evaluation_cohort \
  --target radiance --stage kernel_library_comparison \
  --candidate out/artifacts/targets/radiance/<same-frozen-package> \
  --dest out/artifacts/capsule-bench/radiance/<run>/kernel_library_comparison \
  --predecessor-cohort out/artifacts/capsule-bench/radiance/<run>/derived_gsim \
  --predecessor-score out/artifacts/capsule-bench/radiance/<run>/derived_gsim_score.json
```

Stage 3 opens only after that predecessor evidence proves the exact declared names passed L3 on an
RTL-backed engine, the score is clean and gradeable, and the frozen candidate path and digest match.
The comparison remains independent: PR sources may inform a separately labelled information treatment,
but submitted compiler code may not copy, link, call, or dispatch on the reference library, capsule names,
or expected outputs.  Search results and the two post-search scores must be reported separately.
