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

The paid loop contains the fourteen `_l2` capsules synthesized from real contraction classes observed in
the declared SmolVLA and LSTMNetVIT captures.  These retain the full application shapes and run through
the fast functional oracle.  No capsule selected because it resembles Radiance PR #1 is in this cohort.

## Stage 2: frozen-candidate GSIM evaluation

After search converges, freeze the package and materialize `derived_gsim`.  It contains the exact same
fourteen full application-shape capsules—not their reduced `_l3` lookalikes.  Materialization removes
the search-time L2 ceiling and makes L3 mandatory in the isolated copy; the committed source capsules
remain unchanged.  The reduced siblings are suitable for GSIM smoke tests only and cannot contribute to
the proper derived-workload score.

```bash
PYTHONPATH=merlin/python .venv/bin/python -m merlin.targetgen.evaluation_cohort \
  --target radiance --stage derived_gsim \
  --dest out/artifacts/capsule-bench/radiance/<run>/derived_gsim
```

The command exits 2 if the selected GSIM executable is unavailable, is not the descriptor-requested
engine, or lacks a receipt binding its bytes to the elaborated FIRRTL.  A populated cohort is therefore
not automatically runnable or certified.  Once preflight is green, grade the frozen package against that
single materialized root with `merlin.targetgen.capsule_grade`.

## Stage 3: independent kernel-library comparison

Only a package that passes `derived_gsim` advances to `kernel_library_comparison`.  That cohort contains
the 24 public capsules corresponding to the external PR's GEMM, MX, attention, normalization, activation,
embedding, patch, and fused-operation workloads.  It is also materialized with mandatory L3 GSIM.

```bash
PYTHONPATH=merlin/python .venv/bin/python -m merlin.targetgen.evaluation_cohort \
  --target radiance --stage kernel_library_comparison \
  --dest out/artifacts/capsule-bench/radiance/<run>/kernel_library_comparison
```

The comparison remains independent: PR sources may inform a separately labelled information treatment,
but submitted compiler code may not copy, link, call, or dispatch on the reference library, capsule names,
or expected outputs.  Search results and the two post-search scores must be reported separately.
