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

## Scope: an admitted search covering set, not whole-model coverage

The paid loop contains sixteen `_l2` capsules derived from the declared SmolVLA and LSTMNetVIT
captures: the original fourteen contraction-class members, one exact-rank fp32 add slice at
`tensor<1x113x1xf32>`, and one exact-rank fp32 multiply slice at `tensor<32xf32>`. The add closes the
highest-frequency missing captured family. The multiply then covers the largest still-unrepresented
operation inside the only other standalone family Radiance admits: 1,533 captured `mul` regions, of
which 168 have the exact admitted shape. Both operations have semantics the target-neutral PyTorch
writer and Muon emitter reproduce; their sources point to byte-pinned capture regions, not PR-kernel
shapes. These retain application shapes and run through the fast functional oracle. No
capsule selected because it resembles Radiance PR #1 is in this cohort.

This is the current **model-derived admitted search covering set**.  A 16/16 result is not an E2E
SmolVLA/LSTMNetVIT result and must not be reported as whole-application readiness.  The census contains
13,046 classified regions. The fourteen contraction shapes plus the two exact maps do not represent
attention, movement, normalization, reduction, or other elementwise operations. Those families and
operations remain fail-closed/unrepresented until the target admits and implements them; future
model-derived cohorts should extend this set at that point, rather than silently broadening today's claim.

The family census is deliberately broader than the cohort claim.  Across the six declared captures it
finds 13,186 regions: 5,780 elementwise maps, 3,469 movements, 1,457 contractions, 1,341 normalizations,
915 reductions, 84 attention regions, and 140 unclassified regions. Before the application maps, the
fourteen search members represented only contraction (1,457/13,046 classified occurrences, 11.168%).
The exact add and multiply capsules do not qualify any other elementwise operation or shape. Movement,
normalization, reduction, and attention also remain outside search because Radiance's effective
capability map does not currently admit those families for a `must_accelerate` capsule; the 140
unclassified regions remain evidence for
neither coverage nor a gap.  The machine-readable census and L2 receipt are under
`out/artifacts/capsule-bench/radiance/application_family_coverage_v2_20260908/`.

## Stage 1: converge and seal the exact L2 pass

Run the paid loop only on the descriptor's sixteen-member `search_cohort`.  A score can open Stage 2
only when its per-capsule evidence names exactly those sixteen members (including the exact add and
multiply), every row records an L2 numeric pass and an execution digest, and the suite reports 16/16.
Seal that score with the candidate and source-tree digests; the seal must live outside the candidate.

The self-check runner defaults to the repository's default target when no descriptor is supplied, so
Radiance selection must be explicit.  A canonical clean Stage-1 run is:

```bash
MERLIN_TARGET_EXPERIMENT=merlin/experiments/capsule_bench/targets/radiance/target_experiment.yaml \
MERLIN_ELF_BUILD_CACHE=0 MERLIN_MUON_TRUSTED_COMPACT_NUMERIC=1 \
PYTHONPATH=merlin/python .venv/bin/python \
  merlin/experiments/capsule_bench/harness/agent_selfcheck.py \
  --submission out/runs/radiance/capsule-bench/<run>/submission \
  --sim spike --tiers L2 --capsules all --workers 4 --timeout 300 \
  --out out/artifacts/capsule-bench/radiance/<run>/l2_score.json \
  --progress-out out/artifacts/capsule-bench/radiance/<run>/l2_progress.json
```

`MERLIN_MUON_TRUSTED_COMPACT_NUMERIC=1` changes only the target-owned result transport: the complete
full-shape kernel still executes, and the private post-compile nonce check still covers every output.
It is not a reduced-shape evaluation or a substitute for the numeric verdict.

```bash
PYTHONPATH=merlin/python .venv/bin/python -m merlin.targetgen.evaluation_cohort \
  --target radiance \
  --candidate out/runs/radiance/capsule-bench/<run>/submission \
  --search-score out/artifacts/capsule-bench/radiance/<run>/l2_score.json \
  --create-search-pass-seal out/artifacts/capsule-bench/radiance/<run>/search_l2_pass.json
```

The command refuses 15/16, a missing or extra capsule, a non-L2 row, a missing execution digest, a
non-model-derived source, a stale score, or a candidate/source tree changed after the run.  Creating the
seal does not certify unrepresented operation families or an E2E model.  The candidate digest excludes
only Python interpreter byproducts (`__pycache__`, `.pyc`, `.pyo`), so importing the frozen package during
grading does not invalidate it; authored source, manifests, configuration, schedules, and binaries remain
strictly content-addressed.

## Stage 2: frozen-candidate GSIM evaluation

After the sealed search pass, materialize `derived_gsim`.  It contains the exact same
sixteen full application-shape capsules—not their reduced `_l3` lookalikes.  Materialization removes
the search-time L2 ceiling and makes L3 mandatory in the isolated copy; the committed source capsules
remain unchanged.  The reduced siblings are suitable for GSIM smoke tests only and cannot contribute to
the proper derived-workload score.

```bash
PYTHONPATH=merlin/python .venv/bin/python -m merlin.targetgen.evaluation_cohort \
  --target radiance --stage derived_gsim \
  --candidate out/runs/radiance/capsule-bench/<run>/submission \
  --search-pass-seal out/artifacts/capsule-bench/radiance/<run>/search_l2_pass.json \
  --dest out/artifacts/capsule-bench/radiance/<run>/derived_gsim
```

Materialization happens only after all gates pass.  It refuses an unavailable or non-GSIM engine, an
override that is not the canonical install, an unbound receipt, or any digest mismatch; it records the
canonical emulator and receipt paths and SHA-256 identities.  The current canonical emulator digest is
resolved at runtime rather than copied into this document.  The cohort manifest also seals the complete
candidate tree and the Stage-1 seal.  Validation fails if any compiler, score, capsule source, emulator,
or receipt byte changes between convergence and grading.

Grade the frozen package against only that materialized root:

```bash
PYTHONPATH=merlin/python .venv/bin/python -m merlin.targetgen.capsule_grade \
  --target radiance \
  --package out/runs/radiance/capsule-bench/<run>/submission \
  --capsules out/artifacts/capsule-bench/radiance/<run>/derived_gsim \
  --runs-root out/artifacts/capsule-bench/radiance/<run>/derived_gsim_runs \
  --score out/artifacts/capsule-bench/radiance/<run>/derived_gsim_score.json
```

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
Its manifest preserves hashes of the predecessor manifest, predecessor materialized tree, score,
candidate, canonical GSIM binding, and original Stage-1 seal.  A changed engine/receipt or edited
predecessor therefore closes the gate.
The comparison remains independent: PR sources may inform a separately labelled information treatment,
but submitted compiler code may not copy, link, call, or dispatch on the reference library, capsule names,
or expected outputs.  Search results and the two post-search scores must be reported separately.

## Superseded artifact

`out/artifacts/capsule-bench/radiance/staged_derived_gsim_exact_sealed_20260908` predates this protocol.
It contains 14 capsules, names `hand_v0`, and records the old unreceipted-engine failure.  It is preserved
as historical failure evidence, but it cannot be resumed, extended, or cited as Stage 2.  The workflow
must create a new destination only after a current exact 16/16 L2 seal exists.  No GSIM run should start
before that gate, and no PR #1 capsule belongs in compiler search.
