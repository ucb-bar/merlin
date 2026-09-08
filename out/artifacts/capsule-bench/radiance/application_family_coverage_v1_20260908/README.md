# Radiance declared-application family audit

This artifact audits only the six `workload_spec.applications` captures declared by Radiance.  It never
uses the PR-kernel comparison cohort as application evidence.

The structural census contains 13,186 regions (13,046 classified): 5,780 elementwise maps, 3,469
movements, 1,457 contractions, 1,341 normalizations, 915 reductions, 84 attention regions, and 140
unclassified regions.  All 14 pre-audit search capsules were contractions.  The new 15th member is an
exact `add` slice from `smolvla_fp32_consistent`, preserving `tensor<1x113x1xf32>`; that exact semantic
shape occurs 99 times across the six captures.

`l2_receipt.json` records a Cyclotron L2 positive pass over all 113 elements and an answer-key negative
control that fails at the perturbed element.  The public command buffer and emitted LLVM MLIR are
byte-identical between the arms.  Cyclotron is not GSIM or physical RTL, and the receipt qualifies only
this exact capsule program.  It does not qualify other elementwise operations or any broader family.

The remaining non-contraction disposition is fail-closed:

- movement (3,469), normalization (1,341), reduction (915), and attention (84) have no search capsule;
  Radiance's effective capability map does not admit them for a genuine `must_accelerate` probe;
- the other elementwise operations and shapes remain unqualified by the one add receipt;
- the 140 unclassified regions remain unclassified rather than being assigned to a convenient family.

Recompute the structural census (about 31 seconds on this host):

```bash
PYTHONPATH=merlin/python .venv/bin/python \
  out/artifacts/capsule-bench/radiance/application_family_coverage_v1_20260908/audit.py --write
```

Replay an L2 arm with `run_l2.py --case positive|negative --work ... --publish ...`.  Verify the sealed
report and receipts without simulation:

```bash
PYTHONPATH=merlin/python .venv/bin/python \
  out/artifacts/capsule-bench/radiance/application_family_coverage_v1_20260908/verify.py
```
