# Radiance declared-application operation audit, version 2

This artifact audits only the six `workload_spec.applications` captures declared
by Radiance. It never uses the PR-kernel comparison cohort as application
evidence.

The structural census remains 13,186 regions (13,046 classified): 5,780
elementwise maps, 3,469 movements, 1,457 contractions, 1,341 normalizations,
915 reductions, 84 attention regions, and 140 unclassified regions. Movement is
the largest missing family, but Radiance's effective capability map does not
admit it. Normalization, reduction, and attention fail the same admission check.

Within the admitted standalone `elementwise_map` family, `mul` is the largest
operation not represented after the exact add probe: 1,533 captured regions.
The new 16th cohort member preserves the exact `tensor<32xf32>` operands and
result from `smolvla_fp32_consistent` structural region 904
(`prov.region_id=mul_22`). That exact semantic shape occurs 56 times in the
anchoring capture and 168 times across the six declared captures.

`l2_receipt.json` records a Cyclotron L2 positive pass over all 32 elements and
an answer-key negative control that fails at the perturbed element. The public
command buffer and emitted LLVM MLIR are byte-identical between the arms.
Cyclotron is not GSIM or physical RTL, and this receipt qualifies only this exact
capsule program. It does not qualify other elementwise operations or shapes.

Recompute the structural census:

```bash
PYTHONPATH=merlin/python .venv/bin/python \
  out/artifacts/capsule-bench/radiance/application_family_coverage_v2_20260908/audit.py --write
```

Replay an L2 arm with `run_l2.py --case positive|negative --work ... --publish
...`. Verify the sealed report and receipts without simulation:

```bash
PYTHONPATH=merlin/python .venv/bin/python \
  out/artifacts/capsule-bench/radiance/application_family_coverage_v2_20260908/verify.py
```
