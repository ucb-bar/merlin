# Phase-2 compute-feature existing-evidence audit

Conclusion: **incomplete; no compute feature receipt is admissible**.

`preparation.json` is the canonical output of `phase2_feature_calibration` over the
17 content-addressed candidate files named by `request.json`. It re-read all candidate bytes, found
zero valid controlled points, and refused to derive a coefficient: two distinct points are required
for the one fitted compute parameter.

The closest old evidence is insufficient for different reasons:

- `counter_partition_gsim_20260908` has exact command-buffer MAC counts and measured GSIM cycles,
  but has no controlled-variable/observation contracts, no exact target/source binding in the
  measured result, and no warm predecessor. Execute-busy cycles were not treated as issued MACs.
- `development_masked_program_warm_20260907` has one warm and one measured invocation, but its
  original receipt is refused, its supplement is one point only, and it lacks target/controls and
  observation bindings.
- The two complete-source-pair reduced GSIM runs have warm/measured counts and a shared exact target
  digest, but they were compiler before/after probes rather than a sealed compute-volume series.
  Their selected arms record `declared_plan_status=refused` and `target_route_verified=false`, and
  they expose no exact emitted-compute quantity in the Phase-2 observation schema.
- Spike warm/measured receipts were excluded because Spike is not FPGA or RTL performance truth.

No target, simulator, complete layer/model, FireSim, or L3 execution occurred during this audit.
See `AUDIT.json` for the exact missing-proof inventory and content hashes.
