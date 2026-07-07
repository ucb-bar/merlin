# Residency plot semantics (`decision_weight_residency`)

Source: `data_movement_table.csv` (repeated-head regions, `weight_bytes` + `invocations`).

- **y-axis = weight BYTES MOVED**, not bandwidth. We never divide by a deadline here, so there is no implied
  time and no implied hardware. (A bandwidth *requirement* is a different plot — `realtime_requirement`.)
- **reload-every-step** curve = `weight_bytes × K` (grows linearly with the loop count). **resident** = the
  flat `weight_bytes` line (loaded once). The gap is the avoidable reload.
- **The dot = the model's actual K**, taken from `invocations`, which is the **`scf.for` trip count recovered
  from the IR** (Tier A) for the loop-preserving captures. Where a workload's K came from source/config
  rather than the IR it is labeled as such; for this corpus the repeated-head K is IR-recovered.
- **Scale = captured-config** (the byte magnitudes are from the reduced captures); the *shape* of the curve
  (linear vs flat) is the structural point and is scale-independent.
- Title: *"Loop-visible residency: non-resident weight traffic grows with K."* Caveat subtitle:
  *"bytes moved (not bandwidth); dot = model's K (IR scf.for); captured-config scale."*
