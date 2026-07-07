# Known limitations

Explicit, so no later phase obscures what is already defensible. Each limitation names what is
real, what is *not* claimed because of it, and what would lift it.

## 0. Corpus coverage: 7 of 11 registry models studyable
The committed study covers **rdt, openvla, small_llama, tiny_llama, rdt2, groot_n1d7, molmoact**.
Four registry models are captured (via model2MLIR with `prov.fqn`) but **deferred**, each for a
concrete, named reason — not a silent omission: **xr0** is batched-attention DiT where only 2 of 19
matmuls are plain-2D, so the 2D-matmul-centric geometry skips it (*lift:* `linalg.batch_matmul`
support in `extract_matmuls`); **bitvla** and **smolvla** parse-fail in the ingest xDSL on
`tensor.collapse/expand_shape` with typed-reassociation (`[[0 : i64, …]]` / `output_shape`) syntax
(*lift:* a reassociation-syntax normalizer or an xDSL bump in `mlir_m2m._parse_module`); **pi05** is
the full PaliGemma+expert VLM (~13k ops) — too large/imbalanced for the comparable small-config
corpus (*lift:* a denoise-step-only tap or a layer-reduced variant). All four can be promoted as the
lift lands; the captures already exist.

## 1. Magnitudes are small, random-init instances
The recaptures are small-config / random-init architectures (no weight download). **Structure and
provenance are real; absolute magnitudes are not the deployed models.** Per-workload weight figures
differ in scale because they are different toy configs (`resident_state_table.csv`: rdt 391 MB,
tiny_llama 614 MB, openvla 3 MB, small_llama 1.6 MB) — read these as "the machinery works", not as
deployed-model sizes. *Lift:* full-depth recaptures with real weights.

## 2. K, H, control-rate, replan deadline are assumed references
All loop counts and timing are `assumed_reference`, not measured from a deployment. Every derived
requirement (compute rate, bandwidth, command rate, avoidable reload) is correct analytical math
**resting on those assumed inputs**. *Lift:* real K / control-rate from the deployed loop.

## 3. Accuracy gate is int8-only and host-interpreter
`accuracy_gate_results.csv` records **measured** W8A8-vs-fp32 (5 models pass) — but on the host
interpreter (`docs/results.md`), and only int8. fp8/int4/fp4/fp6 are honestly `unavailable`.
*Lift:* run the per-format accuracy gates.

## 4. Dispatch cost is not measured (only the count is)
`dispatch_coupling.csv` measures dispatch **count** (matmul proxy under-counts ~12–14×). The
per-dispatch host **cost** is host-interpreter timing, not the deployable runtime, so no latency or
speedup is derived from it. *Lift:* one real command/sync latency measurement on the target runtime.

## 5. No per-component cycle calibration
chipyard/spike/FireSim are unavailable in this environment. The earlier whole-model calibration was
a ~147,000× honest miss (`cost_calibration.md`) and is demoted to a sanity anchor — it is **not** a
predictor. *Lift:* per-component microbenchmarks (compute/memory/dispatch/epilogue) on an available
target.

## 6. Numerical metadata erased by the flat capture
Scale / zero-point / group-size and sparsity metadata are not present in the post-dequant capture;
`numerical_contract.yaml` marks them `erased_or_unavailable` rather than inventing them. The
cross-zoo finding (low-bit stored, f32 computed) is itself real. *Lift:* a capture path that
preserves packed layout + scale metadata.

## 7. State-lifetime table is currently thin
`resident_state_table.csv` surfaces only `weights` (loop-invariant). The boundary-crossing /
loop-carried machinery exists and is unit-tested, but the recaptured topologies do not populate
`produces`/`consumes`/`loop_carried_state` (no hand-authored sidecar), so prefix/KV/latent crossings
do not appear for these workloads. Honest gap, not a bug. *Lift:* a Level-0 topology sidecar (or
Level-2 loop-preserving capture).

## 8. Command-graph is structural-only (loop unrolled)
`dispatch_granularity_table.csv` / `command_graph.yaml` give a matmul-count **proxy** for commands
per step (a lower bound — measured dispatch granularity is ~12–14× higher) and the dispatches-per-
replan proxy scaled by K. **Syncs, per-step dependencies, and in-loop allocations are `unavailable`**
because `torch.export` unrolled the host loop — a true command graph is not recoverable from the
flat capture. This artifact is largely an honest restatement, by design. *Lift:* Level-2
loop-preserving capture.

## 9. Memory envelope omits intermediate / layout-conversion traffic
`traffic_table.csv` covers weight + activation traffic and avoidable reload (recovered/derived).
Intermediate-materialization and layout-conversion bytes are **not** exposed by the flat capture and
are emitted `unavailable`, never estimated.

## 10. Abstraction-pressure detectors are coarse
All four workloads suggest most axes; `evidence_strength` is `strong` only for the two head-attributed
axes (`resident_action_head_weights`, `packed_layout_preservation`) and `structural_only` otherwise.
The ranking is a structural **count**, never a speedup.

## By design (not deficiencies)
Merlin does not run DSE, does not pick a design, and claims no speedup/cycle/energy/area. It emits a
contract + a search-space template + a measurement plan for a future DSE engine to consume.
