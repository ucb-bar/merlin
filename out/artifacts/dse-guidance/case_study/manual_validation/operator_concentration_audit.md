# Operator-concentration audit (P19 Phase 4)

Validates the P16/P18 operator-concentration story against source + MLIR. Magnitudes structural-only.

## Findings

- **rdt "one giant op" is real but smaller and non-generalizing.** The dominant op is
  `model.blocks.1.cross_attn.kv` = `nn.Linear(2048, 4096)` over a **4096-token image context**
  (RoboticsDiffusionTransformer/models/rdt/blocks.py:95; flat MLIR matmul_20 = 4096×2048×4096). The true
  share is **84.6%** of recovered MACs (not 87%). **It is a depth=2 capture artifact**: only block 1 sees
  `img_c` (`conds[i%2]`, model.py:158), so at the real depth=28 it would be ~14 such ops, not 1. **rdt2 —
  the sibling — has NO such op** (largest is a 0.12 GMAC FFN `fc1`); it captures the KV-cache *as inputs*
  (host-computed), so it is FFN/projection-dominated. ⇒ **"one hot op dominates" does NOT generalize; it is
  an rdt-at-depth-2 phenomenon.** Present it as "RDT *at this capture depth*", never as a corpus law.

- **pi05 is "many instances of few shapes", not shape-diverse.** 777 `linalg.matmul` = the gemma_expert
  depth-18 layers ×~43 matmuls (openpi/src/openpi/models/gemma.py:73), NOT loop-unroll (capture is one
  denoise step). **Distinct (M,N,K) = 17 — Merlin is CORRECT** (the earlier "20" was a CSV misparse of
  `prov_fqn` containing `slice(None, 18, None)` commas). Top 3 shapes = 83.6% of MACs, top 6 = 98.3%; one
  vision shape appears 324×. ⇒ the operator-Pareto "many ops for 50% MACs" is **instance-count
  concentration**, correctly captured now by `n_distinct_shapes`/`top_shape_multiplicity`.

- **Attention concentration was mis-measured — now fixed (P19).** Two real classifier bugs:
  (1) **xr0 under-counted** — SDPA-fused attention lowers to `linalg.generic` tagged `prov.op="sdpa"`/
  `family="attention"`, which the classifier missed → `n_attention=0`, `visible_linear_fraction` wrongly
  1.0. Fixed → 14 attention ops, fraction 0.9923. (2) **groot over-counted** — `CategorySpecificMLP`
  `torch.bmm` (fqn `head.action_encoder.W*` / `action_decoder.layer*`) shares `prov.op="batch_matmul"` and
  was labeled attention (11 ops, ~96% of "attention" MACs were MLP bmm). Fixed → 4 true attention
  (fqn `...attn1`) + 7 `batched_matmul` (341M MACs, separated). Root cause: keying attention on
  `prov.op=="batch_matmul"` alone is both incomplete and imprecise.

- **Top-op tables are source-real elsewhere** (openvla, molmoact, smolvla, *_llama): the top GEMMs map to
  real q/k/v/o/gate/up/down projections with correct shapes (each cited in the per-workload audit).

## Verdict for the concentration plots
- `operator_cumulative_mac` / Pareto: **main-slide, with the rdt caveat** ("at depth 2; rdt2 shows the
  opposite"). The few-giant (rdt) vs many-even (pi05) contrast is real but must be framed as capture-depth-
  and architecture-dependent, not a universal regime split.
- `work_coverage_by_workload` / `visible_linear_fraction`: **valid after the P19 attention fix** (xr0/groot
  corrected); now also report `batched_matmul` MACs so MLP-bmm is not conflated with attention.
