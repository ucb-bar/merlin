# Final judgment (P19 Phase 6 · updated P21)

Source-grounded verdict on every current result/plot, after the 11-workload forensic audit (S1) and the
correctness fixes (S3). Classification ∈ {main-slide, backup, QA-only, needs-fix, invalid}. All structural;
no perf claims. "Evidence" = source/MLIR audit status.

## P21 UPDATE — every caveat / blocked item resolved with IR or config evidence

The "torch.export-blocked" frontier is broken: model2MLIR now lowers `torch.while_loop → scf.for`
(commit `aec234b`), so the K-step loop is preserved in the IR. Net status of the formerly-open items:

| caveat / blocked item (P19/P20) | status now | evidence |
|---|---|---|
| **assumed/configured K** | **resolved** | K is the `scf.for` upper-bound constant (`prov.op=while_loop`): smolVLA/pi0.5 K=10, openVLA K=7 — `loop_preserving_recovery.csv` |
| **KV-state sizing (blocked)** | **resolved** | openVLA static KV cache is a recovered `scf.for` iter_arg (`2×4×128×27×2×4 = 221184 B`); the formula re-applied at Llama-2-7B scale = 137.9 MB bf16 — `kv_cache_sizing.csv` (IR-validated) |
| **loop-carried state (erased)** | **resolved** | the denoise latent `[1,50,32]` (smolVLA/pi0.5) and the KV cache (openVLA) are explicit `iter_args` — `loop_preserving_recovery.csv` |
| **region roles low-confidence** | **resolved (structural)** | the `scf.for` body IS the repeated head (597–6208 ops), recovered structurally, not by fqn heuristic |
| **structural-only / tiny-config magnitudes** | **resolved** | deployment-real magnitudes are config-exact compositions (`prefix + per-layer × n_layers`): openVLA 6.74 B / 13.5 GB bf16, pi0.5 2.82 B, smolVLA 0.58 B — `real_config_magnitudes.csv` (weight VALUES irrelevant) |
| **native low-bit datapath (blocked)** | **resolved (storage+scale)** | bitvla `BITVLA_NATIVE_QUANT=1` recapture shows 391 packed-int2 ternary weight tensors + per-tensor absmean scale — `native_lowbit_datapath.csv` (residual: bit-unpack stays opaque, see note) |

Verified end-to-end: `verify_implementation.py` 679/679 (P21 checks re-derive K/KV/params/native-storage
independently from the IR/config). **Corpus-complete:** all **11/11** workloads now have a loop-preserving
capture (one sub-agent per model, each numerically verified vs the eager unrolled loop), so the
capture-fidelity headline reads *recovered (K=N, IR scf.for)* / *recovered (B, IR iter_arg)* for the entire
corpus — K∈{4,5,7,8,10}; the flow/diffusion models carry the action latent (rdt also carries DPM-solver
state), the autoregressive models carry a static KV cache as an explicit `scf.for` iter_arg. See
`loop_preserving_recovery.csv` (11 rows) and the per-model wrappers under `p21_loop_preserving/`.

**P22 follow-on (additive, flat corpus untouched):** (A) deployment-real magnitudes extended to the
standard decoder-LLMs whose config is fully sourced — `real_config_magnitudes.csv` now covers 6 models
(tiny_llama composes to exactly 1.1 B = TinyLlama-1.1B; DiT/diffusion + bitvla omitted, no guessed
fields). (B) `loop_aware_contract.csv` joins the IR loop facts (K, repeated region, resident-eligible vs
loop-carried operands) with the flat-capture weight bytes into one per-workload contract — the avoidable
reload (`resident_weight_bytes × (K−1)`) is now IR-backed (K from `scf.for`, weights proven
loop-invariant). (C) `residency_from_ir.csv` (all 11). Verifier 682/682.

| result / plot | class | what it gets right | what it gets wrong / caveat | answers a DSE decision? |
|---|---|---|---|---|
| **capture-fidelity matrix** (+ erasure evidence) | **main-slide** | the central result: what export/lowering erase (loop/KV/low-bit) vs preserve (shapes/dtype); now demonstrated from IR (scf.for only in smolvla = gather artifact; no low-bit types) | attention row must say "recovered (re-parsed from generics)", not erased | yes — sets which axes a DSE tool can even ask about |
| **work_coverage / visible_linear_fraction** | **main-slide** (post-S3) | answers "95% or 40%": linear-GEMM is 82–99% of recovered MAC work; attention recovered from IR (no config) | only valid after the S3 attention fix (xr0/groot); magnitudes structural-only | yes — how much of compute the linear datapath serves |
| **primitive-set frontier** (+ by-threshold) | **main-slide w/ 2 caveats** | a 2-primitive set covers the corpus where 1 fails; robust across thresholds/LOO | (1) linear-GEMM-only (attention excluded); (2) "GEMV-like" is **capture-M-induced** (tiny/small_llama M=4/8 prefill, not decode); specific pair is threshold-sensitive | yes — search primitive SETS, not one tile |
| **operator cumulative-MAC / Pareto** | **main-slide w/ caveat** | real few-giant (rdt) vs many-even (pi05) contrast; pi05 = instances-of-17-shapes (correct) | rdt's 84.6% giant op is **depth-2 artifact, does NOT generalize** (rdt2 FFN-dominated) — frame as "RDT at this depth" | yes — hot-op specialization vs broad coverage |
| **abstraction necessity matrix** (categorical) | **main-slide w/ caveat** | discriminating; low-bit + KV correctly blocked with precise reasons (source-has/export-erases) | resident_weight_object/loop necessity rests on **configured K**; region roles low-confidence in single-step captures | yes — what to commit to vs merely permit |
| **residency vs K / capacity×dtype** | **main-slide** (K-caveat) | reload-grows-with-K vs resident-flat is structurally valid; int4<int8<bf16 is exact byte-scaling | K is assumed (loop unrolled); absolute bytes random-init | yes — residency + capacity/dtype knobs |
| **requirements envelope** (P17) | **main-slide** | requirements (work/deadline), explicitly not measured; residency removes K× bandwidth | command-rate proxy-only except small_llama; all inputs config/sweep-tagged | yes — maps to robotics deadlines |
| **capture-level ablation** (P18-B) | **main-slide** | real: high-level→named attention, qdq→dequant; loop-preserving torch.export-blocked (honest frontier) | only 4 workloads re-captured at extra levels | yes — capture fidelity is a methodology axis |
| **batched_matmul split** (S3) | **backup/QA** | separates MLP bmm (groot) from attention | new; small corpus footprint | partial — prevents miscounting |
| **sharding per-top-op** | **backup** | per-op M/N/K shard bytes for hot ops | depends on rdt's non-generalizing giant op | partial |
| **sharding aggregate / inter-op parallelism** | **backup** | structural shardability + low inter-op parallelism | inter-op parallelism is single-step-capture-bound | weak |
| **shape-class MAC share** | **backup** | context | too coarse; M-source caveat | no (superseded by Pareto) |
| **boundary placement heatmap** | **backup** | full enumeration | too many abstractions; descriptive not decision | no (use necessity matrix) |
| **evidence-type by workload/phase** | **QA-only** | provenance traceability | not a result | no |

## Summary

**Strong, present now (source-verified):** capture-fidelity matrix + erasure evidence; work_coverage /
visible_linear_fraction (post-fix); primitive-set frontier (with the linear-only + capture-M caveats);
operator concentration (with the rdt-depth caveat); abstraction necessity (categorical); residency vs
K/dtype; requirements envelope; capture-level ablation. **The methodology contribution** — "what a compiler
can recover from a flat capture, and which DSE axes are blocked-by-capture vs blocked-by-proof" — is the
headline and is fully source-grounded.

**True but needs a caveat on the slide:** all K/residency/loop claims (configured/assumed K); region roles in
single-step captures (low-confidence). Magnitudes everywhere are structural-only (only tiny_llama is a real
checkpoint, and even it is truncated to 2 random layers).

**Resolved with evidence (P20-S3, see capture_shape_sensitivity.md):** rdt "one giant op" — confirmed a
**depth-2 artifact**: at depth=6 the top-op share drops 0.871→0.292 (cross-attn-to-image is the dominant op
*class* ~88%, distributed, not one op). "GEMV-like is capture-M" — tiny_llama at **true decode M=1** is also
`gemv_like` (= the committed M=4), so the GEMV finding genuinely holds in the decode regime (it is not a
small-M artifact; only a large *prefill* M would change the class).

**Fixed this pass (were wrong):** xr0 attention under-count (SDPA-fused, now recovered); groot attention
over-count (MLP bmm now batched_matmul); xr0 K drift (10→5). pi05 "17 shapes" confirmed CORRECT.

**Needs source/capture fixes (not analysis) — ALL RESOLVED in P21 (see the P21 UPDATE table above):**
KV-state sizing (loop-preserving `scf.for` capture now carries the KV iter_arg); native low-bit datapath
(bitvla `BITVLA_NATIVE_QUANT` packed-int2 recapture); region-role recovery (the `scf.for` body is the
repeated head). The remaining true-decode (M=1) re-capture of the other AR workloads is a refinement (P20-S3
already showed M=1 confirms the GEMV regime), not a blocker.

**Next tools (see next_tools.md):** ✅ BUILT in P20 — Tool A mapspace_seed_extractor → timeloop_problem_
shapes.yaml + dataflow_candidate_table.csv (P20-S1); Tool E quant_metadata → quant_metadata_visibility.csv
(P20-S2); Tool B operand_locality → operand_locality_table.csv + capacity_requirement_table.csv (P20-S2).
C/D/F remain extensions of existing P17/P18 tools.

**Next captures (highest leverage) — DONE in P21:** (1) ✅ loop/decode-preserving capture (smolVLA/openVLA/
pi0.5 via `torch.while_loop → scf.for`; unblocked K, KV-state, loop-carried state, region roles — the single
biggest unlock); (2) true-decode (M=1) — P20-S3 already confirmed the GEMV regime holds at M=1; (3) ✅
native-low-bit capture of bitvla (packed-int2 ternary + absmean scale). The largest blocks of
"blocked-by-capture" are now recoverable facts. Remaining frontier is purely runtime/physical (per-K-step
wall-clock, area/energy from a design YAML) — orthogonal to the static contract.
