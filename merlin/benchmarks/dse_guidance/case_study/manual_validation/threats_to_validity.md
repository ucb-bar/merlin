# Threats to validity (self-critique, conference-grade)

Written adversarially — as the reviewer we expect, not the authors we are. Each threat is classified:
**[INTENTIONAL]** = a deliberate scope boundary (the thesis), **[ADDRESSED]** = fixed in this round (P24),
**[SCOPED]** = a real limit with the exact input that would close it (we don't fabricate around it).

## What we claim — and, precisely, what we do not
Merlin recovers a **structural workload contract** for accelerator DSE from real model captures: the op graph
and shapes, the multi-rate loop structure (K, loop-carried state, KV cache, the repeated region), region roles,
operand locality / residency, the dtype/numerical contract, and a HW/SW-boundary search space — each with an
**evidence tier**. It is **not** a DSE optimizer and makes **no performance, cycle, area, or energy claim for any
built or hypothetical chip.** Reviewers should hold us to exactly that line; anything beyond it we flag below.

## Threats

### T1 — Magnitudes vs structure  **[ADDRESSED]**
*Risk:* the loop captures use reduced/random-init configs (openVLA 2-layer → ~0 GMAC; pi0.5 full 18-layer), so
mixing their absolute MACs/bytes across the corpus is apples-to-oranges and not deployment-real.
*Resolution:* we **decouple** the two. STRUCTURE (op shapes, ratios, role split, loop facts) is capture-sourced
and is the contribution. ABSOLUTE MAGNITUDES (params, bytes, MACs, residency, KV) are reported from a
**deployment config-composition** (`real_config_magnitudes.csv`) — exact for layer-identical transformer stacks
(`embed + Σ per-layer × real n_layers`), anchored by exact matches (openVLA = 6.74 B = Llama-2-7B; tiny_llama =
1.10 B = TinyLlama-1.1B). Every magnitude figure is tagged **deployment-composition** vs **captured-config**.

### T2 — Random-init weights  **[INTENTIONAL / SCOPED]**
*Risk:* the captures are random-init (only tiny_llama is a real checkpoint), so weight VALUES are meaningless.
*Why it's OK:* MACs/bytes/shapes depend on shapes, not values — the structural contract is value-independent
(stated explicitly). Anything that WOULD depend on values (accuracy) is taken only from the measured W8A8 gate.
*Scoped:* full-size real-checkpoint captures would let value-dependent studies run — required input noted.

### T3 — Verification is internal-consistency, not ground truth  **[ADDRESSED-as-far-as-possible / SCOPED]**
*Risk:* the 628 verifier checks re-derive from the same captures — they prove the package is self-consistent,
not that it predicts reality.
*Resolution:* (a) the loop wrappers are each **numerically exact** vs the eager unrolled loop (cos ~1.0 /
bit-exact) — so the recovered structure provably equals the model's real computation; (b) two exact external
anchors (openVLA→Llama-2-7B, tiny_llama→1.1B params) cross-check the magnitude composition; (c) measured legs
(below) are surfaced as independent sanity anchors. Ground-truth perf is out of scope by design (T6).

### T4 — Evidence-tier honesty  **[ADDRESSED]**
*Risk:* presenting assumed values as if measured. Of the recovered facts, ~A 24% (IR/measured), B 40%
(recovered + recompute-checked), C 34% (config/assumed), D <2% (unavailable).
*Resolution:* every fact and every figure carries its tier (A/B/C). Config-sourced values (K when not IR,
control rate) are labelled C; we never round a C up to A.

### T5 — K is the captured decode length  **[ADDRESSED for structure / SCOPED for deployment]**
*Risk:* K is recovered exactly from the `scf.for` trip count (Tier A), but the captured K (e.g. 7 decode
tokens) is a capture choice, not the deployment decode length.
*Resolution:* K-as-loop-structure is IR-exact; K-as-deployment-length is a runtime parameter — residency/reload
plots show the **dependence on K** (a curve), not a single K, so the conclusion holds for any K.

### T6 — No performance / the unknown-hardware roofline  **[INTENTIONAL]**
*Risk:* "where's the speedup / latency?" There is none — and that is the thesis: the HW is unknown, so the tool
emits hardware-INDEPENDENT requirements, never performance. Our roofline is done the only honest way: the
x-axis (**arithmetic intensity** = MACs/bytes) is a property of the workload (Tier A, no HW); residency's effect
on AI is HW-free; the compute-vs-memory regime is a **ridge-point partition over possible machine balances**, not
a chosen chip. Any absolute "speed-of-light" latency is shown ONLY as a parametric sensitivity over a stated
(peak, bandwidth) range, every value `design_assumption`, never a prediction for a built chip.
*Sanity anchors (not products):* FireSim cycles (6 models incl the xr0 silicon datum), W8A8 accuracy (5 models),
host-dispatch counts (3) — each kept with its caveat (the matmul-only cost model is crude: xr0 is a 4.7× outlier).

### T7 — Low-bit datapath only native for bitvla  **[ADDRESSED with honest tiering / SCOPED]**
*Risk:* only bitvla has a native packed-low-bit capture; the rest are dequantized-f32.
*Resolution:* a per-workload tier — **native** (bitvla: packed int2 + scale + `quant_ext.unpack_int2`),
**qdq-int8** (storage + per-channel scale visible), **dequant-only** (honest). Int8 candidates are ratified by
the measured accuracy gate; fp8/int4 stay `unavailable` (never assumed). Native packed fp8/int4 for all models
needs model-specific quant exports — scoped, not faked.

### T8 — Attention is recovered but not loop-level structured  **[SCOPED]**
*Risk:* attention MACs are recovered (re-parsed from generics), but inside the loop the KV/attention is not
structured at the loop level (SDPA-fused / lowered).
*Scoped:* a Level-2 loop-preserving, attention-not-lowered capture would expose it; the KV STATE across the loop
is already recovered as an `scf.for` iter_arg.

### T9 — Corpus is 10 models, one synthetic  **[ADDRESSED]**
small_llama (synthetic toy) is excluded from the analyzed corpus (its functional-weight loop wrapper lowered
GEMMs to `linalg.generic` → 0 `linalg.matmul`); 10 real architectures remain. bitvla is in the structural/
low-bit corpus but omitted from deployment-magnitude composition (its real config is not in the repo — not
guessed).

### T10 — Plot legibility / honesty  **[ADDRESSED]**
*Risk (a hostile reviewer's figure check):* 6–7 pt fonts, stacked-bars-on-log, toy-vs-deployment not marked,
K-provenance ambiguous, dense heatmaps unreadable, headline table overflow.
*Resolution:* conference-grade pass — fonts ≥ 8 pt, grouped (not stacked) bars on log, every figure carries an
evidence-tier badge + a scale-source tag (deployment vs captured) and marks reduced-config workloads, K dots
labelled "IR-recovered K", dense heatmaps trimmed to top-N with readable labels, headline table widened.

## Bottom line for a reviewer
The **structural contract recovery + the loop-preserving capability** are the result and they are
final-quality, verified, and numerically faithful. **Magnitudes** are deployment-real by config-composition
(exact for layer-identical stacks), clearly separated from the reduced-config captures. **Performance is
intentionally absent** — the roofline is hardware-independent (arithmetic intensity + ridge-point regime), and
nothing is claimed for any chip. The honest residuals (T2, T5, T7, T8) each carry the exact input that closes
them.
