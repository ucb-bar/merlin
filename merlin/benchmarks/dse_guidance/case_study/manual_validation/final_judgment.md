# Final judgment (P19 Phase 6)

Source-grounded verdict on every current result/plot, after the 11-workload forensic audit (S1) and the
correctness fixes (S3). Classification ∈ {main-slide, backup, QA-only, needs-fix, invalid}. All structural;
no perf claims. "Evidence" = source/MLIR audit status.

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

**True but needs a caveat on the slide:** rdt's giant op (depth-2, non-generalizing); "GEMV-like"
(capture-M, prefill-vs-decode); all K/residency/loop claims (configured/assumed K); region roles in
single-step captures (low-confidence). Magnitudes everywhere are structural-only (only tiny_llama is a real
checkpoint, and even it is truncated to 2 random layers).

**Fixed this pass (were wrong):** xr0 attention under-count (SDPA-fused, now recovered); groot attention
over-count (MLP bmm now batched_matmul); xr0 K drift (10→5). pi05 "17 shapes" confirmed CORRECT.

**Needs source/capture fixes (not analysis):** KV-state sizing (needs loop/decode-preserving capture —
torch.export-blocked); native low-bit datapath (qdq capture is torchao-int8, not the model's native ternary
— a capture gap); region-role recovery (needs a loop/region-preserving capture). Several "decode" workloads
are captured as prefill (use_cache=False) — re-capture as true single-token decode (M=1) to get the real
GEMV regime instead of the M=4/8 prefill artifact.

**Next tools (see next_tools.md):** Tool A mapspace_seed_extractor (highest new value), Tool E
quant_metadata_capture (unblocks low-bit, qdq capture exists), Tool B operand_locality_analyzer; C/D/F are
extensions of existing P17/P18 tools.

**Next captures (highest leverage):** (1) loop/decode-preserving capture (unblocks K, KV-state, residency
proof, region roles — the single biggest unlock); (2) true-decode (M=1) re-capture of the LLMs/VLAs; (3)
native-low-bit capture of bitvla (packed ternary + scales). These three convert the largest blocks of
"blocked-by-capture" into recoverable facts.
