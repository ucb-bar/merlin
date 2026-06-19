# P17 final report — what is robust, what is blocked (all)

> Per headline finding: a verdict from the adversarial audit (robust / winner-robust-but-magnitude-fragile / blocked-by-capture / descriptive), then the captures that would most improve the study. Structural only; timing rows are requirements, not measured performance.

## Finding verdicts

| conclusion | verdict | slide |
|---|---|---|
| primitive-set frontier: a 2-primitive set covers the corpus where one fails | claim robust (a 2-set suffices); specific pair is threshold/LOO-sensitive | main |
| dense-GEMM is corpus-narrow (skinny/GEMV dominates the MAC mass) | winner robust, magnitude fragile | main |
| resident_weight_object is necessary (K-loop weight residency is a search axis) | blocked-by-capture (needs a loop-preserving capture to confirm K) | backup |
| skinny_gemm_or_gemv_engine is necessary/useful across the corpus | robust (name now honest: skinny OR gemv) | main |
| low-bit abstractions are blocked (cannot be evaluated) | blocked-by-capture | backup |
| attention/KV abstractions are blocked (attention lowered) | blocked-by-capture | backup |
| bounded_loop_command / loop_carried_state_handle are useful | blocked-by-capture (loop erased; useful, not provable) | backup |
| capture fidelity is the limiting factor (flat captures erase loop/KV/low-bit) | robust | main |
| HW/SW boundary placement is itself a DSE search axis | descriptive (enumeration, not a single decision) | backup |

## Most valuable next captures (to unblock the blocked findings)

1. **Loop-preserving capture** — turns configured/reference K into IR-recovered loop structure; unblocks `resident_weight_object` / `bounded_loop_command` and grounds the timing-requirement envelope.
2. **Loop/KV-preserving capture** — attention bmm / softmax / norm are already recovered (see `omitted_operator_accounting.md`); what remains is the KV *state* across the decode loop and the K-loop trip count.
3. **Low-bit recapture** (packed weights + scales + per-format accuracy) — unblocks the low-bit abstractions (blocked today by the f32 fake-quant capture).
4. **Measured dispatch counts** beyond small_llama — turns the command-rate envelope from proxy-only into a real requirement.

## Headline for the talk

Compiler-derived workload contracts tell a DSE tool which search axes to include before any hardware exists — primitive-SET coverage, loop/rate residency, operator concentration, HW/SW boundary placement — **and** reveal when the capture itself is too flat to decide (low-bit, KV, and the K-loop are erased). Capture fidelity is part of the methodology.

