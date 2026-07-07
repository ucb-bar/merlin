# Why our whole-model runtime is slower than the experts — deep investigation

**Question:** on openvla/rdt2 we're ~1.6× slower whole-model than XNNPACK; *why does our runtime suck?*
This records the investigation (4 subagents + board measurement), honestly, including a real
attribution caveat.

## 0. First, what the four-way actually measures (important)
The K1 four-way is a **kernel swap on ONE runtime**, not "our runtime vs theirs." The XNNPACK/OpenBLAS
arms are **our compiled whole model with only `linalg.matmul` rerouted** to an expert ukernel via a
`func.call` shim (`runtime/backends/xnnpack_board/xnn_gemm_rvv_shim.c`); attention, softmax, norm,
activations, layout all lower through **our** pipeline in every arm. So the experts' *own* runtime is
never tested — and our runtime's non-matmul cost is what's exposed.

## 1. Most of the gap is non-matmul — but our matmul kernel IS slower too (now MEASURED on both arms)
`scripts/k1_dispatch_breakdown.py` + the shim bracket the matmul loop with `rdtime`
(`-DMERLIN_DISPATCH_TIMING`) and subtract it from the wall. The attribution caveat below is now
**closed**: we added an `ours` kernel-backend (`runtime/backends/ours_board/`) that routes the matmul to
**our own v3 MR=4 ukernel through the IDENTICAL rdtime bracket**, so both arms self-time their matmul.

**MEASURED, apples-to-apples** (same baseline non-matmul lowering, only the routable matmul swapped,
both timed — `xnnpack_kernels` vs `ours_kernels` in `dispatch_breakdown_measured.json`):
- **openvla (n=3):** ours-v3 matmul bucket **145.7 ms** vs XNNPACK-7x4v matmul **18.4 ms** — **our matmul
  is 7.9× SLOWER** (26 calls each, cos≥0.99999). Walls: ours_kernels 692 ms, xnnpack_kernels 626 ms;
  dispatch buckets 547 vs 608 ms (comparable). So on openvla's real matmul shapes the v3 kernel is
  genuinely worse — but matmul is still only **~21% of the ours wall**; dispatch dominates.

⚠️ **What this corrects.** The earlier note here attributed the ours matmul bucket *equal* to XNNPACK by
decode-identity ("52.9 ms, 8%, gap is 100% dispatch"). That was **wrong on two counts**: (a) it used
XNNPACK's *weak* 1x4v kernel (the fair 7x4v matmul is 18.4 ms, not 52.9); (b) our matmul is **not** equal —
measured 7.9× slower, matching the isolated-cube result that 7x4v beats our v3 (the `register_block: null`
blind spot in `KERNEL_JOURNEY.md`). **Corrected decomposition of the openvla gap** (fair `ours_wholemodel_vf`
1095 ms vs XNNPACK 627 ms ≈ 468 ms): ~**27% is our slower matmul** (+127 ms) and ~**73% is non-matmul**
(the `accumulator_resident_wholemodel_vf` schedule's non-matmul path is ~340 ms slower than baseline —
note `ours_kernels` with *baseline* non-matmul is only 692 ms, far below `ours_wholemodel_vf`'s 1095 ms).
**Robust claim (now measured, not assumed): the majority (~70%+) of the whole-model gap is non-matmul, but
our matmul kernel is also a real, measured contributor — it is slower, not equal.**

- **rdt2:** see `dispatch_breakdown_measured.json` (measured split, same method).

## 2. WHY the non-matmul path is slow (structural, with the worst offenders measured)
1. **Scalar transcendental activations — the dominant measured hotspot.** GELU/sigmoid/SiLU lower via
   `convert-math-to-libm` → **scalar `erff`/`expf` call loops** → **11–18× slower than XNNPACK's
   vectorized polynomial** (`cross_framework_ops_k1.md`: GELU 10–17×, sigmoid 9–11×).
   **A fix EXISTS but is OFF:** `vectorized_transcendental_activation` (`impr_features.py`) rewrites them
   to inline minimax polynomials → vfmacc chains (~3.7× faster, cos>0.999) — but it is **not applied by
   default in whole-model runs.** So part of the slowness is self-inflicted (a built, certified feature
   left disabled).
2. **No cross-op fusion → DRAM round-trips.** The model is one compiled binary, but ~**4,484
   `tensor.empty()` → `memref.alloc`** and ~**1,956 call instructions** (smolvla): **every op
   materializes its output to DRAM and the next op reads it back.** Matmul output is always a hard
   boundary. The experts' arm pays the same glue, but our schedule adds materialization the libraries'
   fused kernels wouldn't.
3. **Softmax stays scalar** (`exp` via libm) — *intentionally* not vectorized (vectorizing it corrupted
   numerics, cos→0.541). Deferred.
4. **Norm / transpose / reshape / gather** rely on clang autovectorization (unreliable) or are scalar.
5. **Per-op setup:** vsetvl reconfig + memref-descriptor unpack + loop setup on every op.

## 3. The honest punchline
We lose on **two** fronts, now both measured: (1) our **matmul kernel is slower** than the fair expert
ukernel (openvla: 7.9× slower matmul bucket, ~27% of the gap) — the `register_block: null` blind spot in
KERNEL_JOURNEY means we never mined/compared MR, so the v3 MR=4 kernel loses to XNNPACK's 7x4v; and (2)
the **larger share (~70%+) is the non-matmul / glue path**: scalar activations (with an unapplied fix), no
cross-op fusion, per-op materialization, plus the `accumulator_resident_wholemodel_vf` schedule's own
non-matmul overhead (its baseline-non-matmul sibling `ours_kernels` is much faster — 692 ms vs 1095 ms on
openvla). The good news is unchanged: a **kernel library structurally can't fuse across ops** — so the
non-matmul majority is exactly where a *compiler* can beat them, if the runtime is made competitive (next
section). But the clean isolated-GEMM ceiling is now honestly **against** us, not equal.

## 4. Where a COMPILER can beat XNNPACK/OpenBLAS (opportunities a frozen library can't do)
Ranked, with maturity (from the opportunities probe; estimates are rough, not measured):
1. **Attention fusion (QK·softmax·V in one tiled pass)** — removes 2–3 DRAM round-trips/attention block.
   Helps all VLAs (openvla/rdt2 are 92–97% non-matmul). *Not built.* Est. the biggest single lever.
2. **Matmul + bias + activation + residual epilogue fusion** — one output write instead of ~4 reads.
   All models. *Partially built* (elementwise fusion exists; cross-dispatch merge doesn't).
3. **Softmax reduce+elementwise fusion** — fuse the reduce into the exp/normalize. *Partially built.*
4. **Norm + matmul fusion** — avoid materializing the normalized activation. *Partially built.*
5. **Turn ON `vectorized_transcendental_activation` whole-model** — the already-built activation fix
   (~3.7× on the 11–18× hotspot). *Built, disabled.* ← cheapest real win to verify first.
6. **Native low-bit datapaths the libraries can't run at all:**
   - **W8A8 int8 fold-dequant-into-GEMM** whole-model (`passes_quant_int.py` exists, isolated today).
   - **bitVLA W1.58 ternary native** (XNNPACK/OpenBLAS have *no* ternary GEMM — they can only run it
     dequantized to fp32). Real *categorical* advantage, but needs integer softmax → speculative.
- Already built / no further win: small-N attention N-tail clamp (`accumulator_resident_ntail`),
  basic elementwise-chain fusion.

**Combined realistic estimate** (attention + epilogue + softmax fusion + activation-on): the opportunities
probe put openvla/rdt2 at roughly **60% → ~73%** of XNNPACK — i.e. it would *narrow* the gap, not
obviously surpass, on fp32. The clean *surpass* story is **native low-bit** (bitVLA ternary), which the
libraries categorically cannot match — but that's the most speculative and least built.

## 5. Fair whole-model standings (cos-gated, fair/strongest expert kernels: XNNPACK 7x4v, OpenBLAS 16x8_zvl256b)
With the *fair* kernels (not the weak 1x4v/8x8 we first benchmarked), the whole-model four-way is:
- **bitVLA:** ours-v3 **148 ms WINS** vs XNNPACK 167 ms (1.13×), OpenBLAS 181 ms (1.22×). Headline holds.
- **openvla:** ours-vf 1095 ms **LOSES** vs XNNPACK 627 ms (0.57×), OpenBLAS 681 ms. Dispatch-bound.
- **rdt2:** ours-vf 30.2 s **LOSES** vs XNNPACK 18.6 s (0.61×), OpenBLAS 20.1 s. Dispatch-bound.
So the honest claim is **"we beat both libraries whole-model on bitVLA"** — not on the big dispatch-bound
VLAs, where our slow non-matmul path (and a slower matmul, §1) dominates.

## 6. TODO / open items
- ✅ DONE: independently measured the *ours* matmul bucket (`ours_board` backend + rdtime bracket) — §1.
  Result overturned the equal-matmul assumption: our matmul is 7.9× slower on openvla.
- Verify the activation fix actually helps whole-model (turn on `vectorized_transcendental_activation`,
  re-measure openvla/rdt2). Cheapest test.
- Fix the matmul blind spot: mine `register_block`/MR so the compiler picks MR=7-class blocking instead of
  the hand-fixed MR=4 v3 (would shrink the 7.9× matmul gap).
- Prototype attention fusion (highest-ROI compiler-only lever).
- These are *opportunities*, not results — present them as "where a compiler can win," not as wins.
