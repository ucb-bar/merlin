# Kernel-Mining Journey — driving example, slide flow, speaker notes

**Purpose.** The advisor couldn't picture the kernel-mining / compiler-improvement flow because the
deck threw full diagrams (old slides ~16, 18) cold. This section instead **follows one real kernel
end-to-end** so the audience sees *what we extract → what we compare → why → how we get it from the
compiler → how we improve → how we certify → why it wins*. (DSE is not presented.)

**The one driving example: a plain f32 GEMM, 64×64×64.** Every step below is a *committed artifact*
(paths given). The mining is real (we lift the experts' kernels into a CCA and route divergences to
typed compiler actions); the **7.9× isolated** number is our fused-vfmacc fork beating *our own*
`vfmul+vfadd` baseline (cos=1.0) — NOT beating the experts. The honest competitive result is whole-model:
on **bitVLA we beat both libraries' best kernels, 148 ms vs XNNPACK 167 ms (1.13×) / OpenBLAS 180 ms
(1.22×)**, fair + cos-gated — but the win is whole-model *scheduling*, our GEMM kernel is actually slower
(see the Supervisor-answer section for the full measured story; on openvla/rdt2 we lose).

Figures live in `output/presentation/` (gitignored): `fig20`–`fig24` (new) + reuse `fig7` (CCA cards),
`fig9` (gate ladder), `fig6` (beam tree), `fig1` (four-way win).

---

## Slide flow (figure · what to say · artifact)

### J0 — "The journey at a glance"  ·  `fig22_journey_strip`
> "We'll follow ONE kernel all the way through: kernel → decode → CCA → compare → action → fork →
> certify → win. Every box is a real artifact, not a cartoon."
Orientation only; sets the spine so nothing later feels cold.

### J1 — "Start with one simple kernel"  ·  `fig20_kernel_input`
A 64×64×64 `linalg.matmul` (f32). One op, one shape.
*Artifact:* `output/rvv_workloads/matmul_f32_64x64x64/model.mlir`.
> "This is the actual input we compile. We follow *this* kernel for the rest of the section."

### J2 — "What we extract & compare"  ·  `fig21_asm_diff`
Our compiler's baseline emission vs the fused form, same kernel.
- **Baseline (ours today):** 4× `vfmul.vv` then 4× `vfadd.vv`, then `vse32.v` — multiply and add are
  *separate*, and the accumulator *spills* to memory. `contraction_form = mul_add`.
- **Fused form (expert's, and our fork):** `vfmacc.vf` — one fused MAC, accumulator stays in vector
  registers, no per-K store. `contraction_form = fused_fma`.
*Artifacts:* baseline `runs/rvv_bench/hand_v0_matmul_f32_64x64x64/generated/objdump.txt` (lines ~121–133:
`vfmul.vv`×4, `vfadd.vv`×4, `vse32.v`); the fused form is what the certified fork emits (J5).
> "We disassemble both down to RVV asm — the one substrate every framework lowers to — and read off
> the *structural decision*: fuse the MAC, keep the accumulator resident."

### J3 — "Lift both to a Common Compute Abstraction"  ·  `fig7_cca_example` (cards)
Decode → lift each asm into the CCA (compute facet + vector facet + provenance). The structural decode
*agrees* on most fields and *diverges* on the ones that matter.
*Artifacts:* `artifacts/kernel-mining/rvv/mining_rvv_v3_20260619T092631/{our_cca,expert_cca}.yaml`
- ours: `contraction_form: mul_add`, `widening: false`, `lmul: 2.0`, `vl_strategy: vsetivli_fixed`
- expert: `contraction_form: fused_fma`, `widening: true`, `lmul: 4.0`, `vl_strategy: vsetvl_loop`,
  `accumulator_resident: true`
> "This is *what we extract*: not the kernel bytes — the **decisions**. The CCA is target-agnostic;
> RVV fills the vector facet, a systolic target would fill the spatial facet."

### J4 — "The divergence routes to a typed action — and the catalog has four levers"
First `fig7` right (the single PASS), then **`fig24_action_classes`** (the full taxonomy).
*Artifacts:* `artifacts/kernel-mining/rvv/mining_rvv_v3_*/{divergences,actions}.yaml`; `kernels/action_catalog.py`.
The catalog picks the **cheapest lever that works** and *escalates*:

| class | mined divergence | target_seam | status |
|---|---|---|---|
| **FLAG** | `compute.contraction_form` | `cflag: -ffp-contract=fast / -ffast-math` | tried first — clang did **not** fuse (`vfmacc=0`) → escalate |
| **KNOB** | `vector.lmul` | `schedule:vector_sizes` (widen N → LMUL↑) | forkable today · ~1.05× |
| **HEURISTIC** | `compute.nr_is_vsetvlmax` / `mr_adapts_to_m` | `schedule:NR=vsetvlmax` · `MR=min(MR,M)` | forkable today · small-N & M=1 vectorize |
| **PASS** | `compute.contraction_form` | `impr_features:fused_vfmacc_contraction` | **the winner** — new lowering pattern · 7.9× |

> "Same goal — fuse the MAC. We tried the *cheap* levers first: a compiler FLAG (`-ffp-contract`),
> schedule KNOBs. They measured `vfmacc=0` — clang wouldn't fuse it — so the catalog **escalated** to a
> PASS (a new lowering pattern). FLAG/KNOB/HEURISTIC are forkable *today* via `schedule.mlir`+cflags;
> a PASS needs compiler code (a default-off feature)."
- **int8 / Gemmini bridge (one line):** the *same* mining run also flags `compute.widening: true vs
  false` → a KNOB routing the i8 matmul through the `vwmacc` (i8×i8→i32) datapath. "The same CCA that
  learns f32 fusion also captures the int8 widening MAC an int8 systolic target like **Gemmini** lives
  on." (Gemmini section carries its own int8 example, `A2_single_tile_matmul`, RTL-certified 316 cyc.)

### J5 — "Fork the compiler (default-off) and certify it"  ·  `fig9_beam_gates`
The fork enables one default-off `ImprFeature`; the baseline stays byte-identical. Every candidate
clears the **K-ladder**: K0 load · K1 non-perturbation · K2 build · K3 spike cos-gate · K4 instruction
histogram (`vfmacc` must appear, not a scalar fallback) · K5 K1 cycles · K6 Δ-vs-baseline (fail-closed).
*Artifacts:* `merlin/python/merlin/llvmlower/impr_features.py` (`fused_vfmacc_tiled`,
`accumulator_resident_microkernel_v3`); `runs/rvv_bench/impr_rvv_v5_*/results.yaml` (cos = 1.0,
`vfmacc.vf` present).
> "Correctness-gated before any speed claim — cos = 1.0, and we *prove* the asm now carries `vfmacc`."

### J6 — "Beam search composes features"  ·  `fig6_beam_tree`
The beam explores single features at depth 1, keeps the top-k survivors, composes at depth 2; prunes
no-gain branches, kills regressions. Winner: `v3 + lmul_widen`.
*Artifact:* `artifacts/kernel-mining/rvv/beam_rvv_v2_*/ranking_bitvla.yaml`.

### J7 — "Measured win + why it beats XNNPACK"  ·  `fig23_why_we_win` (+ `fig1`)
The "kernel ≠ system" payoff and the supervisor answer (next section). The fair, measured result: ours
beats both libraries' best kernels **whole-model on bitVLA (148 vs 167/180 ms)** via whole-model
scheduling, while our GEMM *kernel* is measured slower (3–14×) — kernel ≠ system, exactly.
*Artifacts:* fair four-way `output/rvv_bench/k1_e2e_fair_{bitvla,openvla,rdt2}.json`; measured matmul-vs-
dispatch split `output/rvv_bench/dispatch_breakdown_measured.json` (ours_board backend, both arms timed).

---

## What each action class actually CHANGES — one real example per class

The catalog's four classes differ by *what they edit*. FLAG/KNOB/HEURISTIC all ride the existing
`schedule.mlir` + `cflags` seams (forkable today, no new compiler code); PASS adds a new lowering
pattern. All snippets verified from `merlin/python/merlin/llvmlower/impr_features.py`,
`rvvgen/{registry,apply,report}.py`.

**FLAG — a compiler flag** (`cflags` seam; frozen `RVV_CFLAGS` untouched). The honest example is the
one *tried and demoted*:
```python
cflags = ["-march=rv64gcv", "-mabi=lp64d", "-O2", "-ffp-contract=fast"]   # ← the FLAG
# build_app(cflags_override = pkg.cflags + _CFLAGS_COMMON)
# outcome (rvvgen/report.py): v1–v4 tried -ffp-contract=fast / -ffast-math → vfmacc=0
#   (clang won't fuse the K=1-tiled contraction) → escalated to a PASS
```

**KNOB — a schedule parameter** (`lmul_widen_n`, real registration):
```python
register(ImprFeature(name="lmul_widen_n", action_class="KNOB",
    edit_schedule=lambda t: t.replace("tile_sizes [4, 8, 1]",  "tile_sizes [4, 16, 1]")
                             .replace("vector_sizes [4, 8, 1]", "vector_sizes [4, 16, 1]")))
# change: [4, 8, 1] → [4, 16, 1]  (widen N 8→16 → LMUL m2→m4)  ·  ~1.05× on K1
```

**HEURISTIC — a selection rule** (catalog routes `compute.nr_is_vsetvlmax` / `compute.mr_adapts_to_m`
→ HEURISTIC, seam `schedule:NR=min(NR,N)` / `MR=min(MR,M)`; realized via the schedule seam, forkable today):
```python
edit_schedule=lambda _t: _accumulator_resident_pre_schedule(4, 16, 16, MR_mm=1)   # MR = min(MR, M)
# rule: clamp the register block to the ACTUAL dim → small-N attention (N=8) & M=1 token-decode
#   matmuls vectorize FULL (no mask) → vfmacc, instead of the LLVM-23 masked-write PipelineError
#   → silent SCALAR fallback.
```
**This is the most interesting class — measured ~10× swing** (`output/kernels/ceiling/RESULTS.md`):
the kernel-only `vfmacc` PASS, applied *whole-model* WITHOUT the clamp, **regresses to 0.75× / 0.68×**
(bitvla / openvla — silent scalar fallback, slower than baseline). Adding the N-tail/M-tail heuristic
→ `accum_ntail` **7.75×** (bitvla) and `accum_wholemodel` **8.10× / 4.97×** — the **openvla/rdt2
winner**. The heuristic is what makes the kernel win *survive* whole-model.
```text
vfmacc PASS, whole-model, no clamp :  0.75×  (regression — small-N/M=1 fall back to scalar)
+ N-tail/M-tail HEURISTIC clamp     :  7.75× → 8.10× / 4.97×  (vfmacc, attention-safe; wins openvla)
```
*(Note: no impr_feature is tagged HEURISTIC — the label lives in `action_catalog`; the clamp is realized
through `schedule.mlir`, which is why it's "forkable today.")*

**PASS — a new lowering pattern** (needs compiler code; `fused_vfmacc_tiled` + `_VFMACC_TILED_SCHEDULE`):
```python
register(ImprFeature(name="fused_vfmacc_tiled", action_class="PASS",
    edit_schedule=lambda _t: _VFMACC_TILED_SCHEDULE, schedule_replace=True))
```
```mlir
// _VFMACC_TILED_SCHEDULE (transform dialect — what the PASS actually does)
%t, %l:3 = transform.structured.tile_using_for %mm tile_sizes [4, 16, 16]
transform.structured.vectorize %t vector_sizes [4, 16, 16]
transform.apply_patterns to %f { transform.apply_patterns.vector.reduction_to_contract }            // rebuild vector.contract
transform.apply_patterns to %f { transform.apply_patterns.vector.lower_contraction "outerproduct" } // → vector.fma
transform.apply_patterns to %f { transform.apply_patterns.vector.lower_outerproduct }                // → vfmacc
// change: vfmul+vfadd → fused vfmacc  ·  7.9× isolated on K1 (cos = 1.0)
```

**⚠️ Did each make a difference, and is each on the 64³ journey example? (be precise on the slide):**
- **PASS** — yes & yes: **7.9×** on the 64³ GEMM. The headline win.
- **KNOB** — yes (small) & yes: **~1.05×** measured on the same 64³ GEMM.
- **FLAG** — tied to the same goal, but it **FAILED** (`vfmacc=0`) — shown *because* it failed (the
  escalation point), NOT a win.
- **HEURISTIC** — yes, **big & measured, but on real models not the 64³ cube**: the N-tail/M-tail clamp
  turns the whole-model `vfmacc` from **0.75× (regression, scalar fallback)** into **7.75× / 8.10×**
  (and **4.97×**, the openvla/rdt2 winner). It's a no-op at M=64; it earns its keep on small-N attention
  (N=8) and M=1 token-decode. Narrative: *the 64³ GEMM teaches FLAG/KNOB/PASS; the HEURISTIC is what
  makes the kernel win survive whole-model on real shapes.*

## The artifacts, step by step (real committed snippets — the evolution)

These are the actual file contents at each step, so you can *see* what changes. Put each on its slide.

**① INPUT — `output/rvv_workloads/matmul_f32_64x64x64/model.mlir`**
```mlir
func.func @forward(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<64x64xf32>
  %1 = linalg.fill   ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %2 = linalg.matmul ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>) outs(%1) -> tensor<64x64xf32>
  return %2 : tensor<64x64xf32>
}
```

**② BASELINE emit — `runs/rvv_bench/hand_v0_matmul_f32_64x64x64/generated/objdump.txt` (lines 121-133)**
```asm
182: vfmul.vv v26,v26,v8     ; 4x separate multiply ...
192: vfmul.vv v8,v22,v8
196: vfadd.vv v10,v26,v10    ; ...then 4x separate add
1a2: vfadd.vv v8,v8,v16
1a6: vse32.v  v10,(s3)       ; accumulator SPILLS to memory
1aa: vse32.v  v12,(s9)
```
histogram: `vfmul.vv: 4 · vfadd.vv: 4 · vfmacc: 0`

**②′ FORK emit — `runs/rvv_bench/impr_rvv_v5_*/generated/objdump.txt`**
```asm
flw       fa5, 2040(sp)      ; scalar A
vfmacc.vf v8, fa5, v16       ; c += a*b  — ONE fused MAC (accumulate into v8)
vfmacc.vf v8, fa5, v24
vfmacc.vf v8, fs10, v16
```
histogram: `vfmacc.vf: 8065 · vfmul.vv: 0 · vfadd.vv: 0`

**③ CCA — `artifacts/kernel-mining/rvv/mining_rvv_v3_*/{our,expert}_cca.yaml` (side by side)**
```yaml
# ours (baseline)                         # expert (mined from XNNPACK/OpenBLAS)
compute:                                  compute:
  contraction_form: mul_add                 contraction_form: fused_fma
  widening: false                           widening: true
  accumulator_resident: null                accumulator_resident: true
vector:                                   vector:
  lmul: 2.0                                 lmul: 4.0
  vl_strategy: vsetivli_fixed               vl_strategy: vsetvl_loop
provenance: {level: asm, source: ours}    provenance: {level: policy, source: mined_policies}
```

**④ DIVERGENCE — `mining_rvv_v3_*/divergences.yaml`** (only the differing fields)
```yaml
- {axis: compute.contraction_form, expert: fused_fma,   ours: mul_add}
- {axis: compute.widening,         expert: true,        ours: false}     # the int8/Gemmini bridge
- {axis: vector.lmul,              expert: 4.0,         ours: 2.0}
- {axis: vector.vl_strategy,       expert: vsetvl_loop, ours: vsetivli_fixed}
```
**ACTION — `mining_rvv_v3_*/actions.yaml`**
```yaml
- axis: compute.contraction_form
  class: PASS
  target_seam: impr_features:fused_vfmacc_contraction
  expected_effect: "vfmacc replaces vfmul+vfadd; MEASURED 7.9x faster on K1 (64^3 f32, cos=1.0)"
```

**⑤ FEATURE — `merlin/python/merlin/llvmlower/impr_features.py` (the registered, default-off fork)**
```python
register(ImprFeature(
    name="fused_vfmacc_tiled", action_class="PASS",
    description="Tiled-vfmacc: tile [MR=4,NR=16,KC=16] then vectorize_children -> "
                "outerproduct -> vector.fma -> vfmacc. Bounded-code, whole-model-safe.",
    edit_schedule=lambda _t: _VFMACC_TILED_SCHEDULE, schedule_replace=True))
```
**CERTIFY — the K-ladder gate + the proof.** ⚠️ Do NOT screenshot the raw fork `results.yaml`: in
`runs/rvv_bench/impr_rvv_v5_*/results.yaml` the build + structural rungs pass and the histogram proves
the feature fired, but **K3/K6 are `not_run`** (no refs threaded in that record) so its own
**`status: fail`** means *incomplete*, not *wrong*. Show the **histogram** + the cos/speedup from the
gated benchmark run instead:
```yaml
ladder: {K0: pass, K1: pass, K2: pass, K3: not_run, K4: pass, K5: pass, K6: not_run}  # status: fail = INCOMPLETE
instruction_histogram: {vfmacc.vf: 8065, vfmul.vv: 0, vfadd.vv: 0}   # K4 proof: the fusion fired
# cos = 1.0 · 7.9×  ← from k1_gemm64_benchmark.md (the separate cos-GATED measurement run)
```

**⑥ WIN — `output/rvv_bench/k1_e2e_fair_bitvla.json` + `output/rvv_bench/dispatch_breakdown_measured.json`**
```text
isolated 64^3, K1 wall (ours vs OUR baseline): baseline 12,857,888 ns -> vfmacc 1,629,809 ns = 7.889x (cos 1.0)
  ^ this 7.9x is ours-vs-our-own-baseline, NOT vs the experts. On K1 ticks the experts' 7x4v BEATS our v3.
spike instret (DIRECTIONAL ONLY, a mirage — do NOT rank kernels with it): XNN 101,705 · OB 84,483 · v3 53,207
K1-measured matmul bucket (both arms timed): bitvla v3 32.4ms vs XNN-7x4v 10.2ms -> our kernel 3.18x SLOWER
whole-model bitvla, K1 (FAIR, cos-gated): ours-vf 148.3ms  vs  XNN-7x4v 167.3ms (1.13x)  ·  OB-16x8 180.5ms (1.22x)
  ^ ours wins WHOLE-MODEL (the vf schedule's non-matmul path), despite the slower kernel. bitVLA only.
```

## ✅ RESOLVED (no-caveat, fair re-run on the K1) — the bitVLA whole-model headline HOLDS

After the unfair-kernel scare below, the fair re-run is done. **FAIR whole-model bitVLA — real K1,
same-pass, N=3, cos-gated, each expert with its BEST K1 kernel** (`output/rvv_bench/k1_e2e_fair_bitvla.json`):

| bitVLA whole-model | min wall | cos | ours vs it |
|---|---|---|---|
| **ours-v3** | **148.3 ms** | 0.999995 | — |
| XNNPACK **`7x4v`** (MR=7, its best) | 167.3 ms | 0.999993 | **ours 1.13× faster** |
| OpenBLAS **`16x8_zvl256b`** (MR=16, VLEN-256, its best) | 180.5 ms | 0.999993 | **ours 1.22× faster** |

**Ours beats BOTH experts' best kernels whole-model on bitVLA.** The unfair kernels gave 1.28×/1.34×;
the fair kernels give **1.13×/1.22×** — margins shrank but ours still wins, with everything cos-gated.
(XNNPACK improved 191→167 ms with `7x4v`; OpenBLAS 188→180 ms with `16x8`.) Method: swapped the
whole-model shims `runtime/backends/{xnnpack,openblas}_board/*_gemm_rvv_shim.c` to `7x4v` / `16x8_zvl256b`
(.bak files kept), re-ran `scripts/k1_e2e_xnnpack.py`.

**What's still true from the scare (now fully measured):** the *isolated single-GEMM cube* claim is dead
— on 32/64/128³ cubes XNNPACK `7x4v` beats ours-intrinsic (175 vs 224, 10 341 vs 14 256 ticks), AND the
per-model matmul bucket is now measured on both arms (`ours_board` backend): our v3 matmul is **3.18×/7.9×/
13.6× slower** on bitvla/openvla/rdt2. So **we do NOT have a better kernel anywhere.** The bitVLA whole-
model win is real but comes from our **`vf` whole-model schedule's NON-matmul lowering**, not the GEMM:
the pure kernel-swap `ours_kernels` (178 ms) is *slower* than `xnnpack_kernels` (169 ms) on bitVLA; only
the full `vf` schedule (148 ms) wins. Honest framing: *"our whole-model compilation of bitVLA beats
swapping either library's best GEMM into the same model — via non-matmul scheduling, not a better kernel;
on neutral cubes and on openvla/rdt2 the experts win."*

✅ Last no-caveat item CLOSED: the *ours* matmul bucket is now independently instrumented (rdtime bracket
in `ours_board`), so the matmul-vs-dispatch split is measured on both arms — see
`dispatch_breakdown_measured.json` and `RUNTIME_INVESTIGATION.md` §1. It overturned the old equal-matmul
attribution (our kernel is slower, 3–14×).

## ⚠️ (superseded by the RESOLVED block above) the scare — the comparison used XNNPACK's weakest kernel

The "we beat XNNPACK" claims (isolated AND the bitVLA whole-model headline) were measured against
XNNPACK's **`1x4v` (MR=1)** microkernel — its *worst* f32 RVV GEMM kernel. XNNPACK also ships **`7x4v`
(MR=7)**, which its runtime would normally select. Re-running on the **real K1 board** (bit-exact,
rdtime ticks, same session):

| K1 ticks | ours-intrinsic | XNN `1x4v` (we used) | XNN `7x4v` (real) | vs 7x4v |
|---|---|---|---|---|
| 32³ | 224 | 260 | **175** | **XNNPACK 1.28× faster** |
| 64³ | 1406 | 1985 | 1465 | tie (ours 1.04×) |
| 128³ | 14256 | 31933 | **10341** | **XNNPACK 1.38× faster** |

**Consequences:**
- **The isolated "ours beats XNNPACK" claim is dead.** Against `7x4v`, XNNPACK is faster at 32³/128³ and
  ties at 64³. We only "won" by benchmarking its MR=1 kernel. The committed
  `cross_framework_matrix_k1.md` "ours-intrinsic beats BOTH experts at all shapes" is **false** vs `7x4v`.
- **BUT the bitVLA WHOLE-MODEL headline SURVIVES the fair kernel — RE-RUN DONE (this session, cos-gated,
  N=3, real K1):**

  | bitVLA whole-model | min wall | cos |
  |---|---|---|
  | ours-v3 | **149.3 ms** | 0.9999946 |
  | XNNPACK with `7x4v` (fair) | 169.5 ms | 0.9999927 |
  | → **ours 1.135× faster** | | |

  XNNPACK's whole-model dropped 191.3 → 169.5 ms with `7x4v` (the swap worked; `n_xnn=15` matmuls
  routed), so the margin shrank from 1.28× to **1.135×** — but **ours still wins bitVLA whole-model.**
  Reconciliation: isolated cubes favor `7x4v`, but bitVLA's real M=32 / large-N,K shapes, integrated
  whole-model, still favor ours-v3. *(Method: swapped `runtime/backends/xnnpack_board/xn_gemm_rvv_shim.c`
  to `7x4v` MR=7 tiling; `scripts/k1_e2e_xnnpack.py --model output/bitvla_fp32_consistent`. Result file:
  `output/rvv_bench/k1_e2e_xnnpack_bitvla_7x4v.json`.)*
- **OpenBLAS fair re-run (`16x8_zvl256b`) — still TODO** (we used `8x8_zvl128b`).

**Status:** isolated re-run DONE (above). **TODO before presenting any "beats XNNPACK/OpenBLAS":**
re-run the whole-model bitVLA four-way with XNNPACK `7x4v` + OpenBLAS `16x8_zvl256b`. Driver variant
`ceiling_drivers/xnnpack_gemm_driver_7x4v.c` exists and is bit-exact.

### Root cause — why our *mining* missed it (not just the benchmark)
Verified from the committed mining artifacts:
- **We had `7x4v`** — it's indexed in `artifacts/kernel-mining/rvv/rvv_mined_v1_*/xnnpack_index.json`
  (`src/f32-gemm/gen/f32-gemm-7x4v-rvv.c`). It wasn't hidden.
- **But every CCA has `register_block: null`** — `{our,expert}_cca.yaml` in v1/v2/v3 all show
  `register_block: null`. The decoder extracts `contraction_form`, `accumulator_resident`, `widening`,
  `lmul`, `vl_strategy`, but the **register-block size (MR) comes back null for everyone.**
- **So MR is never a divergence and never an action** — `divergences.yaml` has no register-block axis.
  "The experts pick a high MR and reuse B — raise yours" was never discovered or routed. (`mr_adapts_to_m`
  is the opposite: it *clamps* MR down to M for correctness, not *raises* it for reuse.)
- **Our MR=4 came from our own spill-free tile sweep, not from the experts.**

**Implication:** our extraction captures the *compute* structure (which ops to fuse / keep resident) but
is **blind to register blocking — the #1 GEMM data-movement parameter.** That single `null` is why we
(a) never learned the experts' MR family, (b) hand-chose MR=4, and (c) then benchmarked XNNPACK's
weakest member (`1x4v`). It also voids the "library is frozen to MR=1, the compiler picks MR" framing:
XNNPACK *isn't* frozen (it ships MR=1…7 and selects), and *we* didn't mine MR either.

**Open TODOs (tracked):** (1) whole-model bitVLA four-way re-run with `7x4v` + `16x8_zvl256b`;
(2) fix the `register_block`/MR decoder so MR becomes a real mined divergence/action; (3) the deep
"why our whole-model runtime is slower on openvla/rdt2" investigation (separate section, in progress).

## Supervisor answer — "how did we beat XNNPACK if we learned FROM it?"

**We DO mine their kernels** — we disassemble 675 expert kernels (XNNPACK + OpenBLAS) to RVV asm. What
we keep is the structural **decision** (fused `vfmacc`, accumulator residency, widening, LMUL grouping,
VL strategy), baked into our **compiler** as general passes/knobs — *not the kernel copied verbatim*.
That generalizes: the compiler re-derives the decision shape-specialized per model and fuses it
whole-model. (Project invariant: *abstract WHY experts win into general compiler capabilities — no
hand-kernels, no shape-overfit.*)

> **CORRECTION (rigorous fair+measured pass — supersedes ALL earlier versions of this answer).** Two
> prior framings were wrong and are retracted: (a) "experts ahead at the ceiling" (used the wrong fork),
> AND (b) its over-correction "ours-v3 wins the kernel too" (that leaned on **spike instret**, which the
> deck itself calls a mirage). The K1-silicon truth, now MEASURED on both arms, is below — and it is less
> flattering: **our GEMM kernel is slower than the experts' best on every model; the bitVLA win is a
> whole-model *schedule* win, not a kernel win.**

The honest two-level picture, all on **K1 silicon** (spike is directional-only, deck slide 51 — do NOT
use its instret to rank kernels):

**Level 1 — the isolated GEMM kernel: we LOSE.** On neutral 32³/64³/128³ cubes (real K1 ticks, bit-exact)
XNNPACK's best `7x4v` beats our v3 (32³ 175 vs 224, 128³ 10 341 vs 14 256; 64³ ties). And per-model the
matmul bucket is now **measured on both arms** (`ours_board` backend + rdtime bracket,
`dispatch_breakdown_measured.json`): our v3 matmul is **3.18× slower on bitVLA · 7.9× on openvla · 13.6×
on rdt2**. Root cause: every CCA has `register_block: null` (below) — we never mined MR, hand-chose MR=4,
and the experts' MR=7/16 register-blocking wins. **We do not have a better GEMM kernel.**

**Level 2 — the whole model: we win bitVLA only, and it's the SCHEDULE, not the kernel.** Fair four-way
(K1, N=3, cos-gated, each expert with its BEST kernel `7x4v`/`16x8_zvl256b`,
`k1_e2e_fair_{bitvla,openvla,rdt2}.json`):
- **bitVLA:** ours `accumulator_resident_wholemodel_vf` **148.3 ms WINS** vs XNNPACK 167.3 ms (1.13×),
  OpenBLAS 180.5 ms (1.22×).
- **openvla:** ours 1095 ms **LOSES** vs XNNPACK 627 ms (0.57×). **rdt2:** ours 30.2 s **LOSES** vs 18.6 s.

**Why bitVLA wins despite a slower kernel — the measured mechanism.** It is NOT the matmul: the apples-to-
apples kernel swap `ours_kernels` (baseline non-matmul + our v3 matmul) = **178 ms** is actually *slower*
than `xnnpack_kernels` = 169 ms on bitVLA (because our matmul is 3.18× slower). The bitVLA win comes
entirely from our **whole-model `vf` schedule's NON-matmul lowering** (148 ms total < 178 ms), i.e. how we
compile attention/norm/activation/layout — the part a frozen kernel library *cannot* touch. The catch:
that same `vf` schedule *hurts* the big VLAs (openvla `vf` 1095 ms vs its own baseline-non-matmul 692 ms),
so it's a bitVLA-specific schedule win, not a general one.

**So the defensible claim is narrow and exact:** *"On bitVLA, our compiler's whole-model lowering is
1.13×/1.22× faster than taking the same model and swapping in XNNPACK's or OpenBLAS's best RVV GEMM —
measured on K1 silicon, fair, cos ≥ 0.9999. The advantage is whole-model scheduling of the non-matmul
path; our GEMM kernel itself is slower (a `register_block` blind spot), and on openvla/rdt2 we lose."*
Do NOT claim a better kernel, a spike-based win, or a win beyond bitVLA. **Fairness:** same-pass, experts'
resident-weight pack excluded from both timings, all cos ≥ 0.9999 (SpacemiT K1, VLEN=256, `cycle_accurate=false`).

---

## Artifact trail (every step is real & committed)

| step | artifact | shows |
|---|---|---|
| input | `output/rvv_workloads/matmul_f32_64x64x64/model.mlir` | the 64³ `linalg.matmul` |
| baseline asm | `runs/rvv_bench/hand_v0_matmul_f32_64x64x64/generated/objdump.txt` | `vfmul.vv`×4 + `vfadd.vv`×4 + `vse32.v` spill |
| CCA (both) | `artifacts/kernel-mining/rvv/mining_rvv_v3_20260619T092631/{our_cca,expert_cca}.yaml` | compute + vector facets |
| divergences | `…/mining_rvv_v3_*/divergences.yaml` | `contraction_form`, `widening`, `lmul`, `vl_strategy`, … |
| actions / classes | `…/mining_rvv_v3_*/actions.yaml` + `kernels/action_catalog.py` | FLAG/KNOB/HEURISTIC/PASS routes |
| feature | `merlin/python/merlin/llvmlower/impr_features.py` | `fused_vfmacc_tiled`, `accumulator_resident_microkernel_v3` |
| certify | `runs/rvv_bench/impr_rvv_v5_*/results.yaml` | K-ladder, cos = 1.0, `vfmacc.vf` present |
| fork-vs-own-baseline | `output/rvv_bench/k1_gemm64_benchmark.md` | baseline 12.86 M ns → vfmacc 1.63 M ns = **7.9×** (vs OUR baseline, NOT vs experts) |
| whole-model win (FAIR) | `output/rvv_bench/k1_e2e_fair_bitvla.json` | ours 148.3 ms vs XNNPACK-7x4v 167.3 (1.13×) · OpenBLAS-16x8 180.5 (1.22×) |
| measured matmul split | `output/rvv_bench/dispatch_breakdown_measured.json` | our v3 matmul SLOWER: 3.18×/7.9×/13.6× (bitvla/openvla/rdt2) |
| ceiling framing | `output/kernels/ceiling/{dispatch_breakdown,packing_residual}.md` | kernel ≠ system; the win is non-matmul scheduling |
| int8 bridge | the `compute.widening` divergence above; Gemmini `experiments/gemmini_capsule_bench_v0/.../A2_single_tile_matmul` (int8, 316-cyc RTL) | i8×i8→i32 `vwmacc` / systolic |

---

## Slide integration plan (maps to the real deck, `Merlin-9.pdf`)

The kernel-mining section already exists as deck slides **26–35**; the new journey figures slot in
around them, and one slide gets deleted:

- **DELETE slide 30** ("Beam search as optimization evidence", the old dual diagram) — it is tagged
  *"TODO: Simplify this diagram OR replace it."* It is replaced by slides 31 (`fig7`) + 32 (`fig9`).
- Order the section as: **26** (section title) → **27–29** (motivation: slow compilers, one-to-many,
  generated-kernels-overfit) → **`fig22`** (journey strip) → **`fig20`** (kernel) → **`fig21`** (asm
  diff) → **31** (`fig7`, CCA→action — *fixed*, see below) → **`fig24`** (action classes) → **32**
  (`fig9`, gate ladder) → **33** (`fig6`, beam tree) → **34** (`fig8`, candidates) → **35** (`fig1`,
  four-way win) → **`fig23`** (why we beat XNNPACK — the supervisor answer).
- Slides 33–35 already carry the whole-model story (beam winner `v3+lmul`, the four-way `ours WINS`
  on bitvla and `ours = 60% / 63%` on openvla/rdt2) — keep them; `fig23` is the capstone that explains
  *why* (kernel ≠ system).

## Fixes applied this pass (factual accuracy)
- **`fig7` / slide 31 — number attribution corrected.** The `CompilerAction` card (the
  `fused_vfmacc_contraction` PASS) previously read "MEASURED → 16.77× whole-model". That 16.8× belongs
  to `accumulator_resident_microkernel_v3 + lmul` (the *beam composition*, slides 33–35), NOT to this
  PASS. Now reads **"7.9× isolated · 64³ GEMM (cos=1.0) · composes to 16.8× E2E — see beam."** Also
  dropped the "no per-K accumulator spill" line from this card (that is the v3 micro-kernel's
  achievement; the fused-vfmacc transform path still spills — per `impr_features.py`/`action_catalog.py`).
- **`fig21` — made byte-real + honestly labeled.** Right card now shows the fork's *actual* objdump
  (`vl8r.v` + `flw` + `vfmacc.vf v8,…` with the real histogram `vfmul=vfadd=0, vfmacc=8065`,
  `impr_rvv_v5`), labeled "Fused vfmacc — our fork." Footer notes that XNNPACK/OpenBLAS emit the same
  fused form (their literal asm is in `merlin/python/tests/data/cca_asm/`).

## Feature → number attribution (keep these straight)
- `fused_vfmacc_contraction` (PASS, the J2–J5 teaching feature) → **7.9× isolated** 64³ GEMM (cos=1.0).
  Fuses mul+add → `vfmacc`; does **not** by itself fully close accumulator residency.
- `accumulator_resident_microkernel_v3` (+ `lmul_widen_n`) → **16.8× whole-model bitvla** (the beam
  winner, slides 33–35). This is the genuine zero-in-loop-spill, register-resident micro-kernel.
- CCA mapping: the two divergences shown on slide 31 route to these two features —
  `contraction_form → fused_vfmacc` and `accumulator_resident → microkernel_v3`. (Note: in
  `mining_rvv_v3/divergences.yaml` only `contraction_form`/`widening`/`reduction`/`epilogue`/`lmul`/
  `vl_strategy` are emitted as divergences; `accumulator_resident` lives in the CCA with `ours=null`
  and is routed via the catalog — fine to show as a divergence conceptually, but it is the v3 feature,
  not the fused-vfmacc PASS, that closes it.)

## bitVLA's quant format — what actually ran (verified, repo-grounded)

bitVLA's name implies a 1-bit model, so an informed audience *will* ask "how do all the frameworks
support that weird quant format?" The honest answer: **they don't — the four-way comparison ran bitVLA
in fp32; none of XNNPACK / OpenBLAS / ours executed bitVLA's native ternary.**

Verified facts (with sources):
- **Native format = W1.58 ternary.** `dse_guidance/models.py:70` → `note="BitNet ternary VLA."`;
  `dse_guidance/quant_metadata.py:69` → `"W1.58 ternary (BitLinear, packed int2: 4 ternary values per
  i8 byte)"`, per-tensor absmean scale.
- **The fair bitVLA race ran fp32.** `output/rvv_bench/k1_e2e_fair_bitvla.json` is `bitvla_fp32_consistent`,
  "fp32 cos", experts at their best fp32 kernels (`7x4v` / `16x8_zvl256b`); every matmul in
  `output/bitvla_fp32_consistent/model.mlir` is `…xf32`. The ternary weights are **dequantized to dense
  fp32** before the race.
- **Why fp32:** it's the only format all three can execute. XNNPACK/OpenBLAS are fp32/int8 BLAS — neither
  has a ternary (W1.58) GEMM kernel. fp32 makes it the same op / shapes / numerics for all three
  (cos ≥ 0.9999) — a fair fp32 matmul race, not a quant-format race.
- **Native ternary is *captured* (for DSE), not *executed* (in the race).** `quant_metadata.py:23` →
  `recaptures_native/bitvla` captures the packed-int2 ternary storage + absmean scale + a named
  `quant_ext.unpack_int2` op. That is structural analysis for the (un-presented) DSE track — it is not
  run as a GEMM by any framework in the perf comparison.
- **The int8 captures are torchao int8 weight-only — NOT native ternary** (`quant_metadata.py:9-10`).
  bitVLA does run W8A8 int8 on the K1 board (the int8 capstone), but it was not raced vs the experts and
  is still not native 1.58-bit.

**Slide guidance:** label the result **"bitVLA (fp32)"** on the four-way (slide 35 / `fig1`) and on
`fig23`. Speaker note: *"bitVLA's W1.58 ternary weights are dequantized to fp32 for a fair
cross-framework race; the native ternary is captured for analysis but not executed by any framework
here, and the int8 path is a separate, non-native track."*

## Honesty notes to keep on the slides
- We **mine** (disassemble + abstract) the kernels; we never **transplant** an expert kernel.
- **Our GEMM kernel is SLOWER than the experts' best — say so.** Measured on K1 (both arms timed), our v3
  matmul is 3.18×/7.9×/13.6× slower (bitvla/openvla/rdt2); on neutral cubes `7x4v` beats it too. Do NOT
  use spike instret to claim a kernel win — it's a mirage. The bitVLA whole-model win (148 vs 167/180 ms,
  1.13×/1.22×) is the **vf schedule's non-matmul path**, not the GEMM. The honest *losses* to state:
  openvla/rdt2 (0.57×/0.61×, dispatch-bound, and the vf schedule itself hurts there).
- int8 was **not** the measured RVV mining target (mining ran on f32); the `widening` axis is a real
  *bridge*, not a measured int8 win. Gemmini is genuinely int8-only and has its own certified example.
- All board numbers are K1 wall time (`cycle_accurate=false`); RTL-cycle confirmation is out of scope.
