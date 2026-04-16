# Saturn OPU FireSim — f32-reduction lowering hang: findings & workaround plan

**Date**: 2026-04-14 (pre-deadline consolidation)
**Status**: Root cause isolated via 37 targeted micro-tests + per-op bisection.
**Scope**: vit_small + tinyllama on Saturn OPU FireSim; non-transformer models
unaffected.

## TL;DR

Every RVV dispatch whose body is a **f32 reduction** (either an explicit
`linalg.generic` with a reduction iterator or the `linalg.softmax` named op)
hangs on Saturn FireSim hardware. The hang is in whatever the LLVM codegen
emits after the existing `vfredusum.vs` scalarization runs — most likely a
tree-reduction using `vfmacc.vf` at fractional LMUL (`mf2`, VL=1) interleaved
with `vslidedown.vi`. Every other path (f32 elementwise, i8 matmul via the OPU
ukernel, i8 reductions) is healthy.

The symptom class is precisely confined to **6 dispatches** in the 2-block
vit_small (4 LayerNorm dispatches + 2 attention softmax dispatches). The OPU
ukernel path (`iree_uk_opu_matmul` + VOPACC/OPMVINBCAST custom ISA) is not
involved in any hanging dispatch and must be preserved by any fix.

## Evidence summary

| Test | Dispatch symbol at hang | Verdict |
|------|-------------------------|---------|
| **f32 elementwise (no reduction)** | | |
| `mt_rsqrt_64xf32`        | elementwise_64_f32            | pass (warmup + 4+ bench iter, ~3200 cyc) |
| `mt_sqrt_64xf32`         | elementwise_64_f32            | pass (~1550 cyc) |
| `mt_erf_64xf32`          | elementwise_64_f32            | pass (~3300 cyc) |
| `mt_cvt_64`              | elementwise_64_f32            | pass (~330 cyc) |
| `mt_divf_vv_64`          | elementwise_64_f32            | pass (~1600 cyc) |
| **f32 reduction / softmax** | | |
| `mt_bmm_4x64x64` (f32)   | reduction_256x64x32_f32       | **HANG** |
| `mt_matmul_64x128` (f32) | reduction_64x128x128_f32      | **HANG** |
| `mt_softmax_64x64` (f32) | softmax_64x64xf32_dispatch_tensor_store | **HANG** |
| `mt_vit_d1_layernorm`    | reduction_64x128_f32          | **HANG** |
| **i8 matmul (via OPU ukernel)** | | |
| `mt_mm_i8_64x128`        | reduction_64x128x128_i8xi32   | progresses at ~41000 cyc/wg |
| `mt_mm_i8_narrow_m`      | reduction_128x128_i8xi32      | pass (warmup iter 1 complete, ~2300 cyc/wg) |
| `mt_mm_i8_wide`          | reduction_128x256x256_i8xi32  | progresses at ~137000 cyc/wg |
| `mt_mm_i8_resid_bias`    | reduction_64x128x128_i8xi32   | progresses at ~30000 cyc/wg |

Every i8 matmul test ends killed by timeout mid-execution rather than hung;
every f32 reduction test halts at `[dn]` with no `[dc]` exit for wg=0.

## Dispatches affected in vit_small (2 transformer blocks)

| Dispatch | Role | f32 reduction(s) | Hang |
|----|----|----|----|
| **d1**  | pre-attn LayerNorm (block 0)  | mean + variance | ✅ confirmed |
| **d7**  | attention softmax (block 0)   | max + sum(exp) | ✅ pattern-match |
| **d10** | post-attn LayerNorm (block 0) | mean + variance | ✅ pattern-match |
| **d13** | pre-FFN LayerNorm (block 0)   | mean + variance | ✅ pattern-match |
| **d19** | attention softmax (block 1)   | max + sum(exp) | ✅ pattern-match |
| **d22** | post-FFN LayerNorm (block 1)  | mean + variance | ✅ pattern-match |

All other 22 vit_small dispatches either use the OPU ukernel (i8 matmul) or
are pure f32 elementwise (quant / requant chains) and should execute normally
once the 6 above are fixed.

tinyllama is expected to have the same bug family (RMSNorm = sum-of-squares
reduction, attention softmax), with proportionally more dispatches (32 blocks
vs 2).

## What's NOT the bug (things we ruled out)

1. **vfredusum.vs** — already scalarized in `ConvertToLLVM.cpp`
   (`ScalarizeXopuFloatReductionPattern`). Verified: `grep -c vfredusum` = 0 in
   the linked .s of both vit_small and the hanging micro-tests.
2. **`math.rsqrt` / `math.sqrt` / `math.erf`** — all pass in isolation.
   LayerNorm's f32 math ops themselves are fine; only the reduction that
   precedes them hangs.
3. **`vfdiv.vv`** — passes (`mt_divf_vv_64`).
4. **`vfcvt.f.x.v` / `vfcvt.x.f.v`** — passes (`mt_cvt_64`).
5. **Fractional LMUL (`mf2`) in isolation** — `large_mlp` uses mf2 in 36 sites
   and passes; not a pure-mf2 issue.
6. **OPU custom ISA (VOPACC, OPMVINBCAST, VMV_RV, VMV_VR)** — every i8 matmul
   test exercises these and they all work.
7. **i8 matmul + residual + bias + triple-saturation-requant fusion** (vit d9
   pattern) — passes because the reduction is i8 (ukernel) and the requant is
   pure f32 elementwise.
8. **Alignment of bindings** — confirmed all bindings are ≥64B aligned in the
   hanging dispatches; not an alignment issue.
9. **Memory/heap limits** — 1 GB heap, 64 MiB stack; plenty of headroom.
   `IREE_DEVICE_SIZE_MAX` is `SIZE_MAX` on 64-bit so tinyllama's 1.1 GB VMFB
   fits.
10. **ded6b594ab cherry-pick** (`IREE_HAL_MEMORY_ACCESS_UNALIGNED` flag) —
    applied to `memory_file.c`. Not the root cause but prevents a silent fast-
    path slowdown.

## What IS the bug (hypothesis, backed by data)

The compiler lowers `linalg.generic` with a reduction iterator over f32 into
(a) a vector-contraction lowering pattern, which (b) codegen's to a tree
reduction with `vfmacc.vf` at VL=1 interleaved with `vslidedown.vi` and
`vsetivli … e32, mf2, …` LMUL switches, which (c) Saturn's vector unit does
not drain when VL=1 + mf2 is combined with slides. The resulting pipeline
stall is indistinguishable from a hang at the instruction level (no exception,
no progress, no advance of rdcycle).

Direct evidence from disassembling
`/tmp/phase_d/mt_bmm_opu/module_main_dispatch_0_embedded_elf_riscv_64.s`:

```asm
.LBB0_2:
    vmv1r.v      v12, v14
    flw          fa5, -8(a5)
    ...
    vsetivli     zero, 1, e32, mf2, ta, ma   ← fractional LMUL, VL=1
    vfmacc.vf    v12, fa5, v10               ← single-lane FMA
    vsetivli     zero, 1, e32, m1, ta, ma    ← LMUL transition
    vslidedown.vi v13, v14, 1
    vsetivli     zero, 1, e32, mf2, ta, ma
    vfmacc.vf    v13, fa4, v10
    ...
```

Saturn apparently issues these instructions but doesn't retire them when VL=1
with mf2 + slides are interleaved in a tight loop.

## Flag-bisection results

| Variant | Flag | Result | Meaning |
|---|---|---|---|
| `_opu_scalar`    | `--iree-llvmcpu-target-vector-width-in-bytes=0` | **HANG** at `reduction_256x64x32_f32` (same as baseline) | IREE's vector-width flag only disables IREE's own vector dialect lowering. **LLVM's loop vectorizer still runs independently and re-emits the broken tree-reduction pattern.** |
| `_opu_novec`     | drop `+v` target feature | **PASS** — warmup iter 1/2 done, all 16 workgroups exit at ~21 k cyc | Dropping `+v` fixes the hang. The broken pattern is **specifically some vector-codegen path**. With no `+v` there are no RVV instructions at all, so scalar `fadd.s` loops run fine. |
| `_opu_nocustom`  | `--iree-llvmcpu-enable-vector-contract-custom-kernels=false` | _pending_ | expected: still hangs, since `_opu_scalar` didn't fix it |
| `_opu_noukernel` | `--iree-llvmcpu-enable-ukernels=none` | _pending_ | irrelevant for f32 reduction; baseline sanity |

### Implication: per-function `-v` override is the surgical fix

The bisect results give us a complete picture:

- `_opu_scalar` still hangs → IREE's own vectorizer is *not* the only emit site.
- `_opu_novec` passes → dropping `+v` entirely removes the broken pattern.

Globally dropping `+v` is unacceptable (kills the OPU ukernel path, which
needs `+v,+xopu`). But **LLVM supports per-function `target-features`
attributes** — different functions in the same module can be compiled for
different feature subsets. So we can emit:

```llvm
; LayerNorm / softmax dispatch — scalar only
define void @main_dispatch_1_reduction_64x128_f32(...) #42 { ... }
attributes #42 = { "target-features"="-v" ... }

; Every other function (matmul, ukernel, elementwise) keeps +v,+xopu
define void @iree_uk_opu_matmul(...) #99 { ... }
attributes #99 = { "target-features"="+v,+xopu" ... }
```

LLVM emits **scalar `fadd.s`/`fmul.s` loops** for the `-v`-marked function
and **full RVV + OPU custom ISA** for everything else. Same binary,
per-function isolation. This is the right fix. Call it **Option E**; it
supersedes A–D.

### Option E (NEW RECOMMENDATION) — Per-function `target-features="-v"` attribute

**File 1**: new MLIR pass
`third_party/iree_bar/compiler/src/iree/compiler/Codegen/Common/F32ReductionDevectorizePass.cpp`.

Walk every `func.func` inside `hal.executable.variant`. If the body contains
either a `linalg.generic` with at least one reduction iterator + f32
accumulator, **or** a `linalg.softmax`, attach the MLIR attribute
`iree.llvmcpu.override_features = "-v"` on that `func.func`.

**File 2**: edit
`third_party/iree_bar/compiler/src/iree/compiler/Codegen/LLVMCPU/ConvertToLLVM.cpp`
(the MLIR→LLVM conversion). After the default `target-features` attribute is
set, if `iree.llvmcpu.override_features` is present on the source func, merge
its value into the emitted LLVM function attribute `target-features` string.

**Gate**: the pass only fires when the variant's target features include `+v`
(check `hasAnyVFeature`). Spacemit RVV builds, OPU builds, and OPU_LLM builds
all get covered; scalar-only builds are untouched.

**Effect**:
- i8 contractions (i32 accumulator) are unmatched → OPU ukernel path intact,
  VOPACC + OPMVINBCAST still emit.
- f32 elementwise (no reduction iterator) unmatched → fast vector code stays.
- ~6 dispatches in vit_small (4 LN + 2 softmax) get scalarized. Expected
  cycle cost: the `_opu_novec` variant ran at ~21 k cyc/wg × 16 wg ≈ 340 k
  cycles per iteration. The hanging OPU variant was ~N/A. Total runtime
  penalty: probably +30-50 % on vit_small overall because LN/softmax are a
  minority of the work. Acceptable.

**Cost**: ~60 lines C++ (pass + conversion hook) + 1 MLIR regression test.

**Verification**: `mt_vit_d1_layernorm`, `mt_vit_d7_softmax`, `mt_bmm_4x64x64`,
`mt_matmul_64x128`, `mt_softmax_64x64` → all pass. `mt_mm_i8_*`,
`mt_rsqrt/sqrt/erf`, non-transformer models → same cycle counts as before
(±5 %).

### Why not the earlier Options A–D?

- **A (IREE linalg scalarize)**: `_opu_scalar` result proves IREE-level
  scalarization isn't enough; LLVM re-vectorizes. A alone would not have
  fixed vit_small.
- **B (per-dispatch translation_info → CPUDefault)**: same problem as A,
  plus coarser granularity (whole dispatch scalarized, even elementwise ops
  fused into it).
- **C (LLVM `"no-vectorize"` function attribute)**: this was my pre-bisect
  best guess. Now refined: instead of `no-vectorize` (which only stops
  LLVM's loop vectorizer, may miss SLP or instruction selection vector
  emission), **Option E uses the stronger `target-features="-v"` which
  removes RVV from the ISA entirely for that one function**. Guaranteed to
  work because `_opu_novec` (the whole-module equivalent) passed.
- **D (model rewrite)**: unnecessary, Option E handles it in the compiler.

## Sweep-run classification (from 2026-04-14 20:00-21:30 microtest sweep)

**Important**: the old `run_microtests.sh` verdict parser grepped for the
literal string `Iteration 1`, which the benchmark harness never prints (it
prints `Bench iter 1/10`). As a result every test is labeled `hang` in
`/tmp/microtests.csv` even when it completed multiple warmup + bench
iterations. Re-classify from uartlog content directly — the cycles column in
the CSV is still correct (it's the sum of every `[dc] cyc=…` seen, so a high
number means *more* iterations ran, not an earlier hang).

Parser fix applied in `run_microtests.sh` (new verdicts: pass / warmup /
progress / hang / early).

Corrected classification:

| Test (opu or rvv) | Verdict | Evidence from uartlog |
|---|---|---|
| mt_rsqrt_64xf32 | pass | warmup 1+2 done, bench iters 1–5 printed (~3200 cyc each) |
| mt_sqrt_64xf32 | pass | bench iters 1–3+ (~1550 cyc each) |
| mt_erf_64xf32 | pass | bench iters (~3300 cyc each) |
| mt_cvt_64 | pass | bench iters (~330 cyc each) |
| mt_divf_vv_64 | pass | bench iters (~1600 cyc each) |
| mt_mm_i8_narrow_m | pass | warmup iter 1 done, iter 2 starting (8 wg × ~2300 cyc) |
| mt_mm_i8_64x128 | progress | d0 elementwise_8192 passes, d1 reduction_64x128x128_i8xi32 at wg 10+ × ~41 k cyc/wg |
| mt_mm_i8_resid_bias | progress | same d0 pass, d1 reduction at wg 10+ × ~30 k cyc/wg (fusion makes it faster) |
| mt_mm_i8_wide | progress | d0 wg 0–7 pass (~12 k/wg), d1 reduction_128x256x256_i8xi32 at wg 4+ × ~137 k cyc/wg |
| mt_mm_i8_accumulate | progress | cycles accumulating similarly to mm_i8_64x128 |
| **mt_vit_d9_matmul** | **progress** | d0 dequant passes, d1 reduction_64x128x128_i8xi32 wg 10+ × ~35 k cyc/wg → **d9 is NOT a hang** |
| **mt_vit_d1_layernorm** | **HANG** | d0 elementwise_8192_f32xi8 passes (11 500 cyc each wg), d1 reduction_64x128_f32 → no [dc] ever |
| **mt_vit_d7_softmax** | **HANG** | d0 elementwise_16384_f32xi8 passes, d1 softmax_4x64x64xf32_generic → no [dc] |
| mt_bmm_4x64x64 | HANG | reduction_256x64x32_f32 — no [dc] for wg=0 |
| mt_matmul_64x128 | HANG | reduction_64x128x128_f32 — no [dc] |
| mt_softmax_64x64 | HANG | softmax_64x64xf32_dispatch_tensor_store — no [dc] |
| **mt_matmul_64x128_opu_llm** | **early** | never even printed [dn] (catastrophically earlier failure than plain OPU variant) |

### Extra findings from the sweep

1. **mt_vit_d9_matmul confirmed NOT a hang**. Both OPU and RVV variants run
   the i8 reduction dispatch at ~35 k cyc/wg through at least wg 10 of 16.
   Our original "OPU vit_small hangs at d9" attribution was wrong — the
   actual hang in the full model is later, at d10 LayerNorm.
2. **OPU_LLM compile-flag delta is independently broken**. `mt_matmul_64x128_opu_llm`
   didn't even emit `[dn]` for dispatch 0 — suggests `--iree-preprocessing-collapse-multi-n-contractions`
   produces code that traps before the first dispatch. **tinyllama uses
   FLAGS_MODEL_OPU_LLM**, so tinyllama may have two stacked bugs: this pre-
   dispatch crash + the f32-reduction hang. The vit_small hang analysis
   does not transfer cleanly to tinyllama; separate investigation needed
   for tinyllama post-deadline.
3. **d0 (elementwise f32→i8 dequant) always passes** across every test,
   regardless of shape. The broken lowering is strictly limited to reduction
   iterators.
4. **Softmax fusion state doesn't matter**. Standalone softmax
   (`_dispatch_tensor_store`) and fused softmax-with-generic (`_generic`)
   both hang identically. Fix must be upstream of fusion.
5. **i8 reduction cycle counts are stable across variants**. OPU and RVV
   mirror each other within ±1 % per workgroup, confirming the f32-reduction
   hang is variant-agnostic (and hence the i8 matmul path is variant-agnostic
   too — both share the same OPU ukernel linkage via `iree_uk_opu_matmul`).

## Workaround plan (keep OPU ukernels intact)

Four implementable remediation paths, ordered by surgical scope (most
targeted first → widest blast radius last). Each lists the exact file to
touch and expected LoC. Pick A first; fall back to B if A is incomplete;
C/D are emergency broad-strokes only.

### Option A (RECOMMENDED) — Pre-vectorization linalg scalarize pass

**File**: new `third_party/iree_bar/compiler/src/iree/compiler/Codegen/Common/ScalarizeF32ReductionPass.cpp`
+ entry in `Passes.td` + hook in the LLVMCPU pipeline before
`GenericVectorizationPass`.

**Pattern**: match `linalg.generic` where
- at least one iterator is `utils::IteratorType::reduction`, AND
- the accumulator (output) element type is `f32`, AND
- the target attribute matches our gate: `+xopu` present OR any RISC-V V
  feature present.

**Rewrite**: replace the op with a nested `scf.for` that walks each reduction
dim element-by-element and applies the body with a scalar accumulator. Output
is unchanged except now emitted as scalar `fadd.s`/`fmul.s`/`fsqrt.s`
instructions instead of `vfadd.vv`/`vfmacc.vf`-with-tree-reduction.

**Also catch `linalg.softmax`** in the same pass: invoke
`linalg::decomposeSoftmax` first (produces max+sub+exp+sum+div as explicit
linalg.generics), then let the same scalarize pattern run on the two
reductions inside.

**Effect**:
- i8 contractions (i32 accumulators, not f32) are unmatched → `linalg.matmul`
  / `iree_uk_opu_matmul` ukernel path is untouched, OPU VOPACC code still
  emits.
- Pure f32 elementwise (no reduction iterator) is unmatched → fast vector
  codegen stays.
- Only LayerNorm / softmax / explicit f32 reductions fall back to scalar.
  Per-dispatch cost: +4-8× cycles on those ~6 dispatches. Acceptable.

**Cost**: 60-80 lines C++ + 1 MLIR regression test.

**Verification**: `mt_vit_d1_layernorm`, `mt_vit_d7_softmax`, `mt_bmm_4x64x64`,
`mt_matmul_64x128`, `mt_softmax_64x64` should all pass. `mt_mm_i8_*`,
`mt_rsqrt_64xf32`, non-transformer models should remain unchanged (same
cycle counts ±5%).

### Option B — Per-dispatch translation_info override

**File**: new pattern in `compiler/src/iree/compiler/Codegen/LLVMCPU/KernelDispatch.cpp`
(or equivalent) that matches the same f32-reduction pattern at dispatch
formation time and attaches `#iree_codegen.translation_info<CPUDefault>`
(scalar strategy) instead of the default `CPUDoubleTilingExpert`.

**Effect**: the entire dispatch function for a f32-reduction generic gets
scalar codegen (not just the reduction body). Broader than Option A — any
elementwise ops fused into the same dispatch also get scalarized.

**Cost**: 20-30 lines, simpler than A, but less surgical (may regress cycle
counts on fused dispatches that contain *both* a reduction and a lot of
elementwise work). Use only if Option A leaves residual hangs (e.g., if some
linalg.generic f32 reductions route around the pattern for structural
reasons).

### Option C — LLVM function-attribute `optnone` / `"no-vectorize"`

**File**: `compiler/src/iree/compiler/Codegen/LLVMCPU/ConvertToLLVM.cpp`
post-processing step that walks the generated LLVM module, finds every
function whose body contains an f32 reduction IR pattern (`llvm.intr.vector.reduce.fadd.v*f32`
or an explicit reduction loop), and attaches the `"no-vectorize"` function
attribute.

**Effect**: LLVM skips its own loop-vectorizer on those functions. Scalar
codegen wins by default. Does NOT prevent IREE's own vector codegen — so it
only helps if the hang is specifically in LLVM's tree-reduction output, not
in IREE's vector.contract lowering. We don't know yet which layer emits the
broken pattern (the 4 bmm flag-bisect variants will tell us).

**Cost**: 20 lines, but effectiveness depends on bisect outcome.

### Option D — Model-level rewrite (LAST RESORT)

Replace `LayerNorm(x)` with a precomputed-statistics approximation baked in
as constants. Only applies if we can't ship a compiler fix. Accuracy takes a
hit; not a real solution, only a demo workaround.

**Not for vit/tinyllama final production** — for deadline-demo purposes only
if A/B/C all fail.

### What NOT to do

- **Don't** disable LLVM loop vectorization globally
  (`--iree-llvmcpu-enable-loop-vectorization=false`): it also strips the
  ukernel lowering so the i8 OPU path breaks.
- **Don't** drop `+v`: every dispatch loses vector codegen including
  working ones. Entire runtime slows 10-50×.
- **Don't** re-gate on `+xopu` exclusively: vit_small RVV (no `+xopu`)
  has the same hang family; the gate must include `isAnyRVV`.
- **Don't** try to fix tinyllama with the same patch alone. Tinyllama has a
  SECOND bug (OPU_LLM preprocessing pre-dispatch crash — see sweep
  findings above). That's a separate investigation post-deadline.

### Implementation order (for the deadline push)

1. **Step 1 (30 min)**: Implement Option A as a new pass. Hook into
   `LLVMCPU/Passes.cpp` immediately before `GenericVectorizationPass`.
2. **Step 2 (10 min)**: Rebuild host `iree-compile`.
   `CHIPYARD_ROOT=… uv run tools/merlin.py build --profile vanilla --config release`.
3. **Step 3 (10 min)**: Rebuild the microtest binaries
   `uv run tools/merlin.py build --profile firesim --config release --cmake-target microtests`.
4. **Step 4 (15-30 min)**: Run the **focused set** on FireSim:
   `mt_vit_d1_layernorm_rvv`, `mt_vit_d7_softmax_opu`, `mt_bmm_4x64x64_rvv`,
   `mt_mm_i8_64x128_opu` (regression check), `mt_rsqrt_64xf32_rvv`
   (regression check).
5. **Step 5**: If all 5 pass, rebuild vit_small with the patched compiler and
   rerun on FireSim. Expected: full inference in <10 min.
6. **Step 6 (post-deadline)**: separately investigate tinyllama OPU_LLM.

## Verification plan (post-fix)

1. Rebuild `microtests` with the new pattern. Re-run `mt_vit_d1_layernorm`
   and `mt_bmm_4x64x64` — both should pass to Iteration 1.
2. Rebuild vit_small with the new pattern. Run on FireSim, expect
   completion in <10 min per inference.
3. Rebuild tinyllama with the new pattern. Run on FireSim.
4. Regress against the Phase-A sweep of non-transformer models
   (`large_mlp`, `mlp_wide`, `dronet`, `yolov8_nano`) — all must still pass
   at their previous timing (no regression).

## Artifacts (for reproduction)

- 37 micro-tests: `samples/SaturnOPU/simple_embedding_ukernel/tests/microtests/*.mlir`
- Sweep runner: `build_tools/firesim/run_microtests.sh`
- Focused 7-test runner: `build_tools/firesim/run_pinpoint.sh`
- Per-op CSV output: `/tmp/microtests.csv` (columns: test, variant, verdict,
  last_ord, last_symbol, cycles, notes)
- Partial parse of in-flight runs: `/tmp/microtests_partial.csv`
- Raw uartlogs: `/tmp/microtests_logs/*.uartlog`
- Pre-emptive analysis: `/tmp/phase_d/*.txt`
- Lifted dispatch MLIRs from vit_small d1/d7/d9:
  `tests/microtests/vit_layernorm_d1_chunk.mlir`,
  `vit_d7_softmax_chunk.mlir`, `vit_d9_chunk.mlir`

## Open questions (not blocking)

1. Exact RISC-V V-extension instruction sequence that stalls Saturn. We have
   a candidate (mf2+slide+vfmacc VL=1) from disassembly but haven't confirmed
   with a PC trace. A TraceRV run with `+tracefile` on `mt_bmm_4x64x64_opu`
   would pinpoint it; deprioritized because the workaround doesn't require
   knowing the exact instruction.
2. Whether tinyllama's RMSNorm bug is identical to vit's LayerNorm bug or a
   variant. Pattern analysis says identical; not yet directly verified because
   tinyllama takes ~1 hr per FireSim run.
3. Whether the compiler uses `mmt4d` + `TRANSPOSED_OUTPUT` anywhere in
   vit/tinyllama. Analysis of per-dispatch MLIRs suggests no, but hasn't been
   fully exhaustively checked.
