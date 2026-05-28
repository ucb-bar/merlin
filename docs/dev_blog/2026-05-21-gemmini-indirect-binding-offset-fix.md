# 2026-05-21: Bug A — Gemmini × dronet × FireSim: the Indirect-binding offset journey

> **Status:** Landed
> **Author:** Agustin (pair-debugged with Claude)
> **Test:** `benchmarks/firesim_shuttle/run_hetero.sh dronet_with_intermediate gemmini gemmini 1` → 4 output hashes match scalar baseline bit-perfectly.

## Context and Goal

For two days the dronet × Gemmini × FireSim pipeline had been emitting four output hashes that didn't match anything — not the scalar baseline, not ONNX runtime, not IREE x86 host. The handover doc from 2026-05-20 had localized the symptom to dispatch_16 / dispatch_18 (the two `matmul_1x1x2048` FC heads) and proposed a "Fix #3" walkback that recovered the `hal.interface.binding.subspan` `byte_offset` for the bufferized memref pointer feeding `gemmini.mvin` / `gemmini.mvout`. The IR-level traces showed the fix was emitting the right address. FireSim hashes were still wrong.

Goal: figure out why Fix #3 didn't actually change the on-board output, and produce a fix that makes dronet × Gemmini bit-perfectly match the scalar baseline so we can move on to the rest of the matmul lowering work.

## TL;DR — the root cause

The Fix #3 walkback (in `compiler/src/merlin/Dialect/Gemmini/Transforms/LegalizeForLLVMExport.cpp`) had a flag-conditional that **skipped applying `byte_offset` for bindings flagged `Indirect`**, on the (empirically wrong) assumption that the IREE local-task runtime pre-resolves the offset for those bindings.

Empirically: the standard memref→LLVM CPU codegen path applies the offset via the memref descriptor's offset slot regardless of the `Indirect` flag. So for a binding with `offset(%c2048) flags(Indirect)`:
- the **Gemmini MVOUT** lowering was emitting an address of `binding_ptrs[2] + 0` (offset skipped)
- the **next dispatch's CPU codegen** was reading the matmul result at `binding_ptrs[0] + 2048` (offset applied)

The writer and reader disagreed on the address by exactly `byte_offset = 2048`. The matmul result was silently dropped. The reader instead picked up zeros from the `linalg.fill` that pre-initializes the output tensor — so the final output was just `0 * scale + bias = bias`, with the matmul ignored.

The fix is a one-line revert to "always apply": drop the Indirect-skip, always return `subspan.getByteOffset()`.

## What we tried first (and why none of it moved the on-board hash)

The investigation surfaced an unusually long list of "obvious" fixes that all proved orthogonal to the actual bug. Each one was IR-verified, on-board-tested, and produced **zero change** in the four output hashes. Documenting them here because the trail of dead ends is the most useful part of the story.

### 0. The premise that turned out to be wrong

The handover doc framed the situation as "scalar baseline gives the correct value (`16,777,216` for the linear1 matmul i32); Gemmini produces a different broken value (`-9937`)". This was based on bcontent dumps of `dispatch_state->binding_ptrs[i]` at offset `+0`.

It turned out the bcontent was looking at the wrong slot. d16's actual matmul write address (with Fix #3's skip-Indirect path) was `binding_ptrs[2] + 0`, while the bytes `00 00 00 01 ...` at that slot were LEFTOVER from d15's quantized-i8-activation write, not a matmul result. d16 wrote at `+0`, then `linalg.fill` wrote 0 at `+2048` (the codegen offset), then d17 read from `+2048` — getting the linalg.fill zero, not the matmul.

So both "scalar's 16M" and "gemmini's -9937" were misnomers throughout the investigation. The actual matmul result was being produced correctly all along; it just wasn't reaching the reader.

### 1. Bit-63 toggle (`xor %ptr, -2^63`) on the mvin rs1

The LLVM IR for `gemmini.mvin` was emitting `add i64 %ptr_as_i64, -9223372036854775808` — toggling bit 63 of the pointer. The chipyard `gemmini.h` reference (the canonical SW stack) passes a clean 64-bit DRAM address with bit 63 clear. We hypothesized that Gemmini's RTL DMA path on FireSim was failing silently for bit-63-set addresses (TLB truncates to `paddrBits=56`).

We added an explicit `arith.AndIOp` with `0x7FFFFFFFFFFFFFFF` after the index-cast to mask bit 63. Verified at IR level the mask survived all optimizations. Verified at assembly level (`and a0, s5, s10`). Verified at runtime via an injected trace store to a fixed DRAM address (0x80300000) that captured the actual values flowing into mvin.

On-board hashes: **unchanged**. The bit-63 hypothesis was a red herring.

### 2. `noBias` accumulator-init: switching `k0 == k - 1` to `k0 == 0`

Gemmini's OS dataflow accumulator is stateful across K iterations. When `noBias=true`, the MVIN-D loop is skipped, so the accumulator is never explicitly initialized. The code at `LegalizeForLLVMExport.cpp:858` toggles an "OVERWRITE" bit on the spad address `outSpAddr` for the last K-tile (`k0 == k - 1`) when `noBias` is set. An explore agent suggested this should be `k0 == 0` (toggle on the FIRST K-tile so the accumulator gets zeroed before accumulation).

We changed it. Hashes: **unchanged**. Reverted: the chipyard `sp_tiled_matmul_os` reference (`gemmini.h:508`) uses `k == K-1`, not `k == 0`. The WS path at `gemmini.h:638` uses `k == 0`, but that's a different dataflow.

### 3. D-matrix sized buffer + `linalg.fill` zero-init

`LowerTileToISA.cpp` was allocating an `i32 memref<0x0xi32>` (a 0-byte stack alloca) for the D operand of `gemmini.tile_matmul`. Combined with `noBias=true`, this meant MVIN-D was skipped and the accumulator picked up whatever uninitialized stack memory was at D's alloca address.

We changed the allocation to `memref<16x16xi32>` (1024 bytes) and added a `linalg::FillOp` to zero-init it. With a non-zero-dim D shape, the `noBias` detection (which scans for any `0` in `getShape()`) flipped to `false`. The MVIN-D loop then fired and loaded the 1024 explicit zeros into the accumulator's `(0,0)` slot via the standard MVIN.

IR confirmed: `%18 = alloca i32, i64 256` (256 i32 = 1024 bytes) plus a nested zero-init loop plus an explicit `mvin(%42, ...)` for D. Hashes: **unchanged**.

### 4. Coherency `fence rw,rw` before MVIN-D

The CPU's zero-init stores into the D stack buffer might still be in the L1 D-cache when Gemmini's DMA fires for MVIN-D, in which case the DMA would read stale (uninitialized) memory at the L2/DRAM coherency point. We added `insertFence(loc, rewriter)` (which emits `LLVM::FenceOp` with seq_cst ordering) right before the MVIN-D loop in `spTiledMatmulOs`.

The existing `embedded_elf_loader.c` already had a `fence rw,rw` before the dispatch ELF call (separate, covers the runtime's binding-ptrs setup), but not before MVIN-D inside the dispatch.

Hashes: **unchanged**.

### 5. Poison tests on binding 1 contents

We added a runtime hook in `embedded_elf_loader.c` that overwrites bytes of binding 1 with `0x55` right before dispatch_16 / dispatch_18 fire. Three rounds:

- Poison `[0..2048)`: Gemmini should pick up garbage at base+0 if Fix #3 was bypassed → **hashes unchanged** → wasn't reading from base+0
- Poison `[4864..6912)` (the linear1.weight region): if Gemmini was reading from base+4864 (Fix #3 correct), poisoning would change the matmul → **hashes unchanged** → wasn't reading from base+4864 either
- Poison the entire 313984-byte binding 1: **hashes unchanged** → Gemmini wasn't reading binding 1 at all

This was the loudest negative result of the entire investigation. It's only obvious in retrospect, given the actual root cause: the matmul DOES read binding 1 correctly. Its output is just routed to an address that the next dispatch never reads from, so the final hashes are constant regardless of any matmul input changes.

### 6. Chipyard Scala RTL audit

An Explore agent walked `chipyard/generators/gemmini/src/main/scala/.../LoadController.scala` and `DMA.scala`. The DMA path takes rs1 through a TLB which truncates to 56 paddr bits. Bits 56–63 are undefined when set. The agent concluded "the RTL does NOT cleanly mask bit 63" — which (we now know) was unrelated to our actual bug, but reinforced the bit-63 hypothesis temporarily.

### 7. Force F#1 (force `Indirect` on all bindings)

Considered but never implemented (would have required adding a post-global-opt pass to flip `subspan flags` to include `Indirect`). Killed when we realized the Indirect-skip itself was the bug — flipping every binding to Indirect would have made the situation worse, not better.

## What actually worked: the bprobe trace

The thing that finally broke the case was extending the existing `[bprobe]` instrumentation in `iree_bar/runtime/src/iree/hal/local/loaders/embedded_elf_loader.c` to dump 16 bytes at multiple fixed offsets (`0, 2048, 2816, 4096, 4864, 6144, 18176`) of each binding for ordinals 16, 17, 18, 19 — both BEFORE and AFTER each dispatch ran.

The smoking gun appeared once we compared before-vs-after at the granularity of which bytes changed:

```
# d16 binding[2] (the output) BEFORE d16 ran:
[bprobe] o=16 i=2 +0    bytes=00 00 00 01 ...
[bprobe] o=16 i=2 +2048 bytes=08 0b 03 03 ff 00 02 02 ...

# d17 binding[0] (the same buffer, AFTER d16 ran):
[bprobe] o=17 i=0 +0    bytes=2f d9 ff ff ...     # ← CHANGED: MVOUT wrote here
[bprobe] o=17 i=0 +2048 bytes=00 00 00 00 ff 00 02 02 ...  # ← CHANGED first 4: linalg.fill zero here
```

Two writes from dispatch_16: the matmul MVOUT at `binding_ptrs[2] + 0`, and the `linalg.fill` at `binding_ptrs[2] + 2048`. The next dispatch's codegen GEP reads at `binding_ptrs[0] + 2048` (= `+ 512 * sizeof(i32)`). Mismatch of exactly `byte_offset = 2048` between the writer's address and the reader's address.

That made the fix obvious: the Gemmini lowering needs to add `byte_offset` to the C operand of MVOUT (and the C operand of PRELOAD/COMPUTE for completeness), exactly as the standard CPU codegen path does. The 2026-05-20 Indirect-skip was the wrong condition.

After the patch (always-apply `byte_offset` in `walkBackToSubspanByteOffset`):

```
# d17 binding[0] AFTER d16 ran:
[bprobe] o=17 i=0 +0    bytes=00 00 00 01 ...      # ← UNCHANGED leftover from d15
[bprobe] o=17 i=0 +2048 bytes=2f d9 ff ff ff 00 02 02 ...  # ← CHANGED: MVOUT now writes here
```

The matmul result is at +2048 where the reader will look. Hashes match scalar bit-perfectly:

```
out[0] steer                       hash=0xa571997617299fca  ✅
out[1] collision                   hash=0x3afaeb1ef0620d61  ✅
out[2] steer_QuantizeLinear_Input  hash=0x665a3c1665127f1d  ✅
out[3] linear_1                    hash=0xe61bf1d44757bd2c  ✅
```

## Implementation Changes

**Critical (the actual fix):**

`compiler/src/merlin/Dialect/Gemmini/Transforms/LegalizeForLLVMExport.cpp` `walkBackToSubspanByteOffset` — removed the `Indirect`-flag conditional. Always returns `subspan.getByteOffset()`.

**Side-effect changes (kept for defensive correctness; do not depend on the matmul fix):**

- `LegalizeForLLVMExport.cpp` (~line 1765): explicit `arith::AndIOp` with `0x7FFFFFFFFFFFFFFF` after each `IndexCastOp` to clear bit 63 of mvin/mvout addresses. Belt-and-suspenders against future RTL DMA issues; no on-board effect in the current build but harmless.
- `LegalizeForLLVMExport.cpp` (~line 765): `insertFence(loc, rewriter)` (LLVM seq_cst fence) before the MVIN-D loop. Closes the in-dispatch coherency hole between the stack-D zero-init and the MVIN that reads it.
- `LowerTileToISA.cpp` (lines 88-110 and 181+): D operand allocation upgraded from `memref<0x0xi32>` to `memref<16x16xi32>` with `linalg::FillOp` zero-init. With a non-zero-dim shape, `noBias` detection (LegalizeForLLVMExport.cpp:965) flips to `false` and the MVIN-D loop fires, properly zeroing the accumulator.

**Diagnostic-only (can be removed at leisure):**

- Trace stores in `LegalizeForLLVMExport.cpp` (~line 1785) that volatile-store A/B/C/D address values to fixed DRAM at `0x80300000+` so the runtime can dump them per-dispatch.
- Extended `[bprobe]` block in `iree_bar/runtime/src/iree/hal/local/loaders/embedded_elf_loader.c` that dumps bytes at 7 probe offsets for ordinals 16/17/18/19, plus a `[mtrace]` block that reads the compiler-emitted trace region after each dispatch.

## What Worked

- Per-dispatch byte-level diff of `dispatch_state->binding_ptrs[i]` at multiple offsets, BEFORE and AFTER each dispatch ran. This was the only instrumentation that surfaced a writer/reader address mismatch at the right granularity.
- Cross-checking against chipyard's `sp_tiled_matmul_os` for the OS-path lowering. Confirmed several other "differences" were red herrings (preload param order, k0 trigger condition).
- Comparing the working `mlp_wide` codegen.ll (which uses WS dataflow, `mvin2` for B, and writes MVOUT into a stack alloca that's then software-dequantized) against dronet's broken codegen.ll (OS dataflow, `mvin` only, writes MVOUT directly into the output binding). The structural difference pointed at the OS path's accumulator init, which led us to the D-buffer fix. The D-buffer fix didn't move the hash but is independently correct.
- Hand-computing the matmul in numpy against `linear1.weight` + ONNX's quantized relu_6 activations. Expected i32 was `-5019`; on-board both scalar and gemmini agree at `-9937`. The `~5000` gap is a separate, lower-priority IREE QDQ-fold semantics divergence (the activations entering the FC differ between ONNX runtime and IREE compile). Not Bug A.

## What Did Not Work

- The bit-63 mask, the `k0 == 0` swap, the D zero-init, the coherency fence, and three rounds of poison-binding-1 tests all changed nothing on-board. In retrospect this was diagnostic: an invariant on-board output regardless of input changes is the signature of "the writer's output is going to a place the reader doesn't read from."
- Trying to disable the Indirect-skip earlier in the investigation (handover doc Phase 6, before the 2026-05-20 patch) reportedly caused "different broken hashes" — which the doc interpreted as "double offset" and motivated Phase 7's Indirect-skip. But the same change now produces correct hashes. Either the runtime behavior actually changed between Phase 6 and 2026-05-21, or Phase 6's results were misinterpreted at the time.
- Reading IREE upstream's `MemRefToLLVM` and HAL-binding-subspan lowering passes for evidence of double-offset application. An Explore agent reported a plausible chain involving `PtrToIntOp` + signed `IndexCastOp` that could double the offset. But the empirical fix is just "always apply, don't try to second-guess what the runtime did" — the IREE plumbing is consistent with itself, and the Gemmini lowering needs to match its convention.

## Debugging Notes

- `[bprobe]` output is a one-pass diagnostic; once you've localized the writer-vs-reader mismatch, the surrounding instrumentation (trace stores at 0x80300000, sentinel poisoning, mtrace block in the loader) is debt to clean up. Recommend leaving the simple `[bcontent]` first-16-bytes dump in place — it's been useful in multiple incidents — but the extended probe-offsets block can come out before the next FireSim release build.
- The 2026-05-19 walkback approach (driving the address from the `hal.interface.binding.subspan` operand) is structurally the right design. The bug was just the flag-conditional, not the walkback machinery.
- If you ever need to chase a similar bug again, the framework is: (a) find an invariant in the on-board output that survives compiler changes you expect to matter; (b) instrument upstream and downstream of the suspect dispatch to dump bytes at every offset that could plausibly be where a writer or reader lands; (c) diff before-vs-after each dispatch run and look for "changed bytes appear at one offset, were-supposed-to-be-read-from a different offset". The Indirect-binding offset class of bug has this fingerprint.

## Test Coverage and Commands

```bash
# Rebuild compiler with the fix
CMAKE_BUILD_PARALLEL_LEVEL=8 ./merlin build --profile gemmini

# Compile dronet for both backends
./merlin compile \
  build/compiled_models/dronet/firesim_shuttle_gemmini_Gemmini_dronet.q.int8.with_intermediate/dronet.q.int8.with_intermediate.mlir \
  --target firesim_shuttle_gemmini --hw Gemmini \
  --output-dir build/compiled_models/dronet_with_intermediate/firesim/gemmini

./merlin compile \
  build/compiled_models/dronet/firesim_shuttle_gemmini_Gemmini_dronet.q.int8.with_intermediate/dronet.q.int8.with_intermediate.mlir \
  --target firesim_shuttle --hw scalar \
  --output-dir build/compiled_models/dronet_with_intermediate/firesim/scalar

# Stage VMFBs at the paths run_hetero.sh expects, then:
benchmarks/firesim_shuttle/run_hetero.sh dronet_with_intermediate gemmini gemmini 1
benchmarks/firesim_shuttle/run_hetero.sh dronet_with_intermediate scalar scalar 1

# All 4 hashes must match across the two runs.
```

## Follow-Up Tasks

1. **Validate the fix on the other Gemmini-targeted models.** dronet.q.int8 (without the intermediate-outputs variant) and yolov8_nano are both in `firesim_shuttle_gemmini.yaml`'s preprocessing list; they should also bit-perfectly match scalar baseline after this fix. (FireSim runs in flight as of writing.)
2. **Re-verify mlp_wide × Gemmini × FireSim.** The handover doc's hash 0xbb3076f6865e2266 is from before any of this work; it would be worth re-establishing the current hash against scalar with this compiler.
3. **Clean up the diagnostic-only edits.** Trace stores at 0x80300000 in `LegalizeForLLVMExport.cpp`, extended `[bprobe]` block in `embedded_elf_loader.c`, and the `[mtrace]` reader.
4. **Investigate the IREE-vs-ONNX i32 divergence.** Both scalar and Gemmini agree at `-9937` for dronet's linear1 matmul; numpy hand-computation using ONNX's quantize-output activations gives `-5019`. The ~5000 gap is in IREE's QDQ-fold semantics (e.g., the activations entering the FC matmul differ between ONNX runtime and IREE compile by a small amount due to QDQ rounding). Not a Gemmini bug, but worth tracking.
5. **Audit the rest of the gemmini lowering for symmetric "Indirect" flag-conditionals.** If any other site in `LegalizeForLLVMExport.cpp` decides to skip applying the byte_offset based on the Indirect flag, it has the same latent bug.

## Retrospective — why this took two days

The fix is one line of C++. The whole investigation arc, with a half-dozen FireSim cycles, two compiler-pipeline reverts, and four hypothesis-then-disproof loops, was avoidable in roughly three places.

### The seven mistakes (in order of how much time they cost)

**1. Trusting the handover doc's framing as ground truth.** The doc said: "scalar reports i32 = 16,777,216 = 0x01000000 = 2^24. Gemmini reports i32 = -9937. Gemmini is broken; the goal is to make Gemmini match scalar." We spent two days trying to make Gemmini match scalar. Neither value was actually the matmul output: 16,777,216 is leftover bytes from d15's quantize output, sitting at the bcontent dump location. The real matmul result was at +2048 — never observed until the last hour. A 5-minute numpy hand-computation on the actual ONNX-quantized A bytes and linear1.weight would have given `-5019` and immediately raised the question "neither scalar nor gemmini is correct — what are we actually comparing?".

The lesson: **do a ground-truth hand-computation BEFORE trusting "X is the baseline, Y diverges from it." A numpy line costs nothing.**

**2. Trusting `bcontent` to dump from "the right address."** `[bcontent]` prints 16 bytes at `dispatch_state->binding_ptrs[i] + 0`. We treated this as showing "the contents of the binding the dispatch reads/writes." It isn't — it shows the contents at the runtime-resolved binding pointer's first 16 bytes. For Indirect bindings with non-zero byte_offset, the dispatch's actual read/write addresses are at `binding_ptrs[i] + (codegen GEP)`, not `binding_ptrs[i] + 0`. The bytes we kept staring at had nothing to do with the matmul.

The lesson: **the instrumentation has its own model of "where the dispatch operates." Verify that model before trusting any diff.** Concretely, before running the next FireSim cycle, look at the codegen.ll for the dispatch and confirm which offset the GEP applies. If the GEP adds +2048 to the binding ptr, your bcontent at offset +0 isn't showing you what the dispatch sees.

**3. Misinterpreting the poison tests.** Three rounds of binding-1 poison (offset 0, weight region 4864, and the entire 313984-byte buffer) ALL produced zero change in the output hashes. We hypothesized: "Gemmini's mvin is silently failing on the FireSim RTL — bit 63 isn't being masked." That's an *explanation*; it's not the *simplest* explanation. The simplest one was: "the writer's output is being thrown away before the reader picks it up; the contents of binding 1 don't matter because the matmul result doesn't propagate."

The lesson: **invariance under perturbation is the strongest signal in debugging. If poisoning every input has no effect, the output isn't a function of those inputs — the dispatch's output is being lost, not corrupted.** "Hardware silent failure" is too convenient an explanation; prefer the explanation that says "data is being correctly computed but routed to the wrong place."

**4. Conflating "the assembly matches chipyard reference" with "the address arithmetic is correct."** An Explore agent walked our `spTiledMatmulOs` lowering line-by-line against chipyard's `sp_tiled_matmul_os` reference and reported "matches chipyard reference." This was true at the level of "the gemmini instruction stream is correct." It was orthogonal to the actual bug, which was at the *boundary* between the Gemmini lowering and the standard CPU codegen — specifically, whether both apply the binding's byte_offset consistently. The chipyard reference doesn't speak to that boundary at all (it's pure C-style address arithmetic; the IREE memref/HAL Indirect flag is an IREE-specific construct).

The lesson: **when you've confirmed "subsystem X is internally consistent against its own reference," that does NOT confirm "subsystem X is consistent with the surrounding subsystem Y." The bug class "writer/reader disagree on address" only shows up when you instrument both sides together.**

**5. Chasing IR-level fixes instead of semantic-level fixes.** We made four IR-verified compiler changes (bit-63 mask, k0 swap, sized D buffer, coherency fence) before the right fix. Each was justified by a plausible local reading of the LL. None changed the on-board hash. In retrospect, the right move after the second failed IR-level fix should have been to step back and ask "is my model of *which bytes the dispatch reads/writes* even correct?" Instead we doubled down on local plausibility.

The lesson: **two failed-but-IR-verified fixes is a sign that your model of what bytes flow where is broken. Don't escalate to fix #3 — escalate to instrumenting downstream of the dispatch.**

**6. Asymmetric trust in the runtime vs the codegen.** The handover doc's Indirect-skip patch had a comment: "the IREE local-task scheduler has already added the byte_offset to the binding pointer before invoking the dispatch — re-adding it here would double-offset the pointer." We trusted that comment for two full days. The first patch was based on a Phase-6 experiment that "produced different broken hashes" when offset was always applied. We never re-ran that experiment to see what the hashes actually were. They could have been correct — they probably *were* correct, and were misinterpreted at the time as "different broken" because we were comparing against the wrong baseline.

The lesson: **don't trust a code comment that asserts what an upstream subsystem does. Verify with a one-off instrumentation. The runtime-side question "does the runtime resolve the Indirect offset?" is answerable in ~10 lines of printf added to the local-task driver. We never did it.**

**7. Treating "the FireSim run is the test" instead of "the FireSim run is the slow validator."** Each FireSim cycle was ~30 minutes of wall time. We did six of them with insufficient instrumentation to localize the actual problem, when the right move was 2-3 cycles with much heavier instrumentation. The bprobe extension to multiple offsets was the diagnostic that finally cracked it — and it could have been added on day one.

The lesson: **when each iteration of the outer debug loop is expensive, invest in better instrumentation BEFORE running. The cost of one extra `fprintf` in the loader is zero; the cost of a misdirected FireSim cycle is 30 minutes.**

### The 30-minute version of how this should have gone

```
00:00 — Read handover doc. Note: "scalar i32 = 16,777,216; gemmini i32 = -9937."
00:05 — Compute expected matmul i32 in numpy using onnx quantized A and linear1.weight.
       → expected ≈ -5019. Both scalar (16M) and gemmini (-9937) are WRONG.
       → Stop. Reframe: neither backend is the baseline.
00:10 — Extend [bprobe] to dump at multiple offsets (0, 2048, 4096) for the matmul dispatch
       AND the dispatch immediately downstream (d17 reads d16's output).
00:15 — One FireSim cycle.
00:45 — Read the uartlog. Observe: d16 writes at +0; d17 reads at +2048. Mismatch.
00:50 — Look at the codegen.ll: d17's `getelementptr i32, ptr %6, i64 512` confirms +2048.
       Look at the Gemmini lowering: walkBackToSubspanByteOffset skips Indirect.
       That's the bug.
00:55 — Change walkBackToSubspanByteOffset to always apply byte_offset.
01:00 — Recompile, FireSim cycle.
01:30 — Hashes match scalar. Done.
```

The two days happened because we never reframed the question "which value is correct?" with a ground-truth computation, and because the bcontent diagnostic was lying about where the dispatch operates.

### Process recommendations for next time

1. **Ground truth FIRST.** Before believing any "X is broken, Y is correct" framing in a handover doc, compute the expected output in the simplest available way (numpy, hand math, pencil and paper). If both X and Y disagree with ground truth, the framing is broken, not the implementation.

2. **The poison test is a structured probe — treat its results as conclusive.** If the output doesn't change when the input is poisoned, the dispatch isn't reading the input. Stop hypothesizing exotic hardware failures.

3. **Diagnose at writer-vs-reader pairs, not single-side.** Single-side instrumentation (look at one binding's bcontent) almost always undersamples the problem. Always instrument the writer and the reader's-corresponding-binding together, and diff before-vs-after the writer's dispatch.

4. **Question your debug tool's model of the system.** `[bcontent]` dumps from a particular address; before trusting it, verify that address matches the dispatch's actual operating address. If the codegen adds a GEP offset, the dispatch is operating at a different address than bcontent reports.

5. **Track "what costs each iteration" explicitly.** When iteration cost is high (FireSim, hardware-in-the-loop, multi-hour reproductions), the budget should be heavily skewed toward better diagnostics per iteration, not more iterations.

6. **When two IR-verified fixes both fail to change behavior, escalate to "is my mental model of what bytes flow where wrong?"** That escalation is the cheapest, most-likely-fruitful next step.

7. **Don't trust comments asserting runtime behavior. Verify with a printf.** "The runtime already does X" is a claim, not a fact. Especially in mid-investigation code where the comment was written by someone trying to explain a fix that may or may not have been correct.

## Addendum (2026-05-22): partial coverage on non-intermediate models

The walkback-offset fix is **definitively correct for the FC matmul codegen path** (small output, `tensor<1x1xi32>`, direct MVOUT-to-binding pattern). Validated on:
- `dronet.q.int8.with_intermediate` × Gemmini × FireSim — all 4 hashes match scalar baseline ✅

However, `dronet.q.int8` (non-intermediate, full-conv-stack-through-Gemmini-via-im2col) × Gemmini × FireSim **still diverges** from its scalar baseline:
- scalar steer = `0xa571997617299fca` (matches with_intermediate scalar — correct)
- gemmini steer = `0x4c03397622201e9e` (does not match)

The conv-stack matmuls (dispatch_2 / 7 / 10 / 16 / 19 / 25 / 28 in non-intermediate dronet) use a different codegen pattern: each emits ~390 MVOUTs into a `alloca i32, i64 100352` stack buffer (400 KB), followed by a SW dequantize/requantize loop that copies the matmul i32 to the binding's i8 output. The FC matmul pattern (single MVOUT direct to a `tensor<1x1xi32>` binding) is simpler and is what the offset fix corrects. The conv-stack pattern's bug — if any — lives in a different region of the codegen (likely the MVOUT-tile-stride or the SW dequant loop's address arithmetic), distinct from the binding-offset class.

yolov8n almost certainly inherits the same problem since it has substantially more conv-stack matmuls and the same im2col preprocessing.

**Status**: original Bug A (FC matmul output silently dropped due to writer/reader offset mismatch) is closed. The non-intermediate / yolov8 case is a separate, follow-on bug that should be tracked independently.
