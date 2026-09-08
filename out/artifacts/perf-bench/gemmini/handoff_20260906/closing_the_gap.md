# Closing the ResNet-50 gap on Gemmini: experiment design, capsules, tests, optimizations

**Audience:** a fresh session picking this up with no prior context.
**Status:** written 2026-09-06; hardware progress updated 2026-09-08. Every number here is either measured and attributed, or explicitly
marked as an estimate. Where an earlier version of this analysis was wrong, the correction is kept
rather than the original, because the wrong versions are the ones that cost time.

**Completion:** the deterministic infrastructure and shared compiler candidates described here are
implemented. Target-neutral scalar conversion and affine-im2col specialization are now exact,
full-model, hardware-promoted Phase-2 wins: q530's 1,704,064,223 cycles fell to q534's
1,537,416,019 and then q535's 1,316,619,699, a cumulative 1.29427x speedup. The end-to-end plan is
not finished: exact epilogue/residual boundary deletion, compatible narrow-layout propagation,
warm-safe native convolution selection, and transformer placement remain open. The authoritative
current state is §29 of `phase2_handoff.md` and `PHASE2_CURRENT_CHECKPOINT_20260908.md`.

Read §1 and §2 first. They are the difference between optimising the thing that matters and
optimising what happens to be easy to measure — which is the mistake this document exists to stop
you repeating.

---

## 1. The situation, in measured numbers

### 1.1 Where we stand against other frameworks

Measured on the same U250 FireSim hardware configuration, ResNet-50:

```
Merlin champion   1.00x
Voyager           5.81x faster than Merlin
TVM-Gemmini      59.60x faster than Merlin
AutoComp        102.66x faster than Merlin
```

**This table is provided context, not something reproduced locally.** No artifact in this repository
contains the 102.66x, the 5.81x or the 59.60x. Treat it as the goal to beat, not as a verified
measurement, and do not cite it as ours without regenerating it.

### 1.2 The two artifacts being compared, and why the comparison is not what it looks like

| Field | Older "2B" run | Current "11B" run |
|---|---:|---|
| FireSim request | `20260904T070237Z-13628` | `20260906T014122Z-1256426` |
| FireSim terminal cycles | 2,353,594,112 | 20,187,983,552 |
| Workload/model counter | 98,316,713 | 11,183,959,730 |
| Frontend | existing `resnet50.c` | canonical PyTorch capture through Merlin/MLIR |
| Gemmini path | direct calls to prebuilt `tiled_*_auto` | compiler-generated OOT device kernels + scalar MLIR |
| Terminal status | PASS / FireSim PASSED | Spike numerical pass / FireSim PASSED |

On comparable FireSim terminal counters the current artifact is **8.58x slower**. Comparing the
current model counter to the older terminal counter gives 4.75x, but that **mixes counter scopes and
is not the preferred comparison** — do not use it.

**The older artifact is a Gemmini library baseline, not a compiler result.** Its source audit records
**124 accelerator calls**:

- 18 `tiled_conv_auto`
- 2 `tiled_conv_downsample`
- 87 `tiled_matmul_nn_auto`
- 16 `tiled_resadd_auto`
- 1 `tiled_global_average_auto`

Those routines supply prewritten tiling, loop ordering, DMA scheduling, accumulator management and
fusion decisions. Its build manifest begins from an already-generated C file and invokes RISC-V GCC
with `-O2 -ffast-math`. **So the 8.58x does not establish that the older Merlin generated better
schedules — it did not generate them at all.**

The current OOT path lowers the captured model automatically and emits 21 distinct Gemmini device
signatures across 54 on-mesh calls, leaving substantial work in scalar/generic MLIR. Its inclusive
profile contains 2,988 marks and attributes **9,689,674,051 cycles to `linalg.generic`**. Other
material categories: batch normalisation, im2col/matmul convolution lowering, tensor insertion,
transpose, pooling, residual. **These counters overlap hierarchically and must never be summed as
disjoint totals.**

### 1.3 The measured composition of the 11.18B

- `linalg.generic`: **86.6%** (9.69B cycles)
- on-mesh contractions: **5.9%**

Making the accelerator infinitely fast removes under 6% of runtime.

### 1.4 Why the older program is faster — the mechanism

1. Direct optimised Gemmini convolution vs generic `convolution_im2col_matmul` plus scalar
   surrounding operations.
2. Prearranged C buffers/layouts vs generated memref handling, packing, insertion, copies, device
   shims.
3. Fused library epilogues vs separately lowered batch norm, quantisation, activation, residual and
   layout operations.
4. A compact static call graph (124 calls) vs thousands of profiled generic operations and repeated
   device-resource transitions.
5. No MLIR memref descriptors, no generic shape handling, no device-dispatch layer, no per-operation
   profiling.
6. It avoids repeated allocation, copy, pack and evict traffic.

### 1.5 Specific defects already identified in the current path

These are reported findings; each should be confirmed before being treated as fixed.

- **Fences.** A capacity-safe schedule issued a CPU/Gemmini fence after **every output tile** —
  **4,096 fences for a single 1024x1024 QK slice**. With 12 batch slices and repeated attention
  layers that is tens of thousands of blocking round trips, and FPGA execution appeared stalled. The
  fix keeps arithmetic, tiling, addresses and commands identical and batches synchronisation: one
  fence before the kernel, Gemmini's reservation station orders intervening scratchpad/accumulator
  hazards, one load-bearing fence at the end so output DMA completes before the CPU reads. A second
  redundant terminal fence in the LLVM emitter was removed. **4,098 fences -> 2**, same numerics.
- **Batch matmul was illegal by contract.** The target contract declared `ranks: [2, 4]`, which
  Merlin reads as allowed contraction output ranks. A batched matmul has output rank 3
  (`B x M x K` @ `B x K x N` -> `B x M x N`), so the legality check rejected **every** batch matmul
  before it reached the device rewrite — while the rewrite already supported `(B,M,N,K)`, the device
  builder already produced the `(M,N,K)` kernel, and the shim already looped over `B`. The machinery
  existed; the contract prevented its use.
- **False residency.** Large matrix shapes address memory outside Gemmini's scratchpad/accumulator,
  and a schedule that claims to reuse resident weights is actually reloading them.
- **Crash on very large kernels** (e.g. 1k x 700 x 700). Low priority.

### 1.6 The reference for what good looks like

An Exo-generated kernel for `exo_mm32_16x8208x16` (i8 A[16,8208] @ B[8208,16] -> i32 C[16,16]):

- **Five `config_*` calls issued ONCE**, before any loop: `config_st_acc_i32`, `config_matmul` (WS),
  `config_ld_i8_id2`, `config_ld_i8_id1`, `config_zero`.
- The accumulator tile is allocated once (`gemm_acc_malloc`), zeroed once via `mvin` of 0, and stays
  **resident across all 513 k-steps**.
- The k-loop body is exactly: `mvin2` (A panel), `mvin3` (B panel), `preload`, `compute_preloaded`.
- **One `mvout`** at the end, then free.
- **No fences. No per-tile reconfiguration. No reallocation.**

That single artifact is simultaneously the synchronisation claim, the residency claim, and a
command-encoding-efficiency axis the current corpus does not have.

---

## 2. What the current experiment can and cannot measure

### 2.1 The perf corpus is 38 single-operation on-mesh capsules

| family | declared lever | members | matches a measured defect? |
|---|---|---:|---|
| PQ | `redundant_synchronization_removal` | 6 | **yes** — the fence defect |
| PR | `operand_residency` | 6 | **yes** — false residency |
| PC | `dma_issue_before_wait` | 2 | yes — CPU/accelerator overlap |
| PL | `cross_regime_amortization` | 4 | partly |
| PK | `reduction_depth` | 4 | the k-axis bounding utilisation |
| PM | `parallel_extents` | 16 | **no** |

**Nothing in this corpus can see the 9.69B `linalg.generic` cycles**, because every member is a
single on-mesh operation. Absent entirely: layout conversion, repacking, `tensor.insert_slice`,
memref descriptor handling, device-dispatch shims, epilogue fusion across a layer, allocation /
copy / pack / evict traffic, and CPU/accelerator round trips at layer scale.

### 2.2 A campaign that proved the point

The first phase-2 campaign was scoped to **PM** — chosen because it was the largest single-claim
cohort (16 of 38). That was the wrong criterion, and the result shows why. Measured over 16 members:

```
PM00   4,096 MACs -> 14.3 MACs/cyc ( 5.6% of 256 peak)
PM05  16,384      -> 31.6          (12.3%)
PM10  36,864      -> 41.7          (16.3%)
PM15  65,536      -> 42.8          (16.7%)
TOTAL 409,600 MACs in 11,451 cycles = 35.8 MACs/cyc (14.0% of peak)
```

Every PM member holds **k = 16** (MACs = m·n·k; PM15 = 65,536 = 64·64·16). The family sweeps the two
extents that do NOT set the ceiling on a weight-stationary mesh, where utilisation is governed by how
much work each loaded weight tile amortises — i.e. by k. A two-point fit gives roughly **200 cycles
of fixed per-capsule overhead** and a **marginal rate near 49 MACs/cycle**, so even an unboundedly
large PM member tops out near 19% of peak on this schedule.

The agent moved 11,487 -> 11,469 cycles (**0.157%**), with several members regressing. That is close
to all the room the family has.

### 2.3 The attainable-rate defect is fixed

The old `attainable_total_cycles` (2,873 for PM) used one global best rate of 142.5 MACs/cycle over
97 measured points, including deep-k shapes that a k=16 PM member cannot resemble. That made
`headroom_open` impossible to close.

`perf_model.MeasuredPoint` now records the command buffer's derived reduction-depth signature.
`DevelopmentGsimFeedback` admits only seed/baseline points with the member's exact signature, writes
the matched rate and its basis into each cell, and computes the family stopping denominator as the
sum of each member's MACs divided by its own matched rate. A deep-k point is therefore excluded from
a k=16 attainable claim. The regression is in `test_achievable_ceiling.py`; the focused stage,
unmeasured-cell and ceiling suites pass 120 tests.

---

## 3. The structural change: capsule -> layer -> model

### 3.1 Why the unit must change

The 8.58x lives **between** operations (layout conversion, `insert_slice`, packing, memref
descriptors, dispatch shims) and **across** them (epilogue fusion). No sum of single-operation
capsules contains it. The unit of measurement has to become an interval.

### 3.2 Feasibility — what is expressible TODAY (verified)

- `merlin/python/merlin/targetgen/capsule_golden.py` implements **`im2col`** (line 46) and
  **`_apply_epilogue`** (line 108). The epilogue stages it implements are exactly `EPILOGUE_STAGES`:
  bias stages, `requant`, `acc_scale`, `relu`, `maxpool`.
- A **synthetic conv with scalar scale, per-column accumulator bias, ReLU and maxpool** is therefore
  expressible as one device commit. This is useful for testing the commit implementation.
- **Correction: the captured ResNet batch norm is not that epilogue.** It is a per-channel f32
  subtract / rsqrt / multiply / add chain after conversion from the accumulator. Treating it as the
  existing scalar `acc_scale` plus i32 accumulator `bias_add` changes the numeric contract. A real
  fold requires either offline weight/bias folding followed by re-quantization under an explicit
  accuracy policy, or a per-channel-scale ABI extension.
- `produce_gsim_certificate.derive_workload` has a **general fallback** for operations it does not
  know: the workload identity derives from exact named input roles/shapes plus semantics, with "no
  guessed shape algebra". So a new operation kind can still be certified.
- Capsule `kind: model_slice` already exists, and capsules already carry an `interface_mlir` file.

**Not expressible in the commit ABI today:** the **residual add**. `EPILOGUE_STAGES`
(`merlin/python/merlin/runtime/commandbuffer.py:76`) is
`(bias_add, bias, requant, acc_scale, relu, maxpool)` — a residual needs a second full tensor
operand, not a per-column vector. Closing this is a prerequisite for a true residual-block capsule.

**Not measurable in the ordinary split capsule path:** the compiler's host/device layer interval. That
path runs the accelerator lane under the target oracle while host work runs in-process on this
workstation; subtracting them would compare different clocks. The new explicit whole-program ABI (§20)
can instead call one fused target ELF under one counter, but `PB` remains unmeasured until its paired
corpus and a new post-freeze compiler candidate exist. A rich device commit capsule must not be
presented as a measurement of host-work deletion.

**Existing exporter is not sufficient:** `merlin/python/merlin/targetgen/model_slice_export.py`
states each slice "reduces to a single weight-stationary matmul" — MLP linears, attention
projections, QK^T, PV, no softmax. It must be extended or paralleled for conv-with-epilogue slices.

### 3.3 The hard sizing constraint — measure before you design

**Measured GSIM throughput: ~193 cycles/second aggregate**, from the live campaign's own receipts —
32 runs (16 members x 2 arms) totalling ~22,974 cycles in a **119 s** median feedback call. Per-run
process startup dominates for small capsules (~3.7 s per run).

Consequences, and this is the constraint that shapes the whole design:

- Whole ResNet-50 is **11.18B cycles**. On GSIM that is **not remotely feasible** — it is FireSim-only.
- A real ResNet 3x3 conv, 64->64 channels at 56x56, is ~115.6M MACs -> roughly 2.9M cycles ->
  **~4 hours of GSIM per run**. Also infeasible as an iteration loop.
- **Target 10k-50k cycles per layer capsule.** At ~193 cyc/s aggregate a 5-member family with two
  arms is roughly 10-25 minutes per feedback sweep — slow but workable inside an agent loop.

**Therefore layer capsules must use REDUCED shapes.** This is sound because what they measure are
*ratios*, not absolute sizes: im2col materialisation factor, fused vs unfused epilogue traffic,
layout conversion overhead, commands per unit work. A conv3x3 with Cin=Cout=16 at 16x16 spatial is
~589k MACs -> ~15k cycles, which exercises every mechanism at a tractable size.

Full-size shapes belong on FireSim, run periodically, never in the loop.

---

## 4. Proposed capsule families

### 4.1 Layer objectives and the current execution boundary

The campaign unit decision is **a residual block or other complete layer interval**, with PR/PQ/PK
retained as diagnostics. It is not yet honest to materialize that unit as an ordinary GSIM capsule:
the explicit same-counter ABI now exists, but the independently predeclared paired workload, hidden
variants and post-freeze compiler candidate do not.

| objective | mechanism it must expose | readiness |
|---|---|---|
| stem: conv 7x7 s2 + maxpool | im2col amplification and direct-conv selection | runner ready; paired capsule/emitter needed |
| conv + real BN + ReLU | per-channel affine semantics, fusion and layout conversion | blocked on numeric policy plus paired capsule/emitter |
| conv 1x1 projection | packing/layout without a window | runner ready; paired capsule/emitter needed |
| downsample block | stride/shape change and repacking | runner ready; paired capsule/emitter needed |
| classifier tail | global average pool + FC and rank legality | runner ready; paired capsule/emitter needed |
| residual block | several convs, second tensor operand and cross-op liveness | additionally blocked on residual ABI support |

A reduced rich-epilogue **device commit** may be added as a diagnostic, but it cannot score the
compiler interval. When materializing the paired corpus, derive reduced extents from target facts; do not
hardcode the first machine's mesh dimension.

### 4.2 Model family (FireSim-scale) — the global objective

| member | scope | oracle |
|---|---|---|
| `GM0_resnet50_int8` | whole model, real shapes | FireSim only, periodic |

This is the number that must fall. It is not an iteration loop — it is the citable claim, run at
milestones. Its inputs exist:
`out/artifacts/recaptures/resnet50_v1_5_int8_w8a8_consistent/` with weights, inputs, and
`golden_w8a8.independent.npy`.

### 4.3 Keep the micro families as diagnostics

PR (residency), PQ (synchronisation), PC (DMA overlap), PK (reduction depth) explain **why** a layer
member is slow. They are the mechanism; the layer is the objective. Do not report a micro-family win
as a model result.

---

## 5. The optimisations to land

Ordered. Sequencing is not optional: (2) before (5), or int8 makes the model **slower**.

1. **Route on semantic family, not spelling.** DONE — commit `99857ed5`. `compute_units.supports_op`
   previously string-matched `prov.op` against declared `ops: [matmul]`, so all 53 ResNet convolutions
   (tagged `convolution_im2col_matmul`) were refused to the host lane by a string comparison.
2. **Fix the quantisation/vectoriser interaction. DONE.** The quant pass can preserve a canonical
   2-D i8xi8->i32 contraction as mixed-type `linalg.matmul` through the
   `named_int8_contraction` enabler; non-canonical and batched forms fail closed to the generic form.
   The proposer does not waste a fork on the inert enabler alone: it couples it to refinements that
   act on the named operation. `test_named_int8_contraction.py` proves the 15 -> 0 defect and the
   restored 15 named operations. Full ResNet static audits now contain vector instructions.
3. **Delete the im2col intermediate.** The default conv lowering materialises the activation
   `kh*kw/(sh*sw)` times over — on ResNet conv1 a `3x7x7x1x112x112` intermediate, 1,843,968 elements,
   the activation written **12.25x** over. Above `M2M_IM2COL_MAX_ELEMS` the frontend already emits the
   true compound-affine form (`prov.conv_path=direct_contraction`) that materialises nothing, and
   `perop_blocks.CONV_ARM_FEATURE` (`conv_register_block`) tiles and vectorises exactly that form.
   DONE for deterministic validation: a lower-budget ResNet capture was regenerated at
   `out/artifacts/recaptures/resnet50_v1_5_int8_direct_20260906`; independent compiler regressions
   cover signed i8, stride 1/2 and batch 2, and whole-model host replay is byte-identical across the
   direct/base arms. Target timing remains a phase-owned measurement, not a static speed claim.
4. **Make the measured fusions real and searchable. DONE for codegen, not promoted by static data.**
   `fuse_epilogue_loops` now applies on the scalar pipeline and refuses unsafe/vector placements;
   small_llama is byte-identical at **30,488 -> 26,064 instructions (-14.5%)**. On the direct int8
   ResNet, the targeted `fuse_elementwise_post_contraction` arm is byte-identical at
   **136,657 -> 124,730 instructions (-8.73%)** with vector count flat (26,090 -> 26,080). The
   current-compiler report is `scalar_codegen_20260906/resnet_post_contraction_perop_ab_current/`;
   the 53/53 requant census and
   **44,455,936 accumulator bytes** remain the basis for ranking
   `fuse_requant_into_contraction_vec`. None is made a default without target timing.
5. **Build and run the int8 ResNet against the independent golden. PHASE-1 BLOCKED, not hand-fixed.**
   The corrected direct capture and host/RVV codegen are ready, but the certified functional sample
   is 27/28 because the frozen seed has a bias-ABI defect. The seed/submission must be corrected by
   the owning phase; mutating the frozen answer or patching a graded submission here would invalidate
   the claim. Report the eventual result against the fp32 run of the same compiler.
6. **Batch synchronisation. DONE.** The driver retires exactly once; regressions count the one
   pre-kernel and one load-bearing final fence and cover the previous per-tile failure.
7. **Fix the rank legality contract. DONE.** Rank-3 batched contractions are admitted and covered by
   the legality regressions.
8. **Hoist config out of loops. DONE.** A decoded-trace regression asserts one configuration per
   resident kernel rather than per tile.
9. **Hoist weight transposes to compile time. DONE through the safe path.** The general map fold now
   refuses a fold that makes the hot output axis strided. Offline `prepack_weight_layout` removes the
   runtime transpose instead and is bit-identical on Spike, **2,090,995 -> 1,993,040 cycles** in the
   Spike instruction model. This is not target wall time.
10. **`LOOP_CONV_WS`, the native conv FSM.** VERIFIED: `legal_funct` in
    `merlin/targets/gemmini/contracts/rtl_facts/facts.json` contains **15** and **16..21**, and
    `merlin/python/merlin/_data/contract/compute_endpoints.yaml` already declares `LOOP_CONV_WS` and
    `LOOP_CONV_WS_CONFIG_1..6` under the `loop_descriptor` role. **CORRECTION to earlier notes:**
    `CONV2D` is *not* rejected or rewritten to a host gather —
    `gemmini_codegen_mlir.py:181` lowers it "to the same im2col/resident-matmul/commit path used by
    matmul capsules". So this is a **selection problem plus an emitter**, not an ISA bring-up. It
    overlaps item 3. A tracked same-source launcher now prepares the native `ws conv` and im2col
    `ws matmul` arms, pins the source/compiler/binary hashes, warms each arm, and parses the exact
    decomposition. The prepared pair is under
    `out/artifacts/perf-bench/gemmini/handoff_20260906/loop_conv_ws_prepared/`. **FireSim execution is
    now complete:** three post-warm-up repetitions per arm measured `ws matmul / ws conv =
    10.266047x`, with zero observed cycle variance (§19). The compiler emitter and schedule choice
    remain phase-2 agent work.
11. **Stop vector scratch from growing with loop trips. DONE.** `convert-vector-to-scf` created seven
    fixed-size scratch allocations inside the classifier contraction's reduction loop. The generated
    stack grew once per iteration and faulted under a normal 8 MiB limit despite a small static frame.
    A second buffer-hoisting stage now runs immediately after vector-to-SCF conversion. Full ResNet
    host replay succeeds under the normal limit with the same output digest; the RVV object changes
    by 67 instructions. `test_late_vector_scratch_hoist.py` detects any alloca in a cyclic CFG block.

**Do NOT add a general "vectorise the generics" pass.** Measured results from
`mining/wholemodel_proposer.py`:

| lever | measured |
|---|---|
| `vectorize_non_contraction_generics` | **4.9x vector instructions, 1.28x SLOWER** |
| four-lever epilogue stack | **+17.9% instructions, +4.4% wall** |
| `fuse_requant_into_contraction_vec` | **-5.3% instructions** (+1.6% vector) |

The lever that paid **removes a pass**. The host problem is too many passes over too much data, not
too little SIMD.

**Stale-number trap:** the frequently-quoted "44.4% of int8 instructions in the scalar gather / 31.1%
in quantize+amax / 18.2% in the contraction" figures were taken on f32-activation IR with
`quantize_before_gather` OFF, and the source marks them NOT CURRENT. Re-measure; do not cite them.

---

## 6. Experiment mechanics

### 6.1 The objective and its decomposition

Score the **interval's total cycles**, and report alongside it where those cycles went: on-mesh,
`linalg.generic`, movement, synchronisation. Without the decomposition the agent optimises blind.
Every instrument exists: `op_profile`, `work_volume`, `movement_volume` (see the limitation in §8),
`decompose_corpus`, the hardware-counter observations, `trace_check`.

### 6.2 A fair baseline — this is what makes it a compiler measurement

Both arms must start from the **same captured PyTorch graph**, and either both or neither may call
`tiled_*_auto`. Otherwise you are comparing a hand-written library against a compiler and calling it
schedule quality.

Gemmini ships the instrument for this. `gemmini-rocc-tests/imagenet/resnet50.c` (present at
`/scratch2/minh/bringup_chipyard/generators/gemmini/software/gemmini-rocc-tests/`, 66 MB
`resnet50_params.h` with real weights):

- takes `os|ws|cpu` **and** `conv|matmul`;
- prints `Total / Matmul / Im2col / Conv / Pooling / Depthwise convolution / Res add / Other` cycles
  **with percentages** — directly comparable to our own attribution.

Three arms fall out of it, and they answer different questions:

| arm | what it is | answers |
|---|---|---|
| `resnet50 ws conv` | library via the hardware conv FSM | the headline gap |
| `resnet50 ws matmul` | library via im2col + `tiled_matmul_auto` — **our strategy, hand-written** | **B vs this = our compiler gap** |
| our compiler | automatic lowering | |
| `resnet50 cpu` | no accelerator | built-in control |

`conv` vs `matmul` on the library **prices `LOOP_CONV_WS` with no work on our side**.

Compare **cycles only, never outputs** across arms — Gemmini ships its own weights, ours are
recaptured. Hold `elem_t`/int8 identical on both sides.

### 6.3 Two oracle tiers, never mixed

- **GSIM** at layer scale for iteration (~193 cycles/s; size capsules accordingly).
- **FireSim** at model scale for the citable claim, run at milestones.

Label every number with which counter it is. The source report is explicit that comparing an 11B
model counter against a 2.35B FireSim terminal counter mixes scopes and is not preferred.

### 6.4 Correctness gate

Every candidate must match `golden_w8a8.independent.npy`, never the consistent golden alone. A faster
wrong layer is not a result. The certified functional emission guard already refuses candidates that
introduce trace findings, and it **works** — it caught two campaign candidates introducing
`"mode resident_reuse: 1 redundant load(s) rewrite an on-chip destination that already held that
exact source"`, which is precisely the false-residency defect from §1.5.

### 6.5 Campaign scoping

One campaign seals one claim (`perf_claim_dispatch`), so families with different analyzers cannot be
combined. Cohorts are keyed by **(declared analyzer, declared family)**: several families may share
one analyzer while that analyzer's precondition check is defined over a single family — PC, PL and PQ
all declare `perf_paired_claim`, and handing it their combined 12-member cohort is refused outright.

Validate any scope before launching, with both preflights:

```
merlin/experiments/gemmini_perf_bench/scripts/perf_cohort_preflight.py \
    --members <file> --capsule-root merlin/contract/capsules/_perf
merlin/experiments/gemmini_perf_bench/scripts/perf_corpus_preflight.py \
    <descriptor> <tuning-certificate> <members-csv>
```

---

## 7. Tests required and current status

Each is a mutation test: break it deliberately and confirm the check goes red. The standing failure
mode in this repository is a check that cannot fail reporting success — it has been recorded 14 times,
and three of those were inside an instrument built to measure honesty.

1. **Layer capsule golden agreement — BLOCKED on the paired corpus/candidate.** Do not mint a
   device-only golden and call this complete; the test must cover both lanes in the explicit
   whole-program same-counter ELF.
2. **im2col elimination — DONE.** Named and generic direct-conv paths, padded geometry, stride 1/2,
   signed i8 and batch 2 are covered; the recapture records the direct-contraction provenance.
3. **Fusion regression — DONE at compiler/host scope.** Safe scalar fusion and the targeted ResNet
   fusion both preserve output bytes; the requant census records the removable accumulator traffic.
4. **Fence count — DONE.** The emitted driver retires exactly once, with a regression over multiple
   tiles.
5. **Config count — DONE.** The resident trace regression asserts configuration is issued once.
6. **Residency — DONE at the seal path.** `trace_check` compares decoded loads with resident packs and
   already refused two false-residency campaign candidates.
7. **Rank legality — DONE.** Rank-3 admission and rewrite are regression-tested.
8. **Attainable basis — DONE.** Exact reduction-depth signatures prevent a k=16 member from using a
   deep-k rate; the cell records the matched basis.
9. **Vector instructions present — DONE.** Quantized canonical contractions retain a named form for
   the vector schedule, and the ELF audit remains fail-closed.
10. **Loop-local vector scratch — DONE.** A CFG-structural regression rejects LLVM `alloca` in a
    cyclic block after RVV lowering.

---

## 8. Instruments: what they measure, and one that lies

- `work_volume` — MACs from the command buffer.
- `movement_volume` — bytes moved. **LIMITATION, now declared in the block itself:** a resident
  operand is charged **once at its pack**, so it counts the traffic the command buffer *declares*,
  not what the emitted program *issues*. A backend that re-loads a resident tile per output tile
  produces an identical number. It therefore **cannot detect the false-residency defect** it looks
  like it prices. Use `trace_check` for that.
- `trace_check.py:244` — detects redundant loads rewriting an already-resident on-chip destination.
  This is the real residency detector, and it works.
- `decompose_corpus` — roll-up of which resource bound each workload. **Trap:** `binding` names the
  busiest *engine*, and "fixed" is not an engine, so a workload 70% idle can still report "bound by
  the load unit".
- Hardware counter observations — gated on oracle provenance: only a source with
  `derived_from_rtl is True` may carry a counter block. A functional model's counters describe a
  different machine (a 52-cycle window returned per-engine busy totals in the thousands).

---

## 9. Having a chance to beat AutoComp

Closing the gap and beating it are different problems.

**Closing it** is the work in §5: routing, fusion, im2col elimination, synchronisation, encoding. Two
local artifacts suggest our *kernel* quality is already competitive — a 19-capsule VCS comparison
recorded "Merlin is 1.243x faster by paired geometric mean" with wins 16/3, and a separate arm
measured our untuned compiler at **780 cycles** against AutoComp's tuned best of **772** and the C
library seed of **1089**. If both those and the 102x are right, then essentially **all** of the gap is
whole-model lowering, not schedule quality. Verify this before believing it — it is the single most
consequential open question in this document.

**Beating it** requires something AutoComp does not do. Its own search over its own seed measured
**1.41x / 1.0x / 1.0x** on three workloads — modest. Its advantage is the starting point, not the
search. So the opportunities are:

1. **Whole-model scope.** AutoComp optimises kernels. A compiler that fuses *across* layer boundaries,
   keeps weights resident across consecutive layers, and eliminates inter-layer layout conversion is
   doing something a per-kernel autotuner structurally cannot.
2. **The native conv FSM.** `LOOP_CONV_WS` exists in RTL and is emitted by nobody, including the
   library's `tiled_conv_auto` path in `matmul` mode. Measure `conv` vs `matmul` on the library first
   to price it (§6.2).
3. **Automatic quantisation-aware scheduling.** The library's fusion decisions are fixed; a compiler
   can choose them per layer against measured cost.
4. **Search over a space the library cannot express** — im2col-free direct convolution, cross-layer
   accumulator residency, batched synchronisation.

Parity is an engineering result; beating it is a research result. Do not promise the latter.

---

## 10. Traps that have already cost time

- **Do not scope a campaign by cohort size.** PM was chosen because it was the largest (16 of 38) and
  it levers the one axis the evidence does not implicate. It returned 0.157%.
- **Do not trust a preflight that checks a different predicate than the runtime.** A corpus preflight
  green-lit all 38 members using `plan_evaluation` while the runtime's strict admission refused; the
  failure moved from launch time to four hours in.
- **`--schedule continuous`, never `--continuous`.** The legacy path returns 1, hardcodes
  `formal_complete=False`, and runs no post-freeze grade.
- **Shared working tree.** Several agents work in one checkout. Commit with explicit pathspecs — a
  bare `git commit` sweeps the whole index and will capture another session's staged work. This
  happened during the writing of this document and required `git reset --soft` to undo.
- **Harness snapshots.** Each campaign copies a source snapshot; without an `OMIT` set this reached
  14 GB per launch and filled a 3.6 TB filesystem in one night. `perf_snapshot.py` addresses this.
- **`rc=125` from the feedback tool means the host-side evaluator refused**, and the agent sees only
  the exception type. The reason is written host-side to `<work_root>/host_refusals/`. One campaign
  lost 11 of 12 measurement calls this way and was then discarded for "not measuring its final
  candidate bytes"; the underlying event was `evidence["gsim"]` being empty, i.e. **the L3 tier never
  ran**, reported as a certificate-admission failure.
- **UNKNOWN must never read as NO.** When a value cannot be derived, record it and surface it.

---

## 11. Standing constraints

- Derive target facts from RTL or the capability manifest; never hardcode a target name, opcode,
  funct, mesh dimension or address layout in `merlin/python/merlin/**` or `build_tools/scripts/**`.
  Fail closed as UNKNOWN rather than substituting a default.
- No regex in core library code; parse structurally.
- Generated output only under `out/`, via `merlin.common.paths` helpers.
- Tests live in `merlin/tests/<bucket>/test_<area>.py`, one of the eight buckets.
- Goldens and `capsules/hidden/` stay **untracked** — they are answer keys.
- Public repository: no secrets, no local absolute paths in tracked files. This document lives under
  `out/` and is therefore untracked; keep it that way.
- Commit messages: `type(scope): summary` <= 72 chars, imperative, no attribution trailers, no
  process framing ("phase", "step", "round") in the subject — the commit-msg hook rejects those.

---

## 12. CLOSED DEFECT: missing execution is no longer a certificate rejection

The historical signature used `engine=None` and blamed strict certificate admission when no GSIM
execution had occurred. The runner now records two separate facts:

- `execution_outcome.gsim.tier_outcome` says whether L3 ran, was unavailable, failed or was skipped;
- `gsim_qualification.kind` is `execution_missing` when there is no execution to validate, and a
  certificate decision is made only when execution evidence exists.

Development feedback retries `execution_missing + numeric pass + tier unavailable` exactly once,
with the same immutable package SHA, a fresh workspace, and the original wall deadline. It persists
both attempts host-side. A numeric failure, certificate rejection or measured contradiction is not
retried. If evidence is still absent, the stage reports *"no GSIM execution evidence was produced"*
with the tier outcome instead of inventing an engine disagreement.

The redacted feedback denominator and `all_correct` now describe only cells actually measured;
unmeasured cells retain null cycle fields and cannot become an accidental performance result. The
regressions are in `test_run_paired_perf_bench.py` and `test_perf_unmeasured_cell.py`.

---

## 13. Operational note: this host is shared and memory is the binding resource

Three concurrent campaigns died with `ray.exceptions.OutOfMemoryError` on 2026-09-06. The node has
**125 GB**; at the time another tenant held **~90 GB across 11 processes**. One campaign would likely
have survived; three did not.

**Run campaigns SEQUENTIALLY.** They share no state and each seals its own claim, so concurrency buys
nothing except a way to lose all of them at once. Gate the launch on available memory (~25 GB
headroom per campaign) and refuse to start rather than starting into a full node -- a campaign that
dies at minute 30 has spent real agent budget for nothing.

Size for MEMORY, not only for cores. The original three-way launch was sized against 48 CPU cores and
never checked RAM.

---

## 14. §12 RESOLVED: the lost measurements were memory pressure

Running the PR campaign ALONE, with 83 GB free, produced:

```
trial_00: 10 feedback calls -- all rc=0
trial_01: 12 feedback calls -- all rc=0
trial_02: 13 feedback calls -- 12 rc=0, 1 rc=125
```

**34 of 35 succeeded (97%)**, against most failing when three campaigns ran concurrently. The only
variable changed was available memory.

| run | conditions | feedback success |
|---|---|---|
| pm5 | disk full, memory pressure | 1 of 12 |
| pm6 | single campaign, ample resources | 5 of 6 |
| PR/PQ/PK concurrent | memory exhausted, then OOM | most failed |
| PR alone | 83 GB free | **34 of 35** |

**Mechanism:** under memory pressure the L3 GSIM run is killed, `_gsim_l3_adapter` never populates
`evidence["gsim"]`, `engine` is None, and `validate_execution` refuses -- which the stage then reports
as "failed strict certificate admission". The certificate was never the problem.

**A correction worth keeping.** An earlier pass here argued the refusals could NOT be memory-related
because they predated another tenant's 90 GB allocation by ~17 minutes. That reasoning was wrong: the
memory was already under pressure from OUR OWN nine concurrent trials before that tenant arrived. The
lesson is not about the tenant -- it is that nine trials do not fit on this node.

**So §12's "remaining lead" (an earlier tier failing) is superseded.** The instrumentation fixes in
§12 are still worth doing -- the error message is still wrong, and one dead cell should still not cost
a whole feedback call -- but the cause is resource exhaustion, not a tier-ladder defect.

**Operational rule, now evidenced:** one campaign at a time, gated on ~25 GB free. See §13.

---

## 15. Both instrumentation defects are already being fixed -- do not duplicate

As of 2026-09-06 another session had uncommitted work in
`merlin/experiments/gemmini_perf_bench/scripts/perf_agent_stage.py` (+193/-27) and
`run_paired_perf_bench.py` (+30/-8) implementing exactly the two defects recorded above. Read that
work before starting either.

**The attainable-basis defect (§2.3) is being fixed.** The change introduces:

> *"Every host-owned point currently eligible to establish a rate. Unlike the scalar summary ceiling,
> per-member scoring filters these by exact contraction reduction depth."*

That is precisely the correction §2.3 asks for -- a k=16 member must not be scored against a ceiling
established on deep-k shapes.

**The lost-measurement defect (§12/§14) is being fixed**, with a retry:

> *"Retry an unavailable instrument once, never a verdict or certificate rejection. Both attempts use
> the same frozen bytes and original wall deadline. All raw outcomes stay host-private, so an
> infrastructure retry neither teaches the agent nor selects a faster result."*

The design constraints there are the right ones and worth preserving if the code is touched again:
a retry must apply ONLY to an unavailable instrument (never to a real refusal), must reuse the same
frozen bytes and the original deadline (so a retry cannot buy extra time), and must not let the agent
see or benefit from the retry (so it cannot become a way to select a faster number). Attempts are
recorded under `performance-execution-attempts.v1`.

**This does not remove the need for §13's operational rule.** A retry masks a single transient loss;
it does not make nine concurrent trials fit in 125 GB. Run one campaign at a time, gated on memory.

---

## 16. PR campaign result: the residency defect is SYSTEMATIC, not a one-off

The PR campaign (operand residency, 6 members, 3 trials) completed 2026-09-06 with 35 of 36
measurement calls succeeding. **All three trials were refused by the certified functional emission
guard**, and the reason is the most useful experimental result so far.

### Every trial introduced the same defect

```
capsule A6_resident_reuse -- trace_findings_introduced
  "mode resident_reuse: 1 redundant load(s) rewrite an on-chip destination that already
   held that exact source (#12 -> on-chip 16), so the region was NOT resident --
   it was re-materialized per use"
```

3 of 3 independent agents, asked to optimise operand residency, produced a schedule that CLAIMS
residency and re-materialises per use. That is verbatim the defect reported from the ResNet-50 run.

**Interpretation.** This is not a one-off bug in one shipped compiler. Three independent agents
working from the same starting point reproduced it, which says the residency abstraction on this
codegen path is easy to claim and hard to honour -- a systematic property of the path, not an
accident. Fixing residency is therefore a codegen-structure problem, not a matter of one careless
schedule.

### One trial additionally emitted undecodable instructions

`trial_01` broke three further capsules (`GM0_deep_k_fits_single_i8`, `SY_kdepth_fits_double`,
`SY_kdepth_fits_single`) with:

```
MVOUT count 0 != expected Mt*Nt=1 (M=16,N=16)
required instruction class missing: COMPUTE_PRELOADED / MVIN / MVOUT / PRELOAD
trace contains 10-20 UNKNOWN instruction(s) (fail-closed decode)
```

A backend emitting no matmul instructions and instructions the decoder cannot parse. Correctly
refused. Note the decoder FAILS CLOSED on unknown instructions rather than ignoring them -- that is
the behaviour that made this visible at all.

### What this means for the experiment

* **The guard works and is load-bearing.** It caught a real regression in every trial that had one,
  on a capsule built to exercise exactly that property.
* **PR produced a diagnosis rather than a speedup**, which is a legitimate and valuable outcome for a
  DIFFERENTIAL family. Do not report it as a failed campaign.
* **The next question is why the path makes residency hard to honour.** Start from
  `merlin/python/merlin/targetgen/trace_check.py:244` (the detector) and the lowering of
  `MATMUL_RESIDENT` -- the command buffer expresses residency correctly (§11.6 of the companion
  handoff), so the loss happens below it.
* **This raises the priority of the §5 item on residency** above the fence and encoding work: three
  independent agents cannot avoid the defect, so the codegen path must be changed rather than
  scheduled around.

---

## 17. THE HEADLINE RESULT: residency is lost by ANY edit to the contraction lowering

Three campaigns, nine trials, 2026-09-06.

| campaign | lever | clean trials | offender |
| --- | --- | ---: | --- |
| PR | operand residency | 0 / 3 | `A6_resident_reuse` in all three |
| PK | reduction depth | 0 / 3 | `A6_resident_reuse` in all three |
| PQ | redundant synchronisation removal | 2 / 3 | none (the one refusal was a hygiene violation, §18) |

**Six independent trials, across two campaigns pulling DIFFERENT levers, produced the
byte-identical finding on the same capsule:**

```
A6_resident_reuse -- mode resident_reuse: 1 redundant load(s) rewrite an on-chip destination
that already held that exact source (#12 -> on-chip 16), so the region was NOT resident --
it was re-materialized per use
```

Identical text, identical addresses. Six independent agents do not converge on that by chance.

**The conclusion:** the residency property is lost by ANY edit that touches the contraction
lowering. PR (residency) and PK (reduction depth) both touch it and both broke it 3/3; PQ
(synchronisation) does not touch it and broke it 0/3. `A6_resident_reuse` is the canary.

This is the same defect reported from the ResNet-50 production run -- "claims to reuse resident
weights but is actually reloading them" -- and these campaigns show it is not a one-off schedule
mistake. It is a structural fragility: the lowering keeps residency only in its unmodified form.

### The decisive next test (cheap, no agent)

Run a NULL candidate -- byte-identical to the certified functional baseline -- through the emission
guard. The guard computes `candidate_findings - baseline_findings`, so:

* if the null candidate shows the finding, it is a **guard/baseline artifact** and the six trials are
  false positives;
* if it does not, the finding is **genuinely introduced** and the fragility above is real.

Do this before any further residency work. It costs one guard evaluation and decides whether the
headline result stands.

### If it stands, the implication for §5

Move residency ABOVE the fence and encoding items. Scheduling around it demonstrably does not work --
six agents could not avoid it -- so the lowering of `MATMUL_RESIDENT` has to change. Start at
`merlin/python/merlin/targetgen/trace_check.py:244` (the detector, which works) and the emitter that
turns `MATMUL_RESIDENT` into `mvin`/`preload`/`compute_preloaded`. The command buffer itself is
correct: it emits `RES_PACK` / `MATMUL_RESIDENT` / `COMMIT` / `EVICT` in the right order.

### 17.1 The falsification test PASSES -- the result stands (no new run needed)

§17 proposed running a null candidate to decide whether the six identical findings were a guard
artifact. That is unnecessary: **the guard already contains the control**, and the existing records
answer it.

Before diffing findings, the guard emits both the baseline and candidate lowering for each capsule and
compares SHA-256 of the LLVM text and the command buffer. Byte-identical capsules are recorded
`proved_unchanged` and never produce a finding. From a PK trial record:

```
guard rows: 111    changed: 33    proved_unchanged: 78
A6_resident_reuse: status "changed", drives_accelerator [true, true]
```

**78 of 111 capsules were proved byte-identical**, so the guard is not flagging indiscriminately, and
`A6_resident_reuse` is among the 33 whose emission genuinely differs. The residency finding is
therefore truly INTRODUCED by the candidate, not an artifact of the comparison.

Note also that a null candidate would have been a WEAK test: with `base_llvm == cand_llvm` the diff is
empty by construction, so it would have passed trivially and proved nothing. The `proved_unchanged`
control is the right evidence and it was already recorded.

**Conclusion: §17 stands.** Any edit touching the contraction lowering loses residency; six
independent agents across two levers could not avoid it. Move the residency item above the fence and
encoding work in §5, and change the `MATMUL_RESIDENT` emitter rather than scheduling around it.

---

## 18. §7 test status -- what exists, what is blocked

Checked against the tree on 2026-09-06. Four of the nine are done; do not rewrite them.

| § | test | status |
| --- | --- | --- |
| 7.1 | layer capsule golden agreement | **blocked** -- no layer capsules exist yet (§4.1) |
| 7.2 | im2col elimination | **blocked** -- needs `perop_blocks.py` (held) + a recapture |
| 7.3 | fusion regression | **blocked** -- needs `wholemodel_proposer.py` (held) |
| 7.4 | fence count is O(1), not O(tiles) | **DONE** -- `c03943b0`, `merlin/tests/targetgen/test_fence_scaling.py` |
| 7.5 | config issued once, not per tile | **DONE** -- `fa3e382e`, same file |
| 7.6 | residency: mvin count vs RES_PACK | **DONE already** -- `trace_check.py:244`, covered by `test_resident_reuse_configures_once.py`, and it is what caught the six campaign candidates |
| 7.7 | rank-3 batched contraction admitted | **DONE already** -- `test_eligibility_one_rule.py:140`. Note it asserts through `capability_map_for_target`, the LOADER, because the fix was once applied to the packaged contract while the loaded copy still declared `[2, 4]` -- a grep over either path would have missed it. Rank 5 is checked alongside so it cannot pass by admitting everything. The contract now reads `ranks: [2, 3, 4]` |
| 7.8 | attainable basis restricted to comparable shapes | **in progress elsewhere** -- see §15 |
| 7.9 | vector instructions present in the image | **blocked** -- needs a build; the audit exists at `elf_audit.py:271` |

The two added here (7.4, 7.5) close the same class of hole: the checker validated ORDERING of FLUSH
and config-before-use, but never how OFTEN either was issued -- so 4,096 fences and a per-tile
dataflow reconfiguration both read as correct. Both are numerically correct and catastrophically
slow, which is exactly the class no numeric oracle can catch.

Both test SCALING rather than a budget, reusing the tile geometry `_check_tiles` already derives, so
neither introduces a constant. 7.5 is scoped to `CONFIG_EX` on purpose: `CONFIG_LD`/`CONFIG_ST` carry
strides a schedule may legitimately vary per tile, and these findings become REFUSALS through the
emission guard, so a false positive costs a candidate. A test pins that boundary.

---

## 19. Measured FireSim price of native `LOOP_CONV_WS`

The §6.2 same-source Gemmini library experiment completed on 2026-09-06. It used hardware
configuration `alveo_u250_firesim_shuttle_gemmini_opu` and queue jobs 498--506. Each of the nine
executions followed the queue-owned lifecycle exactly:

```
firesim kill
firesim infrasetup
firesim runworkload
firesim kill
```

Every execution ran one unmeasured warm inference followed by one measured inference in the same
ELF. Only the post-warm-up compute-cycle decomposition is admitted below. All three measured
repetitions of every arm were bit-for-bit identical in total cycles:

| arm | total | matmul | im2col | conv | pooling | residual add | other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ws conv` | 152,513,710 | 37,615,878 | 0 | 101,426,425 | 0 | 13,060,170 | 411,237 |
| `ws matmul` | 1,565,712,935 | 81,421,188 | 1,051,625,693 | 0 | 419,154,553 | 13,102,838 | 408,663 |
| `cpu matmul` | 2,148,409,928 | 127,781,645 | 1,053,682,363 | 0 | 419,079,408 | 547,420,435 | 446,077 |

Thus Gemmini's native-convolution library path is **10.266047x faster** than its hand-written
im2col-plus-matmul path for this whole model. The explicit im2col alone costs 1,051,625,693 cycles;
the native conv component costs 101,426,425 cycles. The CPU control is 1.372161x slower than the
accelerated `ws matmul` arm.

This is strong evidence that compiler selection/emission of `LOOP_CONV_WS` is a first-order target,
not a speculative micro-optimization. It is still **not a Merlin compiler speedup**: all three arms
are modes of Gemmini's hand-written `resnet50.c`. The honest compiler comparison remains Merlin
versus the `ws matmul` arm from the same captured graph and numeric contract, followed by an
independent-golden correctness check.

The authoritative machine-readable receipt is
`out/artifacts/perf-bench/gemmini/resnet50_library_baseline_warm_20260906/results.json`. It includes
the pinned source/compiler/ELF identities, every queue job, the lifecycle contract, warm/measured
scope, exact repetition distributions and the decomposition selected from the median repetition.

---

## 20. Integrity correction: a post-candidate mixed-lane harness does not count

An exploratory M3 run exposed a real ABI gap: the Phase-1 candidate encoded two host-only pointer
arguments as unconsumed `RES_PACK` commands with `layout = host_lane_operand`. The frozen
resident-matmul harness correctly rejected those commands because a resident pack describes an
accelerator weight/bias consumer, and neither pointer had one.

A temporary local harness branch was then written after inspecting that candidate. It stripped those
commands from resident-group parsing, inferred the candidate's private pointer order, and made the ELF
run. **Using a cycle count from that branch as Phase-2 evidence would be evaluator adaptation after
seeing the answer and is inadmissible.** The branch and its test were removed. The diagnostic directory
`out/artifacts/perf-bench/gemmini/m3_fused_single_elf_warm_20260906/` carries `INADMISSIBLE.md`; no
number from it may be cited.

The candidate-independent replacement is now implemented in the command-buffer ABI:

- `kernel_abi.kind = whole_program` declares an ordered list of `{tensor, access}` pointer arguments;
- every declared tensor buffer must occur exactly once and an undeclared/duplicate/omitted pointer is a
  contract failure;
- input/weight/bias/scale buffers must be read, output/intermediate buffers must be written, and
  `kernel_abi.outputs` must equal exactly the true model outputs;
- cross-lane scratch uses role `intermediate`; it can no longer masquerade as a model result;
- harness derivation recipes are forbidden on this path, so im2col or another model operation cannot be
  moved outside the measured submitted kernel;
- host operands are not residency commands. An unconsumed `RES_PACK` remains a fail-closed error;
- the runner performs one complete unmeasured warm call, then records only the cycles of one complete
  submitted-kernel call and its completion fence. It performs no LayerNorm, residual, convolution, or
  other model arithmetic;
- `perf_suite.py` seals exact hashes for the ABI contract, JSON schema, Gemmini harness and paired runner
  inside the immutable source snapshot before agent authoring.

The contract-to-harness drift gate now probes this fourth command shape, and the argument-binding RTL
check resolves the explicit row before accelerator-command inference. Focused contract, renderer,
mutation and drift tests pass.

This closes the **generic runner/ABI mechanism**, not the PB experiment. A citable PB result still needs
an independently predeclared island/no-island corpus, hidden variants, and a new compiler candidate
authored only after the suite snapshot is sealed. The existing M3 candidate predates this contract and
is useful only as a diagnostic. Until the new paired corpus and post-freeze candidate pass Spike and
GSIM under the sealed runner, `PB` remains unmeasured and no mixed-lane speedup exists.

---

## 21. Phase 2 is now a cheap-first whole-model optimization loop

Phase 1 is frozen at **92/96**. It will not be regenerated for the present campaign. Phase 2 consumes
that exact read-only snapshot, preserves its four waivers, and makes no correctness claim about source
artifacts added afterward.

The global objective is now explicit. The frozen snapshot predates capsule-level
`performance.global_objective`, so Gemmini's experiment descriptor names
`M2_microvit_gemmini`: the actual composed MicroViT model already in that snapshot and previously
priced at roughly four L3 seconds. Selection is no longer “smallest cross-lane L3 capsule,” which had
silently chosen the focused `M3_host_island_seam_gemmini` harness. Missing, conflicting, duplicate, or
out-of-snapshot objective declarations fail closed. The later full-size ResNet-50 capsule is not
retroactively Phase-1-qualified and will not trigger a new Phase-1 run.

The automatic authoring loop is now:

1. Freeze the Phase-1 package, declared complete-model objective, reduced witnesses, candidate source,
   and evaluation policy before the agent sees feedback.
2. Emit baseline and candidate command buffers for the complete frozen model and run exact structural
   analysis with no simulator: arithmetic expansion, declared movement, residency/fusion, barriers,
   representation changes, and repeated-model projections.
3. Rank the observed gap and map it to exact editable Python symbols declared by the package. Re-run
   emission analysis after every edit to determine whether the intended global effect actually
   occurred. This is the primary iteration loop and does not consume L3 or FireSim.
4. If occupancy or overlap is the one deciding unknown, allow one reduced global profile: one
   unmeasured warm invocation followed by exactly one measured invocation, with total compute cycles
   primary and only the necessary movement/resource/encoding counters retained.
5. Permit at most one exploratory tuning GSIM call in a round, then reserve one final GSIM call for the
   exact bytes being sealed. Static whole-model analysis, command-buffer analysis, and edit-surface
   inspection remain available without consuming this sparse timing budget.
6. Promote only after correctness/residency guards and the final measured receipt agree with the
   structural evidence. Projecting the complete graph is analytical; it is labeled as a projection,
   never reported as measured end-to-end timing.

The target-neutral implementation is split deliberately. `GlobalPlan` and the exact-cover planner
represent multi-operation regions and materializing representation transitions. Command-buffer
analysis constructs a repeated activity timeline and reports compute, movement, synchronization,
encoding, residency, occupancy and overlap with explicit `UNKNOWN` values where evidence is absent.
`GlobalPlanEmission` is a verified adapter protocol: every dispatch, selected region, materializing
transition, and external tensor mapping must be accounted for before a plan can replace executable
dispatches. The agent receives the initial whole-model report, ranked optimization brief, source-symbol
inventory, declared semantic edit surfaces, and the broker actions needed to refresh them.

FireSim is **not required in Phase 2** and its per-round budget is zero. It is optional post-freeze
validation only. If explicitly requested later, it must use the shared queue and every execution must
follow exactly:

```text
firesim kill
firesim infrasetup
firesim runworkload
firesim kill
```

This machinery does not guarantee “maximum performance.” It makes the important global questions
automatic and falsifiable—encodings, data movement, accelerator occupancy, latency hiding, fusion,
residency, and host/accelerator boundaries—while keeping agent iteration under the 5–10 minute ceiling.
Two limitations remain explicit: the old frozen package contains 343 indexed AST symbols but zero
predeclared `optimization_surfaces`, so mappings added during this run must pass structural ownership
validation; and the generic plan-emission protocol still needs a legal target adapter before an
accepted shadow plan changes executable dispatches. No new Phase-2 campaign or compiler speedup is
claimed until those exact frozen bytes are authored, regraded, and measured.

---

## 22. The tools are joined; the live model exposes the remaining compiler seam

The target-neutral Phase-2 analyzer now connects the captured graph, compiler placement, command
buffer, lowered target artifact, decoded trace, CCA axes, and exact candidate AST edit surfaces in one
host-owned action. It was exercised against the immutable 92/96 Phase-1 package and the declared M2
MicroViT objective on 2026-09-06. The analysis took about 2.5 seconds and used neither L3 nor FireSim.

The first result is a macro refusal, not a performance number:

```text
CPU-lane program for @forward needs about 666948 straight-line element evaluations,
past backend 400000 budget; emitted kernel is single-block so no loop to roll them into.
```

Consequently, the M2 command buffer has zero executable commands and the lowered target artifact is a
114-byte empty function. The analyzer now marks those facts `declined` and downstream encoding,
movement, residency, occupancy, and contention as `UNKNOWN`. It does not claim zero work. A cheap scan
of the frozen M0, M1, M2, and M3 interfaces found the same absence of an executable whole-model stream;
Phase 1 was not rerun or changed.

M2 nevertheless exposes the placement decision that led to the refusal:

| structural fact | observed value |
|---|---:|
| captured regions | 132 |
| regions assigned `on_mesh` | 12 |
| regions assigned `scalar_rvv_lane` | 120 |
| adjacent lane transitions | 24 |
| captured contractions | 13 |
| total contraction MACs | 124,928 |
| `on_mesh` contraction MACs | 118,016 (94.4672%) |
| `scalar_rvv_lane` contraction MACs | 6,912 (5.5328%) |
| target-derived contraction capacity regime | 13/13 `fits_double` |

The MAC split is not an acceleration percentage. It excludes non-contraction work and does not price
the 24 boundaries, representation changes, or host execution. Its value is prioritization: all
captured contractions permit double buffering, and most contraction arithmetic is already assigned to
the accelerator lane, but the composed program cannot be emitted. Rolled host loops, larger legal
fused/offloaded covers, and executable plan emission therefore precede further tile tuning.

Every re-emission now produces a `gap_coverage` matrix for ten generic mechanisms: whole-model
placement/coverage, arithmetic lowering, encoding/layout, movement/materialization, cross-operation
residency, fusion/host boundaries, dispatch/loop offload, latency hiding/double buffering,
synchronization, and capacity/contention. Each row contains:

- the current evidence status;
- the semantic CCA axes that control it;
- structurally verified candidate AST edit surfaces;
- the cheap validation route; and
- whether the row is ready, lacks evidence, or lacks an editable seam.

The emitted artifact analyzer derives instruction names and semantic roles from the selected target's
own facts. It contains no Gemmini opcode or tile-size table. The placement analyzer consumes lane names
declared by the compiler. The planner, representation transitions, activity timeline, projection,
emission accounting, warm-profile contract, and promotion gates are all shared. Gemmini supplies one
adapter and one source of validation evidence; it is not embedded in the experiment architecture.

What remains is now precise:

1. Wire a candidate-owned `TargetPlanningAdapter` and legal `GlobalPlanEmitter` to the shared
   exact-cover planner. The shared classes existing is not evidence that the compiler selects or emits
   their plan.
2. Add rolled/structured host-lane lowering, or eliminate enough host work with legal multi-operation
   regions, so M2 produces one executable whole-program command stream.
3. Add candidate-owned `optimization_surfaces`. The frozen package has 343 indexed AST symbols but no
   semantic mappings; the host will accept only real owned symbols and editable CCA axes.
4. Supply a target resource-event adapter for dynamic contention and realized DMA/compute overlap.
   Aggregate instruction counts cannot prove either one.
5. Use the already-enforced warm reduced profile only for the final scheduling unknown: one unmeasured
   invocation, exactly one measured invocation, total compute cycles primary, and only the required
   movement/resource/overlap/encoding counters retained.

The cheap signals are deliberately not combined into a predicted cycle score. Held-out evidence does
not validate the available proxies as reliable same-workload schedule rankers. They can prove work
deletion, fewer materializations/fences/conversions, legal capacity, changed loop offload, and an inert
edit in seconds. A sparse bounded warm measurement decides among structurally plausible schedules.

### Corrections required before using the external compiler comparison

The `/scratch/jack` compiler trees are valuable design evidence but remain read-only and dirty. Their
current state cannot be identified by pinned commit alone; a citable import must also seal tree/patch
and submodule digests, artifact hashes, target identity, counter scope, warm policy, and the callable
boundary. No external tree was modified during this work.

The supplied comparison also needs these factual corrections:

- Exo 1,073 versus TVM 1,265 FPGA cycles is 15.2% fewer cycles relative to TVM, or TVM is 17.9%
  higher relative to Exo. The 36% statement belongs to Voyager 677 versus Exo 1,073.
- The cycle table does not prove the proposed exact tile shapes, padding behavior, prefetch order, L2
  rejects, or fusion causality. Those mechanisms need artifact/trace evidence.
- Exo's identical 1,073 cycles on four named kernels must be checked against the generic 16-column
  shim before treating the rows as four independently specialized schedules.
- The separate Spike proxy table reverses Exo/TVM ordering and gives TVM a different microTVM callable
  boundary. It must not be merged with the FPGA table or used to infer FPGA schedule quality.

The comparison still offers an important hypothesis: local kernel scheduling is plausibly already
competitive, while the live M2 audit proves whole-model emission is not. Phase 2 should test that
hypothesis by repairing the composed program first, then using the joined structural tools and sparse
warm timing to optimize encodings, data movement, residency, and overlap across the complete graph.

## 23. Live integration correction: compile the model, debug the experiment directly

The user explicitly requested starting one live global optimization iteration and debugging failures
as they occur, rather than expanding pre-launch tests or requiring repeated trials. Multi-trial
confirmation belongs to a final performance claim. Authoring is permitted with explicit unknown
costs and unqualified candidate semantics; promotion is not.

The primary M2 emission failure in §22 was traced further. The mixed-program path omitted a host
argument from its resident-only ABI. The 666,948-element host-budget refusal was the fallback error,
not the first defect. An isolated development candidate now uses the existing whole-program ABI,
folds provably constant tensor-carried gather indices, lowers concatenation, and recognizes canonical
generic integer matmul structurally. None of these changes modified the frozen Phase 1 compiler.

Two further real defects were found while integrating that candidate:

- Host/device RAW, WAR and WAW crossings needed dependency-derived completion/ordering barriers.
- Source i8 matmul has modular output semantics, while the candidate's native narrowing saturated.
  The second development candidate reads a full accumulator into an explicit intermediate and emits
  host truncation. This increases real readout/host work; the command buffer and plan report it.

The terminal v2 compilation receipt is
`../development_full_model_codegen_v2_proof_20260906/compilation_receipt.json`:
58.18 seconds compilation plus serialization, 454/454 original operations owned exactly once,
691,704 LLVM operations, 288 inline assembly operations, 36 mesh commands and 59 explicit pointers.
Twelve pointers are declared accumulator temporaries. Compiler sources and the two shared helpers
were fingerprinted before and after compilation. No model or accelerator execution was performed.
This proves emission and accounting, not whole-model numerical correctness or a speedup.

The exact generated host-narrowing mechanism was independently translated through LLVM/clang and
executed on six accumulator cases. Its receipt is
`../development_full_model_codegen_checks_20260906/modular_readout/witness_receipt.json`.
For example, 200 now narrows to -56, not 127. That witness qualifies this changed host mechanism only;
it does not qualify the accelerator contraction or all other model operations. Wider-than-target
accumulator source types remain an explicitly reported candidate legality gap.

The shared `compiler_plan_evidence` verifier now checks source/task coverage, source dependencies,
declared tensor ABI and compiler temporaries, actual lowered task ownership and task order. Shared
immutable constants are accounted separately: the candidate's custom constant printer omits their
task annotation. Unowned arithmetic or side effects are not accepted under that exception. Its
result explicitly says numerical equivalence requires independent mechanism witnesses and timing
is unknown. The decoder and verifier can share one host-parsed artifact to avoid a second roughly
two-minute parse of the current large LLVM program.

Real full-model-to-probe extraction is also demonstrated, rather than only a callback protocol.
`../development_mechanism_extraction_20260906/extraction_receipt.json` records three exact initialized
overwrite-primitive matches between the earlier M2 artifact and a separate short PF04 probe, including
active execution configuration and instruction payloads. This is instruction-mechanism evidence,
not an in-context timing equivalence claim. The earlier full-task/readout cost is superseded by v2;
primitive timing still requires a diagnostic setup/body timing boundary and independent context
qualification. Whole-layer/model execution is forbidden even when a small model could finish quickly.

`run_global_perf_experiment.py` is a distinct macro coordinator, not a micro campaign relabeled as
global. Each edit compiles the actual complete model and records its plan/artifact identities; optional
short probes calibrate only their admitted scope. It cannot claim model latency from a micro plateau.
The production bootstrap reuses the original frozen qualification, exact gap IDs and waiver predicates;
a convenient development compiler tree with a different digest is not silently called the 92/96 seed.
The initial actual invocation exposed a relative `PYTHONPATH` that broke shared helper imports after
changing working directory to the candidate. This is being repaired at the generic invocation
boundary, with concise compilation diagnostics retained for the authoring agent.

At this checkpoint no measured end-to-end speedup or completed optimization campaign is claimed.
The immediate next action is the live full-model authoring round, with failures feeding that same
loop. Frozen qualification, model-execution exclusion and honest evidence scope remain mandatory;
exhaustive pre-launch validation does not.

### 23.1 Actual launch and first timed primitive

The live macro authoring run is now `../global_agent_authoring_20260906/`. Its `launch.json` binds
the original `merlincirct_g4p1_biasabi_20260906` submission
`5063331978caec47293c45e64beaeb7a1c534a94fa0032c8a8d9a5228ff78aba`, public 92/96, and these four gaps:
`GN0_layernorm_host_only_bf16_pt`, `M2_microvit_gemmini`, `M3_host_island_seam_gemmini`,
`SY_micro_model`. The eleven accepted completeness-waiver predicates are recorded separately; four
failed capsules do not mean four waiver predicates. M2 authoring therefore remains explicitly
conditional on a waived functional model, not a claim that Phase 1 qualified that model.

`global_iterations/iteration_0000.json` is a real production-invocation receipt: 413.13 seconds,
structural readiness passed, complete 454-operation coverage, 24 emitted tasks and twelve explicit
compiler temporaries. The actual sandboxed authoring process started afterward and is consuming this
full-model context. No optimization result is inferred merely from starting that process.

`../development_primitive_runtime_v2_20260906/execution_receipt.json` records a real, independent
GSIM diagnostic in 7.22 seconds total. It executes only an extracted 16×16×16 compute fragment, not
the source PF04 kernel's full reduction and not a layer/model. Golden output agrees. One warm body
precedes one measured body. The measured 118 cycles include the body call, original preload/compute
pair and completion; initialization/drain, warmup, output readback and verification are outside the
timer. Engine binary and build lineage, ELF, wrapper, source and extracted IR hashes are retained.
The receipt explicitly leaves in-context cycles unknown. A model-level cycle estimate must not
multiply this isolated result across unmatched shapes or unverified contention contexts.

One feedback correction is queued after the immutable live round: the coordinator's current
`candidate_totals.macs=118016` counts command-buffer accelerator MACs, not all model arithmetic.
The full placement diagnostic reports 124,928 contraction MACs including host work. The partial count
must be named accordingly; moving contraction work to the host cannot be credited as global work
deletion. No automatic performance selection is currently made from that counter.

### 23.2 Direct authoring results and evidence corrections

The original frozen `506333…` Phase-1 compiler does emit a mixed M2 program. The earlier
convenience development tree `ad2c8…`, which declined M2, is not that baseline. Neither emission
nor structural verification overturns M2's frozen functional waiver. Earlier descriptions of an
empty baseline stream apply only to that convenience tree, not the actual 92/96 submission.

The v3 candidate emits source-native host loops: full-model compilation/serialization takes
1.83 seconds and emits 9,752 LLVM operations, versus v2's 58.18 seconds and 691,704 operations.
The final artifact translates to upstream LLVM and a RISC-V object. This is compilation and
code-size improvement, not demonstrated inference work deletion. Host CFG accounting still finds
123,924 scalar loads and 101,564 stores, with 453,200/350,304 payload bytes respectively, before
LLVM optimization. These counts are neither DRAM traffic nor machine instructions/cycles.
The agent brief now exposes the largest source-region host-work hotspots and owning edit surfaces.

`../global_agent_authoring_v3_20260906/global_iterations/iteration_0001.json` records an actual
agent-produced generic scheduler change and full-model reanalysis in 159.19 seconds. It preserves
454 source operations and 24 tasks while synchronization instructions fall from 36 to 25;
compute/configuration/movement instruction counts remain unchanged. Actual fences fall 35 to 24;
the synchronization total also includes one unchanged flush.

`../development_boundary_qualification_20260906/qualification_receipt.json` independently reproduces
both emitted artifact hashes. Removing exactly those eleven fences makes the complete LLVM modules
structurally identical. Retained completion fences dominate the deletions, all intervening CFG paths
contain host work only, and all ten existing host-to-device visibility barriers remain. This qualifies
the relative synchronization delta under the pinned completion contract without simulation; it is
not whole-model numerical qualification or measured speedup.

The v3 authoring process exited successfully but its stage was refused for procedural audit issues:
unnecessary required micro smoke actions and a piped broker command. That run remains refused, not
retroactively promoted. Its independently analyzed compiler edit is retained as the next seed.

`../global_agent_authoring_v4_20260906/` has now launched directly, with that seed, the same frozen
92/96 baseline and waivers, an immutable source snapshot, bounded sandboxed analysis, baseline caching,
concise broker reports, optional micro smoke actions, and the real isolated-probe provider enabled.
The initial full-model analysis is the live integration test; no separate readiness sweep precedes it.
Short probes are requested for uncertain costs, not as mandatory ritual after every structural edit.
There is still no measured end-to-end speedup claim and no full-layer/model simulation.

### 23.3 A second global change: remove an encoding materialization

The v4 live bootstrap exposed a real sandbox wiring failure: an exact new compiler helper could
not be mounted below a read-only frozen dependency directory. Baseline compilation succeeded;
candidate compilation did not. The authoring attempt was stopped and its failure artifacts retained.
The fix composes only the already-granted directory and explicitly hashed helpers into a read-only
dependency overlay, retaining answer masks. This is an invocation repair, not a qualification waiver.

A separate development candidate at `../development_full_model_codegen_v4_20260906/` now recognizes
pure-copy, all-parallel affine generics and carries them as lazy views to a single same-host-segment
consumer. The rule checks dtype, map permutation, bounds, fanout and segment boundaries; it contains
no convolution, model-name, shape or weight-pointer match. The full-model host-work report identified
the eligible copy; this is not a micro-capsule objective substituted for the model.

`../development_full_model_codegen_v4_proof_20260906/activity_comparison.json` records:

| Full-model emitted host work | Before | After |
|---|---:|---:|
| Load payload bytes | 453,200 | 397,904 |
| Store payload bytes | 350,304 | 295,008 |
| Static allocation payload bytes | 155,792 | 128,144 |
| Dynamic integer IR arithmetic | 2,254,034 | 1,847,474 |
| Dynamic floating IR arithmetic | 181,648 | 181,648 |
| Dynamic conversions | 146,656 | 146,656 |

Combined load/store payload falls by 110,592 bytes (13.76%); one 27,648-byte intermediate disappears.
The command buffer remains byte-identical, all 454 source operations remain represented in the plan,
and both emitted CFGs pass structural verification. Compilation takes approximately 1.85 seconds.
These are pre-optimization LLVM execution/payload counts, not measured CPU cycles, DRAM traffic,
or inference speed. The independent native copy-map/consumer witness covers sixteen signed-i32
and f32 cases, including permuted output maps; no full model or layer is executed.

The shared structural comparator now reports changes across host work, accelerator instruction
sites, dispatches and declared command-buffer payload. It calls out mixed tradeoffs and missing
dimensions instead of treating a smaller partial counter as a total-model win. The next production
snapshot wires this brief into every iteration. Cost-dependent selection still requires relevant
mechanism evidence; an unchanged isolated compute probe cannot qualify a synchronization or
encoding optimization.

The next actual authoring invocation is now live at `../global_agent_authoring_v5_20260906/`, using
the explicit development candidate digest
`30b88c9497f454e5cf8fec52afe1b273ebda5feeb6cbe76ade5333abae26bb5b` as its starting point. The original
Phase-1 baseline remains unchanged. Its source snapshot includes structural-delta and probe-relevance
feedback; an isolated calibration can be recorded without falsely validating a different changed
mechanism. Shared probe extraction now clones the source's decoded completion instruction rather
than inventing a host ISA instruction. The previously failing candidate invocation was replayed
successfully under the repaired sandbox before this launch. The new run is not yet a completed or
promoted experiment.

### 23.4 General pointwise fusion, short busy-counter profiling, and live isolation repairs

`../development_full_model_codegen_v5_20260906/` adds single-consumer, equal-domain pointwise
producer fusion. It preserves the original scalar operation ordering and dtype rounding, refuses
fanout or reduction/broadcast reuse that would amplify work, and restores nested indexing state.
It removes 71 intermediate allocations in the actual full model. Relative to the lazy-copy seed,
host load/store payload falls from 692,912 to 354,480 bytes (48.84%); relative to the earlier
unfused source-loop seed it is down 55.9%. Floating IR arithmetic remains 181,648 operations,
the command buffer is byte-identical, and all 454 source operations remain covered. Compilation
takes 1.02 seconds. Twelve independent reduced numerical cases cover composed maps, i8 intermediate
overflow, f32 rounding boundaries and the existing tanh implementation. Evidence:
`../development_full_model_codegen_v5_proof_20260906/`. These remain static work reductions, not
measured target memory traffic or end-to-end cycles.

The primitive provider now optionally profiles only the target's joint engine-busy counter set.
It derives event codes and physical slot capacity, proves the Boolean partition in the recorded core
HW artifact, and configures/resets counters after warm completion and before the timer. Snapshot/read
commands are after the timer and before output readback. The measured interval remains the original
compute primitive plus call/completion overhead; setup and counter I/O are excluded.
`../development_primitive_occupancy_20260906/execution_receipt.json` records a real 8.58-second
build/run, correct output, 100 timed cycles, execution-engine token `EX` busy for 72, idle remainder
28, and zero load/store/overlap counts in this deliberately isolated probe. These are controller-busy
counters, not PE utilization. Recorded counter-core provenance and execution-engine lineage remain
separate; no claim of measured full-model occupancy is made. The primitive IR matches the earlier
118-cycle uninstrumented diagnostic, but the wrappers differ, so 118→100 is not a compiler speedup.

V5 authoring exposed eager package initializers importing the correctly masked reference simulator
while loading a pure compiler helper. Lazy public exports now decouple compiler-only imports from
the runtime/evaluator. Both masked reference modules remain unavailable; the repair does not grant
them to the candidate. The previously failing invocation succeeds under the same sandbox with only
the exact initializer overrides.

V6 is now live at `../global_agent_authoring_v6_20260906/`, explicitly seeded with candidate
`4ca249d75a785f9948274822f0c2692d12c0642b430409650fcd953348a51673`. Both baseline and candidate
have emitted LLVM successfully inside the hardened worker; host accounting is underway. It includes
the occupancy provider and exact new runtime receipt, structural comparisons, probe relevance, and
automatic relative-completion qualification. The original 92/96 baseline and waivers are unchanged.

The automatic completion check sharpens the earlier manual result: all eleven device waits are
redundant, but full host-visibility qualification remains conditional because separate pointer
arguments do not by themselves prove physical non-aliasing. The emitted task-3 epilogue writes `t4`,
not `t3`; the suspected same-buffer RAW is absent. This qualified device-completion result does not
silently become unconditional numerical or performance promotion. Its receipt is
`../development_boundary_qualification_20260906/automatic_completion_qualification.json`.

### 23.5 Actual autonomous boundary-store optimization and live integration repairs

V6 finished its authoring stage successfully (`agent_round_0000.status = authored`). Its clean
captured submission at `../global_agent_authoring_v6_20260906/global_iterations/submission_0001`
has digest `bb773e3376e4c179fdf5af35cd2a7e417c4fa8e5ed03a6f582a267e8cbf01b0c`.
The agent made a generic final pointwise producer write directly to its required boundary buffer,
removing a temporary and copy. Actual full-model reanalysis took 48.17 seconds: host load/store
payload fell from 354,480 to 341,552 bytes, allocation bytes from 43,536 to 40,304, and allocation
count from 54 to 43. All 454 source operations, 24 tasks, and accelerator command counts remained
accounted for. This is observed static work deletion, not measured end-to-end acceleration or a
full-model numerical qualification.

The overall V6 launcher nevertheless terminated unsuccessfully: final sealing encountered an
excluded Python cache in the author workspace. The next controller seals the already verified clean
captured submission instead; the original authored-stage result and terminal failure are preserved.
The agent's short profile request separately refused before simulation because an absolute include
path changed the generated wrapper hash despite identical counter-header contents. Neither failure
is retroactively promoted to a successful complete experiment.

The target probe adapter now emits a stable relative include, verifies the first header resolved
by the actual build search order against the discovered schema hash, and records the resolved path
separately. Existing source/ELF identity checks remain strict. A new real diagnostic at
`../development_primitive_occupancy_v2_20260906/execution_receipt.json` passed in 7.74 seconds:
one warm run, one measured run, correct output, 100 compute-interval cycles, EX busy 72 and idle 28.
No full layer/model was simulated. This is isolated calibration, not evidence that the boundary-store
optimization saves 100 cycles. The next bounded experiment will reuse this exact calibration and
preserve the authored compiler change, with focused changed-region checks instead of broad repeated
preflight campaigns. Frozen Phase 1 remains exactly 92/96 with its existing waivers.

### 23.6 Post-backend tradeoff evidence, shared occupancy repair, and the next live round

V7 is live at `../global_agent_authoring_v7_20260906/`, using the preserved authored change plus
the pure-host ABI repair. Initial full-model analysis completed in 167.15 seconds; the agent then
entered authoring. This is not yet a sealed or promoted result. The frozen Phase-1 package is unchanged.

The previous boundary-store result now has a consequential additional check. Compiling its retained,
hash-bound before/after full-model LLVM through the existing target object pipeline and disassembling
both objects takes about 1.2 seconds, with no model execution. Optimized static instruction sites are
6,487 before versus 6,542 after, scalar integer sites 2,434 versus 2,291, scalar floating sites 404
versus 612, and vector sites 2,412 versus 2,405. Object bytes rise from 31,592 to 32,840. Thus the
pre-LLVM memory-work reduction is **not** a dominance result across compiler stages. The shared
structural comparator now reports `mixed_work_tradeoff` when given this bound machine evidence.
These are instruction sites and object bytes, not dynamic retired instructions or a measured runtime
regression. Receipt and reproducible compile-only driver: `../development_machine_delta_20260906/`.

A target-edge machine audit now exposes that inexpensive signal through a required host-supplied
sandbox runner. Even compilation needs the answer mask: assembler directives can read files without
executing the generated kernel. There is no direct-subprocess fallback in this adapter. The next
controller snapshot will run it under the existing per-arm policy and cache immutable baseline work;
the live V7 policy is not mutated. Two runner-boundary and four structural-comparison tests pass.

The target-neutral activity timeline also now unions simultaneous same-kind engine intervals before
counting compute/movement overlap. Previously two compute engines active for the same four cycles
could report eight hidden cycles against only four available cycles. The reproduction is fixed and
eleven focused timeline/planning tests pass. This is shared multi-resource accounting, not a target
special case.

Bounded source-context extraction now finds four actual M2 queued-movement/compute windows in about
2.2 seconds. The first keeps twelve commands, including seven competing loads, rather than draining
all transfer work outside the measured interval. Exact order, configuration, row relationships, source
artifact and task ownership are retained. These are proposed short windows, not yet executable timing
receipts; physical concurrency, queue/cache state and full-context composition remain unresolved.
Evidence: `../development_queued_context_20260906/extraction_receipt.json`.

The changed-region numerical provider has two explicit integration limitations being repaired outside
the live snapshot: compact host-activity summaries do not contain every task, and the pure-host entry
initially did not exercise the mixed-program direct-boundary-output path. A separate development
revision aligns that path without changing full M2 emitted bytes. A 120-second diagnostic timed out
during synchronous policy preparation before any compiler/native witness child started; it did not
produce numerical evidence. The next diagnostic reuses recorded exact grants for only the short
extracted mechanism. Neither a source-only pass nor a setup timeout qualifies full-model correctness.

### 23.7 Correction: use the production target ISA for machine feedback

The §23.6 machine-code comparison used the older `build_object` helper's shared `rv64gcv` default.
The actual package grading path, `contract.compile.llvm_mlir_to_object(target=...)`, already overrides
that default with `harness_build_recipe(target).march()`. Thus the earlier vector-capable diagnostic
does **not** price the production target and must not select its compiler optimization. Its receipt
is retained rather than overwritten.

The new machine-audit adapter now applies exactly the production override. The corrected receipt is
`../development_machine_delta_20260906/production_receipt.json`: same hash-bound full-model artifacts,
`rv64gc` final march, 1.11 seconds for both compilations/audits, zero model execution. Before/after
static instruction sites are **5,977 → 5,623**, scalar integer sites **2,885 → 2,664**, floating sites
**1,846 → 1,846**, vector sites zero in both, and object bytes **36,272 → 33,592**. The bound shared
comparison therefore reports `decrease_in_observed_metrics`, with no increased observed dimension.
This corrects the target-selection conclusion in §23.6; it remains static downstream evidence, not
a measured end-to-end speedup. A focused test now asserts the target's final march override.

The numerical diagnostic has also progressed beyond the policy-setup timeout. Reusing the recorded
V7 answer-masked policy, with explicit candidate/scratch rebindings and equal shared import closure,
the v8 development source witness passed three independent-reference cases in **11.03 seconds**.
It extracts the actual changed task-12 scalar chain at reduced 3×3 extents and demonstrates the
direct-output lowering path: 18 bytes of input loads, 36 bytes of final stores, zero intermediate
allocations and no output reload. This is a short native mechanism check, not model execution or
target timing. The receipt explicitly leaves full-shape/backend and global numerical correspondence
unproven: `../development_full_model_codegen_v8_proof_20260906/`
`reused_policy_source_witness_attempt4/development_receipt.json`.

### 23.8 Executed queued context and a second autonomous global edit

The controlled queued-prefix probe now executes successfully, not merely extracts. Receipt:
`../development_queued_context_runtime_20260906/runtime/execution_receipt.json`. A standalone
pure-SSA slice copies 17 original commands, with 12 in the timed region, and three pointer bindings.
All seven competing loads remain inside the measured prefix. Original execution/load configuration
is restored after warmup; the source's strided i32 readout and completion operation are preserved.
One warm run plus one measured run completes build and GSIM execution in **9.27 seconds**, verifies
the reduced tile, and measures **257 cycles**: EX-only 45, simultaneous LD+EX 77, LD-only 92, other
joint combinations zero, active union 214, idle remainder 43. No layer/model was executed. This proves
real movement/compute overlap in the controlled prefix, **not** complete model-context equivalence,
cache/queue equivalence, or an end-to-end schedule speedup. Broker admission is being connected as a
scoped diagnostic, separately from the stricter gate for extrapolatable cost calibration.

V7 is now terminal and its process reaped. It produced another actual compiler edit before the
authoring deadline: generic dead-initializer deletion for a pure full-domain pointwise operation whose
scalar body never reads its `outs` argument. Unlike fusion, it also removes the unnecessary copy when
the output must stay materialized, such as fanout. The captured submission digest is
`63faddd3724d0c55ad5b6b008ad13450a1ee6c94619a1615757003c42c1410b0` at
`../global_agent_authoring_v7_20260906/global_iterations/submission_0001`. Actual full-model analysis
took **49.21 seconds**, verified all source/plan bindings, and reduced host stores **119,328 → 94,368
bytes** with host loads, floating arithmetic, MAC counts and accelerator commands unchanged. Integer
IR executions drop by 37,267 and conversions by 768. These are still static work counts.

The overall V7 stage is **refused**, not completed/promoted: the authoring transport hit its deadline
and omitted final telemetry. One duplicate concurrent analysis also refused while the original
49-second request succeeded. The useful verified checkpoint is retained, with semantic and global
cost qualification still required. The next snapshot will coalesce equal concurrent requests and
reuse already prepared exact sandbox policy for short probes. A new development seed will preserve
this edit and the earlier return-path fix; its reduced semantic example must retain the actual fanout
so it exercises materialization rather than accidentally bypassing the new lowering.

### 23.9 Fanout qualification, integrated validation, and the actual generality gap

The v9 development checkpoint preserves both autonomous edits and the aligned pure-host return path;
its digest is `f40141377963bcb188a5b784b50a545bcabcd2135ddf63f347bc7c181bef5ff0`.
The numerical witness now preserves an actual source fanout: producer 244 feeds operation 246 twice
and operation 262 once. Three native cases, each with two outputs, passed the independent reference
in **9.45 seconds**. The same reduced source compiled before/after retains its 36-byte intermediate
materialization and 252-byte load payload, while stores fall **144 → 108 bytes**, exercising precisely
the dead-initializer deletion rather than bypassing it through fusion. Evidence:
`../development_full_model_codegen_v9_proof_20260906/reused_policy_fanout_witness_attempt3/`.
This remains reduced-mechanism evidence, not full-shape/model numerical certification.

A fresh deterministic production validation is live at `../global_checkpoint_validation_v9_20260906/`:
full-model comparison/candidate analysis, scoped numerical qualification, controlled-prefix warm
profiling, then an exact conditional checkpoint. It does not require another paid authoring session
or its final telemetry, and does not retroactively accept the refused V7 session. Its source snapshot,
compiler candidates and the original 92/96 Phase-1 package are fixed independently.

Compile-only checks of other **existing frozen complete models** expose why M2 alone is insufficient:
M0 captures 718 logical nodes and M1 captures 1,403, but both still decline at the inherited host-work
budget gate. Their frozen semantics are f32 computation with quantized weights, not M2's integer
contraction semantics, and must not be silently recast to make offload appear successful. The M3 seam
does compile with 54/54 source operations and three verified tasks, but is explicitly **not a full
model** and cannot substitute for those declines. Diagnostics took **6.56 seconds**, with no model
execution, simulation, grading, Phase-1 rerun or changes to waivers. Evidence:
`../development_full_model_codegen_v9_proof_20260906/held_out_static_models_attempt2/`.

The next isolated generality revision must use real loop-backed host lowering and canonical task/ABI
ownership for zero-mesh full models; merely deleting the budget check while leaving unrolling active,
or accepting a return-only decline artifact, does not close this gap. Separately, a new scheduling
candidate is being prepared to interleave RHS-tile staging with compute in the actual compiler.
Any short A/B for that edit must retain identical load/compute work and competing traffic; the
current work-contract gate correctly rejects a speedup obtained by stopping the new prefix earlier.

### 23.10 Integrated checkpoint completed; broader full-model lowering and schedule feedback

The deterministic V9 production validation in §23.9 is now **completed**, including consumer
verification of its sealed artifact. `../global_checkpoint_validation_v9_20260906/validation.json`
records the exact V8/V9 comparison, reduced fanout semantic pass, and 257-cycle controlled-context
measurement. Initial and changed full-model analyses took 174.48 and 55.68 seconds; the scoped
numerical action took 14.41 seconds and warm/context action 15.30 seconds. The resulting
`global_iterations/global_candidate.json` is explicitly `unqualified_candidate_for_review`:
`full_model_cycles: null`, `global_speedup_proven: false`. M2's original Phase-1 waiver remains;
this does not establish full-model numerical correctness. The frozen 92/96 baseline was reused.

The optional machine audit remained UNKNOWN in that immutable run because canonicalizing a granted
tool symlink selected an ungranted sandbox path. The adapter now preserves the granted executable
spelling and selects the production LLVM disassembler. Replaying only that step with the exact
recorded policy and no new grants succeeded in 4.35 seconds. Evidence:
`../development_machine_replay_v9_20260906/receipt.json`. No kernel was executed.

**Instruction-count scope correction:** the earlier instruction-site totals in §§23.6–23.7 and that
machine replay counted decoded standard instructions only, omitting raw `<unknown>` custom sites.
The shared comparison now refuses to treat those legacy totals as complete encoded-site totals.
The adapter reports undecoded sites explicitly instead of silently dropping them. Hash-verified
re-disassembly of the same retained V9 object finds **5,747 encoded sites = 5,494 decoded + 253
undecoded**, not 5,494 total ISA sites. `coverage_supplement.json` records this correction separately;
old receipts and the immutable validation are unchanged. These remain static counts, not cycles or
proof of ISA compatibility.

The V10 compiler schedules RHS staging between compute groups in the actual complete M2 lowering.
Its relative proof preserves the 277-command multiset and verifies 330 disjoint operand-row
load/compute crossings across 11 tasks. A fixed-work slice preserves all 12 timed commands and all
nine loads, including loads now following compute, with final completion after all selected loads.
Warm1/measured1 execution took **9.45 seconds**, giving **257 → 256 cycles**, with identical joint
busy counters: EX-only 45, LD+EX 77, LD-only 92. Only idle changes 43 → 42: **no meaningful occupancy
improvement is demonstrated**. Evidence:
`../development_full_model_codegen_v10_proof_20260906/fixed_work_comparison.json`.
This is controlled same-work scheduling feedback, not complete model-context equivalence. The
separate paired-context controller action is implemented; its production invocation is next.

V11 removes the stale host-only unrolling gate through real source-native loop lowering, reuses
existing quantization normalization without recasting f32 semantics, and emits exact immutable
dense-constant globals. The actual **complete M1 model** now verifies all 1,403 source operations,
its 103 input/three output ABI, 949 loops and 2,848 reachable CFG blocks, with zero fabricated
accelerator instructions. LLVM translation succeeds and both globals preserve the original bytes.
The compile-only diagnostic took 66.07 seconds and translation/byte check 5.72 seconds. Evidence:
`../development_full_model_codegen_v11_proof_20260906/RESULT.md`. This exposes substantial global
costs to optimize: 19,459,472 bytes of static allocation payload, 694,872,028 bytes of modeled scalar
loads and 244,246,812 bytes of modeled scalar stores. These are pre-machine LLVM work/payload, not
stack footprint, DRAM traffic or elapsed cycles. M0 still declines precisely at `math.cos`; no
approximation or unsupported runtime call was substituted to claim success. V11 remains unqualified
for full-model numerics and runtime linkage. No model/layer execution or Phase-1 rerun occurred.

### 23.11 Automatic materialization guidance and explicit frozen-model selection

The experiment selector now accepts an explicit `objective_capsule` while preserving the existing
default. Selection resolves only a public model with its declared L2 screen in the original frozen
snapshot; it does not replace that snapshot or its qualification. Actual selection of M1 succeeded:
seven files, 4,623,741 bytes, capsule-tree SHA-256
`883f0b7f22f2e63b38d0c322a45f2c8bed6368bbb4bac6a0ad6fd95a3e72ed28`.
The first direct attempt exposed that frozen M1 has no `lanes` field. Explicit selection now preserves
this absence rather than excluding the model or inventing accelerator coverage. Default selection
keeps its previous metadata requirements. Three focused selector tests cover default preservation,
missing legacy lanes, immutable bytes, and rejection of private/layer/missing/escaping names.
Selection alone does not prove that every `kind: model` capsule is a complete application; seam
scope must still follow the captured workload. The launcher exposes `--objective-capsule` and a
`--validation-only --semantic-only` path that does not require a device probe.

Agent feedback now includes the three largest individual allocations and the three largest scalar
memory-payload buffers, not just the five largest tasks. Reanalysis of retained M1 in **10.58 seconds**
finds `alloca:330` at 9,437,184 bytes and 28,311,552 bytes of scalar load/store payload. Smaller
`alloca:106` and `alloca:170` each account for 51,578,880 bytes of scalar payload. Evidence:
`../development_full_model_codegen_v11_proof_20260906/materialization_hotspots.json`. The full-model
artifact SHA-256 is `4737a5b4d1479d957391f50c5892aae3a3b12703c33b0d644e474e939f2f9353`.
Buffer SSA identity and task ownership are derived; exact source-operation attribution remains
UNKNOWN, as do physical traffic and runtime costs. No recompilation or execution was necessary.

The next compiler revision uses this full-model evidence to select generic pointwise/dequantization
fusion into a one-use contraction operand, retaining f32 rounding and reduction order. The short
qualifier must exercise that actual mechanism, not an unrelated pointwise chain in the same large
host task. Its implementation and qualification are in progress, not yet a promoted result.

### 23.12 Completed paired scheduling loop and a second full-model work-deletion result

`../global_paired_validation_v10_20260906/validation.json` is now a successful **production**
paired-context invocation, and its sealed checkpoint passed the independent consumer under the
immutable source snapshot. Complete-model analyses took 245.11 and 113.36 seconds; preparation plus
both short executions took **26.88 seconds**. Each arm used warm1/measured1 on exactly the fixed work.
The result remains 257 → 256 cycles with identical EX/LD/overlap counters: no meaningful occupancy
win. The schedule-only edit correctly leaves the unrelated host semantic qualifier unresolved.
Both candidate machine audits find 5,747 encoded sites, including 253 undecoded, with unchanged
counts. The original frozen baseline machine audit reached its 60-second cap and remains UNKNOWN.
Future iterations retain that failed attempt's exact build-policy identity so the unchanged failure
need not be repeated. No timing winner or full-model numerical qualification was manufactured.

V12 now implements the selected **generic dequantization/pointwise → contraction fusion** in an
isolated compiler revision, SHA-256
`a9ca9b88d39a1f81c456c19149c052d2844d57f83043a4781e262a5ff478e581`.
The complete M1 comparison verifies all 1,403 source operations and finishes in **30.06 seconds**:

| Full-model emitted work | V11 | V12 |
| --- | ---: | ---: |
| Static allocation payload, bytes | 19,459,472 | 10,020,752 |
| Modeled scalar load payload, bytes | 694,872,028 | 675,994,588 |
| Modeled scalar store payload, bytes | 244,246,812 | 234,808,092 |
| Modeled floating arithmetic operations | 128,952,888 | 128,952,888 |
| Modeled conversions | 9,332,718 | 9,332,718 |

The transformation deletes two materializations, not useful floating-point computation. Its legality
requires a sole source consumer and a contraction input mapping that does not amplify producer
evaluation. F32 dequantization operations and contraction reduction order are unchanged. A shared
quantization helper changed during development; the before/after comparison uses the same exact
current helper bytes, and V11 re-emits its previous full-model LLVM byte-identically before comparison.

Reduced actual-source native checks took **31.68 seconds** for both arms and refusal controls.
They preserve signed extremes, nonzero zero points, negative/small/round-boundary scales and initial
accumulation, and pass bit-exact f32 output comparisons. Increased operand reuse and source fanout
correctly retain the materialization. Full-model source-to-specific-fused-loop attribution and
full-shape numerical correctness remain UNPROVEN; the reduced result does not establish them.

Production-target post-LLVM comparison of the retained complete artifacts took **5.35 seconds**,
without linking or execution: object bytes **75,400 → 75,144** and encoded instruction sites
**13,865 → 13,827**, with zero undecoded or vector sites in either. This checks that the IR deletion
does not introduce a downstream static code-size increase. It is not a dynamic machine-work,
physical traffic, stack-frame, or end-to-end cycle measurement. Evidence:
`../development_full_model_codegen_v12_proof_20260906/RESULT.md`,
`full_m1_attempt2/comparison.json`, `dequant_matmul_witness/`, and `machine_comparison.json`.
The shared automatic source qualifier is being extended for this mechanism before the integrated
M1 semantic-only validation; V12 is not yet a promoted global performance result.

### 23.13 M1 production loop passed; generic scalar-runtime link repair

The M1 V11→V12 **production semantic-only validation is complete**, at
`../global_semantic_validation_m1_v12_20260906/validation.json`. Full-model analyses took
**77.49 and 68.85 seconds**. The automatically selected actual-source pair 1254→1258 passed three
exact-f32 native cases in **22.86 seconds**, and the emitted reduced LLVM demonstrates direct
dequantized scalar use in the contraction without the dequantized tensor allocation. No simulator
ran. The seal and independent consumer both passed under source snapshot `046eb7fe…`, with the
exact V12 compiler identity above. The frozen baseline still declines this full-model invocation at
its inherited expansion limit; the successful comparison is **V11 versus V12**, not a manufactured
timing comparison against the frozen compiler. The existing 92/96 evidence is unchanged.

This is connected evidence for full-model analysis → global compiler change → source-mechanism
qualification → sealed artifact on M1, complementing the M2 scheduling loop. Full-shape numerical
correctness, exact full-model changed-chain attribution, and end-to-end cycle improvement remain
UNPROVEN. Neither successful seal promotes the compiler to a globally measured performance winner.

The M0 scalar-runtime investigation corrects an earlier feasibility concern: the installed target
math archive already contains `cosf` and `sinf`. A tiny trusted link-only program failed under the
existing recipe because libraries were placed **before** the objects referencing them. The same
objects and libraries linked when the libraries followed the objects. No approximation, new math
runtime, or harness measurement change was required.

The shared `HarnessBuildRecipe` now has ordered trailing `ldflags`, distinct from compile flags.
Both link paths preserve library/group order after all inputs; object-only compilation excludes
those flags. The target edge supplies its existing `-lm -lgcc` through that generic field. Nine
focused recipe checks passed, and the actual repaired recipe linked the retained trusted trig
objects in **1.49 seconds**, with no unresolved symbols and no manually appended diagnostic flags.
The existing build-cache token includes the exact link command and distinguishes library order.
Evidence: `../development_full_model_codegen_v12_proof_20260906/full_m1_attempt2/`
`runtime_feasibility/fixed_recipe_receipt.json`; the original failure remains in `receipt.json`.
This code landed after the immutable M1 run snapshot and did not alter that run. A separate V13
compiler revision is extending exact scalar lowering for M0; full M0 support is still in progress.

### 23.14 One real M1 authoring run launched from the validated checkpoint

`../global_agent_authoring_m1_v12_20260906/` is a **single actual agent-authoring launch**, not another
deterministic validation or a trial sweep. It starts from the preserved V12 compiler, explicitly
selects frozen M1, and uses the existing configured authoring model with a 600-second round and
300-second per-analysis limit. At recording, PID 376613 was live and had re-executed immutable source
snapshot `0e2aae26bb83c55233fc553dfab940378b2225dc1dc0b8dec47dd0b6214f181f`.
Its terminal outcome is not yet known. The snapshot contains allocation/buffer hotspot guidance,
dequantization source qualification, explicit objective selection, the generic library-order fix,
and guidance to reserve time for the final authoring response. Only whole-model analysis is mandatory;
device profiles are optional diagnostics, and full layer/model simulation remains forbidden.

### 23.15 Broader M0 compilation and honest treatment of a no-op performance change

V13, a separate isolated revision from V12, now compiles the complete frozen M0 graph rather than
declining at trig/comparison operations. Standard LLVM trig intrinsics, source-width-preserving
integer comparisons, and typed selection verify all **718 source operations**, 1,339 reachable
CFG blocks and 446 loops in **8.52 seconds**. Reduced actual-source trig/comparison/selection
witnesses pass in **10.61 seconds**; comparisons and selection are bit-exact, while trig uses an
explicit rtol=1e-6/atol=1e-7 host-reference tolerance. A separate unsigned-i8 bit-pattern check guards
against comparing sign-extended i64 values instead of the declared source width.

Production-target object compilation took **4.59 seconds**: 83,920 bytes, 16,193 encoded sites,
zero undecoded/vector sites. Its unresolved object symbols are `cosf`, `sinf`, `memcpy`, `memset`.
This does not establish full-model linkage/capacity or target-libm numerical equivalence; the earlier
tiny trig link and native witnesses have their own narrower scopes. No model was executed.
Evidence: `../development_full_model_codegen_v13_proof_20260906/RESULT.md`; compiler SHA-256
`ab4087d7172846c1645c890206b7783a64420b2fc2bacdabc3b11cdd7f0cbb07`.

The actual M1 authoring run produced an index-address simplification and a full-model reanalysis
in **68.32 seconds**. It removes 801,995,460 modeled pre-LLVM integer operations, but all memory,
floating-point, MAC and dispatch counts remain unchanged. More decisively, the pre/post target
object is **byte-identical**, SHA-256
`8ba134700b4fa30ccf70690a1a535a7ee88f37210f4c5d25bc7c9b22a069e556`.
The agent recognized that this is not a runtime gain. The authoring run is still live at recording;
this partial checkpoint is not a terminal/promoted result.

Future structural feedback now exposes source-bound `machine_object_comparison` separately from IR
count deltas, and warns against pricing an IR simplification as emitted-kernel work deletion when
the actual object is identical. Equal instruction counts or object sizes alone do not establish
identity. Reanalysis of the retained current-run receipts confirms this exact-byte result without
compilation or execution; the immutable running snapshot is unchanged.

Two other reporting corrections prevent misleading experiment guidance: a verified compiler-owned
source/task/ABI/CFG plan is now distinguished from **unproved use of the shared exact-cover solver**,
and refused/declined evidence no longer appears ready merely because its status string omits
`UNKNOWN`. Finally, portfolio cycle rows carry `measured`, `model_estimate`, or `unspecified` basis.
Estimates remain useful search feedback but cannot pass the post-freeze measured-performance gate.
This reporting gate does not request or execute full-model measurement during search. Receipt
authenticity remains the upstream admission layer's responsibility.

### 23.16 Actual autonomous global reduction fusion; native-convolution scope is still open

The M1 authoring run is now terminal **SUCCESS**, with complete telemetry, exit code zero and no
timeout. It finished in 561.809 seconds of authoring time, discarded the address-identity experiment,
and retained generic sole-use pointwise→full-domain reduction fusion. Final compiler SHA-256:
`c20ec7edb897decb428cfa34ea405515fc41b21da175e9230dd551154a039ef0`.
The final complete M1 analysis took **68.66 seconds**, preserving 1,403 source operations and
55,537,312 contraction MACs. Relative to V12, allocation and store payload each decrease by
**209,792 bytes**, load payload by **419,584 bytes**, target instruction sites by **711**, and
object size by **2,832 bytes**. Floating arithmetic and graph dependencies remain unchanged.
The sealed artifact is `unqualified_candidate_for_review`, not a measured performance winner.

Relevant qualification is still being completed. The old qualifier could pass the already-fused
dequantization pair in the same large host task, which does not qualify this new reduction fusion.
Its full-model changed-chain attribution was explicitly UNPROVEN and remains so. The next-source
qualifier now compiles the same reduced source through both bound revisions and demands an actual
relevant materialization change; unchanged supported mechanisms are skipped. A targeted supplemental
qualification will reuse the retained complete-model analyses and exact sandbox policies rather
than rerun those analyses or rewrite the original sealed receipt. No simulator ran in authoring.

The original §5.10 native-convolution implementation is **still missing**: both shared CONV2D
normalization and the isolated compiler select im2col/resident matmul rather than emit native
convolution descriptors. The CPU direct-convolution register-blocking pass is not that emitter.
Read-only inspection of existing frozen complete models found no immediately eligible explicit
integer convolution: M1's eleven convolutions and M2's depthwise convolution use f32 operands.
They cannot be silently recast to claim native offload.

M2 does have a recoverable patch path: zero padding at source 18, overlapping 6×6 stride-4 gather at
24, bijective layout views at 26–32, the exact f32 conversion `fptosi_i8(3*tanh(0.8*x))` at 36–44,
and modular-i8 matmul at 48. The pinned FIR confirms ordinary native-convolution output narrows with
saturation; it does not implement that modular result (3×127 gives 125 modulo i8 versus 127 saturated).
A compute-only native descriptor followed by explicit full-width accumulator readout and host
truncation is feasible in principle, but accumulator ownership, initialization, capacity and
completion are not yet proved. It is not an enabled or qualified route.

The next source-preserving work-deletion step is therefore a generic bounded read-only gather view
through those layout views into the pointwise consumer. A subsequent conversion-before-gather
rewrite can reduce repeated conversions from 576 patch elements to 256 original pixels if exact
padding-value transformation, scalar rounding and use/layout conditions are proved. This advances
the original im2col-materialization goal without falsely substituting illegal native output or
shrinking the plan to the already successful M1 example. Both steps are development work, not yet
performance results.

### 23.17 Relevant authored-fusion qualification completed without another model run

The supplemental replay of original V12 against sealed `c20ec7ed…` now **passes** the actual changed
pointwise→reduction mechanism at source operations 608→611. Both bound compilers compiled the same
reduced source; the resulting direct scalar edge removes 36 bytes of allocation, 36 bytes of stores
and 72 bytes of loads while preserving floating arithmetic. Three exact-f32 cases passed in
**27.25 seconds**. The unchanged dequantization mechanism did not qualify this edit.

The supplemental writer and consumer verified linkage to the original sealed candidate without
modifying its verdict. Evidence:
`../development_authored_m1_reduction_qualification_attempt2_20260906/semantic_supplement.json`.
An initial tool-path refusal is preserved separately; the successful retry used the byte-identical
translator at its already-granted retained-sandbox path. There was no complete-model recompile,
model/layer execution, simulator, or new Phase 1 run. Full-shape numerical equivalence and full-model
speedup remain unproven; the result is a relevant reduced correctness witness, not a timing result.

### 23.18 M2 gather materialization removed; finite-buffer analytical projection enabled

V14 is a stable isolated development compiler, SHA-256
`887b29743978baeb6bd283c30fe248a0feddb083aa6281fecd622fb933884e7e`.
Its generic read-only gather deferral proves source indices bounded through SSA integer intervals,
follows only sole-use bijective layout views, and retains every original scalar conversion and
source read. It does not recognize M2 by name or substitute native saturated output for modular i8.

Complete M2 before/after compilation and accounting took **10.34 seconds**. Both 454-operation source
plans verify, with identical command buffers, tensors, ABI and accelerator commands. Allocation
payload falls 40,304→38,000 bytes, scalar loads 222,224→217,616 bytes and stores 94,368→89,760 bytes.
Floating arithmetic (181,648) and conversions (129,120) remain unchanged. Production-object comparison
took 4.45 seconds and removes 240 object bytes and 44 decoded sites, with no undecoded-site increase.
The extracted overlapping/padded gather→views→conversion witness matches signed i8 outputs exactly;
a fanout control retains the intermediate and exact gathered f32 bits, including signed zero.
Evidence: `../development_full_model_codegen_v14_proof_20260906/RESULT.md`. These are development
results, not an autonomous run, full-model numerical qualification, or measured inference speedup.

The shared pipeline estimator now accepts explicit, provenance-bearing buffer-slot constraints.
Slots stay owned from producer start through consumer completion; one slot can serialize stages
whose engines are otherwise independent, while two slots may permit overlap. Completion is computed
using max-plus matrix powering in logarithmic repetition count, without replaying layer/model tiles.
An omitted buffer contract is labeled an **unlimited-buffer idealization**, never a capacity proof.
Finite-slot overlap remains UNKNOWN when only completion and per-resource busy time are established.
The interval resource floor uses lower-end costs, not upper-end costs. Eleven focused checks pass,
including a trillion-item analytical projection, tiny independently expanded recurrences, missing
buffer-edge refusals, and uncertainty handling. This proves the estimator's stated model, not an
actual target's stationary costs, independent resources, or allocation/lifetime correspondence;
no full-model timing prediction is claimed from it yet.

### 23.19 Exact conversion-before-gather work deletion and the remaining legality guard

V15, SHA-256 `237da7c124356cd82c6764339a139e1ceb007946cb086b3e04abc8895ceb079f`,
moves the unchanged uniform scalar conversion chain onto the interior input and padding scalar
before gathering. Complete M2 compilation/accounting took **11.56 seconds**, preserving its verified
454 source operations, 24 tasks, complete command buffer and ABI. Conversion-chain evaluations fall
576→257; emitted floating arithmetic falls 181,648→172,716 and conversions 129,120→123,896.
Allocation becomes 37,028 bytes, scalar loads 215,888 bytes and stores 88,020 bytes. The narrower
encoded padded input is explicitly allocated and counted. Production-object comparison took 4.03
seconds: 32,192→31,080 bytes and 5,703→5,608 instruction sites, with 253 undecoded sites in each arm.

Reduced actual-source overlap/padding, nonzero transformed padding and fanout controls pass in
31.71 seconds; reference and candidate retain the original scalar rounding and signed conversion.
Evidence: `../development_full_model_codegen_v15_proof_20260906/RESULT.md`. These are not measured
inference cycles or full-model numerical qualification.

Review found an additional **general-rule legality obligation**, not resolved by the passing M2
example: Q must not be newly evaluated on an interior/padding value never accessed by the original
gather when Q contains partial operations such as float-to-int conversion. Actual M2 and the reduced
witness cover the relevant coordinates, but V15's rule needs a source-index coverage proof before
being treated as generally legal. V15 is preserved; a new unified revision will add this guard and
combine the exact authored M1 reduction fusion with the M0/M2 improvements. No frozen Phase 1 result
is changed, and no separate best-of-model compiler is presented as a unified winner.

The experiment qualifier now has source-derived bounded gather extraction/evaluation, with no
model-specific source ordinals or geometry substitutions. It admits only a proper gather/layout/
pointwise subgraph, excludes the whole source application and contractions, preserves all source
intermediate uses, and caps tensor extents, aggregate storage and reference scalar steps. Large or
unsupported mechanisms abstain. Ten focused source-witness checks pass. Its first production replay
of retained V14/V15 complete-model artifacts is running at recording; a passing development script
is not being substituted for that integrated outcome.

### 23.20 Production gather qualification passed; occupancy feedback reaches the agent

The actual automatic gather qualifier is terminal **PASS** on its first invocation: **23.69 seconds**
for the action, 34.32 seconds including restoration/driver work. It reused the hash-bound V14/V15
complete-model source/plan artifacts and existing answer-masked compiler policies, selected source
gather 24 and its index/padding/layout/conversion subgraph, and compiled only that proper mechanism.
The original bounded geometry is 256 f32 inputs and 576 i8 outputs; no contraction or full layer was
included. Both reduced emissions exhibit the relevant changes: −972 allocation bytes, −1,728 load
bytes, −1,740 store bytes and −8,932 floating operations. Three independent-reference i8 output
cases pass exactly. This is relevant source-subgraph qualification, not full-model equivalence.
Receipt: `../development_production_gather_qualification_v15_20260906/iterations/`
`semantic_0001_1788755033248680187.json`. The 24 exact interpretation-policy source files are preserved
read-only in that output's `policy_sources/`; a separate sealed binding manifest links their hashes
to the original experiment/semantic receipts without rewriting those receipts. It is not a complete
runtime/toolchain snapshot. There was no complete-model recompile, execution, simulator, or Phase 1.

The paired-context feedback connection is also implemented. Actual retained V10 counters now reach
the agent's next-step guidance, are carried forward explicitly as prior-revision evidence, and are
re-derived during sealing/consumption. They correctly say **no observed overlap or busy-work change**:
257→256 cycles, unchanged overlap and EX/LD/ST busy counts. The source-bound task/source operation
links to `schedule.py::Scheduler.run`; it does not price an end-to-end speedup. Full-model occupancy,
contention and timing remain UNKNOWN, capacity is unproved, and pipeline extrapolation is not admitted.
Evidence: `../development_paired_feedback_20260906/RESULT.md`; 44 focused controller/guidance checks
pass. No additional simulator or authoring trial was run for this evidence join.

### 23.21 One unified compiler preserves improvements across M0/M1/M2; real M2 authoring launched

V17 is ready, SHA-256 `75b84222a3ce260542c549fd6bb11c0d96ebb67f5fa4f01e80482b43464d52c2`.
It combines V13 scalar lowering, the exact authored `c20ec7ed…` reduction-fusion change, bounded gather
deferral and conversion-before-gather with a new source-index coverage guard. All three complete
models compile and verify under this **same compiler** in 21.37 seconds; no per-model winner switch.

| Complete source | Source operations / tasks | Allocation payload | Scalar loads | Scalar stores |
|---|---:|---:|---:|---:|
| M0 | 718 / 1 | 1,936,288 B | 46,012,992 B | 15,801,824 B |
| M1 | 1,403 / 1 | 9,810,960 B | 675,575,004 B | 234,598,300 B |
| M2 | 454 / 24 | 32,932 B | 207,696 B | 83,924 B |

The authored reduction rule also deletes work in M0 and M2 without model-specific matching. The
coverage guard leaves all three emitted artifacts identical to the pre-guard unified V16; its new
reduced controls reject incomplete interior coverage and unused NaN padding, retaining V14's exact
LLVM and matching independent numerical references. Those controls completed in 21.42 seconds.
Evidence: `../development_full_model_codegen_v17_proof_20260906/comparison.json` and
`coverage_qualification.json`. Full-model numerical correctness and cycles remain unqualified.

One actual full-M2 authoring round has now launched from V17 at
`../global_agent_authoring_unified_m2_v17_20260906/` (launcher exec session 59226 live at recording).
Budget: 600 seconds authoring, 300 seconds per analysis, 40 tool calls; no extra deterministic
validation round or trial sweep. Its immutable source snapshot includes automatic relevant gather
qualification and source-bound paired occupancy decision feedback. Snapshot preparation is in
progress at recording; terminal authoring outcome is not yet known. Full-model/layer simulation and
execution remain forbidden, and frozen Phase 1 remains exactly 92/96.

### 23.22 New ResNet evidence is diagnostic, not a replacement Phase 1 or a timing certificate

The supplied `../resnet50_merlin_w8a8_single_run_firesim_bundle_final_20260906/` contains
53 integer convolutions plus FC, 1,240 uniquely owned source operations and 109 scheduled
host/device tasks. Its compilation receipt pins target MLIR
`ec926b37bd35c4a5ec1abc13db2c124c9f1e95fe243a53891edb78d0df1f52d5` and compiler
`2b2f07a7151b2792c0d79faeca3464edbde70bac1c1c5230c83feae67d074134`.
The matching compiler is the retained `development_full_model_codegen_v4_streamconv_20260906`.
This artifact is held-out development evidence; the frozen 92/96 input is unchanged.

Two measurement caveats are essential. The supplied Spike receipt is **warmup=0**, measured=1,
so it does not satisfy our warm timing contract. The 16,489,325 counter is named
`loop_matmul_active_cycles`, not verified mesh-MAC activity; it cannot establish “3.3% from peak.”
The implausible DMA-byte counters remain unsupported. The integer-golden functional pass is
not an apples-to-apples comparison to Jack's differently quantized/runtime-boundary result.
No new full-model/layer execution was performed to change these facts.

Exact epilogue semantics are now inspectable: recovered normalized source SHA
`76c26171096661650e2ee440fb545926936b1e770d2bfd0206a6509e6e835e94` matches the original
receipt. Its per-channel sequence includes multiple separately rounded f32 multiplies, f32
bias, ReLU and round-even requantization. A single accelerator scale plus pre-scale integer
bias is not generally equivalent. An actual-parameter one-point witness gives 4 for the
staged source and 5 for that proposed replacement at accumulator −2145; full-model reachability
of that particular value is not claimed. See `../development_resnet_epilogue_analysis_20260906/`.
These are Phase 2 compiler opportunities, with candidate-specific semantic checks, not reasons
to repeat Phase 1 or silently change its quantization contract.

### 23.23 Actual authoring exposed and repaired a generic accounting limitation

The §23.21 immutable V17 authoring round is terminal **REFUSED**, not promoted: the agent
timed out before final telemetry/sealing. Its valid intermediate submission
`d0c325c51b4b0b5f4e299c770c4e8d34cd07d26b9f3bf06cbd6af4dd776226b2` compiled the full M2
graph in 51.41 seconds, preserving 454 operations / 24 tasks. It introduced an SSA-carried
host reduction accumulator; the original accounting instrument could not count a loop with
more than one header argument. Original receipts and the unfinished draft remain unchanged.

`host_cfg_activity.py` now proves constant induction separately from additional carried
values, including induction at a nonzero argument position. Recurrence-dependent bounds or
steps remain UNKNOWN. A separate re-accounting of the exact before/after emitted artifacts
takes 2.44 seconds, without compiling or executing either model:

| Emitted host IR metric | Candidate minus V17 |
|---|---:|
| Static allocation payload | +3,072 B |
| Dynamic load payload | −27,648 B |
| Dynamic store payload | −24,576 B |
| Integer arithmetic operations | −78,336 |
| Floating operations / conversions | unchanged |

Evidence: `../development_carried_cfg_accounting_20260906/receipt.json`. This is a mixed
structural tradeoff, not numerical qualification, measured DRAM traffic or a cycle verdict.

The next actual sustained run is live at `../global_sustained_authoring_unified_m2_v17_20260906/`,
snapshot `0eb0deaad1f2d0b4328564684ef4088e1da742e6a92338188c725410049ce25f`, PID 696201,
launcher exec 21521. One process owns two 600-second authoring rounds, total authoring budget
1,200 seconds, with 300-second actions. Its initial V17 seed was actually sealed and consumed
after a 229.99-second full-graph analysis; authoring started at 2026-09-07T04:55:02Z.
This permits recovery to a real checkpoint after ordinary author timeouts, while integrity
failures still stop the run. No failed draft was silently promoted or injected as a hint.
The two-round terminal outcome is not yet known at recording.

### 23.24 Compiler-owned compact ABI: actual ResNet object, no arena reuse claim

New generic `merlin.llvmlower.compact_abi.compact_pointer_entry` converts an explicitly bound
all-pointer entry to base pointers plus byte offsets. It clones rather than mutates the input,
requires complete bindings and target-derived pointer index widths, uses a distinct symbol,
and verifies every original CFG operation, operand and successor under the pointer substitution.
It adds no alignment/no-alias assertions and infers no memory-lifetime or DMA-completion facts.

The actual supplied ResNet artifact now has a separately emitted **393→2 pointer entry** and
RV64GC object, produced in **120.47 seconds**. Existing constant/mutable offsets and extents
are preserved exactly; arena bytes saved are **zero**. Data-layout width is obtained from the
selected compiler, not a shared-code target literal. Its original model bundle was not modified.
Evidence: `../development_resnet_compact_abi_20260906/receipt.json`, `abi_contract.json`,
`target.mlir`, and `kernel.o`. Pass SHA:
`97b0e9d54d9eb5424fe52eff9597dd2cbab9ab098fe3e511d804b41c65a74b4e`.

A two-load/one-add/one-store native address mechanism passes three cases in 0.053 seconds,
with one warm and one observed invocation per arm/case. It is not a layer/model execution or
timing measurement. Eight targeted ABI checks pass; combined accounting/ABI/pipeline/activity/
report checks are 56 passed in 0.27 seconds. Target-name and regex guards pass without new waivers.

Remaining integration is explicit: the experiment's common emitter/verifier/harness must
consume a compiler-emitted ABI contract before this can be an automatically selectable
candidate transformation. The old many-pointer harness intentionally cannot link to the new
symbol. A standalone emitted object is not being presented as completed experiment integration,
arena lifetime reuse, or an end-to-end speedup.

### 23.25 Explicit performance surfaces, not unrestricted compiler exploration

The user clarified that important performance levers must be exposed explicitly instead of
giving the agent unrestricted repository access. The next-source shared guidance now produces
`compiler_edit_contract_v1`: host-frozen AST symbols, explicit owning new-helper directories,
protected evaluator/model/reference/hardware/measurement controls, and required source/plan-bound
hypothesis, expected delta, semantic obligations, cheap evidence and stopping condition.
Candidate manifest edits cannot enlarge this contract. The catalog itself is not enforcement;
the launcher is integrating a pre-execution AST-change gate separately, without changing the
currently running immutable experiment.

The actual retained compiler now has a separately generated **26-surface host-approved catalog**:
8 existing surfaces plus 18 for global partition/placement, tensor bindings, entry ABI, encodings,
resident operands, convolution/batching routes, exact epilogues/readout, global issue/fences,
capacity tiles, pipeline issue, scalar loop state, host residuals and floating primitives, and
target IR. All declared locations resolve to real source AST symbols. The shared code contains
no target package paths: those are data in the target-specific artifact.

Catalog: `../development_global_surface_catalog_20260906/PERFORMANCE_SURFACES.md`.
Machine contract: `edit_contract.json`, canonical SHA
`698e1a0386067ac908d7203fcfd30e7bbd2663f8b7fe2e8f03aeccfa249e40c9`; source files and full
inventory are bound by `receipt.json` and `inventory.json`. Guidance now has 13 mechanism-gap rows,
explicitly including exact quantized epilogues/residuals, arena lifetime reuse and ABI/runtime
overhead. Missing proof remains UNKNOWN, not a ready or zero-cost classification.
Eighteen focused guidance checks pass. Repository structure checking still reports 28 problems
in unrelated directories/library boundaries; none are being waived or presented as a clean gate.

### 23.26 Sustained progression and the ResNet integration distinction

The sustained experiment actually completed round 0, sealed and consumed candidate
`292f10db81b27064df17b13384aeebb5aff7b27d5b8c7b790ac668b84bbdfdda`, and started round 1
automatically in the same process/source snapshot. Reported first-round full-model deltas are
−3,760 B each of scalar load/store payload, −7,520 conversions and −22,679 integer operations;
machine sites 5,456→5,250 and object bytes −976. Unresolved optional probes did not stop the
authoring lifecycle. This is an authored/consumed checkpoint, not semantic or performance promotion.

The historical ResNet compiler was separately updated with exact staged host epilogue fusion:
source, command buffer, ABI and accelerator schedule unchanged; LLVM operations
151,745→135,009. Static dynamic scalar load/store payload drops by 277,342,848 B **each**,
with floating arithmetic unchanged. An exact-source one-point native witness passes in 6.60 s,
warm1/observed1. No full model/layer was executed and reserved arena capacity is unchanged.
Evidence: `../development_resnet_epilogue_analysis_20260906/staged_host_activity_comparison.json`
and `staged_native_witness_attempt2/qualification.json`.

**Do not label this a new V17 optimization:** direct source comparison shows V17 already has the
same staged pointwise fusion plus more general reduction/gather optimizations. The missing union
is different: V17's full-model placement path lacks the historical source-convolution recognition,
`_mesh_conv` route and explicit integer-preparation/input-prologue entry. Integrating those routes
into a fresh unified candidate is next work. It must keep the held-out ResNet's designated integer
contract explicit and leave frozen Phase 1 and the other source semantics unchanged.

### 23.27 Unified source-convolution and explicit storage ABI now compile together

V18 integrated source convolution with the existing V17 global host optimizations. V19 adds
complete target-neutral storage contracts for all 393 ABI tensors, consumed by buffer allocation
and packed access pitches. The actual normalized ResNet LLVM is byte-identical between V18/V19;
all 1,240 operations remain uniquely owned, with 53 convolution tasks, one FC and 55 host tasks.
Default M0/M1/M2 artifacts remain unchanged. See
`../development_resnet_epilogue_analysis_20260906/V19_ENCODING_RESULT.md`.

The generic storage checker derives exact logical-axis grouping, strides and bounded injective
addresses rather than accepting equal tensor volumes or layout names. OIHW→CoK preserves
the original C-order grouping; FC NK→KN explicitly swaps axes. Scalar values have explicit
single-element storage views. Padding geometry is supplied by the compiler's target-derived
contract, never invented by the shared checker.

Actual full source/plan/CFG/ABI verification now passes in **42.82 seconds**, for all **1,240
source operations, 109 tasks and 393 encoding contracts**. Evidence:
`../../development_storage_encoding_20260906/full_graph_attempt2/receipt.json` and `RESULT.md`.
This reads retained artifacts, not a full-model execution or new Phase 1. The same checker also
passes retained M0/M1/M2 graphs (718/1,403/454 operations), with legitimate host contractions.

The preceding refusal is retained. It exposed a separate classifier defect: `prov.family` was
used as if it were computation semantics, rejecting pointwise host epilogues carrying inherited
contraction labels. The repaired gate checks actual reduction/yielded MAC def-use for declared
contraction work; source labels never establish accelerator placement. Forged constant or dead
multiply-as-contraction claims remain refused. Actual placement remains a separate target-bound
emitted-instruction obligation.

The common caller now consumes complete explicit layouts for data-format setup/readback with
an unchanged warm compute window. It refuses candidate-selected input permutations without
host-owned immutable-weight prepacking authority; declaring `role: weight` is not enough.
For V19 that leaves only the FC transpose as an explicit caller prepacking obligation. The
metadata check materializes no model inputs. Short native caller tests and scoped encoding
copy/address checks pass; neither is a full-model numerical certificate.

Also completed: actual full-M2 physical conversion emission, independently checked allocation/
address/load-store/consumer dataflow and nine short native source checks. The forced copy
regresses; Pareto retains the byte-identical incumbent. This is proof that the loop can price
an actual global encoding choice, not a new speedup. See
`../development_physical_transition_codegen_proof_20260906/RESULT.md`.

The previously safely blocked but unterminated broker scope-refusal path now records terminal
refusal receipts (HTTP 400) for both unauthorized edits and baseline-integrity violations;
unexpected faults are not disguised as ordinary refusals. Existing run receipts stay unchanged.

Next-source policy now pins all new storage interpreters. Agent feedback replaces only its
presentation copy of the 189 KB storage map with a 2.1 KB digest-bound summary, preserving
explicit unresolved caller/consumer counts and links to the full unchanged evidence. Combined
focused regressions: 187 passed in 3.91 s.

No full layer/model execution, simulation, FireSim, or Phase 1 rerun occurred. Remaining work
includes fully bound caller/consumer proofs, compact-ABI consumption, asynchronous storage reuse,
broader relevant probes, and model-wide cost ordering. The overall goal is not complete.

### 23.28 Captured weights now have explicit host-owned prepacking authority

The earlier FC input-permutation refusal now has a bounded, generic authorization path.
The constant reader verifies pinned manifest/header/blob bytes and retains only the selected
payload. A host-selected normalization replay preserves actual entry-argument identity/order/type
and must reproduce the exact normalized source digest. The authority factory joins these facts
to the source's weights-file binding, complete entry bindings, read-only ABI and exact encoding.
This is not inferred from a candidate's `role: weight` label or deserialized candidate evidence.

Actual retained ResNet proof: **217 entry arguments**, **2,048,000 selected payload bytes**,
**2.8965 seconds**, with changed-payload refusal. No full input packing, model/layer execution,
simulation, target recompilation or candidate import occurred. Evidence:
`../../development_prepack_authority_20260906/capture_receipt.json` and `RESULT.md`.

The common compile/link/caller path takes the authorization out of band, requires explicit
logical inputs, checks the actual initializer bytes immediately before packing and uses those
same checked words. It refuses missing inputs rather than falling back to candidate-recorded
values, unsupported backends, changed bindings and serialized grants. Authorized builds bypass
cache reads/writes because an ordinary cached ELF would skip these checks. Tiny caller tests
retain warm1/measured1. Combined focused regression check: **190 passed in 3.91 s**; no-target-name
and no-regex gates pass without growing debt.

This is captured-payload preparation authority, not proof of machine consumer accesses or
normalization numerical equivalence. Automatic experiment-side context construction and exact
consumer witness qualification remain to be connected. The paper tooling summary now records
this distinction. Historical receipts and frozen Phase 1 remain untouched. No performance
promotion, new sustained campaign, or end-to-end speedup is claimed; the overall goal stays open.

### 23.29 Physical copy evidence now runs in production full-graph analysis

`perf/physical_transition_evidence.py` replaces the one-off copy checker with a generic,
host-owned analysis. The global plan verifier automatically invokes it for declared or marked
physical transitions, including orphan source/destination markers. Supported invalid copies
refuse; unsupported mechanisms remain UNKNOWN, separate from source ownership and numerics.
It checks actual typed GEP scaling, affine address domain, scalar type, allocation extents,
load-to-store bit identity, destination pointer uses, consumer ordering and charged byte counts.
Producer/consumer arithmetic is explicitly outside that proof. No target geometry is assumed.

The retained forced-transition M2 graph passes the final production checker: **454 operations,
24 tasks, 6.032 s**. Actual copy charges remain **3,072 B load + 3,072 B store + 3,072 B destination**.
The earlier 3.634-second receipt is preserved, not upgraded. These are checks of an existing
regressing control, not a speedup. Evidence:
`../../development_automatic_transition_evidence_20260906/full_graph_final.json` and `RESULT.md`.

The native correctness runner now admits bounded explicit logical offsets or strided maps,
including non-row-major views, after validating complete allocation/count/type/access limits.
Warm1/observed1 remains; there are no timing or prepacking permissions in this runner. Combined
focused checks: **237 passed in 6.31 s**. The new static checker is pinned by experiment host
policy and appears in digest-bound compact agent feedback. No-target/no-regex debt is unchanged;
the structure gate retains its same 28 unrelated problems.

A broader mocked compile-routing test run found four expectation mismatches in unchanged
`compile_cli._mesh_verify`: Spike screening precedes final-engine certification, and unavailable
screen results aggregate to partial. Those tests stub certification and never enter the new
prepack implementation. **15 passed, 4 failed, 2 skipped**; no actual simulator ran. Do not report
the whole repository suite as green or change Phase 1 to resolve these separate expectations.

Next active integration is the automatic semantic qualifier for a changed internal physical
copy: select its actual whole-graph source edge, compile the exact before/after short witnesses,
prove the changed representation is reproduced and use existing sandboxed native execution.
Do not inject full-capture prepack grants into synthetic/reduced witnesses. Captured-weight
context construction, arbitrary emitted-consumer indexing, compact caller ABI, asynchronous
reuse and model-wide cost ordering still remain. The goal is active and incomplete.

### 23.30 Automatic internal-copy qualification passes a real controller replay

The launcher now dispatches declared/marked internal physical copies to a dedicated host
qualifier before the legacy changed-region provider. Missing or unsupported copy evidence
does not fall through to an unrelated successful witness. Both full artifacts are checked;
the actual selected source producer/consumer pair is cloned at bounded extents, compiled by
the exact preceding/current revisions and checked for matching copy type and orientation.
Native admission checks actual bounded allocation, memory payload and operation counts,
as well as a closed supported scalar operation set. Full-capture prepack grants are not used.

An actual controller replay used two NEW immutable normal-entrypoint revisions (off/forced),
not flags smuggled into a probe or relabeled historical analyses. Fresh full M2 analysis took
**26.03/22.66 s**. The production qualification action selected edge **284→288**, compiled its
**3×3** witness under both revisions and passed **six exact-bit native cases**, warm1/observed1,
in **46.68 s** (below its 60-second bound). Total replay was 102.71 s. Evidence:
`../development_physical_transition_production_qualification_20260907/RESULT.md` and `receipt.json`.
This remains a deliberately regressing control, not an improvement or full numerical certificate.

Parallel development also produced generic compact-pointer caller binding with byte-preserving
setup/readback and explicit host-supplied alignment/index-width facts; 62 related tests pass.
The target renderer still needs explicit compact-entry dispatch and real target-derived facts.
Those tests do not establish that compact ABI is enabled in the launched experiment.

The user has now explicitly requested an initial search before every feature is finished.
Prepare a fresh source-snapshotted full-MicroViT segment on the same frozen 92/96 baseline,
with three 600-second authoring rounds, 600-second iteration bounds and retained-checkpoint
recovery. Report emitted full-model work/movement/storage changes and scoped measured cycles
separately; do not claim full-model measured speedup. The current launcher only selects frozen
functional model objectives, so the separately captured ResNet is not silently substituted.
No Phase 1 rerun or model/layer simulation is authorized or needed for this launch.

### 23.31 Initial V19 full-model search segment launched

The user explicitly requested launching before every integration is finished. The new
source-snapshotted launcher is live in execution session **67667**, PID **1632981**, output
`../global_sustained_authoring_m2_v19_20260907/`. `launch.json` confirms full M2 objective,
V19 initial SHA `781808c981fed365a7c5c3ddd4f15be0e53f710f39f68641de23e44f803c3321`, three
600-second rounds, 1,800 authoring seconds, 600-second iteration bounds and checkpoint recovery.
All 27 previously approved AST surfaces were repinned without adding authority. The new
`ChangedRegionQualifierDispatch` is enabled. Frozen Phase 1 remains exactly 92/96 and full-model
simulation is false. Startup is outside the authoring budget. No new improvement is reported yet.

Re-poll the live handle rather than restart. Also discovered an independent older live launcher,
PID 1147159, under `global_phase2_perf_opt_m2_v1_20260906`; its state is being audited read-only.
Do not terminate or modify it based on this note. Source snapshots isolate the two runs.

### 23.32 Separate runs, external model input, and emitted-work priorities

Read-only recheck on 2026-09-07 found the V19 launcher PID 1632981 live in round 1,
with round 0's clean retained candidate `40182b8ff813335fe47cd778ec15ca2e4d15f7b2376ab57c7f05ed68529df495`.
The other session's PID 1147159 was absent; its `continuation_failure_0006.json` records
`recovery: stop` after an authoring-audit failure, preserving round-2 candidate `d925b87...`.
These are independent source snapshots on the same `feat/target-generalization` branch.
The old run's two full-model GSIM checks passed exact golden outputs but had no warm invocation:
2,110,042 → 2,103,648 cycles, 0.303% fewer cycles. They are cold observations, not this
search's warm bounded-probe evidence, and were not rerun here.

The existing structural comparison already warned that 61,440 of the older run's 88,576B
pre-LLVM payload reduction left the optimized target object identical. New production
`agent_analysis_view` now lifts that bound comparison into the main optimization brief,
with separate object/caller/linked-ELF/timing scope. No duplicate classifier or synthetic
cycle score was added. 68 combined macro/structural tests passed. Exact audit:
`../development_existing_run_audit_20260907/ir_payload_vs_machine_work.md`.

Generic occupancy conversion now retains adapter-declared engines absent from counters.
Missing compute observations yield UNKNOWN utilization and UNKNOWN idle-capacity opportunities;
explicit zero observations remain known idle time. Duplicate counters/resource roles refuse,
and a missing declared movement counter cannot pass the complete-occupancy gate.
74 occupancy/planning/pipeline/guidance tests plus 17 external-objective tests passed together
in 0.56 s. No hardware/model execution was involved, and live frozen snapshots are untouched.

The new host-pinned external objective path seals only normalized MLIR and fixed metadata,
separate from unchanged Phase-1 92/96 qualification. Actual normalized ResNet input sealed in
0.94 s; no capture weights/reference files were granted and numerical equivalence is UNPROVEN.
See `../development_external_objective_20260907/RESULT.md`. This does not yet prove a complete
ResNet experiment: the historical V19 proof explicitly enabled `--source-convolution`, whereas
the default manifest does not. A new isolated production-controller compile/static preflight
is testing the unchanged frozen Phase-1 compiler and unchanged V19 manifest against this source.
Do not silently replace the comparison compiler, add flags to historical evidence, or rerun Phase 1.

### 23.33 Real external objective, separate comparator, full MAC coverage

The actual default-entrypoint ResNet preflight completed in 230.75 s, with 1,240 source ops
verified across three tasks (host/FC/host). The frozen Phase-1 compiler explicitly declined
162,772,535 straight-line host evaluations above its 400,000 limit. Its rc=0 placeholder was
not a valid performance comparator. The old snapshotted controller still admitted candidate
structural readiness, while correctly leaving baseline work unknown. New paired admission
rejects declined baseline/candidate/cache arms before accounting; six regression cases pass.
See `../development_external_objective_preflight_20260907/RESULT.md`.

The controller/launcher now support an explicit immutable `--optimization-baseline` with exact
SHA and reason, independently of unchanged `baseline_sha256`/Phase-1 verification. Separate
comparison identities are bound in analysis, checkpoint, resume and sandbox records; original
`/perf-functional-base` remains the qualification compiler. `--analysis-only` uses the real
isolated controller but skips authoring, telemetry, probes and sealing. Default behavior is
unchanged except the new declined-comparator refusal. No Phase-1 rerun was introduced.

New generic `perf/model_macs.py` leaves the matmul-specialization observer unchanged and counts
actual yielded MAC recurrences over proved static affine domains, including multi-axis convolutions.
Unsupported domains/call multiplicity remain UNKNOWN. Explicit multi-function entry selection
comes from the objective descriptor. The two new analysis helpers are host-policy pinned.
Actual retained ResNet accounting now covers **54 contractions / 4,089,184,256 MACs**:
default V19 declares 2,048,000 on mesh and 4,087,136,256 on host; explicit source-convolution
V19 declares all on mesh. This fixes the old matmul-only 100% fraction, not a timing measurement.
Per-operand footprint accounting also preserves i32 accumulator width beside i8 inputs.
58 focused source-MAC/placement/plan tests pass; see
`../development_full_model_mac_coverage_20260907/RESULT.md`.

A NEW real launcher analysis-only comparison is running in session **33030**, output
`../development_external_sourceconv_analysis_20260907/`, source snapshot
`snapshot.d5cf0052a62394a7024c24f6ba18dbdb031a00e6c1c3d4efd6b7526fbff65eb2.json`.
It compares original V19 `781808c...` against new immutable manifest-only source-convolution
variant `a5cfa426b985d920682ffb0e92f67d2ae5f2c91bd7d4e52f81112d4ac4cc195e`, on the same
normalized source `76c261...`; functional compiler remains `506333...` at 92/96. This is
development global-lowering evidence, not agent-authored improvement or numerical/performance
qualification. All simulators and model execution are disabled, with a 600-second analysis bound.
Poll session 33030 rather than restart; at this note's creation no terminal result exists.

The V19 MicroViT search session 67667 is now reaped: `budget_complete`, `promotion: unqualified`.
Only round 0's `40182b8...` was consumed. Rounds 1/2 timed out and drafts remain unaccepted.
The frozen source lacked the newer hard finalization reserve. A terminal-fence deletion was
correctly refused by wrapper ownership; its authoring logic also risks missing required native
completion when global hazard metadata is absent. Preserve the refusal. Exact terminal/fence
audit: `../development_v19_terminal_audit_20260907/REPORT.md`.

Parallel compact-caller work now has actual compiler/header-derived ABI facts (compile-only),
not host pointer guesses. Renderer/ELF integration is in progress after this source cutoff;
the current ResNet comparison does not use compact ABI. Never retroactively attribute later
renderer changes or corrected header-selection probes to the frozen comparison snapshot.

### 23.34 Source-convolution manifest failure reproduced and repaired

Session 33030 is terminal: the candidate command-buffer entrypoint failed on missing
`im2col_row` target-dialect support, then a later LLVM timeout masked the first failure.
The 784-MAC actual-source witness reproduced the same error in about 0.3 seconds;
the 1×1 witness exposed missing `loop_ws_block` support too. LLVM emission already
worked on these short inputs. Historical internal-pipeline success was not proof
that every normal manifest entrypoint worked.

New immutable V20 candidate `4f0b992b25e32c6c6db3466e8071a64dc22f292c62bb492afa5c2e618e070648`
adds the two real target-dialect operations and verifier-backed mappings without
changing the manifest, scheduler or LLVM emitter. Thirteen actual-entrypoint,
target roundtrip and malformed-geometry/signature checks pass. This remains
streamed CPU im2col + LOOP_WS, not native LOOP_CONV_WS. See
`../development_sourceconv_dialect_repair_20260907/RESULT.md`.

The shared runner now stops on first-emitter failure and baseline failure, retaining
entrypoint diagnostics; no later compiler can mask the first cause. Explicit
optimization-baseline probe APIs preserve exact comparison/source/artifact/policy
identities, distinct from both previous-iteration and Phase-1 qualification.

A fresh analysis-only full-ResNet comparison is live in session 2998 under
`../development_external_sourceconv_v20_analysis_20260907/` at this note's creation.
It uses the same 600-second analysis bound, unchanged Phase-1 92/96, and no simulator.
Re-poll rather than restart. Complete short-program CPU→device warm qualification
is still being connected; identity-only baseline accessors are not semantic proof.

Compact caller integration now has a real tiny 3-pointer→2-base compile/link proof,
5.25 seconds, with exact target-derived ABI facts and explicit logical payloads.
See `../development_compact_caller_build_attempt2_20260907/RESULT.md`. This is not
execution proof or the complete ResNet 393-pointer caller conversion.

### 23.35 Full-ResNet production preflight passes; bounded bridge advances

Session 2998 is terminal and reaped, not live. The V20 production analysis-only
preflight passed in **469.359 seconds**, within 600 seconds. It verified all
1,240 source operations across 109 tasks (53 convolution + FC + 55 host), covered
4,089,184,256 source MACs, and compiled actual target objects for both arms.
Original V19 declares only 2,048,000 MACs on mesh; V20 declares all there. Object
size increases 385,240→949,088 bytes. No whole-model execution, numeric certificate
or cycle speedup is inferred. See `../development_external_sourceconv_v20_analysis_20260907/RESULT.md`.

After that source snapshot, the shared analyzer gained a once-per-comparison
verified baseline-plan cache bound to source/LLVM/raw-CB/compiler/host-policy/proof
hashes. The source-convolution preparation helper consumes cached full proofs,
selects a real host→convolution owner, compiles complete reduced programs with both
normal manifests and retains independent signed inputs/oracles. It does not run
or admit a simulator. This prevents probe-time repeated full-model verification.

The target edge now offers an exact pinned GSIM command for the existing bounded
process-group runner, sharing argv construction with legacy execution. Fourteen
tests pass without emulator execution. The legacy environment override names an
unreceipted copy; the existing canonical `out/build/rtl_engines/gemmini/gsim/`
installation contains identical binary bytes and a bound build receipt. Selecting
that existing installation passed command preparation/revalidation without adding
permissions or manufacturing evidence. See `../development_bounded_gsim_command_20260907/RESULT.md`.

These pieces still need a real automatic complete-short-program build/run/readback
demonstration with warm1/measured1 and numerical checks. Model-wide cost ordering,
broader encodings/consumers, asynchronous lifetime reuse and sustained promotion
remain open; the full goal is not complete.

### 23.36 Search scope correction and actual launch blocker

ResNet authoring (`global_resnet50_authoring_v20_20260907`) was operator-cancelled
before an authored round when the user excluded ResNet from search. The preceding
compile-only evidence remains historical evidence, not a search result. ResNet is
post-freeze evaluation, but is not an untouched blind holdout after that inspection.

The non-ResNet M1 replacement (`global_phase2_development_v20_20260907`) stopped at
initial analysis: the baseline declined with “no region of this module is a
contraction the mesh admits.” No authored candidate or improvement was produced.
The captured source's 49 MAC operations use floating-point operands; the new
source-convolution selection must preserve the existing host-only path rather
than change their numerical semantics to force acceleration. A new candidate
repair is in progress; the failed immutable run is preserved.

The read-only progress index is `../../phase2_progress_20260907/README.md`. It
separates this development attempt from the other session's live M2 continuation.
Future launcher initial failures now write an explicit terminal-failure receipt;
the dashboard does not infer a terminal outcome merely from a missing process.

A deterministic controller regression also exposed uncharged failed probe time.
Runner failures, invalid observations and overruns now consume the original
iteration budget without earning performance receipts. Four outcome/retry checks
pass; the controller suite passed 108 tests before the final retry assertion.

The bounded complete-program builder now produces a tiny warm1/measured1 ELF in
a deadline-owned process group (10 tests, one actual build). Production masked
sandbox admission and emitted/source numerical correspondence remain separate
pending integration, not completed global proof.

Counter correction: the user reports Spike misprediction near 2×. Do not infer
FireSim timing from Spike or apply a universal multiplier. Approximately 2B
reported FireSim cycles requires over 50% reduction to go below 1B; this is an
unproven research target. Full-model simulation remains prohibited in search.

### 23.37 Capability selection repaired; a new non-ResNet segment launches

V21 (`development_full_model_codegen_v21_20260907`, SHA256
`f22f8b3cddc2242f4112c712d0170cdde3caf4eb7ed89884d3de57f203dc8085`)
selects the source-convolution extension only when an actually eligible source
convolution exists. Otherwise it uses the existing fresh mixed planner. Actual
normal-manifest pairs preserve M1's 1,403 source operations (11.41 seconds) and
M2's 454 operations/24 tasks including 12 contractions (4.21 seconds); tiny f32
and eligible 7x7 integer-convolution regression checks pass. These times exclude
subsequent verification and are not inference measurements. See
`../development_full_model_codegen_v21_proof_20260907/RESULT.md`.

The repair is host-maintained seed development, not an experiment-authored win.
The 27-symbol catalog is unchanged; no new edit authority or Phase-1 qualification
was granted. A new `global_phase2_development_v21_m2_20260907` segment was launched
with six 600-second rounds, 600-second iteration limit and 3,600-second authoring
budget, comparing authored changes against V21 itself. Its actual current state
is on `../../phase2_progress_20260907/development_v21_m2/STATUS.md`.

The new segment's **production initial analysis passed in 65.8839 seconds**,
with `ready_for_probe_admission` and no readiness blockers; the initial seed was
sealed. `global_iterations/iteration_0000.json` is the authoritative record.
Source snapshot SHA256 is
`1b464cecfda9f16f2c723b3075e5a5c811a8575da08ec5f3417ade67cfc9055a`.
This is an initial seed analysis, not an authored improvement. Semantic and
global-cost promotion blockers remain.

The generic target-cycle fitter now requires host-validated engine/configuration/
counter identity, mechanism/repetition domain and independent error evidence.
Raw engine-relative observations remain available without that authority; no
production engine was silently authorized. Controller plus mechanism tests pass
144 cases after receipt propagation. The fitter is not yet used by production
providers, and this repair does not reproduce or explain the reported Spike error.

The real complete-short-program build integration exposed a genuine dependency
boundary: the existing whole-program builder imports backend packages that the
answer policy masks. These masks were not weakened. Pure build recipe/renderer
dependencies must be separated from oracle/backend discovery before this new
automatic build path can work inside the existing sandbox. Current source-only
preparation is not runtime qualification or a model speedup.

### 23.38 First V21 round stops; JSON cache and host-guidance gaps repaired

The V21/M2 sourceworker is terminal (session 6062 reaped exit 1); it did not
complete six rounds. Initial analysis passed, but the next edited candidate
encountered `retained baseline global-plan evidence binding changed` before its
emitter ran. A subsequent revert exhausted its small remaining analysis budget.
The round audit refused, continuation recorded `recovery=stop`, and only the seed
was retained. `terminal_failure.json` and `continuation_failure_0000.json` are
authoritative; no improvement was established.

The mismatch was solely proof serialization: three integer-keyed task maps become
string-keyed over JSON, changing sorted order for task 10 versus task 2. Source,
LLVM, raw command buffer, compiler and policy pins were unchanged. The producer
now normalizes the proof to JSON document form before hashing and retaining it.
A 12-task default-verifier transport regression failed before the fix and passes
afterward. Old receipts were not rewritten or accepted under weaker checks.
A new two-analysis full-M2 production replay remains required before relaunch.

The agent feedback separately used only eight candidate-manifest surfaces despite
27 host-approved symbols. New source loads the explicitly hash-pinned host
inventory, validates its AST/component ownership, refuses descriptions outside
the unchanged authority, freezes the inventory identity and uses it in full-model
briefs and inventory-tool responses. Offline real V21/M2 replay maps 12/13 global
CCA gaps instead of zero exact-axis mappings, preserving all unknown costs and
semantic obligations. See `../development_v21_host_catalog_guidance_20260907/RESULT.md`.

The new `development_v21_loop_offload_catalog_20260907` adds only a documented
`dispatch.loop_offloaded` implementation opportunity to the existing authorized
`Scheduler.contraction` surface, with exact existing target-emission support.
It does not claim current M2 offload or authorize descriptor reuse. All 27
permissions and compiler source pins are unchanged. Source-convolution override
methods are not covered by ordinary mixed-planner surfaces; reachability must
still be checked for each selected workload.

### 23.39 Corrected full-M2 replay passes; authoring resumes

The first production cache replay failed before reaching the cache check: its
sandbox mounted the historical Phase-1 schema where the compiler searched for
the current whole-program API. The historical schema rejects `intermediate`
tensors; the current schema already permits them. The fix separately pins and
read-only mounts the current compiler API, and the host validates both emitted
command buffers even if candidate-side validation is omitted. Historical
Phase-1 bytes and the 92/96 result are unchanged; no schema was relaxed.

The actual two-worker full-M2 replay then passed: 69.00 seconds first analysis,
18.15 seconds second analysis, one baseline compilation and two candidate
compilations. The cached proof survives JSON transport with exact source, LLVM
and raw command-buffer identity. Evidence:
`../development_baseline_cache_production_replay_attempt2_20260907/receipt.json`.
No model or simulator ran in this proof.

At the user's request for immediate results, development authoring launched
while that replay completed, using the same immutable source snapshot
(`7bab569ad7e8376d059dfdc15912a778f0c944cbf6ab64b288b4c8a9f371b610`).
`global_phase2_development_v21_m2_authoring_20260907` passed its own full-model
initial analysis in 75.33 seconds and entered round 00. Its budget is six
600-second rounds / 3600 authoring seconds; source remains V21 as the explicit
optimization comparator, not a new Phase-1 qualification. Latest state:
`../../phase2_progress_20260907/development_v21_m2_authoring/STATUS.md`.

The live agent selected eligible contraction hardware-loop offload as its first
work order. This is intent, not a verified improvement. Existing paired-context
checks require the same command multiset and cannot certify replacement of
manual commands by a hardware loop. Complete reduced-source programs and their
independent numerical/runtime witnesses remain the needed evidence for that
implementation change; a provider being installed does not prove coverage.

Separately, generic mount-visibility indexing preserves the actual 408-surface
policy verdict while reducing one coverage check from 5.700 to 0.024 seconds.
That is a sandbox-check speedup, not model or full-setup performance. This later
change is not in the active authoring snapshot. Preparation feedback now uses
the same frozen host inventory as main analysis rather than silently reverting
to the smaller candidate-manifest catalog; the regression passed after first
demonstrating the missing approved surface.


## 24. 2026-09-08: the gap is now being closed on the full model

Two target-neutral compiler changes have been promoted through exact warm/measured full ResNet-50
runs on the same FireSim/U250 configuration:

| checkpoint | measured cycles | gain from prior | cumulative gain from q530 |
| --- | ---: | ---: | ---: |
| q530 slot-complete schedule | 1,704,064,223 | — | — |
| q534 native scalar conversion | 1,537,416,019 | 1.10839x | 1.10839x |
| q535 affine im2col spans | **1,316,619,699** | **1.16770x** | **1.29427x** |

Every run uses one warm inference, an untimed mutable-arena reset, one uninstrumented measured
inference, and exact checking of all 1,000 logits. Both hardware promotions used only the FireSim
queue and the exact `kill -> infrasetup -> runworkload -> kill` lifecycle.

q535 is especially important for the experiment design: its accelerator command buffer, DMA
volume, launch count, and fence count are identical to q534, yet it saves 220,796,320 hardware
cycles by deleting repeated host-side im2col address, bounds, and clamp work. The cheap Spike proxy
predicted the direction and magnitude well enough for promotion (245,561,917 predicted cycles
saved, a 1.11216x overprediction). This is the intended Phase-2 loop: complete-graph static
attribution, exact cheap A/B qualification, four-model non-regression, then one bounded hardware
measurement only for a material candidate.

The optimization is not ResNet-specific. It derives valid spans from target-neutral static
convolution geometry and was compile-qualified unchanged on TinyLLaMA, LSTMNetVIT, and SmolVLA.
Those graphs still expose different remaining gaps, which prevents a single-model result from being
mistaken for general compiler completion.

The remaining q535 gap is now sharper. It still moves 1,196,945,312 Gemmini DMA bytes, performs
14.6 MB of logical host packing, and retains 1,050 fences. Loop-matmul occupancy is only 4.065% of
the end-to-end measured window. The next high-value change is not a smaller tile tweak: form exact
non-residual and true two-tensor residual epilogues, propagate a compatible narrow encoding, make
native convolution reentrant under warm execution, and then delete packing/boundaries. Only after
that should queue depth and overlap become the primary work order.

Exact artifacts:

- `../q535_affine_im2col_hardware_result_20260908.md`
- `../q535_affine_im2col_hardware_receipt.json`
- `../phase2_affine_im2col_q535.patch`
- `../resnet50_merlin_phase2_affine_im2col_spans_w8a8_warm_measured_firesim_candidate_20260908`

### 24.1 q536 is a native-convolution opportunity bound, not the next compiler checkpoint

Queue job 536 measured **555,991,472 cycles** with all 1,000 int8 logits exact and the required
queue-only `kill -> infrasetup -> runworkload -> kill` lifecycle. It proves that 53
Merlin-generated native `LOOP_CONV` kernels, including i32 bias loading and narrow scalar
epilogues, can execute the complete ResNet convolution set on the pinned hardware. It also proves
that the native-convolution lever has enough measured headroom to cross one billion cycles.

It is not an end-to-end Merlin compiler result. The executable retains a TVM-generated host runner
for the activation arena, call graph, 20 residual blocks, first padding/max-pool, global average
pool, flatten, and dense output. Its 2.368x arithmetic ratio versus q535 therefore cannot be
reported as a fair compiler speedup. The stitcher and mixed activation arena are excluded from the
canonical compiler. Only the target-neutral kernel mechanisms and their warm/reentrant tests may
be integrated through Merlin's source graph.

The ownership audit and machine receipt are `../q536_native_loopconv_hybrid_diagnostic_20260908.md`
and `../q536_native_loopconv_hybrid_receipt.json`. q535 remains the accepted compiler-owned
hardware checkpoint until the same mechanisms pass Merlin's four-model compile gate and exact
warm/measured full-model qualification.


### 24.2 Final combined compiler release

The directly invocable Phase-2 compiler is now:

`../development_phase2_final_combined_exact_residual_global_encoding_20260908`

Its compiler tree SHA-256 is
`48694957d14c9608960f7ac2b6cd08b72b15c55e4a11ff26e1cf952e0cd7607e` over 45 files; the review
patch is `final_combined_compiler.patch`, SHA-256
`0e611e3c9c2231196653bfdb906a45843b89f8ee14f4ffbefe683116a9f2477a`. The release verifier
passes and binds 65 focused tests, two exact same-process warm/measured witnesses, all four complete
model compiles, the opt-in transformer bridge, source ownership, ABI, refusal/encoding evidence,
the final receipt, and q535 hardware lineage.

The artifact composes general mechanisms only: q534 native scalar lowering, q535 affine segmented
im2col, exact ordered non-residual epilogue formation, exact true second-tensor residual formation,
global encoding/fanout/lifetime/capacity planning, fail-closed native convolution, warm-safe
read-only zero-bias initialization, and the explicit default-off dynamic-weight bridge. Integration
regressions enforce unique `bias -> scale -> activation` order, real-bias hazard reads, mixed
source-convolution plus exact-matmul formation, post-fusion encoding facts, and the preservation of
all geometry/layout/overflow/reduction-residency guards. No model, layer, or fixed-shape dispatch is
present.

Exact cheap qualification establishes both the positive mechanisms and the current boundary:

- The native epilogue witness checks 323 outputs after one warm run and measures 56 Spike proxy
  cycles, versus 13,760 before fusion. It removes two host tasks, 1,292 full-width intermediate
  bytes, 969 output-DMA bytes, and one logical device-to-host boundary.
- The global-encoding witness checks 96/96 outputs after one warm run and measures 102 proxy cycles.
  Two native convolutions retain an internal i8 NHWC value, with zero im2col tasks and one true
  device-dependency fence.
- ResNet selects 15 exact residual formations. Relative to q535's compiler it changes tasks
  109->105, host tasks 55->51, ABI pointers 393->389, and physical intermediate storage by
  -7,340,032 bytes. Its byte-identical residual-child target transfers the exact warm/measured
  Spike result 506,265,226->492,147,976 cycles (2.7885%; 1.028685x), with 1,000/1,000 logits exact.
  This changed object has no hardware result; q535 remains the latest honest full-Merlin FireSim
  checkpoint at 1,316,619,699 cycles.
- ResNet admits 0/53 exact native-narrow convolutions. All 53 carry per-channel scale and FP32 bias
  not exactly representable by the current one-scalar-scale, narrow-store `LOOP_CONV` endpoint.
  Conv1's naive fold mismatches 143/802,816 values before pooling and 33/200,704 after pooling.
  The conditional boundary estimate 44,455,936->11,113,984 bytes is opportunity, not realized work.
- TinyLLaMA, LSTMNetViT, and SmolVLA compile sequentially and non-regress. Byte identity is not a
  performance win. TinyLLaMA's bridge remains explicit opt-in, non-bit-exact, and not a dataset
  accuracy claim.

The exact target-neutral next contract carries scale granularity/axis, zero points, bias domain,
ordered floating-point stages, residual domains, rounding, saturation, and an output endpoint
`{narrow_i8, full_i32, accumulator_handle}`. Legal routes are full-i32/accumulator output followed
by the exact fused host/RVV epilogue; narrow native output only after equivalence proof; or the
existing fallback. A native-aligned W8A8 flow is a separate calibration/golden/accuracy/provenance
contract. The q536 555,991,472-cycle hybrid remains an opportunity bound, not a Merlin speedup.

No final-combined FireSim, L3, or queue action occurred. No sub-billion compiler claim is made; no
formal roofline is inferred; Phase 1 remains frozen at 92/96.

Downstream use:

```sh
artifact=/scratch/agustin/projects/oscar-merlin/out/artifacts/perf-bench/gemmini/development_phase2_final_combined_exact_residual_global_encoding_20260908
PYTHONDONTWRITEBYTECODE=1 python "$artifact/verify.py"
MERLIN_PYTHON=/scratch/agustin/projects/oscar-merlin/.venv/bin/python \
  "$artifact/run-gemmini-opt" --source-convolution --convert-iface-to-gemmini \
  --emit-command-buffer=/absolute/output/command_buffer.json \
  --emit-target-artifact -o /absolute/output/target.mlir /absolute/input/model.mlir
```

Object/ELF construction is runner-owned and must use the selected target's pinned compiler, ISA,
linker script, harness ABI, and hardware facts. Evidence index:

- final report/status/receipt/verifier/patch:
  `../development_phase2_final_combined_exact_residual_global_encoding_20260908`
- q535 hardware result/receipt/bundle: `../q535_affine_im2col_hardware_result_20260908.md`,
  `../q535_affine_im2col_hardware_receipt.json`,
  `../resnet50_merlin_phase2_affine_im2col_spans_w8a8_warm_measured_firesim_candidate_20260908`
- q536 diagnostic/receipt: `../q536_native_loopconv_hybrid_diagnostic_20260908.md`,
  `../q536_native_loopconv_hybrid_receipt.json`
- feature receipts: `../development_phase2_exact_native_epilogue_20260908/validation/exact_native_epilogue_receipt.json`,
  `../development_phase2_canonical_multimodel_affine_im2col_residual_epilogue_20260908/validation/residual_integration/receipt.json`,
  `../development_phase2_canonical_affine_global_encoding_20260908/validation/global_encoding/receipt.json`
- quantization/capability audit:
  `../universal_narrow_epilogue_recovery_v1_20260908/resnet50_narrow_epilogue_inventory.json` and
  `../universal_narrow_epilogue_recovery_v1_20260908/conv1_native_requant_proof.json`
