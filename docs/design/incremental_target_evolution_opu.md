---
title: "Design: incremental target evolution — RVV + Saturn OPU as the driving delta"
kind: design
status: draft
owner: targetgen
last_verified: 2026-08-10
related: [beam_cca_architecture, target_onboarding, lowering_pipeline, llvm_integration, reproducibility]
code_refs:
  - merlin/python/merlin/targetgen/compute_units.py
  - merlin/python/merlin/targetgen/families.py
  - merlin/python/merlin/targetgen/routing.py
  - merlin/python/merlin/targetgen/capability_manifests.py
  - merlin/python/merlin/targetgen/rtl/spatial_introspect.py
  - merlin/python/merlin/llvmlower/custom_isa.py
  - merlin/python/merlin/llvmlower/perop_blocks.py
  - merlin/python/merlin/llvmlower/impr_features.py
  - merlin/python/merlin/kernels/cca.py
  - merlin/python/merlin/kernels/cca_contract.py
  - merlin/python/merlin/rvvgen/registry.py
  - merlin/python/merlin/kernels/census.py
  - merlin/python/merlin/kernels/shapes.py
  - merlin/python/merlin/llvmlower/passes_quant_int.py
  - merlin/python/merlin/llvmlower/op_profile.py
---

# Incremental target evolution — RVV + Saturn OPU as the driving delta

## The question

Merlin has a certified RVV int8 target: `out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8`,
`status: k1_verified`, `publication.champion: true`, fingerprint `2255a08a…`, certified by run
`k1_int8_wholemodel_ab_tiny_llama`. It compiles whole models (deepjscc, spectformer, whisper) to
multicore-RVV binaries.

The hardware is now evolving from `CPU + RVV` to `CPU + RVV + OPU`, where the OPU is Saturn's
outer-product unit — a small ISA extension with a matrix accumulator, not a new accelerator.

This document is **not** a plan to add OPU support. It asks:

> Given a certified target **T** and a hardware delta **ΔH**, can Merlin synthesize **T + ΔH** by
> reusing every artifact whose semantic contract is unchanged, generating only what ΔH invalidates,
> and then deciding — **by measurement, not by legality** — when the new capability should replace or
> coexist with the existing lowering path?

Concretely `T` = the RVV target above, `ΔH` = the OPU, and the intended end state is *not* a second
independent compiler and *not* a large new dialect, but:

```
                            existing certified RVV path
                          /                            \
 linalg / model IR -------                              --> executable
                          \                            /
                            new OPU lowering path

                                     ^
                          legality -> candidates
                                     ^
                      measured cost / CCA / search selects
```

Two separable notions of "incremental" run through everything below, and conflating them is the main
way this work could go wrong:

- **Structural incrementality** — which *compiler artifacts* must be regenerated after ΔH.
- **Semantic incrementality** — which *equivalent implementations* become legal after ΔH.

## 1. What the OPU actually is

### 1.1 Derived semantics, cross-checked three ways

Everything in this section is derived from the target's own sources, never from prose. Three
independent sources agree, which is what makes them usable as facts:

1. **Saturn RTL** — `$MERLIN_CHIPYARD/generators/saturn` @ `opu-int8` `ea373800`:
   `src/main/scala/common/Consts.scala` (`OPMFunct6`), `src/main/scala/exu/OuterProductUnit.scala`
   (`OPUParameters`, `HasOPUParams`), `src/main/scala/exu/OuterProductSequencer.scala`,
   `src/main/scala/common/Parameters.scala` (`opuParams`, `opuInsns`).
2. **Expert software** — `$MERLIN_CHIPYARD/generators/saturn/benchmarks/common/bme.h` plus the nine
   `benchmarks/opu-*` kernels.
3. **XNNPACK's `opu` branch** — `src/xnnpack/bme.h` and the generator template
   `src/qs8-gemm/opu-rvv.c.in` (fetched; not checked out locally).

All four OPU ops live in OP-V (`opcode 0x57`) in the **reserved even `funct6` slots** — the odd
neighbours 41/43/45/47 are `vmadd`/`vnmsub`/`vmacc`/`vnmsac`, so the choice is collision-free.

| mnemonic (RTL name) | funct6 | funct3 | operands | semantics |
| --- | --- | --- | --- | --- |
| `VOPACC` (`opmacc`) | 40 | 2 (OPMVV) | `md`, `vs1`=LHS rows, `vs2`=RHS cols | `md[i][j] += vs1[i] * vs2[j]`, int8×int8 accumulating in int32 |
| `VMV_RV` (`opmvin`) | 42 | 6 (OPMVX) | `md`, `rs1`=row **value**, `vs2` | write `vs2` into row `rs1` of `md` |
| `OPMVINBCAST` | 44 | 6 (OPMVX) | `md`, `rs1`=`x0`, `vs2` | broadcast `vs2` down **all** rows of `md` |
| `VMV_VR` (`opmvout`) | 46 | 6 (OPMVX) | `vd`, `rs1`=row **value**, `ms2` | read row `rs1` of `ms2` into `vd` — the only readout path |

Consequences the compiler has to respect:

- **`vl` is load-bearing on every one of them.** `VOPACC` consumes `vl` LHS rows and `vl` RHS columns;
  `VMV_VR` writes `vl` int32 elements. This is the origin of the narrow-M failure in §1.3.
- **Geometry is a derived fact, not a constant.** Physical MACC array is `(dLen/8) × (dLen/8)`; one
  *logical* matrix register is `(vLen/8) × (vLen/8)` int32 held as `(vLen/dLen)²` physical subtiles;
  there are `nMrfRegs = 4` matrix registers. The `HW = 16` appearing in expert kernels is
  `dLen/8` for the `vLen = dLen = 128` configuration they were written against — it is **not** a
  property of the OPU. Merlin reads `tile.rows`/`tile.cols`/`mrf_depth` from
  `targetgen/rtl/spatial_introspect.py`'s fact bundle.
- **No saturation — and the hazard is unreachable, so it is retired.** `OuterProductUnit.scala` carries
  `// TODO: Need to check for overflow and saturate to accumulator width`, so where int32 accumulation
  overflows it **wraps**, and any numerical reference must wrap too. But the bound is arithmetic:
  int8 × int8 into int32 cannot exceed the accumulator below `K = (2³¹−1)/127² ≈ 133,144`, and the
  longest reduction in the workload census is three orders of magnitude below that (whisper's largest is
  1152). So on the int8 datapath the missing saturation cannot be provoked by any contraction a model
  produces. `kernels/opu_corpus.reference` still wraps — the reference must match the hardware wherever
  the hardware is defined — but the corpus deliberately contains **no** wrap case, because writing one
  would require a `K` no model emits, and a case that claims to exercise wrapping while staying inside
  the range is worse than none. The wrap is unit-tested directly on synthetic wide values.
- **Bias init is one instruction.** `vle32.v v0, (bias)` then `OPMVINBCAST` broadcasts a bias row
  across the tile. Zero-init is the same instruction after `vmv.v.i v0, 0`.
- **Operand layout is forced, not chosen.** Saturn's own `benchmarks/opu-gemm/kernel.h` indexes
  `at[k*M + i]` and `b[k*N + j]`: **both operands are K-major**, so the LHS must be transposed/packed.
  That packing is a real cost term, not an implementation detail — see §4.3.
- **Readout is row-serial** and row 0 is the slowest (`scalar_row_latency = (yDim+1) -
  scalar_cluster_row_idx`), so extraction cost scales with rows, which is what makes an
  accumulator-resident epilogue worth representing.

#### 1.1a Every field in that table is reachable structurally, from four files

The table above is not transcribed from prose — the derivation chain was walked and each link confirmed,
which is what Phase 2's `opu_isa.py` implements (and cross-checks against `bme.h`, agreement being the
evidence). No regex is required at any step; every link is a tokenizer over Scala declarations.

| field | source | how it is read |
| --- | --- | --- |
| opcode `0x57` | `common/Consts.scala`, `trait HasVectorConsts` | `def opcVector = "b1010111".U` |
| funct3 | same trait | `def OPMVV = "b010".U(3.W)` → 2; `def OPMVX = "b110".U(3.W)` → 6 |
| funct6 | `common/Consts.scala`, `object OPMFunct6 extends ChiselEnum` | a `ChiselEnum`'s value IS its ordinal, so funct6 = the count of `Value` slots declared before the name, counting `val a, b = Value` as two and a placeholder `val _ = Value` as one. Walking it yields `opmacc = 40`, `opmvin = 42`, `opmvinbcast = 44`, `opmvout = 46` — and shows the odd neighbours they sit between (`madd` 41, `nmsub` 43, `macc` 45, `nmsac` 47) |
| which ops exist, and each one's VV/VX form | `common/Parameters.scala`, `def opuInsns` | `OPMACC.VV`, `OPMVIN.VX`, `OPMVINBCAST.VX`, `OPMVOUT.VX` — so `VOPACC` is OPMVV (funct3 2) and the other three are OPMVX (funct3 6), which is the split the table shows |
| operand roles | `insns/Instructions.scala` | `object OPMACC extends OPMInstruction { val props = Seq(F6(OPMFunct6.opmacc), ReadsVS1.Y, ReadsVS2.Y, WritesVD.N) }`. `ReadsVS1`/`ReadsVS2`/`WritesVD` are the roles; note `OPMVOUT` is the only one with `WritesVD.Y`, which is why it is the only readout, and the three `.VX` forms gain their scalar operand from `OPMVXInstruction` not adding `ReadsVS1.Y` |

The `ChiselEnum`-ordinal step is the one worth guarding: it makes funct6 a function of *declaration
order in a file*, so an upstream edit that inserts a `Value` silently shifts every later opcode. That is
precisely why the derivation must be re-run per RTL revision rather than cached as a constant, and why
disagreement with `bme.h` must fail closed instead of picking a side.

### 1.2 There is no `+xopu`, and no LLVM fork

`+xopu` appears **nowhere** in this repo, and no march token is needed. Every real OPU binary — the
Saturn benchmarks, the Merlin-IREE ukernel, XNNPACK's generated kernels — is raw `.insn r` on stock
LLVM, with matrix registers `#define`d to integer-register spellings (`#define m0 "x0"`) purely so
`.insn r` will accept them. `bme.h` says so directly: *"HACK reuse the scalar registers to avoid
assembler hacking for now"*. There is no assembler support for the mnemonics anywhere.

This matches Merlin's standing no-forked-toolchain rule, and the seam already exists and names this
exact case — `merlin/python/merlin/llvmlower/custom_isa.py:1-21`:

> The escape hatch for a truly novel encoding (e.g. a Saturn vcix instruction) is the assembler
> `.insn` directive inside the inline asm … Standard rv64gcv needs none of this; it is the on-ramp
> for Saturn custom instructions.

So the OPU delta needs **no** LLVM change, and `rvvgen/registry.py:25`'s cflags allowlist already
admits `-march` without code changes if one is ever wanted.

### 1.3 The historical failure is the acceptance bar

Merlin-IREE (`third_party/baselines/merlin-iree`, `docs/dev_blog/2026-04-06-opu-utilization-e2e-benchmarking.md`)
records an OPU integration that reported **100% matmul OPU coverage** on multiple models and then hung
for 12+ hours on the first warmup iteration of an MLP. Verbatim:

> The original "100% MLP OPU coverage" claim above was based purely on **static analysis** of the
> linked binary (counting `.insn r 87 ...` opcodes). MLP had never actually been executed end-to-end
> on FireSim until the new `bench_model_*` runner was wired up — at which point the runtime bug
> surfaced.

Root cause, in order: all three MLP matmuls were **vecmat** (rank-1 LHS), so the encoding's
contraction-dim inference produced `M0 = 1`; the kernel assumed the hardware width; `vle8.v` read 15
bytes past the LHS panel; `VOPACC` accumulated the garbage. Compounding it, **LLVM's
`RISCVInsertVSETVLI` treats a standalone `asm volatile vsetvli` as a no-op for vl tracking**, so both
loads used `vl = 16` regardless. The fix was to fuse `vsetvli`+`vle`+`vsetvli`+`vle`+`vsetvli`+`VOPACC`
into a single opaque `asm volatile` block. Their coverage tool still has **no counter for `VMV_RV`**,
so the accumulate/bias-reload path is invisible to it even now.

Three rules follow, and they are enforced rather than advised:

1. **Static `.insn` occurrence is never evidence of support.** It may be reported only as
   `emitted_opu_ops`. "Coverage" is cycle-weighted and execution-gated.
2. **Narrow-M is a named regression test**, from `M = 1` upward, not a corner to revisit later.
3. **A test with no oracle records `not_run`**, never a pass.

### 1.4 The local spike fork is not a usable model

`opu-bme-spike` (branch `opu-bme`) implements the extension in 464 lines and disagrees with the RTL in
five confirmed ways: `funct3 = 7` where RTL and `bme.h` use 6 (so **no `bme.h`-built binary decodes**);
the row index read from the `rs1` *field* instead of its *value* (and no model of the RTL's bit
swizzle); an int8-truncating overwrite register file with a bolted-on int32 side array; `vl`/`vtype`
ignored entirely (`get_tlen()` hardcoded to 512, so it only functions at `VLEN = 512`); and an FP8
opcode **bit-identical to `vmadd.vv`** which, because custom instructions are matched first, silently
turns every `vmadd.vv` in ordinary RVV code into an FP8 outer product.

Decision: **the OPU has no functional simulator in this pass.** Correctness comes from RTL. The
consequences are stated in §5, not worked around.

## 2. Datapath class is not software exposure

This is the structural blocker in Merlin today, and the one core refactor the work requires.

`targetgen/families.py:54-69` keys a `FamilyProfile` off the compute-unit `kind`, and the `spatial`
profile sets `endpoint_kind_default = "command_buffer"` with the rationale (`:61-66`) that the OPU is
*"a grid of accumulator cells driven by a COMMAND BUFFER over one-hot op ports (macc/mvin/shift) — NOT
a RoCC command ISA"*.

That is **correct about the datapath and wrong about the exposure.** The one-hot op ports are how the
OPU is driven *inside the vector unit*; software drives it with vector instructions in OP-V. Those are
independent axes:

| axis | OPU's value | what it decides |
| --- | --- | --- |
| datapath family | `spatial` (rank-1 outer-product accumulate) | RTL fact extraction, perf fields, grading tiers |
| software exposure | `.insn` on stock LLVM (`inline_asm_insn`) | which codegen artifact is produced |
| state model | matrix accumulator, `nMrfRegs` deep | dependence/legality reasoning |

`endpoint_kind` is *already* an overridable contract field with FACTS > residual > family-default
precedence (`capability_manifests.py:424-431`), and `_endpoint_from_facts:264-287` already derives it
from the decoder rather than from a name. The blocker is narrower than it looks: a manifest carries
exactly **one** `kind` and **one** `endpoint_kind`, both taken from `_primary_kind(units)`
(`target_experiment.py:190-209`). A hybrid RVV + OPU target has two units with two different exposures
at once, and cannot be expressed.

Fix: exposure becomes a **per-compute-unit** axis (`ComputeUnit.exposure`), resolved unit → target
`endpoint_kind` → family default, so every existing single-unit target resolves exactly as it does
today. `spatial` keeps `encoding_required = False` and no `rocc_insn` trace gate — the OPU has no funct
decode table in the RoCC sense; its encodings are derived per §1.1 and cross-checked.

The general rule this generalises to, and the one to hold when the next target arrives: **do not let a
hardware class imply a software exposure.** `if target == "saturn"` is not the failure mode to guard
against here — the failure mode is a taxonomy that cannot express the machine.

## 3. Routing must return candidates

`targetgen/routing.py:55-74` returns one `RouteResult` per demand and selects the **first legal unit in
contract-declaration order**. There is no cost model, no preference, no tie-break, and no way to see
the alternatives.

For a hybrid target that is not merely limited, it is wrong: an int8 matmul is legal on **both** RVV and
OPU, and legality is not profitability. Which one wins depends on `M`/`N`/`K`, layout, whether the
operands are already K-major, whether the epilogue can stay accumulator-resident, and dispatch
overhead. So:

```python
route_candidates(demands, units) -> list[RouteCandidates]   # every legal unit, honest gap when none
select(candidates, context, cost_model) -> list[RouteResult] # separable, pluggable, ablatable
route(demands, units) -> list[RouteResult]                   # first-candidate wrapper: unchanged today
```

Keeping `route()` as the wrapper preserves the four tests in `merlin/tests/targetgen/test_routing.py`
and both existing consumers (`frontends/gguf_reader.py:134`, `frontends/adapters/gguf.py:54`).

Cost models are swappable specifically so the ablation is real: `eager` (every legal contraction →
OPU) exists as a deliberately bad baseline, and `measured` interpolates the Phase 2 microbenchmark
corpus rather than inventing architectural constants.

## 4. What is reused, and what ΔH invalidates

### 4.1 The parent is reused literally, and stays a control

The OPU delta is expressed as an **additive, default-OFF** feature on the certified parent package —
not a new pipeline. Two invariants already enforced in-tree make this checkable rather than asserted:

- `impr_features.py:11-15` — with `features == frozenset()` the emitted pipeline and schedule are
  byte-identical to baseline (guarded by `test_impr_features`).
- `apply.py:1-7` — `apply_rvv_package(hand_v0)` is byte-identical to `build_app(backend="rvv")`.

So `RVV_TRANSFORM_SCHEDULE` and `build_rvv_pipeline`'s base pass list are not touched, and "old target
+ no OPU feature" is byte-identical by construction, not by inspection.

### 4.2 The artifact DAG

Structural incrementality needs the generation pipeline to be a dependency graph. Today
`targetgen/pipeline.py:100-127` builds a straight-line list and `write_all` overwrites every file;
`EMIT_LAYERS` is a *manual* layer selector, not staleness-driven. Nothing anywhere models a parent
target, a content hash that gates regeneration, or an invalidation set — the closest primitives are
`publish._fingerprint` (a composition of ids, `publish.py:714`), `rtl_check_runner._facts_sha` (a real
content sha, `:60-72`), per-package `manifest["lineage"]` (RVV schedules only), and the unused
`ProductDir.sources` edge list.

```
        RTL / hardware evidence  (facts_sha)
                    |
            capability contract
                    |
        +-----------+------------------+
        |                              |
  semantic / compiler IR         runtime contract
        |
     lowering  ---> codegen ---> CCA / action catalog
        |                    \
        +---------------------> tests / certification
```

An edge means "changing the source invalidates the sink". ΔH = OPU touches hardware evidence and the
capability contract; it should invalidate capability routing, OPU lowering, OPU codegen, the new CCA
routes and the OPU tests, and it should invalidate **nothing** in the RVV schedule, the generic
lowering, the board/runtime support or the elementwise path. `reuse_ratio = reused / relevant` is then
a measured quantity. Line counts are never the result on their own.

### 4.3 Where the OPU lowering attaches

Ranked by how little existing machinery is disturbed:

1. **A default-OFF `ImprFeature`** (`impr_features.py:28-50`) — the intended seam, enabled per-package
   through `knobs.yaml: compiler_features`.
2. **Per-contraction tagging** — `llvmlower/perop_blocks.py` already solves exactly "route *this*
   contraction differently from *that* one", and its docstring records the measured reason the tag must
   be applied after `linalg-specialize-generic-ops` and before `transform-interpreter`: a tag set at
   prepare time does not survive specialization ("20 ops renamed, 0 kept the tag").
3. **`custom_isa.py`** for the `.insn` emission itself.

The microkernel is compiled alongside the model object the way `runtime/c/merlin_op_prof.c` already is.
Its epilogue shape is the interesting compiler question, and it is the one the XNNPACK template answers
in the affirmative: extract a row with `VMV_VR`, convert, scale, clamp, narrow and store **without
round-tripping int32 through memory** — i.e. accumulator-resident requantization. That maps onto
`cca.ComputeFacet.accumulator_resident`, which already exists as the shared target-agnostic concept
with a PASS → CODEGEN escalation ladder.

## 5. Verification ladder, and what this pass does *not* verify

| rung | what it proves | oracle |
| --- | --- | --- |
| L0 | contract legality | capability contract |
| L1 | compiles | clang-23 |
| L2 | emitted-code audit | `kernels/decode/opu.py` (structural, derived opcodes) |
| L3 | microkernel numerics, frozen shape corpus | **OPU Verilator RTL** |
| L4 | single-dispatch numerics incl. epilogue | **OPU Verilator RTL** |
| L5 | whole-model numerics | **none in this pass** |
| L6 | whole-model cycles | **none in this pass** |

RTL sims for `OPUV256D128ShuttleConfig`, `OPUV128D64ShuttleConfig`, `OPUMXV256D128ShuttleConfig` and
`GemminiAndOPUShuttleConfig` are already built under `$MERLIN_CHIPYARD/sims/verilator`, and
`zephyr_model.run_on_verilator(elf, config=…)` (`:1533`) already takes the config as a parameter, so
RTL execution is a parameter away rather than new code. Caveat: in that checkout the OPU config source
is a `.bak` file, so the binaries exist but are not currently rebuildable there.

**The honest bound.** Verilator is ~10⁴ cycles/s and deepjscc is ~4.6×10⁸ cycles, so L5 and L6 are not
reachable with it, and (per §1.4) there is no functional simulator to stand in. Therefore:

- Whole-model rows record `not_run` **with the oracle that would close them** — a repaired OPU spike
  model for L5, an OPU FireSim bitstream for L6. They are never reported as passes.
- Any whole-model number derived by extrapolation is labelled a projection and never printed adjacent
  to a measured value without that label.
- Correctness therefore rests entirely on L3/L4 — which is precisely the surface the historical failure
  escaped through. That is the argument for the frozen corpus being exhaustive rather than
  representative, and for `M = 1` being in it from the first commit.

Kodiak, for the avoidance of doubt, **has no OPU**: `boards.board("chipyard_kodiak")` is 3 harts / 2
vector harts / VLEN 512 with no OPU anywhere in `runtime/boards.py`. The OPU driving example executes
on RTL configs, not on that board.

## 6. Experiment matrix

| arm | what it isolates |
| --- | --- |
| A | RVV baseline — the parent, as a control |
| B | hand OPU — the performance oracle / ceiling |
| C | full regeneration of RVV+OPU as if it were a fresh target |
| D | incremental TargetDelta — reuse the parent, generate only ΔOPU |
| E | D + kernel-mined CCA evidence |
| F | eager OPU — every legal contraction routed to OPU (intentionally bad) |
| G | persistent RVV/OPU alternatives — delayed selection |
| H | full incremental system |

`merlin/experiments/targetgen_evals/` is extended rather than duplicated: a `methods/v7_target_delta/`
arm, a `parent_run` field in `harness/schemas/run_manifest.schema.json` (no parent-run concept exists
today — `compare_runs` groups only by method), and evolution metrics added in the three places that must
agree (`collect_metrics._ALL_COLUMNS`, `build_summary`, `harness/schemas/metrics.schema.json`).

Products go under `out/artifacts/target-evolution/<target>/v<ver>/…` via
`common/artifacts.new_product(...)` — never a hand-built path — carrying `provenance.yaml`,
`parent_target.yaml`, `target_delta.yaml`, `dependency_graph.json`, `invalidation_report.json`,
`workload_census.json`, `routing_report.json`, `cca/`, `eqsat/`, `assembly/`, `microbench/`,
`correctness.json`, `performance.json`, `summary.{json,md}`.

The ablations these are built to answer: is incremental synthesis cheaper than regeneration *at equal
correctness*; does it preserve parent behaviour better; does mined evidence improve the delta; does
eager routing hurt; do persistent alternatives select better; does a measured cost model beat "always
OPU"; does whole-model search choose differently from microkernel ranking; does the synthesized delta
approach the hand implementation; and **where does RVV remain preferable even though OPU is legal**.

## 7. Semantic incrementality (experimental)

xDSL 0.68 ships an `equivalence` dialect (`equivalence.class`, `equivalence.const_class`,
`equivalence.graph` — a graph region permitting cycles — `equivalence.yield`) and six eqsat passes
(`eqsat-create-eclasses`, `eqsat-create-egraphs`, `eqsat-add-costs`, `eqsat-extract`,
`eqsat-serialize-egraph`, `apply-eqsat-pdl`). Merlin uses none of it today.

The intended use is narrow and concrete: before ΔH, `matmul ≡ RVV impl`; after ΔH,
`matmul ≡ RVV impl ≡ OPU impl`, with the equivalence **retained past** packing, tiling, layout and
epilogue formation so that accumulator-residency and QDQ-fusion decisions are made with downstream
information instead of by eager first-legal selection.

Scope is bounded by the source paper's own limitation — Tamagoyaki (arXiv:2602.16707) restricts itself
to *"pure, straight-line functions"* and does not explore structured control flow — so the e-graph
covers one contraction plus its epilogue as a pure region. Never a whole model, never control flow.

Two hypotheses, both ours to falsify:

- **H-EQ1** — persistent alternatives select better than early extraction, measured against the
  `measured` cost model on held-out shapes.
- **H-EQ2** — `saturate(E_RVV, Δ_OPU) ≡ saturate(program, RVV ∪ OPU)`. **The paper is silent on
  incremental re-saturation**, so this is a hypothesis and not a result to be inherited. Measured on
  rebuild vs incremental time, e-node/e-class counts, memory, whether extraction matches from-scratch,
  and stale-equivalence hazards.

Compile-time cost is a first-class reported result: the paper measures a **401× geomean slowdown vs
egg** with only 18.6% of runtime inside saturation. *"EqSat infrastructure retains both
implementations; performance benefit is not yet established"* is an acceptable outcome and must remain
sayable.

## 8. Naming hazards in this repo

Three unrelated things are called "outer product" here, and one is called "OPU" but is not:

1. `lower_contraction lowering_strategy = "outerproduct"` — an **MLIR vector lowering strategy**
   (`impr_features.py`, `rvv_knobs.py:68-70` where it is recorded as a proven no-op). Nothing to do
   with this hardware.
2. `cca.ComputeFacet.contraction_form = "outerproduct"` — a declared but **dead** token; no lifter
   emits it. Phase 5 makes it live for the real hardware.
3. The Saturn OPU itself — `compute_units.py:30`, `families.py:61`, `rtl/spatial_introspect.py`.
4. `runtime/backends/zephyr_model.py:5-7` uses "Saturn-OPU" to mean the **RVV vector tile** of the
   `GemminiAndOPUShuttleConfig` SoC, not the outer-product unit.

Any new code or prose must disambiguate, and reviewers should treat an unqualified "outer product" in
a diff as a defect.

## 8b. What the workload census established

The census (`merlin/python/merlin/kernels/census.py`, one row per contraction joining extents ×
element types × measured ticks × contract legality) was run on the three driving models. Four results
change how later phases are built.

**The IR stage decides the element types, and therefore every legality verdict.** An `int8` bundle's
`model.mlir` is f32 — the W8A8 rewrite is a pipeline pass, so the capture carries quantized *weights*
and an f32 graph. Measured on deepjscc: all 20 contractions read `(f32, f32, f32)` from the capture and
`(i8, i8, i32)` from the prepared module. A census of the capture therefore reports an int8-only unit as
legal for **nothing at all** — the wrong answer, produced by observing the wrong stage. Every legality
claim in this programme must name the stage it was made at.

**The int8 rewrite used to destroy the join key on exactly the ops that matter.** The pass replaces one
captured contraction with an integer contraction plus a requant epilogue, and carried `prov.*` to the
epilogue only. Since a profile's join key falls back to the MLIR op name, all 20 deepjscc contractions
collapsed into one `linalg.generic` bucket while the 383 untouched elementwise regions kept their keys —
so no per-layer cost attribution was possible for the ops that dominate the arithmetic. Both pieces now
carry the source provenance plus a `prov.role` that separates a contraction's own cost from its
epilogue's. Without that, restoring the key would have traded one imprecision for another.

**Narrow-M/N is a correctness hazard, not a throughput lever.** Contractions with a unit parallel extent
are numerous and arithmetically negligible: whisper has 49 of 91 rows with `min(M, N) == 1`, together
0.77% of contraction work; deepjscc has 8 of 20, 0.03%. So `M = 1` earns its named regression test
because it is where the prior integration read 15 bytes past a panel — not because routing it to the OPU
would win anything. Phase 4's cost model should expect to *decline* these, and the frozen corpus must
still cover them exactly.

**`batch_matmul` is a first-class routing question, not an edge case.** It is 24 of 91 rows in whisper
and 16 of 106 in spectformer. A unit whose contract declares only `matmul` gaps every one of them, which
is the correct fail-closed reading of that contract and also a large hole in coverage — the contract has
to say which it computes. Spectformer additionally presents 48 contractions from `fft_rfft2`/`fft_irfft2`
(a DFT expressed as a contraction), a shape family none of the OPU expert kernels resemble.

**The measured ranking disagrees with the arithmetic one, which is why it was worth measuring.** A K1
per-op profile of `whisper_tiny_375pos` int8 against the certified champion gates clean (cos 0.9999999,
profiler coverage 0.999, 5124 marks) and its perturbation against an un-instrumented control is 0.445% —
inside the 1.9% board noise floor, so the breakdown is usable. Joined into the census:

| | rows | share of measured whole-model time (upper bound) |
| --- | --- | --- |
| all contractions | 91 | 96.01% |
| legal on a `matmul`-only int8 unit | 67 | 85.79% |
| illegal (all `batch_matmul`) | 24 | 10.22% |

The heaviest contraction by measured cost — `384×375 / K=1152`, 16.17% of the model on its own — ranks
**9th of 91** by arithmetic; the ops above it on work are each under 5.6% of measured time. A FLOP-ranked
census would have put the single most expensive contraction in the model ninth, which is exactly the
substitution the plan forbids.

Two reporting hazards this surfaced, both now handled in the tool rather than in prose:

- **Per-row shares are not additive.** Several contractions can join one provenance bucket — an attention
  layer's two contractions carry one `prov.fqn` — so summing the 91 rows gives **106.23%** of a model that
  is by definition 100%. The census reports a deduplicated aggregate (79 buckets for 91 rows) and prints
  the warning next to the column.
- **Every share here is an upper bound**, because a bucket that covers a contraction also covers whatever
  else shares its key: the quantize prologue and requant epilogue the int8 rewrite attributes to the same
  layer. That is why "contractions are 96% of the model" must be read as a ceiling, not a measurement of
  the contractions alone. Separating them needs a `prov.role` on the prologue too, which is a Phase 4
  question because that is where the packing/quantize cost term is priced.

**One of three models yields a usable measured ranking; the census declines the other two by name.**
A profile is only joined when it passes its OWN gates, read from the profile rather than re-derived, so
the census cannot disagree with the tool that made the measurement:

| model | profile verdict | census ranking |
| --- | --- | --- |
| `whisper_tiny_375pos` | gated, perturbation 0.445% ≤ 1.9% floor | **ticks** (measured) |
| `spectformer` | numerics exact (cos 1.0000000) but perturbation **2.152% > 1.9% floor** — the producer does not stand behind the breakdown | work, with the reason recorded |
| `deepjscc` | did not gate: cos 0.9176, `rel 0.889` | work, with the reason recorded |

Both refusals are informative rather than merely blocking:

- **spectformer** is a *profiler* limit, not a compiler defect. Its numerics are exact; the per-op marker
  calls simply cost more than the board noise floor on a 25 s model with 5645 marks. Closing it means
  fewer marks (instrument only the contractions) or a longer run to shrink the relative overhead —
  neither of which changes the compiler. Joining it anyway requires an explicit
  `--allow-unusable-profile`, and the census then labels the ranking as resting on a rejected measurement.
- **deepjscc** is a real divergence and is *not* golden selection: the weight-only and W8A8 goldens agree
  with each other at 0.99995, so the board output disagrees with both. A pre-existing defect on one of the
  three named driving workloads, recorded as `not_run` rather than worked around. It must be resolved
  before any whole-model claim rests on that model.

## 8c. The golden delta, and the one thing blocking its RTL certification

The emitter (`kernels/opu_kernel.py`) generates the microkernel with every `.insn` word taken from the
derived table, refuses to emit at all if the derivation disagreed with its cross-check source, and has
been verified **from the object at `-O2`** — not from its source, which is the layer the historical
failure hid below. The configure/zero/configure/load/configure/load/accumulate sequence comes out
contiguous, the two operand lengths use different registers, and the row load keeps its tail-undisturbed
policy so zeroed lanes past a short panel survive. The unfused variant — the shape the failure had — is
compiled in the tests and shown to be rejected by each of those checks, so the regression cannot pass
vacuously.

**Two geometry defects were found on our side and both are fixed.** The kernel was single-tile, and the
corpus held its companion extent at a literal 64 — a hardcoded target fact inside the acceptance surface
itself. On `OPUV256D128ShuttleConfig` the logical tile is `(vLen/8)² = 32×32` (physically `16×16` in
`(vLen/dLen)² = 4` subtiles) and the maximum operand lanes at `e8,m1` is `VLEN/8 = 32`, so that literal
put **18 of 31 cases outside a single tile**, including every named narrow-extent regression.

- The kernel now **tiles M and N**, reading the tile edge from the hardware at run time (a `vsetvli` with
  an unbounded request returns the maximum lane count). Tails go through the *same* tile routine with a
  shorter length, so there is no separate tail path that could be right in the full case and wrong at the
  edge — which matters because the tile boundary is exactly where the narrow-extent hazard lives.
- The corpus's companion extent is now an expression over the target's derived tile edge, resolved
  structurally. Swept extents stay absolute, since they name specific awkward values and mean nothing
  rescaled. `select(tile)` splits the corpus into what a single-tile pass can run and what it cannot,
  returning the remainder **with reasons rather than dropping it** — a report that omitted the shapes out
  of reach would read as full coverage. At a 32-edge all 31 cases run; at 16, 25 run and 6 are deferred.
  The narrow regressions run at both.

**The tiling logic is verified numerically without any RTL.** A compile-time switch swaps the tile body
for a scalar stand-in while leaving the tiling loop untouched, so the tail bounds, pointer arithmetic and
bias column offset are checked on a host with no such hardware. There is **one copy** of that loop, so
what the host validates is the code the device build runs rather than a re-implementation. All 31 cases
are **exactly equal** to the reference at tile edges 4, 8, 16 and 32 — every tail alignment, including
edges no available part has. The stand-in is asserted to contain no device instruction, so a device build
cannot quietly compute correct answers without using the unit.

**What remains for certification is the image, not the logic.** Everything needed is present: the
baremetal harness (`merlin/runtime/baremetal/spike/` — `crt.S`, `htif.c`, `link.ld`, `libc_min.c`), the
four prebuilt OPU sims, and `zephyr_model.run_on_verilator(elf, config=…)` which already takes the config
as a parameter. At ~10⁴ cycles/s a 31-case corpus of tile-sized GEMMs is seconds of simulation, which is
why the corpus was kept small. So the open item is a bare-metal image that runs the corpus and compares
in place — not simulator capacity, not a missing oracle, and no longer the kernel's shape.

## 9. Open, and deliberately unresolved here

- ~~**`VOPACC` operand order.**~~ **Resolved from source; `rs1` = LHS.** Three independent readings
  agree and no RTL run was needed to decide it. The RTL computes `md[i][j] += vs1[i] * vs2[j]`, so
  `vs1` indexes rows (M) and is the LHS. `bme.h` declares `#define VOPACC(md, vs2, vs1)` and expands to
  `.insn r 0x57, 0x2, 0x51, md, vs1, vs2` — i.e. the macro's *argument* order is `(md, vs2, vs1)` while
  the *encoding* puts `vs1` in the `rs1` field. Saturn's own `benchmarks/opu-gemm/kernel.h` then calls
  `VOPACC(m1, v4, v5)` having loaded `v5` from `at[k*M]` (the transposed LHS) and `v4` from `b[k*N]`
  (the RHS) — so the third macro argument, and therefore the `rs1` field, carries the LHS.
  Merlin-IREE's `iree_uk_mmt4d_opu_full_loop`, which emits `rs1`=RHS / `rs2`=LHS, is **wrong**; it went
  unnoticed because a square tile makes the swap shape-safe, and square is the case that was validated.
  An asymmetric run on OPU RTL remains worth doing, but now as a confirmation of a settled fact rather
  than as the thing that decides it. The same kernel also confirms two properties the plan requires of
  our microkernel: the M-lane and N-lane loads carry **separate `vl`** (`ml` for `at[k*M]`, `vl` for
  `b[k*N]`), and bias init is `vle32.v` + `OPMVINBCAST`.
- **`derived_levers` for the OPU.** `rtl_backend.derived_levers:48-60` derives `spatial.dataflow` and
  `spatial.accumulator_resident` from discovered RTL, but `spatial_introspect.py:13-17` warns that
  `discover_mesh_dim` **mis-derives DIM=4** from the OPU's cluster×cell hierarchy. What it actually
  returns for the OPU must be checked before anything relies on it.
- **No facet for accumulator-bank count.** The contract reports `mrf_depth`, and `spatial.pe_rows`/
  `pe_cols` are `IDENTITY` (fixed geometry is not a lever). Whether MRF depth is a lever — 4 registers
  allow a 2×2 sub-tiled 32×32 register block — is a Phase 5 question.
- **fp8.** The contract's dtype list is `[int8]` for the real RTL; the fp8 sub-format is surfaced
  honestly as `unnamed_float_datapaths: ["float8"]` because the RTL does not name it, and `OPFMACC` is
  not in `opuInsns` on the `opu-int8` branch. int8 only, in this pass.
