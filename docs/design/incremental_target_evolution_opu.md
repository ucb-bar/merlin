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
- **No saturation.** `OuterProductUnit.scala` carries `// TODO: Need to check for overflow and
  saturate to accumulator width`, so int32 accumulation **wraps**. Any numerical reference must wrap
  too; a saturating reference would disagree with correct hardware.
- **Bias init is one instruction.** `vle32.v v0, (bias)` then `OPMVINBCAST` broadcasts a bias row
  across the tile. Zero-init is the same instruction after `vmv.v.i v0, 0`.
- **Operand layout is forced, not chosen.** Saturn's own `benchmarks/opu-gemm/kernel.h` indexes
  `at[k*M + i]` and `b[k*N + j]`: **both operands are K-major**, so the LHS must be transposed/packed.
  That packing is a real cost term, not an implementation detail — see §4.3.
- **Readout is row-serial** and row 0 is the slowest (`scalar_row_latency = (yDim+1) -
  scalar_cluster_row_idx`), so extraction cost scales with rows, which is what makes an
  accumulator-resident epilogue worth representing.

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

## 9. Open, and deliberately unresolved here

- **`VOPACC` operand order.** Merlin-IREE's `iree_uk_mmt4d_opu_full_loop` emits `rs1`=RHS / `rs2`=LHS,
  while its own comment, `bme.h` and Saturn's `opu-gemm` all use `rs1`=LHS. Shape-safe (hence silent)
  whenever the tile is square, which is exactly the case that was validated. One asymmetric 16×16 RTL
  run settles it; the answer belongs in §1.1.
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
