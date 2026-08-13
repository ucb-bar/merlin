---
title: "Design: incremental target evolution — RVV + Saturn OPU as the driving delta"
kind: design
status: draft
owner: targetgen
last_verified: 2026-08-13
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
  - merlin/python/merlin/kernels/opu_kernel.py
  - merlin/python/merlin/kernels/opu_corpus.py
  - merlin/python/merlin/kernels/opu_cert.py
  - merlin/python/merlin/kernels/cca_matrix.py
  - merlin/python/merlin/kernels/decode/opu.py
  - merlin/python/merlin/kernels/decode/rvv.py
  - merlin/python/merlin/targetgen/artifact_dag.py
  - merlin/python/merlin/targetgen/persistent_equivalence.py
  - merlin/python/merlin/targetgen/contraction_egraph.py
  - merlin/python/merlin/llvmlower/passes_opu.py
  - merlin/python/merlin/llvmlower/opu_shim.py
  - merlin/python/merlin/common/provenance.py
  - merlin/python/merlin/targetgen/rtl/opu_isa.py
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

**Built (`compute_units.py`).** `resolve_exposure(unit, target_endpoint_kind=...)` implements that
precedence, and a test asserts the family default is reached for every kind in `KINDS` — which is what
makes the change inert for existing targets rather than merely intended to be. An undeclared exposure is
rejected at parse time against `families.ENDPOINT_KINDS`.

One subtlety worth recording, because getting it wrong is silent: **composition unions capability, not
exposure.** `effective()` folds a contained unit's dtypes/ops/accumulate rules into its parent, and
inheriting the child's `exposure` along with them would retarget the parent's codegen to whatever it
happened to embed. The field is carried from the composing unit only.

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
select(candidates, cost_model) -> list[RouteResult]          # separable, pluggable, ablatable
route(demands, units) -> list[RouteResult]                   # first-candidate wrapper: unchanged today
```

**Built (`routing.py`).** `route()` is now literally `select(route_candidates(...), first_candidate_cost)`,
so "the refactor changed no behaviour" holds by construction rather than resting on whichever cases the
tests happen to cover. The four tests in `test_routing.py` and both existing consumers
(`frontends/gguf_reader.py:134`, `frontends/adapters/gguf.py:54`) are untouched.

Cost models are swappable specifically so the ablation is real: `eager` (always prefer the widest
datapath) is implemented as a deliberately bad baseline, and `MeasuredCost` reads a caller-supplied
measurement table rather than inventing architectural constants.

Three properties of `MeasuredCost` are load-bearing, and the first was a defect a test caught:

- **Tile occupancy, not MACs.** Costing a tiled unit as `macs / peak_rate` credits it with the work it
  *would* do at full occupancy, so a partly-filled tile looks as cheap as a full one and every narrow
  shape looks good on the unit — including precisely the ones §8b found to be numerous and negligible. A
  tile is the unit of work, so `M = 1` and `M = 32` cost the same. That is the whole reason a narrow
  extent is expensive, and the earlier formulation could not express it.
- **Declining is a third outcome.** An unmeasured unit that scored well would win on the strength of
  having no data; one that scored badly would be ruled out for the same reason. So `None` means "no
  data", and a demand whose only legal unit is unmeasured still routes and reports that it did so
  unscored — dropping it would turn a missing measurement into a routing gap, i.e. into a capability the
  target does not have.
- **Layout is charged.** A unit that needs K-major operands pays a packing pass over `k*m`; leaving that
  out is how a decision comes out in favour of a unit that then spends longer rearranging memory than
  computing.

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

**Built (`targetgen/artifact_dag.py`, 21 nodes).** The value of writing it down is that the reuse claim
becomes *falsifiable*: "the delta invalidates nothing in the parent's schedule" is the claim that **no
path exists** from the changed node to that one, and `test_artifact_dag.py` looks for the path. The four
parent nodes are declared roots; if a future edge reaches one, that test fails rather than the paragraph
above quietly becoming untrue. The mirror test — that the delta *does* reach routing, lowering, codegen,
CCA and certification — is there because a graph where it reached nothing would satisfy every
"must not reach" test perfectly and mean nothing.

Three things it refuses to fudge. The changed set is diffed from **content hashes**, not declared, since a
hand-written changed set makes the number whatever its author wanted. A node whose sources cannot be read
is `UNKNOWN` and counts as changed on both sides — two `UNKNOWN`s comparing equal would hand back free
reuse for exactly the nodes nothing is known about. And the **denominator is a required argument**
recorded in the result, because a ratio whose relevant set is chosen quietly can be made to say anything.

**The measured ratio is 5/21 (0.238).** That is low, and the reason is worth stating plainly: every one of
the five synthesized plans reads *all* of evidence, so touching hardware facts invalidates all of them.
That is a fact about the **pipeline's granularity**, not about how invasive the delta is — a finer-grained
evidence node set (per-concept rather than one node) would measure higher reuse for the same delta. The
test asserts a *range* rather than a value, with a note that raising it by editing the graph would defeat
the purpose. The honest reading: this pass measures reuse at the resolution the pipeline currently
supports, and improving that resolution is a separate piece of work from the delta itself.

### 4.3 Where the OPU lowering attaches — BUILT

The plan above was followed, with one correction: the feature edits neither the pipeline nor the transform
schedule, because the rewrite happens on the prepared IR. So `opu_matmul` is registered with **both hooks
`None`**, which makes the byte-identity invariant structural rather than merely tested — there is no edit
to apply.

The four pieces, and what each is responsible for:

| piece | responsibility |
|---|---|
| `llvmlower/passes_opu.py` | the structural xDSL rewrite: contraction → `call @merlin_opu_gemm_i8_<n>`, one symbol per distinct signature, plus the sidecar the build reads |
| `llvmlower/opu_shim.py` | the generated C translation unit and its compilation: the certified microkernel, the K-major pack, the descriptor ABI, provenance |
| `contract/matrix_units.yaml` | where the derivation STARTS — the declaration names that are addresses of facts rather than facts |
| `zephyr_model.MatrixRouting` | which unit and configuration a build routes to, so the tile edge has one source |

Three things are worth recording because they were not obvious before building it:

- **The init must be a zero fill, and this is correctness rather than convenience.** `linalg.matmul` computes
  `C_init + A@B`; the microkernel *writes* its output. Those agree exactly when `C_init` is zero. All 90 of
  spectformer's candidates are a `linalg.fill` of `0 : i32`, so requiring it costs nothing — but a model
  whose contractions accumulate onto a live init is now **declined** instead of silently dropping the addend.
- **xDSL drops `arg_attrs` when printing a bodyless declaration.** It stores them correctly and prints them
  for a function *with* a body. So `bufferization.access` never reached the text mlir-opt parses, and
  one-shot-bufferize would have copied the weight operand of every routed contraction. The rewrite repairs
  the printed text and **refuses to write a module** whose declarations lost them.
- **The routing runs before the per-op register blocking**, because a contraction that has become a call is
  no longer on the vector path and the block table must be derived from the IR that remains.

The two halves are joined by a **sidecar** (`opu_signatures.json`) rather than a return value, because the
lowering runs in a subprocess that re-imports these modules — the same reason `impr_features` needs
`_try_lazy_register`. The build generates its C side from the sidecar, so the symbols defined are exactly
the ones the module calls; a set reconstructed independently could drift into a link error.

The microkernel is compiled alongside the model object the way `runtime/c/merlin_op_prof.c` already is.
Its epilogue shape is the interesting compiler question, and it is the one the XNNPACK template answers
in the affirmative: extract a row with `VMV_VR`, convert, scale, clamp, narrow and store **without
round-tripping int32 through memory** — i.e. accumulator-resident requantization. That maps onto
`cca.ComputeFacet.accumulator_resident`, which already exists as the shared target-agnostic concept
with a PASS → CODEGEN escalation ladder.

### 4.4 The CCA lifter for this datapath (`kernels/cca_matrix.py`)

Residency is the same cross-backend question, with different evidence. RVV loses it by spilling the
accumulator inside the MAC loop; a unit whose accumulator is *architected state* has no register to spill,
so the way to lose it is to **read the accumulator out inside the reduction** instead of once after it. The
lifter reads that from the emitted stream and sets `accumulator_resident` on the *compute* facet as well as
`spatial` — under `spatial` alone it would never diverge against a vector expert.

Two methodological points, both of which cost a wrong first attempt:

- **Residency must be scoped to the reduction loop, not to a span between accumulates.** A looping
  reduction emits exactly *one* accumulate statically, so "is there a readout between the first and last
  accumulate" is vacuous for every kernel that actually loops — it can only judge an unrolled one, and
  would report residency for a per-step commit. The loop comes from resolved back-edges instead, and
  when the displacements are unrelocated (an unlinked `.o`) the answer is UNKNOWN rather than a confident
  wrong one.
- **Identity comes from the derived table, never a mnemonic.** These instructions occupy reserved slots
  and a disassembler prints them as unnamed words, so a mnemonic-matching lifter sees an empty stream and
  pronounces every kernel clean. A test shifts every funct6 and requires the same stream to decode as
  containing none of the unit's ops.

The check is only worth having if it can fail, so a variant that reads out inside the loop is compiled in
the tests and required to be caught. `tile_occupancy` exposes what a MAC count cannot: at `M = 1` on a
32-edge tile the kernel is correct, busy, and using one row in thirty-two. Routes register through the
existing `action_catalog` plugin seam, and the CODEGEN route names the regimes where filling the tile is
plausible — so the narrow shapes of §8b stay out of the catalog instead of relying on a later filter.

**A shared-code bug fell out of this.** `decode/rvv.py` matched branch mnemonics against a list holding
only uncompressed forms, so `c.bnez`/`c.beqz`/`c.j` were not branches — and `rv64gcv` includes the C
extension, so at `-O2` a tight loop's back-edge is routinely the compressed form. With the inner back-edge
invisible, `loop_spans()` omitted the reduction loop, `_fma_loop` fell back to the enclosing loop, and a
spill sitting legitimately *around* a reduction was attributed to it: **measured on the added fixture,
`accumulator_resident` reads `False` before the fix and `True` after, for the same resident kernel.** Every
other loop-scoped count (`calls_in_loop`, register block) was mis-scoped the same way, and
`spans_reliable()` shared the defect, so it could vouch for a stream whose real back-edges it had not
looked at. This means RVV residency figures recorded before this fix are suspect wherever the reduction's
back-edge was compressed.

## 5. Verification ladder, and what this pass does *not* verify

| rung | what it proves | oracle |
| --- | --- | --- |
| L0 | contract legality | capability contract |
| L1 | compiles | clang-23 |
| L2 | emitted-code audit | `kernels/decode/opu.py` (structural, derived opcodes) |
| L3 | microkernel numerics, frozen shape corpus | **OPU Verilator RTL — PASSED 31/31** (§8c.2), + 1 of 5 workload shapes (§5.2) |
| L4 | single-dispatch numerics incl. epilogue | **OPU Verilator RTL — PASSED 6/6** (§5.2) |
| L5 | whole-model numerics | **blocked — not by the unit** (§5.3) |
| L6 | whole-model cycles | needs the bitstream of §5.1; L5 first |

### 5.2 L4 passed, and it moved the corpus's centre of gravity

Every case up to L3 was judged on the **int32 accumulator**, which is where the readout stops. A quantised
model does not stop there — it scales, rounds, clamps and narrows to int8 — so a kernel whose accumulator is
right and whose epilogue rounds the other way yields a model that is subtly wrong *everywhere* rather than
obviously wrong somewhere. L3 therefore said less than it appeared to.

Six cases now carry the epilogue and are judged on the **narrowed** output: a square tile, a single output
row, a single output column (a per-column multiplier indexed wrongly is invisible with more than one column
to average over), ragged tails in both directions, full-range operands over a long reduction so the clamp
actually fires, and bias composed with the epilogue. Measured saturation across them spans 0% to 99.6%.

**Result: 6/6 bit-exact on `OPUV128D64ShuttleConfig` RTL**, `uses_unit=True`, no gaps, provenance recorded
against `saturn_opu_int8 @ ea373800`. Landed at
`out/artifacts/measurements/baremetal_OPUV128D64ShuttleConfig/opu_microkernel/l4_requant_dispatch_v1_*`.

The rounding is the one place a C implementation and a numpy one drift silently, so it is arranged rather
than hoped: both add half an LSB and take an **arithmetic** right shift. An arithmetic shift of a negative
value rounds toward negative infinity, so `(x + half) >> s` and `round(x / 2**s)` disagree on exact halves of
negatives — a handful of elements per tensor, which is precisely the error size a cosine gate absorbs.

#### The measured cost model, and a defect it exposed in the assumed one

The same run yields cycles per case, which is what `routing.MeasuredCost` declines a unit for lacking.
Throughput spans **14.3×** on shape alone (0.66 MACs/cycle at 8×1, 9.49 at 8×8 with K=255). Charging tile
occupancy collapses that to **2.3×**, empirically validating the cost model's central premise.

It also falsifies part of it. A tile-occupancy-only model — a full tile charged in *both* dimensions — fits
with 49.5% worst error and systematically **over**-charges a narrow M. The kernel's readout is **row-serial**,
so a one-row output performs one readout, not a tile's worth. Charging rows separately:

    cycles = 90.2 + 5.78 × accumulates + 19.83 × row_readouts        worst error 14.1%

The readout costs about **3.4× an accumulate**, which no MAC-count model can see, and it means a short-K,
tall-M shape is readout-bound rather than compute-bound. Caveat recorded with the artifact: 6 measurements,
3 parameters — the ranking is robust, the coefficients want the full 42-case corpus behind them.

#### Substrate is chosen by output size, not by which shape matters most

Extending the corpus to spectformer's real shapes (Phase C) exposed a property of the certification image
that decides where a case can run at all. Per case the image runs a **scalar triple-loop reference**, zeroes
two buffers and hashes the result with byte-wise `fnv1a64`. All three scale with the **output**, none with the
unit. For `workload_ffn_up` (196×1024, 200,704 outputs) that is ~6.4M scalar MAC iterations and 800 KB hashed
against the OPU kernel's ~204k cycles: **the measurement subject is under 4% of the runtime.**

The scale is now **measured, not inferred**. `workload_classifier` (1×1000, K=32) certifies in **501 s** of
Verilator wall time for its 32,000 reference-loop MAC iterations — **63.9 iterations/s**. Against that
calibration the remaining four workload cases (14.45M iterations) need **~63 h ≈ 2.6 days**, and
`workload_ffn_up` alone needs **~28 h**. The same set on **FireSim at 25 MHz is ~4 seconds**. `screen_only`
(drop the in-image reference) removes the dominant term but not the problem, because the hashing and zeroing
are output-proportional too.

That calibration also corrects an earlier estimate of ~12 days, and the way it was wrong is worth keeping.
The first estimate scaled 13 h of observed runtime by the ratio of remaining-to-done reference MACs, assuming
the 13 h had bought the 31 small cases. It had not: the small cases account for only **~2.9 h**, so ~10 h had
already gone into case 31 — the run was **36% through `workload_ffn_up`**, not finished with the cheap work.
Taking "done" to mean "last thing printed" understated the denominator by 4.5×. The decision it drove was
unchanged (2.6 days against 4 seconds), but a ratio is only assumption-free when *both* of its terms are
measured.

So the split is arithmetic rather than preference: `workload_classifier` is the only workload-scale case
Verilator can certify, and the other four require the bitstream. A corollary for triaging any such run: the
driver captures the image's stdout, so **an empty log is not a hang** — dividing `/proc/<pid>/io`'s `wchar` by
the image's exact per-case `CASE`/`CYCLES` line length locates the running case index precisely, which is what
identified case 31 of 36.

**The result itself, and the cost model's first out-of-sample test.** `workload_classifier` is bit-exact
against *both* the in-image scalar reference and the host digest (`mismatches=0`, `uses_unit=True`, 3×OPMACC /
2×OPMVINBCAST / 1×OPMVOUT), at **7,683 cycles and 4.17 MACs/cycle** on `OPUV256D128ShuttleConfig`, landed at
`out/artifacts/measurements/baremetal_OPUV256D128ShuttleConfig/opu_microkernel/workload_shape_classifier_v1_*`.
The cost model of §5.2 was fitted on six small requant cases; applied to this shape it predicts
90.2 + 5.78×1024 + 19.83×32 = **6,643 cycles** against 7,683 measured — **13.5% under**, in line with its
stated 14.1% worst error and on a shape 30× larger than anything in its fitting set. Note it is the row-serial
readout term that makes this work: M=1 charges 32 readouts (one per column tile), not a tile's worth, which is
exactly the correction the tile-occupancy model lacked.

### 5.3 What blocks L5, and it is not the unit

The whole-model rung is blocked by a defect in the shared bare-metal path that has nothing to do with the
matrix unit: the unrouted control fails identically, and so does a second model (deepjscc). `memrefCopy` is
handed a descriptor whose middle words hold **another copy's descriptor bytes** — proven by address
arithmetic, the rank-4 destination at `0x80085b60` spanning 88 bytes while a rank-2 descriptor sits at
`0x80085ba0`, which is word 8 of it. The stray pointer is then read as a stride and the store lands
gigabytes outside any mapping.

Seven hypotheses have been eliminated (rank mismatch, stride magnitude, wrapper-vs-descriptor rank, wrong
descriptor values, mis-scoped allocas within one copy, two allocator rewrites, and self-copies). A separate
finding came out of it and stands on its own: **12 of 23 copy sites are `memref.copy %x, %x`**, accounting
for 602,112 of 602,113 runtime copies, and the existing default-OFF `erase_self_copy` removes them with a
documented bit-exact 1.88× speedup. It does not fix this fault.

RTL sims for `OPUV256D128ShuttleConfig`, `OPUV128D64ShuttleConfig`, `OPUMXV256D128ShuttleConfig` and
`GemminiAndOPUShuttleConfig` are already built under `$MERLIN_CHIPYARD/sims/verilator`, and
`zephyr_model.run_on_verilator(elf, config=…)` (`:1533`) already takes the config as a parameter, so
RTL execution is a parameter away rather than new code. Caveat: in that checkout the OPU config source
is a `.bak` file, so the binaries exist but are not currently rebuildable there.

### 5.1 No bitstream needs building — one already carries the unit

The plan for L6 assumed a new FireSim build for a wider OPU. It does not need one. **Two OPU bitstreams
are already built and registered** in `config_hwdb.yaml`, and both tarballs are present on disk:

| hwdb entry | config | geometry | built |
| --- | --- | --- | --- |
| `alveo_u250_firesim_shuttle_gemmini_opu` | `FireSimGemminiAndOPUShuttleConfig` → `GemminiAndOPUShuttleConfig` | vLen 128 / dLen 64 → **edge 16, align 8** | 2026-05-06, 57 MB |
| `alveo_u250_firesim-opu-v128-d64-shuttle` | `FireSimOPUV128D64ShuttleConfig` | vLen 128 / dLen 64 → edge 16, align 8 | 2026-03-20, 112 MB |

Targeting the first is now a config string, because every geometry fact is derived from that config's own
declaration. Two things had to be fixed for that to be true, and both were real gaps rather than plumbing:

- **A unit's configurations live in two repos.** The extension's own generator declares the standalone
  configs (one core, the unit, nothing else); the *integrating SoC* declares the heterogeneous ones that put
  the unit on one tile beside something else — and heterogeneous is what real bitstreams are built from. The
  contract now names both sites (`config_scala` / `host_config_scala`), so the compiler can target hardware
  that already exists instead of only hardware described by the generator.
- **Named Scala arguments were invisible.** `vector_unit_params` bound only positional arguments, so
  `WithShuttleVectorUnit(vLen = 128, dLen = 64, params = …, cores = Some(Seq(1)))` — exactly how the
  integrating SoC writes it — returned *nothing at all*, and the config a shipped bitstream was elaborated
  from looked ungroundable. Named arguments now bind by their own name and do not consume a positional slot;
  the second half matters because counting one would shift every later argument onto the wrong name and
  yield a plausible, wrong geometry.

The consequence for the ladder: **L6 is gated on execution, not on synthesis.** The frozen corpus already
covers edge 16 (the `tile`-relative cases resolve against it), so Phase C needs no new hardware either. The
`FireSimOPUV128D64ShuttleConfig` entry is a reproducibility caveat rather than a blocker — that class no
longer exists in the checkout's source, so its bitstream can be *used* but not *rebuilt*; prefer the
`GemminiAndOPUShuttleConfig` one, whose config is still in `TargetConfigs.scala`.

A **Kodiak-exact** bitstream is a separate question and still deferred (§9), and a Kodiak FireSim bitstream
existing does not change that. A built one was handed over on 2026-08-13
(`FireSimCTCKodiakConfig`, bit dated 2025-05-22, sha256 `38219a1066b2c887…`) and it carries **no OPU** —
verified by following its own chain rather than by the name: the bitstream's `metadata` names
`FireSimCTCKodiakConfig`, which composes `chipyard.KodiakFireSimCTCConfig`, which instantiates
`saturn.shuttle.WithShuttleVectorUnit(512, 256, VectorParams.genParams)`; that tree pins
`generators/saturn` at `a898bdc`, and at that revision **`useOpu` does not appear in `common/Parameters.scala`
at all and `opuParams` is not defined**. So the unit is not disabled there, it does not exist — the OPU lives
only on the `opu-int8` fork branch (see the `saturn_opu_int8` pin), 128 commits ahead of the tapeout base.
Note the distinction that made this easy to get wrong: `boards.board("chipyard_kodiak")` in `runtime/boards.py`
is a *software board descriptor* for the physical Kodiak, a different object from any chipyard RTL config, and
a claim about one does not transfer to the other.

That bitstream is still useful, just not for this: it is a vLen=512/dLen=256 Saturn vector unit, which matches
the VLEN 512 the Kodiak board descriptor declares, so it offers a FireSim route for the **RVV** whole-model
work that today runs on the physical board.

Two further facts came out of reading the forks directly, and the first explains why the "Kodiak has an OPU"
claim keeps resurfacing. There is a branch called **`kodiak-multi-chip-firesim-opu-kernels`**, one commit
ahead of the branch that built the bitstream, whose commit message is *"add opu stuff"*. It changes three
lines: it bumps `sims/firesim` (adding an `opu-m2-gemm` **workload**), bumps `firemarshal`, and re-points
saturn's submodule **URL** at the OPU author's fork while leaving the **sha unchanged at `a898bdc`**. It adds
no RTL, and all three Kodiak configs on it still pass `VectorParams.genParams`. The intent was plainly there;
the pin was never moved. An OPU workload existing is not an OPU existing.

Second, the build is **not reproducible from git alone**: the bitstream's `metadata` records
`firesim-commit 727a86ff…-dirty` and a build root under another user's home on a **different machine**, and
the build recipe was never committed — so the `fpga_frequency` it closed at is not recoverable. Everything
else about the target is.

Since the OPU therefore had to be *added* rather than obtained, it was added in this checkout, which already
carries the OPU-bearing saturn and the clock-gate fix below: `chipyard.WithKodiakBase` / `KodiakConfig` /
`KodiakOPUConfig`, plus `FireSimKodiakConfig` / `FireSimKodiakOPUConfig`. The OPU is enabled with
`genParams.copy(useOpu = true)` rather than by switching to `opuParams`, because `opuParams` also changes
`vliqEntries`, `vlissqEntries` and `useElementwiseFP64` — that would produce a different vector unit that
happens to have an OPU, whereas one boolean apart makes the A/B against `KodiakConfig` mean something.

Two caveats belong with any result from it. The **chip-to-chip link is absent**: `KodiakFireSimConfig`'s
`WithSerialTL` + `WithOffchipBusClient` combination does not elaborate (diplomacy rejects the overlap of the
replicated off-chip window and the serial-TL manager window at `0x200000000`), and the config that was
actually built is the CTC one, which has that whole block commented out and which needs a `testchipip.ctc`
package this checkout does not have. Cores, vector units, TCM, L1s, L2 and scratchpad are untouched, so this
is Kodiak's **compute** SoC on one chip, not a reproduction of the multi-chip target. And **area is the open
risk**: the array scales with `dLen` (`yDim = (dLen/aWidth)/clusterYdim`), so each core's array is 8×8
clusters against v256d128's 4×4, and there are two Shuttle tiles — verified in the generated Verilog, where
`TilePRCIDomain` carrying the OPU is instantiated twice and each `OuterProductUnit` contains 64 clusters.
That is **8× the OPU** that measured 81,335 LUTs and 131,584 FFs inside a design at 20.6% LUT utilisation,
which projects to roughly 70% of the device. Whether it closes is a measurement not yet taken.

#### The third bitstream does not close, and the cause is the OPU's own clock gate

`FireSimOPUV256D128ShuttleConfig` (recipe `alveo_u250_firesim_opu_v256d128`, 25 MHz) was synthesised to match
the **edge 32** geometry the corpus is certified against, so that Phase C's shapes and the L3/L4 evidence
would share one geometry. It was **stopped after 15 h** without producing a `.bit`. An earlier reading of this
section treated Vivado's "Timing constraints are not met" as a normal intermediate state of the IDR flow, on
the strength of the shipped v128d64 build having recovered from a similar-looking report. That reading was
wrong, and the convergence *rate* is what shows it:

| | v128d64 (shipped, working) | v256d128 (stopped) |
| --- | --- | --- |
| entering hold fix | WHS **−2.539**, THS **−129,140** | WHS **−4.068**, THS **−840,230** |
| after `phys_opt_design -aggressive_hold_fix` | WHS **−0.366**, THS −375.8 (**−99.7 %** THS) | WHS **−3.972**, THS −834,171 (**−0.7 %** THS) |
| final | WNS +0.031 / WHS **+0.009** / THS 0.000 → `.bit` | never converged; IDR RQS then ran 5.5 h on one thread over 231,911 candidate nets, wrote zero bytes, produced no checkpoint |

Both designs are fully routed with zero routing errors, and setup is met in both. The difference is that one
hold fix removed almost all of the violation and the other removed essentially none.

**The violation is 99.3 % one clock, and it is intra-clock.** Of 260,324 hold-failing endpoints, 258,620 sit
in the group `uart_clock,clock_1000.0MHz,harnessbinder_clock,reference` — 58.8 % of that clock's 439,611
endpoints — while the same clock carries **WNS +16.520 ns** of unused setup slack. Reading the worst path with
`report_timing -hold -path_type full_clock_expanded` gives the mechanism directly:

| | launch (`vu/vopu_ctrl_reg_in_t_*`, ungated) | capture (`vopu/clusters_*`, gated) |
| --- | --- | --- |
| clock path | `BUFGCE_X0Y72` → net (fo=71,743) | `BUFGCE_X0Y72` → net (fo=71,743) → **LUT** → **`BUFGCE_X0Y226`** |
| clock delay | 5.077 ns | **8.868 ns** |

Clock Path Skew **4.207 ns**, slack **−4.068 ns**. Saturn gates the whole OPU cluster array
(`exu/OuterProductUnit.scala:181`, `ClockGate(clock, io.op.clock_enable, "opu_clock_gate")` +
`withClock(gated_clock)`), which lowers to rocket-chip's `EICG_wrapper` — a latch-and-AND in LUTs. Golden
Gate's `midas.passes.xilinx.ReplaceAbstractClockGates` rewrites only its *own* `AbstractClockGate` instances
to `BUFGCE`, so the target's `EICG_wrapper` survives, and Vivado must add a **second global clock net** behind
combinational logic to reach the gated domain's 131,585 loads. Both ends carry the same clock name, so hold is
checked at zero skew against a tree that is 3.79 ns later. `v128d64` closed because its array is 4× smaller
and the gated net's delay nearly matched its parent's. All 300 worst paths are SLR2→SLR2, so this is **not**
an SLR crossing.

Three things that are **not** the fix, each ruled out with a number rather than an argument. **Frequency**:
the recipe is already `fpga_frequency: 25`, and the failing clock has 16.5 ns of setup slack — hold and skew
are both period-independent, so a slower clock adds surplus where there is already surplus. **`CLOCK_DELAY_GROUP`**:
it matches insertion delay across buffers driving one clock, but cannot match a one-buffer tree to a
two-buffer cascade, because the depths genuinely differ. **More physopt passes**: closing this by delay
insertion needs ~3.8 ns on 258,620 paths, tens of LUT1 hops each, which exceeds the device.

The candidate fix is therefore to stop gating the OPU clock on FPGA, by pointing `ClockGateModelFile` at a
passthrough (`merlin/contract/rtl/eicg_passthrough.v`) — a config-level swap that touches no target RTL.
**Whether that is sound is a measurement, not a deduction**, because the gated domain is not uniformly
enable-guarded: `OuterProductCell.regs` is written under an explicit `when`, but `OuterProductCluster.pipe`
— the row-serial readout register — is assigned unconditionally, so it *does* rely on the clock stopping to
hold. The sequencer raises the enable as `clock_enable := valid || mvout_valids =/= 0.U`, i.e. for every cycle
an instruction is in flight or a readout is pending, and `pipe` is re-primed from `cell_outs` at the start of
each readout rather than carried across readouts, so the clobbering ungating introduces should land only on
idle cycles nobody reads. That is validated by re-running the certified corpus against
`OPUV256D128ShuttleNoGateConfig` and requiring bit-identical digests, with the cases chosen to stress `pipe`
(ragged tails, single-row and single-column readouts, the widest tile sweeps, long reductions, bias broadcast
composed with the epilogue, and the multi-tile-column `workload_classifier`). Until those digests are in, the
fix is a hypothesis. Note also that the geometry deriver **refused** an inherited vector length for the new
config rather than defaulting one, and the provenance pin correctly reports drift once `chipyard/OPUConfigs.scala`
carries the added config — so any number from that build records as *pinned revision plus one intended source
change*, not as a clean pinned run.

One point that stays true regardless: **the FPGA is not needed to build a bitstream** — synthesis and
implementation are Vivado work; the board is needed only to *run* one. So L6 remains gated on execution
(§5.1), never on synthesis. Synthesis and placement survived the stop
(`impl_1/overall_fpga_top_{opt,placed,physopt,routed,postroute_physopt}.dcp`), so a retry resumes from
placement rather than from scratch.

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

### 7.1 Built, and what it does *not* show

`targetgen/persistent_equivalence.py` builds one contraction's e-graph with every legal implementation as
an alternative in an `equivalence.class`, attaches the `MeasuredCost` value to each as an `eqsat_cost`
attribute, and runs the real `eqsat-add-costs` + `eqsat-extract` passes. The chosen unit is **read back
out of the extracted IR**, not computed alongside it — computed alongside, this would be `routing.select`
with extra steps and would demonstrate nothing about deciding from the graph.

**H-EQ1 is `not_established`, and the code says so.** With one e-class and no rewrite rules, extraction is
an argmin over the same costs eager selection reads, so it *cannot* decide differently; reporting a win
here would be arithmetic dressed as a result. `agreement()` therefore reports agreement (identical
decisions are the expected outcome) rather than a comparison of quality.

What *is* demonstrated is the mechanism, and it is the precondition for H-EQ1 rather than evidence for it:

- both implementations are present in the IR after construction (`alternatives_in`), and
- `recost()` on an **already-built** graph changes which alternative is extracted.

The second is the falsifiable form of "persistent". A graph that had quietly committed to a unit could not
be moved by a later cost, and `recost` exists precisely because re-costing after the fact is what a
downstream pass (epilogue fusion, layout discovery) would do.

**H-EQ2 is `not_exercised`.** Saturation needs rewrite rules; none are applied, so there is nothing to
re-saturate and no incremental-vs-scratch comparison to make. The paper is silent on incremental
re-saturation, so it cannot be inherited as a result either. Both statuses are held as **data**
(`HYPOTHESES`) and asserted unproven in `test_persistent_equivalence.py`, so promoting either requires
editing a test — a better venue for that argument than a docstring.

A candidate the cost model declines is retained as an alternative but left uncosted: dropping it would
erase a capability the target has, and costing it would let it win for having no data. When every
candidate is declined, extraction fails closed and the caller falls back to declaration order — the same
fallback `select` uses, so the two stay comparable.

## 7.2 The e-graph now decides over real IR, and a rule grows it

§7.1 demonstrated the mechanism on synthetic ops: one `test.TestOp` per candidate carrying a unit name and a
cost. That shows extraction reads a decision out of a graph; it cannot show the decision is one a compiler
would act on, because nothing in that graph is an implementation and nothing can be emitted from the choice.

`targetgen/contraction_egraph.py` builds the same e-class over the **actual** IR: the contraction's own
`linalg.generic`, region and all, beside the `func.call` the rewrite emits. Both yield the same
`tensor<MxNxi32>`, which is why the e-class is type-correct and why the extracted function *is* the compile
path's answer — whichever alternative survives is the implementation that runs. Verified on all 90 of
spectformer's routable contractions: each builds a verifying graph, and a decision takes ~0.4 ms.

The winner is **read back** from the extracted IR rather than computed alongside it. Flipping only the costs
flips which implementation survives, and a tie resolves to the vector path, because declaration order breaks
it and the vector path is the control — a coin-flip must not move work onto a unit whose advantage is
unproven. `egraph_selector` is exactly the `select` callable `passes_opu.rewrite_contractions_to_opu` takes,
so the routing decision the compile path applies is the extraction's rather than a threshold that agrees.

**Saturation runs.** Under `apply-eqsat-pdl` a `pdl.replace` *adds* to the matched value's e-class rather than
replacing, so the rule "a rank-2 int8 contraction is also computable by the microkernel" **grows** the graph:
measured on the real prepared model, one alternative becomes two with the second created by the compiled
rewriter. That changes what a new capability costs — a rule, not a code change. `H-EQ2` moved from
`not_exercised` to `not_established` accordingly: saturation running is not the incremental claim holding,
because nothing re-saturates a parent graph against a delta.

**The rule cannot carry legality, and the generated pattern says so.** PDL matches by op name and by operand
and result types; it cannot express `(i8, i8, i32)`, or the iterator types, or "the accumulator init is a zero
fill". A pattern on `linalg.generic` therefore matches *every* generic, so legality stays in
`passes_opu.routable_contractions` and saturation runs only on a region already known legal. A test asserts
the warning survives in the emitted text, because a rule that stops looking dangerous is the failure mode.

**What still gates a good decision is a measurement, not machinery.** Extraction minimises exactly what it is
given: a crude cost sent 89 of 90 contractions to the matrix unit *including* the 48 FFT-family shapes whose
N is 8 or 14, because a generous rate swamps the occupancy penalty on a shape filling a thirtieth of a tile.
`routing.MeasuredCost` declines a unit absent from its throughput table, so before §5.2 nothing routed at
all — correctly. §5.2 supplies the first measured numbers, and also corrects the occupancy model they feed.

## 7.3 Provenance covers built artifacts, not only checkouts

A pin answers "which checkout was this read from". A bitstream, a compiled simulator and a packaged image have
no answer: they are outputs with no commit of their own, so nothing in the registry could describe the thing a
hardware verdict came from. `hardware_pins.yaml` now carries an `artifacts:` section keyed by content digest,
with `built_from` naming pins rather than repeating their shas and `config` recording which elaborated
configuration the bytes are.

`verify_artifact` distinguishes three states and treats the third as a **gap**: absent; present with a digest
that disagrees; and present with nothing declared to compare against. That last is the point — an artifact
verifying against nothing certifies itself. So the bitstream of §5.1 is registered while its build runs,
honestly, without becoming self-attesting, and `check_provenance --verify-pins` checks artifacts alongside
pins. The 16-lane bitstream that already existed is digested and verifies; its `built_from` is deliberately
**empty**, because nothing records which saturn revision produced it and inventing one would be worse.

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

### 8c.1 The certification image (`kernels/opu_cert.py`)

**Three independent implementations, and the image carries no answer key.** A case is certified only when
the device kernel agrees with an in-image scalar reference *and* a hash of its output matches one computed
host-side by numpy. The in-image comparison is what localises a failure (it knows which element); the
host-side digest is what makes the verdict trustworthy, because two agreeing implementations inside one
image still look like a pass when both are wrong. **Nothing expected is embedded** — an image holding its
own goldens can pass by copying them, and the console cannot tell the difference. A test asserts the only
`int32` arrays in the generated image are declared biases.

Operands *are* embedded, because the corpus draws them from numpy's generator and reproducing PCG64 in C
is not on. That is the stronger choice anyway: the bytes the image computes on are provably the ones the
host derived its expected digests from, rather than two generators that are supposed to agree.

**The geometry is derived, then cross-checked against the running hardware.** `tile_edge_for_config` reads
the vector length from the config's own Scala declaration, binding positional arguments to the names in
the mixin's parameter list (`opu_isa.vector_unit_params`) rather than assuming an order — `(vLen, dLen)`
reversed is a tile edge wrong by exactly 2×, which would silently certify a corpus against the wrong
geometry instead of failing. Verified on all four `OPUV*` configs; a config whose vector length cannot be
grounded raises. The image then reports the edge the hardware actually gives it, and a disagreement with
the edge the corpus was selected for is a **hard failure**, not a warning.

**Anti-cheat.** Between build and run, the image is audited for the unit's derived opcodes: a device image
containing none of them would compute correct answers on the host core and pass every numerical check,
which is the most comfortable way for this whole exercise to be wrong.

**The scalar pre-flight passes 31/31 on spike.** Built with `OPU_SCALAR_TILE`, the same image runs the same
corpus with the scalar stand-in, on a real RISC-V target, in seconds — so operand embedding, K-major
layout, the tiling loop, the comparison and the digests are all confirmed to agree with numpy *before* any
RTL is involved, and a failure on RTL is attributable to the datapath rather than to the harness. That
build must contain none of the unit's instructions, which is checked, so it cannot be mistaken for a device
run.

The tests feed the verdict consoles that are wrong in specific ways — a shape the case does not name, a
digest numpy disagrees with, a missing case, a geometry the corpus was not selected for, an unparseable
line — and require it to say so. A harness that only reports correctly for a working kernel is worthless,
because every interesting failure here is one that looks like a pass.

### 8c.2 The L3 result: 31/31, and the sequencer bug the 31st case found

**`OPUV256D128ShuttleConfig`, 31/31 certified on the unit's RTL.** `uses_unit: true`, no gaps, derived
tile edge 32 confirmed against the hardware's own report. Every case agrees with an in-image scalar
reference AND with a numpy-computed digest; no expected value is embedded in the image.

Getting there found **four** defects, three ours and one in the unit's shipped RTL. All four are silent
in different ways, which is the point of the corpus:

| defect | how it failed | where |
| --- | --- | --- |
| accumulate/readout under `e32`/`m1` | **hung the core** — no trap, no retire | ours |
| accumulator init at the operand vtype | wrote a quarter-row | ours |
| compressed branches not treated as branches | mis-scoped every loop query incl. RVV residency | ours, shared code |
| LHS read-hazard released one iteration early | **wrong arithmetic**, order-dependent | **the unit's RTL** |

#### The RTL bug

Read from the sources the sim binary was built from (saturn `ea37380` on `opu-int8`; the generated
`OuterProductSequencer.sv` in the sim's own build tree carries `@[…OuterProductSequencer.scala:…]`
annotations for these lines, so this is the elaborated design and not a divergent working tree):

```scala
when (io.iss.fire && !tail) {
  when (!macc || row_idx_tail) { rvs2_mask := rvs2_mask & ~UIntToOH(io.rvs2.bits.eg) }  // RHS: gated
  rvs1_mask := rvs1_mask & ~UIntToOH(io.rvs1.bits.eg)                                   // LHS: every iter
```

One `opmacc` iterates `(vLen/dLen)² = 4` subtiles. The LHS element group is `base + row_idx` and the RHS
is `base + col_idx`, so the **two column iterations of a given row read the same LHS group** — and the
first of them clears its intent bit. `rvs1_mask` feeds `seq_hazard.rintent`, which is exactly what a
younger instruction's `war_hazard = vd_write_oh & io.older_reads` tests against, so for the remaining
iteration the hardware no longer advertises the read and a write to that register is not blocked.

**Prediction and observation coincide exactly.** With `M = 8` every output row lives in `row_idx = 0`, so
the single vulnerable iteration is `(row = 0, col = 1)` — columns 16…31. That is precisely the region that
came back wrong, short by exactly one reduction step's outer-product term, with columns 0…15 always
bit-exact. `kernels/opu_forensics.py` decomposes the deltas and names the step.

**Confirmed by A/B**, same 24-case reproducer, identical operands, alignment and arithmetic, differing
only in which register consecutive reduction steps write: **120 mismatches → 0, 23/24 → 24/24**.

Our mitigation (default ON) rotates the left-operand register across steps, so the write following an
accumulate targets a register that accumulate is not reading. The upstream fix is to gate the `rvs1_mask`
clear on the last iteration that reads the group, exactly as `rvs2_mask` already is.

The same file also explains the hang: `col_idx_tail` compares against `(vLen/dLen) << (emul - eew)` with
an **unsigned** subtraction, so `e32`/`m1` (`emul = 0, eew = 2`) underflows, the shift becomes enormous,
the tail condition never matches and the sequencer iterates forever. The rule enforced by
`cca_matrix.vtype_violations` — `lmul * operand_bits >= sew` — is precisely `emul >= eew`.

#### What this cost, and the transferable lesson

Four hypotheses were proposed and refuted before the RTL was read: a shape rule (`M < N`, refuted by a
4×4 grid), the under-initialised broadcast (refuted — fixing it left the mismatch count unchanged),
operand alignment (refuted — a derived 16 B alignment changed nothing), and buffer size (refuted by
appending a large case *after* the target). Three were inferred from pass/fail bits across images; all
three were wrong, and two were stated as causes before being isolated.

Two things would have short-circuited all of it. **Reading the values instead of the verdict** — the
image now dumps `(index, device, reference)`, and one build of that settled what three rounds of
reasoning could not. And **reading the RTL**, which is ours: the mechanism, the exact failing columns and
the fix all fell out of thirty lines of the sequencer. The stable mismatch count (identical across
structurally different binaries) was the signal that something deterministic was wrong, and it was twice
explained away as garbage.

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
- ~~**`derived_levers` for the OPU.**~~ **Checked, and the answer is not the feared one.** It does not
  mis-derive `DIM=4`; it derives **nothing at all** — `legal_opcodes`, `memory_map` and `dim` all come
  back `None`, so `derived_levers` returns `[]`. That is worse than a wrong number in one specific way:
  an empty lever list is indistinguishable from "this accelerator has no structural levers", so a caller
  would report a bare target rather than a missing capability. `lever_derivation_gaps` now surfaces the
  silence (`discovered_nothing` → an explicit UNKNOWN), and a test pins the distinction between "read the
  RTL, found no mesh" and "read nothing". This is also the reason the matrix-unit CCA lifter derives
  residency from the emitted instruction stream rather than from this profile: the stream is there whether
  or not RTL discovery is.
- ~~**No facet for accumulator-bank count.**~~ **Answered: it is a lever, and the golden kernel does not
  use it.** `MatrixStreamFacts.matrix_registers_used` counts the DISTINCT destination registers of the
  accumulate instructions, so it reports what was emitted rather than what a schedule intended. The
  emitted microkernel reads **1**, i.e. it occupies one accumulator bank and leaves the rest of the MRF
  idle — a fact no MAC count or cycle total exposes. A test compiles a two-bank variant and requires the
  facet to read 2, so the lever is visible rather than merely named. Whether using more banks *pays* is a
  measurement, and it needs the L3 verdict first.
- **fp8.** The contract's dtype list is `[int8]` for the real RTL; the fp8 sub-format is surfaced
  honestly as `unnamed_float_datapaths: ["float8"]` because the RTL does not name it, and `OPFMACC` is
  not in `opuInsns` on the `opu-int8` branch. int8 only, in this pass.
