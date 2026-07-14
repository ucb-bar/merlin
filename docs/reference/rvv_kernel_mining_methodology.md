---
title: RVV kernel-mining methodology
kind: reference
status: current
owner: kernels
last_verified: 2026-07-14
related: [kernel_mining, dse]
code_refs: [merlin/python/merlin/rvvgen, merlin/python/merlin/kernels]
---

# Kernel-Mining → Compiler-Improvement: Methodology

*How we mine what expert kernels do, abstract it into target-agnostic structure, route it to
concrete compiler knobs, search the knob space with beam search, and certify every change against a
frozen baseline — with the split between deterministic tooling and LLM/agent stages made explicit.*

> Scope. This document is the **methodology and architecture** companion to the results writeups
> (the generated evidence report the `merlin-rvv-report` CLI writes under
> `out/artifacts/kernel-mining/rvv/`, and `out/artifacts/ceiling/RESULTS.md`). It describes *how the
> system works and how the artifacts interact*, not the measured numbers. RVV is the testbed; every
> abstraction is target-agnostic by construction (§9).

---

## 0. The one-paragraph version

We disassemble expert kernels (XNNPACK, OpenBLAS) and our own compiler's output to the **one
substrate every framework lowers to — assembly**. We lift both into a **Common Compute Abstraction
(CCA)**: target-agnostic facets (compute / vector / spatial / dataflow) reconstructed *structurally*
from the instruction stream, never by regex over text. We diff expert-CCA against ours-CCA to get
typed **Divergences**, route each divergence to a typed **CompilerAction** (FLAG / HEURISTIC / PASS /
KNOB) that names a concrete **seam** in our compiler, express that action as a **default-off feature**
(so the baseline stays byte-identical), and let a **beam search** explore combinations of those
features — certifying every candidate for *correctness first*, structural match to the expert
second, cycles last. The loop is deterministic end-to-end; an LLM is an *optional* proposer whose
suggestions are always clamped and validated before they can touch the compiler.

---

## 1. The mining loop (the whole process in one diagram)

```mermaid
flowchart TD
    subgraph CORPUS["Expert corpus + our compiler"]
        X["XNNPACK / OpenBLAS<br/>kernel objects (.o)"]
        O["our compiler output<br/>(lowered model .o)"]
    end

    subgraph DECODE["② DECODE — structured, no regex (kernels/decode/)"]
        T["objdump.tokenize()<br/>→ list[RawInsn]"]
        R["rvv.decode()<br/>→ InsnStream + vtype state machine"]
        M["memory.analyze_memory()<br/>→ MemFacet (loads/FMA, ladder)"]
    end

    subgraph LIFT["③ LIFT — Common Compute Abstraction (kernels/cca.py)"]
        LA["lift_asm() → CCA  (PRIMARY, level=asm)"]
        LS["lift_source() → CCA  (cross-check, level=source_ast)"]
        AG["cca_agree() → AgreementReport<br/>(validity gate: quarantine on disagreement)"]
    end

    subgraph COMPARE["④ COMPARE + ROUTE (kernels/cca_compare.py, action_catalog.py)"]
        D["compare() → list[Divergence]<br/>(axis, expert, ours, evidence)"]
        AC["build_catalog() → list[CompilerAction]<br/>+ unrouted (never dropped)"]
    end

    subgraph KNOBS["⑤ KNOBS — default-off features (llvmlower/impr_features.py)"]
        F["ImprFeature registry<br/>edit_schedule / edit_pipeline hooks"]
    end

    subgraph SEARCH["⑥ SEARCH — beam over knob/feature combinations (rvvgen/beam.py, autotune.py)"]
        B["run_beam(): expand → certify → rank → keep top-k"]
    end

    subgraph CERT["⑦ CERTIFY — K-ladder (rvvgen/runner.py)"]
        K["K0 load → K3 spike correctness (cos-gate) →<br/>K4 histogram → K5 K1 silicon → K6 Δ vs baseline"]
    end

    X --> T --> R --> LA
    O --> T
    R --> M --> LA
    X -.C source.-> LS
    LA --> AG
    LS --> AG
    AG -->|agree| D
    LA --> D
    D --> AC --> F --> B --> K
    K -->|residual divergence| D
    K -->|certified Δ| REPORT["evidence report<br/>(rvvgen/report.py, mine.py)"]
```

The dashed back-edge — *certified result → residual divergence → re-compare* — is the iterative
heart. Each turn closes a measured gap and re-lifts the residual from fresh asm. (Turn 1 = vfmacc
fusion; turn 2 = `.vf` A-scalarization; turn 3 = packing/memory-traffic facet, which **refuted** the
human "packing" hypothesis by measuring ours-`.vf` already ties XNNPACK on loads/FMA.)

### Why asm is the substrate

XNNPACK lowers from C intrinsics, OpenBLAS from hand asm, ours from MLIR — three different front
ends. The **only** representation where all three are directly comparable is the emitted RISC-V
instruction stream. So the authoritative lift is `lift_asm`; source-level lifting (`lift_source`,
from the clang AST) exists purely as a **cross-check** that gates validity (§3.3), not as the
primary signal.

---

## 2. The measurement substrate (the honesty floor)

Every claim is pinned to a rung of a measurement ladder, and the rung is always reported:

| Rung | What it is | What it measures | `cycle_accurate` |
|---|---|---|---|
| **spike** | functional ISS | `minstret` instruction count (IPC=1 proxy, **no** memory hierarchy) | `false` |
| **K1 board** | real SpacemiT silicon (`root@…`, VLEN=256) | `rdtime` ticks → wall-ns; cycles *derived* (`ticks × 1600/24 MHz`) | `false` |
| **FireSim / VCS** | cycle-accurate RTL sim | true cycles | `true` (queue-gated, never escalated) |

Consequences we state openly: a loads/FMA improvement is invisible on spike (a load costs the same
as an FMA there) and only shows on the board / FireSim; the board's `cycles` is a derived estimate,
so `wall_ns` is the ground truth there. When a rung is unreachable the result is recorded as
`not_run`, never fabricated.

---

## 3. Mining & tooling, in detail

This is the part that must generalize. Everything here is a **deterministic tool** — no LLM in the
extraction path — so the same input always yields the same CCA, the same divergences, the same
actions.

### 3.1 Decode — structured disassembly, zero semantic regex

The old extractor (`kernels/features/rvv_intrinsics.py`) was **7 regex patterns over C source**
(`_E_LMUL = r"_e(\d+)m(f?\d+)\b"` guessing LMUL from a substring of an intrinsic name). It survives
only as a low-confidence source cross-check. The authoritative path is three structured decoders:

```mermaid
flowchart LR
    OBJ([".o object"]) --> TOK
    TOK["objdump.tokenize(triple='riscv64')<br/>llvm-objdump -d -M no-aliases<br/>positional field-split → RawInsn{addr,mnemonic,operands,hexcode}"]
    TOK --> DEC["rvv.decode()<br/>vtype state machine"]
    DEC --> STREAM["InsnStream<br/>list[VInsn{raw, is_vector, vtype}]"]
    STREAM --> HIST["vector_histogram(), vtype_histogram()<br/>has_loop(), loop_spans(), innermost_loop()"]
    STREAM --> MEM["memory.analyze_memory(span)<br/>→ MemFacet"]
```

**`objdump.tokenize()`** disassembles with `llvm-objdump -d --triple=riscv64 -M no-aliases` and parses
each line *positionally* — split on `:` for the address, whitespace for `[hex, mnemonic, operands]`,
commas for operands. No regex guesses meaning; it just tokenizes.

```python
@dataclass
class RawInsn:
    addr: int            # byte address within section
    mnemonic: str        # "vfmacc.vf", "vsetivli", "addi"
    operands: list[str]  # comma-split, stripped
    hexcode: str = ""
    section: str = ""
```

**`rvv.decode()`** runs a **vtype state machine**. RVV encodes element width and grouping in explicit
`vsetvli`/`vsetivli` operands (`e32, m4, ta, ma`); the decoder reads those operand tokens directly,
tracks the *current* `VType`, and stamps every following vector instruction with its **effective**
vtype. This is what lets us say "this `vfmacc.vf` runs at SEW=32, LMUL=4" without ever pattern-matching
a string.

```python
@dataclass(frozen=True)
class VType:                       # parsed from explicit operands, never guessed
    sew: int | None;  lmul: float | None;  tail: str | None;  mask: str | None

@dataclass
class VInsn:
    raw: RawInsn;  is_vector: bool;  vtype: VType | None   # effective vtype at this insn
```

`InsnStream` then exposes the structural queries the lifter needs: histograms, loop-back-edge spans
(`loop_spans()` resolves branch targets robustly across `0x1dc` and `0x1dc <sym+…>` forms),
`innermost_loop()`, and span-scoped counts (`count_in(span, "vfmacc")`).

**`memory.analyze_memory()`** is the turn-3 facet — the CCA's compute/vector facets captured *what*
the FMA does but were blind to *how operands reach it*. Scoped to the K-reduction loop, it classifies
every load by mnemonic prefix (again, the ISA encodes addressing mode in the opcode):

```python
@dataclass
class MemFacet:
    fma_in_loop: int
    vec_unit_loads: int        # vle / vlNre   — packed-panel (the expert shape)
    vec_strided_loads: int     # vlse          — 2-D model-layout gather
    vec_indexed_loads: int     # vlux / vlox   — scatter/gather
    scalar_loads: int          # flw / fld     — the .vf A scalar
    broadcast_ladder_ops: int  # vslideup/vmv/vrgather — rebuilding A per K step (the .vv cost)
    loads_per_fma: float | None
    a_broadcast_per_fma: float | None
    unit_stride_only: bool
```

This single facet is what let us *measure* — not assert — that the `.vf` kernel ties XNNPACK at
**2.0 loads/FMA, unit-stride, 0 ladder ops**, and pin the only remaining lever (OpenBLAS's MR>1
A-reuse) as structurally inapplicable to small-M VLA matmuls.

### 3.2 Lift — the Common Compute Abstraction (CCA)

The CCA is the generalization layer. It is **per-op-region** and **target-agnostic**: a `backend`
tag plus four facets, only the relevant ones populated. RVV fills `compute` + `vector`; Gemmini would
fill `compute` + `spatial`; an NPU fills `dataflow`; a heterogeneous model is N regions each with its
own backend tag.

```python
@dataclass
class CCA:
    op: str
    backend: list[str]                 # ["rvv"] | ["gemmini"] | ["npu","rvv"] | ...
    compute:  ComputeFacet             # target-agnostic
    vector:   VectorFacet | None       # RVV/SIMD
    spatial:  SpatialFacet | None      # Gemmini/systolic
    dataflow: DataflowFacet | None     # NPU
    provenance: dict                   # {"level": "asm"|"source_ast"|"dse", "source", "confidence"}

@dataclass
class ComputeFacet:
    op; contraction_form           # "fused_fma" | "mul_add" | "outerproduct" | "dot" | "systolic"
    accumulator_dtype; widening; reduction_form
    register_block                 # (mr, nr); nr = ("vsetvlmax", lmul) — symbolic, VLEN-relative
    epilogue; accumulator_resident; nr_is_vsetvlmax
```

What makes the lift trustworthy is that every field is **structurally inferred**, not read off a name:

- **`contraction_form`** — `fused_fma` iff `vfmacc/vmacc` count > 0; `mul_add` iff separate
  `vfmul`+`vfadd`. (This is the turn-1 gap: our `outerproduct` lowering emitted `vfmul`+`vfadd`,
  the experts emitted `vfmacc`.)
- **`accumulator_resident`** — a **spill detector**: walk the FMA loop; if a whole-register store
  (`vs1r/vs2r/vs4r/vs8r`) + reload (`vl1re/…`) of the MAC destination appears *inside* the loop, the
  accumulator is round-tripping through memory → `False`; absent → `True`; no FMA loop → `None`.
- **`register_block` (MR, NR)** — MR = count of distinct accumulator vregs fed by the broadcast MAC
  chain in the K-loop; NR = `("vsetvlmax", lmul)` (symbolic lanes, LMUL-scaled relative to VLEN — not
  an absolute lane count, so it ports across VLENs).

### 3.3 Cross-level agreement — the validity gate (why mining doesn't overfit)

The same op is lifted from **two** levels: asm (primary) and the clang-AST source facts
(`clang_ast.extract()` → `SourceFacts`, which resolves SEW/LMUL from the *resolved C type*
`vfloat32m4_t`, not the intrinsic name). `cca_agree(asm_cca, source_cca)` compares them **per facet
field**:

```python
@dataclass
class AgreementReport:
    agree: bool
    disagreements: list[str]   # ["vector.lmul: asm=4 vs source_ast=2", ...]
    compared_fields: list[str]
```

A kernel whose asm- and source-lifted CCAs disagree on a populated field is **quarantined** — it
cannot promote a divergence into a compiler action. This is the built-in defence against reading a
decode artifact as a real algorithmic decision, and it is tested *both ways* (agreement on a curated
kernel; the gate *firing* on a deliberately mismatched pair).

### 3.4 A worked CCA artifact (the concrete example)

Here is the abstraction made concrete: the turn-1 GEMM K-loop, lifted from both expert and our
asm, the divergence between them, and the action it routes to. This is the kind of record that lands
in a `mining_rvv_v{V}_{ts}/` run.

**Expert — XNNPACK `1x4v` f32 GEMM micro-kernel** (lifted from its `.o`):

```yaml
op: matmul
backend: [rvv]
compute:
  contraction_form: fused_fma          # vfmacc.vf present
  accumulator_dtype: f32
  register_block: [1, ["vsetvlmax", 4]]   # MR=1, NR = LMUL-4 lanes
  accumulator_resident: true           # no spill store/reload inside K-loop
  nr_is_vsetvlmax: true
vector: {sew: 32, lmul: 4, vl_strategy: vsetvl_loop, tail: ta}
provenance: {level: asm, source: xnnpack/…/1x4v-rvv.o, confidence: high}
# memory facet (analyze_memory, K-loop one trip):
#   vec_unit_loads=1, scalar_loads=1, broadcast_ladder_ops=0, loads_per_fma=2.0, unit_stride_only=true
```

**Ours — baseline `outerproduct` lowering of the same matmul** (lifted from our `.o`):

```yaml
op: matmul
backend: [rvv]
compute:
  contraction_form: mul_add            # vfmul.vv + vfadd.vv — NO vfmacc
  accumulator_dtype: f32
  accumulator_resident: false          # vs4r spill + vl4re reload inside the K-loop
  register_block: [1, ["vsetvlmax", 4]]
vector: {sew: 32, lmul: 4, vl_strategy: vsetvl_loop, tail: ta}
provenance: {level: asm, source: ours/openvla_fp32/model.o, confidence: high}
```

**Divergences** (`compare()` — only differing, both-non-None fields):

```yaml
- axis: compute.contraction_form
  expert: fused_fma
  ours:   mul_add
  backend: rvv
  evidence: [xnnpack_rvv_gemm, openblas_rvv_gemm]
- axis: compute.accumulator_resident
  expert: true
  ours:   false
  backend: rvv
  evidence: [xnnpack_rvv_gemm]
```

**Actions** (`build_catalog()` — typed, each names a seam):

```yaml
- divergence_axis: compute.contraction_form
  action_class: PASS
  target_seam: "impr_features:fused_vfmacc_contraction"
  change: "form a real vector.contract / inject fastmath<contract> so asm carries vfmacc not vfmul+vfadd"
  forkable_now: true        # registered feature exists and is measured to work
  expected_effect: "vfmacc replaces vfmul+vfadd; MEASURED ~7.9x on the isolated kernel"
- divergence_axis: compute.accumulator_resident
  action_class: PASS
  target_seam: "impr_features:accumulator_resident_microkernel"
  change: "keep MAC chain in vreg group across K; eliminate per-tile spill/reload"
  forkable_now: true
  expected_effect: "removes vs4r/vl4re round-trip inside K-loop"
```

That YAML triplet — **lifted CCA → typed divergence → seam-named action** — is the unit of "what the
expert does that we don't, and exactly where in the compiler to change it." It is human-readable,
diffable, and reproducible from the `.o` files alone.

---

## 4. Utilizing what we capture — the compiler-knob layer

The action catalog answers *what to change*. The **knob layer** is *where and how* — and its hard
invariant is that **the baseline never moves**.

### 4.1 The four action classes and their seams

Every `CompilerAction.action_class` maps to exactly one compiler seam:

| Class | Seam | Mechanism |
|---|---|---|
| **KNOB** | `transform_schedule=` | additive edit to the transform-dialect schedule (e.g. tile/vector sizes, NR=vsetvlmax) |
| **PATTERN** | `schedule.mlir` (full replace) | a feature with `schedule_replace=True` swaps the whole schedule |
| **PASS** | MLIR pass list via `build_rvv_pipeline(features=)` | inserts/edits lowering passes |
| **HEURISTIC** | MLIR pass list | same seam as PASS, semantically a policy choice |
| **FLAG** | `cflags_override=` | C-compiler flag (e.g. `-ffp-contract=fast`) |
| **CODEGEN** | (marker, no edit) | a *ceiling reference* (hand-written driver) — names the work, emits nothing |

### 4.2 The default-off feature registry (`llvmlower/impr_features.py`)

Each PASS/HEURISTIC/PATTERN action is expressed as a named, **default-off** `ImprFeature`:

```python
@dataclass(frozen=True)
class ImprFeature:
    name: str
    action_class: str                  # PASS | HEURISTIC | PATTERN | CODEGEN
    description: str
    edit_pipeline: Callable[[list[str]], list[str]] | None = None   # mutate the MLIR pass list
    edit_schedule: Callable[[str], str] | None = None               # mutate the transform schedule
    schedule_replace: bool = False                                  # full-schedule replacement?
```

The registry today holds ~48 features — the mined headline families plus their MR/NR/KC tuning grids.
The ones that carry the story:

- `fused_vfmacc_contraction`, `fused_vfmacc_tiled` / `_scalable` — the turn-1 vfmacc PASS.
- `accumulator_resident_microkernel` (v1) → `…_v2` (PRE-bufferize subset hoist) →
  `accumulator_resident_microkernel_v3` (the `.vf` A-scalarization) — the turn-2 residency line.
- `accumulator_resident_wholemodel_vf` (v3 + whole-model tail clamps MR_mm=1 / NR_bmm=8) and
  `…_vf_mr4` (the turn-3 MR=4 A-reuse register block).
- `vectorized_transcendental_activation` — minimax-poly exp/erf/tanh + elementwise vectorize.
- `lmul_widen_n` (KNOB), `intrinsic_microkernel` (CODEGEN ceiling marker, no edit).

### 4.3 The byte-identical baseline guarantee

The composition functions are identity on the empty set, which is the whole safety story:

```python
def apply_pipeline(passes, features):
    if not features: return passes                 # identity → byte-identical pass string
    out = list(passes)
    for name in sorted(features):                  # deterministic order
        f = get(name)
        if f.edit_pipeline: out = f.edit_pipeline(out)
    return out

def apply_schedule(text, features):
    if not features: return text                   # identity → byte-identical schedule
    # at most ONE schedule_replace feature (else CompositionError); additive edits layer on top
```

Guarded by `test_impr_features.py::test_empty_features_pipeline_byte_identical` and
`…_schedule_byte_identical`. So `hand_v0` (the frozen baseline, `compiler_features: []`) emits the
exact shipping pipeline; a feature can only change asm *when a fork explicitly enables it*.

### 4.4 Threading and lineage

Features flow `RvvPackage.compiler_features → apply_rvv_package → build_app → lower_model →
lower_to_llvm_ir(features=) → {apply_schedule, build_rvv_pipeline(features=)}`. Every fork is a
versioned directory with a manifest whose `lineage` records `parent_run_id: hand_v0`, `version`,
`depth`, the `action` block (`{class, seam}`), and `source_evidence` (the mined policies that
justified it):

```
out/artifacts/targets/rvv/
  hand_v0/                       # immutable baseline; status=shipped; features=[]
  impr_auto_1_<ts>/              # fork; parent_run_id=hand_v0; features=[fused_vfmacc_contraction,…]
  rvv_tuned_v1_d1_<ts>/          # beam fork: v=generation, d=beam depth
out/artifacts/kernel-mining/rvv/
  mining_rvv_v{V}_{ts}/          # CCAs, agreement report, divergences, action catalog
```

Forks are **isolated**: they never edit `pipeline.py` defaults; they are reproducible from manifest +
source alone.

---

## 5. Beam search — why, and how beams combine

### 5.1 Why beam search

The knob space (tile × vector × contraction strategy × lowering patterns × dtype × feature subsets)
is combinatorial. Two extremes fail: greedy hill-climbing gets trapped (vfmacc alone is neutral until
paired with residency); exhaustive 2^N over features is intractable. Beam search is the middle —
**breadth-first but depth- and width-limited**, expanding only surviving branches, ranked toward the
expert oracle rather than a raw scalar. It is also the natural fit for the iterative loop: each
generation's residual divergences seed the next generation's proposals.

### 5.2 The search (`rvvgen/beam.py::run_beam`)

```python
def run_beam(seed_pkg, model_dir, curated_text, op_key, *, runs_root,
             width=3, depth=2, top_k=2, target="rvv", targets=("spike",),
             certify_fn=certify_rvv, proposer=propose_forks,
             loader=load_rvv_package, minter=mint_fork) -> dict: ...
```

```mermaid
flowchart TD
    S["seed = hand_v0<br/>certify → rank"] --> G1
    subgraph G1["generation d=1"]
        P1["proposer(parent.divergences, parent.knobs)<br/>→ ForkProposal[]  (take forkable[:width])"]
        MINT1["minter: merge parent.knobs + overrides<br/>→ schedule.mlir + manifest (lineage)"]
        CERT1["run_sweep → certify_fn in parallel + isolated"]
        RANK1["rank_results: correctness → structural_match → cycles<br/>keep top_k survivors"]
        P1 --> MINT1 --> CERT1 --> RANK1
    end
    RANK1 -->|survivors become parents| G2["generation d=2 …"]
    RANK1 -.forkable=False.-> DEF["deferred_work_items[]<br/>(needs a compiler feature)"]
    G2 --> OUT["beam_tree.yaml<br/>(nodes, best, deferred_work_items)"]
```

**Search space** = knob overrides derived from the parent's S4 divergences: `op_match` (tile +
vector size lists), `contraction_strategy` (`outerproduct|dot|matmulintrinsics|parallelarith`),
`lowering_patterns`, `dtype_strategy`.

**Scoring is lexicographic — correctness dominates, cycles only break ties:**

```python
def key(n):
    correct = 1 if n["gate_ok"] else 0
    sm      = n["structural_match"] or 0.0     # 0.6*decision_match + 0.4*histogram_cosine
    cyc     = n["cycles"]
    return (correct, sm, -(cyc if cyc is not None else inf))
```

`structural_match = 0.6·(matched RVV decisions / 6) + 0.4·cosine(expert_histogram, ours_histogram)`
— so the beam climbs *toward the expert's fingerprint*, and an incorrect candidate can never
out-rank a correct one. A fork with no correctness gate gets `speedup = None` (fail-closed, §7).

### 5.3 How beams merge / combine

There are **two composition mechanisms**, deliberately:

1. **Implicit knob merge (`beam.py`).** Each fork deep-copies the parent's full knob dict and
   `knobs.update(overrides)`. So a child *inherits everything the parent had* plus one new lever. A
   winner two levels deep is literally `seed → (widen N) → (widen N + outerproduct)` — the beam
   discovers beneficial **knob combinations** by depth traversal, no explicit set algebra.

2. **Explicit greedy feature extension (`autotune.py::beam_search`).** This variant searches over
   **feature *sets*** ranked by *measured K1 wall-time*. A survivor with features `{A,B}` spawns
   candidates `{A,B,C}, {A,B,D}, …` — one feature added per step:

   ```python
   for d in range(1, depth):
       cands = {frozenset(p["features"]) | {f}
                for p in survivors for f in features if f not in p["features"]}
       rs = [evalc(c) for c in cands]
       allcorrect = sorted([r for r in survivors+rs if r["correct"]], key=lambda r: r["min_wall"])
       survivors = allcorrect[:width]
   ```

   This is exactly how the *neutral-alone, synergistic-together* pair was found: `vfmacc` only pays
   off once `accumulator_resident` is also in the set, and greedy extension surfaces the combination
   without enumerating all 2^N subsets.

Branches that can't be expressed as a knob today (forkable=False — e.g. a VL-polymorphic-tail PASS
that doesn't exist yet) are not discarded: they are logged to `beam_tree.yaml["deferred_work_items"]`
as the explicit "next compiler feature to build" list. No silent truncation.

---

## 6. Tooling vs. agents — who does what

The pipeline is **deterministic by default**. An LLM appears in exactly one *optional, sandboxed*
role (RVV) and one *separate* generative system (Gemmini targetgen). Everything else is tooling.

```mermaid
flowchart LR
    subgraph TOOLS["DETERMINISTIC TOOLING (no LLM)"]
        direction TB
        decode["decode/* — objdump, vtype machine, memory facet"]
        lift["cca lift_asm / lift_source / cca_agree"]
        comp["cca_compare.compare → Divergences"]
        cat["action_catalog.build_catalog → CompilerActions"]
        mint["from_strategy.mint_fork — knob merge → schedule.mlir"]
        cert["runner.certify_rvv — K-ladder + cos-gate"]
        rank["sweep.rank_results"]
    end
    subgraph AGENT["OPTIONAL LLM (RVV proposer)"]
        prop["tuning_agent.propose_forks_llm<br/>tool: common.llm.complete"]
        clamp["_clamp_overrides — validate vs _KNOWN_OVERRIDE_KEYS<br/>drop unknown keys, note them"]
    end
    cat --> prop --> clamp --> mint
    cat -. default .-> dgr["rvv_knobs.propose_forks<br/>(deterministic gap-router)"] --> mint
    decode --> lift --> comp --> cat
    mint --> cert --> rank
```

- **Decode → lift → compare → catalog → mint → certify → rank: 100% deterministic.** Same inputs ⇒
  same outputs. This is the auditable spine.
- **Proposer is pluggable.** The default `kernels/rvv_knobs.py::propose_forks` is a deterministic
  **gap-router**: a divergence key → a fixed list of levers (e.g. "lmul_class" → widen-N knob). The
  **optional** `rvvgen/tuning_agent.py::propose_forks_llm` renders a versioned prompt from the
  divergences + knobs, calls `common.llm.complete` (its only tool), parses JSON proposals — and then
  **every override is clamped against `_KNOWN_OVERRIDE_KEYS`; unknown keys are dropped and noted.**
  No LLM output reaches the compiler unvalidated. Both proposers emit the identical `ForkProposal`
  type, so the beam can't tell which produced a candidate.
- **Gemmini targetgen agent** (`merlin/python/merlin/targetgen/agent/`) is a *separate* system: it
  drives the Claude Code CLI headless to generate bare-metal Gemmini C kernels under a
  propose → visible-rung test → feedback → held-out-certification discipline. It is **not** part of
  the RVV beam; it shares only the gate philosophy (nothing is trusted until it certifies).

---

## 7. Certification — the K-ladder and the cos-gate

`runner.certify_rvv` runs every candidate through a fail-closed ladder; the beam consumes its
structured result:

| Rung | Check | Fails closed? |
|---|---|---|
| K0 | load + integrity | yes |
| K1 | non-perturbation (seed == canonical schedule) | yes |
| K2 | build via `apply_rvv_package` | yes |
| **K3** | **spike correctness — cosine(output, golden) ≥ 0.999** | **yes (the gate)** |
| K4 | instruction histogram (evidence, not a gate) | no |
| K5 | K1 silicon cycles (`rdtime`→cycles + `wall_ns`); `not_run` if board unreachable | no |
| K6 | Δ vs baseline — **speedup credited only if K3 passed** | yes |

```python
gate_ok = bool(rec["correctness"]["gate_ok"])
speedup = (baseline_cycles / m["cycles"]) if gate_ok else None   # incorrect ⇒ no number, ever
```

A candidate that is fast but wrong reports `speedup = None`. This is the structural reason the headline
numbers are defensible: nothing that fails the cosine gate can produce a speedup claim, and the rung
(`spike` vs `K1` vs `not_run`) is always attached.

---

## 8. End-to-end artifact interaction (how the pieces talk)

```mermaid
flowchart TD
    objs[".o files (expert + ours)"] -->|decode/*| stream["InsnStream + MemFacet"]
    stream -->|cca.lift_asm| cca["CCA records"]
    src["C source"] -->|clang_ast + lift_source| ccaS["CCA (source)"]
    cca & ccaS -->|cca_agree| gate{"agree?"}
    gate -->|no| quarantine["quarantined (no promotion)"]
    gate -->|yes| div["Divergences (cca_compare)"]
    div -->|action_catalog| actions["CompilerActions (typed, seam-named)"]
    actions -->|forkable| feat["impr_features (default-off)"]
    actions -->|not forkable| deferred["deferred work items"]
    feat -->|proposer → mint_fork| fork["impr_ fork pkg<br/>schedule.mlir + knobs.yaml + manifest(lineage)"]
    fork -->|apply_rvv_package → build_app → lower → pipeline(features=)| elf["zephyr.elf"]
    elf -->|certify_rvv K-ladder| result["certified result (gate_ok, cycles, wall_ns)"]
    result -->|rank_results / beam| best["survivors + beam_tree.yaml"]
    result -->|re-decode residual| stream
    best & actions & div -->|report.py / mine.py| report["evidence report (md)"]
```

Reading it as a sentence: **objects → InsnStream → CCA → (agreement gate) → Divergences → typed
Actions → default-off Features → minted Forks → built ELF → K-ladder certification → ranked
survivors**, with two back-edges (residual re-decode for the next turn; quarantine for kernels that
fail the validity gate) and one terminal sink (the evidence report). Every box is a real module named
in §3–§7.

---

## 9. Why this generalizes (and doesn't overfit to RVV)

Three design choices keep the methodology target-agnostic:

1. **Asm is the common substrate.** `objdump.tokenize()` is ISA-agnostic; `decode(obj, isa)`
   dispatches per ISA. RVV is the first instantiation; the Gemmini RoCC decoder
   (`targetgen/rocc_decode.py`) is the sibling precedent. A new target adds a decoder, not a rewrite.
2. **The CCA facets are split by execution model, not by ISA.** `compute` is universal; `vector`,
   `spatial`, `dataflow` are populated only where they apply, and a composite model (NPU+RVV) is just
   N regions with different `backend` tags. Nothing in the comparator or action catalog is
   RVV-specific except the *route table* (`_ROUTES["rvv"]`), which is data, not code.
3. **The beam is plugin-shaped.** `run_beam(loader, minter, proposer, certify_fn)` are all injectable
   (see `rvvgen/TARGET_PLUGIN.md`); a new target reuses the orchestration unchanged by supplying its
   own four callables. The cross-level agreement gate, the byte-identical-baseline invariant, and the
   fail-closed cos-gate are target-independent.

The validity gate (§3.3) is the specific anti-overfitting mechanism: a divergence can only become a
compiler action if it agrees across asm and source levels, so we never "optimize" toward a decode
artifact. And the iterative loop has twice corrected a *human* hypothesis with a *measured* refutation
(lane-width, then packing) — the system is built to overrule its operators when the asm disagrees.

---

## 10. Pointers (where each thing lives)

| Concern | Module |
|---|---|
| Tokenize / decode | `kernels/decode/{objdump,rvv,memory,clang_ast}.py` |
| CCA + lifters + agreement | `kernels/cca.py` |
| Comparator / divergences | `kernels/cca_compare.py` |
| Typed action catalog | `kernels/action_catalog.py` |
| Default-off feature registry | `llvmlower/impr_features.py` |
| Pipeline / schedule seams | `llvmlower/pipeline.py` (`build_rvv_pipeline(features=)`, `lower_to_llvm_ir`) |
| Fork minting + lineage | `rvvgen/fork.py`, `rvvgen/from_strategy.py` |
| Apply / threading | `rvvgen/apply.py`, `runtime/backends/zephyr_model.py::build_app` |
| Beam (knob merge) | `rvvgen/beam.py` |
| Beam (feature extension, K1-ranked) | `rvvgen/autotune.py` |
| Deterministic proposer (gap-router) | `kernels/rvv_knobs.py` |
| Optional LLM proposer | `rvvgen/tuning_agent.py` |
| K-ladder certification | `rvvgen/runner.py` |
| Ranking / sweep | `rvvgen/sweep.py` |
| Evidence report / driver | `rvvgen/report.py`, `rvvgen/mine.py` |
| Gemmini targetgen agent | `targetgen/agent/`, `targetgen/rocc_decode.py` |
| Tests | `tests/test_decode_rvv.py`, `test_cca.py`, `test_action_catalog.py`, `test_impr_features.py`, `test_rvv_beam.py`, `test_tuning_agent.py`, `test_rvv_runner.py` |

---
---

# Part II — Deep dives (the implementation, exactly)

*Part I is the architecture. Part II is the precise mechanics: the full CCA spec and how it is
generated, the beam search end-to-end (inputs / what's fixed / what varies / outputs / where the
agent plugs in), a real improvement shown as a before/after diff, and the certify-and-measure loop.
Everything here is quoted from or reduced from the actual source — file + symbol named in each case.*

---

## 11. The CCA, in full

### 11.1 The exact spec (`kernels/cca.py`)

A `CCA` is **one op-region**: a `backend` tag plus four facets, only the relevant ones populated,
plus `provenance`. Verbatim dataclasses (`kernels/cca.py:24-78`):

```python
@dataclass
class ComputeFacet:                              # TARGET-AGNOSTIC — every backend fills this
    op:                  str | None = None
    contraction_form:    str | None = None   # fused_fma | mul_add | outerproduct | dot | systolic
    accumulator_dtype:   str | None = None   # f32 | i32 | f64 | ...
    widening:            bool | None = None  # i8×i8→i32 widening MAC (vwmacc / array widen)
    reduction_form:      str | None = None   # tree | vredsum | vfredusum | none
    register_block:      tuple | None = None # (mr, nr) or tile dims
    epilogue:            str | None = None   # requant_narrow | none
    accumulator_resident: bool | None = None # acc stays in fastest storage across the WHOLE reduction?
    nr_is_vsetvlmax:     bool | None = None  # inner width tracks runtime vsetvlmax (VL-adaptive)?

@dataclass
class VectorFacet:                               # RVV / SIMD
    sew:         int   | None = None         # 8 | 16 | 32 | 64
    lmul:        float | None = None         # 8,4,2,1,0.5,0.25,0.125 (mf2/mf4/mf8 → fractions)
    vl_strategy: str   | None = None         # vsetvl_loop | vsetivli_fixed
    tail:        str   | None = None         # ta | tu | none

@dataclass
class SpatialFacet:                              # Gemmini / systolic
    pe_rows: int | None;  pe_cols: int | None;  dataflow: str | None   # ws | os
    accumulator_resident: bool | None = None

@dataclass
class DataflowFacet:                             # NPU
    engine_ops: list[str];  dma_pattern: str | None;  onchip_resident: str | None

@dataclass
class CCA:
    op: str
    backend: list[str]                           # ["rvv"] | ["gemmini"] | ["npu","rvv"] | …
    compute:  ComputeFacet
    vector:   VectorFacet   | None = None
    spatial:  SpatialFacet  | None = None
    dataflow: DataflowFacet | None = None
    provenance: dict        = …                  # {"level": asm|source_ast|mlir|dse, "source", "confidence"}
```

`accumulator_resident` lives on the **shared** `ComputeFacet` on purpose: "does the output stay in
the fastest storage across the whole reduction and commit once?" is the *same question* on every
backend — RVV reads it from a vfmacc-chain / no-in-loop-spill analysis, `SpatialFacet` is the
Gemmini-PE view of the identical concept, an NPU lifter reads it from on-chip-buffer residency. That
is what makes the abstraction portable rather than RVV-shaped.

### 11.2 Kernel-CCA vs compiler-CCA — same schema, different lifter

There is **one** CCA type. "Expert-kernel CCA" and "our-compiler CCA" differ only in which **lifter**
produced them and what `provenance.level`/`confidence` they carry. Four lifters exist
(`kernels/cca.py`):

| Lifter | Input | Level / confidence | Used for |
|---|---|---|---|
| `lift_asm(stream, op, source, backend)` | decoded `InsnStream` (RVV asm) | `asm` / **high** | **both** the expert kernel's `.o` *and* our compiler's `model.o` — the authoritative, comparable lift |
| `lift_source(facts, …)` | `clang_ast.SourceFacts` (resolved C-intrinsic types) | `source_ast` / medium | expert kernels only — the **cross-check** that gates validity (§3.3) |
| `lift_spatial(op_counts, …)` | Gemmini RoCC op counts (`targetgen/rocc_decode`) | `asm` / high | a spatial backend (fills `SpatialFacet`) |
| `lift_dse(op_shape, …)` | `dse_guidance.OperatorShape` geometry | `dse` / low | op + register-block hints only (no micro-decisions) |

The crucial point: **expert-CCA and ours-CCA are both produced by `lift_asm` from the same
instruction stream**, so the comparison in §11.4 is apples-to-apples — XNNPACK (C-intrinsic front
end), OpenBLAS (hand asm), and ours (MLIR front end) all reduce to the one substrate they share. The
"compiler CCA" is *not* lifted from our MLIR in the hot path — MLIR-level lifting (`lift_mlir`,
`level=mlir`) exists only as a faithfulness check ("did our IR lower to the asm we think it did"); the
authoritative ours-CCA is `lift_asm(model.o)`.

`lift_asm` reads **every field structurally** from the stream, never from a name
(`kernels/cca.py`, `lift_asm`):

```python
vfmacc      = stream.count("vfmacc", "vmacc")          # fused MAC present?
vfmul, vfadd= stream.count("vfmul","vmul"), stream.count("vfadd","vadd")
widening    = stream.count("vwmacc") > 0
sew, lmul   = _dominant_vtype(stream)                  # from tracked vtype, not a substring
vl_strategy = "vsetvl_loop" if (stream.count("vsetvli")>0 and stream.has_loop()) else "vsetivli_fixed"
contraction = ("fused_fma" if vfmacc>0 else "mul_add" if (vfmul>0 and vfadd>0) else None)
acc_resident= _infer_accumulator_resident(stream)      # spill-store/reload inside the FMA loop?
reg_block   = _infer_register_block(stream, sew, lmul) # MR=distinct acc vregs; NR=("vsetvlmax",lmul)
```

### 11.3 How a CCA is generated — the inputs and the exact process

**Inputs** (all on-disk, no model/agent in this path): an ELF/`.o` object (expert kernel object, or
our `model.o` from a certified build) and the LLVM tools at `third_party/llvm-install/bin`
(`llvm-objdump`; for the source cross-check `clang-23` + the framework's `riscv_vector.h` path from
its `framework_contract`).

**Process** — deterministic, four steps (`kernels/decode/` → `kernels/cca.py`):

```
.o ──objdump.tokenize()──▶ list[RawInsn]            # positional field-split, no semantic regex
   ──rvv.decode()────────▶ InsnStream               # vtype state machine stamps each insn's effective vtype
   ──memory.analyze_memory(K-loop span)──▶ MemFacet  # loads/FMA, ladder ops, unit-stride (turn-3 facet)
   ──cca.lift_asm()──────▶ CCA{compute,vector,provenance:{level:asm,confidence:high}}
```

The vtype state machine (`kernels/decode/rvv.py:decode`) is the core of "no regex": it walks the raw
instructions, and on each `vsetvli/vsetivli/vsetvl` it parses the **explicit** `e<SEW>,m<LMUL>,ta/ma`
operand tokens into a `VType`, then stamps every following vector instruction with that *effective*
vtype:

```python
def decode(obj_path, triple="riscv64") -> InsnStream:
    raws = tokenize(obj_path, triple=triple)
    cur = VType()                                  # current architectural vtype
    out = []
    for r in raws:
        if r.mnemonic in _VSET:                    # vsetvli | vsetivli | vsetvl
            cur = _parse_vtype(r.operands)         # read e32/m4/ta/ma operands directly
            out.append(VInsn(raw=r, is_vector=True, vtype=cur)); continue
        is_vec = r.mnemonic.startswith("v")
        out.append(VInsn(raw=r, is_vector=is_vec, vtype=cur if is_vec else None))
    return InsnStream(out)
```

So "this `vfmacc.vf` runs at SEW=32, LMUL=4" is *read off the architectural state*, not guessed from
the string `_e32m4`. The old regex extractor (`kernels/features/rvv_intrinsics.py`, 7 patterns over C
text) survives only as a low-confidence fallback.

**For our compiler**, the only difference is where the `.o` comes from: `certify_rvv` builds
`model.o` via `apply_rvv_package`, disassembles it (K4), and the same `lift_asm` runs on it. Same
function, same fields — that is what makes "expert vs ours" a valid diff.

### 11.4 The validity gate, exactly (`cca_agree`)

`cca_agree(asm_cca, source_cca)` compares the two lifts **per populated facet field** (a field that
is `None` on either side is skipped) and quarantines any kernel that disagrees
(`kernels/cca.py:cca_agree`):

```python
for facet in ("compute","vector","spatial","dataflow"):
    for k, (va, vb) in _facet_fields(getattr(a,facet), getattr(b,facet)).items():
        compared.append(f"{facet}.{k}")
        if va != vb:
            diffs.append(f"{facet}.{k}: {a.provenance['level']}={va!r} vs {b.provenance['level']}={vb!r}")
return AgreementReport(agree=not diffs, disagreements=diffs, compared_fields=compared)
```

A disagreement on any populated field ⇒ `agree=False` ⇒ the divergence cannot promote to a compiler
action. Tested **both ways** (`test_cca.py`): agreement on a curated kernel, and the gate *firing* on
a deliberately mismatched pair.

### 11.5 Divergence and action — the exact records

`compare(expert, ours)` (`kernels/cca_compare.py`) emits one `Divergence` per populated **differing**
field; `route(divergence)` (`kernels/action_catalog.py`) maps each to a typed `CompilerAction`:

```python
@dataclass
class Divergence:
    axis: str            # "compute.contraction_form"
    expert: Any; ours: Any
    backend: str
    evidence: list[str]  # kernel ids justifying the expert value

@dataclass
class CompilerAction:
    divergence_axis: str
    action_class: str    # FLAG | HEURISTIC | PASS | KNOB  (+ PATTERN, CODEGEN)
    target_seam: str     # "impr_features:fused_vfmacc_contraction" | "schedule:vector_sizes" | "cflag:…"
    change: str; expected_effect: str
    forkable_now: bool   # is the seam a registered feature TODAY?
    backend: str; evidence: list[str]
```

The route table is **data, keyed by backend** (`_ROUTES["rvv"]`, ~10 routes) — adding a target adds
rows, not code. Routes that aren't expressible as a feature yet carry `forkable_now=False` and become
the explicit "next feature to build" list (never dropped). §3.4 shows the full worked YAML triplet.

---

## 12. The beam search, in full

### 12.1 What a node is, and the two engines

A beam **node** is one *candidate compiler configuration* applied to the frozen baseline, plus its
certified scores. There are two engines, used for two questions:

| Engine | File | A node is a… | Ranked by | Answers |
|---|---|---|---|---|
| **knob-merge beam** | `rvvgen/beam.py::run_beam` | knob dict (tile/vector/contraction-strategy/patterns) | lexicographic *toward the expert* (spike) | "which schedule knobs close the structural gap to the expert kernel" |
| **feature-set beam** | `rvvgen/autotune.py::beam_search` | `frozenset` of `impr_features` names | measured **K1 wall-time** | "which combination of mined features is fastest on real silicon" |

Both exist because the loop has two phases: the knob-merge beam drives the *structural* search
(match the expert's fingerprint on spike, where instruction-count is the proxy); the feature-set beam
drives the *performance* search (rank real silicon wall-time once features exist).

### 12.2 Input files — what the beam reads

- **Seed package** `out/artifacts/targets/rvv/hand_v0/` — the frozen baseline: `schedule.mlir`,
  `knobs.yaml`, `manifest.yaml`. Every search starts here and **never edits it**.
- **The feature registry** `llvmlower/impr_features.py` — the default-off features a node may enable.
- **The route table** `kernels/rvv_knobs.py::_ROUTES` — divergence-key → lever(s) (the deterministic
  proposer). Optional alternative: the LLM proposer `rvvgen/tuning_agent.py`.
- **The expert fingerprint** — `curated_text` (the expert kernel source/fingerprint) for the
  structural score, and the mined `expert_cca.yaml` from a `mining_rvv_v*` run for the divergences.
- **The model workload** `model_dir` (a captured model slice + `golden.npy`) — fixed across the search.

### 12.3 What stays fixed, what varies

**Fixed (the whole safety story):** the baseline `RVV_TRANSFORM_SCHEDULE` and `build_rvv_pipeline`
defaults (byte-identical guard, §4.3); `hand_v0`; the op-key `{op, dtype, shape_regime}`; the curated
expert fingerprint; the workload + golden; the cosine gate threshold.

**Varies (per node):** `op_match` (tile + vector size lists), `contraction_strategy`
(`outerproduct|dot|matmulintrinsics|parallelarith`), `lowering_patterns`, `dtype_strategy`, and the
`compiler_features` set. Each is expressed as an *override* layered on the parent's config — the
baseline is never mutated, only forked.

### 12.4 Scoring and pruning — exact code

Lexicographic, correctness-dominant (`rvvgen/sweep.py::rank_results`):

```python
def key(n):
    correct = 1 if n.get("gate_ok") else 0
    sm      = n.get("structural_match") or 0.0     # 0.6·decision_match + 0.4·histogram_cosine
    cyc     = n.get("cycles")
    return (correct, sm, -(cyc if cyc is not None else inf))
return sorted(scored, key=key, reverse=True)
```

Pruning (`rvvgen/beam.py`, per generation `d`): expand each parent into ≤`width` forks (default 3),
certify all in parallel, then `survivors = [n for n in rank_results(gen) if n["gate_ok"]][:top_k]`
(default `top_k=2`). **An incorrect node can never survive** (gate dominates), and the beam stops if a
generation yields no correct survivors. `structural_match` pulls survivors *toward the expert's
instruction fingerprint*, not toward a raw scalar — that is why the search converges on the expert's
shape rather than on whatever happens to retire fewest spike instructions.

### 12.5 How beams combine — the two merge mechanisms

1. **Implicit knob merge** (`rvvgen/from_strategy.py::mint_fork`): a child *deep-copies the parent's
   whole knob dict* and `knobs.update(overrides)` — so it inherits everything the parent had plus one
   new lever. A depth-2 winner is literally `seed → (widen N) → (widen N + outerproduct)`; beneficial
   knob **combinations** are discovered by depth traversal, no set algebra.

2. **Explicit feature-set union** (`rvvgen/autotune.py::beam_search`): a survivor with features
   `{A,B}` spawns `{A,B,C}, {A,B,D}, …` (one feature added per step), all re-benchmarked, top-`width`
   kept:

   ```python
   for d in range(1, depth):
       cands = {frozenset(p["features"]) | {f}
                for p in survivors for f in features if f not in p["features"]}
       rs = [evalc(c) for c in cands]
       survivors = sorted([r for r in survivors+rs if r["correct"]],
                          key=lambda r: r["min_wall"])[:width]
   ```

   This is exactly how the *neutral-alone, synergistic-together* pair was found: `fused_vfmacc` is
   ~neutral until `accumulator_resident` is also in the set; greedy extension surfaces the
   combination without enumerating all 2^N subsets. (It also surfaces the **composition guard**:
   two full-schedule-replacement features can't coexist — `apply_schedule` raises `CompositionError`,
   recorded as the node's `blocker`, see §12.7.)

### 12.6 Where the agent plugs in (and how it's contained)

The beam is deterministic by default; the proposer is the **one** pluggable seam:

- **Default — deterministic gap-router** (`kernels/rvv_knobs.py::propose_forks`): divergence key →
  fixed lever list. Same divergences ⇒ same proposals.
- **Optional — LLM proposer** (`rvvgen/tuning_agent.py::propose_forks_llm`): renders a versioned
  prompt from the divergences + knobs, calls `common.llm.complete` (its only tool), parses JSON
  proposals — **then every override is clamped against `_KNOWN_OVERRIDE_KEYS`; unknown keys are
  dropped and noted.** No LLM output reaches the compiler unvalidated, and both proposers emit the
  identical `ForkProposal` type, so the beam can't tell which produced a candidate. Everything
  downstream of the proposer (mint → build → certify → rank) is tooling.

(The Gemmini `targetgen/agent/` — Claude Code headless generating bare-metal C — is a *separate*
generative system, not part of the RVV beam; it shares only the "nothing trusted until it certifies"
gate philosophy.)

### 12.7 Outputs — the real artifacts

A run writes a versioned dir `out/artifacts/kernel-mining/rvv/beam_rvv_v{V}_{ts}/` containing a `beam_summary.yaml`
(per-model winner) and a `ranking_<model>.yaml` per model. **Real excerpt**
(`beam_rvv_v2_20260619T132435/ranking_bitvla.yaml`) — note a survivor *and* a killed node with its
honest blocker:

```yaml
model: bitvla_fp32_consistent
ranked:
  - tag: v3_plus_lmul
    features: [accumulator_resident_microkernel_v3, lmul_widen_n]
    speedup: 16.771
    lowering: vectorized
    cos: 0.9999945759773254
    min_wall_s: 0.1506
    status: pass
    blocker: null
  - tag: act_plus_matmul
    features: [vectorized_transcendental_activation, fused_vfmacc_tiled]
    speedup: null
    lowering: null
    cos: null
    min_wall_s: null
    status: not_run
    blocker: 'composition_error: cannot compose multiple full-schedule-replacement features
      [fused_vfmacc_tiled, vectorized_transcendental_activation]: each emits a complete
      transform schedule and would clobber the others.'
```

The knob-merge beam additionally writes `beam_tree.yaml` (`generations[]` with per-gen survivors,
`best`, `ranked`, and a `deferred_work_items[]` list for `forkable=False` branches — the explicit
"next compiler feature to build"). These files are the input to fig6 (the beam tree) and the results
dashboard. Each node's `package_dir` points at its `out/artifacts/targets/rvv/<run_id>/` fork
(schedule.mlir + knobs.yaml + manifest with `lineage.parent_run_id`), so a node is fully reproducible.

---

## 13. Before and after — a real improvement as a diff

The headline turn-1 feature is `fused_vfmacc_tiled` / `fused_vfmacc_scalable` (action class **PASS**,
seam `impr_features:fused_vfmacc_*`). It replaces a degenerate `vfmul`+`vfadd` matmul with a fused
`vfmacc`. Here is exactly what changes.

### 13.1 The schedule diff (what the fork edits)

The baseline (`pipeline.py:RVV_TRANSFORM_SCHEDULE`) tiles `[4, 8, 1]` — **K=1**, which never forms a
real contraction — and lowers with a bare `lower_contraction`:

```diff
  %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 ...
- %t, %l:3 = transform.structured.tile_using_for %mm tile_sizes [4, 8, 1] ...
- transform.structured.vectorize %t vector_sizes [4, 8, 1] : !transform.any_op
+ %t, %l:3 = transform.structured.tile_using_for %mm tile_sizes [4, 16, 16] ...   # KC=16: a real K reduction tile
+ transform.structured.vectorize %t vector_sizes [4, 16, 16] : !transform.any_op  # SCOPED to %t (no vectorize_children → no extract explosion)
  %f = transform.structured.match ops{["func.func"]} in %arg0 ...
+ transform.apply_patterns to %f {
+   transform.apply_patterns.vector.transfer_permutation_patterns
+   transform.apply_patterns.vector.reduction_to_contract        # rebuild a real vector.contract
+ } : !transform.any_op
  transform.apply_patterns to %f {
-   transform.apply_patterns.vector.lower_contraction            # no strategy → degenerate mul+add
+   transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
+ } : !transform.any_op
+ transform.apply_patterns to %f {
+   transform.apply_patterns.vector.lower_outerproduct           # outerproduct{add} → vector.fma
  } : !transform.any_op
```

This is expressed as a registered, default-off `ImprFeature` whose `edit_schedule` *replaces* the
schedule (`schedule_replace=True`); the baseline string is emitted unchanged when the feature is off.

### 13.2 Why `[4,8,1]` is the bug (the recorded root cause)

From the feature's own design comment (`impr_features.py`): `[4,8,1]` with **K=1** collapses the
inner body to a bare `mulf`+`addf` — no contraction can form, so `outerproduct`, K-tiling, and
fast-math were all *measured* no-ops on it. The fix tiles `KC=16` so a real reduction tile survives,
and `reduction_to_contract` rebuilds the `vector.contract` from the scoped-vectorize
`mulf`+`multi_reduction`. Two further constraints are baked in for whole-model safety: vectorize is
**scoped to the tiled handle** (vectorizing the whole func explodes to ~4120 `vector.extract` on a
toy model vs 84 scoped); and `KC=16` keeps tile vectors small (`vector<4x16>`, `vector<16x16>`) so the
register allocator doesn't spill — the unbounded earlier draft spilled 4 KB vectors and faulted the
spike at M≥64. The inner body is a **constant `MR·KC = 64` FMAs regardless of M/N/K**.

### 13.3 The asm diff (the emitted-instruction evidence, K4)

```diff
  # inner K step, baseline (contraction_form = mul_add):
- vfmul.vv  v8,  v0, v16        # t = a * b
- vfadd.vv  v24, v24, v8        # acc += t        ← two ops, extra vreg, extra latency

  # inner K step, fused_vfmacc_tiled (contraction_form = fused_fma):
+ vfmacc.vv v24, v0, v16        # acc += a * b    ← one fused MAC
```

`lift_asm` reports this as `compute.contraction_form: mul_add → fused_fma`; the K4 instruction
histogram confirms `vfmacc` is present and `vfmul`+`vfadd` are gone. The v3 line takes it one step
further — `accumulator_resident_microkernel_v3` scalarizes the A reads so the kernel emits `vfmacc.vf`
(decode-confirmed: the innermost K-loop is **4 `vfmacc.vf`, 0 `vfmacc.vv`, 0 in-loop accumulator
spills**), which is what beats the hand ceiling.

### 13.4 Measured effect (with its rung)

`fused_vfmacc_tiled`: **9.18× bitvla / 3.61× openvla** whole-model on the K1 board (wall-ns, the
real-silicon ground truth). The combined `v3 + lmul_widen_n` winner reaches **16.77× bitvla**
(`cos 0.99999`), beating both the XNNPACK kernel-swap (13.65×) and OpenBLAS. Every number is gated:
no cosine pass ⇒ no speedup credited (§14.3).

---

## 14. The improvement loop, and how we test for improvements

### 14.1 The loop, concretely

```
mining_rvv_v*  ──expert_cca, divergences──▶  action_catalog  ──forkable actions──▶  impr_features
        ▲                                                                                  │
        │ re-decode the residual                                                  proposer (gap-router │ LLM)
        │ from fresh asm                                                                   │
        │                                                                          mint_fork (knob merge)
   certified Δ ◀── certify_rvv (K-ladder, cos-gate) ◀── apply_rvv_package → build_app → model.o ◀──┘
```

Each turn: lift the gap → route to a typed action → enable it as a default-off feature → mint a fork
→ build → certify → measure → **re-decode the residual and repeat**. The loop has twice overruled a
*human* hypothesis with a *measured* refutation (lane-width, then packing): turn-3's `MemFacet`
measured that the `.vf` kernel already ties XNNPACK at 2.0 loads/FMA, 0 ladder ops — so the "packing"
lever was retired as structurally inapplicable to small-M VLA matmuls rather than pursued.

### 14.2 How a fork is applied (the seam)

The fork's `compiler_features` thread through one entry point — `rvvgen/apply.py::apply_rvv_package`:

```python
return zm.build_app(model_dir, work, board=board, backend="rvv",
                    rvv_schedule=pkg.schedule_text,
                    cflags_override=pkg.cflags + zm._CFLAGS_COMMON,
                    features=frozenset(pkg.compiler_features) or None,   # ← threaded here
                    ...)
```

…which calls `apply_schedule(schedule, features)` and `build_rvv_pipeline(sched, features=features)`.
With `features=∅` both are identity (§4.3) — the baseline emits the exact shipping pipeline; a feature
changes codegen **only** when a fork enables it. No global state is mutated; `pipeline.py` defaults are
never touched.

### 14.3 How we test — the K-ladder and the cosine gate

`runner.certify_rvv` runs every candidate through a fail-closed ladder; the beam consumes its
structured result:

| Rung | Check | Gate? |
|---|---|---|
| K0 | load + integrity (cflags allowlist, manifest schema) | yes |
| K1 | non-perturbation: seed schedule == `pipeline.RVV_TRANSFORM_SCHEDULE` | yes |
| K2 | build via `apply_rvv_package` → `model.o` + `zephyr.elf` | yes |
| **K3** | **spike correctness — multi-tier cosine gate** | **yes (the gate)** |
| K4 | instruction histogram (evidence: vfmacc present? — not a hard gate) | no |
| K5 | K1 silicon cycles (`rdtime`→cycles + `wall_ns`); `not_run` if board unreachable | no |
| K6 | Δ vs baseline — **speedup credited only if K3 passed** | yes |

The gate itself is multi-tier (`runtime/backends/zephyr_model.py::_gate`): a W8A8 tier (`cos > 0.999
AND rel < 1e-2`), an fp32 tier (`cos > 0.99 AND top-1 argmax match`), and a legacy fp32 tier
(`cos > 0.9999 AND rel < 1e-3`); `ok = any tier passes`. And K6 is fail-closed:

```python
speedup = (baseline_cycles / m["cycles"]) if gate_ok else None   # wrong ⇒ no number, ever
```

So a candidate that is fast but wrong reports `speedup = None`. This is the structural reason the
headline numbers are defensible: nothing that fails the cosine gate can produce a speedup claim, and
the measurement rung (`spike` / `K1` / `not_run`) is always attached. The byte-identical guard
(`test_impr_features.py`) proves the baseline cannot move underneath any of this.

### 14.4 The measurement ladder (and what each rung can and can't see)

Same ladder as §2, restated as the test discipline: **spike** measures `minstret` (IPC=1 proxy, no
memory hierarchy) — so it ranks *structure* (instruction count) but is blind to a loads/FMA win;
**K1** measures `rdtime`→wall-ns on real silicon (cycles derived, so wall-ns is ground truth) — the
rung the speedups are reported on; **FireSim/VCS** is cycle-accurate but queue-gated and never
escalated. A result unreachable on a rung is `not_run`, never fabricated. This is why, e.g., the
turn-2 `.vf` ladder-elimination shows as a modest whole-model bump on K1 (1.04–1.09×) even though it
removes 8→0 broadcast ops/FMA: the win is in data movement, which only the silicon rung sees.
