# Kernel-mining findings & methodology

A consolidated, auditable synthesis of what we learned by analyzing **2,411 kernels across 6
independent sources**, the tools that make those findings trustworthy *without reading every
kernel by hand*, the metrics we measured on real RTL, and how the insights feed forward into
the compiler. Every number here is reproducible from `output/kernels/` artifacts.

---

## 1. The method: infer from a few, scale to all, never read every file

The core problem you flagged: an agent must **not** read thousands of kernels one by one. Our
loop avoids that entirely:

```
  sample a few kernels  →  hypothesize a DECISION marker (regex/signature/AST)
        →  adapt the scaffold (markers.py / features/)  →  run deterministically over ALL kernels
        →  aggregate by cross-source frequency  →  audit a random sample to confirm precision
        →  promote only what generalizes  →  measure the survivors on RTL  →  feed the compiler
```

Per-kernel extraction is **deterministic and LLM-free** — regex over markers keyed by ISA
family, filename/signature parsing, Exo-lowered C. That is what makes it scale to 2,411 kernels
in seconds, reproducibly, and is why the agent reads ~5 sampled snippets per motif (via
`kernel-audit`) instead of 2,411 files. **Insight comes from frequency across independent
toolchains, not from deep-reading any single kernel.**

We extract **decisions, not constants**: "packed RHS reused", "accumulator lives across the
epilogue", "vector-length-agnostic loop" — never `tile=64` or `LMUL=4`.

---

## 2. What the kernels told us

**Corpus** (after content-hash dedup of 100 kernels triton-cpu vendored verbatim from triton):

| source | kernels | character |
|---|---:|---|
| autocomp (Gemmini) | 910 | matmul×564, conv×346 — LLM-generated variants |
| exo | 696 | compiled-to-C + schedule `.py` mining |
| xnnpack (RVV) | 545 | production microkernels |
| openblas (RVV) | 130 | hand/generated BLAS kernels |
| triton_cpu | 117 | tutorial + kernel-lib |
| triton | 13 | tutorial kernels |

**Motif frequency** (the abstraction signal — breadth of sources matters more than raw count):

| motif | kernels | sources | verdict |
|---|---:|---:|---|
| intrinsic_lowering | 1795 | 6 | structural (too ubiquitous to be a policy) |
| accumulator_lifetime | 1352 | 6 | structural |
| tiling_blocking | 1331 | 6 | structural |
| **packed_rhs** | **1089** | **6** | ✅ policy → `resident_packed_tensor` |
| weight_stationary_dataflow | 896 | 2 | ✅ policy (Gemmini-led → target contract) |
| epilogue_before_commit | 700 | 6 | structural |
| vector_length_polymorphic | 620 | 2 | ✅ policy (parked after measurement) |
| many_small_dispatches | 498 | 1 | ✅ runtime candidate |
| **accumulator_commit** | **436** | **4** | ✅ policy → `accumulator_commit` |
| double_buffering | 22 | 2 | ✅ policy (parked pending cycle sim) |

**Headline finding:** a *small* set of recurring decisions explains a large fraction of
high-performance kernels, and the strongest ones recur across **all six** independent
toolchains. `packed_rhs` (keep the weight resident, reuse it) appears in 1,089 kernels from RVV
microkernels to Gemmini to Triton — it is not a source-specific trick. That cross-source
recurrence is the evidence that it deserves to be a compiler-visible abstraction.

**Promoted artifacts** (all schema-valid, in `output/kernels/`):
- 3 abstraction candidates — `resident_packed_tensor`, `accumulator_commit`, `async_pipeline`
- 5 policy rules, 3 interface candidates (4 lowering variants each), 1 runtime candidate
- 3 L6 dialect requirements (input to TargetGen) + 3 L8 LLVM requirements (all `fork: false`)

---

## 3. The reliability stack — why these findings are trustworthy

Each tool answers a specific "do I believe this?" question so trust is mechanical, not a matter
of inspecting code by hand:

| Tool | Question it answers | How |
|---|---|---|
| **cross-source frequency** | source-specific fluke or real abstraction? | a motif in ≥2 independent toolchains can't be one tool's trick |
| **content-hash dedup** | are counts inflated by copies? | verbatim cross-source duplicates dropped before counting |
| **`kernel-audit`** | does a marker mean what we claim? | samples N kernels/motif (seeded), re-fires the marker, shows the real ±context snippet — the agent eyeballs ~5, not 2,411 |
| **consistency invariants** | are the artifacts internally coherent? | subset rules (`reused_packed_rhs ⊆ packed_rhs`), evidence-ids exist, counts equal a recount, surprise list (motifs on wrong op families). 0 violations on the full corpus |
| **7 evaluation plots** | does the picture make sense? | motif×source heatmap (normalized), prevalence, promotion funnel, reuse/dispatch distributions, co-occurrence, motif×op sanity |
| **negative controls** | does a policy over-fire? | `no_reuse_matmul` / mutable-RHS must stay silent — and they do, symbolically *and* in the cost model |
| **shape-regime matrix** | does it generalize beyond seen shapes? | symbolic sweep R×K×tail; `packed_rhs_policy` fires only at reuse≥2, silent on both controls |
| **schema validation** | is every artifact well-formed? | every emitted YAML validated against `merlin/schemas/*` (0 problems) |
| **`--json` on every CLI** | can agents/CI compose this? | machine-readable summary on stdout; `--strict` gates CI on invariant violations |
| **mandatory caveats** | are we overclaiming? | report states: no kernel executed in mining; plots are evidence-frequency not speedup; Autocomp `score` never trusted |

The promotion ladder makes "skip the unimportant ones" explicit:
`Observation → Motif (≥2 sources) → Policy candidate → Validated (fires on positive, silent on
control) → Measured (RTL)`. Anything that doesn't clear a rung is parked or skipped with the
reason recorded — not silently dropped.

---

## 4. What we measured on real hardware (Stage-F)

We don't stop at "this decision recurs." We measured the four promotable insights on the
**cycle-exact Gemmini RTL** (Spike functional events + Verilator cycles):

| insight | measured result | decision |
|---|---|---|
| **resident_packed_tensor** | RHS traffic ↓ exactly R× (16× at R=16), exploitability **1.00** | ✅ ACT |
| **accumulator_commit** | commit bytes ↓ **4.0×** (i32 round-trip → fused i8), 26,917 CPU instrs of epilogue eliminated | ✅ ACT |
| **command_buffer_batching** | config+fence = **85.2%** of commands at 39 tiles; batching removes **54%** — matches the mined 0.849 small-dispatch fraction | ✅ ACT (runtime first) |
| **vl_agnostic_loop** | tail overhead ≈ **0** at VLEN=128 | 🅿️ PARK — portability only, no perf win |

**The triage worked:** mining ranked `vl_agnostic_loop` highly (620 kernels), but measurement
parked it — *frequency ≠ importance*, which is the entire reason the ladder exists. And
`dispatch_batching`'s measured 85% overhead matched the mined 0.849 fraction almost exactly:
strong evidence the markers measure something real.

### The instruction cost model (the keystone)

`merlin/python/merlin/cost_model/` predicts cycles from event counts without per-candidate RTL:

- coefficients (cycles/command), calibrated against Verilator by relative-weighted least
  squares: `config≈5, mvin≈29, mvin2≈30, compute≈64, mvout≈31, fence≈33`
- **validation MAPE 8.1%** on held-out Stage-F kernels; both decisions preserved
- **it caught an over-claim:** the L2 event count said `resident_packed_tensor` cuts RHS
  traffic 16×, but the cost model shows the end-to-end *cycle* win is only ~1.2× at a single
  16×16 tile (systolic compute dominates) — still ACT (22% margin > 8% band), larger in
  memory-bound regimes. At R=1 the predicted speedup falls *within the band* → correctly no
  action, independently reproducing the negative control.

This is why cycle-grade ranking beats raw counts, and it's the shared currency for everything
downstream.

---

## 5. How the insights advance the compiler

The findings are not a report that ends here — each has a named downstream consumer and a
gate it must clear (the actionability scorecard in the mining report carries all of this):

```
  mined motif ──▶ policy rule (when/action over compiler-visible facts)
              ──▶ interface candidate (compiler_must_prove / hw_must_provide / 4 lowering variants)
              ──▶ L6 dialect_requirement ──▶ TargetGen scaffolds the target dialect op/type
              ──▶ L8 llvm_requirement (fork:false until Stages F/G pass)
  cost model ──▶ ranks the full regime grid in seconds; the reward Autocomp optimizes;
                 the feature axis a learned heuristic ranks transformations over
```

Concrete next moves, in dependency order:
1. **Stage F lowering** — hand the 3 ACT verdicts to TargetGen via `dialect_requirements.yaml`;
   `resident_pack` / `accumulator`+`commit` ops on `toy_npu`, scored by the cost model.
2. **Autocomp as a labeled-trajectory engine** — log its accept/reject decisions, re-cost each
   in our currency. This supplies the *negative* examples mining structurally can't (it only
   sees winners), which is exactly what learning the *right* heuristic requires.
3. **Heuristic learner** — fit `(program features, shape regime, target capability) → ranked
   transformations + predicted Δcost` from those trajectories, gated by the same cross-source
   generalization test so we don't overfit to Gemmini.
4. **Fusion** — the showcase trade-off (fused epilogue saves the 4×-measured commit round-trip
   but costs scratchpad residency that can evict the resident RHS); the cost+capacity model is
   what lets the search find the right frontier instead of guessing.

---

## 6. Where everything lives

| | path |
|---|---|
| Pipeline | `merlin/python/merlin/kernels/` (ingest, markers, features, classify, policy, validate, report, plots, invariants, audit) |
| Tools | `kernel-index`, `kernel-extract`, `kernel-audit` (`[project.scripts]`) |
| Cost model | `merlin/python/merlin/cost_model/` + `gemmini_cost_coeffs.json` |
| Stage-F harnesses | `merlin/experiments/kernel_policy/stageF/` |
| Profiling plan | `merlin/experiments/kernel_policy/profiling_slate.yaml` |
| Generated artifacts | `output/kernels/` (report, plots, audit, features, *_candidates.yaml, stageF/, cost_model/) — gitignored, reproducible |
| Schemas | `merlin/schemas/` (kernel_record, policy_rule, interface/dialect/llvm requirements, …) |

**Caveats that bound every claim:** no kernel was executed during *mining* (decisions, not
speedups); Stage-F numbers are from a functional+RTL Gemmini model (DIM=16), simulator-relative
until FireSim/HW confirms; plots visualize evidence frequency, never speedup; the Autocomp
`score` is never treated as a correctness or performance signal.
