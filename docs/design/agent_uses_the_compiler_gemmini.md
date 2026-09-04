---
title: "Design note: teaching a compiler to fit real models, and measuring an agent against a code writer"
kind: design
status: current
owner: core
last_verified: 2026-09-03
related: [target_publishing, performance_levers_per_archetype, incremental_target_evolution_opu]
code_refs: [merlin/experiments/agent_recipe_select_v0/scripts/census_workloads.py, merlin/experiments/agent_recipe_select_v0/scripts/run_census_campaign.py, merlin/experiments/agent_recipe_select_v0/scripts/compare_arms.py, merlin/experiments/agent_recipe_select_v0/scripts/agent_compile.py, merlin/experiments/agent_recipe_select_v0/compiler/recipe_surface.patch, merlin/contract/claim_models.yaml]
---

# Teaching a compiler to fit real models, and measuring an agent against a code writer

This note is the running account of one experiment: give an LLM a *frozen, RTL-certified* Gemmini
backend and a small set of compiler knobs, point it at the kernels ResNet-50 and TinyLlama actually
need, and compare it against an agent that writes Gemmini C by hand (AutoComp). It is written as a
narrative because most of what it has produced so far is not the headline number — it is a sequence
of measurements that contradicted what we believed when we started, and each contradiction changed
the instrument.

Point-in-time results live in `out/artifacts/recipe-select/`; this note keeps the reasoning.

## The setup, and the two limits it inherited

The v0 round ran two arms on three synthetic GEMM shapes over a frozen backend
(`gemmini_xdsl_rtl_v0`, fingerprint `ad10fc3d…`, certified on five capsules under Verilator at
302/269/525/385/3863 cycles). The recipe arm won 2 of 3 workloads and produced **0 invalid
candidates out of 48** against AutoComp's 12 numerically-wrong out of 66, and the deliverable was a
compiler default flip that generalised on held-out shapes.

It also measured two limits that shaped everything here:

1. On the N-heavy shape the frozen compiler was **2.68× worse than hand-written C**, and the recipe
   space could not close it. A knob set that cannot express the fix is not a search problem.
2. An exact combinatorial control showed the agent was **statistically indistinguishable from random
   search**. On a 20-point dense space that is the honest verdict, and it is also a statement about
   the space rather than the agent.

Both limits point the same way: the interesting question is not "can an LLM search 20 points" but
"what happens when the space is large and the shapes are real".

## The census: what the models actually need

Both captures were already on disk with per-op provenance, and ResNet-50's convolutions are already
im2col'd to `linalg.matmul` in the capture (`prov.conv_path = "im2col_matmul"`). So decomposing the
models into kernels was a structural read, not a lowering exercise.

| | contractions | distinct shapes | MACs |
|---|---|---|---|
| ResNet-50 int8 (224²) | 54 `linalg.matmul` | 21 | 4.087 G |
| TinyLlama int8 W8A8 (seq 8) | 155 2-D | 5 | 8.275 G |

TinyLlama's 45 batched attention contractions are **f32** in the capture, so they are outside the
int8 device path regardless. That is a coverage line to report honestly, not a gap to close here.

## The first thing that was wrong: 10.4% was actually 0%

An early estimate said ~10% of ResNet-50's MACs fit the frozen lowering. That estimate checked only
the scratchpad relation. There are **two** capacity bounds and both must hold for a single block:

> operand store `Kt·(Mt + Nt) ≤ spad_rows/dim`  **and**  accumulator `Mt·Nt ≤ acc_rows/dim`

With `DIM=16`, `SPAD_ROWS=16384`, `ACC_ROWS=1024`, the accumulator bound is `Mt·Nt ≤ 64` — and it
binds first, and binds *everywhere*. The corrected figure is that **0 of 26 distinct shapes fit, and
0.0% of either model's MACs were expressible.** Not a small number: zero.

This is worth dwelling on because the census's own legality column reported 54/54 legal. Its scope is
`op_name + element_types` — *is this op and dtype routable to the mesh at all* — which is a different
question from *can the lowering emit it*. Both columns were true. Only the second is a statement
about the compiler, and reading the first as if it were the second is what produced the 10.4%.

## The second thing that was wrong: it was not producing wrong answers

The plan asserted that past either bound the compiler emits **silent wrong answers**. That claim was
load-bearing — it is the difference between "a correctness bug" and "a missing feature" — so it got
measured, and measurement refuted it:

* `PR06_spills_k8208` (16×16×8208, a 32-row A/B overlap) **passes**, at 28118 cycles. At `Nt=1` each
  tile is written and consumed in the same iteration, so the aliased partners are never live
  together. Accidentally safe.
* 16×512×528 computes a **negative** weight base (−512) and **does not halt** — stopped after 13
  minutes having written 557 MB to the console with no `METRIC`/`DONE`. The same shape under blocking
  is bit-exact at 93366 cycles, cut into two K-blocks.

So the defect was never "wrong answers". It was that **nothing distinguished the two cases**: no
capacity predicate existed on the matmul path at all, so one shape was quietly fine and its neighbour
diverged, with no refusal in between. That is a more precise and more useful bug report, and it only
exists because the original claim was checked instead of repeated.

## The capability fix: blocking

The fix is a real M/N/K blocked schedule in the fork's `_matmul_trace`, with:

* a **derived default** — given `(M,N,K)` and the RTL-derived geometry, exhaustively choose the
  largest legal `(bm,bn,bk)`, ties going to the larger N;
* **agent-selectable overrides** (`block_m`, `block_n`, `block_k`) whose legality is decided
  *jointly* by the compiler, so `choices` refuses illegal picks rather than emitting colliding code;
* K-blocking that preserves accumulation across blocks, with the epilogue applied on the final K
  block only — the host-side path at `compile_cli.py` had already recorded that exact trap.

Result: **26/26 shapes emittable, 100% of both models' MACs.** The equivalence gate holds — with
blocking off, 222 capsules are byte-identical, 0 newly refused, and the 11 that differ are exactly
the shapes past the single-block bound.

### A catalog bug worth naming

The first catalog implementation filtered each dimension's legal values against *the other
dimensions' defaults*. That is wrong in a way that is easy to miss: `block_m=64` is illegal beside
the default `block_n` but perfectly legal beside `block_n=256`, so per-dimension filtering amputated
valid combinations and reported `n_legal == n_total` while doing it. The fix is to offer every value
and let the joint predicate decide — the catalog describes the axis, the compiler decides the point.

### Is blocking also a performance lever where it is not required?

Tempting to assume yes, and the assumption was wrong. Measured on GSIM at 32×32×32, a shape that
fits in a single block and therefore needs no cutting at all:

| recipe | cycles | correct |
|---|---|---|
| frozen default (`per_tile` / `per_mvin`) | 780 | yes |
| `panel` + `on_change` | **721** | yes |
| `panel` + `on_change` + `block_k=16` | 749 | yes |
| `panel` + `on_change` + `block_m=16` | 812 | yes |

So on a shape that does not need it, **cutting costs rather than pays**: `block_k=16` is 3.9% slower
than leaving the block whole, and `block_m=16` is 12.6% slower — one M tile per block makes every
compute `COMPUTE_PRELOADED`. All four are bit-exact, so this is an economic result, not a
correctness one. The four points were measured twice, independently, and replicated exactly.

An earlier session-note claimed `block_k=16` reached 677 cycles here and beat the residency-only
optimum. Re-measurement says otherwise (749 vs 721), and the earlier figure had no persisted
artifact behind it. Blocking is a **capability** fix — it is what makes 26/26 shapes emittable at all
— and it should not be sold as a free speed knob on shapes that already fit.

## Two oracles, one job each, never averaged

| oracle | used for | why |
|---|---|---|
| **GSIM** (`GemminiGsimSerialClkConfig`) | every search candidate | ~70 s + 0.006 s/cycle, parallelisable |
| **FireSim** (`alveo_u250_firesim_shuttle_gemmini_opu`) | the whole-model verdict | the only rung that runs a whole model on our RTL; one shared FPGA |

These cycles are **not interchangeable** and never share an axis: the same capsule reads **510 cycles
on FireSim and 317 under Verilator** — same accelerator, different host core. `oracle_engine` is
therefore a mandatory column on every row carrying a cycle count.

## Why the two arms cannot be ranked by their own speedups

This is the single most important reporting decision in the experiment, and it is easy to get wrong
in a way that flatters us.

The arms do not share a baseline. The **recipe arm** starts from the frozen compiler's default recipe
for that shape. The **AutoComp arm** starts from a hand-written Gemmini C seed kernel. On real
ResNet-50 layers that seed starts **1.5–2.1× ahead of the compiler default**. So "1.79× over my own
baseline" and "1.00× over mine" are answers to different questions, and ranking by them would report
a win that absolute cycles do not support.

The only comparable column is **absolute cycles on the same oracle**. `compare_arms.py` computes the
head-to-head from cycles and reports each arm's own improvement factor *beside* it, never instead of
it.

## What the numbers say so far

Partial, and labelled as such. Recipe arm: 11 of 26 census shapes complete, **0 invalid candidates
out of 176**. AutoComp arm: 0 of 26 complete. On the three shapes where both arms have data:

| shape | recipe base | recipe best | ×own | ac base | ac best | ×own | recipe vs ac |
|---|---|---|---|---|---|---|---|
| `conv1` 48×176×147 | 14431 | 8070 | 1.79× | 9585 | 9585 | 1.00× | 1.19× faster |
| `fc` 1×112×2048 | 52606 | 29960 | 1.76× | 24878 | 24878 | 1.00× | 1.20× slower |
| `layer1.0.conv1` 64×192×64 | 9681 | 6701 | 1.45× | 6289 | 6289 | 1.00× | 1.06× slower |

Geomean 0.975× — AutoComp marginally ahead on absolute cycles, on three shapes, from a better start.

**AutoComp's 1.00× is a floor, not a verdict**, and citing it as a verdict would be dishonest:
across those shapes it generated 32 candidates of which **0 were correct**. Grouping those failures
by message — rather than counting them — shows they are almost all one harness defect, not model
incapacity. See the next section.

### 27 of 29 compile failures were one missing sentence in the prompt

The rule this repo keeps relearning is that **a compile-failure rate must be grouped by message
before it is cited**: a genuinely weak model fails in varied ways, while identical failures are a
harness signature. Grouped:

| class | n |
|---|---|
| implicit declaration / undefined reference to `mvin`, `preload`, `config_ld`, `compute_preloaded`, `mvout`, `config_st`, `config_ex`, `mvin2` | **27** |
| the harness's own `uint64_t` timing bracket | 2 |

Every symbol in that first row is a **real Gemmini macro with the `gemmini_` prefix stripped** —
`gemmini_extended_mvin`, `gemmini_preload`, `gemmini_config_ld`, and so on, all present in
`gemmini.h`. The model had the ISA essentially right and only the naming convention wrong.

Why it had the convention wrong is the defect: `get_backend_specific_rules()` states the shapes, the
bit-exactness requirement, the no-FPU constraint and the timing-bracket contract — but **never names
a single available primitive**, and the seed kernel it is asked to beat uses only the high-level
`tiled_matmul_auto` and `gemmini_fence`. So the arm was asked to out-optimise a library call by
dropping to an ISA whose spelling it was never shown, and then scored on the result.

That is not a measurement of AutoComp. It is a measurement of our prompt. The two `uint64_t` failures
are also a repeat of a defect already recorded as fixed, which is its own lesson about verifying that
a harness fix actually reached the running configuration.

## The failure that would have been invisible

The AutoComp arm tiers two models: planning on `gpt-5.6-sol` at high effort, implementation on
`gpt-5.3-codex-spark` at low. Spark hit its own usage limit. codex reports that on **stdout**, as:

```json
{"type":"error","message":"You've hit your usage limit for GPT-5.3-Codex-Spark. ..."}
```

Our parser reads only `item.completed` and `turn.completed`, so that line was dropped on the floor.
The sample was recorded as an empty string with **0 tokens and no reason**, and AutoComp then spent a
beam slot on an empty candidate. **24 of 56 code-tier calls (43%) failed this way** and appeared in
the ledger as ordinary zero-token calls.

This is the repo's recurring `checks-that-skip-and-report-success` shape: a step that could not run
reported nothing rather than a failure. The lesson is specific and reusable — *when a subprocess is
the boundary, discarding stderr and ignoring unrecognised message types converts an outage into a
plausible-looking measurement.* Fixes required before that arm is rerun: parse `type: error` and
`turn.failed`, record the reason and rc on the usage row, and fail loudly rather than emitting empty
candidates.

A related earlier error is worth recording because it removed an entire experimental axis: the API
answers *any* unknown model slug with "The 'x' model is not supported when using Codex with a ChatGPT
account". Reading that as an entitlement statement led to the conclusion that the seat served only
one model. It serves seven; the probed slugs were simply misspelled. Confirmed by sending an invented
control name and getting the identical message.

## Isolation, because a programme is not one run

v0 used a single shared mutable fork directory. That is fine for one run and wrong for a programme: a
compiler edit made while a run is in flight silently changes what that run is measuring, and two runs
cannot proceed in parallel. Every run now mints its own content-addressed package (0444, own
`SHA256SUMS`, a `lineage:` block naming the parent fingerprint, no `publication.champion`), snapshots
the scripts that produced it, and re-verifies the digest before the first candidate and after the
last. `package_fingerprint` on every row is the join key back to the exact compiler.

The fork itself lives in a gitignored package directory, so the compiler change is *also* mirrored as
a tracked patch (`compiler/recipe_surface.patch`) that reconstructs it from the champion. A codegen
package is tool output; this diff is the deliverable, and a deliverable has to be reviewable.

## The claim-model rule

`merlin/contract/claim_models.yaml` declares both models as **claim models**: they may be compiled
and graded, and their census may never enter requirement derivation, corpus synthesis, or capsule
selection. The recipe space and the derived blocking rule are frozen into a minted package whose
digest **predates** the census artifact. That ordering is what makes this a non-circular
generalisation test, and it is checkable by inspection rather than by assertion.

## Open

* 15 of 26 recipe shapes and all 26 AutoComp shapes remain (the driver takes `--resume`, so finishing
  costs only what is missing).
* `merlin_iface.bias_add` is ingested and then **silently ignored** — `_config_st` reads only
  `acc_scale`, `relu`, `maxpool` — so folded BatchNorm bias vanishes. This gates a numerically
  correct end-to-end ResNet-50.
* The host lane (softmax, RMSNorm, SiLU, RoPE, residual add, avgpool, embedding) is in scope and not
  yet built; the uncovered fraction is the Amdahl bound on any device speedup and belongs on every
  whole-model figure.
* `heavy_oracles.firesim_adapter` still returns `OracleUnavailable` by design; wiring it keeps the
  fail-closed behaviour — no uartlog `METRIC cycles`, no row.
