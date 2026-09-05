---
title: "Design note: the perf corpus cannot express the optimizations that matter"
kind: design
status: current
owner: core
last_verified: 2026-09-04
related: [performance_levers_per_archetype, performance_budget_unit, expert_gap_attribution]
code_refs: [merlin/contract/capsules/profiles/_perf.yaml, merlin/contract/capsules/generate_corpus.py, merlin/python/merlin/targetgen/corpus_spec.py, merlin/python/merlin/runtime/reference.py, merlin/python/merlin/perf/work_volume.py, merlin/python/merlin/targetgen/capsule_golden.py, merlin/python/merlin/targetgen/tier_policy.py]
---

# The perf corpus cannot express the optimizations that matter

## The symptom

The agentic performance campaign converges at roughly **0.2%**. The agent is not the cause: across
three trials it improved 9–14 members each with **zero regressions**. The objective it optimises is
Amdahl-dominated.

Measured on the campaign's own feedback documents (26 priced members, corpus total 171,739 cycles):

| slice | cycles | share of total | position |
|---|---|---|---|
| 7 deep-K `PR` members | 158,811 | **92.4%** | already at 0.59–0.94 of achievable |
| 18 members below 50% of achievable | 12,318 | **7.2%** | perfecting all of them gains **≤4.64%** |
| `PR07_spills_k12288` alone | 53,024 | 30.9% | 21,500 recoverable = **12.5% of the whole corpus** |

The agent improved `PL`, `PM`, `PC` and `PK` members only. It has never improved a `PR` member,
because `PR`'s declared lever is `operand_residency` — spill and refetch behaviour — and the levers
the agent reaches (tiling, hoisting, barrier placement) do not touch it.

So the search is well-behaved and is spending itself on 7% of its own objective.

## The cause is corpus scope, not corpus size

Every one of the 45 members of `merlin/contract/capsules/_perf/` is a single contraction:

```
29  op: matmul            one contraction
16  op: resident_reuse    1-4 matmuls sharing one pushed weight
```

All **150 of 150 operands are rank-2**. Coverage of the declared optimization ladder:

| rung | members |
|---|---|
| `L1_tile` | 20 |
| `L2_intra_layer` | 23 |
| `L3_inter_layer` | 2 (`PC00_k64`, `PC01_k128`) |
| `L4_boundary` | **0** |
| `L5_fusion` | **0** |
| `L6_global` | **0** |

"Large" in this corpus means large in K. `PR08_spills_k16384` carries a 16384×16 weight, but it is
still *one matmul*. The corpus varies depth, never scope. A compiler that fuses an entire attention
block scores identically to one that does not, because no member can observe the difference.

### Three real bugs, and where they would have been caught

Each of these was found outside this loop. Mapped against the corpus:

| bug | rung | corpus coverage |
|---|---|---|
| conv2d reuses stale scratchpad rows, worst under zero padding | L2 intra-layer | **none.** All four conv capsules in the tree declare `padding=[0,0,0,0] stride=[1,1] dilation=[1,1] ci=4 kh/kw=3/3` — identical geometry, four times. `build_conv2d` accepts all three parameters (`corpus_spec.py:1039-1041`); nothing has ever passed a non-default. |
| a QK kernel issued 4,098 fences where 2 suffice | L4 boundary + CPU↔accel | **1/256 of the scale.** `PQ` *is* the synchronization family and is well built, but its largest member is `PQ04_j16_k16` — 16 barriers over 16×16 tiles — and its own declaration says so: *"the members are deliberately SHALLOW … four jobs is enough to separate a per-barrier saving from a constant one."* Never on an attention shape. |
| `ranks: [2,4]` rejected every rank-3 batched matmul before it reached the device rewrite | eligibility, upstream of all timing | **none, and still none.** Fixed to `[2,3,4]` in `merlin/python/merlin/_data/targets/gemmini/contracts/target_contract.yaml:122`, whose comment records *"gemmini shipped none because of this line."* There are zero rank-3 capsules anywhere in the gemmini corpus, functional or perf, so the fix has no regression test and can silently revert. |

## Why nothing model-shaped can be admitted today

The perf corpus is **not a benchmark suite. It is a set of falsifiable claims.** Every entry in
`merlin/contract/capsules/profiles/_perf.yaml` is a `sweep` carrying `fit_axes`,
`comparison_roles`, a `claim ∈ {RECOVERS, PREDICTS, DIFFERENTIAL}`, a `comparand` with `cancels` and
`demand_equal`, a `falsifier` with a `negative_control`, a `gate` with a `capacity` rule, and an
`acceptance` block naming an analyzer. `PQ` alone cancels eleven quantities so that *only* the
barrier count differs between arms.

Admissibility therefore means: **one axis varies, everything else is held identical.** That is what
makes a result citable rather than anecdotal, and it is precisely why a model cannot join a family —
you cannot hold M, N, K, dtype and epilogue fixed across `M2_microvit`. A model cancels nothing.

Family membership is mandatory at four independent layers, so this is not a convention that can be
sidestepped:

1. `_perf.yaml` cannot hand-author capsules — it must generate through `sweeps`
   (`generate_corpus.py:234-237`).
2. A target profile cannot declare a perf entry; `_target_local_perf_declarations`
   (`generate_corpus.py:202`) raises and says to move it to the shared template.
3. `expand_sweeps` runs the full `_validate_performance_block` on every `_perf` sweep
   (`generate_corpus.py:2239`), so a category-`_perf` sweep without a `performance` block raises.
4. The consumer refuses an unfamilied member: `label: dev`, `source_role: derived_sweep`, a
   non-empty `performance.family` and a valid `claim`, else `StageGateError`
   (`perf_agent_stage.py:1217-1227`).

And a stale corpus is a hard campaign refusal: the directory set on disk must equal the `generated`
list in `MANIFEST.yaml` exactly (`perf_agent_stage.py:1233`), with
`performance_generation.<target>.errors == []` (`:1178`). **Hand-dropping a directory into `_perf/`
will not work.**

## The generator-side inventory — what it would actually take

`corpus_spec.BUILDERS` (`corpus_spec.py:1089`) holds 11 builders. `_perf.yaml` names **three**:
`matmul`, `resident_reuse`, `conv2d`. The others are reachable in principle —
`_materialize_performance_entry` gates only on `op in CS.BUILDERS` (`generate_corpus.py:2149`) — so
the question is what breaks downstream. The four gates a member must clear:

- **the golden** — `capsule_golden.golden`, op branches at `capsule_golden.py:588-676`;
- **the reference engine** — `MODELED_OPCODES` in `runtime/reference.py:22`, an unmodeled opcode
  raises `UnmodeledOp` rather than silently dropping a store;
- **work counting** — `perf/work_volume.py:109`; an opcode with no rule makes the whole program
  `is_lower_bound` (`:191`), which drops the member from the achievable-ceiling harvest;
- **declared pricing** — `perf_agent_stage.declared_capsule_macs:2229`, whose `_WORK_OPERATIONS` is
  `("matmul", "resident_reuse")` and which requires rank-2 shapes (`:2253`).

Measured state of each builder:

| builder | golden | reference engine | work_volume | declared_macs | verdict |
|---|---|---|---|---|---|
| `matmul` / `linear` | ✅ | ✅ `MATMUL` | ✅ | ✅ | in use |
| `resident_reuse` | ✅ | ✅ `MATMUL_RESIDENT` | ✅ | ✅ | in use |
| `conv2d` | ✅ | ✅ `CONV2D` | ✅ | ❌ | **one gap: pricing** |
| `attention_qk` | ✅ | ✅ `ATTENTION_QK` | ✅ | ❌ | **one gap: pricing** |
| `movement` | ✅ | ✅ `MOVEMENT` | ✅ (non-compute) | ❌ | one gap: pricing |
| `gemv_batched` | ✅ | ❌ `BATCHED_MATMUL` | ❌ | ❌ (rank-3) | 3 gaps |
| `rmsnorm`, `rmsnorm_qkv` | ✅ | ❌ `RMSNORM` | ❌ | ❌ | 3 gaps |
| `rope_qkv` | ✅ | ❌ `ROPE` | ❌ | ❌ | 3 gaps |
| `attention_mx` | ✅ | ❌ `SOFTMAX` | ❌ | ❌ | 3 gaps; also mx-only dtype regime |

Reference: `MODELED_OPCODES = {RES_PACK, MATMUL_RESIDENT, MATMUL, COMMIT, VECTOR_MAP, VREDUCE,
ATTENTION_QK, ATTENTION_PV, CONV2D, MOVEMENT}`. The emitter can produce `SOFTMAX`, `ROPE`,
`RMSNORM` and `BATCHED_MATMUL` (`targetgen/contract/interface_emit.py:56-83`), none of which is
modeled.

**This is the actionable core of the note.** "We have no large capsules" is not a vague gap; it is
this table. `attention_qk` and `conv2d` are each **one field away** — they clear the golden, the
reference engine and work counting today, and fail only at `declared_capsule_macs`. A pricing rule
for the ops `work_volume` already counts unlocks the two highest-value shapes immediately.

Note that a `None` price is not cosmetic. It nulls `declared_macs`, `ideal_cycles_at_peak`, both
utilizations and both `share_of_achievable` fields, and it **disables the corpus-wide attainment stop
condition for every member** (`perf_agent_stage.py:565`).

## What such members may claim

They should not join a fitted law, and they do not need to. A **paired self-comparison across
compiler revisions** cancels everything by construction — same shapes, same math, two emissions —
which is the identical structure `PQ` already uses (`comparand.kind: paired_run`). The claim is
"candidate X is faster than baseline on this workload", which is exactly what an external ablation
tests, and it is falsifiable without a fitted line.

For the strength of a claim resting on measured small siblings, use the vocabulary that already
exists rather than inventing one — `tier_policy.py:281-311`:

```python
CLAIM_EXTENDS            = "screened_at_cap_resting_on_verified_sibling"
CLAIM_EXTENDS_UNVERIFIED = "screened_at_cap_resting_on_UNVERIFIED_sibling"
CLAIM_SCREENED_ONLY      = "screened_at_cap_resting_on_nothing"
```

`verify_extends` (`:441-488`) fails closed, recording an unverifiable `extends` as *weaker* than
naming nobody, "because an unchecked `extends` reads as certified."

The governing constraint is unchanged and is measured (`tier_policy.py:23-28`): a capsule the cheap
tier REFUTES will not certify (confirmed 12/12); a capsule it PASSES is not certified (one submission
passed the cheap tier 20/20 while RTL passed 1). **A screen may eliminate; it may never certify.**

## What stays out, and why these are backend gaps rather than corpus gaps

Do not re-litigate these here:

- **`PF` — `epilogue_fusion` (L5).** Needs a fused matmul+bias member *and* a standalone bias member.
  This backend implements no bias-add epilogue, so two of the three member kinds fail code
  generation and the capacity rule `complete_fused_and_part_comparison_group` can never be met.
  Restoring it is backend functional work. Note `emitter.status: existing` — the corpus side is ready.
- **`PB` — `host_island_placement` (L4).** Blocked on `fused_single_elf`: the capsule path runs the
  accelerator on the target's oracle and the host lane in-process on the development machine, so a
  seam differential would price the workstation rather than the SoC. Until both lanes execute from
  one ELF under one cycle counter, the cost of an accel→host→accel round trip cannot be stated. The
  round-trip **count** is still an exact, free, actionable fact from the emitted program — report it,
  and refuse to price it.
- **`PT`.** Its cohort's starting depth is a measured property of the target, and no materializer can
  derive it without consuming a recorded overlap probe. Minting the depths from a written-down number
  would repeat `PK`'s own original defect.
- **`PS` and `PG`** are `skipped_inapplicable` on this target by trait refutation, not blocked
  (`self_hosted_program: false`; `multiple_operand_encodings: false`). Both carry full tri-state gate
  evidence in `MANIFEST.yaml:292-345`.

## Two false records this note supersedes

A wrong "blocked" record is worse than no record, because it reads as a settled capability finding.

1. **`MANIFEST.yaml:414-427` still carries `PV` as `blocked_unimplemented`** with the reason that the
   integer reference engine has no convolution definition. That is **false** — `CONV2D` is in
   `MODELED_OPCODES`. It was true only while a library regression was in place. `_perf.yaml:945`
   already unblocked the family (`emitter.status: existing`, four fitted `ci` points at tile
   multiples); what is stale is the on-disk manifest and the absent `PV*` directories. **The corpus
   needs regenerating, not unblocking.** `merlin/tests/infra/test_perf_conv_family.py:51` guards the
   false claim's return.
2. **Eleven `PQ` capsules declare `emitter.entry: merlin.perf.barrier_arms.pair_from_emitter` with
   `emitter.status: existing`, and that function does not exist.** `barrier_arms.py` defines only
   `count_barriers`, `paired_removal` and `analyze_barrier_claim`. The family's second arm has never
   been emitted; its own module docstring records that ten capsules measured one arm against nothing.
   `analyze_barrier_claim` — which decides exactly the claim `PQ` declares — has no caller anywhere.
   This must be repaired before the synchronization family is deepened toward the regime where the
   4,098-fence pathology lives.
