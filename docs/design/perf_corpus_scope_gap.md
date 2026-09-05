---
title: "Design note: the perf corpus cannot express the optimizations that matter"
kind: design
status: current
owner: core
last_verified: 2026-09-05
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
| a large shape addresses memory outside the scratchpad/accumulator, and **claims resident weight reuse while actually reloading** | L2 intra-layer, residency | **structurally unreachable** — see below |
| massive kernels (≈1000×700×700) crash outright | — | unreachable for the same reason; deprioritised, but it is the same region |

### The residency gap is an empty interior, not a missing point

The false-residency bug deserves its own statement because it is the most expensive one — `PR`,
whose declared lever is `operand_residency`, owns **92.4% of all corpus cycles**, and this bug says
that family's central claim can be false without any member noticing.

Measured coverage of the two axes across all 49 members:

| family | M | N | K |
|---|---|---|---|
| `PK` | 16 | 16 | 16 … 128 |
| `PL` | 16 | 16 | 16, 32 |
| `PQ` | 16 | 16 | 16, 32 |
| `PC` | 16 | 64 | 64, 128 |
| **`PM`** (parallel extents) | **16, 32, 48, 64** | **16, 32, 48, 64** | **16 — fixed** |
| **`PR`** (operand residency) | **16 — fixed** | **16 — fixed** | **16 … 16384** |

**No member has both `K > 16` and `M > 16`.** The two axes that jointly produce residency pressure
are each swept while the other is pinned, so the covered region is an L-shape and the bug lives in
its empty interior. `PR`'s own `source_reference` says as much: *"at fixed single-tile parallel
extents."*

Put in bytes, against the derived store — `memory_regime.operand_store('gemmini', dtype='int8')`
returns a **262,144-byte scratchpad, 16,384 rows × 16 int8**:

- largest weight working set in the corpus: `PR08_spills_k16384` at 16384×16 = 262,144 B — **exactly
  1.0× capacity**, and that member is the one whose Verilator capture failed at the 3600 s default;
- the reported failing shape, 1000×700×700 int8: weight 490,000 B + activation 700,000 B = **~1.19 MB,
  about 4.5× the whole store**.

So the corpus probes residency along a *thin needle* (N=16) up to exactly capacity, and the defect
lives at four and a half times capacity in a shape where both dimensions are large. A schedule can
claim residency, silently reload, and every member still passes.

**The detector already exists and is already wired.** `perf/structural_levels.py:102` emits
`residency_restaged` (`L2_intra_layer`) on exactly this pattern — a value staged (`RES_PACK`),
released (`EVICT`), then staged again — and `analyze_command_buffers` already returns it to the
agent for free (`perf_agent_stage.py:1915-1922`). It reports **zero findings on all corpus buffers**,
which is a true negative: no member is large enough to spill. Replayed over the wider run tree it
fires 17 times, every one on a *multi-op* capsule-bench kernel. The instrument works; it has never
been pointed at a workload that could trip it.

The addressing half has an existing guard too: `perf/preflight.py:17-18` refuses a tensor larger than
the DRAM window because such a program "is not rejected — its tail is never loaded. The device
executes the prefix and halts, **and the cycle count describes the prefix.**" Any member in this
region must be run through preflight, or it returns a number, and the number is wrong.

**Corpus action:** the missing member is not "bigger". It is a member with **large M, large N, and a
weight that exceeds the store** — one point in the empty interior — carrying a residency claim that
`residency_restaged` can falsify. That is one new sweep, not a new subsystem.

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

- **`PF` — `epilogue_fusion` (L5): RESOLVED, and it is the worked example of what unblocking costs.**
  The family needs a fused matmul+bias member *and* a standalone bias member, so it requires a
  backend that implements a bias-add epilogue; without one, two of its three member kinds fail code
  generation and the capacity rule `complete_fused_and_part_comparison_group` can never be met. That
  was the state when this note was first written, and the entry said restoring it was backend
  functional work rather than a tuning lever.

  **That work has since been done.** `BIAS_ADD` is modelled in `runtime/reference.py`,
  `corpus_spec.py` derives the bias operand (`_bias_operand`, `_BIAS_STAGES`), and `PF` is a live
  sweep with six members rather than a `blocked_unimplemented` entry. So `L5_fusion` is no longer an
  empty rung, and the ladder table above should be read as the state of a corpus that lacked it.

  ⚠️ **Do not re-derive the block from a branch that lacks the epilogue.** A branch whose backend has
  no bias support will regenerate a corpus with `PF` blocked and a `_perf.yaml` that says so, and
  merging that over a branch which HAS the epilogue silently deletes six working members and a whole
  rung of coverage. The capability lives in the backend; the block record only reports it.
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

## How a member too large to run gets a number

Measured 2026-09-04; artifact `out/artifacts/perf-bench/<target>/composed_band_validation.json`,
produced by `validate_composed_bands.py`, module `merlin/python/merlin/perf/compose_estimate.py`.

A large member cannot be measured — the oracle runs at ~217 simulated cycles/s over a fitted domain of
161..28,118 cycles, and `cert_cost.predict` refuses to extrapolate past twice that. What can be
composed from measurements of smaller siblings is a **band**: a structural floor (priced MAC demand
over the derived peak) and an empirical ceiling. Never a point.

**The obvious ceiling is refuted, and this is worth recording so nobody re-derives it.** The serial
per-command cost model looked correct in principle — its own artifact declares
`fidelity: "L2.5 calibrated (linear, serial; no overlap)"`, and this machine's measured composition is
partial (η = 0.1667), so a serial sum ought to credit no overlap the machine achieves and therefore
over-predict. Over **25 labelled programs on 24 workloads it contained zero of them**, every
measurement above it by **2.0× to 39.2×, median 2.9×**. The cause is structural and is the same one
`tiled-unit-needs-two-k-points` records: the model prices a *histogram of command kinds*, and one
`MATMUL` is one command whether it contracts over 16 or over 16384. Its ceiling sat near-constant at
~133 cycles across workloads whose true cost spans 269..3877. **A per-command constant cannot price a
tiled unit.**

The work-scaled ceiling — the same priced MACs over the slowest rate anything on this machine has been
*measured* at — contains **16 of 17 held-out programs**. Rates are split by compute class because one
global rate, though sound at 18/18, produced bands **95.7× wide** with measurements sitting at a
median position of 0.12 inside them: convolution runs at 2.67 MACs/cycle here and a resident matmul at
94.1, so a single bound covering both is 35× looser than either needs. Per class the median width is
**18.9×**. The single miss is `GP2_conv2d_maxpool_i8`, a fused conv+pool whose pooling work the MAC
counter does not price — so the ceiling under-counts the work and the measurement lands above it.

Two consequences for corpus design:

- **The band eliminates; it never certifies.** `compare` speaks only when two bands are disjoint, so
  the screen is sound whatever the overlap operator does — it never subtracts two composed envelopes,
  which is what `differential.compare` correctly refuses on a partial-overlap machine.
- **It needs about an order of magnitude to speak.** At 18.9× width a 2× difference does not separate,
  and a test pins that. This bounds what the screen is *for*: it will not rank two tilings of one
  shape, and it will separate a kernel issuing thousands of redundant synchronisations from one
  issuing two. That second case is the 4,098-fence pathology, which is the point.

## The corpus sits on the lightest reachable class and misses the heaviest

⚠️ **THIS SECTION SUPERSEDES A STRONGER CLAIM MADE HERE ON 2026-09-05**, and the way it went wrong is
worth more than the claim was. It read: *"27 of 29 OBJECTIVE members classify as `projection_like` — a
class the census does not contain at all … the corpus is not on their manifold."* That was true of the
census as it stood when it was measured. **The census is re-derived from the recapture store, and it
moved the same day**: `projection_like` is now a census class with 148 regions. A finding taken against
a derived artifact expires when that artifact is re-derived, and the only defence is to re-measure
rather than to quote.

Measured again, after the re-derivation, with `dse_guidance.shape_taxonomy.classify_geometry` against
`conformance/gemmini.yaml` → `shape_geometry.required`:

| census class | regions | mac_fraction | reachable | OBJECTIVE members |
|---|---:|---:|---|---:|
| `wide_skinny` | 2248 | **0.821837** | **no** | 2 |
| `squareish_gemm` | 289 | **0.094393** | yes | **0** |
| `unknown` | 36 | 0.046123 | yes | **0** |
| `tall_skinny` | 771 | 0.035248 | **no** | 0 |
| `projection_like` | 148 | **0.002062** | yes | **27** |
| `odd_tail_heavy` | 16 | 0.000327 | yes | **0** |
| `gemv_like` | 66 | 0.000010 | **no** | 0 |

The gap is narrower than the superseded claim and still decisive:

* **Every OBJECTIVE member sits in the lightest reachable class.** All 27 classifiable members are
  `projection_like`, which carries **0.2%** of contraction MAC mass. The heaviest reachable class,
  `squareish_gemm` at **9.4%** — forty-six times the mass — has **none**, and neither does `unknown`
  at 4.6%.
* **85.7% of MAC mass is in classes the census itself marks unreachable** (`wide_skinny` 82.2%,
  `tall_skinny` 3.5%, `gemv_like` 0.001%). The two members that do land in `wide_skinny` are there by
  aspect ratio only, at 1,024 output elements against a class whose real shapes carry a 589,824,000-
  element tensor.

So the corpus is on the manifold, at the one reachable point that matters least. Optimising every
member perfectly moves a share of model time bounded by the mass of the class they occupy.

The threshold-free version of the same fact is **output size**: the OBJECTIVE members produce 256 or
1,024 elements, while the three geometry members the synthesizer actually mints for the reachable
classes produce 8,960, 49,152 and 150,528. This is the same fact as the corpus's cheapness rather than
a separate one — the L3 engine's wall cost tracks output size, so a corpus small enough to sweep per
candidate is small *because* it is unrepresentative, and the two cannot be separated by tuning the
member list.

> **Every affordability number in this note is read from a DERIVED spec and will move when that spec
> is re-derived.** `cert_affordability` in `conformance/<target>.yaml` is regenerated from the
> certifications on disk, and the figures quoted below (`max_elements: 875`, `budget_s: 300`,
> `seconds = 80.6607 + 0.250531 * output`, fitted over 240..1,024 elements) are what it holds today.
> One correction is already known to be coming: `cert_cost.capsule_output_elements` inferred a
> capsule's written extent from the operand it READ rather than its declared result, which priced a
> convolution by its input image — `SY_conv_k16x16` at 16,384 elements and 7,177 s where the truth is
> 256 elements and 81 s — and priced seven flash-attention capsules at zero. Fixed in `f98f0085`; the
> spec has not yet been regenerated through it. This note has already had one finding expire exactly
> this way (see the superseded census section above), so treat a quoted number here as a reading with
> a date, not a constant.

## A fourth wall: the output has to be read back, and that costs simulated cycles

Three walls on capsule size were already recorded here — the affordability cap, the extrapolation
refusal, and the materialization ceiling. A fourth was found by measurement and it is the one that
actually stops a census-anchored certification.

Measured on GSIM, this checkout, a ladder of resident matmuls after the blob path made them buildable:

| shape | MACs | cycles | GSIM wall | achieved |
|---|---:|---:|---:|---:|
| 64x64x64 | 262 K | 4,087 | 150 s | 64 MACs/cyc |
| 128x128x64 | 1.05 M | 12,483 | 346 s | 84 |
| 256x256x128 | 8.39 M | 83,857 | 1,298 s | 100 |
| **1024x256x128** (`tall_skinny`) | 33.6 M | — | 1,438 s | **FAILED** |

Least squares over the three that completed gives **≈ 130 s + 14.0 ms/cycle**, with efficiency still
climbing as the fixed overhead amortises. That law is worth having: it puts a `squareish_gemm`
certification at roughly 1.5 h against the 27.3 h that `cert_cost`'s output-element law projects by
extrapolating a fit made on 240..1,024 output elements out to 49,152.

But the fourth row did not produce a number at all:

```
GemminiError: OUT Y0: expected 131072 values, got 29068
```

The build was fine — a 664-byte harness, a 462 KB ELF, both operands linked as blobs. **The harness
prints the output tensor over a simulated UART, and a 131,072-element output does not finish
printing.** Only 22% of the values came back. Whether the run exhausted `+max-cycles` while still
printing or the capture truncated, the consequence is the same and it is a property of the OUTPUT
SIZE, not of the arithmetic: the readback is part of the simulated program.

Three things follow.

**The affordability cap is guarding something real.** `cert_affordability` measures
`written_output_elements`, and it has been easy to read its 875-element limit as merely conservative
because its seconds law extrapolates so badly. It is tracking a genuine hard ceiling with a bad
formula. Those are different criticisms and only the second one is fair.

**The engine matters, and the two do not agree.** The three geometry members the synthesizer mints
were verified correct on SPIKE at outputs of 8,960, 49,152 and 150,528 elements. This failure is on
GSIM at 131,072. So a capsule can pass its functional oracle and still be unreadable on the
cycle-accurate one, which is exactly the tier a performance claim needs. A member sized against spike
is not thereby certifiable.

**A large member needs a readback that is not a print.** The cost of dumping the answer scales with
the answer, so any member whose output approaches six figures needs its result checked by a digest
computed on the device, or by reading memory out of band, rather than by printing every element. Until
that exists, the certifiable output size is bounded well below the census's reachable classes -- and
that bound, not build size and not simulation speed, is what limits how large a certified anchor can
be.

## The cost model on this branch cannot price a census-anchored member

Sizing those members needs `targetgen.cert_cost.fit_cycles_for`, and on this checkout it returns a
fit whose **domain is 159 … 1,174 cycles** (`n = 34`, `r² = 0.821`, intercept 54.93 s, slope
0.2292 s/cycle). With `_EXTRAPOLATION_MARGIN = 2.0` the fit therefore **refuses any prediction past
2,348 cycles**.

The four reachable census classes need, at this target's peak arithmetic rate, on the order of
16,000 (`gemv_like`, with `M` padded to the tile edge) to 200,704 (`squareish_gemm`) cycles — **7× to
85× past that refusal threshold**. That rate is derived, not assumed: the target's own RTL facts
report one `mesh` array of 16×16 `Tile` elements, 256 instances, `corroborated: true`, so a cycle
retires at most 256 MACs and the figures above are floors on an infinitely fast memory path. Refusing is the correct behaviour; the point is that nothing on this branch can size
them, so the plan's route through composed bands is not a preference but the only available one.

Two causes, both worth repairing and neither visible from the fit's output:

1. **The evidence that would widen the domain is not in this tree.** All 34 records come from
   Verilator-era `capsule-bench` runs plus two perf-bench probe cells; a search for GSIM records
   under this checkout's run root returns **zero**. The GSIM measurements that reach 28,118 cycles
   live in another worktree's `out/runs`, which `_cycle_records` never looks at. A cost model is
   only as wide as the runs its own root can see.

2. **The fit cannot say which engine it describes.** `_cycle_records` accepts any tier declaring
   `cycle_accurate` or `derived_from_rtl` and pools them, and the tier records carry no engine
   field — all 34 resolve to the bare tier name `L3`. The per-cycle cost of the two L3 engines
   differs by roughly fifty-fold, so a pooled fit belongs to neither. The observed per-record cost
   spans 0.185 … 0.903 s/cycle, a 4.9× spread across a 7× cycle range, which is most of why `r²` is
   0.82 rather than near one. This is the same defect the hardware-provenance convention exists to
   prevent, one level up: a number attributed to no particular machine.

Until (1) is fixed the affordability question for every large member is unanswerable rather than
answered pessimistically, and until (2) is fixed widening the domain would pool the two engines
harder rather than resolve them.

## Two false records this note supersedes

A wrong "blocked" record is worse than no record, because it reads as a settled capability finding.

1. **`PV` was carried in `MANIFEST.yaml` as `blocked_unimplemented`** with the reason that the
   integer reference engine has no convolution definition. That is **false** — `CONV2D` is in
   `MODELED_OPCODES`. It was true only while a library regression was in place. `_perf.yaml:945`
   had already unblocked the family; only the on-disk manifest was stale, and the `PV*` directories
   were simply absent. **RESOLVED 2026-09-04** by regenerating the corpus: `PV00_c16 … PV03_c64`
   now exist at `ci ∈ tile×{1,2,3,4}`, the corpus is 49 members, the blocked list is `[PB, PT, PF]`,
   and `performance_generation.gemmini.errors == []`.
   `merlin/tests/infra/test_perf_conv_family.py:51` guards the false claim's return.
   Note the members still carry `padding=[0,0,0,0] stride=[1,1] dilation=[1,1]`, so the stale-row
   conv bug remains unreachable until those axes are varied.
2. **`PQ` declared `emitter.entry: merlin.perf.barrier_arms.pair_from_emitter` against a name the
   module did not define.** This note previously recorded that the function had never existed and
   that no emitter carried a retire knob. **Both halves of that were wrong, and this entry is the
   correction.** `pair_from_emitter` was defined in the tracked module and was dropped when the
   module was rewritten, while the declaration went on naming it; the knob is a parameter of the
   target's own driver emitter, and two tests resolve both. Retargeting the entry at
   `paired_removal` on the false premise contradicted the family's own `regime.separation` ("the
   pair is two emissions of that one capsule") and its `negative_control` ("the single job member
   whose two arms are the byte-identical program"), and left those two tests red.

   **RESOLVED**: the module is now the union of both readings — `pair_from_emitter` BUILDS the pair
   through the emitter's retire knob and proves the arms are the same program plus barriers;
   `count_barriers`/`paired_removal` READ a pair the measurement path already built, where the two
   arms are two compilers. `analyze_barrier_claim` decides the growth falsifier for either. `PQ`
   points back at `pair_from_emitter`.

   The general lesson is not "someone declared vapourware". A declaration and its definition drifted
   apart **in both directions** with nothing between them to notice, and the second drift was caused
   by trusting a grep over the working tree instead of resolving the name.
   `test_perf_family_satisfiability.py::test_an_emitter_declared_existing_actually_exists` is that
   something; it imports the module and getattrs the name, and it catches the drift either way.

## What the FUNCTIONAL corpus can hold at census extents (measured 2026-09-05)

The sections above are about `OBJECTIVE` perf members. The same census also drives the **functional**
side, where `corpus_synth`'s geometry axis mints one `SY_geometry_<class>` member per reachable class
at that class's own recorded extents. That side moved: the emitter now links a large constant operand
as a binary blob rather than spelling it as a C initializer list, so a census-sized member is
buildable at all. Three are minted for this target, and all three were **built and run against the
reference on the functional oracle**:

| member | class re-derived from its own operands | M × K × N | written output | largest tensor | spike |
|---|---|---|---:|---:|---|
| `SY_geometry_projection_like` | `projection_like` | 56 × 480 × 160 | 8,960 | 76,800 | correct, 8,586 cycles |
| `SY_geometry_squareish_gemm` | `squareish_gemm` | 256 × 768 × 192 | 49,152 | 196,608 | correct, 68,871 cycles |
| `SY_geometry_odd_tail_heavy` | `odd_tail_heavy` | 196 × 256 × 768 | 150,528 | 196,608 | correct, 55,270 cycles |

The class is re-derived from the capsule's declared operands rather than from the requirement row
that produced it (`perf.member_geometry.stamp_for`); all three agree with the class they are named
for and all three report `in_census: true`.

**None of the three can be certified, and the numbers are the reason rather than an excuse.** The
budget is 300 s. The measured element fit (`intercept 80.66 s`, `0.2505 s/element`, `r² 0.490`,
domain 240 … 1,024 elements) affords **875 written elements** and, with `_EXTRAPOLATION_MARGIN = 2.0`,
**refuses to predict at all** past 2,048 — so every one of the three returns `None` rather than a
number. Pricing them through the cycle fit instead (`intercept 54.93 s`, `0.2292 s/cycle`, `r² 0.821`,
domain 159 … 1,174 cycles, measured functional-to-cycle-accurate ratio 6.22) gives **3.4 h**
(`projection_like`), **21.9 h** (`odd_tail_heavy`) and **27.3 h** (`squareish_gemm`) — 41× to 328×
the budget. That is why the axis caps each of them at the loop tier and names the certified sibling
it rests on; the cap is a measurement, not a preference.

### The classes that could not be admitted, and the wall that stopped each

The scaling search shrinks a class's heaviest region by a **common** factor on all three extents
until its largest tensor fits the 262,144-element operand store, with a floor of
`min(original, tile edge 16)` per extent so the result is still a problem rather than a degenerate
one. Three classes have no factor that satisfies both, and in every case **the floor is what stops
the search, not the ceiling**:

| class | heaviest region | largest tensor | mac_fraction | stopped at |
|---|---|---:|---:|---|
| `gemv_like` | 3 × 400 × 4096 | 1,638,400 | 0.00001 | factor 2: `M` → 1, below its floor of 3 (a GEMV's `M` is 1 in the application, so its floor is its own extent and it cannot shrink at all) |
| `tall_skinny` | 12288 × 64 × 1024 | 12,582,912 | 0.035248 | factor 5: `K` → 12, below the tile edge. Factor 4 (3072 × 16 × 256) is still 786,432 elements — 3× the store |
| `wide_skinny` | 128 × 2304 × 256000 | 589,824,000 | 0.821837 | factor 9: `M` → 14, below the tile edge. Factor 8 (16 × 288 × 32000) is still 512,000 elements |

These are recorded, with these reasons, in the census itself (`shape_geometry.required[].unreachable`)
and in the corpus profile's `provenance.geometry_classes_unsynthesizable` — the point being that a
class that gets no member has to be *named*, because a class silently absent from that list reads
downstream as a geometry the corpus covers.

A fourth class was absent from both, which is the defect this pass fixed. `classify_geometry`'s
residual label `unknown` covers **36 regions carrying 4.6% of all contraction MAC work** on this
target — more than two of the classes that *did* get a member — and the synthesizer skipped it with a
bare `continue`, so nothing anywhere recorded the hole. It still gets no member (the residual label is
not an aspect ratio: the regions behind it share no ratio, and a member named after the fall-through
would claim a coverage the classifier never asserted), but it is now reported with its mass.

⚠️ The reachability verdicts above are a function of the recapture store and move with it. An earlier
census on this same branch resolved `tall_skinny` at 1024 × 256 × 128 and `gemv_like` at 1 × 256 ×
1000; both became unreachable when the store gained captures whose heaviest region in those classes is
larger. Read the class list from `conformance/<target>.yaml`, never from this table.
