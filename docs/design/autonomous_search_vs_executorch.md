---
title: "Design note: can the search beat ExecuTorch on its own? (int8, K1, from a frozen unoptimized seed)"
kind: design
status: current
owner: rvvgen
last_verified: 2026-09-07
related: [beam_cca_architecture, codegen_vs_handc_wholemodel, expert_gap_attribution]
code_refs: [merlin/python/merlin/mining/beam.py, merlin/python/merlin/mining/select.py, merlin/python/merlin/mining/runner.py, merlin/python/merlin/mining/wholemodel_proposer.py, merlin/python/merlin/llvmlower/impr_features.py, merlin/python/merlin/runtime/backends/zephyr_model.py, merlin/python/merlin/baselines/executorch.py, merlin/python/merlin/kernels/action_catalog.py]
---

# Can the search beat ExecuTorch on its own?

## The experiment, stated so it can fail

We are not a library of hand-written kernels, so our claim has to be that *global* optimization plus a
leaner runtime beats a per-target kernel library. The honest test of that claim is not "can a human
tune our compiler until it wins" — it is:

> Start from the **frozen, unoptimized lowering**. Let the tooling extract, analyse, mine and search
> on its own. Does it reach a configuration that beats XNNPACK/ExecuTorch — and does it do so across
> *several* models, not one?

This note records the first end-to-end run of that experiment on `small_llama` int8 on the SpacemiT K1,
what it found, what it did **not** find, and — the part worth keeping — the several ways the
measurement was wrong before it was right.

Everything below is measured on the K1 (VLEN 256, 1.6 GHz, `rdtime` 24 MHz). The board's noise floor is
**≥1.9 %, band 2.6 %** — deltas under that are not results, and are not reported as such.

## Why this has to be a search, and not a set of good defaults

The tempting shortcut is to take every lever that helped and turn it on by default. The data says that
would ship regressions. Two levers, measured the same way on the same board:

| lever | small_llama int8 | small_llama fp32 | spectformer int8 |
|---|---|---|---|
| weight pre-transposition | positive | positive | positive |
| `promote_buffers_to_stack` | **1.34×** faster | 1.04× | **~1.01× SLOWER** |

Pre-transposition is universally positive — across five bundles it hoisted 15 / 41 / 28 / 302 / 1
transposes with **zero blocked** — so it belongs in the AOT path as a rule. Stack promotion is
model-dependent *with opposite signs*, so it belongs to the search. A blanket default would have
regressed spectformer to buy small_llama.

That asymmetry is the whole argument for the beam. The generalizable artifact is not the winning
configuration; it is the *procedure that finds the winning configuration per model*.

## Step 0: instrument the opponent before optimizing against it

Before any tuning we profiled ExecuTorch itself — runtime, lowering, AOT, encodings and prepacking —
rather than treating it as a single wall-clock number. Two facts reframed the work:

| | ExecuTorch | ours (at the time) |
|---|---|---|
| load vs execute | 13,836,638 ns load / 3,738,641 ns execute | — |
| memory plan | **ONE 32,512-byte arena**, planned AOT | **209 `tensor.empty`**, each its own heap buffer |
| emitted GEMM, 128³ | 798,857 instructions | **394,442** (2.03× *fewer*) |

The third row is the important one and it is easy to misread. **Our emitted GEMM is not the problem** —
it issues half the instructions XNNPACK does for the same shape. We were losing on everything around
it: layout, allocation, and per-tile runtime calls. That is precisely the class of thing a global
compiler should win and a kernel library cannot, which is why it was worth attacking rather than
conceding.

It also told us where *not* to look. Scalar transcendentals are 2.42 % of real work — below the board's
own noise band. `__ieee754_sqrtf` on this target is a single `fsqrt.s`, so RMSNorm's rsqrt was never an
algorithmic lever at all.

## Step 1: the measurement protocol was wrong, and it was wrong against us

This is the correction that mattered most, and it invalidated every ratio computed before it.

`certify_rvv` called `run_on_k1` with the defaults `iters=1, warmup=0`. So the beam **ranked every fork
on a single cold inference**, while ExecuTorch's reported number averages its cold execution into
`--num_executions` and is therefore mostly warm. ET's cold inference is 1.62× its warm one; ours was
*fully* cold. The comparison was skewed by roughly 1.3×, against us.

Sustained (2 untimed warmup + 5 timed) is now the default, extracted by the two-N slope method
`total(N) = cold + (N-1)·warm`. The fix carries a test that pins **both** that the defaults are
sustained **and** that `certify_rvv` actually forwards them to the board — accepting the arguments and
silently dropping them is exactly the failure that produced the bug.

The lesson generalises past this bug: *a benchmark harness that measures the wrong regime does not look
broken.* It produces plausible numbers, in the right units, that rank forks incorrectly.

## Step 2: the run

Seeded from the frozen `hand_v0` package — **zero compiler features**, the naive lowering — with
ExecuTorch's sustained wall passed in as `--expert-wall-ns 3288885` so every fork reports
`attainment_vs_expert = ET/ours` as a first-class output (≥1.0 means matched or beaten) rather than
something computed afterwards by hand.

Width 8, depth 2, all teachers, 25 nodes, 24 forks. The ladder the search climbed:

| depth | lever | proposed by | wall (ns) | vs seed |
|---|---|---|---|---|
| — | seed, zero features | — | 349,877,321 | 1.00× |
| 1 | `erase_self_copy` | teacher:matmul + xnnpack-cca | 24,736,122 | 14.14× |
| 1 | **`perop_register_block`** | census:byte-traffic | 10,767,352 | **32.49×** |
| 2 | + `dtype_strategy: int8_w8a8` | teacher:matmul | **5,186,291** | **67.46×** |

Multi-teacher discovery worked: matmul, softmax, gelu and the xnnpack-cca teacher between them answered
every divergence axis, and `teacher_audit.yaml` records **`unanswered_axes: []`** in all three
generations. No blind spots — a contrast with the recorded `NO_TEACHER_FAMILIES` gap for
`batch_matmul`, which the audit continues to report honestly rather than paper over.

## Step 3: the result, including the part that does not support the headline

**67.46× from the frozen unoptimized seed, entirely by search.** No lever was hand-picked; each was
routed from a measured CCA divergence, forked, built, gated on cos, and ranked on a sustained board
measurement.

**But `attainment_vs_expert = 0.634`, i.e. still 1.58× behind ExecuTorch — and that number is not
apples-to-apples.** The depth-2 step that won is a `dtype_strategy` flip from `fp32` to `int8_w8a8`:
it changed the **datapath**, not the schedule. ExecuTorch's int8 is *weight-only*. So 5,186,291 vs
3,288,885 compares our W8A8 integer datapath against their weight-only one.

That is the same defect class as the contaminated `expert_wall_ns` cells and the mismatched-bundle
ratio — a denominator that silently describes a different computation — now appearing on the
dtype-strategy axis. On the ET-comparable fp32 datapath the best node is
`perop_register_block + erase_self_copy` at **10,683,237 ns**, i.e. **attainment 0.308**.

Reported honestly: the search produced a large, real, autonomous speedup, and it has **not** beaten
ExecuTorch on a like-for-like datapath.

## Step 4: why it stopped, in its own words

The run's `deferred_work_items` is the useful artifact. **Eleven of twelve deferrals carry
`reason: over_width`** — `promote_buffers_to_stack`, `perop_nr_fill_register`, `fuse_transpose_b`,
`mrpad`, `vectorize_reduction` and `perop_register_block` itself were all *proposed and then dropped
for budget* at width 8. The twelfth is the VL-agnostic `vsetvli` loop, which is honestly blocked on
codegen that does not exist (catalog route 327).

So the binding constraint was **search budget, not capability, and not missing teachers.** That is a
much better failure than the alternative.

### Two levers are not in the search space at all

Both are read straight from the environment, so no fork can vary them:

| knob | default | what the sweep measured |
|---|---|---|
| `MERLIN_PROMOTE_STACK_BYTES` | 16 KB | 4 KB 1.00× · 16 KB 1.03× · 64 KB 1.05× · **256 KB 1.34×** · 1 MB saturated |
| `MERLIN_PEROP_MR_CAP` | 4 | MR 8 measured 1.125× on small_llama fp32 |

The beam gets the 16 KB default, which is why its `promote_buffers_to_stack` fork came out *slower*
(11,499,694 ns) than not using the lever at all. The lever is not weak; the search cannot reach the
part of it that works. Making these first-class searchable knobs is the next change, and it is tooling
work rather than tuning — which is the point.

## Step 5: a gate disagreement worth fixing

Two gates in this repo return different verdicts on **identical numbers**
(`cos=0.9999078512191772`, `rel=0.014796391526079668`):

- `zephyr_model._gate` — two-tier and literature-backed for W8A8 (T1 vs a W8A8 reference, T2 vs the
  fp32 golden with an argmax and per-element term), `ok = T1 or T2`. **Passes.**
- the op-profile path — a flat `cos > 0.9999 AND rel < 0.01`. **Refuses to record a wall at all.**

So the configuration one tool crowns is the configuration the other will not measure. `_gate` is the
better-reasoned instrument — a flat `rel < 0.01` is an fp32-strict threshold applied to a W8A8
datapath where the literature expects cos 0.99–0.999 — but they cannot both be the answer, and the
disagreement is currently silent.

This sits alongside the standing rule that aggregate gates accept badly broken kernels: a cos-passing
kernel measured **1209 % off on individual elements** is why the per-element term exists.

## Defects found and fixed on the way

Recorded because each one silently produced plausible-but-wrong numbers, which is the failure mode this
whole effort is trying to make impossible:

- **The XNNPACK fixture harvester stopped normalizing.** `_machine_independent` gated on
  `head.endswith(".o")`; when the pipeline started linking to `.so`, normalization silently stopped.
  Now gated on the `:\tfile format ` separator alone.
- **Instruction counts were doubled.** In the RVV decoder the `""` section key collided with
  instructions whose section is genuinely `""`, double-counting every one. Fixed with a NUL sentinel
  (`"\x00all"`); an existing hermetic test caught it.
- **A 460× claim was really 45×.** The baseline had been measured while the beam was loading the host.
  Re-measured same-moment (6.31 s → 0.14 s) and corrected.
- **"RUNNING" reported three times for a dead job** — `pgrep -f beam_int8_sweep` was matching its own
  command line.
- **A broken commit** from scripted edits where a later assertion failed after earlier substitutions
  had already applied. Now covered by a structural test that parses `main()`'s call.
- **`promote_buffers_to_stack` claimed a family-wide property it did not have.** The recorded note
  said the `accumulator_resident_wholemodel*` family "has no self-copy to erase". True of the MR=1
  member, false of the MR>1 members — and generalising it is what left the lever unactuated. Corrected
  in place with the counter-measurement.
- **`fuse_elementwise_after_generalize` was refuted** (1.22× slower) and is registered *with that
  number in its description*, so it is not re-attempted.

## Open

- **smolVLA** blocks on `BlockAgreementError` for `linalg.matmul:1x32:32` — the op is a named matmul
  in the source but is gone by the time the tagger walks the specialised IR. A matcher disagreement:
  `observe_contractions` prices generics, the tagger matches named ops only.
- **lstmnetvit** is genuinely numerically wrong (`cos=0.9942953`, `rel=0.2591`) with the correct W8A8
  baseline *and* golden. It enters the beam in repair mode, where `correctness_residual` is the
  objective — the same machinery applied to a correctness bug instead of a speed one.
- ExecuTorch sustained walls for spectformer and lstmnetvit, so their beams can report attainment on
  the same basis.
- Deeper/wider re-run once the two environment knobs above are searchable.

## 2026-09-06 whole-model K1 update: grouped direct convolution and fork grain

The original LSTMNetVIT entry above is historical: its bad W8A8 result was traced to the capture, and
the corrected full-output capture now passes both its independently recomputed W8A8 reference and the
fp32 tier. Two compiler changes then moved the measured wall materially:

| full-model LSTMNetVIT W8A8 | 1 hart | 8 harts |
|---|---:|---:|
| previous best grouped-im2col path | 433.026 ms | 86.860 ms |
| grouped-direct convolution | **147.059 ms** | 66.280 ms |
| grouped-direct + `parallel_grain_10000` | — | **57.184 ms** |
| + AOT prequantization of six recurrent weights, launch median | **141.088 ms** | **55.071 ms** |
| + AOT prequantization of all 17 eligible weights, launch median | **133.275 ms** | **50.965 ms** |
| + targeted residual broadcast fold, accepted launch median | **130.880 ms** | **49.640 ms** |
| ExecuTorch qd8 reference | 51.943 ms | 15.345 ms |

The accepted final row is the median of three feature-on launch means from an
off/on/on/off/off/on experiment; each launch contains two warmups and five timed inferences. Its
separate median across all 15 raw iterations is 131.070 ms at one hart and 49.540 ms at eight. All
launches pulled the complete output and produced SHA-256 `adf8308a...`. Consequently this is a real
compiler/runtime improvement, but **not a win**: the remaining gaps are 2.52x at one hart and 3.23x
at eight harts. Merlin scales 2.64x across the matched builds, versus 3.39x for ExecuTorch. The
reference was refreshed on 2026-09-07 from N=1/N=4 warm slopes after
matching the exporter and runtime source at ExecuTorch commit `7fc34bf6f53d2098e3e16c1fa71c23222f607330`;
all four reference runs passed the full correctness gate.

The threshold sweep also closes a prior open question. `parallel_grain_10000` is the measured winner;
30,000 did not beat it and had 3.74% session drift, while 100,000 regressed to about 69.8 ms. Dynamic
per-operation team widths were refuted at about 124--127 ms. Requant/contraction pairing is correct
after removing the paired contractions' parallel tag, but serialising those matmuls regressed the
eight-hart wall to about 69.6--71.0 ms, so it is not part of the current champion.

The post-all17 eight-hart profiler initially appeared to rank contractions at 14.85 ms, quantization
at 8.72 ms, layout copies at 8.06 ms, broadcasts at 5.61 ms, and adds at 4.39 ms. Two independent
guards reject that as an optimizer-selection breakdown. First, instrumentation moves the matched wall
by 2.70%, above the 1.9% acceptance limit. More importantly, a candidate that explicitly folded the
161 profiled sole-use broadcasts produced a byte-identical `model.ll` and `model.o`: the ordinary
uninstrumented lowering already folds them, while the profiling calls pin the intermediates and make
them materialize. Thus at least 4.86 ms of the apparent broadcast class is profiler-induced rather
than residual release-binary work. The raw profile remains useful for diagnosing the instrumented
IR only; the next ranking must intersect its rows with operations proven to survive uninstrumented
lowering, and every performance verdict still requires the uninstrumented bracketed protocol.

That first ranked lever is now built and measured. `prequantize_constant_weights` stores the six
recurrent matrices and their scales in the bundle, removing their runtime scale search and
round/clamp/cast chains. A three-by-three interleaved A/B measured 149.575 -> 141.088 ms median at one
hart (1.060x) and 58.284 -> 55.071 ms at eight harts (1.058x). All twelve launches produced the same
full-output digest and passed both correctness tiers.

The rewrite is now name-independent and recognizes every structurally eligible constant F32
contraction weight, including output-channel-preserving collapse/expand chains and direct-convolution
weights. It prequantizes 17 weights in this capture: the original six recurrent matrices plus 11
convolution weights. Against the six-weight arm, a second three-by-three interleaved A/B measured
139.856 -> 133.275 ms median at one hart (1.049x) and 55.207 -> 50.965 ms at eight harts (1.083x).
Every launch retained the exact three-element output digest. The pass still refuses aliases,
multiple readers, unsupported layout chains, non-finite scales, and reshapes that merge the scale
axis.

The next uninstrumented survivor census found 52 sole-use broadcasts feeding `linalg.add` and 10
feeding `linalg.mul` in the tagged all17 input. Existing post-contraction fusion removes 23 before
the pre-generalization hook, leaving an explicit pass receipt of 31 adds plus 8 multiplies. The
default-off `targeted_named_broadcast_fold` composes only those marked generalized-broadcast source
maps into their all-parallel consumers; it refuses shared uses and never touches a contraction or
reduction. Unlike the earlier 161-broadcast diagnostic negative, its final `model.ll` and `model.o`
both differ from control and the object contains 2,479 fewer emitted instructions. The controlled
board A/B measured launch medians of 135.208 -> 130.880 ms at one hart (1.033x) and 51.005 -> 49.640
ms at eight harts (1.027x). The raw-iteration medians, kept distinct, were 134.700 -> 131.070 and
50.912 -> 49.540 ms. All 12 complete-output launches were byte-identical and passed both W8A8 and
fp32 correctness tiers. This feature is therefore part of the current LSTMNetVIT champion envelope.

## 2026-09-07 TinyLlama: within 5--8% after restoring the wide vector axis

Panel packing had already removed the measured 8x cache-line amplification, but the emitted
MR4/NR16 workers still used `e32,m2`: only 16 of K1's 32 available int32 vector lanes. The MR4/NR32
candidate makes that axis explicit and emits `e32,m4` workers. All 177 wide MAC workers use the
32-lane form, with no vector spills or reloads, and all 155 weights remain packed and parallelized.

| full-model TinyLlama W8A8 | 1 hart | 8 harts |
|---|---:|---:|
| MR4/NR16 same-session control, median | 1,583.485 ms | 474.830 ms |
| MR4/NR32, median | **1,244.614 ms** | **402.107 ms** |
| MR4/NR32, best valid launch | **1,235.399 ms** | **399.829 ms** |
| ExecuTorch qd8 warm slope | 1,179.563 ms | 370.868 ms |

The width change is a controlled 1.272x/1.181x median improvement. Every launch produced the same
complete 256,000-element output digest. Merlin is now 1.047x slower at one hart and 1.078x slower at
eight harts on the best valid launches: close, but not yet a certified win. The fresh ExecuTorch
numbers are N=1/N=4 warm slopes from the same identity-matched `7fc34bf6...` exporter and runtime;
both reference runs pass at each core count.

Increasing MR instead was the wrong lever. MR8/NR16 regressed by 1.430x at one hart and 1.068x at
eight because register pressure changed four scalar A loads into VL=1 vector loads/extracts. This
refutation is retained so the search does not retry it.

## 2026-09-07 ResNet-50: tail panels fix one-core traffic, not multicore scaling

The im2col packer formerly accepted only 27 of 50 eligible contractions. Masked/narrow tail-panel
support now packs all 50 and rewrites all 73 panel regions. The new full-model binary preserves the
complete output digest and passes the shared W8A8 bar. In a same-session board comparison its best
one-hart wall fell from 2,654.273 to **2,253.149 ms** (1.178x), while the eight-hart result was noise-
equivalent to the old path at **1,251.291 ms**. ExecuTorch remains at 884.774 and 235.282 ms,
respectively. This separates two effects: tail packing repairs serial memory/codegen waste, but the
large eight-hart gap is still a parallel scheduling problem. An independent fp32 golden is still
missing, so this row remains diagnostic rather than certifiable.

## 2026-09-07 smolVLA: correct scope and current blocker

The previous `>120 s` Merlin result was explicitly a **one-hart** bounded launch, not an eight-hart
measurement. The fair ExecuTorch one-hart warm slope is 42,309.598 ms, so the only valid statement is
that Merlin's attainment was below 0.353. The captured workload is also a vision/language prefix plus
one flow step, not a complete ten-step action session.

The first eight-hart candidate completed in 146,455.732 ms but failed correctness: its 1,600 outputs
were 50 repeated copies of the final bias, meaning the learned contribution vanished. This wall is
therefore rejected. A 384-boundary trace localized the first divergence to the serial W8-to-BF16
conversion immediately before the first BF16 projection: Merlin's locally defined `__truncsfbf2`
used an integer-class return on RISC-V while clang 23's caller expected the BF16 value in `fa0`.
The initial smoke test accidentally resolved libgcc's correct helper and was therefore vacuous; a
second reproducer linked the exact Merlin runtime object and failed as the full model did.

With the ABI corrected, both the fully traced model and the uninstrumented model are byte-identical
to the host golden. The latter runs in **150,255.120 ms** on eight harts and uses 3.26 GiB RSS. This
is a correctness milestone, not an INT8 performance row: the candidate stores 303 weights as i8 but
dequantizes them to f32/BF16, and its contractions use f32/BF16 activations and dequantized weights
with f32 accumulation. The matched 23.10 s ExecuTorch qd8 reference instead dynamically quantizes
activations and invokes XNNPACK i8 microkernels. The current Merlin diagnostic is therefore about
6.50x slower, and the next required build is Merlin's actual W8A8 path; relabelling the mixed path as
INT8 would violate the requested comparison contract.

The newer v2 capture supplies that contract directly: its flow stage has 140 explicit
i8-by-i8-to-i32 contractions and no BF16 matmuls. Its first session build also exposed why the old
monolithic path could not be repaired merely by turning on `int8_compute`: that path converted only
95 of 298 contractions, leaving 203 BF16 sites and dequantizing 253,870,080 i8 weights into
507,740,160 BF16 bytes at runtime. The v2 build now passes preparation, but LLVM translation stops
fail-closed on 30 residual `vector.mask { vector.contract }` attention tails. They are 15 INT8 score
BMMs and 15 BF16 softmax-by-V BMMs at sequence length 113; MR=4 and NR=8 both leave tails. The next
compiler fix must peel or pad those tails while retaining the full-tile MR4/NR8 kernel. Globally
falling back to MR=NR=1 would compile by destroying the optimization under evaluation and is
rejected. This build also proves a session-specific scheduling gap: the session builder bypasses
`prepare_for_lowering`, so it does not currently derive the per-operation block and parallel-arm
tables used by the monolithic whole-model path.

## Reproducing this

### ⚠️ First, the trap that will silently run someone else's code

The shared venv carries a `.pth` that puts **a different checkout** on `sys.path`:

```console
$ .venv/bin/python -c "import merlin; print(merlin.__file__)"
.../<some-other-checkout>/merlin/python/merlin/__init__.py    # NOT this repo
```

`pytest` is unaffected (its `conftest` inserts the rootdir), but **every `-m module` and script
invocation resolves elsewhere**, so a flag added in this repo appears not to exist and a fix appears
not to work. Prefix every command below with `PYTHONPATH=merlin/python`, and verify once:

```bash
PYTHONPATH=merlin/python .venv/bin/python -c "import merlin; print(merlin.__file__)"
# must print <this repo>/merlin/python/merlin/__init__.py
```

`merlin.mining.pass_slot_wiring.checkout_pythonpath()` exists to compute this for spawned processes.

### 1. Prepare the bundle (weight pre-transposition)

Hoists stored weight transposes out of the runtime. Changes neither offset nor nbytes, so no runtime
change is needed; it is a pure AOT layout rewrite.

```bash
PYTHONPATH=merlin/python .venv/bin/merlin-bundle-pretranspose \
    out/artifacts/recaptures/small_llama_int8_consistent \
    out/artifacts/recaptures/small_llama_int8_consistent_pretransposed
# reports "<n> removed / 0 blocked"; a nonzero blocked count is a refusal to report, not a warning
```

### 2. Get the opponent's sustained wall

ExecuTorch must be measured in the **same regime**. Its cold inference is 1.62× its warm one, so a
single-shot number is not the comparand — take the two-N slope, `total(N) = cold + (N-1)·warm`.

### 3. Run the search

```bash
PYTHONPATH=merlin/python timeout 7200 .venv/bin/python -m merlin.mining.beam_cli \
  --model-dir out/artifacts/recaptures/small_llama_int8_consistent_pretransposed \
  --expert-objdump merlin/tests/data/cca_asm/xnnpack_qd8_gemm_rvv.objdump \
  --teachers all --op matmul --dtype int8 --targets k1 --proposer wholemodel \
  --width 8 --depth 2 --top-k 2 \
  --expert-wall-ns 3288885
```

- `--seed-pkg` is omitted **on purpose**: the default is the frozen `hand_v0`, so the run rediscovers
  the levers instead of inheriting a tuned config. Passing a tuned seed invalidates the experiment.
- `--expert-wall-ns` is what turns `attainment_vs_expert` into a first-class output.
- `--max-workers` defaults to 1 for a `k1` target (board-serialized). Do not raise it.
- Run it under `systemd-run --user` rather than `setsid`, which did not survive restarts.

### 4. Read the results

Everything lands in `out/runs/rvv/beam/matmul/<TS>_cca_beam_seed000_<sha7>/`:

| file | what to read it for |
|---|---|
| `beam_tree.yaml` | `nodes[]` (wall, speedup, `attainment_vs_expert`, `correctness_residual`, `gate_ok`), `best`, `baseline_frozen.verified_unchanged` |
| `teacher_audit.yaml` | `taught_by` per axis and **`unanswered_axes`** — a non-empty list is a teacher gap |
| `beam_tree.yaml: deferred_work_items` | **why the search stopped**: `reason: over_width` = budget-starved; anything else is a real block |
| `targets/rvv/<run_id>/knobs.yaml` | the `compiler_features` a fork actually carried |
| `forks/<run_id>/results.yaml` | the raw `correctness` dict behind `gate_ok` |

Sanity checks before believing any number: `baseline_frozen.verified_unchanged: true` (the frozen seed
still lowers byte-identically) and `repair_mode: false` (the seed was correctness-clean, so the search
was optimizing speed rather than repairing numerics).

### 5. Measure one configuration directly (no search)

```bash
PYTHONPATH=merlin/python .venv/bin/python build_tools/scripts/k1_op_profile.py \
    --model out/artifacts/recaptures/small_llama_int8_consistent_pretransposed \
    --features perop_register_block,promote_buffers_to_stack \
    --warmup 2 --iters 5
```

`--warmup`/`--iters` are the sustained protocol; omitting them measures cold and is not comparable to
ExecuTorch. Note the per-op tick totals accumulate across `iters` while the wall is per-iteration — the
script normalizes by `iters`, and a `profiler_coverage` far above 1.0 is the symptom of that
normalization being missed.

### 6. The two knobs the search cannot reach (today)

Sweep them by hand until they are searchable, always against a **same-session control arm** — the host
and board are shared, so a wall from another day is not a comparand:

```bash
MERLIN_PROMOTE_STACK_BYTES=262144 PYTHONPATH=merlin/python .venv/bin/python \
    build_tools/scripts/k1_op_profile.py --model <bundle> \
    --features perop_register_block,promote_buffers_to_stack --warmup 2 --iters 5
MERLIN_PEROP_MR_CAP=8 ...   # same shape
```

### 7. Prove a lever is not inert *before* trusting a wall

Two levers here were historically inert while looking correctly wired. Require a changed mnemonic
stream, not just a changed runtime:

```bash
PYTHONPATH=merlin/python .venv/bin/python build_tools/scripts/k1_codegen_vs_handc.py --feature <name>
```

and read loop spans off the **linked ELF** — in an unrelocated `model.o` branch displacements are
unresolved, so `loop_spans()` silently reads 0 (measured: 0 spans from the object vs 6,017 from the
ELF).

### 8. Tests

```bash
.venv/bin/python -m pytest merlin/tests/rvv merlin/tests/ir -q
```

## What this note is not

It is not a claim that we beat ExecuTorch. On a like-for-like datapath we do not, yet. It is a record
that the *mechanism* works end-to-end without a human in the loop, that it reports its own budget
starvation accurately, and that the remaining distance is now attributable to two named, fixable gaps
in the search space rather than to anything unknown.
