---
title: "Design note: can the search beat ExecuTorch on its own? (int8, K1, from a frozen unoptimized seed)"
kind: design
status: current
owner: rvvgen
last_verified: 2026-09-03
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
