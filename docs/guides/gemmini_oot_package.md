---
title: Running the certified gemmini OOT backend on a fresh machine
kind: guide
status: current
owner: compiler
last_verified: 2026-09-06
related: [whole_model_on_accelerator, reproducing_whole_model_on_rtl, gemmini_experiment, reproducibility, adding_a_target]
code_refs: [merlin/python/merlin/targetgen/capsule_grade.py, merlin/python/merlin/targetgen/capsule_runner.py, merlin/python/merlin/targetgen/rtl_engine_policy.py, merlin/python/merlin/targetgen/contract/compile.py, merlin/python/merlin/targetgen/publish.py, merlin/python/merlin/targetgen/oot_runner.py, merlin/python/merlin/targetgen/_capsule_bundle_worker.py, merlin/python/merlin/perf/hw_counters.py, merlin/contract/hardware_pins.yaml]
---

# Running the certified gemmini OOT backend on a fresh machine

There is a gemmini out-of-tree MLIR backend that certified **33/33 capsules at L3** (elaborated RTL) on
2026-09-02. This page is what a second person needs to re-run it somewhere else and read the result
correctly. Concepts for the *other* whole-model path live in
[whole_model_on_accelerator](whole_model_on_accelerator.md); its runbook is
[reproducing_whole_model_on_rtl](reproducing_whole_model_on_rtl.md). **This is a different artifact and
a different flow** — an agent-generated OOT backend graded through `capsule_grade`, not
`compile_model`.

## Two published artifacts — do not confuse them

This page documents **two** gemmini OOT backends. They are different artifacts, graded over different
cohorts, and their numbers are not comparable.

| | `gemmini_xdsl_rtl_v0` (2026-09-02) | `gemmini_xdsl_oot_v0` (2026-09-06) |
|---|---|---|
| branch | `stable/gemmini_xdsl_rtl_v0` | `profiling/gemmini_xdsl_oot_v0` |
| tag | `v0-gemmini_xdsl_rtl_v0` | `v0-gemmini_xdsl_oot_v0` |
| headline | 33/33 at L3 over a 34-capsule cohort | **83/96 public**, **14/14 hidden**, over the 96-capsule public set |
| status | `rtl_certified` | `capsule_graded_l3_partial` (`certification: not_certified`) |

The rest of this page's cohort arithmetic describes the FIRST artifact. The second is documented below.

## `gemmini_xdsl_oot_v0` — the 2026-09-06 profiling artifact

    git clone -b profiling/gemmini_xdsl_oot_v0 git@github.com:ucb-bar/gemmini-mlir.git

Agent-generated from RTL-derived facts (`authoring.mode: agent_generated_from_rtl_facts`), lowering an
interface dialect to RoCC command buffers and a linked ELF. Published with `--no-gate`: it is a
**profiling artifact, not a champion**, and its manifest says so.

### The measured result

| | |
|---|---|
| public | **83/96**; tier_reached L0 83, L1 83, L2 82, **L3 78**; RTL-backed passes **76** |
| hidden (held out) | **14/14**, `functional_pass: 1`, highest tier **L3** |
| integrity | `clean`, `gradeable: true` |
| L3 engine | **GSIM**, not Verilator |

The cycle-accurate tier here was **GSIM**. Per this guide's own comparison, GSIM reports 0..+2
cycles per invocation (<=+9 over a capsule window) against Verilator, so these numbers are
elaborated-RTL but **must never be mixed with Verilator numbers in one comparison**.

### Read this before treating the 13 non-passes as bugs

**Ten of the thirteen are the backend DECLINING, not computing anything wrong.** Its own
`DELIVERABLE_OUT_DTYPES` gate refused to hand back an f32/bf16 result it could not encode through the
runner's integer readback at the time. It computed those regions correctly on its scalar lane; lifting
only that gate later, 9 of 10 pass **bit-exact (`max_abs_error: 0`) at L3** against an independent
`host_torch_eager` golden.

The other three fail on three DIFFERENT planes, and none is simply "a routing bug":

| capsule | plane | note |
|---|---|---|
| `SY_epilogue_bias_add` | spike / program-oracle | numeric compare is **exact**, `mismatch_count: 0`; the oracle rejects the `bias_add` epilogue name, and the two engines demand different ranks for the same bias operand |
| `SY_micro_model` | model | whole-model execution proof |
| `M3_host_island_seam_gemmini` | model_execution | `FALLBACK_ON_ELIGIBLE_REGION`, 21 regions — **open question**, see below |

On `M3`, the submission's own `REPORT.md` argues the number is produced by the whole-model plane's own
model compiler and is byte-identical even when `params.lanes` is rewritten to claim every region on the
mesh — i.e. it may not be this package's routing at all. Treat it as unresolved rather than as a
confirmed defect in this backend.

### Getting the capsules it was graded against

The public capsule SOURCES are tracked in the merlin repo and arrive with a normal clone:

    merlin/contract/capsules/{isa,layers,model,model_slices,_perf}/

The run graded a **materialized** public subset, not those directories directly. Regenerate it with
`merlin.targetgen.contract.materialize.public_capsules_for(<target_experiment>)` rather than copying a
cache path — the cache is not a source.

**The authoritative list of what was graded** is `grading_public/score_capsule.json` (96 capsules, named,
with per-capsule status) and `grading_hidden/score_capsule.json` (14). Do not reconstruct the set by
guessing; read those files.

**What is deliberately NOT published:** `merlin/contract/capsules/hidden/` and every `golden.yaml` /
`golden.npy` are untracked on purpose — they are ANSWER KEYS. Publishing them would destroy the
held-out set for every future run and make the 14/14 hidden result unciteable. Their absence is by
design, not an oversight.

**What that means in practice.** For **performance profiling** (cycles, utilization, roofline) goldens
are not needed — run the capsule and measure. For **correctness grading** the pass/fail verdicts cannot
be reproduced without goldens; use `capsule_grade --no-oracle` for structure-only checks, or cite the
recorded verdicts in the score files above.

### Citing cycle numbers from this artifact

Measurements are against gemmini_rtl commit `63f0b68a68f1` **plus** the reviewed off-pin bytes in
`src/main/scala/gemmini/LoadController.scala` and `include/gemmini.h`. Cite it as
"`63f0b68a68f1` plus those bytes" — **never** as "pinned"; the bytes differ from the pinned revision
and a claim derived from them is not a pinned claim.

## Read this before quoting the number

`33/33` is over a **34-capsule cohort, not the corpus**. From
`merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml`:

```yaml
expected_cohort: {source_capsules: 48, admitted_capsules: 34}
capability_exclude_capsules: [GC1…GC6, GF1…GF5]     # 11 bf16 ops this RTL cannot execute
resource_bound:
  policy: representative_l3_capstones_v1
  exclude_capsules: [GX0_interop_rvv_lane, M0_small_llama_gemmini, M1_lstmnetvit_gemmini]
  required_admitted_models: [M2_microvit_gemmini, M3_host_island_seam_gemmini]
```

48 − 11 − 3 = 34.

| | |
|---|---|
| **Established** | 13 ISA capsules + 11 layers + 8 model slices + 1 model capsule, certified on elaborated RTL |
| **Established** | the host/device seam: `M3_host_island_seam` (GEMM → host LayerNorm → GEMM) with an ordered dispatch ledger and per-tile mesh verification |
| **Not established** | whole-model compilation — the three capstones that would show it are excluded by `resource_bound`, and are neither passed nor refuted |
| **Not established** | anything about `M2_microvit`, which ended `budget_exhausted` |

### The number against today's corpus

The corpus grew after certification: **63** capsules now sit in the graded roots. Re-graded on the
merged tree the package gives:

| | count | |
|---|---|---|
| not graded — outside this RTL's declared capability | 13 | the bf16 set; a capability fact, not a backend gap |
| **non-model graded** | **43** | **39 pass, 4 fail** |
| model graded | 6 | 6 `incomplete` — **none of them reached the compiler**, see below |

So the number to quote for "how does it do today" is **39 of 43 non-model capsules**, and it is not
comparable to 33/33: that was a 34-capsule cohort, and 8 of the capsules in today's 43 did not exist
when the package was certified.

The four failures are all `SY_*` capsules added after certification, and they are two different
things:

| capsule | plane | what it actually is |
|---|---|---|
| `SY_reduction_i8_aligned` | `parse` | the package's parser rejects the op — a real gap: it does not implement this reduction |
| `SY_reduction_i8_partial` | `parse` | same |
| `SY_regime_fits_single` | `spike` | **spike itself aborts** (`std::out_of_range`), exit 255 |
| `SY_regime_spills` | `spike` | same |

Only the first two are the backend. The other two are the functional simulator crashing, which says
nothing about the package.

So: **do not tell anyone that 33/33 means a full model compiles.** If that is the question, the
capsule to grade is `M0_small_llama_gemmini` (embed / RMSNorm / RoPE / attention / SwiGLU / lm_head),
and it has not been graded. See [Grading the excluded capstones](#grading-the-excluded-capstones).

## The package

Tracked at `out/artifacts/targets/gemmini/gemmini_xdsl_rtl_v0/`, with per-file checksums:

```bash
cd out/artifacts/targets/gemmini/gemmini_xdsl_rtl_v0
sha256sum -c SHA256SUMS --quiet && echo OK
```

`PROVENANCE.md` beside it records the run, the candidate, and what the verdict covers. ~1,100 lines of
original code; the rest is vendored `xdsl`/`typing_extensions`/`immutabledict`/`ordered_set`, kept as-is
so the directory *is* the graded artifact.

Its contract is one tool with four commands (`manifest.yaml`):

| command | argv |
|---|---|
| `parse` | `gemmini-opt --verify-diagnostics <in.mlir>` |
| `lower_interface_to_target` | `gemmini-opt --convert-iface-to-gemmini <in.mlir>` |
| `emit_command_buffer` | `gemmini-opt --convert-iface-to-gemmini --emit-command-buffer=<out.json> <in.mlir>` |
| `lower_target_to_llvm` | `gemmini-opt --convert-iface-to-gemmini --emit-target-artifact <in.mlir>` |

## Getting it

Two repos, and you need both. `gemmini-mlir` is the **backend**; the thing that compiles a model is
Merlin, which consumes it.

```bash
# 1. the driver
git clone git@github.com:ucb-bar/merlin.git && cd merlin
git checkout feat/target-generalization
cp .env.example .env                                  # then point MERLIN_EXT_* at your toolchain/sims
ln -s <llvm-23-install> third_party/llvm-install       # see the trap below -- do this first

# 2. the certified backend
merlin-target-fetch gemmini --champion stable/gemmini_xdsl_rtl_v0
```

The backend is published at **`ucb-bar/gemmini-mlir`**, branch **`stable/gemmini_xdsl_rtl_v0`**, tag
`v0-gemmini_xdsl_rtl_v0`. That repo uses branch-per-version publishing, so each champion is its own
branch and the landing page on `main` lists them all; nothing is overwritten by a later promotion.

The branch carries its own provenance under `.merlin/`:

```
.merlin/CHAMPION            gemmini_xdsl_rtl_v0 68ae8bb cert_A2_verilator
.merlin/certification.yaml  certification: pass, gate: oot_runner.certify,
                            5 rungs x {oracle: rtl_verilator, derived_from_rtl: true,
                                       cycle_accurate: true}
.merlin/manifest.yaml       the contract: entrypoints + the four commands
.merlin/provenance.yaml     the run and Merlin sha it came from
```

### Why the branch is trustworthy, and what the certification is *of*

The publish gate accepts only `rtl_certified` or an `oot_runner.certify` pass — a *functional*
simulator pass is not accepted as a substitute, deliberately. This package earned it with five
certify runs on **cycle-accurate Verilator**, spanning families:

| capsule | what it exercises | cycles |
|---|---|---|
| `A2_single_tile_matmul` | the base contraction | 302 |
| `A4_acc_scale_i8` | accumulator scaling | 269 |
| `A6_resident_reuse` | two matmuls on one resident weight | 525 |
| `B1_linear_relu_i8` | a fused epilogue | 385 |
| `GC0_conv2d_i8` | convolution via im2col | 3,863 |

All four declared entrypoints pass in each. Those are RTL cycle counts, not model estimates.

### It is the 33/33 bytes

Verified by diffing the published tree against the candidate that earned the verdict
(`_qa_work/cand_01/submission` of run `merlincirct_arm4_func_20260902_codex3`):

```
certified candidate: 476 source files
published package:   476 source files
  only in certified: 0   only in published: 0   differing content: 0
  digest 815d3885d82b6820 == 815d3885d82b6820
```

Not a rebuild and not a repackage. Verify your own clone with `sha256sum -c SHA256SUMS` against the
copy tracked in Merlin at `out/artifacts/targets/gemmini/gemmini_xdsl_rtl_v0/`.

## Prerequisites

| | what | how it is found |
|---|---|---|
| Python | 3.13 venv at `.venv` (uv) | `.venv/bin/python`; plain `python` is not on PATH |
| `.env` | external-path map | copy `.env.example`; without it collection dies with `external path 'chipyard' unset` |
| `third_party/llvm-install` | the LLVM 23 host toolchain | **see the trap below** — a fresh checkout does not have it |
| answer keys | `golden.yaml` + coverage, untracked by design | regenerate: `generate_corpus.py --target gemmini` |
| model2MLIR venv | only for the **model** capsules | `MERLIN_M2M_PYTHON`; without it they are skipped loudly |
| RISC-V toolchain | for the bare-metal harness link | chipyard's `esp-tools`/`riscv-tools` env |
| chipyard + gemmini RTL | the L2/L3 oracles | `MERLIN_CHIPYARD`; **verify the revision** against `gemmini_rtl` in `merlin/contract/hardware_pins.yaml` |
| L2 oracle | spike + `libgemmini` | built in chipyard |
| L3 oracle | one of VCS / **GSIM** / Verilator | see below |

### Choosing the L3 engine

`merlin/python/merlin/targetgen/rtl_engine_policy.py` selects by availability in cost order
(`vcs`, `gsim`, `verilator`) and records every engine it passed over. All three produce an
`elaborated_rtl` verdict — the tier is a **fidelity, not a simulator**.

- **GSIM** is the practical choice, and is what certified 32 of the 33. Point `MERLIN_GEMMINI_GSIM_EMU`
  at the emulator binary.
- **Verilator** is the *citable* instrument for cycle counts. GSIM runs a slightly different elaborated
  design (harness clock ratios collapsed, one pad cell removed from a clock path), and measured
  same-ELF it reports **0 to +2 cycles per kernel invocation, ≤ +9 per capsule window** against
  Verilator. Fine for a functional verdict; **never mix the two in one comparison.**

## Grade it

```bash
export TMPDIR=/path/with/space                       # not /tmp; whole-model builds need room
export PYTHONPATH=$PWD/merlin/python                 # pin it — a shared venv can shadow the tree
export MERLIN_GEMMINI_GSIM_EMU=/path/to/emu_gemmini_gsim   # or leave unset for Verilator

.venv/bin/python -m merlin.targetgen.capsule_grade \
    --package out/artifacts/targets/gemmini/gemmini_xdsl_rtl_v0 \
    --target gemmini --labels public \
    --runs-root out/runs/gemmini/verify --score out/runs/gemmini/verify/score.json \
    --timeout 21600 --workers 6
```

Omit `--capsules`: the default is the target's own graded roots, which is what applies the cohort
policy above. A hand-passed root is how a package ends up graded against the wrong corpus.

Cycle counts are concurrency-invariant, so `--workers` does not change them. **Wall times are not
comparable across different `--workers` values.**

### Read the result correctly

Never quote a bare score. Per capsule, read `tiers.<T>.status`, `cycles`, `derived_from_rtl`,
`cycle_accurate` and `evidence`:

```json
"L3": {"status": "pass", "cycles": 19658, "derived_from_rtl": true,
       "cycle_accurate": true, "evidence": "rtl_gsim_console.log"}
```

A `pass` whose `tiers` is empty, or whose `derived_from_rtl` is `false`, is not an RTL result. `status`
values that are **not** the compiler getting it wrong: `gated`, `budget_exhausted`,
`not_gradeable_no_oracle`, `incomplete` (`NOT_RUN_IS_NOT_PASS`). Do not count those as failures *or*
passes.

## Host ⇄ device: what the seam now establishes

This is the part that improved, and it is worth being exact about.

`M3_host_island_seam_gemmini` is a GEMM → host LayerNorm → GEMM capsule where the host island is the
*subject*, not a by-product. It passes at L3 with:

- an **ordered dispatch ledger** — `{on_mesh: 2, scalar_rvv_lane: 16}`, so both lanes provably carried
  work;
- **per-tile mesh verification** (`mesh_tile_verification.per_tile`), 2 of 2 tiles certified;
- `boundary_execution: A->H->A/pass`, matching an independently derived expectation;
- **`model_execution_check: pass`** with zero violations — the first recorded verdict from that gate,
  which had been computed and discarded on every prior run.

Its L3 evidence is therefore `mesh_execution.dispatch_ledger + mesh_tile_verification.per_tile` rather
than a console log, and that is a stronger claim, not a weaker one: the tiles were verified
individually.

Two honest caveats:

- **The host half executes on x86.** The lane is served by `lower_model(targets=("host",))` +
  `build_host_shared`, run through ctypes. `scalar_rvv_lane` is a *routing label*; the seam is proven as
  a **partition**, not yet as RVV execution on the device.
- Set **`MERLIN_MESH_VERIFY=1`** to get the per-tile verification. It defaults **off**, and a
  whole-model pass taken without it carries `derived_from_rtl: false`, no ledger and `evidence: null` —
  which is a vacuous pass by this repo's own standard.

## Per-unit activity, optionally

The RTL counts the cycles each combination of its three engines (`EX`/`LD`/`ST`) was busy, and the
harness can bracket a kernel with those counters:

```bash
export MERLIN_HW_COUNTERS=1
```

Off by default and deliberately so — a change that altered every run would make one round's verdicts
incomparable with the rounds before it. With it on, each RTL tier record gains a
`timing_observations` block (`busy_cycles.<engine>.in_program`, `overlap_cycles.observed`,
`idle_cycles.no_unit_busy`) plus a `timing_capability` record. The block is refused on functional
tiers: a model that runs the program without modelling the engines returned per-engine busy totals in
the *thousands* for a 52-cycle window.

Measured on the K-sweep with this package, Verilator: 20–41% of each window has **no** engine busy, and
`min(T_compute, T_movement) − realised_overlap` leaves 70–83 cycles of overlap still available.

## Regenerating the answer keys (why a clone grades 43 and not 49)

A clone has no `golden.yaml`, no `expected_instruction_coverage.yaml`, and no
`capsule.weights.safetensors`. That is deliberate, not missing: this repo is public, those files are
the graded answers, and `merlin/contract/capsules/.gitignore` untracks them with its reasoning
attached. What stays tracked is the CONTRACT the backend compiles against —
`capsule.interface.mlir`, `capsule.yaml`, `MANIFEST.yaml` — and the generators that reproduce the
rest.

The consequence to expect: every model capsule comes back

```
whole-model grade error: ValueError: model capsule external weights asset is missing
or a symlink: 'capsule.weights.safetensors'
```

Measured on a fresh worktree, all six graded model capsules fail this way — **including `M3`, which
passes at L3 on a machine that has the file**. So a clone grades the 43 non-model capsules and none
of the 6 model ones, and the failure is an absent asset, not the backend.

### Regenerate them

The corpus generator is tracked and rebuilds goldens, coverage and weights:

```bash
export MERLIN_M2M_PYTHON=<model2MLIR venv>/bin/python     # and MERLIN_M2M_DIR if it is elsewhere
.venv/bin/python merlin/contract/capsules/generate_corpus.py --target gemmini
```

Model capsules are lowered end to end through **model2MLIR** and graded against their host
torch-eager output, so that venv is a hard requirement for them. Without it the generator does not
fail — it skips them loudly:

```
[skip] M0_small_llama_gemmini: model capsule needs the m2m venv (set MERLIN_M2M_PYTHON)
```

Operator, layer and model-slice capsules regenerate without it. So if you only need the 43, you do
not need model2MLIR at all; if you want the model capsules, you do.

### A regenerated key is not automatically the key that was graded

Regeneration reproduces the answers *as the generator computes them today*. Where a capsule's loader
has drifted from the frozen weights, regenerating changes what the capsule asks for — which is a real
state in this corpus, not a hypothetical: see the next section.

## The excluded capstones do not run, and it is not the compiler

If the question is "does a full model compile", the capsules that would answer it are the three the
cohort excludes. **They were force-graded, and all three come back `incomplete`.** The blocker is in
the capsule bundle, before the compiler is reached.

`_capsule_bundle_worker` compares the frozen weights against the loader's `state_dict` and raises if
the name sets differ. Measured:

| capsule | frozen tensors | of which parametrized | loader `parametrize` refs | outcome |
|---|---|---|---|---|
| `M0_small_llama` | 51 | **45** | **0** | `incomplete` |
| `GX0_interop_rvv_lane` | 51 | **45** | **0** | `incomplete` |
| `M1_lstmnetvit` | 98 | **0** | **6** | `incomplete` (the mirror image) |
| `M2_microvit` | 19 | 0 | 0 | `budget_exhausted` |
| `M3_host_island_seam` | 4 | 0 | 0 | **passes at L3** |

`M0`'s weights carry `blocks.N.attn.k.parametrizations.weight.original0/1/2` — a
`torch.nn.utils.parametrize` quantization wrapper — while its loader builds plain
`blocks.N.attn.k.weight`. 45 of 51 names are unreachable. `M1` disagrees the other way round.

So the three capsules the descriptor excludes under `resource_bound` are **exactly** the three whose
bundles are internally inconsistent: they could not have run at any budget, and the exclusion reads
as a cost decision while masking staleness. `M3`, the one consistent multi-lane capsule, passes.

**Whole-model capability on this target is therefore neither established nor refuted.** Fixing it
means re-freezing a capsule's weights (or restoring the parametrization in its loader), which changes
an answer key — do that deliberately, not as a side effect.

### If you do want to grade them

Two things will otherwise waste your time:

- **Grade the models in the same invocation as the operators.** A whole-model capsule is deferred
  unless the operator pass fraction clears its declared `after_op_pass_fraction`. Grading the model
  directory alone computes that fraction as `0.00` and gates every model capsule with
  `whole-model capsule deferred: op pass fraction 0.00 < gate 0.8` — which looks like a model failure
  and is an artifact of what you passed to `--capsules`.
- **Raise `MERLIN_MODEL_BUDGET_S`.** `M2` ends `budget_exhausted` at the default, and
  `budget_exhausted` is not a failure — it is a capsule that never finished.

### The bundle worker throws its own diagnosis away

Worth knowing before you debug one of these. `_capsule_bundle_worker.main()` prints its reason to
**stdout** and returns non-zero; the resulting `SystemExit` traceback goes to **stderr**; and the
caller does `run.stderr or run.stdout`. So every bundle failure is reported as a two-frame traceback
ending at `raise SystemExit(main())` and the actual cause is discarded. Reproduce the worker call
directly, or read the loader and the safetensors key sets yourself — which is how the table above was
obtained.

## Traps

- **A missing host toolchain looks like 43 compiler failures.** A fresh checkout has no
  `third_party/llvm-install`, and every capsule then fails as
  `spike / tool_crash: [Errno 2] No such file or directory: .../bin/clang-23`. Measured: 43 graded,
  43 failed, which reads exactly like a broken backend and is not one. Link or install it first:
  `ln -s <llvm-23-install> third_party/llvm-install`, then confirm `third_party/llvm-install/bin/clang-23`
  resolves. With it in place the same package graded 17/17 on the same commit.
- **A shared venv can shadow the tree.** `import merlin` may resolve to a *different* worktree than
  your `cwd`. Always pin `PYTHONPATH`, and check `merlin.__file__` if a result surprises you.
- **`TMPDIR=/tmp` is not enough space** for whole-model builds, and an empty `TMPDIR` fails oddly.
- **Hardware provenance.** A cycle count without the RTL revision it came from is not citable. Check
  `gemmini_rtl` in `hardware_pins.yaml` by content, not by branch name — branches move and forks share
  them.
- **The 11 bf16 capsules are excluded because this RTL cannot execute them.** That is a capability
  fact, not a gap in the backend.
- **`git commit` is safe under a running grade; `checkout`/`stash`/`merge` are not.** Committing does
  not touch working-tree files, so a grade reading the tree is unaffected. Anything that rewrites
  files mid-grade is not.
- **A fresh worktree has no `out/`**, which is gitignored. The publish `index` command run from one
  listed only the packages that worktree happened to track and silently dropped a package that is
  live on the remote. Run `index` from a checkout whose `out/artifacts/targets/<target>/` holds every
  package you expect listed, and compare its output against `git ls-remote --heads`.

## For maintainers: publishing the next champion

The flow is `materialize` -> `record-cert` -> `promote` -> `publish` -> `index`, and the gate is real:

```bash
merlin-target-publish materialize  --target <t> --from <run>/submission --package-id <id> \
                                   --certified-by-run <run-id>
merlin-target-publish record-cert  --target <t> --champion <id> --results <certify run dirs...>
merlin-target-publish promote      --target <t> --champion <id>
merlin-target-publish publish      --target <t> --champion <id> --execute --confirm-push <fingerprint>
merlin-target-publish index        --target <t> --execute --confirm-push <fingerprint>
```

Notes learned the hard way:

- `--confirm-push` must equal the publish's **content fingerprint**, printed when the push is
  refused. It changes whenever the content does, so it cannot be passed blindly — expect to run the
  command twice.
- `materialize` **overwrites the package directory** from the source submission. If you had added
  anything of your own there (a `PROVENANCE.md`, a `SHA256SUMS`), it is gone; regenerate after.
- `record-cert` preserves each rung's **tier**. A functional-simulator pass and a cycle-accurate RTL
  pass are both `pass` to the gate but are not the same claim, and `certification.yaml` says which
  one it is. Do not reach for `--no-gate` to skip earning it.
- **Verify by cloning the published branch, not by inspecting the assembled tree.** The first Python
  champion published with `entrypoints.tool: build/bin/gemmini-opt` (a CMake output path that never
  exists for an interpreted package) and with the tool at mode `100644`. The assembled tree looked
  right in both respects; only a clone showed otherwise. Both are fixed, and the check to run is:

  ```bash
  git clone -b <branch> git@github.com:ucb-bar/<target>-mlir.git chk && cd chk
  git ls-tree HEAD <target>-opt      # expect 100755
  grep -A1 '^entrypoints' manifest.yaml
  ./<target>-opt --verify-diagnostics <some>.interface.mlir
  ```
