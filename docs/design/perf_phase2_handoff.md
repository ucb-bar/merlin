---
title: "Design: open decisions and remaining work for gemmini phase-2 performance"
kind: design
status: current
owner: core
last_verified: 2026-09-06
related: [perf_phase2_wiring, perf_corpus_scope_gap, capsule_phase_split, derived_capsule_axes, dialect_test_bar]
code_refs:
  - merlin/experiments/gemmini_perf_bench/scripts/run_agentic_perf_experiment.py
  - merlin/experiments/gemmini_perf_bench/scripts/functional_gsim_qualification.py
  - merlin/experiments/gemmini_perf_bench/scripts/perf_claim_dispatch.py
  - merlin/experiments/capsule_bench/harness/cross_validate_engines.py
  - merlin/python/merlin/targetgen/sandbox/bwrap.py
  - merlin/python/merlin/llvmlower/perop_blocks.py
  - merlin/python/merlin/mining/wholemodel_proposer.py
---

# Phase-2 performance: what is decided, what is not, and what is left to build

This is a handoff. It exists so someone who was not present can pick up the phase-2 performance
work without re-deriving the measurements or re-discovering the failures. Read it top to bottom
once; each section states what is *known*, what is *decided*, and what is *open*, and those are
kept apart on purpose.

Two things in here are **decisions, not tasks**. They are marked as such. Do not implement past
them without making the decision explicitly, because both encode a claim about what our evidence
means, and picking one silently is how a benchmark stops measuring what it says it measures.

## 1. State as of 2026-09-06

**A phase-2 campaign is running.** Three concurrent trials, `gpt-5.6-sol` at effort high, seeded
from functional run `merlincirct_g4p1_20260905` (83/96 public, 14/14 hidden, 78 L3-certified).
Scope is the **PM family only** — 16 members, a complete 4x4 m/n grid, declaring
`perf_affine_claim.analyze_affine_claim/v1`. Budget 12 h wall, 11 h round timeout.

It launched **without** the functional cross-validation certificate, explicitly waived and
recorded. See §3.

**Landed this session** (all with tests):

| commit | what |
| --- | --- |
| `99857ed5` | route on the declared semantic family, not the op's spelling — **plan steps 1 and 2** |
| `abe6cb36` | content-addressed engine cross-validation capture store |
| `02a81539` | resolve a capture's engine set from the pin registry |
| `6d5b9934` | file captures from a finished functional grade |
| `72bb4a5c` | separate reference-engine deadline; drop declined capsules |
| `58ad630c` | run the sandboxed agent command from a file (fixes `E2BIG` at spawn) |
| `f4607b26` | correct what admits a launch: shared identities, narrowed cohort, waiver forwarding, certificate waiver |

**Not landed:** plan steps 3, 4 and 5 (§5). Those are the actual performance work.

## 2. The measured performance picture

A ResNet-50 run of the 83/96 compiler on FireSim/U250 measured an **11,183,959,730-cycle** model
interval, in which `linalg.generic` is **86.6%** and on-mesh contractions are **5.9%**.

Against other frameworks on the same hardware:

```
Merlin champion   1.00x
Voyager           5.81x faster
TVM-Gemmini      59.60x faster
AutoComp        102.66x faster
```

That 102x factors cleanly, and the factoring is the single most useful thing in this document:

```
AutoComp total        = 11.184e9 / 102.66 =  108.9M cycles
our on-mesh fraction  = 5.9% x 11.184e9   =  659.9M cycles
659.9M / 108.9M       =  6.06x
```

**AutoComp's entire model is ~6x faster than just our on-mesh portion.** So:

* **~17x** is work that should not be on the host at all — routing, fusion, quantization, layout.
  No amount of schedule search finds this.
* **~6x** is that our on-mesh code is itself inefficient — tiling, dataflow, utilization. This is
  what an autotuning search can win, and it is what the running campaign is aimed at.

The consequence for interpreting phase-2 results: the perf corpus is entirely matmul and residency
shapes, so it optimises the 5.9%. Even an *infinitely fast* accelerator yields `1/(1-0.059)` =
**1.06x at model level** until the 17x is addressed. Report phase-2 outcomes as improvement on the
accelerator-resident fraction, never as a model-level number.

## 3. DECISION 1 — what evidence admits GSIM as an oracle

### What the certificate actually is

`functional_gsim_qualification.py` builds a certificate proving GSIM and the reference engine agree.
`cross_validate_engines.compare_runs` is the comparison, and **it compares output bytes** — per
tensor, by digest, no tolerance — with the console `CHK` value as fallback. It does **not** compare
cycles. It is a claim about the *engine pair*, not about timing and not about any capsule.

GSIM cannot certify itself: `cross_validate_engines.py` refuses a self-comparison outright
("a self-comparison is trivially AGREE and proves nothing"). A second, independent implementation
is structurally required.

### The measured cost

Measured 2026-09-06 on the 80-workload cohort for submission `77004328`:

* GSIM runs a typical case in **5–16 s**; a handful of deep-K members take minutes.
* The Verilator reference leg costs roughly **45 min/capsule** (nine workers ran 49 minutes with
  zero completions).
* Exact-cohort certification is therefore **~6.7 engine-hours**, in front of a run that cannot start.
* Five capsules account for ~90% of that cost and ~6% of the cohort.

`rtl_engine_policy.ENGINE_PRIORITY` ranks `vcs` above `verilator`, and VCS is installed on the
current host, but **no `vcs_run.py` engine home exists for any target**, so the policy silently
passes it over (`program_oracle._RTL_ENGINES` declares the row; nobody built the wrapper). Note
that Verilator is a compiled, cycle-based 2-state simulator and VCS is event-driven and 4-state, so
VCS is **not** obviously faster here — the policy ranks it first for authority, not throughput.
Treat "use VCS" as an unmeasured hypothesis, not a fix.

### What was done, and its exact semantics

`--waive-functional-gsim-certificate` was added to `run_agentic_perf_experiment.py`. It is not a
default and not silent:

* it records the claim it withdraws — the functional regrade's verdicts become GSIM-only,
  uncorroborated by a second engine on the same ELF;
* it marks the run **not gate-clean**;
* it cannot relax the **tuning** certificate, which pins the timing authority and is never waivable;
* five mutation tests hold it narrow (`merlin/tests/infra/test_run_agentic_perf_experiment.py`).

The justification is an asymmetry, not convenience: **phase 1 graded this same submission on GSIM
at L3 and required no equivalence certificate at all.** Requiring one before phase 2 may *start*
held the second phase to a standard the first never met.

### THE OPEN DECISION

> Must every workload in the cohort be cross-validated, or is a declared representative sample
> enough?

The waiver is a stopgap, not an answer. The argument for sampling is strong and gets stronger once
you know the comparison is output bytes: two engines running one ELF over one elaborated design
either agree or they do not, and a 106,458-cycle capsule is not a better test of that than a
3,613-cycle one — it is the same test, 40x more expensive.

A concrete design, should sampling be chosen. `produce_gsim_certificate.derive_workload` already
separates exactly the right two things:

```yaml
operation: matmul
semantics:            # dtypes, epilogue, output dtype, numeric comparison policy
  numeric_policy: {compare: exact_int, dtype: i32}
  operand_dtypes: {lhs: i8, weight: i8}
  operation_attributes: {epilogue: [], output_dtype: i32}
shape: {k: 16384, m: 16, n: 16}    # the extent, i.e. the cost
```

`SY_kdepth_spills` and `SY_kdepth_fits_single` have **byte-identical `semantics`** and differ only
in `shape.k`. So: stratify by `(operation, semantics)`; the cheapest member of each stratum is
mandatory; a declared budget buys additional members; every unsampled member is enumerated in the
certificate with its stratum. Non-negotiable properties:

1. **Pre-register the selection in the sealed declaration**, before any capture runs. The
   declaration is already sealed content-addressed before captures, so this is natural. Without it,
   "sampled" degenerates into "whatever finished in time", which is the failure this repo keeps
   paying for.
2. **The consumer must re-derive the strata itself** rather than trusting the certificate's own
   coverage claim.
3. **Absent coverage metadata must read as "exact"**, never as "sampled" — an old certificate must
   not silently acquire a weaker meaning.
4. `extras` must still refuse a workload foreign to the corpus; only `missing` relaxes.

Code seams: `derive_cases` and the `expected` set in `produce_functional_certificate`
(`_completion` already takes `expected` as a parameter and needs no change);
`_verify_functional_certificate` and `_verify_functional_certificate_provenance` in
`run_agentic_perf_experiment.py`. Exactness is currently enforced in three places — those two, plus
`_capture_paths`/`_completion` — and all three must move together.

Verify by mutation: an exact certificate must still be accepted; a sample missing a stratum must be
refused; a certificate whose declared sample disagrees with the re-derived strata must be refused.

## 4. DECISION 2 — one campaign, one claim

`perf_claim_dispatch.resolve` refuses a cohort declaring more than one analyzer identity:

> "the cohort declares N claim analyzers ...; one campaign seals one claim, so a mixed cohort is
> refused rather than split"

This is deliberate and well-argued in that module's docstring: dispatch is on the contract's frozen
`analyzer` identity, never on a family name, because `PM` and `PV` once shipped declaring `PREDICTS`
with a contract no code path ever evaluated — producing measurements and no verdict while reading
exactly like a family that *was* evaluated.

The consequence nobody has decided on: the 38 certified members split into **four** campaigns.

| analyzer | families | members |
| --- | --- | --- |
| `perf_affine_claim/v1` | PM | 16 |
| `perf_paired_claim/v1` | PC, PL, PQ | 12 |
| `perf_pr_claim/v1` | PR | 6 |
| `perf_pk_claim/v3` | PK | 4 |

> Is "the phase-2 result" one campaign over one claim, or four campaigns that must be reported
> together? What is the unit of the experiment?

Four campaigns is 4x the agent budget and yields four separately-sealed claims that are not
obviously combinable. One campaign covers at most 16 of 38 members. Nothing currently records which
of these was intended. **Decide before spending budget on the other three.**

Related: `PK` requires exactly four frozen descriptors and `PR`'s `fits_double` band requires three
depths, so members cannot be dropped freely to reshape cohorts. `PK00_k16`, `PM00_m16n16` and
`PR00_fits_double_k16` are all the same 16x16x16 workload — the shared anchor of three sweeps — and
each family requires its own. That is legal and intended; `_verify_tuning_certificate` was fixed in
`f4607b26` to group by identity rather than refuse the repeat.

## 5. The remaining compiler work — this is the 17x, and it is the research

Steps 1 and 2 of the prior plan are landed (`99857ed5`): `model_op_demands` mints one demand per
compute operation rather than per textual `prov.op` tag, and `compute_units.supports_op` routes on
the declared semantic family with a `composed_with` capability never claimed standalone.

**Step 3 — stop materialising im2col.** The default conv lowering is `im2col gather + linalg.matmul`,
which materialises the activation `kh*kw/(sh*sw)` times over — on ResNet conv1 that is a
`3x7x7x1x112x112` intermediate, 1,843,968 elements, the activation written **12.25x over**. Above
`M2M_IM2COL_MAX_ELEMS` the frontend instead emits the true compound-affine `linalg.generic`
(`prov.conv_path=direct_contraction`) and materialises nothing, and
`perop_blocks.CONV_ARM_FEATURE` (`conv_register_block`, `perop_blocks.py:60`) is the arm that tiles
and vectorizes exactly that form. Both exist; both are default-off. This needs a **recapture at a
lower im2col budget**, and the arm's own registration says *"NOT MEASURED ON HARDWARE — static
evidence only"*, so it needs a correctness check before any performance claim.

**Step 4 — turn on the fusion that is already measured.** `fuse_requant_into_contraction` and
`fuse_epilogue_loops` are both `False` in `mining/wholemodel_proposer.py`. `requant_fuse` measures
**53 of 53 fusable** on `resnet50_v1_5_int8_w8a8_consistent`, worth 44,455,936 bytes of accumulator
traffic per forward; `epilogue_fusion` claims bit-identical output on small_llama. This is a default
change plus the regression proof.

**Step 5 — build and run the int8 ResNet.** Inputs exist at
`out/artifacts/recaptures/resnet50_v1_5_int8_w8a8_consistent/`, including
`golden_w8a8.independent.npy`. Check against the **independent** golden, never the consistent one
alone. Report against the **fp32 run of the same compiler**, never against the `tiled_*_auto` C
baseline, which is a different workload (batch 4, int8 output, hand-written schedules).

Sequencing is not negotiable: quantization rewrites a contraction into `linalg.generic` while the
RVV transform schedule matches only `linalg.matmul`, so moving to int8 *before* step 3 makes the run
slower — measured 19 -> 0 vectorized contractions on small_llama int8.

**Deliberately deferred, each needing its own plan:**

* `LOOP_CONV_WS` — Gemmini has a native convolution FSM (funct 15, CONFIG 16–21, RTL-attested) that
  **no backend emits**, and `gemmini_codegen_mlir._normalize_command_buffer` actively rewrites
  `CONV2D` into a host-gather recipe. This is the largest single remaining win.
* Residual add on the accelerator — not expressible in `EPILOGUE_STAGES`
  (`runtime/commandbuffer.py:76`), which is `(bias_add, bias, requant, acc_scale, relu, maxpool)`;
  it needs a second full tensor operand, not a per-column vector.
* Re-scoping the perf corpus so capsules reward *moving work onto the mesh* — whole-layer,
  fused-layer, model capsules — rather than only scheduling matmuls already there. Nothing in the
  present corpus can express the 17x.

## 6. Risk to verify: the harness may be capping the agent

The 5.9% on-mesh figure was measured on the **agent's own 83/96 submission**, and the router that
produced it (`merlin/python/merlin/targetgen/compute_units.py`) is **ours**, not the agent's. If
the agent's lowering path routes through `compute_units`, then our string comparison bounded what
that agent could achieve regardless of backend quality — and a phase-2 performance score measures
our ceiling, not the agent's skill.

This repo has recorded that pattern (harness limits reading as agent defects) eleven times. **Verify
whether the agent's out-of-tree backend actually routes through `compute_units.supports_op` before
citing any phase-2 performance number as a property of the agent.** If it does, re-baseline after
`99857ed5` before drawing conclusions.

## 7. Open defect: the argv escape hatch has never fired

`bwrap.compose_command` claims to move an oversized bind list into a file descriptor
(`_wrap_via_args_fd`) when the composed string would exceed the execve per-argument limit. It is
covered by `merlin/tests/infra/test_bwrap_cmd_argv_size.py`, which passes.

**There is not one `bwrap-args` file anywhere in `out/runs/`, in any run, ever** — including runs
whose composed command was measured at 178,303 bytes with the threshold set to 32 KiB. The
mechanism has never engaged in production while its test reports success. `58ad630c` routed around
this by writing the whole command to a file, so agent launches no longer depend on it, but the
composer itself is unexplained and should be treated as a live defect, not a fixed one.

This is the repo's recurring "a check that could not fail reported success" shape. Find it by
mutation, not by reading.

## 8. Standing constraints for whoever picks this up

* Derive target facts; never hardcode a target name, an opcode, or an encoding in
  `merlin/python/merlin/**` or `build_tools/scripts/**`. Fail closed as `UNKNOWN` rather than
  substituting a default.
* No regex in core library code; parse structurally.
* Generated output goes only under `out/`, via `merlin.common.paths` helpers.
* Tests live in `merlin/tests/<bucket>/test_<area>.py`, one of the eight buckets.
* Goldens and `capsules/hidden/` stay **untracked** — they are answer keys.
* Capsule runs use `--schedule continuous`, never the legacy `--continuous`.
* This is a public repo: no secrets and no local absolute paths in tracked files.
* The tree is shared by several agents at once. Commit with explicit pathspecs; a bare commit
  sweeps another session's staged work.

**Operational gap worth closing early:** the phase-2 launcher, the corpus preflight and the cohort
gate currently live only in one operator's scratch directory. They encode hard-won knowledge (the
CHIA wrapper is the entrypoint, not the coordinator; a per-run harness snapshot is required; the
member selection and the nine functional-gate waivers) and they are why this was fragile. Move them
into the repo under `merlin/experiments/gemmini_perf_bench/` before relying on them again.
