---
title: "Design: open decisions and remaining work for gemmini phase-2 performance"
kind: design
status: current
owner: core
last_verified: 2026-09-07
related: [perf_phase2_wiring, perf_corpus_scope_gap, capsule_phase_split, derived_capsule_axes, dialect_test_bar]
code_refs:
  - merlin/experiments/gemmini_perf_bench/scripts/run_agentic_perf_experiment.py
  - merlin/experiments/gemmini_perf_bench/scripts/run_global_perf_experiment.py
  - merlin/experiments/gemmini_perf_bench/scripts/run_whole_program_gsim_smoke.py
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

**Current implementation status is in §18, the completed direct-M2 result is in §22, and the
2026-09-07 four-model experiment/figure handoff is in §24.** Sections 1-17 retain the chronology
that produced the decisions, but §§18, 22 and 24 supersede their future-tense task lists and correct
the earlier batch-norm and layer-capsule assumptions.

Two things in here are **decisions, not tasks**. They are marked as such. Do not implement past
them without making the decision explicitly, because both encode a claim about what our evidence
means, and picking one silently is how a benchmark stops measuring what it says it measures.

## 1. State at the original 2026-09-06 handoff

**At the time this section was written, a phase-2 campaign was running.** Three concurrent trials,
`gpt-5.6-sol` at effort high, seeded
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

### DECISION RESOLVED: declared stratified sampling

> Must every workload in the cohort be cross-validated, or is a declared representative sample
> enough?

The waiver remains a stopgap. The implemented answer is the declared stratified sample in
`functional_coverage.py` (`minimum-per-operation-semantics.v1`). Selection is derived before output
or capture availability is observed; the consumer independently re-derives it; absent coverage
metadata means exact coverage; and foreign extras still refuse. The argument for sampling is that
the comparison is output bytes: two engines running one ELF over one elaborated design either agree
or they do not, and a 106,458-cycle capsule is not a better test of that than a 3,613-cycle one — it
is the same test, 40x more expensive.

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

The 38 certified members split into **four** campaigns.

| analyzer | families | members |
| --- | --- | --- |
| `perf_affine_claim/v1` | PM | 16 |
| `perf_paired_claim/v1` | PC, PL, PQ | 12 |
| `perf_pr_claim/v1` | PR | 6 |
| `perf_pk_claim/v3` | PK | 4 |

**Decision:** one campaign still seals exactly one analyzer claim; those four micro campaigns are
diagnostics, not one combinable model claim. The performance objective unit is a complete
layer/residual-block interval, periodically checked at whole-model scale. Do not spend three more
agent budgets merely to make a 38-member aggregate: run a micro campaign only when its mechanism is
the active diagnostic for the layer objective.

Related: `PK` requires exactly four frozen descriptors and `PR`'s `fits_double` band requires three
depths, so members cannot be dropped freely to reshape cohorts. `PK00_k16`, `PM00_m16n16` and
`PR00_fits_double_k16` are all the same 16x16x16 workload — the shared anchor of three sweeps — and
each family requires its own. That is legal and intended; `_verify_tuning_certificate` was fixed in
`f4607b26` to group by identity rather than refuse the repeat.

## 5. Original compiler work list — status superseded by §18

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

**Step 4 — realize and rank the fusion already measured.** The requant and scalar epilogue features
are registered and default-off. `requant_fuse` measures
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

* `LOOP_CONV_WS` — the target has a native convolution FSM that **no backend emits**. As §15
  corrects, `CONV2D` is accepted but lowered through shared im2col/resident-matmul/commit, not
  rejected to a host-gather recipe. Selection plus an emitter remain agent-owned.
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

## 7. Closed defect: the perf path bypassed the argv composer

`bwrap.compose_command` moves an oversized bind list into a file descriptor
(`_wrap_via_args_fd`) when the composed string would exceed the execve per-argument limit. It is
covered by `merlin/tests/infra/test_bwrap_cmd_argv_size.py`, which passes.

**The mechanism does work and is in use.** A live atlas capsule-bench run was observed executing
`bwrap --args 10 bash -c ...`, and its argument files exist — written next to the workspace, which
for that path is a scratch directory rather than anywhere under `out/runs/`.

The discrepancy is explained: the performance stage supplied its own bwrap callback with a second,
unbounded argv join. It never called `compose_command`, so the shared 32 KiB spill threshold could
not fire. That callback now uses the shared composer. A production-path regression constructs more
than 178,303 bytes of policy, proves the perf callback emits `--args`, checks the exact NUL-separated
payload, and keeps the final command below the kernel limit. A separate live-bwrap regression first
proves the inline form raises `E2BIG` and then executes the spilled form successfully.

`58ad630c` also routed the agent payload through a command file, so both the bind policy and payload
now have bounded transports.

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

**Operational gap closed:** the phase-2 launcher, snapshotter, static/live runtime preflight, corpus
preflight and cohort gate now live under `merlin/experiments/gemmini_perf_bench/scripts/`, with
regressions for the CHIA entrypoint, bounded snapshot contents, member selection and waiver flow.

---

## 9. What is engineering, not an experiment

Everything in this section is a known defect with a known cause. None of it needs a graded agent to
discover it, and none of it should be left for one: an agentic run that rediscovers our bugs measures
our ceiling, not the agent's skill. Work it directly.

The one thing that DOES belong in an agentic experiment is at the end.

### 9.0 Read this first: naive vectorization has been measured, and it lost

Before writing any pass that "vectorizes the host code", read the measured levers in
`mining/wholemodel_proposer.py`. Two of the obvious moves are recorded regressions:

| lever | measured result |
| --- | --- |
| `vectorize_non_contraction_generics` | **4.9x vector instructions, 1.28x SLOWER** |
| the four-lever epilogue stack | **+17.9% instructions, +4.4% wall** |
| `fuse_requant_into_contraction_vec` | **-5.3% instructions**, +1.6% vector instructions |

The lever that paid removes a *pass*; the ones that lost bought vector width. The host problem is
**too many passes over too much data**, not too little SIMD. Optimise for deleting intermediates and
loop trips, and treat any "add vectorization" idea as guilty until measured.

Also note a stale-number trap the source itself flags: the frequently-quoted "44.4% of int8
instructions in the scalar gather / 31.1% in quantize+amax / 18.2% in the contraction" figures were
taken on **f32-activation IR with `quantize_before_gather` OFF**, and are explicitly marked NOT
CURRENT for the shipping build. Do not cite them for the configuration we ship; re-measure.

### 9.1 Harness infrastructure

1. **Certificate sampling — DONE.** `functional_coverage.py` implements the declared,
   consumer-rederived, fail-closed stratified policy; the waiver remains explicitly weaker evidence.
2. **`compose_command` discrepancy — DONE.** The perf callback bypassed the shared composer; it now
   uses it and a >178 KB production-path test proves the args-file branch actually launches.
3. **Launch tooling — DONE.** The launcher, bounded snapshotter, static/live runtime preflight,
   corpus preflight and cohort gate are tracked under the experiment scripts directory.
4. **Campaign unit — DECIDED.** One campaign seals one analyzer claim. Micro campaigns are
   diagnostics; a complete layer/residual-block interval is the objective, with whole-model
   milestones. Do not fund three more micro claims merely to aggregate 38 members.

### 9.2 Getting work OFF the host — this is the 17x, and it is mostly deletion

Steps 1 and 2 are already landed in `99857ed5`; the deterministic status below supersedes the
original future-tense list.

1. **Direct convolution path (step 3).** The highest-value item, because it *deletes* the dominant
   scalar cost rather than scheduling it. The default lowering materialises an im2col gather —
   `kh*kw/(sh*sw)` copies of the activation, 12.25x over on ResNet conv1 — and the gather is then the
   single largest block of scalar work in the model. Above `M2M_IM2COL_MAX_ELEMS` the frontend
   already emits the true compound-affine form (`prov.conv_path=direct_contraction`) that
   materialises nothing, and `perop_blocks.CONV_ARM_FEATURE` (`conv_register_block`) already tiles
   and vectorizes exactly that form. **DONE at deterministic scope:** the lower-budget direct ResNet
   capture exists, compiler tests cover padded/strided/batched signed-i8 geometry, and whole-model
   host replay is byte-identical. Target timing remains phase-owned.
2. **Measured fusions — REALIZED and ranked, not blindly defaulted.** The safe scalar
   `fuse_epilogue_loops` path is byte-identical and removes 14.5% of small_llama static instructions.
   Targeted post-contraction fusion on direct int8 ResNet is byte-identical and removes 8.73%, with
   vector count flat. `fuse_requant_into_contraction_vec` retains the 53/53 and 44,455,936-byte
   basis. Static instruction counts nominate search candidates; target timing decides promotion.
3. **Epilogue semantics — CORRECTED.** The commit ABI speaks scalar `acc_scale`, accumulator bias,
   ReLU and maxpool, but the captured ResNet batch norm is a per-channel f32
   subtract/rsqrt/multiply/add chain after accumulator conversion. It is not semantically the
   existing scalar scale plus i32 bias. Folding it requires offline fold+requantization under a
   numeric policy or a per-channel-scale ABI extension; do not attach a synthetic epilogue and call
   the host work removed.
4. **`LOOP_CONV_WS` — OWN PLAN PREPARED.** As §15 corrects, current `CONV2D` takes the accepted
   im2col/resident-matmul/commit route. The tracked same-source comparison launcher has built and
   pinned warm+measured `ws conv` and `ws matmul` ELFs under
   `out/artifacts/perf-bench/gemmini/handoff_20260906/loop_conv_ws_prepared/`. FireSim pricing and the
   native emitter/schedule remain phase-2 agent work.

### 9.3 The genuine host/scalar residue

What is left on the host after 9.2 is real host work and should be good: residual add (not
expressible in `EPILOGUE_STAGES` — it needs a second full tensor operand), softmax, normalisation,
and layout ops.

1. **Quantization/vectorizer interaction — DONE.** `named_int8_contraction` preserves canonical
   mixed-type matmuls for the named transform schedule and is coupled to refinements that use it.
   Non-canonical forms remain generic rather than being mislabeled. The real-bundle regression
   proves the former 15 -> 0 loss and restoration to 15.
2. **Weight transpose hoist — DONE through offline prepacking.** A general map fold is retained but
   refuses the harmful hot-axis stride flip it discovered. `prepack_weight_layout` removes the
   runtime transposes and is bit-identical on full-output Spike replay (2,090,995 -> 1,993,040 Spike
   instruction-model cycles).
3. **Delete scalar passes — DONE as searchable levers.** Safe affine epilogue fusion and targeted
   post-contraction fusion preserve output bytes and reduce static instructions; neither is promoted
   without target timing.
4. **Vector scratch lifetime — DONE.** A late buffer-hoisting stage moves vector-to-SCF scratch out
   of contraction loops. Full ResNet now executes under the normal host stack, and a CFG regression
   rejects loop-local LLVM allocas.
5. **Do not** add a general "vectorize the generics" pass — see 9.0, it measured 1.28x slower.

### 9.4 What should stay an agentic experiment

The **~6x on-mesh inefficiency**: tiling, dataflow, loop order, residency for work already on the
accelerator. That is a search over schedules against real measurement, which is what an agentic loop
is genuinely good at and what AutoComp does well. It is what the running PM campaign is aimed at.

The deterministic items in 9.1-9.3 are now implemented or explicitly blocked at the ownership
boundary recorded in §18. Do not turn the remaining tiling/dataflow/residency choices into a shared
compiler answer key; those are the phase-2 search surface.

### 9.5 Concurrency warning

As of 2026-09-06 another session is actively editing `merlin/python/merlin/llvmlower/` —
`perop_blocks.py`, `impr_features.py` and a new `quant_hoist.py` are modified or untracked in the
shared working tree. Steps 9.2.1 and 9.3.1 touch exactly those files. Check `git status` and
coordinate before editing; commit with explicit pathspecs.

---

## 10. THE BLOCKER: why the phase-2 campaign produced no result

Measured 2026-09-06 on `perf_agentic_20260906T074702Z_pm5`, all three trials. The campaign launched,
Codex ran, the agents worked for many turns and produced candidates -- and every trial was discarded.
The chain, end to end:

1. The agent called `tuning-gsim-feedback` **12 times**. **11 returned rc=125**; one succeeded.
2. `rc=125` is the host-owned evaluator refusing (`perf_agent_stage.py:3588`). The agent is shown only
   the exception TYPE, so from inside the loop the failure is nearly opaque. The reason is written
   host-side to `<work_root>/host_refusals/round_N_call_M.txt`, and it is the same every time:

   ```
   StageGateError: development GSIM baseline measurement failed:
     StageGateError: development GSIM baseline/PM/PM00_m16n16 failed strict certificate admission
   ```

3. `PM00_m16n16` is the FIRST member of the PM cohort, so every sweep dies on the first member and no
   sweep ever completes.
4. Strict admission is `perf_agent_stage.py:503-509`, four clauses: `qualification.admitted is True`,
   `decision` is a mapping, `selected_engine == "gsim"`, and
   `qualification["decision"]["certificate_sha256"] == decision.certificate_sha256`. The last is the
   discriminating one -- the certificate digest recorded IN the measurement must equal the one the
   stage decided with.
5. Because only one feedback ever succeeded, it evaluated candidate `f3e275688d3642f17ab8...` while
   the agent's final sealed bytes were `4b262afacd15456f44e3...`. The seal gate then refused, correctly,
   with `mandatory tuning GSIM feedback did not evaluate the final round candidate bytes`, and the
   coordinator reported `performance candidate lacks immutable broker registry/receipts`.

**This is a harness defect presenting as an agent failure -- the twelfth recorded instance of that
pattern.** The agent did the right thing twelve times. The harness refused eleven, then discarded the
run for not having measured its final bytes. Nothing in the NO-GO message says any of this; recovering
it took reading the receipts, decoding rc=125, and finding the host-private refusal file.

### What to do

* **Find why `PM00_m16n16` fails clause 4.** Compare the certificate digest the measurement path
  records against `decision.certificate_sha256`. Two certificates are in play at launch (the v5 tuning
  certificate passed as `--gsim-certificate`, and whatever the runner resolves), and the pinned GSIM
  binary `MERLIN_GEMMINI_GSIM_EMU` must be the binary the ACTIVE certificate pins -- the launcher's own
  comment records that a stale export makes every call rc=125 while the compile stages still pass.
* **Suspect the shared workload identity.** `PM00_m16n16`, `PK00_k16` and `PR00_fits_double_k16` are
  all 16x16x16 and share one `workload_sha256` (§4). The certificate holds ONE member for that identity.
  If admission resolves a member by identity and then checks something capsule-specific, the two
  capsules that are not the certificate's representative will fail while the third passes -- which
  matches "the first PM member fails" exactly. Verify before assuming the digest mismatch.
* **Make the preflight check the REAL predicate.** `perf_preflight_corpus.py` verifies members with
  `plan_evaluation(...)` and passed all 38 while the runtime's strict admission refuses PM00. A
  preflight that green-lights a corpus the runtime then rejects is worse than none: it moved the
  failure from launch time to four hours in. Call the same admission the runner calls.
* **Surface the refusal.** rc=125 should carry the refusal digest into the stage error, so a run does
  not have to be autopsied through `host_refusals/` to learn that its measurements never ran.

### Cost of this defect

Three concurrent Codex trials at effort high, roughly forty minutes of agent time each, producing zero
measurements and zero verdicts -- and the run reads as "the agent failed to measure its candidate".

### 10.1 Ruled out (do not re-investigate)

Measured 2026-09-06 against the v5 tuning certificate. Three plausible causes are eliminated:

* **The pinned GSIM binary is correct.** `MERLIN_GEMMINI_GSIM_EMU` hashes to `1a3de02a19ddcb470444`,
  which is exactly the `gsim_binary` pin the v5 certificate carries. The launcher's documented
  stale-export hazard is not this.
* **The certificate envelope is fine.** All 16 PM members are admitted under BOTH phases the code
  uses -- `development_correctness` (the stage, `perf_agent_stage.py:2993`) and `final_performance`
  (the measurement path, `run_paired_perf_bench.py:382`). 16/16 in each.
* **The shared 16x16x16 workload identity is not implicated.** `PM00_m16n16` is simply the FIRST
  member measured, so it is the first to fail; that is not evidence it is special. Static admission
  accepts it.

So the refusal is raised by `GATE.validate_execution(certificate, decision, evidence["gsim"])` at
RUN time, on evidence produced by the execution -- not by the static envelope.
`run_paired_perf_bench.py:797-806` captures that reason into
`measurement["failure"]["detail"]` (plane `gsim_qualification`, category `infra_refusal`), and
`_redact_execution` then discarded it. That discard is now fixed: the reason is appended to the stage
error (host-side only -- the agent still sees just the exception type through rc=125).

**Next step is a single measurement, not a campaign.** Run one paired measurement of `PM/PM00_m16n16`
against the v5 certificate with the pinned binary and read `result["gsim_qualification"]["reason"]`.
That names the failing execution invariant directly. Do NOT relaunch a three-trial campaign to find
out -- that is what cost three agent-hours for zero measurements.

### 10.2 The search is narrowed to one field (2026-09-06, second pass)

Reading how the execution record is BUILT (`run_paired_perf_bench.py:689-706`) eliminates most of
what `validate_execution` can refuse, because most of the fields it checks are not observations:

```python
"status": "pass",                                              # literal
"cycle_accurate": True,                                        # literal
"binary_sha256":  certificate.pins["gsim_binary"]["sha256"],   # copied FROM the certificate
"firrtl_sha256":  certificate.pins["gsim_firrtl"]["sha256"],   # copied FROM the certificate
"model_sha256":   certificate.pins["gsim_model"]["sha256"],    # copied FROM the certificate
```

`validate_execution` then compares those three pin fields against the same certificate they were
copied from, so **those three checks cannot fail** -- worth fixing on its own account, since they read
as verification and verify nothing.

What remains variable is:

* `derived_from_rtl` -- taken from the oracle the run actually used;
* `elf_sha256` -- computed from the emitted ELF;
* `cycles` -- only constrained in the `final_performance` phase;
* `decision.certificate_sha256` vs `certificate.sha256`.

`derived_from_rtl` is `True` unconditionally on the elaborated-RTL branch
(`program_oracle.py:998`), and that branch RAISES rather than returning False when the program does
not halt or the readback is short. The only path that yields `False` is the FUNCTIONAL oracle
(`program_oracle.py:537`), which would mean the measurement did not run on an RTL engine at all --
despite `MERLIN_REQUIRED_RTL_ENGINE=gsim` being exported by the launcher.

**Ruled out in this pass as well:** disk exhaustion. It was a plausible correlation -- eight 14 GB
harness snapshots filled `/scratch` to 100% while the campaign ran, and the one measurement that
succeeded was the one before the fill -- but the run tree contains no `ENOSPC` or "no space left"
evidence anywhere. Do not pursue it without new evidence.

So the question is now narrow: **did the development measurement run on the RTL engine, or did it
fall back to the functional oracle?** Answer it by running one measurement and printing
`evidence["gsim"]["oracle"]`, or by reading the refusal reason, which `566d9213` now appends to the
stage error instead of discarding.

---

## 11. The corpus measures the wrong axis (evidence-grounded, 2026-09-06)

### 11.1 What the six families actually lever

| family | declared lever | members | matches a MEASURED defect? |
| --- | --- | ---: | --- |
| PQ | `redundant_synchronization_removal` | 6 | **YES** -- the fence defect |
| PR | `operand_residency` | 6 | **YES** -- false residency / weight reloading |
| PC | `dma_issue_before_wait` | 2 | yes -- CPU/accelerator overlap |
| PL | `cross_regime_amortization` | 4 | partly |
| PK | `reduction_depth` | 4 | the k-axis that bounds utilization |
| PM | `parallel_extents` | 16 | **no** |

The first phase-2 campaign ran **PM**, chosen because it was the largest single-claim cohort. That is
the one family whose lever corresponds to no defect in the ResNet-50 evidence.

### 11.2 Why PM cannot show a win, measured

Every PM member holds **k = 16** (MACs = m*n*k; PM15 is 65,536 = 64*64*16). The family sweeps the two
extents that do NOT set the ceiling on a weight-stationary mesh, where utilization is governed by how
much work each loaded weight tile amortizes -- i.e. by k.

Measured over the 16 members in one round:

```
PM00   4,096 MACs -> 14.3 MACs/cyc ( 5.6% of 256 peak)
PM05  16,384      -> 31.6          (12.3%)
PM10  36,864      -> 41.7          (16.3%)
PM15  65,536      -> 42.8          (16.7%)
TOTAL 409,600 MACs in 11,451 cycles = 35.8 MACs/cyc (14.0% of peak)
```

A two-point fit gives roughly **200 cycles of fixed per-capsule overhead** and a **marginal rate near
49 MACs/cycle**, so even an unboundedly large PM member tops out near 19% of peak on this schedule.
The agent moved the total from 11,487 to 11,451 cycles -- 0.3% -- with several members regressing.
That is close to all the room the family has.

**A methodological defect follows.** `attainable_total_cycles` (2,873) derives from
`achievable_macs_per_cycle` = 142.5, described as "best rate over 97 measured points", 81 of which
come from the phase-1 functional run -- different shapes, including deep-k ones. At k=16 that rate is
unreachable, so `headroom_open` stays open regardless of what the agent does, and
`attainment_blocked_by: []` asserts nothing is preventing attainment, which is false. The achievable
basis must be restricted to points whose shape the member could actually resemble.

### 11.3 What the corpus cannot see at all

`linalg.generic` accounts for **9,689,674,051** of the 11.18B-cycle ResNet-50 interval. All 38 members
are single on-mesh operations, so nothing in the corpus exercises the work that dominates:

* layout conversion, repacking, `tensor.insert_slice`, memref descriptor handling;
* device-dispatch shims and repeated device-resource transitions (2,988 profiled marks);
* epilogue fusion across a layer -- batch norm, quantization, activation, residual lowered separately;
* allocation / copy / pack / evict traffic;
* CPU<->accelerator round trips at LAYER scale, not single-op scale.

The reference point for what good looks like is an Exo-generated kernel for `16x8208x16`: five
`config_*` calls issued ONCE, then a tight `mvin` / `preload` / `compute_preloaded` loop with the
accumulator resident across all 513 k-steps, and a single `mvout`. No per-tile reconfiguration, no
fences, no reallocation. That single artifact is simultaneously the PQ claim, the PR claim, and an
encoding-efficiency axis the corpus does not have.

### 11.4 Axes the corpus needs

1. **tile / capacity** -- shapes that exceed scratchpad and accumulator capacity. Evidence: large
   matrices address memory outside Gemmini's scratchpad/accum, and a schedule that "claims to reuse
   resident weights is actually reloading them". PR levers residency but only at single-op scale.
2. **layer and fused-layer** -- a whole conv/BN/ReLU/residual block, where epilogue fusion and layout
   conversion are visible. Entirely absent today.
3. **intra-level scheduling** -- issue order, double buffering, overlap of DMA with compute. PC has 2
   members.
4. **CPU/accelerator interaction at scale** -- fence and dispatch cost. A 1024x1024 QK slice issued
   **4,096 fences**, reduced to **2** with identical arithmetic, tiling, addresses and commands. PQ is
   the right lever; it needs members large enough for the effect to dominate.
5. **command encoding efficiency** -- config hoisting, command count per unit of work. Absent.

### 11.5 The comparison to report

The 2B artifact is a **Gemmini library baseline**: an existing `resnet50.c` making 124 direct calls to
prebuilt `tiled_*_auto` routines (18 `tiled_conv_auto`, 2 `tiled_conv_downsample`, 87
`tiled_matmul_nn_auto`, 16 `tiled_resadd_auto`, 1 `tiled_global_average_auto`), compiled with
`-O2 -ffast-math`. Those routines supply tiling, loop order, DMA scheduling, accumulator management
and fusion. The 11B artifact is an **automatic OOT compiler result** from a canonical PyTorch capture.

**8.58x on comparable FireSim terminal counters is real and is the number to beat, but it is not
evidence that the older compiler generated better schedules** -- it did not generate them at all. A
fair comparison requires both arms to start from the same captured graph, and either both or neither
may call `tiled_*_auto`. Until then the two should be labelled as what they are.

### 11.6 Where the residency defect actually lives, and a tool that cannot see it

The emitted command buffer for `PM15_m64n64` is four commands and expresses residency CORRECTLY:

```
[0] RES_PACK          W -> W_res        layout=packed_rhs
[1] MATMUL_RESIDENT   A0 x W_res -> acc0
[2] COMMIT            acc0 -> Y0        epilogue=[]  output_dtype=i32
[3] EVICT             W_res
```

So "claims to reuse resident weights but is actually reloading them" is NOT a command-buffer defect.
The buffer asserts residency; the reload happens when `MATMUL_RESIDENT` is lowered to
`mvin`/`preload`/`compute_preloaded`. Look there, not at the buffer.

**`merlin/python/merlin/perf/movement_volume.py` cannot detect this.** It charges a resident operand
ONCE at its pack -- that is its documented rule. If the backend re-loads the tile per output tile, the
module still reports a single charge and residency reads as achieved. It measures the intent encoded
in the command buffer, not the traffic actually issued, so on precisely this defect it cannot fail.
That is the repository's recurring "a check that could not fail reported success" shape, in a module
added on 2026-09-06.

**The check that would catch it:** decode the RoCC trace and count `mvin`s targeting the weight region
against the number of `RES_PACK` commands. One pack must mean one load. The Exo reference for
`16x8208x16` issues its weight loads inside the k-loop but keeps the ACCUMULATOR resident across all
513 steps with a single `mvout`; the comparison to make is loads-per-pack and mvouts-per-output, both
countable from the decoded stream without a simulator.

---

## 12. STATUS CORRECTION: much of §9 is already being built (2026-09-06)

Several items this document lists as open were, at the time of writing, being implemented in the same
working tree by another session. Do not restart them; read them.

* **DECISION 1 (§3 / §9.1.1) -- the certificate sampling design is IMPLEMENTED**, in
  `merlin/experiments/gemmini_perf_bench/scripts/functional_coverage.py`
  (`POLICY = "minimum-per-operation-semantics.v1"`). Reviewed against the four properties §3 declares
  non-negotiable, and it satisfies all four:
  1. the selection is pre-registered -- `derive` chooses "without reading outputs or capture
     availability", so it cannot degenerate into whatever finished in time;
  2. the consumer re-derives rather than trusting the claim -- `verify` calls `derive` on its own
     cohort and demands `coverage == expected`;
  3. absent coverage metadata reads as EXACT -- `coverage is None` requires `members == full` and
     returns `mode: "exact"`, so an older certificate cannot silently acquire a weaker meaning;
  4. `extras` still refuses, checked BEFORE any mode branch, so it binds in both modes.
* **Snapshot bloat (§9.1.2) is addressed** by `perf_snapshot.py`, whose `OMIT` set excludes
  `__pycache__`, `_data` and `_qa_ws`. That is the defect that put 14 GB into every harness snapshot
  and filled the filesystem on 2026-09-06.
* **§9.3 host work is in progress** as `host_codegen_ab.py` -- "compare work-deletion features on one
  captured host model; never claim accelerator timing", which is the correct framing per §9.0: the
  levers that pay DELETE work rather than add vector width.

`impr_features.py`, `perop_blocks.py`, `wholemodel_proposer.py`, `functional_gsim_qualification.py`,
`run_agentic_perf_experiment.py` and `_pbcommon.py` were all mid-edit at that time. **Check
`git status` before touching anything under §9.2 or §9.3**, and expect the infra suite to show
unrelated failures while that work is in flight -- 142 failures and 30 errors on 2026-09-06 were all
`perf_campaign has no attribute 'verify_functional_host_lane_snapshot'`, none of them regressions.

---

## 13. §10 ANSWERED: the measurement never ran; the gate blamed the certificate

The diagnostic §10 asked for was delivered by the run itself once `566d9213` stopped discarding the
refusal reason. Every failing measurement in the `pm6` campaign records:

```
StageGateError: development GSIM {arm}/PM/{capsule} failed strict certificate admission:
  GsimGateError: eligible development_correctness evaluation must use GSIM, not None
```

`engine` is **None**. That value comes from `execution.get("engine")` where `execution` is
`evidence.get("gsim", {})` -- so the dict was EMPTY. `_gsim_l3_adapter` populates `evidence["gsim"]`
only when the L3 tier actually executes, therefore **the GSIM tier did not run for that cell**.

Nothing was wrong with the certificate. The message "failed strict certificate admission" is the
wrong sentence for this condition; it should say that no GSIM execution evidence was produced.

**This closes several hypotheses.** The failure is not capsule-specific (PM02, PM00, PM00), not
arm-specific (candidate, candidate, baseline) and not position-specific (call_0, call_0, call_4). The
"PM00_m16n16 is special" reading was wrong -- PM00 was simply first in the earlier campaign.

**And it gives the disk explanation a mechanism rather than a correlation.** If the L3 run cannot
write its workdir, no `evidence["gsim"]` is produced and `engine` is None -- exactly this signature.
The observed rates fit: **11 of 12 measurement calls failed with 9.6 GB free; 1 of 6 fails with
600 GB free.** That is consistent, not proven; the adapter's own exception path should be read to
confirm whether a write failure leaves the evidence dict empty rather than propagating.

### What to fix

1. **Name the condition.** `_redact_execution` should distinguish "no GSIM execution evidence was
   produced" from "the certificate refused this workload". They have different causes and different
   remedies, and the current message sent an investigation down the certificate path for hours.
2. **Refuse the round, not the campaign.** One cell whose tier did not run currently costs the whole
   feedback call, and eleven such calls cost the entire campaign -- the run was ultimately discarded
   for "not measuring its final candidate bytes" when the real event was transient measurement loss.
3. **Surface the tier outcome.** A cell that skipped L3 should say so in the measurement record, so
   an empty `evidence["gsim"]` can never be read as an engine disagreement.

---

## 14. The next campaigns are validated and ready to fire

The §11 finding is that PM levers the one axis the ResNet evidence does not implicate, while PQ
(synchronization removal) and PR (operand residency) lever the two defects that were actually
measured costing cycles. Those scopes have now been validated end to end, so choosing one is a
decision rather than a debugging session.

| scope | members | claim | cohort preflight | certificate envelope |
| --- | ---: | --- | --- | --- |
| PR | 6 | `perf_pr_claim/v1` -- operand residency | READY 6/6 | 0 outside |
| PQ | 6 | `perf_paired_claim/v1` -- redundant synchronization removal | READY 6/6 | 0 outside |
| PK | 4 | `perf_pk_claim/v3` -- reduction depth | READY 4/4 | 0 outside |

Each is a SEPARATE campaign: one campaign seals one claim (§4), so these cannot be combined.

Recommended order, on the evidence:

1. **PR** -- directly measures "claims to reuse resident weights but is actually reloading them".
   Read §11.6 first: the defect is below the command buffer, and `movement_volume` cannot see it.
2. **PQ** -- directly measures the fence defect (4,096 fences for one 1024x1024 QK slice, reduced to
   2 with identical arithmetic and commands).
3. **PK** -- the reduction-depth sweep, the only family that varies the dimension bounding
   utilization on a weight-stationary mesh. Useful for a ceiling, less so for a defect.

Not launched here: each costs three concurrent Codex trials, and the scoping choice changes what the
experiment measures. That is an owner's decision, not a default.

---

## 15. LOOP_CONV_WS verified, and a correction to §9.2

The native convolution FSM was asserted earlier in this document on recollection. It is now checked
against the repository, and the substance holds while one detail was wrong.

**Confirmed:**

* `merlin/targets/gemmini/contracts/rtl_facts/facts.json` -- the interface's `legal_funct` list
  contains `15` and `16..21`. The conv FSM and its six CONFIG functs are RTL-attested legal.
* `merlin/python/merlin/_data/contract/compute_endpoints.yaml` -- the gemmini endpoint already
  declares `LOOP_CONV_WS` and `LOOP_CONV_WS_CONFIG_1` through `_6` under the `loop_descriptor` role,
  alongside the `LOOP_WS` family. The contract knows these commands exist.

**Corrected:** this document previously said `CONV2D` was rewritten into a "host-gather recipe" and
had been a schema `protocol_violation`. That is not what the backend does.
`gemmini_codegen_mlir.py:181` states it plainly: *"Lower a whole-op CONV2D to the same
im2col/resident-matmul/commit path used by matmul capsules"*, and the module header adds *"CONV2D
uses the same backend after a shared im2col materialization"*.

So `CONV2D` is ACCEPTED and lowered onto the device -- just by the expensive route. The cost is the
im2col materialization, which writes the activation `kh*kw/(sh*sw)` times over (12.25x on ResNet
conv1), not a fallback to the host.

**What this makes the work.** The FSM is legal, declared, and never selected; the alternative path is
implemented and always selected. So this is a SELECTION problem plus an emitter, not a bring-up from
nothing: the loop-descriptor commands have declared roles to fill, and the decision to use them has to
be made where the conv lowering currently commits to im2col. That is a smaller and better-grounded
task than "emit an unused ISA", which is how §9.2 previously framed it.

Sequencing note: this overlaps §9.2.1 (the direct-conv path, which also removes the im2col
materialization). Do not pursue both independently -- they are two ways of deleting the same
intermediate, and the direct-conv arm already exists and is default-off, while the FSM path needs an
emitter. Price the existing arm first.

---

## 16. pm6 COMPLETED: the loop closes, and §11.6 was wrong

The `pm6` campaign ran end to end -- the first phase-2 campaign to do so.

**trial_00 sealed a clean candidate.** `functional_guard.offenders == 0`, refusal `None`.
GSIM cycles 11,487 -> 11,469 (18 cycles, 0.157%), PM00 304 -> 286. Small, for the structural reason
in §11.2, but it is a complete and admissible phase-2 result.

**trial_01 and trial_02 were refused, correctly, and for the RIGHT defect:**

```
capsule A6_resident_reuse -- trace_findings_introduced
  "mode resident_reuse: 1 redundant load(s) rewrite an on-chip destination
   that already held that exact source"
```

Both agents, while optimising, introduced a redundant reload of an operand already resident on chip.
That is exactly the defect class reported from the ResNet-50 run -- "claims to reuse resident weights
but is actually reloading them" -- caught automatically, before sealing, on a capsule built to
exercise it.

**§11.6 CORRECTION.** That section said residency "must be verified from the decoded instruction
stream, not from this block", implying the check did not exist. It does:
`merlin/python/merlin/targetgen/trace_check.py:244`, invoked as `TCK.check` from the certified
functional emission guard (`perf_agent_stage.py:2807-2820`), which diffs candidate findings against
baseline findings and refuses anything introduced. It is covered by
`merlin/tests/targetgen/test_resident_reuse_configures_once.py`.

So the correct statement is narrower and more useful: `movement_volume` cannot see a reload (that part
stands, and §9.0/§13 keep it), but the REPOSITORY can -- through the trace checker, which is already
wired into the phase-2 seal path and demonstrably works.

**What this means for the campaign order in §14.** PR (operand residency) is still the right next
scope, and this result strengthens it: the trace checker gives the seal path a working detector for
exactly what PR levers, so a PR campaign can distinguish a real residency win from a candidate that
merely claims one.

---

## 17. How to set up a perf experiment that can close the ResNet-50 gap

The PR/PQ/PK campaigns launched on 2026-09-06 are MICRO families. They diagnose mechanisms --
operand residency, redundant synchronization, reduction depth -- on single on-mesh operations. They
cannot close the model gap and must not be reported as if they could: `linalg.generic` is
9,689,674,051 of the 11,183,959,730-cycle interval, and every one of the 38 members is a single op.

### 17.1 The unit has to change from a capsule to a layer

The 8.58x lives BETWEEN operations (layout conversion, `tensor.insert_slice`, packing, memref
descriptors, device-dispatch shims) and ACROSS them (epilogue fusion). No sum of single-op capsules
contains it.

Whole ResNet-50 is 11.18B cycles and therefore FireSim-only -- unusable as an iteration loop. A
RESIDUAL BLOCK remains the right **objective unit** because it contains im2col materialization,
epilogue fusion, layout conversion, the host-only residual add, and CPU/accelerator transitions.
It is not yet a runnable ordinary capsule: the current runner executes the accelerator under the
target oracle and host work on this workstation, so there is no one counter for the interval. The
`PB` family records the required `fused_single_elf` runner as `blocked_unimplemented`.

| objective | what it contains | readiness |
| --- | --- | --- |
| stem: 7x7 conv + maxpool | worst im2col amplification -- activation written 12.25x over | fused runner needed |
| conv + real BN + ReLU | per-channel f32 affine chain, epilogue fusion, layout conversion | numeric policy + fused runner needed |
| residual block | second full tensor operand, cross-op liveness | residual ABI + fused runner needed |
| downsample block | stride/shape change, repacking | fused runner needed |
| classifier tail: GAP + FC | rank legality and layout | fused runner needed |

Keep PR/PQ/PK as DIAGNOSTICS that explain why a layer member is slow. They are the mechanism; the
layer is the objective.

### 17.2 The objective is the interval, with its decomposition attached

Score the layer's total cycles, and report alongside it where those cycles went: on-mesh,
`linalg.generic`, movement, synchronization. The decomposition pieces exist -- `op_profile`,
`work_volume`, `movement_volume` (subject to §13's limitation), `decompose_corpus`, counter
observations and `trace_check` -- but the fused same-counter execution seam does not. Without that
seam, adding the fields to a capsule would produce a plausible-looking number over mismatched clocks.

### 17.3 Deterministic levers are now realized; promotion remains measured

The direct-conv recapture and arm, canonical int8 naming, safe scalar fusion, targeted
post-contraction fusion, offline weight prepacking, rank legality, one-time config, fence retirement
and loop-scratch lifetime all have regressions. The requant census still finds 53/53 opportunities.
These are now usable search candidates, not automatically enabled defaults: the evidence repeatedly
shows that a clean static reduction can lose on target. Per §9.0, prefer work-deletion levers; a
blanket generic vectorizer measured 1.28x slower.

### 17.4 The baseline must make it a compiler comparison

Both arms start from the SAME captured PyTorch graph, and either both or neither may call
`tiled_*_auto`. The existing 2B-vs-11B pair does not satisfy this: the 2B artifact begins from an
already-generated `resnet50.c` making 124 direct library calls. Comparing it to an automatic
lowering measures library-versus-compiler, not schedule quality.

Gemmini's own `resnet50.c` supports this directly: it takes `os|ws|cpu` and `conv|matmul`, and prints
Total / Matmul / Im2col / Conv / Pooling / Depthwise / ResAdd / Other with percentages. Running it in
`matmul` mode gives an im2col-based library arm directly comparable to our lowering, and the
`conv`-vs-`matmul` delta prices `LOOP_CONV_WS` with no work on our side. The tracked
`resnet50_library_baseline.py` launcher has now executed all three arms from one pinned source
snapshot. Three post-warm-up repetitions per arm were cycle-identical: `ws conv = 152,513,710`,
`ws matmul = 1,565,712,935`, and `cpu matmul = 2,148,409,928`, so native conv is **10.266047x**
faster than the hand-written im2col/matmul mode. The queue-owned nine-job receipt is
`out/artifacts/perf-bench/gemmini/resnet50_library_baseline_warm_20260906/results.json`; §19 of
`closing_the_gap.md` records the complete decomposition and measurement boundary. This prices the
native-conv opportunity but is not yet a Merlin compiler result.

### 17.5 Two oracle tiers, never mixed

GSIM is the intended reduced-layer iteration oracle **after** the fused same-counter runner exists;
FireSim is the model-scale periodic citable claim. Until then, GSIM remains a device-only diagnostic.
Counter scopes must not be crossed -- the source report notes that comparing an 11B model counter
against a 2.35B FireSim terminal counter mixes scopes. State which counter every number is.

### 17.6 Correctness gate at layer scale

Each candidate must match `golden_w8a8.independent.npy` from
`out/artifacts/recaptures/resnet50_v1_5_int8_w8a8_consistent/`, never the consistent golden alone.
A faster wrong layer is not a result. The functional emission guard already refuses candidates that
introduce trace findings, and it works: it caught two pm6 candidates introducing a redundant load
against an already-resident destination. The current certified functional sample remains 27/28 due
to a frozen-seed bias-ABI defect; correcting that seed is phase-1 ownership, not permission to edit a
graded submission or weaken this gate.

## 18. Authoritative implementation and ownership status

This section is the current answer to "is the plan finished?" It supersedes future-tense wording in
the chronology above. The answer is **no for the end-to-end performance plan**, and **yes for the
deterministic infrastructure/compiler work assigned to this checkout**, subject to the explicit
exceptions below. A prepared arm, an instruction-count reduction, or a routing plan is not a target
performance result.

| item | status | evidence / remaining owner |
| --- | --- | --- |
| Stratified functional certificate policy | **implemented** | `functional_coverage.py`; selection is predeclared, independently re-derived, exact-by-default, and rejects extras |
| Oversized `compose_command` discrepancy | **closed** | the perf callback bypassed the shared composer; production-size and live `bwrap --args` tests now cover the actual path |
| Repository launcher, immutable source snapshot, live runtime preflight, cohort gate | **implemented** | `perf_snapshot.py`, `perf_suite.py`, and the existing campaign scripts now seal and verify the run inputs before authoring or launch; new candidate records name the actual paired runner, while immutable schema-v3 records with the former consumer name remain readable |
| Campaign unit | **decided** | one campaign seals one analyzer claim; PR/PQ/PK/PM are mechanism diagnostics, while a residual block or complete layer interval is the objective |
| Direct convolution without materialized im2col | **implemented and functionally checked** | direct named/generic paths, padded geometry, stride 1/2, signed i8 and batch 2 have regressions; the recapture records `direct_contraction` provenance |
| Requant/scalar epilogue deletion | **implemented as measured candidate, not promoted blindly** | the current-compiler targeted ResNet host A/B is byte-identical and reduces 136,657 to 124,730 instructions (-8.73%); target timing still decides promotion |
| Quantization-to-vectorizer coupling | **implemented** | canonical named int8 contractions survive quantization and feed the contraction schedule; vector-presence checks remain fail-closed |
| Compile-time weight layout | **implemented on the safe path** | offline prepacking is content-addressed and Spike measured 2,090,995 -> 1,993,040 cycles with identical output; a harmful axis fold is refused |
| Scalar/RVV scratch lifetime | **implemented** | a late buffer-hoisting stage removes loop-local vector scratch growth; full ResNet replays under an 8 MiB stack with unchanged output |
| Fence, config, rank and residency checks | **implemented** | the driver retires once, resident execution configures once, rank-3 legality is covered, and decoded traces reject false residency |
| `LOOP_CONV_WS` comparison | **measured; compiler emitter/selection still phase 2 owned** | three warm-plus-measured FireSim repetitions per arm are cycle-identical; `ws matmul / ws conv = 10.266047x`. The result prices the library mechanism, while emitting/selecting it from Merlin and measuring the same-source compiler arm remain agent work |
| Fused same-counter layer runner (`PB`) | **generic ABI/harness implemented; focused boundary measured, layer objective still open** | `kernel_abi.kind=whole_program` supplies a candidate-independent explicit pointer boundary, intermediate role, warm whole-kernel window and fail-closed validation. `pb_formal_scalar_regred_v4_20260906` passes 8/8 exact PB00--PB03 grades and establishes the island/no-island boundary delta at both K values; this validates canonical saturating-requant deletion, not a residual-block or model speedup. A predeclared real layer/residual corpus and hidden variants remain phase-2 work before an end-to-end claim. |
| Real BatchNorm / residual offload | **not implemented; phase 2 compiler/backend owned** | ResNet BN is a per-channel f32 affine chain, not the scalar epilogue already supported; residual add needs a second full-tensor operand ABI |
| Functional cohort 28/28 | **not complete; phase 1 seed owned** | 27/28 is the honest result; the remaining bias-ABI defect is in the frozen seed and must not be repaired by editing the graded submission |
| On-mesh tiling/dataflow/residency search and target campaigns | **authoring ran; no admitted end-to-end claim; phase 1/2 agent owned** | PM/PR/PQ agents exercised the search surface, but no later run has completed the predeclared all-trial, tuning-plus-held-out campaign needed for a citable phase-2 result |

The ownership boundary is deliberate. Shared code supplies the legal candidates, trustworthy
measurement plumbing and fail-closed checks. It does **not** encode the schedule, loop-FSM selection,
dataflow or residency answer the graded agent is supposed to discover. Consequently no phase-2
performance number should be cited until that phase emits and measures its own artifact through the
sealed runner.

## 19. Post-handoff PQ campaign result and final open gate

`perf_agentic_20260906T120643Z_pq2` is terminal **NO-GO**. This is not a target-performance result.
The campaign predeclared `all_trials_all_cells_no_best_of_no_drop`; one of its three authoring stages
was refused, so the coordinator correctly did not continue by selecting the two surviving trials.
Doing so after seeing their feedback would turn the campaign into an undeclared best-of experiment.

The refusal was concrete and correctly detected. `trial_00` copied
`submission/mlir_oot/lowering/schedule.py` to `/tmp/arm4_schedule_variant_a.py`; the combined
transcript audit recorded `candidate_code_copied_outside` at line 55. Its final functional guard was
otherwise clean and its final tuning-feedback receipt matched the sealed bytes, but those facts do
not waive the isolation violation. `trial_01` and `trial_02` sealed consumable, clean candidates, but
their author-visible tuning deltas are not held-out campaign claims and must not be cited as one.

The shared prompt now states the already-enforced protocol explicitly: all source, executable and
scratch variants remain inside the Arm4 submission, including comparison copies; temporary variants
belong under `submission/performance/`. This is a measurement-integrity instruction, not a schedule
hint. The prompt, stage and coordinator regressions pass after the change.

A fresh campaign through `perf_suite.py` is gated before agent spend on a non-waived functional GSIM
certificate. Re-deriving the implemented stratified policy on the current cohort selects the known
`SY_epilogue_bias_add` workload, whose frozen-seed bias ABI is the remaining failed qualification.
Do not exclude it as "declined" or weaken the suite back to the waiver. Phase 1 must repair and seal a
new seed; phase 2 may then prepare a fresh immutable suite and resume the schedule search. This is the
remaining ownership boundary, not unfinished deterministic scalar/compiler plumbing in this tree.

## 20. Post-handoff PK campaign result: the residency guard refused every trial

`perf_agentic_20260906T125936Z_pk2` is terminal **NO-GO**. The Chia completion receipt records
`status: failed`, return code 2 and 1,917.189 seconds wall time. This campaign used the old functional
certificate waiver, never reached held-out evaluation, and produced no target-performance claim.

All three predeclared authoring stages sealed a candidate, and the host-owned full functional
emission guard refused all three for the same independently detected regression:

```
capsule A6_resident_reuse -- trace_findings_introduced
  mode resident_reuse: 1 redundant load(s) rewrite an on-chip destination
  that already held that exact source
```

The records are
`out/runs/gemmini/perf-bench/agent_stages/perf_agentic_20260906T125936Z_pk2__trial_00/performance_candidate.json`
through `__trial_02/performance_candidate.json`; each has `state: refused`,
`functional_guard.status: offending`, and `admission.consumable: false`. The terminal orchestration
receipt is
`out/runs/gemmini/gemmini-perf-bench/perf_chia_20260906T125936Z_pk2/chia/completion_receipt.d177ebdb4631e72191c0e4f3a9aae890437465746eee334c2a66b2768227d374.json`.

This repeats the useful finding from `pm6`: the decoded-trace residency check is live in the seal
path and catches candidates that appear faster on the tuning cells by re-materializing supposedly
resident work. Do not quote the agents' author-visible tuning deltas. The predeclared selection was
all trials with no dropping, every trial is non-consumable, and there is no campaign manifest to
support a performance statement. A fresh PK run is not the next action; the non-waived phase-1 seed
and functional certificate remain the admission gate for the tracked suite.

## 21. Focused scalar boundary result and direct M2 authoring

The original `pb_formal_scalar_regred_v1_20260906` aggregate is invalidated: its analyzer did not
require each timing row's correctness/contract grade to pass, so two lane-invalid members could
still produce `ESTABLISHED`. `comparison_group_claim.py` now refuses failed or missing `correct`
rows. The clean replacement is `pb_formal_scalar_regred_v4_20260906`, sealed with candidate SHA
`dc05f4dd750149563da126154f7f02c76ad1db9f5b13d3dc5c412ac42f77db6c` and a separately sealed
corrected adjudicator.

All eight exact PB00--PB03 grades pass. Both repetitions are deterministic. The island minus
no-island complete-kernel deltas are 1,546 GSIM cycles at K=32 and 1,511 cycles at K=64. This is
evidence for deleting the canonical host saturating-requant pass at that boundary. It is not a
full-layer or model timing result.

A subsequent full-M2 validation exposed a source-task scheduling regression in the first fusion
patch: unrelated outputless host work appearing before a mesh task was incorrectly consumed by
that contraction, eventually forcing the compiler into a scalar-only fallback past its 400,000
element budget. The repaired scheduler consumes only the transitive initializer setup owned by
the contraction and carries unrelated host work forward. On the frozen M2 interface the repaired
plan owns all 454 source operations exactly once across 24 tasks and emits 36 accelerator commands;
the focused requant regression remains green.

The direct whole-M2 phase-2 sequence is `global_phase2_perf_opt_m2_v1_20260906`. It starts from
that repaired compiler, spends no budget on another 38-member functional sweep, and gives each
authoring round host-owned complete-model structural evidence plus separately admitted bounded
probes. Until it seals a semantic witness and global cost evidence,
`full_model_timing_status` remains `UNMEASURED_FULL_MODEL` and no model speedup is established.

## 22. Completed post-freeze M2 execution and selected Phase-2 candidate

The `UNMEASURED_FULL_MODEL` statement at the end of §21 is now superseded for the frozen M2
workload. The retained candidates were compiled to complete-model ELFs and executed on GSIM against
the frozen capsule's real model weights and runtime inputs. Every measured output is byte-for-byte
equal to the recorded PyTorch eager golden in its result document.

| candidate | GSIM cycles | change from initial | disposition |
| --- | ---: | ---: | --- |
| `b2941ebc...` | 2,110,042 | baseline | initial compiler |
| `d925b87a...` | 2,103,648 | -6,394 (-0.3030%) | retained first checkpoint |
| `6798e318...` | 2,103,504 | -6,538 (-0.3099%) | independently validated v19 draft |
| `ec949d0b...` | **2,101,367** | **-8,675 (-0.4111%)** | **selected** |

The selected result is a 1.004128x speedup over the initial M2 candidate. Its optimization commutes
padded-gather writes into disjoint padding slabs plus one interior write, deleting 256 dynamic host
store bytes, 2,008 dynamic integer operations and 256 conversions in the structural witness. The
selected candidate seal is
`global_phase2_perf_opt_m2_v2_continuation_20260907/global_iterations/round_0000_candidate.json`;
the independently executable result is
`global_phase2_perf_opt_m2_v2_continuation_20260907/postfreeze_execution_smoke/candidate/result.json`.
The final digest-bound choice and all comparison result paths are recorded in
`global_phase2_perf_opt_m2_final_20260907/selection.json`.

The evidence binding is stronger than the original smoke path: the result pins the candidate,
analysis-worker record, raw retained command buffer, lowered program, compiled ELF, frozen capsule,
canonical runtime inputs and frozen model bundle. The selected result records command-buffer SHA
`c4fcdf88...`, lowered SHA `a09570d4...`, ELF SHA `8b13a0cf...`, canonical-input SHA
`ed2c5e59...`, and model-bundle SHA `15169062...`. `run_whole_program_gsim_smoke.py` is the reusable
runner for reproducing that chain.

Four orchestration defects discovered by the rounds are fixed in the repository. Exact model
capacity failures now resume from the last checkpoint without admitting the unfinished draft; a
600-second round reserves 120 seconds for finalization; and expired broker requests receive an audit
receipt. Candidate lookup also searches prior iterations for an exact hash and unchanged dependency
set, so reverting to an earlier verified revision can be sealed instead of being rejected merely
because it was not the last attempted edit. The runner, controller, broker and certificate suites
pass 218 focused tests.

There is no live paid optimization round now. The v1 sequence accepted `d925b87a...`, rejected or
rolled back three later edits, then encountered model capacity. The v2 continuation accepted
`ec949d0b...`; v3 explored two more changes, found no retained improvement and exited. Re-launching
the same speculative search was stopped after the actual complete-model executions established the
winner.

The 38-member performance set is **not** rerun for every candidate. This direct-M2 path deliberately
reused the frozen functional boundary and ran bounded structural checks while authoring, followed by
one complete-model golden comparison for each retained candidate. The earlier functional-certificate
waiver remains a recorded weakening of campaign-wide evidence; it is not being presented as a new
38-member proof.

Scope this result narrowly. It establishes one exact-correct frozen-M2 GSIM comparison, not a
predeclared all-trial campaign claim, independent Verilator/FireSim agreement, a full ResNet-50
speedup, or closure of the reported 102x gap. A second full-model Verilator/FireSim run was dropped
from this iteration because it is not needed to choose among same-engine GSIM candidates and was the
long path repeatedly blocking iteration. It remains required if the number is promoted to a formal
cross-engine/publication claim. The high-value remaining Phase-2 work is still real BatchNorm and
residual offload, compiler emission/selection of `LOOP_CONV_WS`, and on-mesh tiling/dataflow/residency
search; those are algorithmic compiler/search tasks, not missing deterministic launch plumbing.

## 23. Multi-model full-graph Phase 2 (2026-09-07)

The M2 work above proved the loop, but M2 is too small and too narrow to be the optimization
objective. Phase 2 now supports an ordered portfolio of complete graphs. One candidate compiler
snapshot is compiled sequentially for every member under one 600-second deadline. There is no
full-layer or full-model simulation in this loop.

The first four-model portfolio is:

| role | objective | source status | what it stresses |
| --- | --- | --- | --- |
| primary | ResNet-50 W8A8 | host-pinned external normalized graph | convolution, residual boundaries, epilogues, im2col, arena lifetime |
| training | TinyLlama | exact graph from the frozen 92/96 input snapshot | attention/MLP, KV-shaped movement, long-lived encodings |
| training | LSTMNetViT | exact graph from the frozen 92/96 input snapshot | vision + recurrent state, mixed spatial/state lifetimes |
| training | SmolVLA W8A8 denoise step | host-pinned external normalized graph | large VLM attention/MLP, vision-language crossings, action expert |

Do not misread the source status. ResNet-50 was added after the frozen Phase-1 snapshot, and
SmolVLA is a separate recapture. TinyLlama and LSTMNetViT are present in that snapshot but were not
members of the reported 92 passing rows. None of the four receives a new numerical qualification
from this setup. The exact old 92/96 qualification and four waivers remain unchanged; Phase 1 is
not rerun.

The external model declarations are:

- `out/artifacts/perf-bench/gemmini/objectives/resnet50_w8a8.external_objective.json`, SHA-256
  `b66f597511e2436d7e60ef465ef29e5b6e2c0fa94ce3cb37becee869a8ead279`.
- `out/artifacts/perf-bench/gemmini/objectives/smolvla_denoise_step_w8a8.external_objective.json`,
  SHA-256 `3922381e63def060a56d2ad7aef4424888e509142bcd579f8c094f2b792be37d`.

Both declarations pin exact already-normalized MLIR plus provenance hashes. Only normalized MLIR is
sealed into the read-only agent grant; weights, goldens and capture scripts are not exposed as
executable authority. The SmolVLA graph is 4,605,210 bytes, has entry `forward`, and has no
unresolved `func.call`. It represents the SmolVLM2 backbone plus action expert for one flow-matching
denoise step. It is not the full 1x-prefix + 10x-denoise + 1x-decode session; that later claim needs
a multi-program scenario contract so prefix work is not incorrectly repeated ten times.

### 23.1 What happens for every candidate edit

1. The host checks the frozen Phase-1 receipt, target descriptor, model portfolio, compiler edit
   authority, shared compiler dependencies, historical references and candidate digest.
2. It copies the candidate once into a read-only submitted snapshot.
3. It compiles the optimization baseline and candidate for each of the four complete graphs,
   sequentially. Remaining time is divided by remaining members, so an early timeout cannot starve
   all later graphs and unused time rolls forward.
4. Each member independently verifies complete logical-graph capture, source/task ownership,
   dependency coverage, global-plan-to-emission binding, command buffer, lowered program and target
   machine artifact where the backend supports it.
5. The host compares each model only with the same model in the prior compiler revision. It does
   not add unlike MACs, bytes, dispatches or guessed cycles into one score.
6. If any member fails emission or binding, the aggregate revision is blocked and cannot be probed
   or sealed. A successful revision is a Pareto candidate, not a measured speedup.
7. Only a changed, relevant mechanism may use a reduced witness. The required measurement shape is
   one warm run followed by one measured run, reporting compute cycles and only the counters needed
   for occupancy/movement/latency-hiding diagnosis.
8. FireSim remains post-freeze publication validation. If used, the required lifecycle is exactly
   `firesim kill`, `firesim infrasetup`, `firesim runworkload`, `firesim kill`, through the queue.

### 23.2 Evidence exposed to the optimizing agent

For every model, the agent receives a compact host-derived view and a pointer to the immutable full
record. The compact view contains the highest-ranked shared compiler surfaces and the global signals
that decide which lever to pull:

- exact work placement and host-versus-device ownership;
- logical and physical movement, payloads, repeated transfers and queued-movement context;
- storage encodings, layout conversions, prepack opportunities and representation transitions;
- accumulator/input/weight residency and capacity contracts;
- dispatch/configuration counts, fences, dependency frontiers and issue ordering;
- host CFG activity, dynamic integer/address work and materialized intermediates;
- machine-object changes and instruction motifs;
- structural lower bounds, roofline inputs and explicit unknowns;
- edit-surface path, AST symbol, intended emitted effect, validation and revert condition.

The full secondary graphs are not copied into broker stdout. The compact adapter retains at most
four ranked actions per secondary model and links the complete evidence by JSON pointer. This keeps
the authoring context bounded while preserving auditability.

The agent instruction explicitly requires generalized algorithms: no capsule/model-name
specialization, no summing unlike workloads, and preference for a transformation that improves
multiple families or deletes a shared bottleneck. MicroViT and kernel capsules remain smoke tests
and mechanism calibrators; they cannot decide global acceptance.

### 23.3 Code and receipt structure

The main seams are:

- `launch_global_agent_experiment.py`: ordered portfolio CLI, immutable external-source loading,
  resource lease, launch/resume binding and status receipts.
- `run_global_perf_experiment.py`: one candidate snapshot, sequential per-model analyses, fair
  deadline allocation, Pareto records, compact agent view, aggregate seal/consume checks.
- `perf_agent_stage.py`: answer-free read-only grants for multiple external objectives and the
  existing host analysis broker.
- `merlin/perf/external_objective.py`: exact source/spec/provenance validation with collision-free
  per-objective grant paths.
- `merlin/perf/analysis_worker.py`: one bounded, process-group-owned compile/static worker reused
  sequentially; no simulator.
- `merlin/perf/structural_delta.py`, `storage_encoding.py`, `host_cfg_activity.py`,
  `task_instruction_evidence.py`, `model_placement.py`, `model_macs.py` and
  `artifact_activity.py`: the target-neutral diagnostic planes.

The binding hierarchy for a paper figure is:

```text
frozen Phase 1 + target + four pinned graphs + edit authority
                         |
                         v
              immutable candidate snapshot
                         |
          +--------------+--------------+--------------+
          v              v              v              v
       ResNet          TinyLlama     LSTMNetViT      SmolVLA
       compile          compile        compile        compile
       + static         + static       + static       + static
          +--------------+--------------+--------------+
                         |
                 per-model deltas
                         |
              Pareto/regression decision
                         |
       optional admitted short changed-mechanism witness
                         |
                 immutable checkpoint
                         |
        optional post-freeze queued FireSim validation
```

The checkpoint binds the ordered portfolio hash, every member identity, the shared candidate hash,
each member's workload hash and readiness, and the complete iteration record. Resume must match the
same portfolio hash. A legacy single-model checkpoint cannot silently become a multi-model run; the
portfolio starts a new explicit optimization segment.

Focused regression status for this implementation: 295 controller/broker tests passed before the
multi-external extension, followed by 295 tests over `test_perf_agent_stage.py`,
`test_external_objective.py` and `test_global_perf_experiment.py`; no simulation was run by those
tests. A real four-model compile/static preflight is the next gate, after the already-running
single-M2 guarded sequence releases the host-wide experiment lease.

## 24. Four-model v13 experiment and paper-figure handoff (2026-09-07)

This section is the current, figure-ready description of Phase 2. It supersedes the final sentence
of §23: the four-model compile/static preflight did run, the v13 experiment authored and retained a
Round-0 candidate, and a later authoring round was refused. It does **not** establish a measured
portfolio, ResNet-50, or end-to-end speedup.

The live artifact root is
`out/artifacts/perf-bench/gemmini/global_phase2_portfolio_r50_tiny_lstm_smol_w8a8_v13_20260907/`.
Its exact portfolio SHA is
`5ab012efcf4d2a23c276921d71ad0b5201f05a6c62b3c73bb62681918edde5b4`; the target SHA is
`667990eef06958951d9ee46d51535d64bcff9557a17c982d146bcf1acc15985e`; and the immutable
optimization comparison compiler SHA is
`fbc39087e50e57f5ff0038820a04081233764d2a106e11d5226e4bf259df135b`. The frozen Phase-1
submission remains `5063331978caec47293c45e64beaeb7a1c534a94fa0032c8a8d9a5228ff78aba`, with the same
92/96 result and the same four declared gaps. `global_iterations/experiment.json` is the launch
receipt (file SHA-256 `2f1384e4036342607c2241516541f95971b17907e396c71e33470686afe17cfa`), and explicitly records
`full_model_simulation_allowed: false` and `firesim_stage: optional_post_freeze_validation`.

### 24.1 Objective: optimize one compiler against four complete graphs

One candidate compiler is evaluated against the same ordered portfolio on every accepted
iteration. Each model is compared only with its own preceding compiler revision. There is no sum
of unlike MACs, bytes, dispatches, or estimated cycles.

| model family | exact v13 objective | capsule/objective SHA | status in the loop |
| --- | --- | --- | --- |
| CNN | `resnet50_w8a8_static_full_graph_prepared_v1` | `9a3fcb08faed5d0f6b2a2f0de88a49cf506b82b59b2a822f8f4debd7d16afbe6` | complete-graph compile/static only |
| decoder transformer | `SY_model_tiny_llama` | `8dbfeb062749844f64f1da7d998790577314791329c50f4bb5c53633ea73a511` | complete-graph compile/static only |
| vision + recurrent | `lstmnetvit_w8a8_full_graph_prepared_v1` | `9815bffce3445191287980e0911e62344bff91413c5c55b990af8d12c1fe7fe2` | complete-graph compile/static only |
| VLM/action | `smolvla_flow_denoise_w8a8_canonical_xdsl` | `973c629e07f6bb7a2ee9650605c69faca438f0e46b6a79ac3bff8cf79c0b3e3c` | one complete denoise-step graph, compile/static only |

The three external objective declarations sealed into the v13 input manifest have these exact
spec/source pairs:

| objective | spec SHA-256 | normalized-source SHA-256 |
| --- | --- | --- |
| ResNet-50 | `206602694e3308087e4ee24aeb6a095be99c455a0698addd563e5c7ef77c768f` | `76c26171096661650e2ee440fb545926936b1e770d2bfd0206a6509e6e835e94` |
| LSTMNetViT | `5d015ddd1763c0270519ecea313b68ba5296359439096c450f7e70b772e3b26a` | `65a32b514dcb7b5ce7bba4421f0cf6425f3d7b7b7f16c9740c8dea4798c6016c` |
| SmolVLA | `62a649a1ca9fbc3731f56aef0605ce93e2911d603cc509cf5ebae5face82f17a` | `4e0b5e2f17398f069901848e64b54163b921f4a56025d5a3510d97fa1459a2e8` |

Those identities are in `_agent_inputs/agent_input_manifest.json` (file SHA-256
`17dda06943b155ea3b713951564cfb3688a82d3b5b25c980f62a90663ea96a41`). They are optimization
objectives, not additions to Phase 1: their full-model numerical equivalence remains `UNPROVEN`, and
full-model execution is disabled during search. TinyLlama comes from the frozen corpus; the other
three are pinned external normalized sources. SmolVLA is one denoise step, not a complete
prefix/ten-step/decode session.

The **full graphs are the unit of static reasoning**. Full graph capture, compilation, placement,
task ownership, dependencies, command buffers, lowering, target artifacts, movement, host work and
representation are inspected. The full graphs are **not** executed in GSIM, L3, or FireSim inside
the optimization loop. Only an admitted, bounded, mechanism-equivalent witness may execute, and it
cannot by itself prove a full-model speedup.

### 24.2 One coherent mechanism per round

The experiment is not an unrestricted search over arbitrary compiler changes. Each round advances
one causal mechanism through this state machine; a mechanism may need coordinated edits in more
than one authorized symbol, but it must have one stated cause and one predicted emitted effect.

| state | required input | host decision/evidence | transition |
| --- | --- | --- | --- |
| **1. Hypothesis** | a full-graph bottleneck and exact source/task/plan identity | state the mechanism, why it matters across model families, semantic obligations and stop/revert condition | proceed only with a falsifiable expected delta |
| **2. Opportunity** | ranked global/movement/encoding/occupancy evidence | select exact source-operation IDs and current plan digest; do not substitute an unrelated capsule | bind the proposed mechanism to actual portfolio work |
| **3. Edit surface** | host-frozen edit contract | name surface IDs, paths and AST symbols; candidate metadata is descriptive and cannot grant permission | author only inside the admitted compiler surface |
| **4. Static Pareto** | immutable submitted candidate snapshot | compile/analyze all four complete graphs; require verified emission, graph/task/source/dependency coverage, and no unexplained regression for any member | reject on failure/regression; otherwise remain an unmeasured Pareto candidate |
| **5. Changed-region semantics** | actual emitted/source region changed by this candidate | run a host-selected reduced semantic witness or an exact relative proof | reject incorrect/unattributed change; unresolved stays a blocker |
| **6. Bounded controlled profile** | cost is still decision-relevant **and** the witness/provider/relevance gate admits it | exactly one warm invocation, then one measured invocation; record compute cycles and only diagnostic counters needed for this mechanism | use only as local causal evidence; never project it into full-model cycles |
| **7. Retain or refuse** | all preceding receipts and exact candidate/portfolio identities | retain a digest-bound checkpoint only for a clean Pareto candidate; otherwise refuse and preserve the last checkpoint | the next round starts from the last retained candidate |

In compact form, the required control edge is:

```text
hypothesis -> opportunity -> authorized edit surface -> four-model static Pareto
           -> changed-region semantics -> [optional admitted warm+measured profile]
           -> retain | refuse
```

This state machine prevents three common errors: treating smaller IR as faster without an emitted
change, using a microkernel result for a different global mechanism, and allowing a timed-out draft
to replace the retained compiler. A plateau in a micro witness does not stop macro search.

The v13 budgets recorded in `agent_workspaces/round_00/STAGE_CONTEXT.json` are 1,200 seconds per
authoring round, a 1,020-second broker window, a 180-second final-response reserve, up to 1,200
seconds for host-owned post-authoring full-portfolio static validation, and at most 600 seconds for
an admitted reduced witness. The experiment receipt records four portfolio analysis workers. This
is resource-controlled static compilation, not four simultaneous simulations.

### 24.3 Trusted boundary versus agent-editable boundary

The paper diagram should draw a hard trust boundary. The host owns identity, evidence, admission,
scoring and checkpointing; the agent owns only declared compiler transformations.

**Trusted, immutable or host-owned:**

- frozen Phase-1 receipt, 92/96 status and waivers;
- ordered model portfolio, normalized sources, inputs, weights and reference semantics;
- target facts, hardware/configuration pins, ISA-derived constants and target digest;
- optimization baseline and all candidate/source/command-buffer/lowered/object digests;
- analysis workers, graph/task/source/dependency verifiers, semantic qualifiers and profile
  admission;
- measurement wrapper, warmup/ROI/counter interpretation, timeouts and process ownership;
- evaluator, scoring/Pareto policy, checkpoint/resume logic, sandbox and launch commands;
- compiler edit contract itself. A candidate's `manifest.yaml` can describe a change but cannot
  widen authority.

The host-verification policy SHA is
`fb736a9fdac815136b82b055a75b0c969498d7085aa79b5dbdc38291e95a3237`. The sealed performance
corpus manifest is `_frozen_corpus/performance_corpus_manifest.json`, SHA-256
`cc71c50229c62a8f75a0be0c4785fbafc7474d2b704681546bb53d2559b58d9b`.

**Agent-editable:** exactly the compiler AST symbols named by the host-frozen contract. In v13 this
is 30 existing symbols across seven candidate-relative files:

- `mlir_oot/lowering/model_lane.py`;
- `mlir_oot/lowering/source_conv_model_lane.py`;
- `mlir_oot/lowering/schedule.py`;
- `mlir_oot/codegen/builder.py`;
- `mlir_oot/codegen/fpbuilder.py`;
- `mlir_oot/codegen/loop_host_linalg.py`;
- `mlir_oot/codegen/llvm_emit.py`.

There are no v13 helper-extension directories. The contract SHA embedded in
`agent_workspaces/round_00/STAGE_CONTEXT.json` is
`1214457e3c66b7a735913145d1fe3ab1e9e2b4b5e0ffd172570f65f40e888f55`; the durable host copy is
`global_iterations/compiler_edit_authority.json` (file SHA-256
`14a3f270549658b4fda9efbfffb00e7d11e0c94a97aab1a6d58f57f60e91a2e3`). The required work order
names surface IDs, source-operation IDs, current plan digest, hypothesis, expected emitted delta,
semantic obligations, cheap validation and stop/revert condition.

This is CCA-style guidance without hard-coding Gemmini into the search algorithm: a target may
advertise different editable symbols and target facts, while the host keeps the same evidence and
admission protocol.

### 24.4 What Phase 2 can see, and what remains `UNKNOWN`

The analyzer deliberately separates exact/static evidence from measured/dynamic evidence.

| question | exposed evidence used to choose a lever | still `UNKNOWN` without an admitted execution |
| --- | --- | --- |
| **Global lowering** | complete logical graph; source-to-task ownership; dependency coverage; host/device placement; contraction MACs by lane; task/dispatch count; host dynamic operation estimates; emitted command buffer/lowered/object identity | full-model numerical equivalence for the three external objectives; real end-to-end cycles; unmodeled runtime/library work |
| **Data movement and residency** | command-buffer declared input/output bytes; host load/store/allocation payload; repeated logical transfers; boundary tensors; queued-movement source context; working-set/capacity regime | physical DRAM/cache traffic; descriptor-expanded traffic; actual cache misses, bank conflicts, evictions and persistence across calls; whether two legal issue orders overlap better |
| **Encodings and layout** | source tensor dtypes/shapes; declared storage directives; lazy/materialized transpose/gather/copy paths; conversion counts in host CFG; physical egress evidence when a target adapter can bind it to emitted bytes | executed conversion count when only declarations exist; physical width/encoding for unmatched writes; representation cost not linked to an emitted artifact |
| **Occupancy and latency hiding** | configuration, loop-descriptor and synchronization issue sites; dependency frontier; barrier count; issue ordering; bounded queued-load context; target peak when derivable | dynamic mesh occupancy, DMA/compute overlap, queue stalls, scratchpad-bank contention, descriptor-expanded work and cycle attribution |
| **Headroom / roofline** | exact captured contraction MACs; on-/off-device fraction; declared bytes; capacity regime; theoretical target peak/lower bound when target facts support it; same-engine historical calibration when identity-compatible | an achievable MAC/cycle ceiling for an uncalibrated graph, attainable bandwidth, and a cycle winner between two schedules when static ordering signals fail validation |

For the current ResNet record, all 54 captured contractions and 4,089,184,256 structural MACs are
placed on mesh, while the untiled full-operand capacity classifier marks their regimes as spilling.
That is useful prioritization evidence, not physical traffic or a proved schedule. The v13 analyzer
reports no derived structural target peak, so both lower-bound arms are unavailable. Its historical
ordering-signal audit refuses every tested command-buffer proxy as a schedule oracle: command count
is below chance, dependency makespan is at chance, and the apparently better signals have too few
or contradictory pairs. Accordingly, Phase 2 may reject *more declared work* but may not call one
legal issue order faster without measurement.

The roofline should therefore be drawn as a **partially observed diagnostic**, not as a fabricated
performance model:

```text
arithmetic work (known MACs) / declared minimum bytes
                 |
                 +--> capacity and placement regime
                 +--> target theoretical peak, when derived
                 +--> measured compatible ceiling, when available
                 |
                 `--> explicit UNKNOWN: physical bytes, overlap, contention, achieved cycles
```

The agent is shown both the observation and the missing denominator/counter. It should pull a
global lever when host work, materialization, dispatches or boundaries dominate; an encoding lever
when conversions/materialized views dominate; a movement/residency lever when boundary or repeated
payload dominates; and an issue/latency-hiding lever only when a short matched context can resolve
an otherwise undecidable ordering. It must not replace an `UNKNOWN` with a guess.

### 24.5 v13 Round 0: retained static result, not measured speedup

Round 0 implemented one generalized mechanism: fuse a one-use reduction result directly into its
pointwise consumer. The host compiled and statically analyzed all four complete graphs. The exact
candidate SHA is `65ef41ecf3215c6f4100eb26bdcf67f309f263b6b2a9bfef0287bf108de853af`; the checkpoint is
`global_iterations/round_0000_candidate.json`, SHA-256
`cf6955c52da10cb7c37f2c0301efe1bd337e0fb2d25ed6e581747cf97227592f`.

| model | retained static deltas from its own baseline | unchanged invariants relevant to the claim |
| --- | --- | --- |
| ResNet-50 | no observed change | kernel object byte-identical (`1e231ab7...`); same 109 dispatches, 25,508,960 declared command-buffer bytes, 4,089,184,256 contraction MACs, 652 synchronization sites and 14,758 loop-descriptor sites |
| TinyLlama | allocation/load/store payload each -11,360 B; dynamic integer operations -14,200 | same accelerator work, 311 dispatches, 1,051,230,208 declared bytes and 30,048 loop-descriptor sites |
| LSTMNetViT | allocation/load/store payload each -10,712 B; dynamic integer operations -13,390; machine sites -174; object size -544 B | same accelerator work, 75 dispatches, 5,804,274 declared bytes and 426 loop-descriptor sites |
| SmolVLA | allocation/load/store payload each -41,864 B; dynamic integer operations -133,930; conversions -38,400 | same accelerator work, 233 dispatches, 129,821,760 declared bytes and 9,879 loop-descriptor sites |

No observed static metric increased. This is a valid cross-family structural improvement, but the
cost is still `UNKNOWN`: no full model was timed, occupancy/overlap was unresolved, the
changed-region qualifier found no applicable actual-source witness, and the reduced profile was
refused because the host mechanism-equivalent provider was unavailable. The semantic receipt is
`global_iterations/semantic_0001_1788815133980475150.json`, SHA-256
`735a0ad7ca9a5715fbe94025a60f1b8d6b59e6f981fc62b748424bcd30d5a4b5`; it records
`full_model_executed: false`, `full_model_correctness: UNPROVEN` and `timing_measured: false`. The
profile refusal is preserved in `global_iterations/host_refusals/round_None_call_3.txt`, SHA-256
`e9fb1668d123f2c11c0abbbbb5a1ec41fe28d45fa18cb3b1e28a0d3a7b94df6a`.

The authoritative detailed comparison is `global_iterations/iteration_0001.json`, SHA-256
`392493fc5c758b82422d6075feb661b0404bca3a9033d1e53d423dc507186454`; the agent's concise summary
is `rounds/round_00.final.txt`, SHA-256
`6e712d62d0f175ec3faaabb5f306ab33b5f1d43a542dfc0005a0587fdfe90ede`. Neither is a cycle result.

### 24.6 v13 Round 1: timeout refusal

Round 1 produced changed candidate bytes but did not finish the clean authoring protocol. The
agent process exited 124, post-authoring validation did not start, and the required real final
telemetry file was absent. The host therefore refused candidate
`ecf8dadcb3479e025c90555817eba463f697f0566b95981437da5b9dbb0608b1` instead of letting an
unfinished draft replace Round 0. The audit is
`global_iterations/agent_round_0001.json`, SHA-256
`aedf4876c91752bbec9b41495e42aebddbe0488cb8ec50f2215e1bced708573d`; the continuation failure
states `macro agent round did not finish with a clean authoring audit`.

The last verified iteration reuses exact static analysis for the retained Round-0 candidate rather
than recompiling unchanged bytes. `global_iterations/iteration_0003.json` records
`exact_analysis_reused: true`, 4/4 ready members, 16.88 seconds elapsed and
`UNMEASURED_FULL_MODEL`. This is cache/reuse evidence, not performance evidence.

### 24.7 Tooling status: implemented, active, and under review

Keep these statuses separate in prose and figures:

| tool/mechanism | status on 2026-09-07 | evidence and limitation |
| --- | --- | --- |
| exact baseline-emission cache | **implemented and active in v13** | `experiment.json` binds `_global_phase2_baseline_emission_cache_v1` to compiler-dependency SHA `f9da1591...` and imports four exact entries from v5; corruption/identity mismatch is fail-closed |
| exact immutable static-analysis reuse | **implemented and exercised in v13** | four `global_iterations/analysis_reuse_*.json` receipts exist; iteration 3 reuses the exact retained candidate/member bindings without starting analysis workers |
| secondary-member changed-region selection/qualification | **fix implemented in the shared tree, under review; not established by sealed v13** | v13 Round 0 changed only secondary models but its qualifier selected no applicable actual-source witness; focused tests now cover selection/recomputation of a changed secondary member, but a future sealed cohort must exercise and receipt it |
| batched-matmul command semantics/accounting | **implemented in committed generic runtime/perf tooling** | commit `92099da1` generalizes rank-3/rank-4 materialization, reference, simulation, MAC and movement accounting; commit `e0ac400e` admits the target contract path. v13 exposes rank-general/broadcast/capacity compiler edit surfaces, but Round 0 did not change batched placement or prove a batched performance gain |
| deployment admissibility core | **implemented after the v13 source seal** | `merlin/perf/deployment_admissibility.py` is target-neutral and consumes exact deployment profiles plus physical-egress and wrapper-event evidence; it is for a future checkpoint/final deployment gate, not evidence attached retroactively to v13 |
| Gemmini deployment evidence/LOOP decoder | **implemented target adapter after the v13 source seal** | target code can derive physical writeback and wrapper ordering evidence from exact artifacts and fail closed on unmatched outputs; a final candidate still needs a complete SHA-bound contract/config/header/bitstream profile before FireSim |

The qualifier/controller files are currently a reviewed-work boundary rather than something to
silently fold into the running v13 snapshot. Do not mutate v13 to pick up those changes; start a new
sealed cohort or use them at final promotion after review.

### 24.8 Target-general core and target adapters

The architecture is intentionally split so the experiment is not a Gemmini benchmark harness with
model names removed.

**Target-general core** owns:

- external objective identity and immutable source grants;
- compiler edit-surface contracts and submitted-tree hashing;
- captured graph, task/source/dependency coverage and model placement;
- contraction work, structural deltas, host CFG activity and storage transitions;
- per-model Pareto comparison, exact cache/reuse, checkpoint and resume;
- changed-region/probe admission contracts and the generic deployment-admissibility state machine;
- explicit `UNKNOWN` propagation.

Representative modules are `merlin/perf/external_objective.py`, `analysis_worker.py`,
`structural_delta.py`, `storage_encoding.py`, `host_cfg_activity.py`,
`task_instruction_evidence.py`, `model_placement.py`, `model_macs.py`, `work_volume.py` and
`deployment_admissibility.py`.

**Target adapters** supply facts the generic core must not guess:

- target artifact production and disassembly/decoding;
- instruction roles, descriptor semantics, physical movement and readout encodings;
- scratchpad/accumulator capacities and theoretical peaks;
- completion ABI, runtime header, wrapper events, counter semantics and hardware identity;
- construction of a digest-bound deployment profile.

For Gemmini these include the existing backend/codegen plus
`merlin/targets/gemmini/backend/gemmini_deployment_evidence.py` and
`gemmini_loop_matmul_decode.py`. A different out-of-tree target supplies its own adapter while using
the same objective, evidence, Pareto, semantic, profile-admission and checkpoint protocol.

The boundary rule is simple: generic code asks a semantic question such as “what physical encoding
left the device?” or “did completion precede ROI end?”; a target adapter answers from emitted bytes
and pinned target artifacts. Missing linkage returns `UNKNOWN`/refusal, never an inference from a
declaration.

### 24.9 FireSim is final-only

FireSim is not part of the per-round optimizer. It is reserved for a frozen, deployment-admissible
candidate when a publication-quality hardware-aligned number is required. Every invocation must go
through the existing `/scratch/firesim_queue`; agents must not bypass the queue or operate the farm
directly.

The required queued lifecycle is exactly:

```text
firesim kill
firesim infrasetup
firesim runworkload
firesim kill
```

Do not reorder, omit, or combine those steps. Before admission, the deployment profile must bind
the exact compiler candidate, contract/configuration, runtime header, wrapper, ELF/object and
bitstream/hardware database identities. A mismatch is a refusal, not a warning.

The runtime measurement contract is one warm invocation and one measured invocation, with device
completion on both sides:

```text
warm invocation -> completion
counter reset -> ROI start -> measured invocation -> completion -> ROI end
validation outside the measured window
```

Report the accelerator compute-cycle window as the primary result. Capture only counters needed to
interpret compute occupancy, movement or overlap; do not time reference generation, validation,
programming, setup, teardown or an all-purpose debug trace. FireSim confirms the final number and
deployment compatibility; it is not how the agent explores schedules.

### 24.10 Current dashboard and source-of-truth paths

| purpose | path |
| --- | --- |
| human status | `out/artifacts/perf-bench/phase2_progress_20260907/portfolio_v13/STATUS.md` |
| machine-readable status | `out/artifacts/perf-bench/phase2_progress_20260907/portfolio_v13/status.json` |
| run root | `out/artifacts/perf-bench/gemmini/global_phase2_portfolio_r50_tiny_lstm_smol_w8a8_v13_20260907/` |
| immutable experiment source | `out/artifacts/perf-bench/gemmini/global_phase2_portfolio_r50_tiny_lstm_smol_w8a8_v13_20260907.source/` |
| experiment/portfolio/cache policy | `global_iterations/experiment.json` |
| frozen candidate edit authority | `global_iterations/compiler_edit_authority.json` |
| agent-visible identity manifest | `_agent_inputs/agent_input_manifest.json` |
| frozen performance corpus | `_frozen_corpus/performance_corpus_manifest.json` |
| seed full-graph evidence | `agent_workspaces/round_00/INITIAL_FULL_MODEL_EVIDENCE.json` |
| agent protocol/context | `agent_workspaces/round_00/STAGE_CONTEXT.json` and `TASK.md` |
| Round-0 detailed result | `global_iterations/iteration_0001.json` |
| retained checkpoint | `global_iterations/round_0000_candidate.json` |
| Round-0 broker audit | `global_control/round_0000/receipts.jsonl` |
| Round-0 semantic result | `global_iterations/semantic_0001_1788815133980475150.json` |
| Round-1 refusal | `global_iterations/agent_round_0001.json` |

At the last recorded dashboard refresh, state was `live_launcher_observed`, four iterations had been
processed, the latest retained candidate was `65ef41ec...`, and the latest terminal audit was
Round 1 refused with exit 124. A prepared `round_02` workspace is not proof that useful authoring or
validation completed. The dashboard correctly states `Full-model speedup: UNPROVEN`.

### 24.11 Figure 1: overview node and edge inventory

Use these node IDs directly when drafting the overview figure.

| node | label | visual grouping |
| --- | --- | --- |
| O1 | Frozen Phase-1 qualification (92/96 + waivers) | trusted host input |
| O2 | Pinned four-model portfolio | trusted host input |
| O3 | Target facts + optimization baseline + edit authority | trusted host input |
| O4 | Agent: one mechanism hypothesis + authorized compiler edit | untrusted/agent-authored |
| O5 | Immutable candidate snapshot | trust-boundary crossing |
| O6a-d | Full-graph compile/static analysis: ResNet, TinyLlama, LSTMNetViT, SmolVLA | four parallel portfolio lanes |
| O7 | Per-model structural deltas + explicit unknowns | host evidence |
| O8 | Four-model Pareto gate | host decision |
| O9 | Changed-region semantic gate | host decision |
| O10 | Optional bounded warm+measured mechanism profile | controlled execution, dashed |
| O11 | Digest-bound retain/refuse checkpoint | host decision/state |
| O12 | Optional final deployment gate + queued FireSim | post-freeze, dashed |

Edges:

1. `O1,O2,O3 -> O4`: bounded context, ranked evidence and legal edit surfaces.
2. `O4 -> O5`: submit exact bytes; hash and revoke write access.
3. `O5 -> O6a-d`: same compiler snapshot, four complete graphs, static-only.
4. `O6a-d -> O7`: graph/plan/artifact/movement/encoding/host-work facts.
5. `O7 -> O8`: compare each model only with itself; no cross-model scalar score.
6. `O8 -> O9`: only a non-regressing relevant mechanism advances.
7. `O9 -> O10`: dashed conditional edge, only if cost is unknown and a witness is admitted.
8. `O9,O10 -> O11`: retain if all obligations hold; otherwise refuse and loop to O4 from the last
   checkpoint.
9. `O11 -> O12`: dashed post-freeze edge; exact deployment profile, then queued final validation.

Recommended visual encoding: blue boxes for trusted host state, orange for agent-authored compiler
logic, white for per-model workers, diamonds for admission/Pareto decisions, solid edges for every
round, dashed edges for optional execution. Put “no full-model simulation” over O6a-d and “cycles
only after admission” over O10/O12.

### 24.12 Figure 2: detailed Phase-2 node and edge inventory

The detailed figure should show the causal evidence planes inside one round.

| node | label/content |
| --- | --- |
| D1 | Trust root: Phase 1, portfolio SHA, target SHA, optimization-baseline SHA, host policy SHA |
| D2 | Portfolio action digest: four readiness summaries + ranked shared bottlenecks |
| D3 | Work order: surface IDs, source-op IDs, plan digest, hypothesis, expected delta, semantics, stop rule |
| D4 | Edit gate: path/AST-symbol authority + masked shared dependencies |
| D5 | Submitted compiler tree/content hash |
| D6 | Per-model compiler worker: normalized source -> command buffer + lowered IR + target artifact |
| D7 | Capture/ownership plane: logical graph, tasks, source ops, dependencies, ABI bindings |
| D8 | Global work plane: placement, contractions/MACs, host dynamic ops, dispatches |
| D9 | Movement/residency plane: logical payload, host payload, boundaries, working set, queued context |
| D10 | Encoding plane: dtypes, storage directives, conversions, lazy/materialized views, physical egress when decoded |
| D11 | Issue/occupancy plane: configs, loop descriptors, fences, barriers, dependency frontier, ordering |
| D12 | Headroom/roofline plane: MACs, declared minimum bytes, capacity, theoretical/measured compatible ceilings, UNKNOWNs |
| D13 | Artifact identity plane: candidate/source/CB/lowered/object digests and before/after byte relation |
| D14 | Per-model structural delta |
| D15 | Portfolio Pareto gate |
| D16 | Selected actual changed region/mechanism |
| D17 | Reduced semantic qualifier or relative proof |
| D18 | Probe relevance/provider/admission gate |
| D19 | Warm run -> completion; reset/start -> measured compute -> completion/end; validation outside ROI |
| D20 | Decision feedback: measured local cost or explicit UNKNOWN/refusal |
| D21 | Retained candidate/checkpoint + continuation memory |
| D22 | Post-freeze deployment admissibility: exact profile, physical egress and wrapper-event order |
| D23 | `/scratch/firesim_queue`: kill -> infrasetup -> runworkload -> kill |
| D24 | Final publication receipt: exact cycles + correctness + identities |

Detailed edges:

1. `D1 -> D2 -> D3 -> D4 -> D5`: evidence-guided, authority-bounded authoring.
2. `D5 -> D6`, once per portfolio member under resource allocation.
3. `D6 -> D7,D8,D9,D10,D11,D12,D13`: derive orthogonal evidence from the same emitted candidate.
4. `D7..D13 -> D14`, then four `D14 -> D15` edges, one per model.
5. `D15 refuse -> D21(previous)`; `D15 advance -> D16`.
6. `D16 -> D17`; unresolved or failed semantics returns refusal to `D21(previous)`.
7. `D17 -> D18` only if cost affects selection. A host-only proven work deletion need not force a
   device profile.
8. `D18 admit -> D19 -> D20`; `D18 refuse -> D20(UNKNOWN)`.
9. `D17,D20 -> D21`; the checkpoint records what is structural, semantic, measured and unknown.
10. `D21 -> D2` for the next round, using prior feedback as search history rather than calibration
    for new bytes.
11. `D21(final freeze) -> D22 -> D23 -> D24`; no edge from D19 directly to a full-model speedup
    claim.

For the paper caption, the essential sentence is: **Phase 2 optimizes one compiler snapshot against
four full graphs using exact static evidence, invokes execution only for an admitted reduced causal
witness, and reserves queued FireSim for final post-freeze validation.**

## 25. Macro-first correction and exact ResNet opportunity census (2026-09-07)

The experiment must search from larger scopes to smaller scopes. A flat list of actionable compiler
surfaces allowed an agent to choose an easy scalar reduction even while model-wide boundaries and
materializations remained. The shared tooling now exposes this target-neutral order:

| tier | scope | examples of evidence, not target-specific recipes |
| ---: | --- | --- |
| 0 | correctness and regression repair | failed emission, added work/movement/fences, new trace defects |
| 1 | whole-program work deletion | placement/coverage, fusion and host boundaries, exact epilogues/residuals, arena/ABI work |
| 2 | global dataflow and representation | encoding/layout selection, materialization, movement, cross-operation residency |
| 3 | global execution and latency hiding | dispatch/loop offload, overlap, synchronization, capacity/contention |
| 4 | operator/layer/tile efficiency | arithmetic lowering and per-layer scheduling |
| 5 | local scalar cleanup | peepholes after larger opportunities are terminal |

An agent may descend only after every higher tier has either produced a retained structural change
or a source/plan-bound refusal or no-op for the current revision. Within a tier it prefers quantified
dynamic extent and then cross-model coverage. Unlike cost units are not added; a warm-one,
measured-one reduced causal witness selects between a byte/compute/overlap tradeoff. An easy local
rewrite or capsule win cannot bypass an actionable higher-tier mechanism. This contract is emitted
by `merlin/perf/agent_guidance.py::macro_optimization_order`; the next sealed experiment also places
it in the four-model action digest and agent task.

### 25.1 v13 Round 2 and Round 3 outcomes

Round 2 attempted to extend lazy reduction/pointwise fusion through reshape/transpose view chains.
The candidate failed complete-portfolio analysis, including a concrete transpose-index failure on
SmolVLA, and was reverted to retained candidate `65ef41ec...`. The post-authoring host check then
reused the exact retained analysis and restored 4/4 readiness. No new transformation or performance
claim was retained.

Round 3 attempted loop-carried SSA accumulation for contiguous trailing multi-axis reductions,
targeting ResNet `reduce_0`. Whole-portfolio validation exhausted its budget with only 1/4 members
verified. The host retained `65ef41ec...` unchanged. This was a clean method outcome but a poor
impact choice; it motivated the macro-first order above. Round 4 subsequently selected shared
host-side materialization/fusion across the portfolio, but the host resource guard terminated the
worker before it edited the compiler because an unrelated approximately 36 GB GSIM process drove
the machine below the memory threshold and filled swap. The retained candidate stayed
`65ef41ec...`; Round 4 produced no optimization result. This is evidence that the resource guard
worked, not a refusal or no-op for the materialization mechanism, so that tier remains first in the
next experiment.

### 25.2 Frozen ResNet epilogue semantics: exact correction

The frozen ResNet graph does **not** contain an unfused BatchNorm subtract/rsqrt chain. BatchNorm is
already folded. Each contraction is followed by an ordered expression of this general form:

```text
i32 accumulator
  -> f32 conversion
  -> activation-scale multiply
  -> per-channel weight-scale multiply
  -> f32 bias add
  -> ReLU or intervening source operations
  -> reciprocal-output-scale multiply
  -> round-even -> zero-point add -> clamp -> i8
```

Combining the FP32 multiplies, moving the FP32 bias into the integer accumulator, or replacing
round-even with an integer half-up requantization changes results in general. Therefore the existing
`[bias, acc_scale, relu]` device epilogue is not an exact drop-in.

Exact census from the frozen command buffer and compilation receipt:

| item | count/volume |
| --- | ---: |
| accelerator contractions | 53 convolutions + 1 FC, all producing i32 |
| host tasks | 55 |
| device-to-host boundaries | 54; 44,459,936 logical bytes |
| host-to-device boundaries | 50; 9,058,816 logical bytes |
| hypothetical i32-to-i8 egress saving | 33,344,952 bytes (75%), upper bound only |
| normalized quantization sites | 50 `roundeven`, 50 `fptosi` |
| other conversion / FP32 multiply sites | 104 `sitofp`; 158 FP32 multiplies |
| fences | 652 total; 546 associated with streamed convolution packing |

The first global epilogue mechanism is therefore: extract the exact ordered typed chain; offer
target-neutral global-plan alternatives; fuse admitted host evaluation without materializing its
intermediates; and use a native target readout only for a region whose complete expression is proven
equivalent. A second dynamic tensor operand terminates this mechanism; residual add requires its own
two-input quantized operation. The 33.34 MB figure guides prioritization but is not an achieved or
currently eligible saving.

### 25.3 Convolution packing versus native convolution

The frozen model contains 53 convolutions and 4,087,136,256 convolution MACs:

| group | count | MACs | generated gather bytes | row-packing fences |
| --- | ---: | ---: | ---: | ---: |
| direct-DMA 1x1 stride 1 | 33 | 1,811,152,896 | 0 | 0 |
| 3x3 | 16 | 1,849,688,064 | 12,418,560 | 385 |
| 7x7 stem | 1 | 118,013,952 | 1,843,968 | 112 |
| 1x1 stride 2 | 3 | 308,281,344 | 351,232 | 49 |
| **total** | **53** | **4,087,136,256** | **14,613,760** | **546** |

The 20 packed convolutions account for 55.7% of convolution MACs. The current
`Scheduler.convolution` emits row-streamed im2col followed by one fence per packed row. Of the
14,613,760 generated input reads, 639,316 read a clamped address only to discard it and write zero.
Interior/border specialization can remove those reads and repeated bounds/address work exactly, but
it retains the 14.6 MB window and fence structure.

The existing native `LOOP_CONV_WS` adapter is eligible for **0/53** frozen convolutions: it requires
a single capacity-fitting NHWC, stride-1, no-padding operation and performs signed narrow output,
whereas the frozen graph requires raw i32 outputs. A higher-ceiling feasibility route is suggested by
the pinned RTL: suppress the native store, retain the accumulator, then explicitly read out full i32
from a proven accumulator layout. This is unimplemented and must first prove initialization,
partial-reduction continuation, layout, lifetime, completion and non-overwrite on a reduced signed
i8-to-i32 convolution whose values exceed i8 range. If that bounded proof fails, the experiment
immediately returns to exact fused-host epilogues or specialized existing packing rather than
spending repeated rounds forcing an inexact native interface.

### 25.4 What the updated figures must show

Insert a vertical **macro-lever ladder** between the portfolio evidence plane and the work-order
node. Draw a feedback edge from each tier to a retained result or bound refusal; only that terminal
edge unlocks the next smaller tier. The work-order/edit gate must eventually show two checks:

1. cumulative initial-seed to candidate authority; and
2. retained round-start to candidate AST delta restricted to one host-defined mechanism group.

The second check prevents a round from mixing unrelated authorized optimizations. A mechanism may
span multiple files, target adapters and model families. The host catalog—not candidate manifest
edits—defines its member surfaces. Cache reuse still requires a fresh round work-order binding and
cannot inherit a prior round's closure.

### 25.5 V14 exact four-model work assignment

V14 no longer asks the authoring agent to infer a promising site from aggregate counters. The host
replayed the retained v13 `iteration_0001.json` through a target-neutral logical-DAG/source-task
join and produced an immutable portfolio inventory:

`development_v14_epilogue_site_bindings_20260907/portfolio_site_inventory.json`

Its raw SHA-256 is
`cfcf2ab7b5e0562544f436fc119bd297e708cffb2df4d0616dca255cc2c2e95f`, its canonical
self-hash is `522771a00ba83f597cb8af349858907caccc450ebde42601035312d3a032fa53`,
and it is read-only. The extractor revalidates the candidate, source, plan, command-buffer, lowered,
logical-graph, target-facts and host-policy identities independently for each graph. It identifies
candidate sites, not legality or performance:

| complete graph | candidate chains | exact graph-local source-operation IDs |
| --- | ---: | ---: |
| ResNet-50 | 64 | 271 |
| TinyLlama | 607 | 5,532 |
| LSTMNetViT | 26 | 168 |
| SmolVLA | 542 | 4,958 |

The executable assignment is the separate immutable
`development_v14_epilogue_site_bindings_20260907/active_t01_01_work_order.json` (3,817,190
bytes, raw SHA-256
`6f8b11ba12091f4e5936a57b3524c61ba60c366aa42f6b088a9c0e967a358f64`, canonical
self-hash `0c38b0a3417b252630dbbe5fe0cfb4910a3d5e92aad8731922d7fc8ffa2dfedf`). It contains
only `t01_01_exact_host_epilogue_materialization_deletion`; source IDs remain partitioned by graph,
and each compact chain carries exact sites, one-use edges and input/output boundaries. The controller
freezes this file before authoring, revalidates it against the current four-model static analysis,
and rejects ambiguous flat IDs, missing members, hash drift, a different active mechanism, or an
empty assignment. This authorizes where to investigate; exact arithmetic equivalence, legal fusion,
emitted work deletion and cycles still require their normal gates.

The first clean detached-worktree analysis attempt exposed two environment constraints before doing
compiler work. Detached worktrees do not inherit `.env` and do not automatically contain the frozen
Phase-1 run under their repo-relative `out/runs`; V14 therefore explicitly supplies the external
toolchain path and stages an exact private copy of the immutable 92/96 receipts. After that path fix,
the machine resource guard refused launch because an unrelated Radiance GSIM process occupied about
66 GB and drove `MemAvailable` below the 64-GiB floor while swap was full. No simulator was started
by V14 and no candidate bytes changed in that attempt. The later admitted attempts and resulting
controller fixes are recorded below.

### 25.6 V14 launch corrections and live segment

The cold replay subsequently proved three graphs and exposed a portfolio-budget defect rather than
a lowering defect: SmolVLA completed in 428.4 seconds, TinyLlama in 580.5 seconds and ResNet-50 in
156.9 seconds, leaving LSTMNetViT only 29.2 seconds under the old shared 1,200-second deadline. That
failed evidence is preserved as
`global_phase2_portfolio_r50_tiny_lstm_smol_w8a8_v14_cold_analysis_failed_portfolio_budget_20260907`.
The three completed emissions exactly reproduced the retained V13 round-start hashes.

Commit `cdc583071ed02a49ba9cb1df41e8773e5797171a` changes only the host-only static portfolio
ceiling from 1,200 to 2,400 seconds; the reduced-witness execution ceiling remains 600 seconds.
Commit `d4417f6db2c83be9cf35afb1898902192dc09f3b` fixes the work-order bootstrap cycle: the untouched
compiler seed may be analyzed before a work-order analysis binding exists; the controller then
binds all four graphs to that immutable evidence, appends a no-compile bound seed iteration, and
only afterward opens authoring. Later rounds reuse that original host assignment rather than trying
to rebind it to edited artifacts. The complete controller suite passes (230 tests), as do all 17
execution-policy tests and repository pre-commit checks.

The full V14 segment is live at:

`global_phase2_portfolio_r50_tiny_lstm_smol_w8a8_v14_20260907`

It runs from the clean detached worktree
`/scratch/agustin/projects/oscar-merlin-phase2-v14` at exact commit `d4417f6d...`, uses the frozen
92/96 Phase-1 receipts, all four complete compile/static objectives, the immutable T01.01 catalog
and exact work order, six 1,200-second authoring rounds, and a 2,400-second host validation ceiling.
No complete model or complete layer is executed during search. The derived live dashboard is kept
outside the immutable run at
`global_phase2_portfolio_r50_tiny_lstm_smol_w8a8_v14_20260907_status/STATUS.md`.

For final FireSim certification, the queue-safe atomic operation is `firesim-queue
runworkload-full`. It machine-enforces exactly one lifecycle:

```text
firesim kill -> firesim infrasetup -> firesim runworkload -> firesim kill
```

The staged ELF must therefore execute one unmeasured warm inference followed by one measured
inference inside that single `runworkload`, fence before/after the measured window, validate outputs
outside timing, and print exactly one measured compute-cycle metric. Two queue-level
`runworkload` calls are not atomic and must not be used.

## 26. 2026-09-08 hardware-promoted macro checkpoint

The first substantial full-model Phase-2 win is now implemented and hardware-measured. The
compiler's scalar lane formerly expanded round-to-nearest-even and float-to-signed-integer
conversion into IEEE bit manipulation for every quantized tensor element. The promoted candidate
emits the standard target-neutral LLVM operations instead; the RISC-V backend selects native
`fcvt` instructions. No ResNet name, layer ID, shape guard, Gemmini tile, command, DMA request,
fence, or task boundary changes.

The exact one-warm/one-measured comparison is:

| engine | q530 predecessor | optimized | saved | reduction | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spike comparative proxy | 976,190,118 | 751,827,143 | 224,362,975 | 22.9835% | 1.2984x |
| FireSim/U250 q530 -> q534 | 1,704,064,223 | **1,537,416,019** | **166,648,204** | **9.7795%** | **1.1084x** |

Both runs checked all 1,000 logits exactly (`bad=0`, `nonfinite=0`, top-1 258, checksum
`c6e777c3fe0aae90`). Spike therefore ranked a material win correctly but overpredicted saved cycles
by 1.3463x. This validates the Phase-2 policy: use the cheap engine for correctness and within-engine
candidate ranking, then use hardware only for a materially changed promoted checkpoint.

Queue job 534 used only `/scratch/firesim_queue/bin/firesim-queue` and completed the required
`firesim kill -> firesim infrasetup -> firesim runworkload -> firesim kill` lifecycle. Eight sparse
completion markers were confined to the untimed warm entry; the mutable arena reset was untimed;
the measured entry carried no markers. Package, staged and executed ELF SHA-256 all equal
`0b3284a815da3bd6f8b8e0a014f936b0ffc5c5a2c6666916ced45f6c716f62c5`. The HWDB remained
byte-identical throughout. The sealed receipt is:

`resnet50_merlin_phase2_native_scalar_ops_w8a8_warm_measured_firesim_candidate_20260908/validation/firesim_queue_job_534_success.json`

The independently verified reusable compiler artifact is:

`development_bf62_target_neutral_epilogue_fusion_20260908`

Its compiler tree SHA-256 is
`0b1e14e9b61dd3a4eb3472a87702cc53c5a05b98198c08cec3bfbce85363db8a`; 14/14 focused tests pass.
All four portfolio graphs compile with the same snapshot. ResNet is exact locally and on FireSim;
TinyLLaMA still exposes the known all-host placement gap; LSTMNetViT and SmolVLA preserve their
prior command-buffer identities; generic one-byte `i1` storage sizing repairs SmolVLA compilation.

q534 also sharpens the next work order. It still retires 751,827,149 guest instructions and moves
1,142,808,576 read-DMA plus 54,136,736 write-DMA bytes, while `loop_matmul_active` is only
53,479,788 cycles (3.48% of the measured interval). The next macro lever remains exact contraction
epilogue/boundary formation, then narrow-layout propagation into capability-selected native
convolution, then a real second-tensor residual operation. Blanket synchronization removal is
deprioritized: q530 already proved it worth only 0.6277%. Dynamic-activation quantization for
weight-only transformer graphs must remain an explicit, default-off numeric contract because f32
source bit-equivalence cannot be claimed.

The concise current invocation, four-model table, and artifact map are in the tracked top-level
`PHASE2_CURRENT_CHECKPOINT_20260908.md` beside this handoff.

## 27. Canonical compiler assembly and cross-model promotion gates

The current directly invocable compiler is now a single isolated assembly rather than three
disconnected prototypes:

`development_phase2_canonical_multimodel_20260908`

Compiler tree SHA-256:
`38e42aea90f29e271ddc510daca09901cc36970323e2926d1d9426a6e6ee1fdf` over 42 files.
It combines only orthogonal, byte-traceable mechanisms:

| mechanism | default | evidence | current full-model effect |
| --- | --- | --- | --- |
| standard LLVM `roundeven` + `fptosi` | on | exact q534 FireSim | 9.7795% ResNet cycle reduction |
| capability/layout/semantics-selected native `LOOP_CONV_WS` | on, fail closed | 20 tests + two exact reduced Spike cases | 0/53 ResNet; records narrow/NCHW dependency |
| dynamic-activation/per-channel-static-weight i8 bridge | off | 15 tests + frozen TinyLLaMA numeric oracle | TinyLLaMA 0 -> 15 mesh contractions when explicitly enabled |
| byte-addressed `i1` storage | on | unit + full SmolVLA compile | removes SmolVLA compile blocker |

The combined suite passes 31/31. The default compiler recompiles ResNet, TinyLLaMA, LSTMNetViT,
and SmolVLA sequentially with a maximum observed compiler RSS of 308,860 KiB. ResNet target LLVM
is byte-identical to q534; its only command-buffer addition is a 53-entry native-convolution
selection/refusal audit. The other three default command buffers and target modules are
byte-identical to the q534 scalar artifact. This is the compiler artifact a downstream evaluator
should invoke; its README contains the exact command and its verifier binds all transferred
evidence.

The TinyLLaMA opt-in command is:

```text
--dynamic-weight-only-contract symmetric_per_output_channel_roundeven_v1
```

It produces 15 mesh contractions, 45 accelerator commands, 3,424,256 declared accelerator MACs,
and 540,032 declared transition bytes. A cheap independent NumPy oracle over the frozen capsule
witness reports 0/2,048 declared-tolerance violations, max-absolute error 0.018019676,
relative-L2 0.007236148, cosine 0.999973894, and identical top-1 for 8/8 tokens. The contract remains
default-off because this is not source-f32 bit equivalence and one frozen witness is not a dataset
accuracy claim.

### 27.1 Exact native-convolution dependency, now executable

The target dialect, funct-15-through-21 encodings, capability selector, and reduction tiler exist
and execute. A stride-2/pad-1 3x3 witness checks 192 outputs exactly; a forced `CI=1024`
two-descriptor reduction checks 144 outputs and proves accumulator continuation. The selector
admits only i8 NHWC + HWIO + narrow i8 NHWC with no unrepresented epilogue.

This target's `LOOP_CONV_WS` store is narrow. `trans_output_1203` is RTL-proven HWNC, not NCHW.
Per-channel store scale is global `CONFIG_ST` state and would race concurrent slots unless
serialized. Therefore the current frozen ResNet boundary—raw i32 NCHW—is legitimately ineligible.
The remaining work is not “implement loop-conv”; it is form an exact narrow epilogue and globally
choose/propagate a compatible encoding, or prove a separate full-width accumulator readout.

## 28. q534 empirical headroom and honest roofline status

The exact machine-readable headroom artifact is:

`q534_whole_model_headroom_roofline_20260908`

It reuses the repository's `merlin.perf.roofline` implementation and deliberately preserves its
refusal. q534 supplies an exact workload observation but not enough calibration evidence for a
physical empirical roofline. In particular, no bandwidth is inferred from request bytes divided by
whole-program time, no theoretical bandwidth is substituted for a measured sustainable peak, and
no max/sum/partial-overlap composition operator is guessed.

Available exact q534 observations:

| quantity | value |
| --- | ---: |
| measured compute cycles | 1,537,416,019 |
| content-addressed workload MACs | 4,091,330,560 |
| end-to-end rate | 2.661 MAC/cycle |
| Gemmini read + write DMA | 1,196,945,312 bytes |
| arithmetic intensity | 3.418 MAC/DMA-byte |
| loop-matmul occupancy | 3.479% |
| reservation-station occupancy | 3.505% |

The q530-to-q534 pair has identical accelerator schedule and DMA volume. Removing 229,709,973
instructions saved 166,648,204 hardware cycles, giving a local sensitivity of 0.72547 cycles per
removed instruction. A two-point scalar-only extrapolation reaches about 991,986,308 cycles, only
about 8.0 million below the one-billion target. This is explicitly non-predictive: it folds fixed
memory cost and overlap into an intercept. Its use is prioritization, not publication as a
roofline—it proves that host scalar deletion alone leaves essentially no safety margin and that
epilogue/boundary deletion must be coupled to movement/im2col reduction.

A resolved formal empirical roofline still requires the RTL-derived calibration plan, at least four
distinct raw samples per fitted resource, four exact empty-run baselines per measurement protocol,
RTL-bound traffic counter receipts, a complete joint-occupancy partition for composition/eta, and
the full expected workload observation set. Until those exist, the Phase-2 controller should expose
the exact utilization/intensity/headroom observations and the refusal, not a fabricated limiter.


## 29. q535 affine-im2col hardware promotion

The next macro candidate optimized the host implementation of row-streamed im2col using only
static convolution geometry. It classifies the common valid output-column interval, hoists row
bases and y-validity decisions, emits affine guardless interior copies, and retains the exact
guarded gather for borders. There are no model names, layer IDs, shape allowlists, arithmetic
approximations, or target-specific schedule decisions.

The complete ResNet analysis accounts for 14,613,760 logical packed bytes. Repeated x/y bounds and
clamps are removed for 13,628,944 bytes (93.26%), and 12,757,195 bytes (87.30%) pass through fully
guardless interior loops. The accelerator command buffer is byte-identical to q534: 3,787
`LOOP_WS` launches, 1,050 fences, and exactly the same declared DMA/schedule decisions.

Exact one-warm/one-measured evidence:

| comparison | predecessor | candidate | saved | reduction | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spike q534 -> affine im2col | 751,827,143 | 506,265,226 | 245,561,917 | 32.6620% | 1.48505x |
| FireSim q534 -> q535 | 1,537,416,019 | **1,316,619,699** | **220,796,320** | **14.3615%** | **1.16770x** |
| FireSim q530 -> q535 cumulative | 1,704,064,223 | **1,316,619,699** | **387,444,524** | **22.7365%** | **1.29427x** |

All 1,000 logits remain bit-exact. Spike overpredicted saved hardware cycles by 1.11216x, but
correctly ranked a substantial win. FireSim reports the same 1,142,808,576 read-DMA and 54,136,736
write-DMA bytes, isolating the gain to host dynamic-work deletion.

Queue job 535 used only `/scratch/firesim_queue/bin/firesim-queue`. The daemon executed and
hash-verified exactly
`firesim kill -> firesim infrasetup -> firesim runworkload -> firesim kill`.
The ELF performed one sparse-marker warm inference, an untimed mutable-arena reset, and one
uninstrumented measured inference. Package, staged, and executed ELF hashes all equal
`86aea9b8bc6aad86487e652a51629f76dcd472201c85e19077ecb7fac3dbde78`.
The measured window alone is the result; 3,945,377,522 simulator-total cycles include warmup and
harness work and are excluded.

The sealed package is
`resnet50_merlin_phase2_affine_im2col_spans_w8a8_warm_measured_firesim_candidate_20260908`.
Portable evidence is in `q535_affine_im2col_hardware_result_20260908.md`,
`q535_affine_im2col_hardware_receipt.json`, and `phase2_affine_im2col_q535.patch`.

This result improves the end-to-end rate from q534's 2.661 to 3.107 MAC/cycle, but loop-matmul and
reservation-station occupancy remain only 4.065% and 4.096%. Those are diagnostics, not a calibrated
roofline. q535 still performs all 14.6 MB of logical packing and retains 546 packing-related
synchronization points. The next macro lever remains deleting the epilogue/layout boundary so
capability-selected native convolution can delete packing, then handling true two-tensor residual
chains. Tile search and latency hiding follow boundary/movement deletion.

## 30. q536 hybrid native-convolution diagnostic

Queue job 536 measured **555,991,472 cycles** with 1,000/1,000 int8 logits exact under the required
queue-only `firesim kill -> firesim infrasetup -> firesim runworkload -> firesim kill` lifecycle.
It validates 53 Merlin-generated native `LOOP_CONV` kernels and demonstrates sub-billion headroom,
but a TVM-generated runner owns the graph, arena, residuals, pooling, and dense. Its 2.368x ratio
versus q535 is not a Merlin compiler speedup. The controller may mine only the general kernel,
bias, descriptor, capability, and warm/reentrant mechanisms. Evidence is
`q536_native_loopconv_hybrid_diagnostic_20260908.md` and
`q536_native_loopconv_hybrid_receipt.json`.

## 31. Final combined compiler and Phase-2 output

The directly invocable artifact is
`development_phase2_final_combined_exact_residual_global_encoding_20260908`, with 45-file compiler
SHA-256 `48694957d14c9608960f7ac2b6cd08b72b15c55e4a11ff26e1cf952e0cd7607e` and review-patch SHA-256
`0e611e3c9c2231196653bfdb906a45843b89f8ee14f4ffbefe683116a9f2477a`. Its verifier binds 65
focused tests, two exact warm/reentrant executions, all four complete-model compiles, the opt-in
bridge, ownership/ABI/encoding/refusal audits, the final receipt, and q535 hardware lineage.

| general mechanism | final state | fast signal |
| --- | --- | --- |
| native scalar conversion | preserved from q534 | exact full-model hardware win in lineage |
| affine segmented im2col | preserved from q535 | exact full-model hardware win in lineage |
| ordered quantized epilogue | implemented, fail closed | 323 exact outputs; 13,760->56 proxy cycles |
| second-tensor residual | 15 ResNet sites | tasks 109->105; 7,340,032 intermediate bytes removed |
| global encoding/lifetime solver | implemented, fail closed | 2 native convs, 0 im2col, 96/96 exact, 102 cycles |
| native convolution | capability-selected | q536 headroom; frozen PT2E 0/53 exact narrow |
| dynamic-weight bridge | explicit opt-in | qualified numerically; no source-f32 bit claim |

Four-model simulator-free gating compiles ResNet (1,240 source ops, 54 accelerator/51 host tasks),
TinyLLaMA (718, 0/1), LSTMNetViT (2,302, 37/38), and SmolVLA (11,910, 116/117), peaking at
308,264 KiB RSS. The other-model target identities are non-regressions, not performance wins.

The final ResNet target is byte-identical to the residual child and transfers its exact warm/local
A/B: 506,265,226->492,147,976 Spike proxy cycles (2.7885%; 1.028685x), 1,000/1,000 logits exact.
There is no hardware result for the changed object; q535 at 1,316,619,699 cycles remains the latest
honest full-Merlin FireSim checkpoint.

Frozen PT2E ResNet admits 0/53 exact native-narrow convolutions because its per-channel scales and
FP32 bias cannot be represented by one scalar store scale. Conv1's naive fold mismatches
143/802,816 values before pooling and 33/200,704 after pooling. The global solver exposes a
conditional 44,455,936->11,113,984-byte boundary opportunity without claiming it was realized.
The next exact target-neutral endpoint is `{narrow_i8, full_i32, accumulator_handle}` plus scale
granularity/axis, zero points, bias/residual domains, ordered FP32 stages, rounding, saturation,
completion, and lifetime. Native-aligned W8A8 remains a separate calibration/golden/accuracy and
provenance contract.

```sh
artifact=/scratch/agustin/projects/oscar-merlin/out/artifacts/perf-bench/gemmini/development_phase2_final_combined_exact_residual_global_encoding_20260908
PYTHONDONTWRITEBYTECODE=1 python "$artifact/verify.py"
MERLIN_PYTHON=/scratch/agustin/projects/oscar-merlin/.venv/bin/python \
  "$artifact/run-gemmini-opt" --source-convolution --convert-iface-to-gemmini \
  --emit-command-buffer=/absolute/output/command_buffer.json \
  --emit-target-artifact -o /absolute/output/target.mlir /absolute/input/model.mlir
```

Object/ELF construction remains runner-owned. No final-combined FireSim, L3, or queue action was
performed; no sub-billion compiler claim or formal roofline is made; Phase 1 stays frozen at 92/96.
The final artifact itself indexes the report/status/receipt/verifier/patch. q535/q536, standalone
epilogue/residual/encoding receipts, and `universal_narrow_epilogue_recovery_v1_20260908` provide
the complete evidence lineage.
