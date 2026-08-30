# Task register — the performance layer

Merlin derives **what a target permits**. This program builds the other half: **what its choices
cost**. Every task the plan requires, with its current state.

`DONE` means implemented **and** verified by a test or a measured run. `PARTIAL` means the mechanism
exists but a required property is unproven. `OPEN` means not started. **A task is not `DONE` because
code exists** — R0 has already found three numbers in this tree that looked settled and were wrong.

Blocking order is noted where one task's output is another's input. Agent-costed tasks say so: the
measured price of one Atlas run is 900–31,061 s and up to $147 / 222 M tokens.

Plan of record: `~/.claude/plans/cryptic-foraging-sparrow.md`.
Rationale for the cost decisions: `docs/design/performance_budget_unit.md`.

---

## The flow

```text
R0  ground truth, cost, eligibility        [gate: decides R6's unit and R7's shape]
     |
     +-- N1  layer-scale workload generation    <-- HARD BLOCKER, newly discovered
     |
     +----------------+----------------+
     |                |                |
    R1 record/term   R2 DMA volume    R3 profile/contract
     |                |                |
     +----------------+----------------+
                      |
                     R4 envelope + gap attribution
                      |
                     R5 harvest + rule registry
                      |
                     R6 bounded selection
                      |
                     R7 headline experiment
```

R2 runs in the mlc repo and reaches the others only through R1's record fields, so it is independent.
R1 and R3 share Lane A. R4 is disjoint until it imports the record type.

---

## R0 — Ground truth, cost reality, eligibility  *(gate)*

| id | task | state |
|---|---|---|
| 0.1 | Measure the budget unit at **capsule** scale | **DONE** — 42 samples/tier, serial. arc L3 median 3.68 s, Verilator L4 median 0.276 s, no build step (llvm-mc 5.7 ms), cost linear in halt cycles (arc 3.63 ms/cyc, L4 0.255 ms/cyc). Artifact + script under `out/artifacts/capsule-bench/atlas/tier_policy/` |
| 0.1b | Measure the budget unit at **layer** scale | **IN PROGRESS** — 0.1 priced 178–2614-cycle toys; the whole corpus is that size (largest tensor anywhere = 5400 elements). 0.255 ms/cyc is ~4000 cyc/s, which is *slow*; the wall was small because the work was trivial. Must validate the linear law over the widest range available, find every `max_cycles` cap, and price a real layer |
| 0.1c | Written verdict naming the scarce unit | **PARTIAL** — recorded for capsule scale (synthesis call dominates by 2–3 orders of magnitude); layer scale is explicitly `pending measurement` in the design note, not decided |
| 0.2 | Pin the ground truth | **DONE** — `atlas_arc_cycle_suite` registered by sha256 `1c36e13f…`; `muon_arc_model` advanced to `add52b0` (fast-forward, `cosim_muon.py` byte-identical). 9 pins + 4 artifacts verify; gate green |
| 0.3 | mlc hygiene | **DONE** — 2 commits in ModeLIR (`92889d0`, `add52b0`): the untracked passes, `circt_study/`, and the characterization evidence base. 35 → 18 dirty, all remaining generated or separately-pinned submodules. Fast suite 822 passed / 8 failed / 5 skipped |
| 0.4 | Freeze the perf-eligible set | **DONE** — **21 capsules** (14 public + 7 hidden) at elaborated RTL, from `merlincirct_atlassg1` (integrity clean). Digest-stamped set under `out/artifacts/capsule-bench-perf-eligible/atlas/v1/latest/`. Corrects a misreading: `functional_pass: 0` is an all-or-nothing flag, not "nothing passed" |
| 0.5 | Decide "unseen shape" in writing | **PARTIAL** — established that shipped programs bind a static `.S` with hardcoded immediates, and `ParameterizedMatmul*Program` does **not exist** in npu_model (only in the dead `calibrate_npu.py`). So shapes can only come from merlin's own emitter. The consequence is written into the design note; the formal decision record is not |
| 0.6 | Enumerate the Atlas agent output contract | **DONE** — a single `kernel.S` of `.word`/`.insn` directives, stock `llvm-mc` + `llvm-objcopy`, EBREAK-terminated, 419–1218 words observed. `kernel_slot.py` is **gemmini-only** and is not the Atlas path |
| 0.7 | Radiance GSIM cycle extraction *(timeboxed side-quest)* | **OPEN** — GSIM passes L3 with `cycles=None`; the adapter already calls `_cycles_from_rtl_report`, but the passing path exits via `stopSim` before a cycle line prints. Drop if it exceeds the timebox — Radiance is fan-out |

---

## N — Discovered during R0 (not in the original plan)

| id | task | state |
|---|---|---|
| **N1** | **Layer-scale workload generation** | **OPEN — HARD BLOCKER.** There is no layer-scale Atlas workload anywhere: 21 eligible capsules are 32×32 tiles, 25 shipped programs are static `.S`, and the "full model shapes" ([241,960], 1024×3072, (50,720)) are docstring text with no program, golden or run. Only merlin's emitter can produce them. **Every performance claim about layers depends on this.** Blocks 0.1b's large run, R7, and the meaning of R5's rules |
| N2 | Re-declare the Atlas measurement authority against L4 | **OPEN** — L4 Verilator is 13× cheaper than arc *and* `derived_from_rtl: true` / `elaborated_rtl` where arc is not, and the two report **identical cycles on 14/14 capsules where both tiers ran** (directly measured, 42 samples, `oracle_query_cost_atlas.json`). NOTE the evidence provenance: `score_capsule.json` cannot support this claim — its tier records are bare strings (`"pass"`), and `cycles_diagnostic` is L3-only, so an earlier citation of "identical across all 21, e.g. AT2 = 3078 at both" was wrong on both count and number (3078 is one submission's L3; a different submission gives 1090 at both). The declared `cycles_from: arc_program` / `cycles_tier: cycle_model` understates what is obtainable; re-declaring upgrades every number to tier `rtl` at 1/13 the cost. Edit `out/artifacts/targets/atlas/contracts/residual.yaml` |
| N3 | `cycles_diagnostic` harvests only L3 | **OPEN** — `capsule_grade.py:336` reads `tiers["L3"]["cycles"]`, so the 12 failing capsules' L4 cycles never reach the summary. Small fix; also the natural place to give the dict a comparand (R5) |
| N4 | Delete `merlin/python/merlin/dse/calibrate_npu.py` | **OPEN** — orphaned third cost model; `available()` is permanently `False` (looks in a nonexistent `work_dir()/tmp/dse/npu_model`), and the programs it names are absent from npu_model. Delete the file and its allowlist entry |
| N5 | Persist `tier_policy.record_cost` | **OPEN** — `_COST` is a module dict with no file I/O, so every grader process starts uncalibrated and re-pays the unmeasured-tiers-first probe. ~30 lines, no signature changes. Do **not** reuse `.oracle_timing.json` (different consumer) |
| N6 | Audit `max_cycles` caps on both tiers | **IN PROGRESS** (folded into 0.1b) — `atlas_verilator_run.py::run_program` defaults to `max_cycles=20000`, which a real layer blows through. A run that silently truncates and reports a cycle number is a *wrong* number, not a slow one |
| N7 | mlc `test_discover_runtime_abi.py` — 8 failures | **OPEN, low priority** — pre-existing (`ae8314f`, none of ours). Test-harness bug: a Scala source string passed where a path is expected → `OSError: File name too long`. ModeLIR's SIMT discover layer; unused by Atlas |
| N8 | `speed_of_light: null` — no attainment denominator | **OPEN** — the one candidate model is GEMM-only (3 of 21 kernels) and imports matplotlib at module scope. Until a denominator is *derived*, performance claims are **kernel-relative only** and "% of peak" is unclaimable. Record the reason in `residual.yaml` rather than leaving the field null |
| N9 | Three measured defects in mlc's Atlas **L2 functional core** | **OPEN — a peer session offered to take it.** In `mlc/backends/func_program_atlas.py`: (a) `dma.config.chN` and `dma.wait.chN` encode to the identical word — `RType.to_bytecode` never encodes the `imm` field that separates them; (b) VMEM read as byte-addressed where the RTL is word-addressed; (c) `VUNPACK_FP8_BF16`'s scale read as a divisor where the RTL reads a biased exponent. **Corroborated three ways**: the atlas descriptor already declares L2 inapplicable citing (a); and L2 overcounts on **three same-submission points** (`_tierpolicy_v3` kernels, from `L2_functional_probe` in `oracle_query_cost_atlas.json`), where L3 and L4 agree exactly: AT1 543 vs 178 (3.05×), AT2 3081 vs 1090 (2.83×), CT0 6781 vs 2614 (2.59×). **Same submission, so this is a real mis-model, not submission variance** — do not confuse 3081 with merlincirct_atlassg1's AT2 = 3078, which is a different submission's L3 number three apart. The **fix gate** is those three exact values (178 / 1090 / 2614) on the `_tierpolicy_v3` kernels: three points, each with a known-wrong current value to move away from. A new third number is not a fix. That the ratio *varies* by program (2.59–3.05×) is itself consistent with the dma-word collision — a constant collapse to 1.00× would mean the collision was the whole story; a partial close means VMEM addressing contributes too. Payoff if fixed: a cheap tier that can see inside the 12 currently-undiagnosable atlas failures, **and** per-op observability for R5 we otherwise have to get from GSIM |
| N10 | `DispatchFacet.dma_overlap` reports overlap that does not exist | **OPEN — owned by a peer session.** `merlin/python/merlin/kernels/cca.py:701` computes `bool(counts.get("dma"))` while documented as "movement issued to OVERLAP with compute", returning True for all 114 atlas kernels that provably have none (DMA ops 2567, DMA.WAIT 2567 — exactly 1:1). Matches our independent measurement of overlap **exactly 0.0** across all 21 kernels. Asked that the fixed facet distinguish *DMA present* from *DMA overlapped* — R2.3 needs the second |
| N11 | Consume the peer's RTL-derived `timing` fact class; do not build a second | **OPEN — integration point.** A peer is adding per-unit pipeline depth / initiation interval / in-flight depth to `targetgen/rtl/circt_introspect.py`, derived structurally, UNKNOWN when the walk cannot establish it. R3 consumes these as terms with provenance. Their finding, which R5.8 must respect: npu-model ships a flat `MXU_OP_LATENCIES{vmatmul.acc.mxu0: 96}` that **conflates II with completion latency** — the RTL carries two numbers (II 33, completion ~94, `inflightDepth` 3), and the shipped corpus schedules to both (matmul `op_stream` delays: 34, 34, 32, **96**, 32). Importing the flat dict would have made merlin conclude the corpus was under-delaying. Cross-check against the zero-fit characterization terms (MXU0 per-tile 192 = 130 + 2·DIM−2; MXU1 132 = 130 + numPipeCuts+1); a disagreement is a finding, not noise |
| N12 | Oracle ladder short-circuited on a refuting tier failure | **FIXED by a peer (merlin `396c3f7a`)** — a mandatory tier failure raised from inside the tier loop, so every tier ordered *after* the refuter recorded nothing. Verified on `merlincirct_atlassg1`: all 14 passing capsules carry both L3 and L4; **11 of the 12 failures carry L4 with no L3 record, and 1 the reverse** — because `tier_order` runs atlas's cheap Verilator before arc. Consequence for us: a missing tier is *absence of a record*, not disagreement. Any atlas cycle comparison drawn from a pre-fix grade must be re-checked against a post-fix one |

---

## R1 — Performance record + minimal term  *(Lane A; concurrent with R2)*

| id | task | state |
|---|---|---|
| 1.1 | `performance_record.schema.json` (real JSON Schema) | **OPEN** — the digest triple must be a **required** field from the first record written, or every artifact produced before it is uncitable |
| 1.2 | `performance_term.schema.json` + minimal `PerformanceTerm` | **OPEN** — `value \| unit \| provenance \| confidence \| validity \| bounds`. **UNKNOWN is a distinct inhabited state that cannot be read as 0.0** (no float default) |
| 1.3 | Emit a record per kernel from `compose_program_cycles` + `attribution.py` + `npu_model_compare` as they stand | **OPEN** — all 21 kernels, under `out/artifacts/` |
| 1.4 | Test: composed prediction reproduces `mxu` 158 (one tile) / 284 (k_chain) exactly | **OPEN** |
| 1.5 | Test: writing a record with a missing digest **raises** | **OPEN** |
| 1.6 | Test: npu_model cycles/`exu_stats` can never source a term | **OPEN** — they disagree with arc by up to 3× (rms_norm 3972 vs 1273); diagnostic only |
| 1.7 | Defer the five-lattice provenance unification | **DEFERRED by design** — until ~10 real terms exist. Five representations exist today; unifying before there is anything to unify will churn |

## R2 — DMA byte-volume and overlap  *(Lane B; mlc repo only)*

**The highest-leverage item in the program.** DMA is 60–93.7% of every Atlas cycle count and the MXU
is never above 13.9% busy on any of the 21 kernels. Measured compute/DMA overlap is **exactly 0.0**
suite-wide.

| id | task | state |
|---|---|---|
| 2.1 | Structural DMA footprint predictor from program descriptors | **OPEN** — extends `predict_dma_volume.py` / `compile_dma_descriptor.py`. Today `footprint_bytes` is a workload *input*; `compose_program_cycles.py` flags it open in its own docstring ("Finding 6", transfer amplification) |
| 2.2 | Match arc-measured `(reads + writes) · beat_bytes` for ≥18/21 kernels | **OPEN** — the ≤3 failures named, cause recorded **UNKNOWN, never fitted** |
| 2.3 | The overlap-policy term | **OPEN** — zero measured overlap means double-buffering / descriptor pipelining is the compiler's largest available win (~2×) and nothing models it |
| 2.4 | Falsifier | **OPEN** — if structural prediction fails, the cycle claim is downgraded **in writing** to "given the byte volume" |

## R3 — Target profile + performance contract  *(Lane A; after 1.2)*

| id | task | state |
|---|---|---|
| 3.1 | `perf/profile.py` — archetypes + traits, **derived never named** | **OPEN** — copy `capability_manifests.derive_manifest`'s 3-source pattern exactly (CIRCT facts + `families.py` + `residual.yaml`) |
| 3.2 | `perf/contract.py` — emit the contract | **OPEN** |
| 3.3 | Keep VMEM capacity **UNKNOWN** | **OPEN** — npu_model's `HardwareConfig` says 1 MiB, RTL says `0x180000` (1.5 MiB), mlc refuses to classify 39 SRAMs. `HardwareConfig` is residual-tier at most, **never a fact source** |
| 3.4 | Test: the same code derives a profile for Atlas **and** Gemmini | **OPEN** — the anti-overfit proof |
| 3.5 | Test: deleting a fact from `facts.json` yields UNKNOWN, not a default | **OPEN** |
| 3.6 | Gate: `check_no_target_name.py` clean **and** `--coupling` adds zero entries under `perf/**` | **OPEN** — `perf/` is by construction "a generic module", so any target-named import registers as coupling debt. Do **not** add allowlist entries |

## R4 — Envelope + gap attribution  *(Lane C; disjoint from A)*

| id | task | state |
|---|---|---|
| 4.1 | Extend `design_envelope.py` / `arithmetic_intensity.py` with the **DMA-bound ridge the data shows** | **OPEN** |
| 4.2 | Map gap components 1:1 onto `attribution.py`'s buckets | **OPEN** — `compute / dma / stall / control / host`; `residual` stays `assumed` and never vanishes |
| 4.3 | Each attributed gap names the optimization family it implies | **OPEN** |
| 4.4 | Invariant test: no prediction falls below the structural bound, across all 21 | **OPEN** |
| 4.5 | Test: attributed buckets sum to the arc total exactly | **OPEN** |

## R5 — Instruction-timing harvest + rule registry

Cheap enough to run over the whole corpus rather than a sample. `npu_model_suite.json`'s `op_stream`
already carries `[unit, mnemonic, n]` per op.

| id | task | state |
|---|---|---|
| 5.1 | Adapters emit a `timing_observations` block | **OPEN** — arc per-op, GSIM per-cycle CSVs, npu_model `exu_stats`, RoCC timestamps. An adapter with no timing capability emits **nothing, never zeros** |
| 5.2 | `perf/harvest.py` + retro-mine runs already on disk | **OPEN** |
| 5.3 | Rule: harvested latencies are **contended upper bounds** | **OPEN** — `trace_derived`, validity domain naming what else was active, spread recorded. Test-enforced |
| 5.4 | Rule: a harvested term can never be promoted to `calibrated` without a dedicated experiment | **OPEN** — test-enforced |
| 5.5 | Rule: only substrates the `MeasurementAuthority` declares citable contribute | **OPEN** |
| 5.6 | Rule registry as YAML data under `merlin/contract/perf_rules/` | **OPEN** — YAML is not scanned by the name gate; a `.py` there naming a target *is* |
| 5.7 | Fitted structural equations, **≥2 points per fitted parameter** | **OPEN** — one `macs_per_cycle` cannot price a tiled unit |
| 5.8 | Registry re-derives the known constants and nothing else | **OPEN** — `DIM`, `fill = 2·DIM−2`, `beat_bytes`, DMA `base_latency`. `npu_model_suite.json`'s `_meta` gives beat_bytes 32, mxu_dim 32, vpu_lanes 16, reset_cycles 12 |
| 5.9 | Give `cycles_diagnostic` a comparand | **OPEN** — pairs with N3; a small change, **not** a new capsule kind |

## R6 — Bounded candidate selection  *(unit set by 0.1b)*

| id | task | state |
|---|---|---|
| 6.1 | Two axes only: DMA tiling / descriptor shape, and overlap policy | **OPEN** — 21/21 reference kernels are single-op single-tile; a five-level hierarchy is unsupported |
| 6.2 | Selection via `tier_policy` + `oracle_schedule` + the three sanctioned methods, or `mining/beam.py` | **OPEN** — **no new beam under `perf/`**; `dse/search/AGENT.md` states a repo-level stance, and choosing a different directory to route around it is rules-lawyering |
| 6.3 | Denominate the budget in the **measured** scarce unit | **BLOCKED on 0.1b** |
| 6.4 | Drop `Generality` from VOI | **DECIDED** — with one target it is a constant, hence not a factor |
| 6.5 | Stop conditions as predicates with unit tests over a fake evaluator | **OPEN** |

## R7 — Headline experiment  *(split; no reference exists off-corpus)*

| id | task | state |
|---|---|---|
| 7.1 | **"Recovers"** — fraction of reference on the 21 eligible capsules | **OPEN** — kernel-relative only (N8: no % of peak) |
| 7.2 | **"Predicts"** — prediction accuracy at shapes merlin emits | **BLOCKED on N1** — no shipped reference exists off-corpus, so only predicted-vs-measured is claimable |
| 7.3 | Report 7.1 and 7.2 **separately** | **OPEN** — conflating them reports a prediction result as a recovery result |
| 7.4 | Small→large: error, Spearman/Kendall, top-K recall, regret **vs size** | **BLOCKED on N1** |
| 7.5 | Convergence curve over the measured scarce unit | **BLOCKED on 0.1b** |

---

## Standing constraints

- **Cycles are a property of the submission, not the capsule.** `AT2` measured 1090 / 3078 / 8889
  across three submissions — an 8.2× spread on identical inputs. Freeze the capsule *set*; never
  freeze a cycle number.
- **Every Atlas number carries a source digest.** The atlas pins report permanent drift by design, so
  a commit sha alone is not provenance.
- **Report per-query cost with its concurrency.** A 16-worker grade inflated the same arc query from
  3.7 s to 23.4 s (6.3×). A cost measured under parallelism is a throughput figure wearing a latency
  figure's clothes.
- **Tests go in an existing bucket** — `{dse, gemmini, infra, ir, kernels, runtime, rvv, targetgen}`.
  There is no `perf` bucket and the list is an enum. Profile/contract/record → `targetgen`;
  envelope/attribution/selection → `dse`.
- **A check that could not run is `not_run`**, never a pass and never a zero.
- **mlc lives in a NESTED git repo.** `$MERLIN_MLC_DIR` = `/scratch2/agustin/mvp-lhwir/modeling` is its own
  repo (`copparihollmann/ModeLIR`, branch `feature/discover-datapaths`); the outer `mvp-lhwir` repo
  separately tracks copies of the same files and reports a different branch, a different HEAD, ~971 dirty
  paths, and this pin's commit as a *missing object*. Two sessions have now lost time to it. Resolve the
  path the way the code does: `provenance.verify("muon_arc_model", checkout=mlc_bridge.mlc_dir())`.

## Explicit non-goals

The A0–A8 ablation matrix and its config flags · Radiance and Gemmini adapters · the generic lift of
`compile_ilp_rate.py` (1160 lines for 7% of Atlas cycles, whose customer is deferred) · a new `perf`
capsule kind · learned residual models · rewriting mlc's dialects into a real MLIR IR · agentic
optimization loops (Codex enters only at fan-out).
