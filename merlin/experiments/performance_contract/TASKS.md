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

## The deliverable is TOOLS, not findings

Everything in this register that reads as an Atlas fact was derived **by hand, in one session**. That
is the failure mode this program exists to end: a target should be onboarded by *running tooling*, not
by an agent re-deriving "is this DMA-bound?", "where is the headroom?", "which capsules are the right
instrument?" from scratch each time.

So the rule for every task below:

> **Ship a generic, trait-gated analysis. The Atlas numbers already measured are its regression
> fixtures — the tool is wrong if it does not reproduce them, and overfit if it cannot produce a
> different, correct answer for a second target.**

Gating is on **traits**, never on archetype and never on a target name (`archetype` is only a prior —
it decides *which questions to ask*; the RTL-derived traits decide *which of them apply*). The
no-target-name gate over `merlin/python/merlin/**` enforces this by construction.

| hand-derived today | becomes | trait that gates it | must reproduce (Atlas) |
|---|---|---|---|
| "Atlas is DMA-bound, 60–93.7%" | `perf/decompose.py` — bottleneck decomposition from any per-unit activity source | target exposes a per-unit activity decomposition | matmul dma 86.2% / mxu 6.6% |
| "overlap headroom is 14.7%, not 2×" | `perf/headroom.py` — Amdahl bound `min(a,b)` over any concurrency-capable resource pair | ≥2 engines with independent ports + explicit completion | 4,457 cyc total; gelu_tanh 23.9%; matmul 6.6% |
| "amplification is 9–28×" | `perf/amplification.py` — bytes moved vs bytes useful | explicit DMA / managed scratchpad | matmul 16.0×, rms_norm 24.0×, gelu_tanh 28.0× |
| "these 12 capsules are the optimize set" (N22) | `perf/workload_roles.py` — classify each workload by dominant term and available headroom | any target with a cost decomposition | the 12/1/3 OPTIMIZE / dma-calib / fixed-calib split |
| "cost = a + b·cycles + c·words" (R0.1b) | `perf/oracle_cost.py` — two-term fit **plus the isolation protocol** (a program whose first word is the halt separates the load term) | any substrate that runs a program | L4 0.131 ms/cyc, L3 1.553; 2.8% median error on held-out |
| "these RTL facts matter" | `perf/feature_rules.py` + `merlin/contract/perf_rules/*.yaml` | — (it *is* the registry) | emits exactly the experiment set that re-derives DIM, `2·DIM−2`, `beat_bytes`, DMA base latency |

**The anti-overfit gate, applied to every one of them:** the same code must run on Atlas *and* Gemmini
and produce **different, correct** answers. A tool that only works where it was written is the manual
overfitting with extra steps. Gemmini is the right second target here precisely because it is a
different archetype (decoupled queue/systolic vs tensor/dataflow), so a tool that silently assumes
Atlas's shape will visibly fail rather than quietly agree.

**What stays hand-derived, legitimately:** the *fixtures*. Someone has to measure Atlas once to know
what the tool should say. The error is stopping there.

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
| 0.1b | Measure the budget unit at **layer** scale | **DONE** — the law has **two terms**: `sim_s = a + b·cycles + c·words` (IMEM load), and the word term is ~2× the cycle term on both tiers. A cycles-only fit overstated L4 by 1.77×. Real rates: **L4 7,641 cyc/s (0.131 ms/cyc), L3 644 cyc/s (1.553 ms/cyc)**; L4 is 11.9× faster *and* higher fidelity. Linear across **12,500×** (r² 0.99997); two-term fit predicts 52 held-out corpus queries at 2.8% median error |
| 0.1c | Written verdict naming the scarce unit | **DONE** — toy scale: oracle is 3.8% of a run (77 s L3 + 7 s L4 for all 26, vs median agent run 2,184 s). Layer scale: **L4 never binds** (worst ~35 min), **L3 does** (1.6–6.8 h/query, 2.7–11.3× a median agent run, and it is the tier most capsules declare). **But the real verdict is that cost is not the binding constraint — correctness is** (N13–N15) |
| 0.2 | Pin the ground truth | **DONE** — `atlas_arc_cycle_suite` registered by sha256 `1c36e13f…`; `muon_arc_model` advanced to `add52b0` (fast-forward, `cosim_muon.py` byte-identical). 9 pins + 4 artifacts verify; gate green |
| 0.3 | mlc hygiene | **DONE** — 2 commits in ModeLIR (`92889d0`, `add52b0`): the untracked passes, `circt_study/`, and the characterization evidence base. 35 → 18 dirty, all remaining generated or separately-pinned submodules. Fast suite 822 passed / 8 failed / 5 skipped |
| 0.4 | Freeze the perf-eligible set | **DONE** — **21 capsules** (14 public + 7 hidden) at elaborated RTL, from `merlincirct_atlassg1` (integrity clean). Digest-stamped set under `out/artifacts/capsule-bench-perf-eligible/atlas/v1/latest/`. Corrects a misreading: `functional_pass: 0` is an all-or-nothing flag, not "nothing passed" |
| 0.5 | Decide "unseen shape" in writing | **PARTIAL** — established that shipped programs bind a static `.S` with hardcoded immediates, and `ParameterizedMatmul*Program` does **not exist** in npu_model (only in the dead `calibrate_npu.py`). So shapes can only come from merlin's own emitter. The consequence is written into the design note; the formal decision record is not |
| 0.6 | Enumerate the Atlas agent output contract | **DONE** — a single `kernel.S` of `.word`/`.insn` directives, stock `llvm-mc` + `llvm-objcopy`, EBREAK-terminated, 419–1218 words observed. `kernel_slot.py` is **gemmini-only** and is not the Atlas path |
| 0.7b | **Atlas GSIM evaluated — SUPPLEMENT: replace the L4 runner, keep the L4 tier** | **DONE.** **50.7× Verilator** (377,335 vs 7,441 cycles/s), both arms re-measured in one session so the ratio is load-cancelling; the Verilator re-measure lands within 3% of the prior artifact, cross-validating the method. **17/17 runs match Verilator exactly** on cycles, output bytes and DMA read/write counts. **Arbitrary `kernel.S` runs with zero recompiles** via the `imemTL` TileLink port the Verilator harness already uses. Bring-up was **62 s**, no FIRRTL regeneration, zero GSIM engine patches, assertions left live. **The unblock**: the prior characterization built M2 (`SystolicArray`) and M3 (`ChipTop`) and *neither is the oracle's DUT* — the oracle verilates `AtlasCore`, whose `.fir` was already on disk. Layer projection: `gelu_tanh` 8.5–36 min → **10–42 s** (extrapolates ~500× past the 7,817-cycle measured ceiling — flagged as extrapolation, not measurement) |
| 0.7 | Radiance GSIM cycle extraction *(timeboxed side-quest)* | **OPEN** — GSIM passes L3 with `cycles=None`; the adapter already calls `_cycles_from_rtl_report`, but the passing path exits via `stopSim` before a cycle line prints. Drop if it exceeds the timebox — Radiance is fan-out |

---

## N — Discovered during R0 (not in the original plan)

| id | task | state |
|---|---|---|
| **N1** | **Layer-scale workload generation** | **DONE — the blocker is cleared, verified independently.** A **416×832×416** matmul (**143,982,592** MACs, 13×26×13 = 4,394 tile passes) **ran, halted and was bit-exact** on the Verilator tier (`derived_from_rtl: true`, `fidelity: elaborated_rtl`): **3,300,328 cycles**, which is **1,263× the largest capsule** (2,614). Three properties matter more than the size: **(1)** the kernel is **96 words, looped** — unrolled it would be 108,011 and overflow IMEM 3.3×, so N16 was solved rather than dodged; **(2)** the footprint is **1,038,336 B fitting disjointly inside the 1,048,576 B window with no wrap**, so the layer was sized to the N13/N26 hazard instead of producing a cycle count from an aliasing run; **(3)** the two traps were **measured, not assumed** — the branch encoding carries its own provenance (*"a 8-trip loop with a 200-cycle body halted at 1636 cycles with imm_scale=2, delay_slots=1"*, confirming N17 experimentally) and so does the settle time. **And the cost law survived a 1,263× extrapolation**: measured **7,750 cyc/s** against the fitted **7,641**, 1.4% apart — validation of `oracle_cost.py` rather than a restatement of it. Artifact: `out/artifacts/perf-workload/atlas/v1/` (record + command buffer + kernel.S) |
| N2 | Re-declare the Atlas measurement authority against L4, **run by GSIM** | **OPEN — revised by 0.7b.** The tier stays L4 (elaborated RTL); the *runner* becomes GSIM, 50.7× faster and cycle-exact against Verilator on 17/17. merlin's L4 already imports its runner by path from `<vsim_dir>/verilator_run.py`, so this needs only a sibling wrapper — the GSIM binary already speaks the identical spec-JSON contract. Original rationale below still holds: — L4 Verilator is 13× cheaper than arc *and* `derived_from_rtl: true` / `elaborated_rtl` where arc is not, and the two report **identical cycles on 14/14 capsules where both tiers ran** (directly measured, 42 samples, `oracle_query_cost_atlas.json`). NOTE the evidence provenance: `score_capsule.json` cannot support this claim — its tier records are bare strings (`"pass"`), and `cycles_diagnostic` is L3-only, so an earlier citation of "identical across all 21, e.g. AT2 = 3078 at both" was wrong on both count and number (3078 is one submission's L3; a different submission gives 1090 at both). The declared `cycles_from: arc_program` / `cycles_tier: cycle_model` understates what is obtainable; re-declaring upgrades every number to tier `rtl` at 1/13 the cost. Edit `out/artifacts/targets/atlas/contracts/residual.yaml` |
| **N2b** | Switching the measurement authority is VERDICT-AFFECTING — coordinate it | **OPEN, deliberately not done.** `citable` is a floor, so raising `citable_tier` from `cycle_model` to `rtl` **retroactively makes every already-reached `cycle_model` result non-citable**, and changing `cycles_from` stops `pick()` matching records labelled `arc_program`. Both are correct to want — the Verilator tier is 11.9× cheaper *and* `derived_from_rtl: true` where arc is false, with identical cycles on 14/14 — but landing them quietly while other sessions grade against this contract would silently change what their results mean. The refuted "no FIRRTL leg" claim in the residual **has** been corrected (no behaviour change). The fast FIRRTL simulator is **not** declared, because it is not yet wired as a merlin tier and declaring an authority no adapter can run is precisely what that field exists to prevent |
| N3 | `cycles_diagnostic` harvests only L3 | **OPEN** — `capsule_grade.py:336` reads `tiers["L3"]["cycles"]`, so the 12 failing capsules' L4 cycles never reach the summary. Small fix; also the natural place to give the dict a comparand (R5) |
| N4 | Delete `merlin/python/merlin/dse/calibrate_npu.py` | **OPEN** — orphaned third cost model; `available()` is permanently `False` (looks in a nonexistent `work_dir()/tmp/dse/npu_model`), and the programs it names are absent from npu_model. Delete the file and its allowlist entry |
| N5 | Persist `tier_policy.record_cost` | **OPEN** — `_COST` is a module dict with no file I/O, so every grader process starts uncalibrated and re-pays the unmeasured-tiers-first probe. ~30 lines, no signature changes. Do **not** reuse `.oracle_timing.json` (different consumer) |
| N6 | Audit `max_cycles` caps on both tiers | **DONE** — split into N13–N16 — `atlas_verilator_run.py::run_program` defaults to `max_cycles=20000`, which a real layer blows through. A run that silently truncates and reports a cycle number is a *wrong* number, not a slow one |
| N7 | mlc `test_discover_runtime_abi.py` — 8 failures | **OPEN, low priority** — pre-existing (`ae8314f`, none of ours). Test-harness bug: a Scala source string passed where a path is expected → `OSError: File name too long`. ModeLIR's SIMT discover layer; unused by Atlas |
| N8 | `speed_of_light: null` — no attainment denominator | **OPEN** — the one candidate model is GEMM-only (3 of 21 kernels) and imports matplotlib at module scope. Until a denominator is *derived*, performance claims are **kernel-relative only** and "% of peak" is unclaimable. Record the reason in `residual.yaml` rather than leaving the field null |
| N9 | Three measured defects in mlc's Atlas **L2 functional core** | **OPEN — a peer session offered to take it.** In `mlc/backends/func_program_atlas.py`: (a) `dma.config.chN` and `dma.wait.chN` encode to the identical word — `RType.to_bytecode` never encodes the `imm` field that separates them; (b) VMEM read as byte-addressed where the RTL is word-addressed; (c) `VUNPACK_FP8_BF16`'s scale read as a divisor where the RTL reads a biased exponent. **Corroborated three ways**: the atlas descriptor already declares L2 inapplicable citing (a); and L2 overcounts on **three same-submission points** (`_tierpolicy_v3` kernels, from `L2_functional_probe` in `oracle_query_cost_atlas.json`), where L3 and L4 agree exactly: AT1 543 vs 178 (3.05×), AT2 3081 vs 1090 (2.83×), CT0 6781 vs 2614 (2.59×). **Same submission, so this is a real mis-model, not submission variance** — do not confuse 3081 with merlincirct_atlassg1's AT2 = 3078, which is a different submission's L3 number three apart. The **fix gate** is those three exact values (178 / 1090 / 2614) on the `_tierpolicy_v3` kernels: three points, each with a known-wrong current value to move away from. A new third number is not a fix. **CAUSE REFUTED (peer collision census):** instrumenting `_resolve_collision` shows the dma.config/dma.wait ambiguity fires **zero times** on all three capsules where the overcount is measured. The collision is a real ISA fact but **cannot explain the gap**, so N9's original stated cause — and the atlas descriptor's — attribute it to the wrong defect. Whatever drives the uniform ~2.6–3× overcount is now first; the collision is a separate correctness issue for DMA-driven kernels. Amend the descriptor's stated reason too. Payoff if fixed: a cheap tier that can see inside the 12 currently-undiagnosable atlas failures, **and** per-op observability for R5 we otherwise have to get from GSIM |
| N10 | `DispatchFacet.dma_overlap` reports overlap that does not exist | **OPEN — owned by a peer session.** `merlin/python/merlin/kernels/cca.py:701` computes `bool(counts.get("dma"))` while documented as "movement issued to OVERLAP with compute", returning True for all 114 atlas kernels that provably have none (DMA ops 2567, DMA.WAIT 2567 — exactly 1:1). Matches our independent measurement of overlap **exactly 0.0** across all 21 kernels. Asked that the fixed facet distinguish *DMA present* from *DMA overlapped* — R2.3 needs the second |
| N11 | Consume the peer's RTL-derived `timing` fact class; do not build a second | **OPEN — integration point.** A peer is adding per-unit pipeline depth / initiation interval / in-flight depth to `targetgen/rtl/circt_introspect.py`, derived structurally, UNKNOWN when the walk cannot establish it. R3 consumes these as terms with provenance. Their finding, which R5.8 must respect: npu-model ships a flat `MXU_OP_LATENCIES{vmatmul.acc.mxu0: 96}` that **conflates II with completion latency** — the RTL carries two numbers (II 33, completion ~94, `inflightDepth` 3), and the shipped corpus schedules to both (matmul `op_stream` delays: 34, 34, 32, **96**, 32). Importing the flat dict would have made merlin conclude the corpus was under-delaying. Cross-check against the zero-fit characterization terms (MXU0 per-tile 192 = 130 + 2·DIM−2; MXU1 132 = 130 + numPipeCuts+1); a disagreement is a finding, not noise |
| N12 | Oracle ladder short-circuited on a refuting tier failure | **FIXED by a peer (merlin `396c3f7a`)** — a mandatory tier failure raised from inside the tier loop, so every tier ordered *after* the refuter recorded nothing. Verified on `merlincirct_atlassg1`: all 14 passing capsules carry both L3 and L4; **11 of the 12 failures carry L4 with no L3 record, and 1 the reverse** — because `tier_order` runs atlas's cheap Verilator before arc. Consequence for us: a missing tier is *absence of a record*, not disagreement. Any atlas cycle comparison drawn from a pre-fix grade must be re-checked against a post-fix one |
| **N13** | **1 MiB DRAM window WRAPS silently on both tiers** | **OPEN — highest priority.** `cosim_atlas._DRAM_WINDOW` and `sim_main_prog.cpp:32` mask addresses with `window-1` and silently drop bytes past the end. A `gelu_tanh` 1024×3072 layer aliases **12×** over it; a 512³ fp8 matmul lands at exactly 1,048,576 B against a 1,048,576 B window. **Nothing in the returned dict flags it — the only cap that returns a WRONG NUMBER rather than an error.** A slow oracle costs money; a silently-aliasing one costs the result. Fix: widen it, or fail closed |
| **N14** | **L4 never passes a cycle budget, and the harness blames the agent for it** | **FIXED.** `program_verilator_adapter.run()` omits `max_cycles`, so every capsule runs at the 20000 default while L3 sizes its budget with `derive_cycle_budget`. Probed: a 25,096-cycle program raises `ProgramDidNotHalt` on the shipped path, and `capsule_runner.py` converts that raise into a tier FAIL **explicitly attributed to "the AGENT's bug."** This is the recurring *harness-limit-reported-as-agent-defect* pattern. **SETTLED — real defect, zero impact on the record.** A peer classified every failing tier record across all atlas runs: **L4 `did_not_halt` = 0 everywhere**; the only two L4 failures are hidden capsules failing on tolerance. So the denominator does not move. Still worth fixing — it fires the moment anything runs longer. **But the same census found the cap DID fire on L2**: 10 `did_not_halt` in `merlincirct_glmnp1`, every one with **L3=None and L4=None** — the tiers that would have decided the capsule never ran, and the run is recorded 0/11. On capsules we jointly measure executing in 178–2614 cycles on RTL. That run must not be cited as an agent result; it is another instance of harness-limit-reported-as-agent-defect, and the strongest argument for the ladder-completion fix (N12), which would have made the misattribution visible in the artifact **FIXED**: the Verilator adapter now sizes its budget with `derive_cycle_budget(cb)` exactly as the arc adapter does. 4 tests guard the symmetry, the scaling and the floor. `derive_cycle_budget`'s own docstring already described this failure — a correct program on a working oracle reported as a missing oracle — so the arc tier had been fixed for it and the Verilator tier was simply missed |
| **N15** | The arc tier has no wall-clock bound, and it is not fixable in merlin | **OPEN — mechanism identified, fix belongs in mlc.** `timeout` reaches only `emit_bundle`; the arc cycle loop is unbounded. A layer-scale command buffer authorising ~10⁸ cycles runs for **days**, holding a worker, undetectably. **Why merlin cannot fix it:** `mlc.backends.cosim_core.large_stack_call` runs the simulator in a *thread* with a bare `t.join()`, and a Python thread executing a native extension cannot be killed — `join(timeout)` would return control while leaking a native thread burning CPU for days, which is worse than blocking. The fix needs a **subprocess boundary in mlc**, or a pre-flight refusal that projects wall clock from the cycle budget and declines to start. Deliberately not half-fixed here |
| N16 | IMEM is 32768 words and today's kernels are fully unrolled | **OPEN** — 3 of the 4 projected layers overflow it 4×–52×. Constrains N1: layer-scale kernels need loops, not unrolling |
| N17 | Atlas control flow is not the RISC-V reading | **RECORDED (fact).** The PC is a **word index** and `branch_target = s1_pc + (imm>>1)`, so a B-type immediate moves `imm/2` *instructions*; there is **one architectural delay slot** (`PcControl.scala`, `ScalarCore.scala`). Encoding a byte offset silently never closes a loop and reads as "this core has no control flow at all" — a derived-fact trap for N1 and for any emitted loop |
| N18 | Buffered-trace OOM hazard — **cleared** | **DONE, no action.** No per-instruction or per-cycle stdout on either tier: a 2,049× longer run produced 0.13% more output (8,277 → 8,288 B, exactly the extra digits of the cycle count) at identical peak RSS (15,360 kB). The 72 GB hazard seen on another simulator here cannot occur |
| **N19** | **Role table fabricated a DMA engine** | **FIXED by a peer (in tree).** `isa_model.roles["memory"]` names a VMEM→mreg load (VLOAD), and `FROM_ISA_ROLE` mapped `memory → dma`. So the `dma` role fired on **local register loads**, while the 32 actual `DMA_{CONFIG,LOAD,STORE,WAIT}_CH*` mnemonics carry **no role at all** and there is no `sync` role. Three facets read it — `dispatch.dma_overlap`, `dispatch.double_buffered_banks`, `memory.dma_pattern` — so all three **answered confidently about a DMA engine they had never seen**; the old `bool(counts["dma"])` was True everywhere because VLOAD is everywhere. Now mapped to `operand_load`. **Confirmed at the instruction level**: `DMA.CONFIG` / `DMA.LOAD` / `DMA.WAIT` all carry `isa_role: 'scalar'` (the catch-all), while `VLOAD` carries `isa_role: 'memory'` — the role table had it exactly backwards, calling the local load "DMA" and the DMA "scalar". **Still open**: roling the real DMA mnemonics (`dma` for LOAD/STORE, `sync` for WAIT), **derived from the ISA model, not guessed** — an honest UNKNOWN ("the overlap question is unanswerable on this target until the ISA model distinguishes asynchronous channel movement") is worth more than a facet that answers. Same gap as the 17 unroled atlas_isa identities; pairs with R2.1 |
| N20 | RTL-derived `timing` fact class — **reconciles three ways** | **LANDED by a peer** (`targetgen/rtl/timing.py`, `rtl-introspect-v4-timing`). Structural walk over the use-def graph counting `seq.firreg` crossings; Atlas 84 modules, **43 resolved / 41 refused**. Independently reproduces our zero-fit terms: **SystolicArray depth 62 = 2·DIM−2**, **InnerProductTrees depth 2 = numPipeCuts+1**, PEMesh 31 = rows−1 — and `numPipeCuts+1` is literally what `InnerProductTreeParams.latency` computes in the Chisel. So three independent sources agree: our measurement, their structural walk, the RTL's own expression. The 41 refusals are correct behaviour — outputs reached through feedback have no finite longest path, so `pipeline_depth` stays UNKNOWN and an acyclic maximum is recorded under a *separate* name (`partial_depth`). **This is what R3 consumes; N11 is now half-closed.** **The corroboration is one target's, NOT the fact class's** — verified on the second target and it does not carry: gemmini resolves only **31 of 116** modules (73% refused), and the mesh container `Mesh` **refuses with 36/36 outputs cyclic**, because weight-stationary accumulation routes back through the array so no finite wiring depth *is* its latency. `Tile` resolves to **depth 0** (0/10 cyclic) — a legitimate combinational answer, not a missing one, and a consumer writing `if not depth:` would misread it exactly as `UNKNOWN` was misread as `0.0`. **Consequence for R4.6: per-resource peaks are NOT generally structurally derivable.** On a systolic array with feedback the walk correctly refuses and the latency must come from the sequencer's limits or from measurement. The refusals — 41 on one target, 85 on the other — are where R5's probes buy the most, and that set is *computed*, not chosen |
| N21 | Overlap is **available**, not impossible — measured | **DONE (peer).** Per-channel issue→wait distance over 137 kernels: 2,567 descriptors paired, **0 unpaired**. Distance 0 on 95.4%, but a real tail — 4.6% carry work between issue and wait, out to distance 50 — and the tail is held by exactly the kernels you'd predict (`dma_overlap.S` at 50, `dma_lsu_stall_mixed.S` 29, `dma_8_channel_burst.S` 14, `perf_fused_attention_mxu{0,1}` 8). **A per-issue WAIT emitted by construction would give a uniform distance**, so this refutes the assembler-convention hypothesis the bare 1:1 count could not. 97 of 114 DMA-bearing kernels are fully serial, including nearly all `smolvla_*`. Conclusion: the ISA expresses overlap, the hardware does it, the programs mostly don't — *available*, not hypothetical. Provenance: direct mnemonic pairing over the decoded `.S`, **not** the `dma_issue_to_wait` facet (which correctly returns UNKNOWN on all 137 until DMA is roled — keep both until they agree once) |
| **N27** | **The RTL facts record an input they never read** | **OPEN — blocks R3's validity domains.** Two resolvers disagree: `circt_introspect._soc_hw_path()` builds a cache-dir path `<target>_soc.hw.mlir` (its own docstring calls it a legacy fallback) which **does not exist for this target**, so `_sha()` failed and the cached facts carry `inputs.hw_mlir: "atlas_soc.hw.mlir"`, `inputs.hw_sha: "missing"`. Meanwhile `mlc_bridge.core_hw_mlir()` resolves the file discovery and the depth walk actually consume (`atlas_hw.mlir`, now pinned as `atlas_core_hw_dialect`, digest `d0b4135a…`). **The facts name an input they did not read and omit the one they did.** Until `inputs` records the resolved dialect, any performance term's validity domain is *asserted rather than evidenced* — it cannot say which elaboration it holds for. Also note the cached facts are `rtl-introspect-v3`, predating the timing walk, so they carry no `timing` block yet |
| **N29** | The facts cache never invalidates on a code change | **OPEN — makes N27 half-effective.** `inputs.extractor_sha` is recorded and its comment claimed "code change -> cache invalidates", but **nothing compares it**: **the comparison is written, correct, and DEAD**: `dump_facts` does compare `inputs` and `generator` and would rebuild, but `ensure_facts` returns on an existence check (`if p.is_file(): return p`) and never reaches it. So a stale cache serves facts from an older extractor indefinitely — which is why the atlas cache sat at `rtl-introspect-v3` with no `timing` block while the code was already v4. Consequence for validity domains: a fact-derived term can silently describe an **older extraction than the one the pin names**. Comment corrected; the comparison deliberately NOT implemented here, because it would force a live CIRCT re-extraction for every target on the next read — expensive, and it fails closed where the toolchain is absent. Measured: a fresh `build_facts('atlas')` takes **69.2 s** and yields v4 with `timing`, while `load_facts` still served v3 — **every cache on this host was stale**. **Same failure shape as the other caps: a check that cannot fire.** Merge with N27 — both are one piece of provenance plumbing, and together they mean an artifact can name an input it did not read *and* have been produced by code that no longer exists. Interim workaround for R3: call `circt_introspect.dump_facts(...)` directly, which bypasses the dead check. Not fixed here: correct semantics force a ~70 s rebuild per target on first access across every suite, on a host where a test already crossed the 900 s ceiling — a rollout decision with real blast radius, not something to land quietly |
| **N30** | **One merge dropped FOUR things from the grading path; three restored, one outstanding** | **PARTIAL.** `test_cert_tier_is_representative.py::test_suppressed_tiers_are_skipped_never_failed` fails with `ImportError`; the name exists **only in the test** (4 references) and the definition is gone from `capsule_runner.py`. Present at `27396b1a`, absent by `f9dbad27`; the only commits touching the file in between are the two big merges (`a544c055`, `7a2d9112`). **Same shape as the recorded merge that reddened 21 ir tests.** NOT caused by the ladder fix `396c3f7a` — that diff neither removes nor renames it. The invariant at stake is exactly the anti-fabrication one: *a tier that never ran says `skipped`, never `fail`*, because `not_run_is_not_pass` reads a recorded `fail` as evidence the capsule was certified and found wrong. Restore-vs-delete is the owner's call, since ladder completion may make suppression unreachable except on the deliberate `MERLIN_FULL_LADDER=0` path. Current state is the worst of both: the invariant is unenforced **and** the bucket is red. Body recovered verbatim and sent; **restored by the owner as `074bec94`** — and their restore surfaced a worse bug of their own: the `MERLIN_FULL_LADDER=0` opt-out still did `raise _cf`, so **the escape hatch reintroduced the exact defect the ladder fix removed**. A fix whose opt-out restores the bug is worse than no fix, because the opt-out is the path taken when iterating fast and trusting the tool.\n\n**THE MERGE DAMAGE WAS FOUR THINGS, NOT ONE.** Diffing `capsule_runner.py` across the same boundary (`27396b1a` → `f9dbad27`) shows three dropped top-level defs — `_clip`, `_finalize_capsule_result`, `suppressed_tier_result` (all three now restored) — **plus `TierResult.toolchain`.** That field records WHICH PROGRAM was graded: a block-scaled MX capsule is graded on the harness's own reference kernel, not the submission, and the field is what keeps a score decomposable. Its test exists because a run once **reported 40/40 where 9 passes were the fixture** and nothing recorded the difference. **SETTLED — all four restored and the boundary swept clean.** The owner diffed the full top-level def set at `27396b1a` against the current file: *"top-level defs still missing: NONE"*, so nothing else from that merge is outstanding. **Why this one was the worst of the four:** the other three failed LOUDLY — an ImportError, a NameError. A dropped dataclass field does not raise; it just stops emitting, and the score it decomposes silently goes back to reading 40/40 when 9 were the harness's own kernel. Same shape as `codegen_ok: true` on a check that never ran, and as an absent tier reading as agreement: **a green signal that quietly stopped carrying its evidence.** Back-compat verified rather than assumed — a `TierResult` with no toolchain gains no key, so existing artifacts stay byte-identical. — the owner landed `toolchain` after the flag; the 21 anti-fabrication tests in `test_mx_pass_is_attributable` + `test_cert_tier_is_representative` pass. It had been about to be deleted as a stale test, which would have removed the only surviving record of why the field exists. Lesson: when a merge drops one symbol, diff the whole file; it dropped four |
| N31 | targetgen bucket baseline, measured | **RECORDED.** With the known-slow arc test deselected: **1 failed, 254 passed, 24 deselected in 6m41s**. The single failure is N30. Useful as the pre-existing baseline so a later red bucket is not misattributed to the perf work | **Corrected by the owner:** a competing 20-failure figure from a 2h51m full-suite run is **contaminated** — it ran while three sessions were committing, so pytest imported half-edited files and 6 of the 20 were tests fixed *during* the run. The 6m41s deselected run is the trustworthy instrument. Same failure shape as everything else today: a number that looks like the measurement you want |
| **N28** | A CI timeout is not a constant on a shared host | **RECORDED (finding).** `test_capacity_fit_second_target.py` fails on a **900 s pytest-timeout** under load, and it reproduces with the suspected change reverted — the hang is in `compile_cli → mlc_bridge.discovered_memories → dump_cache → xDSL parse`, an inherently long arc-oracle path, not a logic break. This is the **6.3× concurrency effect surfacing as a test failure**: five agent sessions were active. Two consequences: a ceiling tuned on a quiet host is not a ceiling on a busy one, and **any timing measured while the suite runs is a throughput figure wearing a latency figure's clothes**. Do not raise the ceiling to make it green — that hides the signal. Owner undecided |
| **N23** | **Pin the `AtlasCore.fir` revision before GSIM grades anything** | **OPEN — blocks 0.7b's use as an oracle.** The `.fir` came from the spec-lifting tree, where 13 task dirs are **not byte-identical**; AT0 was used. Cycle-exact agreement with the independently-built gold Verilator is strong evidence it is the same design, but that is evidence, not provenance. Needs a `hardware_pins.yaml` entry (digest, not just a sha) before any verdict cites it |
| **N24** | **Per-unit busy is a TOP-LEVEL PORT — Verilator can read it too** | **OPEN, and cheaper than assumed.** `io.dbg.{mxu0,mxu1}{Comp,Data}Busy`, `lsuBusy`, `xluBusy`, `dmaBusy[8]` is a port on `AtlasCore`; the existing Verilator harness simply does not read it. **Do not credit GSIM for the per-unit waterfall** — R4's gap attribution can have it today on the tier we already run. GSIM's genuine exclusive is the **18,416 internal signals** (VPU/XLU FSM state, LSU busy, scalar PC) that a no-`--trace` Verilator build cannot reach; trace overhead 23%, streamed to disk, RSS flat |
| **N25** | First traced run already found two things | **RECORDED (finding).** On AF3: **MXU1 is 0.0% busy across all 7,817 cycles**, and **76.7% of cycles have no unit busy at all**. A second matrix engine sitting entirely idle is a compiler opportunity, not a hardware fact; and 76.7% idle bounds what any compute-side optimization can win. Both feed R4's attribution and N22's role split — and both were invisible before per-unit observation |
| **N26** | The DRAM wrap is quantified: harmless now, fatal at layer scale | **OPEN — sharpens N13.** Instrumented: DRAM base `0x90000000` means **100% of accesses wrap**, but the touched span is ~10 KB so `alias_collisions = 0`. That ~100× headroom evaporates at layer scale — one 1024×3072 bf16 tensor is **6 MiB** against a 1 MiB window. Same defect on all three tiers; GSIM neither fixes nor worsens it. A detector now exists; the fix does not |
| **N22** | **Split the corpus by performance ROLE, not by pass/fail** | **OPEN — do before R6.** Functional eligibility is the wrong axis for a performance set. Derived from the arc table: **OPTIMIZE (12)** — both DMA and compute substantial, so a lever has something to act on (gelu_tanh headroom 1234, attention 777, gemma_attention 456, gemma_mlp 381, softmax 331, k_chain 284, …); **DMA-term calibration** — `smolvla_rms_norm` at 93.7% DMA and **0% compute**, which isolates the memory term with zero compute confound and is the cleanest input to R2.1's predictor; **fixed/word-term calibration** — the ~688-cycle elementwise trio, where process-start and IMEM-load dominate and which R0.1b already showed carry the largest fit error. A capsule with no headroom is not a failed optimization target, it is the wrong instrument |

---

## R1 — Performance record + minimal term  *(Lane A; concurrent with R2)*

| id | task | state |
|---|---|---|
| 1.1 | `performance_record.schema.json` (real JSON Schema) | **DONE** — the digest triple must be a **required** field from the first record written, or every artifact produced before it is uncitable |
| 1.2 | `performance_term.schema.json` + minimal `PerformanceTerm` | **DONE** — `value \| unit \| provenance \| confidence \| validity \| bounds`. **UNKNOWN is a distinct inhabited state that cannot be read as 0.0** (no float default) |
| 1.3 | Emit a record per kernel from `compose_program_cycles` + `attribution.py` + `npu_model_compare` as they stand | **DONE** — all 21 kernels, under `out/artifacts/` |
| 1.4 | Test: composed prediction reproduces `mxu` 158 (one tile) / 284 (k_chain) exactly | **DONE** |
| 1.5 | Test: writing a record with a missing digest **raises** | **DONE** |
| 1.6 | Test: npu_model cycles/`exu_stats` can never source a term | **DONE** — they disagree with arc by up to **4.92×** (elementwise trio 3387 vs 688; 3972/1273 is `gemma_rms_norm` at 3.12×); diagnostic only |
| 1.7 | Defer the five-lattice provenance unification | **DEFERRED by design** — until ~10 real terms exist. Five representations exist today; unifying before there is anything to unify will churn |

## PROVEN: generated kernels beat the shipped atlas kernels, bit-exact

Head-to-head, **same computation, same oracle tier the baseline was measured on**, bit-exactness by
`plan.matches()`. Artifact: `out/artifacts/headtohead/atlas/v1/`.

| shipped kernel | shape | shipped | ours | speedup | bit-exact | bytes moved |
|---|---|---:|---:|---:|:--:|---|
| `matmul` | 32×32×32 | 2383 | **513** | **4.64×** | ✅ | 65,536 → **4,096** |
| `smolvla_matmul` | 32×32×32 | 1485 | **513** | **2.90×** | ✅ | 36,864 → **4,096** |
| `smolvla_matmul_k_chain` | 32×64×32 | 3102 | **801** | **3.87×** | ✅ | 81,920 → **6,144** |

(Superseding the first run at 705/705/1121 = 3.38/2.11/2.77×, before the third lever below.)

**Where the win comes from — two levers, both previously only estimated:**

1. **Transfer amplification, now demonstrated rather than projected.** The shipped `matmul` moves
   **65,536 B for a computation needing 4,096 B — 16×**. Ours moves the minimum. Since DMA is
   **2054 of the shipped kernel's 2383 cycles (86%)**, cutting movement is nearly the whole win. This
   is `compose_program_cycles`'s open "Finding 6", closed by construction: the generator computes its
   own descriptors instead of inheriting a movement pattern.
2. **The settle margin.** The device contract carries settle at **2× the measured minimum** (128 vs
   64). At the minimum, `matmul` drops 1217 → **705**.
3. **Per-class settle — a third lever, found by decomposing our own result.** At 705 cycles the kernel
   is **stall-dominated, not DMA-dominated**: 8 stall sites × 64 = **512 of those 705 cycles are
   settle**. And `Settle` carries separate `tensor`/`mxu`/`vpu` fields that `probe_settle` fills with
   **one uniform value**. Probing each class independently:

       tensor=32  bit-exact on 5/5 operand salts     ->  705 → 513
       tensor=24  WRONG on 5/5                       ->  a real boundary, not data luck
       mxu=32     WRONG                              ->  the matrix class genuinely needs 64

   So the uniform floor was over-delaying the tensor class by 2×. Measuring three numbers instead of
   one is strictly more information by the same method — the safety argument is unchanged, and the
   failure at 24 across every operand set is what makes it a floor rather than a lucky value.

**The falsifier fired correctly, and it is what makes this a measurement.** At settle=32 — below the
measured minimum — the kernel runs in 449 cycles and is **`bit_exact: false`**. So 64 is a real
architectural floor, not a lucky value, and 705 is a legitimate result rather than an under-delayed one.

**Tier confound removed:** arc and vsim return **identical cycles** on every generated variant, which
independently re-confirms the 14/14 tier-agreement finding. The baseline is arc-measured and ours is
quoted on arc.

### The comparison set is complete, not cherry-picked

Of the 21 shipped kernels, exactly **7 have MXU work**, and of those exactly **3 are pure matmul** —
`matmul`, `smolvla_matmul`, `smolvla_matmul_k_chain`. Those are precisely the three compared. The other
four (`gemma_attention` 5 vector ops, `smolvla_attention` 8, `gemma_mlp` 1, `fused_matmul_bias` 1) mix
in vector work that `plan_matmul` cannot emit, so they are not a fair comparison rather than an
unfavourable one. **3 of 3 of the fair set, not 3 of 21 cherry-picked.**

### The differential comparator, validated against measured hardware

The two settle variants differ **only** in resolved stall cycles; the MXU initiation interval and the
movement rate are UNKNOWN in both, with identical demand. So they must cancel, and the difference must
be exactly predictable with **no absolute model**:

    emitted stall sites: 8    (counted in the kernel, both variants)
    settle delta:        128 -> 64 = 64 cycles per site
    PREDICTED delta:     8 x 64 = 512
    MEASURED delta:      1217 - 705 = 512          exact

`differential.compare` returns `basis=exact`, `faster=settle_64`, `|delta|=512`, `cancelled=('mxu_ii',
'dma_rate')` — **while neither total is computable** (`a.cycles is UNKNOWN`). That is R8's thesis
confirmed against hardware rather than against a fixture: two schedules ordered exactly, with the
unknowns cancelling, on a workload neither model can price.

(One correction to my own first reading: `delta_cycles` is `b - a` with *positive means `a` is faster*,
so the tool's `-512.0` is the right sign for "b is faster by 512" and my equality check was
sign-confused, not the tool.)

### The levers scale — measured at layer scale, 4,394 tile passes

| 416×832×416 | cycles | bit-exact | wall |
|---|---:|:--:|---:|
| uniform 128 (2× margin) | **3,300,328** | ✅ | 493 s |
| per-class 32/64/64 | **1,288,552** | ✅ | 196 s |

**2.56×**, 2,011,776 cycles saved over 4,394 tile passes (≈458 cycles/pass). The baseline reproduces the
recorded N1 layer measurement of **3,300,328 exactly**, which validates the setup rather than just the
result.

This is a **self-comparison, not a vs-shipped number** — the shipped corpus has no layer-scale kernel to
compare against. What it establishes is that the win is **not a toy-scale artifact**: the per-pass stall
saving multiplies by the tile-pass count, so the lever is worth *more* at realistic sizes, not less.

### What is left on the table, and which tool would take it

Our 513 cycles against a **crude serial floor of ~286** — MXU busy 158 (measured, and the same on the
shipped kernel) plus 128 beats of movement at the measured 0.999 beats/cycle. So we are at roughly
**56% of a naive serial bound**, with ~227 cycles unaccounted.

That remainder is almost entirely **compiler-inserted separations**: on a non-interlocked machine every
producer→consumer pair needs its stall, and ours are now at their measured per-class floors. The only
way to remove a separation that is architecturally required is to **cover it with independent work** —
which is precisely the DMA/compute overlap lever, still unused, and precisely what a dependence-aware
scheduler decides.

Two honest limits on taking it:
- A 32×32×32 matmul has **one K-step**, so there is no cross-step work to overlap with. The lever needs
  `k_chain` or larger, where the next step's movement is independent of this step's compute.
- Reordering is an **emitter** change, not a settle constant, so it risks bit-exactness in a way the
  settle work did not. It should be ranked by `differential` before it is emitted — which is exactly the
  rank-before-generate split this plan already committed to.

So the next increment is W3's dependence graph, and the measurement above is what it would be judged
against: any schedule it proposes must beat 513 while staying bit-exact.

**What this does NOT claim:**
- Not a whole-model or layer-scale win — these are 32-tile kernels, the scale the shipped corpus has.
- Not an *optimality* claim. We beat the shipped kernels; we do not know the machine's floor. The
  structural bound says a 32×32×32 matmul cannot finish before `fill(62) + completion(96) = 158` MXU
  cycles plus its movement, and 705 is well above that.
- Not a claim the shipped kernels are badly written, and here is the concrete reason. Decoding
  `matmul.hex`: its DMA descriptors take their transfer length from register **x6, which the program
  NEVER WRITES** (it writes only x1, x2, x3, x10, x11). The size is **inherited harness state**, not a
  size chosen for the computation — which is exactly why it moves 16x what it needs. So the 16x is an
  inherited default rather than a considered choice, and the comparison is fair on cycles and unfair on
  intent. What our generator does differently is *choose its own descriptors*.
- Not a claim about the other 18 kernels — three were run.

## The new perf capsules, RUN — and the L1 sweep separates rate from intercept

W2's L1 K-sweep (`PK00`–`PK03`, K = 32/64/128/256 at m=n=32), generated kernels at the per-class settle
floor, all bit-exact:

    PK00   K=32    1 k-tile     513
    PK01   K=64    2 k-tiles    801
    PK02   K=128   4 k-tiles   1370
    PK03   K=256   8 k-tiles   2510

    least squares:  cycles = 229.3 + 285.1 x k_tiles
    residuals:      [-1.4, +1.5, +0.2, -0.3]   -- +/-1.5 cycles across an 8x span of K

**This is the measurement the shipped corpus could not produce.** It has only two K points, and the
repo's own rule says a unit whose cost is a rate *plus* a fixed overhead cannot be priced by one rate —
two points fit a line through any two numbers and cannot show it is a line. Four points give
±1.5-cycle residuals, which is what makes the split credible rather than merely fitted.

The intercept is consistent with the structural terms already measured independently: fill/drain **62**
(read from the circuit, `handshake.py`) plus reset **12** leaves ~155 cycles of prologue, epilogue and
store — a residual with a plausible home rather than a fudge.

`PK00 = 513` reproduces the standalone 32×32×32 measurement exactly, so the capsule path and the
direct path agree.

**What this does not settle:** the k-chain Δ30 (measured MXU 284 vs naive 254) is about *MXU busy
cycles* from the activity buckets, a different quantity from these whole-kernel totals. It stays
UNKNOWN; this sweep does not touch it.

## W3 — dependence graph: DONE, and it answers the 76.7%

`aeb464aa` (operand direction) + `8f37632c` (graph + liveness + driver), 34 tests.

**3.1 operand direction, derived behaviourally.** Seed state, execute one instruction, observe what
changed. Of 374 operands: **54 DEF, 96 USE, 224 UNKNOWN, 0 refused** — and the refusals separate two
cases that look alike: *"the effect was identical at every probed value"* (provably didn't matter)
versus *"the instruction changed no observable state"* (may be real and simply unobserved). Neither is
reported as established.

**3.3 the critical path is a real bound.** On our 32×32×32 kernel: 59 instructions, **46 edges (23 RAW,
5 WAR, 18 WAW)**, critical path **130.5 cycles** against a measured **513** —
`"consistent: the bound sits below the measurement, as a bound must"`. That is the first number in this
tree that speaks to the 76.7% idle, because it prices *separations* rather than occupancy.

**3.4 the ranking is EXACT, and it independently reproduced a prediction I had made by hand:**

    stalls_tightened   693.5   (-384.5 vs as-emitted)
    as_emitted        1078.0
    movement_hoisted  1079.0   (+1.0 -- SLOWER)

All three pairwise comparisons come back `basis: exact` with the unresolved class cancelling. And
**`movement_hoisted` being 1 cycle slower is the tool deriving, unprompted, what I had reasoned
separately**: a 32×32×32 matmul has one K-step, so there is no independent work to hoist movement over.

**Three honesty properties worth keeping:**
- **Loop-carried edges are excluded, not weighted.** *"Its separation is UNKNOWN and measured to exceed
  the naive sum of the published per-operation latencies, so giving it a weight would be fitting rather
  than deriving."* Every per-region number is therefore a bound on ONE iteration.
- **Register pressure refuses where capacity is unknown**: `mrf 2 of 64 live -- fits`; `scalar 10 live,
  capacity UNKNOWN, so whether that fits was NOT checked`.
- **Unpriced separations are tracked, not dropped** (`separation.scalar × 19`), so the critical path
  reports `complete: false` and claims `AT LEAST 130 cycles`.

**The one caveat:** the schedule *estimates* are not bounds. `stalls_tightened` predicts 693.5 where the
measured tightened kernel is **513**, so the estimate is conservative. The *bound* (130.5) is the sound
number; the estimates rank correctly but should not be quoted as cycle predictions.

## W1.2 RUN — the whole corpus on both RTL engines, and the overlap question settled

`cycle_sweep.py` (`merlin/perf/occupancy.py` + 19 tests). **27 capsules, both elaborated-RTL
engines, per-cycle traces, 10 s wall serially.** GSIM and Verilator return **identical cycles on
27/27** — extending the previous 17/17 cross-validation to the whole corpus.

The two engines are blind to different things, which is the point of running both: the Verilator tap
reads all 8 DMA channel ports but no unit lacking a top-level port; the GSIM trace reads the internal
FSMs and the PC but only one aggregate DMA signal. **Each engine alone charges its own blind spot to
idle.** The union is the honest denominator.

**Three ways a per-cycle trace fabricates an answer — all three fired here, all three now gated:**

1. **A signal counted beside its own components.** `lsuBusy == vloadBusy + vstoreBusy` exactly, and
   `lsuBusy` is *never* busy alone. Counted naively this produced **204 cycles of "overlap"** on a
   kernel with none.
2. **A unit with no busy port, read as permanently idle.** The **VPU has no top-level busy port** and
   is invisible to the Verilator tap — yet it is the busiest unit on the vector kernels (AF3: 1493
   cycles). The harness's `unmeasured_units` said only `[scaleRegs, dbg0, dbg1]`, so it **claimed a
   completeness it did not have**. Including the VPU moves AF0's idle from **89.9% to 39.2%**.
3. **Two instruments' views of one unit, merged as two.** GSIM lags Verilator by exactly one cycle
   (derived, not assumed), so shared units land in adjacent cycles and read as overlap; and an
   aggregate bus-valid signal beside the per-channel ports of the same bus added **6.8% fabricated
   overlap**. Both folded by containment, derived from the measurement.

**The corrected corpus figures** (union, 42,661 cycles, 27 capsules):

    idle by the top-level ports alone : 25,308  (59.3%)
    idle by the UNION of both engines : 17,752  (41.6%)
    overlap                           :      0  (0.0%)

**So the motivating number of this whole plan was substantially an instrument gap.** AF3's idle is
**46.2%**, not the 76.7% the plan quotes — the rest was a unit nobody could see. The 76.7% figure
should not be requoted.

### The zero overlap is a property of the SCHEDULES, not of the machine — CORRECTION

`composition_operator` now returns an operator instead of `Unavailable`: **SUM, eta = 0.0000**, over
27 joint (`partitioned=False`) workloads. **That is true of this corpus and must NOT be stated as a
device trait.** Generating the *same* kernel at tightening separations and re-measuring the joint
vector:

    settle (t/m/v)   cycles   idle   overlap>=2
    128/128/128       1217     726        0
     64/ 64/ 64        705     240       26
     32/ 64/ 64        513      79       57
     32/ 64/ 32        481      47       57

**Tightening the separations creates overlap.** The engine busy vectors are byte-identical across all
four rows — only idle moves — so this is the schedule changing, not the work. Measured twice by
independent routes (the ablation, and a separate re-derivation through this module's own path).

Consequence: the operator's validity domain is **"schedules at or above the 2x settle margin"**. On
the shipped corpus eta is 0; outside it, it is not. Quoting SUM as what the hardware does would
repeat the exact error the joint vector was built to prevent, one level up. `compose_program_cycles`'s
`overlap_cycles = 0` default remains refuted.

### What the corpus sweep also exposed

- **`mxu1` is never busy on any of the 27 capsules**, and **dma4-7 are never used**. Half the matrix
  units and half the DMA channels are dark corpus-wide.
- **6 capsules are byte-identical programs** (`AS0`, `AT0`, `AT2`, `AT7`, `AT8`, `BT0` — all 1090
  cycles, identical busy vectors). `AT7_matmul_mxu0` and `AT8_matmul_mxu1` are supposed to exercise
  *different* MXUs; they emit the same program and `mxu1` never fires, so **AT8 does not test what its
  name claims**.

## The lever ablation — what is separable, and one claim NOT confirmed

A 2x2 over (descriptor byte volume) x (settle), run twice by independent routes.

    cell                    cycles    idle   overlap>=2   dma0 busy
    amp= 1  128/128/128       1217     726        0           133
    amp= 1   32/ 64/ 64        513      79       57           133
    amp=16  128/128/128       2625     726        0          1541
    amp=16   32/ 64/ 64       1921      79       57          1541

**The interaction is exactly 0** — the settle effect is -704 in both rows and the descriptor effect
is -1408 in both. Confirmed independently by two constructions. The per-unit ports say why the two
levers are separable rather than merely appearing so: **the descriptor lever moves only `dma` busy;
the settle lever moves only idle.** Every engine total is byte-identical across the settle factor,
and `mxu0Comp+mxu0Data = 158` in every cell — the same 158 the shipped kernel's own bucket reports,
and the same 158 that `fill(2*DIM-2 = 62) + completion(96)` predicts. Three independent routes to one
number.

**So the win is dominated by movement, not by scheduling.** The MXU term is identical between the
shipped kernel and ours; what we changed is how many bytes get moved to feed it.

**NOT CONFIRMED — the "settle alone loses" reversal.** A concurrent ablation reported that the settle
lever at the shipped byte volume gives 2598 against the shipped 2383, i.e. that we *lose* without the
descriptor lever. This construction gives **1921**, which still wins by 1.24x. The two disagree
because **neither reproduces the shipped kernel's movement cost**: the shipped `matmul` spends 2054
cycles in DMA, while these amplified cells reach 1541 and 2128 respectively. The amplified cell is a
stand-in for the shipped movement pattern, not a reproduction of it, so **no reversal claim should be
quoted from it in either direction** until a cell matches 2054. What both routes DO support is the
sign and the rough magnitude: movement is the dominant term, worth roughly 75-100% of the 1870-cycle
win, and the scheduling terms net near zero or slightly against us.

**A new lever, measured here and previously untested: the `vpu` settle class.** `32/64/32` runs in
**481** cycles (4.95x against the shipped 2383) versus 513 at `32/64/64`. The `vpu` class had never
been probed separately. **Bit-exactness for this variant is NOT yet verified in this tree** — the
concurrent ablation reports 6/6 operand salts clean, and until that is reproduced here 481 is a
measured cycle count with an unverified correctness claim, which is not yet a result. 513 remains the
citable number.

### The `vpu` settle class is INERT on this kernel — and the occupancy vector is what proves it

Probing the third settle class, six operand salts each, through the same oracle path the record used:

    32/64/64   513   bit-exact 6/6
    32/64/32   481   bit-exact 6/6
    32/64/16   465   bit-exact 6/6
    32/64/ 0   449   bit-exact 6/6      <- the falsifier NEVER fires

**A class that never fails has established nothing.** The reason it never fails is measurable, and
GSIM measures it: `vpu_fsm_state` is **0 on every cycle of this kernel** (the Verilator tap cannot
see it at all — the VPU has no top-level busy port). The generated matmul never uses the VPU, so the
emitter is inserting VPU separations into a kernel with no VPU hazard to protect against. Those 64
cycles are pure padding — **12.5% of the 513**.

Contrast with the classes that DO bind: `tensor` fails bit-exactness at 24 on 5/5 salts, and `mxu`
fails at 32. Those floors are real because the falsifier fired at them.

**The rule this yields is derived and checkable, not a constant:** *a settle class may be zeroed for
a kernel in which the joint occupancy vector shows that unit is never busy.* That makes "which
separations can I drop?" a measured per-kernel question instead of a global constant, and it is
exactly the kind of fact this layer is supposed to produce.

**What may and may not be quoted.** **513 stays the citable number** for the 32x32x32 matmul under
the general device contract. 449 (5.31x) is legitimate *for this kernel with its precondition
stated* — VPU unused, proven by occupancy — and quoting it without that precondition would be an
overclaim of exactly the kind this register exists to catch. The intermediate 481 was reported
elsewhere as a new best; it is bit-exact, but it inherits the same precondition and is not a floor.

Corollary for the corpus: `mxu1` and `dma1-7` are never busy in this kernel either, so any
separation the emitter inserts for them is inert here by the same argument, and has not been probed.

## Generalizing the occupancy layer — Gemmini, and Radiance especially

Radiance is the hard case and therefore the useful one: it declares a **SIMT cluster that CONTAINS a
systolic MX PE** (`simt_cluster.contains: [mx_pe]`, kinds `simt` and `systolic`), so one device holds
two microarchitectures whose concurrency is exactly what a perf layer should measure.

### A live bug this exposed, fixed before anything else

Derived containment **cannot** distinguish a unit's own sub-signals from an engine nested inside
another: both nest identically in the data. On a Radiance-shaped trace the shipped rule folded
`mx_pe` into `simt_cluster` and reported **overlap 0** between the two microarchitectures — deleting
half the silicon, which is the very error the muon contract's own comment records ("Declaring only
the SIMT cluster described half the silicon"). Fixed (`833a7e8d`): the engine a column belongs to is
**declared**, and columns in different declared engines are never folded into each other; folding
still applies *within* an engine, where it is right.

`declared_engines()` reads the engine set from each target's own compute-unit declaration
(`c1937f89`). One code path, four shipped declarations, four archetypes, three topologies:

    gemmini   systolic_mesh (systolic)                          flat
    muon      simt_cluster (simt) CONTAINS mx_pe (systolic)     nested, heterogeneous
    saturn    rvv_lanes (vector) + opu (spatial)                siblings
    toy_npu   pe_array (systolic)                               flat

### What still has to change, in order

| id | task | why |
|---|---|---|
| **R-2** | Activity semantics beyond a busy bit: `width` (active/total lanes) and `commit`, alongside `binary_busy` | A divergent warp is 100% *busy* on 1 of 32 lanes. Radiance is also the only target with a real denominator (`speed_of_light: simt_lane_peak`; atlas's is `null`), so lane-cycles are computable there and utilization must never be read off a busy bit |
| **R-3** | Open cyclotron's `on_cycle` tap | `cosim_muon.run()` **already takes `on_cycle`** and already returns `per_core_active` (cycles each core had a valid **commit**) — merlin passes neither. Same "built, never called" shape as the atlas io.dbg tap. A commit signal is strictly better than the pc-advance proxy. **Measure callback overhead first**: a Python callback per cycle on a slow engine can double wall |
| **R-4** | Cross-instance contention, and **the falsifier for the whole method** | Atlas returned overlap 0 everywhere. If the same tooling reports 0 on a device with two concurrent microarchitectures and multiple cores, **the instrument is broken, not the machine**. Radiance is where this method earns trust or loses it |
| **R-5** | The MX PE <-> SIMT interaction | The question the layer exists to answer. mlc's G12 already found a composition-level effect neither component's isolated model predicts (the SIMT offload stream exceeding the MX scratchpad DMA-queue depth — a real RTL backpressure assertion) |
| **R-6** | **Gemmini first**, as the cheap validation | mlc's GSIM full-SoC harness already emits `commit_valid, wb_pc, wb_inst, is_rocc, gem_ex_state, gem_load_state, gem_store_state, gem_rs_busy, axi_ar_valid, axi_aw_valid` — CPU + accel + memory in exactly the right shape, and `gem_rs_busy` pairs with the `gem_*_state` FSMs for the idle calibration. Validates the generalization before the harder Radiance work. **Not run**: no raw trace on disk, the harness needs building |

Order: R-0 (done) -> R-6 -> R-2 -> R-3 -> R-4/R-5.

**Limits to carry, not paper over.** Radiance's `facts.json` is EMPTY, so occupancy cannot be tied to
derived geometry. Its citable tier is `cycle_model` (cyclotron), **not** elaborated RTL, so any
Radiance occupancy claim is weaker in provenance than the atlas ones by construction. 9 MX capsules
are unwinnable on the fork-free MLIR path, and radiance declares 8 semantic families while evidencing
2. None of this blocks the occupancy work; all of it bounds what a Radiance perf claim may say.

## CPU <-> accelerator on atlas — measured

The GSIM trace carries `pc_reg`, so the cycles with no accelerator unit busy can be split by whether
the scalar core retired an instruction. Over 42,688 cycles:

    accelerator unit busy         24,943   58.4%
    SCALAR CORE only (pc advances) 13,684   32.1%
    nothing advancing (true stall)  4,034    9.4%

**81% of the "idle" time is the scalar core executing control code** — address arithmetic, loop
bookkeeping, descriptor setup — and with overlap at zero on these schedules it is fully serialized
with the accelerator. The spread is workload-dependent and large: `AF8_rope` 54.4% scalar,
`AT3_k_accumulation` 50.7%, the matmuls 37.2%, `AF4_gelu` 2.2%.

So the largest remaining lever on this target is **hiding scalar control under accelerator
execution**, not more MXU or less DMA. No instrument before the joint vector could see it.

**Caveat on the method:** "pc advances" is a proxy that leans on atlas being a single non-interlocked
scalar core with one delay slot. On gemmini use `commit_valid`; on a SIMT device neither works. The
control-processor activity signal must become a **declared producer input**, exactly like unit kinds
and engine bindings — that is part of R-2.

**The external host lane is a separate question and is NOT honest yet**: `scalar_rvv_lane` exists in
routing/place and `lane_report()` fails closed, but the lane actually executed in the capsule path is
x86, not RVV, with no per-lane cycle accounting and two clocks with no common denominator. Not
re-verified this session. Any end-to-end number there stays a **vector, never a sum** — the failure
mode is a router win reported as a backend win.

## W3.5 RUN — the graph confronted with GSIM, and what it did NOT refute

The dependence graph priced its edges from published latencies and from UNKNOWNs read as zero, so
it was **plausible, not falsifiable**. GSIM's per-cycle `pc_reg` settles it: the cycle each
instruction executed is observable, and a run that was BIT-EXACT left at least the required
separation. For an edge `u -> v` with required separation `R`, a correct run left `S`, so `S >= R`:
`S` is an **UPPER bound on the requirement** and may never be promoted to "the latency"; a predicted
weight `W > S` is **refuted**, the one direction a single run can falsify the model.

**RETRACTED — the first run of this reported two refuted edge weights. It was wrong.** It read the
traced counter as the index of the executing instruction. It is not: the counter is a pipeline
register that **leads the executing instruction by 2 slots** on this target, so a correctly
separated pair whose counter values arrive 2 cycles apart looked like it violated a 32-cycle weight
it had actually respected. The concrete case, from the emitted kernel:

    [23] VLOAD vd=1, rs1=7, imm=0
    [24] DELAY imm=32          <- the separation IS emitted, right between them
    [25] VTRPOSE_XLU vd=1, vs1=1

Reported as "predicted 32.125 > measured 2" — while a 32-cycle stall sits between the two
instructions. The stall was real; the reading of it was not.

**The lead is derived, not assumed.** Two runs of the same program differing only in the DELAY
immediates: the counter values whose dwell changed are the changed instructions displaced by the
lead. All 8 sites map with a single shift, and the dwell is `imm + 1` exactly:

    DELAY at  24 26 28 36 38 42 45 48
    dwell at  26 28 30 38 40 44 47 50      -> lead = 2 slots, 8 of 8 explained
    128 -> 129 cycles      32 -> 33      64 -> 65

`derive_counter_offset` requires EVERY changed instruction to be explained before it reports
`established`, and an underived offset leaves per-instruction claims unavailable rather than off by
a pipeline depth.

**Result with the offset applied** (generated 32x32x32 matmul, 59 instructions, straight-line, each
executed once): 46 edges, **all 46 checked, 0 skipped, ZERO refuted** — minimum slack **+1 cycle**.
The graph's weights survive their first confrontation with a machine. Two UNKNOWN classes were
narrowed from above (`separation.ADDI` n=16 tightest 2; `separation.LUI` n=3 tightest 1) — the first
numbers those unknowns have carried that are not guesses.

**Where the 513 cycles actually go**, charged to the executing instruction and reconciling exactly
with the emitted stalls (6 sites at 33 + 2 at 65 = 328):

    DELAY           328   63.9%
    DMA_LOAD_CH0     68   13.3%
    DMA_STORE_CH0    68   13.3%
    ADDI             28    5.5%
    ADD               7    1.4%
    LUI               3    0.6%

**Nearly two thirds of the kernel is separation padding**, which is why the settle lever alone took
1217 -> 513, and it is now measured per instruction rather than inferred from a whole-kernel total.
The 8 `DELAY` sites in the decoded program independently re-derive the register's "8 stall sites".

**Dropped as unsound:** a longest path over MEASURED separations. Elapsed time is not a dependence,
so on a straight-line kernel that path degenerates to the span — it returned "506 of 513 over 2
instructions", recovering the makespan from one early-to-late edge and explaining nothing.

**The lesson, which is this register's recurring one.** The first result was produced by tooling that
ran cleanly and returned a plausible, interesting number. What caught it was reading ONE concrete
example — an emitted kernel with the stall visible between the two instructions the tool called
under-separated. A number that survives its tooling has not been checked; a number that survives an
example has.

## L5 fusion is NOT the cheap level — a scoping error, corrected

This register called L5 fusion "the cheapest level to build and the only one whose comparand needs no
model at all". **The comparand half is right and the cheap half is wrong.** `cycles(fused)` against
`cycles(A) + cycles(B)` needs no model — but it needs three KERNELS at one shape, and two of the
three cannot be emitted:

* `plan_matmul` emits matmul only. There is no `bias_add` and no fused epilogue in the generator.
* The shipped corpus cannot supply the missing halves either. `AF9_fused_matmul_bias` is
  **16x16x16 bf16**, `AF10_bias_add` is **16x16 bf16**, and the nearest shipped matmul
  (`AT2_single_tile_matmul`) is **32x32x32 fp8** — a different shape AND a different dtype. There is
  no matmul at AF9's shape anywhere in the corpus.
* `comparison_group` is declared on `AF9`/`AF10`/`AF11`/`AF12` but each capsule sits ALONE in its own
  group, so the field was never consumable: the groups have one member each.

So L5 needs a bias-epilogue emitter (a vector add over the accumulator drain, a golden, and its own
falsifier) before any fusion number exists. That is real work, not a free run, and it should be
scheduled as such rather than kept on the register as the cheap next step.

**The levels that ARE runnable with today's generator** are L2 and L3: `PL00`/`PL01` declare
`op: matmul` at 224x224x224 and 224x448x224, and `PC00`/`PC01` declare `op: k_chain` at 128x128x128
and 128x256x128 — all of which the existing emitter covers.

## W1.0 — the free fidelity run, and what it found

`mlc/spec/validate_fidelity.py` against the 2,219 totals already on disk. **Zero new measurement.**

**Result 1 — the spec formula and the RTL-derived model agree EXACTLY.** `dev% = 0.000` on all 21
kernels, every row `normative`, no discrepancies. Two independently-written cost models, one from a
spec formula and one compiled from RTL, land on the same integer 21 times. That is a real cross-check
and it had never been run.

**Result 2 — against MEASURED arc, one kernel is a 16.5% outlier** while the other 20 are under 2.5%:

    smolvla_rms_norm   spec 3824   arc 3282   +16.5%
    (next worst: gelu_tanh 2.44%, fused_silu_gate 1.75%, gemma_rms_norm 1.65%)

**Result 3 — the cause, and it refutes a premise this whole layer rests on.** The component drift:

    vpu            spec 428   |   arc measured vpu = 0
    data_movement  spec 3072  |   arc dma_busy    = 3075
    overhead       spec 324   |   arc none        = 208

The program contains **7 real vector ops** (`vsquare, vredsum, vmul, vadd, vsqrt, vrecip, vmul`), and
`perf/vector_cycles.py` — exact on 15/17 kernels — prices them at **428 cycles, complete**. Arc reports
**vpu = 0**. The instrument is not blind to them: **`gemma_rms_norm` uses the SAME op classes and
registers 621 vpu cycles.** The difference between the two is DMA saturation — 93.7% vs 30.5%.

So those 428 cycles of vector work ran **while DMA was busy**, and the partition charged them to
`dma_busy`. **That is measured overlap.** It contradicts the "both-active fraction EXACTLY 0.0000"
claim that `compose_program_cycles` defaults `overlap_cycles = 0` on, and it is a second independent
demonstration of the partition problem (the first being `buckets == truth + 1` on all 21).

**Two consequences that change other tasks:**

- **A correction to my own classification.** `workload_roles` called `smolvla_rms_norm` the cleanest
  memory-term calibration instrument, on the basis of *93.7% DMA and 0% compute*. That reading is
  wrong: it is not a compute-free kernel, it is a kernel whose compute is **already fully overlapped**.
  That makes it a *poor* DMA calibration instrument — its `dma_busy` silently contains 428 cycles of
  vector work — and an *excellent* overlap demonstrator. W2's role classification needs re-deriving
  once a joint-occupancy instrument exists.
- **The 14.7% overlap headroom is measured on the wrong instrument.** It was computed as
  `min(dma, compute)` from the same partitioned buckets, so a kernel whose compute is already hidden
  reads as having no compute to hide. The number is not necessarily wrong, but its provenance cannot
  support it, and W1.2's joint occupancy vector is what settles it.

**What this does NOT establish:** how much overlap, on which kernels, or whether it is deliberate. A
partition can only show that a bucket is *missing* work; it cannot count concurrent cycles. Only the
joint occupancy vector (W1.2) can.

## R2 — DMA byte-volume and overlap  *(Lane B; mlc repo only)*

**The highest-leverage item in the program.** DMA is 60–93.7% of every Atlas cycle count and the MXU
is never above 13.9% busy on any of the 21 kernels. Measured compute/DMA overlap is **exactly 0.0**
suite-wide.

| id | task | state |
|---|---|---|
| 2.1 | Structural DMA footprint predictor from program descriptors | **OPEN — method established, owned with N19.** Today `footprint_bytes` is a workload *input*; `compose_program_cycles.py` flags it open ("Finding 6"). The size is **not on the DMA instruction** — it arrives in a register, so recovering `{vmem offset, DRAM address, byte count, channel}` per descriptor needs forward constant propagation of scalar writes (`LI`, `ADDI rd, x0, imm`) to each DMA op. Two rules agreed with the peer building it: **(a)** which operand is the size comes from the ISA model's field layout for that mnemonic — reading it positionally is the pick-by-field-name bug — and if the layout does not name operand roles, the byte count is **UNKNOWN**, not operand 2; **(b)** a size from a loop induction or runtime value is not a constant, so that descriptor gets `bytes: None` and **the kernel reports a lower bound, never a total** — summing only the descriptors that resolved would understate the footprint in the flattering direction. Same unit of work as roling the DMA mnemonics (both need the ISA model to describe DMA), so do them together |
| 2.2 | Match arc-measured `(reads + writes) · beat_bytes` for ≥18/21 kernels | **OPEN** — the ≤3 failures named, cause recorded **UNKNOWN, never fitted** |
| 2.3 | The overlap-policy term | **OPEN — evidence basis corrected.** Overlap on Atlas is zero, but **the arc table cannot be the evidence**: its buckets *partition* cycles (`dma_busy + mxu + vpu + none = truth + 1` on all 21, a constant fencepost), so it returns zero overlap whether or not overlap exists. The real evidence is the direct mnemonic count — **2567 DMA ops and 2567 DMA.WAIT, exactly 1:1**, read from the `.S` rather than through the role table (which, per N19, was fabricating DMA anyway). **And the 1:1 count has its own caveat**: an exact pairing cannot by itself distinguish *"every issue is immediately awaited"* from *"the assembler emits a paired WAIT per issue by construction"*. The supporting argument is that WAITs carry explicit channel operands and the engine holds **8 in flight**, so pairing is a choice the programs make rather than a syntax rule. **Stronger measurement available and preferred**: `dispatch.dma_issue_to_wait` (N19's sibling metric) gives the instruction distance directly — a sampled kernel shows DMA.LOAD at 6, DMA.WAIT at 7, i.e. distance 1. Report that distribution over the corpus and the question is answered by measurement rather than by inference from a count. **"~2×" WAS WRONG — I propagated it unchecked.** Perfect overlap saves exactly `min(dma, compute)`, and summed over the corpus that is **4,457 cycles = 14.7%** of the affected kernels' total (best single case: gelu_tanh 23.9%; matmul just 6.6%). Compute is far too small relative to DMA for overlap to be worth a multiple. **Overlap is a ≤15% lever, not a 2× one.** It is still worth taking, and it is now proven *available* rather than hypothetical (N21), but it is no longer R2's headline |
| **R4.6** | **Generalized roofline: the composition operator is a derived trait** | **OPEN — the model spec.** Classic roofline `T = max(T_compute, T_mem)` silently assumes perfect overlap; Atlas overlaps ~0, so its truth is `T = Σ`, and applying textbook roofline understates runtime by a **MEASURED 13.36%** corpus-wide (SUM 42388 vs MAX 36725 cycles; 15.00% over engine cycles only; worst single kernel `gemma_rms_norm` **30.46%**, 38.45% engines-only). **The "~40%" I first quoted was never measured and is wrong** — it is roughly the worst single kernel with idle excluded, not a corpus figure. Pin 13.36% / 30.46%; do not tune toward 40%. General form: `T_lower = compose({demand_r/peak_r}, critical_path, fixed_terms)` with `compose ∈ {max | sum | sum − η·min(pairs)}` **derived from traits and measured η**, never defaulted to `max`. Four generalizations: **(a)** N resources from the resource graph, not two axes — report the *binding* resource and margin to second (mlc's `core_ipc` already returns this as `limiter`); **(b)** the bytes axis is **moved** bytes, since amplification makes an algorithmic-bytes roofline optimistic by that factor — wrong in the flattering direction. **But the ratio itself is only half-derived**: moved bytes are measured, while `useful_bytes` is NOT derivable from the pinned suite (it carries no shape or dtype field), so the 9–28× figures came from hand-entered operand sizes. The gate — an `ALGORITHMIC` demand refusing without a measured factor — is enforced and tested, but the family split is exercised on explicitly SYNTHETIC observations until real shapes are added to the measurement artifact; **(c)** fixed terms are first-class intercepts, not noise (`fill = 2·DIM−2`), so a rate-only roofline mispredicts every small tile; **(d)** UNKNOWN propagates, and it is the COMMON case, not the edge: the structural walk resolves 43/84 modules on one target and only **31/116 on the other**, refusing the mesh itself where accumulation is cyclic. A composer that assumes derivable peaks works on one archetype and collapses on the next. Also: a resolved depth of **0 is a real answer** (combinational), distinct from UNKNOWN — `if not depth:` repeats the `UNKNOWN`-as-`0.0` bug one level up. On this target `DmaEngine` is among the refused, so the **dominant** term's peak is not structurally derivable and must be measured by capsule, never guessed |
| **R4.7** | Validation fixture: the analytical model already predicts 6/7 | **PARTIAL — measured.** From two independently derived facts and **zero fitting**: `mxu_busy = fill(2·DIM−2 = 62) + completion(96)` gives **158**, matching `matmul`, `smolvla_matmul`, `fused_matmul_bias`, `gemma_attention` exactly, and `2 × 158 = 316` matching `gemma_mlp` and `smolvla_attention`. **The 7th refutes the naive extension**: `smolvla_matmul_k_chain` (K-accumulate) predicts `62 + 2×96 = 254` against a measured **284** (Δ30), so the accumulate path carries a cost the single-tile law does not express. Recorded **UNKNOWN, not fitted** — two K points show the naive law is wrong without showing what is right, which is exactly the ≥2-points-per-parameter rule at model level. This 6/7 is the regression fixture for the generic composer |
| **2.5** | **DMA transfer amplification — the actual lever** | **OPEN — this is R2's headline, not overlap.** **PROVENANCE WARNING — half of this is hand-entered, not derived.** The *moved* bytes are measured (`(reads+writes)·beat_bytes`), but `npu_model_suite.json` carries **no shape or dtype field** (`['arc','footprint_bytes','npu_cycles','op_stream','program','vpu_tile_elems']`), so `useful_bytes` **cannot be derived from the pinned artifact**. The ratios below used operand sizes typed in by hand from the program sources; they are plausible and the shapes were separately confirmed, but `amplification.py` cannot reproduce them from the suite and its family routing is tested on observations explicitly labelled SYNTHETIC. **Wiring real shapes in is a prerequisite before any amplification number is cited as measured.** The demo tiles move 9–28× more bytes than they use: `matmul` moves 65,536 B (2,048 beats × 32) for ~4,096 B of useful operand+output data = **16.0×**; `rms_norm` 24.0×; `gelu_tanh` 28.0×; `elementwise_add` only 2.0×. With DMA at 60–93.7% of cycles, cutting amplification is worth **multiples** where overlap is worth ≤15%. This is `compose_program_cycles.py`'s open "Finding 6" (transfer amplification) and it makes R2.1's structural footprint predictor the *optimization instrument*, not just an accuracy requirement — you cannot reduce what you cannot predict. **Caveat to check at layer scale:** part of this ratio is an artifact of a 32×32 demo tile paying a fixed per-tile transfer cost; proper layer tiling amortizes it, so the toy-scale amplification likely OVERSTATES the available win. Measure it again under N1 before claiming it |
| 2.4 | Falsifier | **OPEN** — if structural prediction fails, the cycle claim is downgraded **in writing** to "given the byte volume" |


### R2 result — the falsifier did NOT fire; my first conclusion was wrong

**Retracted:** I first reported that movement volume was unpredictable because the derived ISA model
carries no encoding for that family — `opcode_table` empty, `by_mnemonic['dma.load.ch0'] is None`, and
all 146 decoded movement instructions collapsing into one class with all-zero operands. Every one of
those observations is true. **The conclusion drawn from them was not.**

The encodings exist upstream, in the target's own shipped ISA definition:

    DMA_LOAD_CH0 : opcode=0b1111011, funct3=0b000, funct7=0b0000000   (funct3 = channel)
    DMA_STORE_CH0: opcode=0b1111011, funct3=0b000, funct7=0b0000001
    DMA_WAIT_CH0 : opcode=0b1111111, funct3=0b000, funct7=0b0000001

Decoded against those fields, **all 242 movement instructions across the 25 shipped programs resolve
cleanly: 67 loads, 29 stores, 146 waits** — with real register operands (`load.ch0 rd=4 rs1=1 rs2=7`),
and the length register shared across the loads of a program. merlin's disassembler had seen only the
146 waits and missed every load and store, because *its* extraction drops the family; the information
was in the programs and in the vendor ISA the whole time.

**The lesson is the session's own recurring one, and I walked into it while writing it down.** I
concluded "underivable" from *merlin's view of the ISA* rather than from the ISA. A missing field in a
derived model is evidence about the derivation, not about the machine. Checking one file upstream
turned a blocker into a build.

**Shipped:** `merlin/python/merlin/perf/dma_volume.py` + 10 tests (`ce975bc0`). Identification is by
ENCODING, not by role — which is why the role table was never the real prerequisite, and why nothing
in the module names a target, a mnemonic spelling or a channel count. The two R2.1 rules are enforced
and pinned: the size operand comes from the ISA's declared field layout (a form declaring no size
field yields `None`, never "operand 2"), and any unresolved descriptor demotes the **whole kernel** to
a lower bound rather than letting a partial sum present as a total. Constant propagation kills a
register it cannot evaluate and a backward branch invalidates everything, since a loop-carried value
is not a constant.

**R2.2 RUN — the acceptance criterion is not met, and the reason is measured.**
`movement_volume.py --target <t>` (`632bf16e`), over all 21 kernels of the pinned suite:

    5 exact,  16 consistent floors,  0 bound violations,  of 21 kernels
    60 of 83 movement descriptors unresolved

**R2.2 asked for >=18 of 21 within tolerance. It is 5.** But the shape of the failure is the finding:

- **Where a program sets its own transfer lengths, the prediction is EXACT** — all 5 land at ratio
  1.000, not merely inside a tolerance. The derivation is right.
- **Where it does not, the floor holds.** 16 consistent lower bounds and **zero bound violations**:
  the predictor never once claimed more traffic than was measured. The demotion rule works on real
  data, not just in its unit tests.
- **The cause: 52 of 83 descriptors (63%) name a length register the program NEVER WRITES.** The
  shipped programs are not self-contained — they inherit register state from the harness that runs
  them. No amount of constant propagation over the program text can recover a value the text does not
  contain.

**So >=18/21 is unreachable from program text alone on this corpus, for a named reason** — not a gap
in the predictor. Closing it needs the harness's entry state (or a target whose programs set their own
descriptors), and that is a different input, not a better analysis.

**Consequences, carried forward honestly:** the cycle claim stays *"given the byte volume"* for the 16;
`compose_program_cycles`'s Finding 6 stays open; and R2.5's amplification ratios stay half-derived,
since `useful_bytes` needs the same lengths. What DID change: `footprint_bytes` is no longer purely an
input — for 5 of 21 kernels it is now predicted exactly from the program, and for the rest there is a
sound floor with its shortfall attributed to a specific missing input.

**One thing this corrected in the module.** The size operand is now read from the register each form's
own executable body reads to size its transfer (`length = state.read_xrf(self.<operand>)`, recovered by
`ast`), not from a name-order fallback. My first version listed a generic register name as a last-resort
guess — which is the pick-by-position bug wearing a different hat, in the very module written to
prevent it.

## R3 — Target profile + performance contract  *(Lane A; after 1.2)*

| id | task | state |
|---|---|---|
| 3.1 | `perf/profile.py` — archetypes + traits, **derived never named** | **DONE** — copy `capability_manifests.derive_manifest`'s 3-source pattern exactly (CIRCT facts + `families.py` + `residual.yaml`) |
| 3.2 | `perf/contract.py` — emit the contract | **DONE** |
| 3.3 | Keep VMEM capacity **UNKNOWN** | **DONE** — npu_model's `HardwareConfig` says 1 MiB, RTL says `0x180000` (1.5 MiB), mlc refuses to classify 39 SRAMs. `HardwareConfig` is residual-tier at most, **never a fact source** |
| 3.4 | Test: the same code derives a profile for Atlas **and** Gemmini | **DONE** — the anti-overfit proof |
| 3.5 | Test: deleting a fact from `facts.json` yields UNKNOWN, not a default | **DONE** |
| 3.6 | Gate: `check_no_target_name.py` clean **and** `--coupling` adds zero entries under `perf/**` | **DONE** — `perf/` is by construction "a generic module", so any target-named import registers as coupling debt. Do **not** add allowlist entries |


**R3 verified independently (not taken from the agent's report).** `derive_profile(target)` and
`derive_contract(target)` produce *different, correct* answers on two archetypes:

| trait | atlas (tensor/dataflow) | gemmini (decoupled queue/systolic) |
|---|---|---|
| `explicit_dma` | **UNKNOWN** — the role census "cannot separate an asynchronous channel move from a local operand load" (this is N19) | **True** — `facts.interfaces[dma_tlb]: FrontendTLB` |
| `host_dispatched_queue` | UNKNOWN | **True** — RoCC `ReservationStation` |
| `independent_engine_ports` | UNKNOWN — no second engine group evidenced | **True** |
| `self_hosted_program` | **True** — 42 opcodes -> `external_backend` | **False** — 26 opcodes -> `inline_asm_insn` |
| `persistent_configuration_state` | UNKNOWN | **True** — CONFIG_EX/LD/ST |

Contracts: atlas 3 resources (`mxu` peak **1024** MACs/cyc = DIM², `operand_store.capacity_bytes`
**UNKNOWN** as required since RTL 1.5 MiB and config 1 MiB disagree); gemmini 5 resources
(`systolic_mesh`, `data_movement`, `accumulator` 65536 B, `scratchpad` 262144 B).

Three properties worth keeping: **`satisfied=None` (UNKNOWN) is distinct from `False`** (evidenced
negative) — `self_hosted_program` is False on gemmini, while `explicit_dma` is None on atlas; every
trait carries **both `evidence` and `missing`**, so an UNKNOWN states what would resolve it; and
`Archetype` carries a `questions` tuple, making the archetype a *prior about which questions to ask*
while traits decide which apply — the design the plan called for.

**184 tests pass** across the landed perf layer (term, record, decompose, headroom, amplification,
workload_roles, oracle_cost, profile, contract), run directly rather than trusted from a report.
## R4 — Envelope + gap attribution  *(Lane C; disjoint from A)*

| id | task | state |
|---|---|---|
| 4.1 | Extend `design_envelope.py` / `arithmetic_intensity.py` with the **DMA-bound ridge the data shows** | **OPEN** |
| 4.2 | Map gap components 1:1 onto `attribution.py`'s buckets | **OPEN** — `compute / dma / stall / control / host`; `residual` stays `assumed` and never vanishes |
| 4.3 | Each attributed gap names the optimization family it implies | **OPEN** |
| 4.4 | Invariant test: no prediction falls below the structural bound, across all 21 | **OPEN** |
| 4.5 | Test: attributed buckets sum to the arc total exactly | **OPEN** |

### R4 — verified independently

`520e83cb` (2680 insertions, gates green, 0 coupling entries under `perf/`).

**The composition operator is derived per target, and the second target refuses.** The NPU corpus
derives `SUM` with eta **0.0** — from an overlap observation *independent of the buckets*, since from the
buckets alone `composition_operator` correctly returns `Unavailable("…partition…")`. The
weight-stationary mesh derives **no operator at all**: one engine group, so all three concurrency traits
are missing, and even given an overlap observation it refuses — *"no pair has any overlappable time, so
the operator is unobservable"*. A different, correct answer on a different archetype.

**The 6/7 fixture reproduced exactly, zero fitting.** `fill = 2·32−2 = 62` from the fill law **and
independently** `SystolicArray.pipeline_depth == 62` from the RTL walk; delay 96 read from the program's
own schedule. 158 on four kernels, 316 on two; `smolvla_matmul_k_chain` naive 254 vs measured **284**,
Δ30, recorded UNKNOWN with the accumulate reason, fill lower bound surviving. `floor_violations == ()`
and `bound_violations == ()` across all 21. Buckets attribute to the arc total exactly, residual **−1**
on every kernel (the constant fencepost), emitted unconditionally as `assumed`.

**Coverage implemented as a time share, not a module count** — and the gap is stark:

| target | module-count share | structurally-resolved TIME share |
|---|---:|---:|
| NPU corpus | 43/84 = **51.2%** | **3.65%** |
| weight-stationary mesh | 31/116 = **26.7%** | **0.0%** |

A **14× disagreement** on the first target. `Coverage` deliberately has **no `confidence` field** (asserted
by test) so the number cannot be misread as one, and omitting `structural_resources` yields UNKNOWN
rather than 0.

**The hard constraint bites on the PRIMARY target, not the second.** Only **2 of 21** workloads bound end
to end: the vector engine has no valid demand, because the op-count proxy is *refuted* by
`smolvla_rms_norm` (7 `Vector` ops, **0** vpu cycles) and `VectorEngine` refuses structurally (31/31
cyclic). So 19/21 carry a partial bound only. Correct behaviour, and a direct measurement of how far
UNKNOWN propagates in practice.

**Not claimed:** the 443-test `dse` bucket **did not complete** (two runs killed past 15–20 min at load
10–18). All 443 collect; the 139 perf-selected pass in 80 s; the 65 new ones pass. The other 304 are
untouched and are reported as not-run, never as passed.

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

## R8 — Relative comparison: ordering schedules without pricing them  *(added; not in the original plan)*

**The plan's own §23 says ranking accuracy matters more than absolute error, and the machinery was
built absolute-first.** R7 then measured the consequence: only **2 of 21** workloads bound end to end,
which reads as near-fatal. It is not, for ranking. If two schedules leave the **same** resources
unresolved and ask the **same** work of them, those resources cost the same in both and **cancel out of
the difference** — what remains is a difference between the parts the compiler actually controls.

That is the common case for the two axes that matter: retiling a transfer changes movement and leaves
compute demand alone; changing overlap policy changes composition and leaves every demand alone.

**Demonstrated on real evidence** (`gelu_tanh`, dma 3592, vpu 1234 unresolved) — three tiling variants,
**no computable total for any of them**, all three pairs ordered exactly:

    serial   vs tiled_2x -> exact: tiled_2x faster by 1006 cycles (1 unresolved resource cancelled)
    serial   vs tiled_4x -> exact: tiled_4x faster by 1401 cycles
    tiled_2x vs tiled_4x -> exact: tiled_4x faster by  395 cycles

| id | task | state |
|---|---|---|
| 8.1 | `perf/differential.py` — `compare`, `comparable`, `rank_schedules` | **DONE** (`9e1934df`, 11 tests) |
| 8.2 | **A matching unresolved SET is not sufficient** | **DONE** — the trap: same unpriceable unit, *different work* asked of it. Differencing there hands a real gap to the wrong term and returns a confident number. Unequal demand is REFUSED; silence about demand is refused too, since silence is not evidence of equality |
| 8.3 | What each operator permits, derived not assumed | **DONE** — `SUM` is additive so ordering **and** magnitude transfer (EXACT); `MAX` is monotone but not additive, so only the ordering transfers, because an unresolved resource may dominate both and shrink the true gap to nothing (ORDERING_ONLY, magnitude withheld); `PARTIAL` couples pairs so neither survives — REFUSED rather than approximated |
| 8.4 | Incomparable pairs are returned, not dropped | **DONE** — a candidate excluded for want of evidence is a hole in the search, not a verdict about the candidate |

**What this changes.** The 19 workloads R7 could not bound are not unusable — they are unusable for
*absolute prediction* and usable for *choosing*. Since choosing is what a compiler does, this converts
the headline limitation from fatal to bounded. It resolves no missing peak and needs no new
measurement: the same evidence, differenced instead of totalled.

## R6 — Bounded candidate selection  *(unit set by 0.1b)*

| id | task | state |
|---|---|---|
| 6.1 | Two axes only: DMA tiling / descriptor shape, and overlap policy | **DONE** — 21/21 reference kernels are single-op single-tile; a five-level hierarchy is unsupported |
| 6.2 | Selection via `tier_policy` + `oracle_schedule` + the three sanctioned methods, or `mining/beam.py` | **DONE** — **no new beam under `perf/`**; `dse/search/AGENT.md` states a repo-level stance, and choosing a different directory to route around it is rules-lawyering |
| 6.3 | Denominate the budget in the **measured** scarce unit | **DONE** |
| 6.4 | Drop `Generality` from VOI | **DECIDED** — with one target it is a constant, hence not a factor |
| 6.5 | Stop conditions as predicates with unit tests over a fake evaluator | **DONE** |


**R6 verified independently — `4eb0860c`, 63 tests pass.** API: `select.{STOP_CONDITIONS, StopPolicy,
StopVerdict, check_stop, Axis, AxisEvidence, Candidate}`, `budget.{Budget, budget_from_channels,
Calibration, Channel}`. Two design choices worth keeping: `GENERALITY_DROPPED` is an explicit named
constant rather than a silently-omitted VOI factor — the reason it is dropped (one target makes it a
constant) is recorded where a reader will find it; and `budget` distinguishes `MEASURED` from
`PROJECTED` cost, so a plan built on an extrapolation cannot be mistaken for one built on a measurement.

Four more properties worth keeping:

- **`scarce_unit` refuses rather than guesses.** An unpriced channel returns `Unavailable` naming every
  unpriced channel — because an unpriced channel cannot be ruled out as the expensive one, *which is
  exactly the error that produced the wrong R0 verdict*. Per-item price and items-per-datapoint are kept
  separate, since a cheap oracle called 26 times per datapoint is not cheap.
- **The unit flips with the regime, proven by running the same code twice.** Synthesis-dominated
  (2184 s/call vs 0.276 s x 26) elects `synthesis_call`; simulation-dominated (2700 s x 26) elects
  `deep_sim_query`. The oracle share is pinned at **0.33%** of a datapoint, inside the measured
  0.2–0.4% band. The convergence x-axis inherits the elected unit and carries its measured price.
- **The blocked axis reports UNKNOWN with the evidence named**, not a default candidate: `dma_axis`
  returns `established=None` with `missing=("the per-command byte volume …")` exactly when the movement
  mnemonics carry no role. `established` is tri-state — `None` (no evidence), `False` (a real finding
  that the axis has no lever here), `True` (candidates).
- **A derivation bug caught and test-pinned:** the observed command count is `redundancy_factor *
  transfers_min`, **not** `moved_bytes / block_bytes` — with heterogeneous descriptors the block is the
  *largest* command, so the naive division undercounts (8 commands read as 4) and would silently drop a
  real descriptor shape from the sweep.

Composed from existing machinery rather than rebuilt: `dse.search.grid.grid_search` for the within-axis
sweeps, `tier_policy` for the budget ledger and its `may_certify` refusal shape, and
`oracle_schedule.CapsuleState`/`Verdict` as the re-evaluation cache — so *"a verdict earned by different
bytes is not a verdict about this submission"* is the rule deciding here too. **No beam, MCTS, Bayesian
optimizer or new search loop under `perf/`.**

**Deliberately not done:** no real-data convergence curve was emitted. Producing one needs R7's
datapoints, and manufacturing points from the fake evaluator under a product manifest would be a
fabricated result. The machinery is exercised end-to-end in tests instead.

## R7 — Headline experiment  *(split; no reference exists off-corpus)*

| id | task | state |
|
### R7 result — and the number NOT to quote

`6a19c5d2`, 5 tests. Run: `headline.py report --target <t>`.

| claim | n | result |
|---|---|---|
| **7.1 RECOVERS** — fraction of a *shipped reference's* cycles | **2 of 21** | **0.916** |
| **7.2 PREDICTS** — fraction of *merlin's own emitted program* | 8 of 8 shapes | 0.2646 corpus-calibrated; **1.0095** adding the program's own scheduled stalls |

**The 1.0095 is a falsification, not a fit — do not quote it as accuracy.** It is a *lower* bound
sitting **above** its measurement on all 8 shapes. A structural bound that exceeds the thing it bounds
means an input is wrong. It is the single most quotable-looking number the experiment produced and it
is the one that must not be quoted; the driver emits it as a WARNING and a test pins that wording.

**7.1's n is 2, not 21** — only two workloads bound end to end (the vector engine's demand proxy is
refuted, and `VectorEngine` refuses structurally). The 19 partial bounds are reported beside it under
their own name and never averaged in: a bound over a subset of resources answers a different question
from a bound over all of them.

**The two claims are not the same measurement.** 7.1's denominator is somebody else's kernel; 7.2's is
merlin's own — *"a perfect 7.2 would still say nothing about whether the emitted code is fast."*

**Four things the experiment states it does NOT support:** any % of peak or attainment (no denominator
is derived — `speed_of_light` is null); a fraction-of-reference at generated shapes (no reference
exists off the corpus); a claim that the model discriminates between *alternatives* (0 iso-work shapes
— rank agreement over differently-sized shapes is mostly the ordering of the work itself); and an
end-to-end bound at generated shapes (every one carries an unresolved resource).

---|---|---|
| 7.1 | **"Recovers"** — fraction of reference on the 21 eligible capsules | **OPEN** — kernel-relative only (N8: no % of peak) |
| 7.2 | **"Predicts"** — prediction accuracy at shapes merlin emits | **UNBLOCKED (N1 done)** — no shipped reference exists off-corpus, so only predicted-vs-measured is claimable |
| 7.3 | Report 7.1 and 7.2 **separately** | **OPEN** — conflating them reports a prediction result as a recovery result |
| 7.4 | Small→large: error, Spearman/Kendall, top-K recall, regret **vs size** | **UNBLOCKED (N1 done)** — the 1,263× span between the capsule corpus and this layer is the size axis |
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
