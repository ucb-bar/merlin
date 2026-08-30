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
| **N1** | **Layer-scale workload generation** | **OPEN — HARD BLOCKER.** There is no layer-scale Atlas workload anywhere: 21 eligible capsules are 32×32 tiles, 25 shipped programs are static `.S`, and the "full model shapes" ([241,960], 1024×3072, (50,720)) are docstring text with no program, golden or run. Only merlin's emitter can produce them. **Every performance claim about layers depends on this.** Blocks 0.1b's large run, R7, and the meaning of R5's rules |
| N2 | Re-declare the Atlas measurement authority against L4, **run by GSIM** | **OPEN — revised by 0.7b.** The tier stays L4 (elaborated RTL); the *runner* becomes GSIM, 50.7× faster and cycle-exact against Verilator on 17/17. merlin's L4 already imports its runner by path from `<vsim_dir>/verilator_run.py`, so this needs only a sibling wrapper — the GSIM binary already speaks the identical spec-JSON contract. Original rationale below still holds: — L4 Verilator is 13× cheaper than arc *and* `derived_from_rtl: true` / `elaborated_rtl` where arc is not, and the two report **identical cycles on 14/14 capsules where both tiers ran** (directly measured, 42 samples, `oracle_query_cost_atlas.json`). NOTE the evidence provenance: `score_capsule.json` cannot support this claim — its tier records are bare strings (`"pass"`), and `cycles_diagnostic` is L3-only, so an earlier citation of "identical across all 21, e.g. AT2 = 3078 at both" was wrong on both count and number (3078 is one submission's L3; a different submission gives 1090 at both). The declared `cycles_from: arc_program` / `cycles_tier: cycle_model` understates what is obtainable; re-declaring upgrades every number to tier `rtl` at 1/13 the cost. Edit `out/artifacts/targets/atlas/contracts/residual.yaml` |
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
| **N14** | **L4 never passes a cycle budget, and the harness blames the agent for it** | **OPEN.** `program_verilator_adapter.run()` omits `max_cycles`, so every capsule runs at the 20000 default while L3 sizes its budget with `derive_cycle_budget`. Probed: a 25,096-cycle program raises `ProgramDidNotHalt` on the shipped path, and `capsule_runner.py` converts that raise into a tier FAIL **explicitly attributed to "the AGENT's bug."** This is the recurring *harness-limit-reported-as-agent-defect* pattern. **SETTLED — real defect, zero impact on the record.** A peer classified every failing tier record across all atlas runs: **L4 `did_not_halt` = 0 everywhere**; the only two L4 failures are hidden capsules failing on tolerance. So the denominator does not move. Still worth fixing — it fires the moment anything runs longer. **But the same census found the cap DID fire on L2**: 10 `did_not_halt` in `merlincirct_glmnp1`, every one with **L3=None and L4=None** — the tiers that would have decided the capsule never ran, and the run is recorded 0/11. On capsules we jointly measure executing in 178–2614 cycles on RTL. That run must not be cited as an agent result; it is another instance of harness-limit-reported-as-agent-defect, and the strongest argument for the ladder-completion fix (N12), which would have made the misattribution visible in the artifact |
| **N15** | L3 has no wall-clock timeout at all | **OPEN.** `timeout` reaches only `emit_bundle`; the arc cycle loop is unbounded. A layer-scale command buffer authorising ~10⁸ cycles runs for **days**, holding a worker, undetectably |
| N16 | IMEM is 32768 words and today's kernels are fully unrolled | **OPEN** — 3 of the 4 projected layers overflow it 4×–52×. Constrains N1: layer-scale kernels need loops, not unrolling |
| N17 | Atlas control flow is not the RISC-V reading | **RECORDED (fact).** The PC is a **word index** and `branch_target = s1_pc + (imm>>1)`, so a B-type immediate moves `imm/2` *instructions*; there is **one architectural delay slot** (`PcControl.scala`, `ScalarCore.scala`). Encoding a byte offset silently never closes a loop and reads as "this core has no control flow at all" — a derived-fact trap for N1 and for any emitted loop |
| N18 | Buffered-trace OOM hazard — **cleared** | **DONE, no action.** No per-instruction or per-cycle stdout on either tier: a 2,049× longer run produced 0.13% more output (8,277 → 8,288 B, exactly the extra digits of the cycle count) at identical peak RSS (15,360 kB). The 72 GB hazard seen on another simulator here cannot occur |
| **N19** | **Role table fabricated a DMA engine** | **FIXED by a peer (in tree).** `isa_model.roles["memory"]` names a VMEM→mreg load (VLOAD), and `FROM_ISA_ROLE` mapped `memory → dma`. So the `dma` role fired on **local register loads**, while the 32 actual `DMA_{CONFIG,LOAD,STORE,WAIT}_CH*` mnemonics carry **no role at all** and there is no `sync` role. Three facets read it — `dispatch.dma_overlap`, `dispatch.double_buffered_banks`, `memory.dma_pattern` — so all three **answered confidently about a DMA engine they had never seen**; the old `bool(counts["dma"])` was True everywhere because VLOAD is everywhere. Now mapped to `operand_load`. **Confirmed at the instruction level**: `DMA.CONFIG` / `DMA.LOAD` / `DMA.WAIT` all carry `isa_role: 'scalar'` (the catch-all), while `VLOAD` carries `isa_role: 'memory'` — the role table had it exactly backwards, calling the local load "DMA" and the DMA "scalar". **Still open**: roling the real DMA mnemonics (`dma` for LOAD/STORE, `sync` for WAIT), **derived from the ISA model, not guessed** — an honest UNKNOWN ("the overlap question is unanswerable on this target until the ISA model distinguishes asynchronous channel movement") is worth more than a facet that answers. Same gap as the 17 unroled atlas_isa identities; pairs with R2.1 |
| N20 | RTL-derived `timing` fact class — **reconciles three ways** | **LANDED by a peer** (`targetgen/rtl/timing.py`, `rtl-introspect-v4-timing`). Structural walk over the use-def graph counting `seq.firreg` crossings; Atlas 84 modules, **43 resolved / 41 refused**. Independently reproduces our zero-fit terms: **SystolicArray depth 62 = 2·DIM−2**, **InnerProductTrees depth 2 = numPipeCuts+1**, PEMesh 31 = rows−1 — and `numPipeCuts+1` is literally what `InnerProductTreeParams.latency` computes in the Chisel. So three independent sources agree: our measurement, their structural walk, the RTL's own expression. The 41 refusals are correct behaviour — outputs reached through feedback have no finite longest path, so `pipeline_depth` stays UNKNOWN and an acyclic maximum is recorded under a *separate* name (`partial_depth`). **This is what R3 consumes; N11 is now half-closed.** The 41 refusals are exactly where R5's Verilator probes buy the most |
| N21 | Overlap is **available**, not impossible — measured | **DONE (peer).** Per-channel issue→wait distance over 137 kernels: 2,567 descriptors paired, **0 unpaired**. Distance 0 on 95.4%, but a real tail — 4.6% carry work between issue and wait, out to distance 50 — and the tail is held by exactly the kernels you'd predict (`dma_overlap.S` at 50, `dma_lsu_stall_mixed.S` 29, `dma_8_channel_burst.S` 14, `perf_fused_attention_mxu{0,1}` 8). **A per-issue WAIT emitted by construction would give a uniform distance**, so this refutes the assembler-convention hypothesis the bare 1:1 count could not. 97 of 114 DMA-bearing kernels are fully serial, including nearly all `smolvla_*`. Conclusion: the ISA expresses overlap, the hardware does it, the programs mostly don't — *available*, not hypothetical. Provenance: direct mnemonic pairing over the decoded `.S`, **not** the `dma_issue_to_wait` facet (which correctly returns UNKNOWN on all 137 until DMA is roled — keep both until they agree once) |
| **N27** | **The RTL facts record an input they never read** | **OPEN — blocks R3's validity domains.** Two resolvers disagree: `circt_introspect._soc_hw_path()` builds a cache-dir path `<target>_soc.hw.mlir` (its own docstring calls it a legacy fallback) which **does not exist for this target**, so `_sha()` failed and the cached facts carry `inputs.hw_mlir: "atlas_soc.hw.mlir"`, `inputs.hw_sha: "missing"`. Meanwhile `mlc_bridge.core_hw_mlir()` resolves the file discovery and the depth walk actually consume (`atlas_hw.mlir`, now pinned as `atlas_core_hw_dialect`, digest `d0b4135a…`). **The facts name an input they did not read and omit the one they did.** Until `inputs` records the resolved dialect, any performance term's validity domain is *asserted rather than evidenced* — it cannot say which elaboration it holds for. Also note the cached facts are `rtl-introspect-v3`, predating the timing walk, so they carry no `timing` block yet |
| **N29** | The facts cache never invalidates on a code change | **OPEN — makes N27 half-effective.** `inputs.extractor_sha` is recorded and its comment claimed "code change -> cache invalidates", but **nothing compares it**: `facts.ensure_facts` regenerates only when the cache is cold (`if p.is_file(): return p`). So a stale cache serves facts from an older extractor indefinitely — which is why the atlas cache sat at `rtl-introspect-v3` with no `timing` block while the code was already v4. Consequence for validity domains: a fact-derived term can silently describe an **older extraction than the one the pin names**. Comment corrected; the comparison deliberately NOT implemented here, because it would force a live CIRCT re-extraction for every target on the next read — expensive, and it fails closed where the toolchain is absent. Purging the cache is the interim workaround |
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
| 2.1 | Structural DMA footprint predictor from program descriptors | **OPEN — method established, owned with N19.** Today `footprint_bytes` is a workload *input*; `compose_program_cycles.py` flags it open ("Finding 6"). The size is **not on the DMA instruction** — it arrives in a register, so recovering `{vmem offset, DRAM address, byte count, channel}` per descriptor needs forward constant propagation of scalar writes (`LI`, `ADDI rd, x0, imm`) to each DMA op. Two rules agreed with the peer building it: **(a)** which operand is the size comes from the ISA model's field layout for that mnemonic — reading it positionally is the pick-by-field-name bug — and if the layout does not name operand roles, the byte count is **UNKNOWN**, not operand 2; **(b)** a size from a loop induction or runtime value is not a constant, so that descriptor gets `bytes: None` and **the kernel reports a lower bound, never a total** — summing only the descriptors that resolved would understate the footprint in the flattering direction. Same unit of work as roling the DMA mnemonics (both need the ISA model to describe DMA), so do them together |
| 2.2 | Match arc-measured `(reads + writes) · beat_bytes` for ≥18/21 kernels | **OPEN** — the ≤3 failures named, cause recorded **UNKNOWN, never fitted** |
| 2.3 | The overlap-policy term | **OPEN — evidence basis corrected.** Overlap on Atlas is zero, but **the arc table cannot be the evidence**: its buckets *partition* cycles (`dma_busy + mxu + vpu + none = truth + 1` on all 21, a constant fencepost), so it returns zero overlap whether or not overlap exists. The real evidence is the direct mnemonic count — **2567 DMA ops and 2567 DMA.WAIT, exactly 1:1**, read from the `.S` rather than through the role table (which, per N19, was fabricating DMA anyway). **And the 1:1 count has its own caveat**: an exact pairing cannot by itself distinguish *"every issue is immediately awaited"* from *"the assembler emits a paired WAIT per issue by construction"*. The supporting argument is that WAITs carry explicit channel operands and the engine holds **8 in flight**, so pairing is a choice the programs make rather than a syntax rule. **Stronger measurement available and preferred**: `dispatch.dma_issue_to_wait` (N19's sibling metric) gives the instruction distance directly — a sampled kernel shows DMA.LOAD at 6, DMA.WAIT at 7, i.e. distance 1. Report that distribution over the corpus and the question is answered by measurement rather than by inference from a count. **"~2×" WAS WRONG — I propagated it unchecked.** Perfect overlap saves exactly `min(dma, compute)`, and summed over the corpus that is **4,457 cycles = 14.7%** of the affected kernels' total (best single case: gelu_tanh 23.9%; matmul just 6.6%). Compute is far too small relative to DMA for overlap to be worth a multiple. **Overlap is a ≤15% lever, not a 2× one.** It is still worth taking, and it is now proven *available* rather than hypothetical (N21), but it is no longer R2's headline |
| **R4.6** | **Generalized roofline: the composition operator is a derived trait** | **OPEN — the model spec.** Classic roofline `T = max(T_compute, T_mem)` silently assumes perfect overlap; Atlas overlaps ~0, so its truth is `T = Σ`, and applying textbook roofline understates runtime ~40%. General form: `T_lower = compose({demand_r/peak_r}, critical_path, fixed_terms)` with `compose ∈ {max | sum | sum − η·min(pairs)}` **derived from traits and measured η**, never defaulted to `max`. Four generalizations: **(a)** N resources from the resource graph, not two axes — report the *binding* resource and margin to second (mlc's `core_ipc` already returns this as `limiter`); **(b)** the bytes axis is **moved** bytes, since amplification 9–28× makes an algorithmic-bytes roofline 9–28× optimistic — wrong in the flattering direction; **(c)** fixed terms are first-class intercepts, not noise (`fill = 2·DIM−2`), so a rate-only roofline mispredicts every small tile; **(d)** UNKNOWN propagates — `DmaEngine` is among the 41 refused modules, so the **dominant** term's peak is not structurally derivable and must be measured by capsule, never guessed |
| **R4.7** | Validation fixture: the analytical model already predicts 6/7 | **PARTIAL — measured.** From two independently derived facts and **zero fitting**: `mxu_busy = fill(2·DIM−2 = 62) + completion(96)` gives **158**, matching `matmul`, `smolvla_matmul`, `fused_matmul_bias`, `gemma_attention` exactly, and `2 × 158 = 316` matching `gemma_mlp` and `smolvla_attention`. **The 7th refutes the naive extension**: `matmul_k_chain` (K-accumulate) predicts `62 + 2×96 = 254` against a measured **284** (Δ30), so the accumulate path carries a cost the single-tile law does not express. Recorded **UNKNOWN, not fitted** — two K points show the naive law is wrong without showing what is right, which is exactly the ≥2-points-per-parameter rule at model level. This 6/7 is the regression fixture for the generic composer |
| **2.5** | **DMA transfer amplification — the actual lever** | **OPEN — this is R2's headline, not overlap.** The demo tiles move 9–28× more bytes than they use: `matmul` moves 65,536 B (2,048 beats × 32) for ~4,096 B of useful operand+output data = **16.0×**; `rms_norm` 24.0×; `gelu_tanh` 28.0×; `elementwise_add` only 2.0×. With DMA at 60–93.7% of cycles, cutting amplification is worth **multiples** where overlap is worth ≤15%. This is `compose_program_cycles.py`'s open "Finding 6" (transfer amplification) and it makes R2.1's structural footprint predictor the *optimization instrument*, not just an accuracy requirement — you cannot reduce what you cannot predict. **Caveat to check at layer scale:** part of this ratio is an artifact of a 32×32 demo tile paying a fixed per-tile transfer cost; proper layer tiling amortizes it, so the toy-scale amplification likely OVERSTATES the available win. Measure it again under N1 before claiming it |
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
