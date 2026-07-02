# Figure provenance & self-critique notes

Conference-grade provenance trail for every figure in `reports/`. For each: **N** (sample size),
**oracle tier**, **measured vs estimated**, and **caveats**. The guiding rule: nothing estimated is
drawn as if measured; every figure states its N and tier in an on-figure caption; single-datapoint
claims say so.

Oracle ladder: **L0** numpy · **L1** ref==sim · **L2** spike (FUNCTIONAL — cycles plateau ~120, NOT
timing) · **L3** verilator (cycle-accurate RTL) · **L4** VCS · **L5** FireSim (FPGA, cycle-accurate).
**arc** = arcilator JIT of the isolated `@Gemmini` (CIRCT firtool→HW→arcilator), bit-exact RTL, no SoC
boot. Fixed method→colour map (documented in `perf_style.py`): golden=steel, baseline=salmon,
merlin-gen=gold, IREE=sage, native=tan — colours are per-method, not "ours = brightest".

Regenerate: `gen_perf_plots.py`, `gen_arc_plots.py`, `gen_agentic_plots.py`, then `build_results_png.py`.
Data sources: `runs/perf_full_0001/{perf_results.json,firesim_arm_results.json}`,
`merlin/targets/gemmini/contracts/rtl_facts/arc_results.json`,
`experiments/gemmini_capsule_bench_v0/reports/agentic_results.json`.

---

## Section A — arcilator middle-tier

### A1 · fig_arc_landscape — oracle landscape
- **N:** 20 arc capsules (median wall). **Tier:** conceptual placement + measured wall.
- **Measured:** throughputs = 1/wall from the timing re-run (arc median 3.7 ms; verilator median 654.8 s;
  FireSim ~210 s/run). The ~177,453× arrow is measured, not derived.
- **Caveats:** y-axis is *ordinal fidelity* (spike functional < arc/verilator/FireSim cycle-accurate),
  not a measured fidelity score — stated in caption. spike placed low because it does not model the
  systolic array (see B3).

### A2 · fig_arc_bitexact — bit-exactness + per-capsule cycles
- **N:** 20 capsules. **Tier:** arc (RTL-faithful JIT), cross-checked vs L0 golden.
- **Measured:** 20/20 bit-exact; per-capsule cycle counts are arc's RTL cycle counter. Also backed by
  500/500 random differential + a negative control (not plotted).
- **Caveats:** arc runs the accelerator under an *ideal* memory model (see A4 for latency sensitivity).

### A3 · fig_arc_speed — wall time, arc vs verilator/FireSim
- **N:** 20 arc capsules (bars); verilator reference = median over perf kernels; FireSim = typical/run.
- **Measured:** all three. Speedup badge computed from measured arc ÷ measured verilator wall
  (`measure_arc_wall.py`, min-of-5). **No hardcoded `ARC_RATE`/`VERILATOR_WALL`** — both read from
  `arc_results.json::rtl_wall_ref`.
- **Caveats:** verilator wall is boot-dominated (the honest reason arc is so much faster is *no SoC
  boot*, not a different RTL); stated in caption. VCS≈verilator (same RTL).

### A4 · fig_arc_latency — timing realism under memory latency
- **N:** **1 capsule** (A2's matmul) swept across modelled DRAM latencies. **Tier:** arc.
- **Measured:** arc cycles at each latency setting.
- **Caveats:** single-capsule sensitivity study (explicitly labelled). Shows arc is not fixed to ideal
  memory — cycles scale with latency — but is not a corpus-wide claim.

### A5 · fig_arc_hostcomm — host↔accelerator telemetry
- **N:** 1 capsule. **Tier:** arc (RoCC command feed + TileLink DMA beats counted in the harness).
- **Measured:** RoCC control ops + DMA traffic from the arc harness counters.
- **Caveats:** illustrative of the instrumentation, single capsule.

## Section B — performance / fidelity (cross-approach)

### B1 · fig_cycles — cycle-accurate cycles, generated vs hand-tuned
- **N:** 24/24 kernels (those with a cycle-accurate cell). **Tier:** **L3 verilator (plain bars, ≤32K
  MACs) + L5 FireSim (hatched)** — same RTL, directly comparable; tier shown per-bar (hatch) + legend,
  not a footnote.
- **Measured:** cycles, single run per cell.
- **Caveats:** IREE omitted here (different verification path + ~10–40× outlier that crushed the scale);
  baseline ≡ native (bit-identical); merlin-gen v1 differs a few cycles on epilogues. Series shown =
  golden vs merlin-gen as the generated representative.

### B2 · fig_capability — functional correctness & op coverage
- **N:** 24 kernels × 5 approaches. **Tier:** **spike L2 — FUNCTIONAL ONLY, explicitly NOT timing**
  (relabelled from the old "capability" framing).
- **Measured:** compiles + exact-int == golden (✓/✗); "·" = not attempted (golden conv template
  deferred).
- **Caveats:** spike does not model the array — this figure makes no timing claim (timing lives only in
  A2/A3/B1). Only merlin-gen (v1) covers conv2d + movement.

### B3 · fig_spike_not_timing — why spike ≠ performance
- **N:** golden on each kernel (one point/kernel). **Tier:** spike vs verilator/FireSim.
- **Measured:** reported cycles, single run/point.
- **Caveats:** the methodological figure justifying B2's "not timing" label — spike cycles plateau ~120
  regardless of MACs (would imply util > 100%); RTL cycles scale with work.

### B4 · fig_iree_profile — IREE profiled on FireSim L5 (its correct oracle)
- **N:** 12 kernels where golden+merlin+IREE all have FireSim L5 cells. **Tier:** FireSim L5 (cycle-
  accurate); IREE measured by its own per-dispatch rdcycle dump (`iree_merlin_dump_cycles`).
- **Measured:** cycles (log scale) + PE-array utilization. IREE runs 10–40× more cycles at 1–6% util.
- **Why FireSim, not verilator:** the IREE ELF embeds the full IREE runtime (530KB .text vs golden's
  5.7KB); at verilator's ~kHz clock it can't reach the kernel in reasonable time. FireSim (~MHz) is its
  oracle. Switching the host VM module to **EmitC** (no bytecode interpreter) cut .text to 294KB (−45%)
  but the shared HAL/runtime-init still dominates on verilator — see EmitC status below. The low IREE
  util is a real result of the dialect lowering, not a measurement artifact.
- **Verify:** IREE = all-ones self-check (rc=0), NOT exact-int golden. **Asm-verified** offload (below).

## IREE oracle, EmitC rebuild, and asm verification
- **Asm verification** (`scripts/verify_gemmini_asm.py`): disassembles the embedded dispatch ELF inside
  an IREE binary and counts Gemmini RoCC custom-3 (opcode 0x7b) ops. A correct 16×16×16 dispatch carries
  **14** custom-3 ops = genuinely offloaded to the systolic array (not a scalar/RVV fallback that would
  still pass a numeric self-check). Both the bytecode build and the EmitC build pass identically (14 ops,
  byte-equivalent dispatch) — the EmitC switch changed only the host VM module, not the kernel.
- **EmitC build** (`/scratch2/agustin/merlin`, additive target `bench_gemmini_spike_matmul_emitc`):
  builds + links + asm-verified; .text 530KB→294KB (VM bytecode interpreter fully removed). **Verilator
  verdict (CONFIRMED):** a 40-min probe (2400s, 99% CPU) reached only the UART banner — no `invoking`/
  `DONE`. The wall is the SHARED IREE runtime init (HAL local-sync + embedded-elf loader + allocator +
  module registration), which EmitC does not touch; at verilator's ~kHz that init exceeds 40 min before
  the matmul invoke. So removing the VM interpreter was necessary-not-sufficient. **IREE's cycle-accurate
  column is therefore FireSim-only (B4)** — the honest and correct representation. The EmitC work still
  paid off: it validated the lighter build is feasible + gemmini-correct, and empirically pinned the
  verilator wall to runtime-init (a precise finding, not a hand-wave).
- **IREE-on-verilator runner** (`run_iree_small_verilator.py`) is now feasibility-aware: a short probe
  records `infeasible` with a reason instead of hanging for hours / leaking verilator children.

## Section C — RTL-derived static checks

### C1 · fig_arc_checks — static checks vs oracle
- **N:** 242 oracle-labelled trace decisions. **Tier:** deterministic decoded-trace checks vs the oracle
  verdict.
- **Measured:** **0/242 false positives**; recall 65% → 88% with the extended check set.
- **Caveats:** dialect-level FileCheck is advisory (multiple legal MLIR surface forms); the verdict
  checks are the format-agnostic decoded-trace ones. Recall < 100% by design (fail-open on unknowns).

### C2 · fig_arc_mutation — pre-screen catches RTL failures fast
- **N:** **1 capsule** (A2), a mutation battery. **Tier:** pre-screen (ms) vs verilator (s).
- **Measured:** screen wall (ms) vs the real verilator wall it would have spent.
- **Caveats:** single-capsule demonstration of iteration-saving (labelled); not a corpus aggregate.

## Section D — agentic authoring effort (the previously-unplotted axis; PILOT)

> **Pilot scale, stated everywhere:** valid converged runs **baseline N=3** (incl. one explicit-C++
> outlier ~$47), **merlin N=1**. Shown as **individual points labelled by run-id with a median bar — NO
> error bars / no fabricated variance**. The merlin full-suite audit is not yet run (pilot 4/4 only).
> Source: `agentic_results.json` from `agg_agentic_results.py`.

### D1 · fig_agentic_effort — authoring effort A/B
- **N:** baseline 3, merlin 1. **Tier:** real run telemetry (`cost_time_toolcalls.yaml`).
- **Measured:** cost (real API pricing), tokens, tool-calls, wall, rounds — per run.
- **Caveats:** N=1 merlin ⇒ no spread claimed; merlin pilot lands at/below the baseline range on
  cost/tokens/tool-calls/rounds, not a significance claim.

### D2 · fig_agentic_convergence — per-round convergence
- **N:** baseline 3, merlin 1. **Tier:** per-round QA-loop telemetry (`qa_loop_summary.yaml`).
- **Measured:** capsules passing after each round (single trace/run).
- **Caveats:** pilot 4-capsule loop; merlin converged in 2 rounds, baseline 2–4.

### D3 · fig_agentic_coverage — capability coverage by op-class
- **N:** baseline full-suite audit, **2 audited runs**. **Tier:** full_suite_audit pass/class.
- **Measured:** capsules passed per op-class; **conv + movement = 0 for baseline (the capability gap)**.
- **Caveats:** the merlin-assisted *v1 backend* compiles conv+movement (see B2), but the merlin *agent*
  full-suite audit is not yet run — annotated on-figure to avoid over-claiming.

### D4 · fig_agentic_per_capsule_effort — downstream efficiency
- **N:** baseline 3, merlin 1. **Tier:** run telemetry ÷ 4 (all valid runs converged 4/4).
- **Measured:** cost/capsule, tokens/capsule.
- **Caveats:** pilot; individual points, no variance.

---

## Open self-critique (what a reviewer should still push on)
- **Agentic N is small** (3 vs 1). Plotted honestly as a pilot; the headline scaling claim is not made.
  Larger N + the merlin full-suite audit are the obvious next runs (budget-gated, out of scope here).
- **A4/A5/C2 are single-capsule** — labelled as such; corpus-wide sweeps would strengthen them.
- **arc ideal-memory** — A4 shows latency sensitivity but arc's default is optimistic vs a real SoC; arc
  is positioned as a *fast pre-FireSim* cycle estimator, not a FireSim replacement.
- **B1 single run per cell** — RTL is deterministic so variance is ~0, but stated rather than assumed.
