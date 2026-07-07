# RTL-derived checks layer — build summary (branch `feature/rtl-derived-checks`)

Deterministic, RTL-grounded checks for the Gemmini codegen agent — **compiled from the hardware via
CIRCT, not LLM-lifted** — plus an isolated agentic A/B track. Built + validated; the agentic A/B run is
deferred to a user trigger (no agent budget spent). Companion: `rtl_checks_layer_design.md`.

## Pillar 1 — static RTL-derived checks (DONE, validated)

**A. Deterministic facts from RTL (`merlin/python/merlin/targetgen/rtl/circt_introspect.py`).**
`firtool --ir-hw` lowers the elaborated Gemmini SoC `.fir` (125 MB) to the CIRCT HW dialect in **11.6 s**
(cached at `merlin/targets/gemmini/contracts/rtl_facts/gemmini_soc.hw.mlir`). From it we extract, each with
evidence + provenance, and cache to `facts.json` (keyed on input SHAs):
- accumulator depth/banks/bytes/addr-width — from `@AccumulatorMem` HW port widths (the gap grep-v1 left
  `None`): **2 banks × 32 KB = 64 KB, addr i9**;
- mesh + scratchpad — reused from the v1 grep;
- the **legal RoCC funct decode table {0..25}** + custom opcode `0x7b` — from `GemminiISA.scala`.

`validate()` cross-checks all of it: mesh/scratchpad/dtypes/accumulator **reproduce** `target_contract.yaml`,
and `rocc_decode._FUNCT_CLASS ⊆ RTL legal set` — **zero divergence**. (En route it caught a real bug in my
own extractor: per-bank vs total accumulator; Chisel `acc_banks=2, acc_capacity=64KB` confirmed the contract.)

**B. Checks compiled to FileCheck (`rtl_check_compiler.py` + `rtl_check_runner.py`).**
Per the user's direction, the facts + a capsule's *declared* shape **compile into FileCheck assertions**
(`CHECK-DAG`, `CHECK-COUNT`, end-of-line-anchored numeric `CHECK`) run by the LLVM `FileCheck` binary over:
- the agent's gemmini-dialect MLIR (`lowered.target.mlir`): `res_pack`→`matmul`→`commit`, `output_dtype`;
- a canonical render of the decoded RoCC trace: exact tile coverage `MVOUT_COUNT = ⌈M/DIM⌉·⌈N/DIM⌉`,
  `ILLEGAL_FUNCT_COUNT 0` (funct ∈ RTL legal set), `COMPUTE_PRESENT`.
Arithmetic is evaluated at check-generation time, so RTL-grounded literals are baked into the CHECK lines;
FileCheck's diagnostic (the offending IR line) is the feedback. Numeric capacity / multi-matmul lower
bounds that don't fit FileCheck stay in the Python `rtl_checks.screen()` (now reading the CIRCT facts +
the `T0.decode_funct_legal` check). Every `expected` derives from RTL facts + declared shape — **no golden**.

**Validation (`merlin/python/tests/test_rtl_checks.py`, 7 passing):**
- **0 false rejects on 20/20 known-good** capsule_bench_v1 runs (both FileCheck levels + Python screen);
- **catches** synthetic corruptions with precise diagnostics: over-commit MVOUT (tile coverage), illegal
  funct, missing COMPUTE, over-capacity scratchpad address;
- regression test that `MVOUT_COUNT 1` does **not** substring-match `MVOUT_COUNT 16` (the `{{$}}` anchor).

This is the "save VCS/verilator iterations" pre-screen: a hard reject is a near-certain RTL-oracle failure,
surfaced in ms as a specific message instead of a multi-minute failed run + log dive. `prescreen()` lets a
caller skip the oracle on reject (frozen runner never consulted).

## Pillar 1 — isolated agentic A/B track (DONE, validated, NOT run)

`merlin_assisted_rtlchecks`: an **exact mirror of `merlin_assisted` + the rtl_checks feedback**, built with
**zero edits to any existing tracked file** (verified by `git diff`; frozen grader/runner/contract and the
baseline loop/qa_check untouched). Purely additive:
- `input_bundles/merlin_assisted_rtlchecks_public_v0/` — copy of the merlin bundle (identical allowed/denied)
  + an `rtl_checks` section appended to `TASK_ADDENDUM.md`;
- `scripts/qa_check_rtlchecks.py` — wraps the real `qa_check.run`, appends an **answer-free** `rtl_checks`
  block (FileCheck pass/diag + RTL-derived findings with expected/got/fix_hint) to the redacted verdict the
  agent reads each round;
- `scripts/run_rtlchecks_qa_loop.py` — reuses the **unmodified** baseline loop, swapping only (in-process)
  the merlin bundle → rtlchecks bundle and `qa_check` → the wrapper.

Validated dry (no agent spend): swaps take, the loop reuses the untouched baseline module, and the
`rtl_checks` block assembles answer-free over an existing 20-capsule runs tree. **To run the A/B:**
`run_rtlchecks_qa_loop.py --run-id rtlchecks_0001 --model claude-opus-4-8` vs the existing `merlin_assisted`
arm — same task/accounting, only difference is the RTL feedback.

## Pillar 2 — dynamic "middle" simulator (arcilator): feasibility spike → DEFER

Goal recap: a tier between spike (fast, not faithful) and verilator/VCS (faithful, slow) — RTL-faithful
numerics+cycles without the SoC-boot overhead, used to triage candidates that pass the static checks.

**Spike findings (timeboxed):**
- arcilator is built and runs a fixed JIT pipeline (`--run`, `--observe-memories`, `--jit-vcd-file`).
- It does **not** ingest the `firtool --ir-hw` output directly: the HW MLIR carries `sv.macro.ref`
  assertion macros arcilator rejects. **Surmountable** — re-emit with `firtool --verification-flavor` /
  strip via `circt-opt` (`--arc-lower-verif-simulations`, `--convert-to-arcs`).
- The **real cost is untouched**: the accelerator is 731 mangled/quoted modules with no clean Gemmini root;
  isolating its subtree **and** building a RoCC + scratchpad/DMA drive harness to feed it the decoded
  command stream and read back the accumulator is a genuine multi-day sub-project.

**Verdict: feasible, worth doing later, not a quick add.** Matches the locked decision (build Pillar 1 now;
Pillar 2 = pre-screen+triage, spiked only). The static checks already deliver the pre-screen value today.

## Corroboration against REAL agent output (383 runs) — honest results

Ran the RTL checks over **383 real agent-authored capsule runs** with both an oracle verdict and a decoded
trace, then *characterized provenance* (because a first cut implied more than the data supports).

**What the 141 "failures" actually are.** 126/141 are **intermediate agentic-round scratch dirs**
(`raw_baseline|merlin_assisted/.../_qa_work/runs_01,02/...`) — the agent's early drafts before it converged,
NOT final submissions. By plane: **125 failed at the grader's own `trace_check`** (a cheap structural gate
that runs *before* spike), 10 at spike, 3 verilator, 1 VCS. The converged/final backends pass (as expected).

**1. RTL extraction is VALIDATED (the core question).** Of the **181 runs that passed verilator/VCS**
(genuinely-correct code), the RTL pre-screen rejected **0** — zero false positives. A wrong DIM / capacity /
tile formula / legal-funct set would have false-rejected some of those 181; it didn't. The compiled facts
are right.

**2. The catches are NOT new signal.** The caught failures are a subset of failures the grader **already
flags at its own `trace_check` plane**. The RTL checks largely *re-derive* `trace_check` on the
intermediate drafts.

**Recall closed (follow-on).** After adding three general WS-protocol invariants — `preload_before_compute`,
`config_before_use`, and especially **`fence_bracket`** (trace must open+close with FENCE; the dominant
real-draft violation) — recall on real failures rose **65% → 88% (124/141)** with the **0/242
false-positive rate held**. The remaining 17 misses are the genuinely-uncatchable-by-static class: 9
numerical `functional_mismatch` (need the oracle / arc middle-tier by design), plus a few tool-crashes and
one mode-specific (resident_reuse CONFIG_EX count). So the static layer now catches essentially all the
*structural* failure classes a general (non-overfit) checker can, and cleanly hands the numerical class to
the arc tier.

**3. Honest limitation — no demonstrated oracle-iteration saving in this corpus.** Of the failures that
slipped *past* `trace_check` to the expensive tiers (spike/verilator/VCS), the static checks caught **0** —
those were `functional_mismatch` (**numerical**), out of scope for static checks by design. So in this data
the layer does not save expensive-oracle iterations beyond the existing `trace_check` gate.

**VCS / expensive-tier reconciliation (checked, not assumed).** Tier coverage across the 383 runs:
L0/L1 all pass; L2 spike 224 pass / 10 fail; **L3 verilator 181 pass / 3 fail**; **L4 VCS: 1 run, fail**.
Every expensive-tier "fail" is a **`tool_crash`**, NOT a codegen bug: the 3 verilator fails are all
`G06_64×64×64` (the verilator-*infeasible* kernel — it passes on FireSim L5 at 7080 cyc), and the 1 VCS
run is a VCS-environment crash on the simplest matmul (wrong config, gdb stack dump). VCS and verilator simulate the **same Gemmini RTL**, so for correctness they are the same verdict
(which one ran is incidental). So there are **zero real codegen failures at the RTL-sim tier** in the
data: the RTL-sim verdict is **181 pass / 0 real codegen fails** (the 3 verilator + 1 VCS "fails" are all
tool_crashes). The converged code **passes RTL simulation** — i.e. the verilator-L3 passes ARE, for
practical purposes, the "passes in VCS". Consequence: this corpus has **no real RTL-tier codegen failure
for a pre-screen to catch**, so the "saves an RTL-sim iteration" claim has no positive example here — the
layer is validated as *correct + safe* (0 FP on 181 RTL-passing runs), not yet *demonstrated as
cost-saving*. The checks correctly call all tool_crash runs `ok`.

**Where the layer's distinct value actually lies** (not iteration-saving, as first overstated):
- it derives expectations from **RTL facts + declared shape — no per-capsule golden**, i.e. the *general,
  non-overfit* form of the frozen, per-capsule-expected `trace_check` (honours abstract-into-compiler);
- it gives **richer feedback** (`expected`/`got`/`ratio`/`fix_hint` + FileCheck line pointers) vs pass/fail —
  the real lever for the agentic A/B;
- the **numerical gap is the arc middle-tier's job**, confirming Pillar 2 is the part that would add the
  expensive-tier coverage static checks can't.

**This corroboration also caught a real bug.** The dialect-level FileCheck was brittle to the agent's
multiple legal MLIR surface forms (op-form `%x = gemmini.<op>` vs attribute-encoded `gemmini.program=[...]`)
and false-failed passing runs (89 FPs). Fix: the **verdict rides only the format-agnostic decoded-trace
checks**; the dialect check is **advisory-only** (skipped on non-op-form). FPs 89 → 0. Lesson: RTL-grounded
checks must key off the *canonical decoded RoCC stream*, not MLIR surface syntax.

## Status
Tasks #136–#141 done; #131 complete. Corroborated on 383 real runs (0 false positives). Nothing here
touches the FireSim L5 backfill (#130) or the perf-bench reports.
