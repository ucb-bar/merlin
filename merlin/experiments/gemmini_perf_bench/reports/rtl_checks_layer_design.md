# RTL-derived checks layer — design (task #131, untouched half)

Status: DESIGN + one un-wired first module (`rtl_checks.py`). No frozen contract / grader /
schema semantics are touched. Nothing here gates anything yet.

Repo: `/scratch/agustin/projects/oscar-merlin` (the ACTIVE project — never the deprecated IREE
merlin at `/scratch2/agustin/merlin`).

---

## 1. Motivation

Today correctness/perf of a generated Gemmini kernel is judged by *running the full oracle
ladder*:

```
L0 numpy golden → L1 reference==simulate(cb) → trace_check → L2 spike (functional)
                → L3 verilator (RTL, cycle-accurate) → L4 VCS → L5 FireSim
```

(`merlin/python/merlin/targetgen/capsule_runner.py:run_capsule`.) The expensive truth lives at
L3+ (RTL). Two problems follow:

1. **Cost.** Every candidate the codegen agent emits pays for a spike run (L2) and, for perf, an
   RTL run (L3/L5). Most *broken* candidates are broken in ways that are visible *before*
   simulating — a malformed ISA stream, a tile count that cannot cover the declared shape, a
   scratchpad address past the array's capacity. Paying a full RTL run to discover "you emitted
   zero COMPUTEs" is waste.
2. **Feedback poverty.** When a run fails the agent gets a single `CertFailure` plane +
   `trace_check.violations`. That is pass/fail-shaped. The agent (which *writes the code* — these
   checks never write kernels for it) would iterate faster with **richer, RTL-grounded, graded
   feedback**: not just "trace_check failed" but "you emitted 240 MVOUTs but the declared
   16×256 output over a 16×16 mesh needs Mt·Nt=16 output tiles — you are over-committing ~15×",
   or "your largest spad address (0x4_8000) exceeds the introspected scratchpad capacity
   (0x4_0000) — this will alias/wrap on real silicon".

The **RTL-derived checks layer** mines *cheap, RTL-grounded invariants* that:

- **(a) pre-screen** a candidate before paying for a full RTL/spike run (catch obviously-broken
  codegen at near-zero cost), and
- **(b) give the codegen agent richer-than-pass/fail feedback** (severity-graded, with the
  specific instruction index, the expected-vs-got quantity, and a one-line "why").

### The non-overfit invariant (project principle)

Per memory `abstract-into-compiler-not-overfit`: checks must be **general compiler-level
invariants of the target**, *not* per-capsule golden values or shape-specific magic numbers. The
discriminator used throughout this doc:

> A check is **general** iff its expected value is derived *only* from (i) the target's RTL
> facts (mesh dims, scratchpad/accumulator capacity, datapath widths — from
> `rtl/introspect.py` / `target_contract.yaml`), (ii) the capsule's *declared* tensor shapes and
> modes (which are part of the problem statement, not the answer), and (iii) Gemmini's ISA
> structural rules. It must **never** depend on a recorded golden trace, a per-capsule expected
> instruction list authored by a human, or a hand-tuned constant for one shape.

This is exactly why these are "RTL-*derived*" checks: the bound comes from the *hardware*, and is
the same predicate for every shape the agent might emit.

---

## 2. What already exists (and why this is the *cheap-screen* gap, not a re-implementation)

The repo already has the static decode + a structural verifier:

| module | what it gives us | cost |
|---|---|---|
| `rocc_decode.py` | Runner-owned decode of the package's emitted `lowered.llvm.mlir` into a structured trace (class histogram + per-instruction decoded fields: `spad_addr`, `acc_addr`, `readout`, `acc_scale`, `out_stride_bytes`, `rows/cols`, dram `arg_index`/`offset`, `garbage`). **Fail-closed**: any unrecognised `.insn` → `UNKNOWN`. | **zero simulation** — pure text parse |
| `trace_check.py` | Verifies the decoded trace against a capsule's `expected` block: required/forbidden classes, FENCE bracketing, config-before-use, PRELOAD/COMPUTE pairing, declared-mode exercise, and an optional `_check_tiles` MVOUT == Mt·Nt geometry cross-check. | zero simulation |
| `rtl/introspect.py` | Structure-only RTL facts from elaborated FIRRTL: `mesh.rows/cols` (16×16), scratchpad `bytes` (262144), accumulator presence, datapath widths (i8 / i32). Provenance-tagged; reproduces `target_contract.yaml` capacities. | one-time extraction (cached `rtl_facts.yaml`) |
| `coverage_report.py` | Aggregates class/mode coverage across the suite. | zero sim |

**The key observation:** the decode is rich and the RTL facts exist, but the *only* consumer of
the decode today is `trace_check`, whose contract is **per-capsule-`expected`-driven and
pass/fail-shaped**, and whose one capacity-aware check (`_check_tiles`) is (a) single-matmul only,
(b) not RTL-fact-derived (hard-codes `/16`), and (c) bails silently on anything it doesn't
recognise. There is **no layer that** (1) screens against *hardware* bounds (capacity, mesh,
address legality) independent of a hand-authored `expected`, or (2) produces *graded* feedback
the agent can act on. That is precisely the untouched half of #131.

`trace_check` stays the frozen *gate*. The new layer is an **additive, advisory pre-screen +
feedback producer** that reuses the same decoded trace and the RTL facts. It never changes a
pass/fail verdict; it informs one.

---

## 3. Catalog of proposed checks

Each check declares: **id**, **what it catches**, **input** (and therefore **cost tier**),
**severity**, and **why it is general (not overfit)**.

### Cost tiers

- **T0 — pure static, no simulation.** Operates only on the decoded RoCC trace
  (`rocc_decode` output) + RTL facts + the capsule's declared shapes/modes. ~milliseconds. This
  is the pre-screen tier — run it *before* spending any spike/RTL cycle.
- **T1 — cheap L2 byproduct.** Needs only the spike *functional* pass (L2), which the ladder
  already runs and which (per memory `gemmini-perf-bench`) is **correctness-only, no timing**.
  Uses spike's cheap byproducts (console markers, DONE, commit count) — never an RTL run.
- **T2 — needs RTL (L3+).** Listed only to mark the *boundary*: these are NOT part of this layer
  (they are the expensive oracle). Documented so the catalog is honest about what stays costly.

### T0 checks (pure static — the pre-screen)

| id | catches | input | severity | general because |
|---|---|---|---|---|
| `T0.decode_clean` | malformed / unrecognised ISA stream (any `UNKNOWN` instruction); empty trace | decoded trace | **error** (hard pre-screen reject) | "no UNKNOWN custom-3 forms" is an ABI property of the target, identical for every shape. |
| `T0.fence_bracket` | trace not opened *and* closed by FENCE → DMA/commit may race | decoded trace | error | Gemmini ordering rule; shape-independent. (Mirrors a `trace_check` rule but graded + reported as advisory severity, not a gate.) |
| `T0.config_before_use` | a CONFIG_EX/LD/ST emitted *after* the first instruction that consumes it | decoded trace | error | structural ISA rule. |
| `T0.preload_compute_pairing` | COMPUTE not immediately preceded by PRELOAD; PRELOAD/COMPUTE counts unequal | decoded trace | error | WS micro-sequence rule (memory `gemmini-public-isa-headers`). |
| `T0.movement_compute_balance` | "all movement, no compute" (MVIN+MVOUT > 0 but COMPUTE == 0) for a capsule whose declared op is a matmul/conv; or "compute with no output commit" (COMPUTE > 0, MVOUT == 0) | decoded trace + capsule declared op | **error** | derived from the *declared* operation class (part of the problem), not a golden count. A matmul that emits no COMPUTE is wrong for *any* shape. |
| `T0.tile_coverage` | MVOUT count cannot cover the declared output shape over the mesh: `mvout != ceil(M/mesh_r)·ceil(N/mesh_c)` (single-matmul); for multi-matmul, `mvout < Σ tiles` (lower bound). Reports **expected vs got + ratio**. | decoded trace + declared shapes + **RTL mesh facts** | **warn** (advisory; geometry can legitimately vary with fusion) | the tile count is computed from `mesh.rows/cols` (RTL fact) and the *declared* output shape — never a recorded golden. Generalises `trace_check._check_tiles` by (1) sourcing the mesh from RTL facts not a hard-coded 16, (2) handling multi-matmul as a bound, (3) emitting a graded delta instead of bailing. |
| `T0.spad_capacity` | any `spad_addr` (MVIN) or resident region that exceeds the introspected scratchpad capacity (would alias/wrap on silicon) | decoded trace + **RTL scratchpad bytes** | **error** | bound is the hardware capacity (`resident_storage_bytes` from `introspect.py`), identical for all shapes; this is `must_respect_scratchpad_capacity` from `target_contract.yaml.compiler_obligations`. |
| `T0.acc_addr_region` | MVOUT/PRELOAD readout-region bits inconsistent (e.g. claims i32 readout but addresses the i8-scaled region, or sets accumulate-onto on the *first* write to a tile) | decoded trace | warn | the C_ACC / ACC_I8 / ACC_ACCUM region encoding is a fixed ISA layout (documented in `rocc_decode.py` header), not shape-specific. |
| `T0.acc_scale_present` | capsule declares a requant/acc_scale mode but **no** CONFIG_ST carries a non-identity `acc_scale`; or a non-identity scale appears when the declared output is i32-raw (scale silently ignored) | decoded trace + declared modes | warn | keys off the *declared* mode, not a golden scale value. |
| `T0.readout_dtype_match` | declared `output_dtype: i8` but every MVOUT uses i32 readout (or vice-versa) | decoded trace + declared output dtype | warn | declared dtype is part of the problem statement. |
| `T0.dram_operand_legality` | a DMA operand whose base is `UNKNOWN`/`unresolved` (decode couldn't tie it to an `%arg` base+offset) — a likely address-computation bug the RTL would fault on | decoded trace | warn | "DMA base must resolve to an argument region" is an ABI property. |

### T1 checks (cheap spike byproducts — only if L2 already ran)

| id | catches | input | severity | general because |
|---|---|---|---|---|
| `T1.spike_completed` | spike ran but produced no `DONE` / non-zero rc → kernel hangs or faults functionally (catch *before* paying for L3 RTL) | spike console (L2 byproduct) | error | a completion marker is a target-level liveness property. |
| `T1.commit_count_plausible` | number of output-commit markers spike reports is implausible vs `T0.tile_coverage`'s expected tile count (cross-check static prediction against the cheap dynamic count) | spike console + `T0.tile_coverage` | warn | both sides derived from RTL facts + declared shape; cross-validation, no golden. |

> **Explicit non-members (T2, NOT in this layer):** cycle counts, throughput, anything that needs
> L3 verilator / L5 FireSim. Per memory `gemmini-perf-bench`, spike has *no* timing, so any perf
> claim must stay at the RTL tier. The checks layer never fabricates a perf number from a
> functional run.

### The 3–5 most valuable

1. **`T0.tile_coverage`** — *expected-vs-got* MVOUT tiling vs `ceil(M/mesh)·ceil(N/mesh)`; the
   single most informative "are you even covering the shape" signal, and the richest feedback.
2. **`T0.spad_capacity`** — hard pre-screen against the introspected scratchpad bound; a real
   silicon-correctness obligation (`must_respect_scratchpad_capacity`).
3. **`T0.movement_compute_balance`** — catches the "all MVIN/MVOUT, zero COMPUTE" and
   "compute but never commits" classes instantly, with zero simulation.
4. **`T0.decode_clean`** — any `UNKNOWN` instruction means the RTL will reject it; reject now for
   free instead of after a spike run.
5. **`T1.spike_completed`** — gate the *expensive* L3 RTL run on the *cheap* L2 actually finishing.

---

## 4. Where it slots into the pipeline

```
run_capsule():
   parse → lower_* → emit_cb → lower_to_llvm
   L0 golden ─┐
   L1 ref==sim│   (unchanged, frozen)
   trace_check│   <-- FROZEN GATE (per-capsule expected). Verdict authority unchanged.
              │
   ┌──────────▼─────────────────────────────────────────────────┐
   │  NEW: rtl_checks.screen(trace, capsule, rtl_facts)          │  ← T0, pure static, ~ms
   │   -> CheckReport{ verdict, checks[], advisories[] }         │
   │   * advisory-only by default (writes rtl_checks.json)       │
   │   * OPTIONAL pre-screen mode: if any severity==error, the   │
   │     CALLER (a wrapper / mining loop) MAY skip L2..L5 to save │
   │     cost — capsule_runner itself stays unchanged & frozen   │
   └──────────┬─────────────────────────────────────────────────┘
   L2 spike ──┤
   (T1 checks consume the spike console byproduct here, advisory)
   L3 verilator … L5 FireSim   (unchanged)
```

Two integration modes, **both keeping the frozen runner intact**:

- **Advisory (default, zero-risk).** A thin wrapper (or the mining/feedback loop, *not*
  `capsule_runner`) calls `rtl_checks.screen(...)` on the already-decoded
  `instruction_trace.json` and writes a sibling `rtl_checks.json`. The grader's verdict is
  untouched; the report is pure extra feedback surfaced to the agent.
- **Pre-screen (opt-in, caller-side).** The *mining/codegen driver* (the thing that decides
  whether to spend an RTL run) calls `screen()` first; if `verdict == "reject"` it short-circuits
  before invoking the expensive oracle. `capsule_grade` / `capsule_runner` semantics never change;
  the saving is realised by the *caller* choosing not to run them.

This respects the constraint: **no frozen file is modified, no gate semantics change.** The layer
is consumed *around* the runner, not *inside* it.

---

## 5. Data structure surfaced to the agent

A single JSON object (`rtl_checks.json`), designed to be both human- and agent-readable and to
carry *actionable* deltas, not just booleans:

```jsonc
{
  "schema": "rtl_checks/v0",            // NOT a frozen contract schema; advisory artifact
  "capsule": "A6_resident_reuse",
  "source_trace": ".../instruction_trace.json",
  "rtl_facts": {"mesh": [16,16], "scratchpad_bytes": 262144, "from": "target_contract|introspect"},
  "verdict": "reject" | "warn" | "ok",  // reject == at least one severity:error
  "n_error": 0, "n_warn": 1,
  "checks": [
    {
      "id": "T0.tile_coverage",
      "tier": "T0",
      "severity": "warn",               // error | warn | info
      "status": "fail",                 // pass | fail | skipped (e.g. shape undeclared)
      "message": "MVOUT=240 but declared 16x256 over 16x16 mesh needs Mt*Nt=16 tiles",
      "expected": 16, "got": 240, "ratio": 15.0,
      "evidence": {"instruction_indices": [/* first few offending MVOUT idxs */]},
      "fix_hint": "over-committing outputs ~15x; check the N-tiling loop bound"
    }
  ],
  "skipped": [ {"id":"T1.spike_completed","reason":"L2 console not present (static-only run)"} ]
}
```

Design choices:

- **Graded** (`severity`) so the agent can prioritise; **`status` separate from `severity`** so a
  passing high-severity check is still reported (positive signal).
- **`expected`/`got`/`ratio`** make the delta machine-actionable (the agent can compute "I'm 15×
  off" without parsing prose).
- **`fix_hint`** is a *general* nudge derived from the check, never a per-capsule answer.
- **`skipped`** is explicit (mirrors the repo's `not_run_is_not_pass` honesty — a check that
  couldn't run is never silently a pass).
- It is an **advisory artifact with its own `schema` tag**, deliberately *not* registered in
  `bench_contract/schemas/` so it cannot be confused with the frozen contract.

---

## 6. Phased implementation plan

**Phase 0 — first static check (DONE here, un-wired).**
`merlin/python/merlin/targetgen/rtl_checks.py` implementing the highest-value, zero-risk T0
checks against the *already-decoded* trace: `decode_clean`, `tile_coverage`,
`movement_compute_balance`, `spad_capacity` (+ the `CheckReport` data structure). Pure function of
`(trace_dict, capsule_dict, rtl_facts_dict)`. No import of frozen runner internals; no wiring into
any gate. Ships with a tiny `__main__` so it can be pointed at an existing
`instruction_trace.json`. (See §7.)

**Phase 1 — complete the T0 catalog + tests.**
Add the remaining T0 checks (`fence_bracket`, `config_before_use`, `preload_compute_pairing`,
`acc_addr_region`, `acc_scale_present`, `readout_dtype_match`, `dram_operand_legality`). Add unit
tests under `merlin/python/tests/test_rtl_checks.py` driven by *synthetic* decoded traces and the
RTL facts — **no shape-overfit fixtures** (assert the *predicate*, e.g. "over-commit ratio > 1
fails tile_coverage", never "capsule X must have 16 MVOUTs"). Wire `rtl_facts` to load from
`target_contract.yaml` with `rtl/introspect.dump_facts` as the authoritative override.

**Phase 2 — advisory surfacing.**
A thin `rtl_checks_report.py` (or a flag on the mining driver) that, after a capsule run, reads
the run's `instruction_trace.json`, calls `screen()`, and writes `rtl_checks.json` next to
`capsule_result.json`. Add `rtl_checks` summary into the agent-feedback bundle. Still advisory;
grader untouched.

**Phase 3 — opt-in pre-screen in the mining/codegen driver.**
In the *driver that decides whether to pay for an oracle run* (NOT `capsule_runner`), call
`screen()` on the static trace before invoking L2+; on `verdict == "reject"` skip the oracle and
hand the agent the report. Measure the saved spike/RTL invocations on the existing corpus to
quantify the screen's value (false-reject rate must be ~0 by construction, since every error-tier
check is a true ISA/hardware violation, but verify on the corpus).

**Phase 4 — T1 spike-byproduct checks.**
Add `T1.spike_completed` / `T1.commit_count_plausible` consuming the L2 console the runner already
writes (`*_console.log`). Cross-validate `commit_count_plausible` against `T0.tile_coverage`.

**Phase 5 (optional) — close the loop with mining.**
Feed aggregate `rtl_checks` failure modes into the kernel-policy miner (branch
`feature/kernel-policy-mining`) so recurrent violation classes become *general* compiler
capabilities — never per-shape kernels (memory `abstract-into-compiler-not-overfit`).

---

## 7. The first module (Phase 0, shipped here, UN-WIRED)

`merlin/python/merlin/targetgen/rtl_checks.py` — a new, self-contained module. It:

- imports nothing from the frozen runner/grader (only stdlib);
- is a pure function `screen(trace, capsule, rtl_facts) -> CheckReport`;
- implements four T0 checks (`decode_clean`, `movement_compute_balance`, `tile_coverage`,
  `spad_capacity`) plus the graded `CheckReport`/`Check` data structures from §5;
- defaults `rtl_facts` to the `target_contract.yaml` capacities (mesh 16×16, scratchpad 262144)
  so it is runnable with no RTL run, and accepts an override dict from `introspect.dump_facts`;
- has a `__main__` so you can run it against any existing `instruction_trace.json`:
  `python -m merlin.targetgen.rtl_checks <instruction_trace.json> [--capsule capsule.yaml]`.

It is **not imported by `capsule_runner`, `capsule_grade`, or any schema**, and changes no
existing file. It is advisory scaffolding for Phases 1–3.

> Generality guard built into the module: every `expected` value is computed from RTL
> facts + declared shape only; there is no path that reads a golden trace or a per-capsule
> expected instruction count. `tile_coverage` for multi-matmul degrades to a *lower-bound*
> `warn` rather than asserting an exact count it cannot derive generally.
