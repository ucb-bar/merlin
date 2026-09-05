---
title: GSIM as the elaborated-RTL cert engine — where models live, how a target adopts one
kind: guide
status: current
owner: targetgen
last_verified: 2026-09-04
related: [adding_a_target, hardware_pins, artifact_layout]
code_refs:
  - merlin/python/merlin/targetgen/gsim_emulator.py
  - merlin/python/merlin/targetgen/rtl_engine_policy.py
  - merlin/python/merlin/targetgen/program_oracle.py
  - merlin/python/merlin/targetgen/capsule_runner.py
  - merlin/contract/external/gsim/model_build/recipes.yaml
  - merlin/python/merlin/perf/cycle_trace.py
  - merlin/python/merlin/perf/observations.py
---

# GSIM as the elaborated-RTL cert engine

The L3 cert tier is a **fidelity**, not a simulator. VCS, GSIM and Verilator all run the elaborated
design and all produce an `elaborated_rtl` verdict; which one answers is an availability and cost
decision. `rtl_engine_policy` has encoded that for a long time, ranking `vcs > gsim > verilator`, and it
was never the thing that was wrong.

What was wrong is that GSIM kept losing anyway, on every target, for three separate reasons that all
looked identical from the outside — a run that certified on Verilator.

## The three reasons, and what fixes each

**1. There was no home, only an environment variable.** GSIM emits a standalone C++ model built *out of
tree*, so unlike Verilator there is no simulator rule inside the RTL checkout whose output path can be
derived. Every backend therefore resolved its emulator through a bare env var. On a machine where nobody
had exported it, the probe answered False and the policy correctly fell through.

The fix is a derived home: **`out/build/rtl_engines/<target>/<engine>/`**, resolved by
`merlin.targetgen.gsim_emulator.engine_home`. Installing a build there **is** registering it. Two shapes
live under it, both legitimate:

| shape | contents | driven with |
|---|---|---|
| self-contained emulator | `emulator`, `build_receipt.json` | an ELF (`+loadmem`), the chipyard flavour |
| engine directory | `<engine>_run.py` beside its own binary | assembled program words, the program-oracle flavour |

Env overrides still win — a caller pointing at a freshly built model must not have to install it first —
but they are now the exception. Note the precedence consequence: a stale env registration **shadows** a
correctly-provenanced derived install. If a target's `experiment.env` names an engine directory, that
directory is what certifies.

**2. `os.environ` is not the configuration.** The repo's gitignored `.env` already declared a built GSIM
emulator for one target, and the oracle never saw it, because it read only the process environment. The
machine had the fast engine, the configuration named it, and every cert still ran on Verilator. Overrides
now resolve through `merlin.common.paths.env`, which honours `.env` (process environment still wins).

**3. Falling back was silent.** This is the deepest one. A cert that ran on Verilator because the fast
engine was absent produced the same artifact as one that ran on Verilator because it is the only engine
there is. Nothing distinguished them, so "why does the cert tier cost hours per capsule" had no answer
anywhere, and the fast engine stayed unbuilt or unregistered for months.

Now the selection travels:

- `capsule_runner.describe_l3_engine(target)` answers "which engine would certify, and what was it chosen
  over" in one shape whatever the target's routing (chipyard / bespoke sim / program oracle);
- `readiness_check` prints it and prints every passed-over engine **with its reason**, which is the
  moment a human is actually looking;
- each `capsule_result.json` tier record carries `engine`, `engine_selection` (considered + passed over +
  why) and `sim_provenance` (the digest of the binary that produced the numbers), alongside
  `console_log`/`console_bytes` — the telemetry a perf-tuning pass reads.

## Provenance: a binary in the right place proves nothing

A model must carry a `build_receipt.json` (schema `merlin.gsim-model-build.v2`, written by
`produce_gsim_certificate.py`) binding **these bytes** to the FIRRTL and the tools that produced them.
`gsim_emulator.resolve` enforces it as three states, never two:

- **bound** — the receipt's binary digest is this file. Cite it.
- **absent** — no receipt. Usable, and the reason string says `provenance UNRECORDED` every single time it
  is selected, so the sentence reaches the record. `MERLIN_GSIM_REQUIRE_RECEIPT=1` makes it fatal for a
  run that must not certify on unattributable bytes.
- **invalid** — a receipt that binds a *different* digest. **Refused.** It says nothing about these bytes
  and reads, to anyone who opens it, as though it did. Refused at install time too, so a mis-bound pair
  never lands.

`refused` and `absent` are deliberately different states: absent is work not done yet, refused is bytes
that exist and may not be trusted.

## Choosing which build is canonical

By provenance, never by filename or mtime. The worked example: seven built emulators existed for one
target; three carried receipts; all three bound the *same* elaborated FIRRTL, so none was off-RTL relative
to the others. What separated them was evidence of agreement — one carried a 38-member GSIM-vs-Verilator
equivalence certificate (all AGREE, bytes match, zero unresolved), the others one member each. The four
unreceipted binaries were unattributable and were not candidates at all.

If two candidates match on provenance and nothing else separates them, **stop and say so**. Certifying
against the wrong RTL revision is the hazard the whole convention exists to prevent.

## Building a model for a new target

`merlin/contract/external/gsim/model_build/` holds the centralized recipe — `recipes.yaml` (the
declaration: designs, stages, fixes, harnesses, and an honest capability line per config) plus every fix
script, harness and blackbox stub it references. `../gsim-merlin-patches.diff` carries merlin's deltas
against the pinned upstream `gsim_compiler`.

The one distinction to get right before wiring anything is **capability class**, and `recipes.yaml`
records it per config:

- **`program_oracle`** — can be given an arbitrary program and return results. This is what a capsule
  needs, and only this may hold a cert tier.
- **`register_observation`** — steps the design and exposes every register, but has no way to receive a
  program. Genuinely useful (it is the per-cycle substrate for cycle accounting) and **not** a cert
  engine. A full SoC that boots its bootrom and then stalls with nothing to execute is in this class.

A probe that reports a `register_observation` model as an available cert tier is exactly the silent
degradation the engine policy exists to prevent, and would be worse than the Verilator fallback it
replaced.

## Timing observations: folded from the trace, not re-simulated

The program-oracle GSIM harness returns `{halted, cycles, outputs, reads, writes}` and no
`timing_observations`, so every consumer reading occupancy telemetry saw it as an instrument with no
timing capability — and adopting the faster engine silently cost a perf campaign its per-capsule
decomposition. That was never a fidelity gap. The GSIM harness *does* sample the design's own activity
ports on every cycle; it just **dumps** them (`argv[2]`, one CSV row per cycle) instead of accumulating
the buckets in-sim the way the Verilator harness does.

So the block is produced by **reduction**, in `merlin.perf.cycle_trace`: the trace the engine already
wrote is folded into the same quantities, with the same spellings, that an in-sim harness emits
(`busy_cycles.<unit>.in_program`, `idle_cycles.no_unit_busy`, `overlap_cycles.observed` /
`.across_kinds`, `sampled_cycles.dbg_tap`). Every number is a count of rows in a file the engine
produced — nothing is modelled, estimated or scaled. `program_oracle` asks for the trace whenever the
engine declares its columns, folds it, and deletes the temporary.

**Which column is a unit's busy signal is DECLARED, never guessed.** A trace is a table of integers,
and every way of inferring occupancy from it is wrong in the flattering direction: picking columns
whose names end in "busy" reads a role out of an identifier and drops any port spelled otherwise
(making that unit read as permanently idle); "nonzero means busy" is simply false on a state register,
where state 0 is a state. So each engine home carries `timing_columns.json`
(`merlin.cycle-trace-columns.v1`) written by whoever chose the columns, binding each to a unit and a
kind and naming what it could not read. **An engine with no declaration reports no timing capability**,
exactly as before — an absent instrument stays absent rather than becoming a block of zeros.

Two rules the declaration must get right, both of which produced wrong numbers on the first pass here:

- **Columns that nest fold into one unit.** `vloadBusy` and `vstoreBusy` sit inside `lsuBusy` (measured:
  68 + 68 = 136 on `spec_AT4_bf16_scale`). Binding them as separate units charged the LSU three cycles
  for one and reported 136 cycles of overlap in a kernel with none — the unit overlapping with itself.
- **Idle is relative to what was read.** A unit whose port the trace omits contributes no busy cycles,
  so every cycle it alone was busy in lands in idle. When anything is unmeasured the entry says so and
  calls itself an upper bound, rather than being quietly compared against an instrument reading more.

**Cross-validated against the in-sim instrument (2026-09-04).** The same two programs were run on the
GSIM engine (folded from its trace) and on the Verilator engine over the same design (accumulated
in-sim). Every unit both instruments carry agreed **exactly** — on `spec_AF3_attn_full_bf16_pt`, `lsu`
1088, `mxu0Comp` 376, `mxu0Data` 224, `mxu1Comp` 0, `mxu1Data` 0, `xlu` 130 — as did
`sampled_cycles.dbg_tap` (7818) and both overlap readings. The two differ only where Verilator reports
eight per-channel DMA units and the DMA-beat counters that this trace does not carry: those are named
in `unmeasured_units`, and the idle figures differ by precisely their cycles, which is why idle from a
trace is a bound and not a figure to compare across engines.

## Why atlas and radiance cannot be promoted by a rebuild (2026-09-04)

Both engines are refused under `MERLIN_GSIM_REQUIRE_RECEIPT=1` — radiance because no receipt sits
beside it (`lineage: unrecorded`), atlas because its adoption record covers the bytes but nothing built
them here (`adopted`, not `bound`). The obvious fix, and the one both records propose, is a
receipt-writing rebuild from the FIRRTL already on disk. **That would not close either gap, and it
would make the record worse.**

A build receipt binds a binary to the FIRRTL it was compiled from. It answers "which bytes did I
elaborate?", not "which hardware revision is this?". The chain only reaches hardware if the FIRRTL
itself is attributable, and for both targets it is not:

- **atlas.** `AtlasCore.fir` verifies by content as the registered artifact `atlas_core_firrtl_gsim`
  (`eb99a31f`), and that entry says in full why the digest is all there is: *"A COMMIT CANNOT IDENTIFY
  THIS FILE"* — it was produced under a gitignored build dir in a checkout carrying 67 uncommitted
  changes, and the lifting worktree that made it has since been deleted. `atlas_npu` has meanwhile
  drifted from the pinned `569b7c31` to `d6770b6d` on a different branch. Rebuilding from those bytes
  yields a receipt whose `firrtl_sha256` traces to an artifact that is explicitly unattributable.
- **radiance.** The emulator's source is `RadianceGsimConfig.fir`, elaborated 2026-08-14 under the
  chipyard checkout. That checkout is now at `d45f86f4` (2026-09-02) with `generators/radiance` at
  `82cd2e1f`, and the elaboration recorded nothing — the `.d` and `chisel.log` beside the FIRRTL carry
  parameters, not revisions. `radiance_muon_rtl` additionally reports the checkout is a *different
  repository* from the one the pin declares (`ucb-bar/radiance` vs `ucb-bar/chipyard.git`) at a
  different commit. So the revision the model was elaborated from is not recoverable from what exists.

Minting a receipt over either would flip a status that currently reads UNRECORDED — truthfully — to
one that reads bound, while the hardware question stayed exactly as unanswered. That is the failure
this convention exists to prevent, stated in its own words: *a result attributed to the wrong device is
worse than no result, because it gets cited.* The `recipes.yaml` debt note for atlas already names the
real fix — **re-elaborate from the pinned checkout and mint a receipt** — and that is a fresh
elaboration whose output must then be re-qualified against Verilator, not a repackaging of the bytes on
disk. Both remain open, deliberately.

## Radiance runs, but its completion could not be observed (2026-09-04)

Acceptance for radiance had never been backed by a run, so one was made: capsule `R0_gemm_fp32`, cb
from the interface MLIR, kernel from the reference emitter (`emit_kernel_mlir`), compiled fork-free to
rv32, fused into the rv64 SoC carrier and run on the GSIM model through `gsim_muon_adapter`. It
executed — 386,090 cycles of real Muon dispatch/issue/writeback, exit 0, well inside an 8M cycle cap —
and the adapter returned a pass.

**The pass was vacuous.** The console carried none of the four contract markers: no `Cycles:`, no
`finished execution`, and equally no `Timeout exceeded` and no `FINISHED: cycles=`. The emulator's own
stats line read `dram_aw=0 dram_w=0 writes_resultpage=0 uart_chars=0` — the kernel wrote nothing to
DRAM and printed nothing (the same silent-console symptom recorded for this target elsewhere). The
completion test was a double negative, asking only whether the two FAILURE markers were absent, so it
could not tell "the GPU went idle having finished" from "this harness never printed a word".

The test now requires a positive witness and reports honest-unavailable without one, which is the
standard the Verilator sibling already held (`_run_verilator` grades on the `finished execution`
marker). Radiance L3 on GSIM therefore reports **unavailable** rather than passing — the engine and
the toolchain are demonstrably working end to end, and what is missing is the observability to grade
the result, which is a harness gap to close and not a verdict to keep.

## Verified end to end (2026-09-04)

The claim "GSIM runs our accelerators" is only worth what a reproduction says, so both flavours were
re-run from the installed homes rather than from the build trees they came out of.

**Self-contained emulator (gemmini).** `out/build/rtl_engines/gemmini/gsim/emulator` (`1a3de02a`) was run
on capsule `PL01_k16`'s ELF (`8683753b`, the exact ELF the equivalence certificate names). It exited 0 in
27.9 s wall, reported `METRIC cycles 746`, and its stdout digests to `0a765541` — **byte-identical** to the
console the certificate records for that member. The link consumes all sixteen `TestHarness0..15` objects,
which is the check `gemmini_gsim_model_testharness` warns about: a short link still runs, while silently
simulating an incomplete design.

**Engine directory (atlas).** `out/build/rtl_engines/atlas/gsim/gsim_run.py` ran
`evidence/spec_AF4_gelu_bf16_pt.json` to `halted=True` in 1252 cycles, 512 bytes of output, in 0.01 s.

All three targets that have a model now select GSIM through the real path
(`capsule_runner.describe_l3_engine`): gemmini and atlas as `gsim [elaborated_rtl] (over vcs)`, radiance as
`gsim [elaborated_rtl]`.

### The defect this surfaced: a probe blind to one of its own two shapes

The table above has always listed two legitimate shapes, and `record_adoption` exists to install the
directory-shaped one — but `resolve()` looked **only** for the `emulator` binary. So atlas, whose GSIM
engine is cycle-exact against Verilator on 17/17 programs at 32x the speed and sits installed in its
derived home, made `gsim_emulator.probe('atlas')` answer **False**. It selected GSIM anyway only because
`program_oracle` resolves the wrapper by a separate path — two modules disagreeing about the same fact,
with the pessimistic one authoritative anywhere the optimistic one was not consulted.

`resolve()` now answers for both shapes and reports which one answered (`Resolution.flavour`). The
lineage standard is not weakened to fit the new shape: a wrapper home carries no build receipt binding
the wrapper's own bytes, so its adoption record must **cover those exact bytes** to count, the status is
reported as `adopted` rather than `bound`, and `MERLIN_GSIM_REQUIRE_RECEIPT=1` refuses it — an adopted
lineage is weaker than a built-and-bound one and the two are not quietly equated.

### Engine drift in the cost model

`cert_cost` fitted certification seconds over samples with **no engine discriminator**, while the same
capsule costs 3.31 s on GSIM and 86.83 s on Verilator. The per-capsule record carried `engine` and the
reshaping into `by_tier` dropped it, so the mixture was not merely unhandled — it was invisible. The
engine now survives that reshaping and rides in the sample's `basis` string, so a fit over two engines is
readable off its own sources.

### Toolchain pin reconciliation

`gsim_compiler` declared `50b371c6`/`master`; the checkout had moved to `65a1f89a` on
`submission/ASPLOS2026-gsim`. The drift is **nominal**: `50b371c6` is an ancestor, and all nine files in
the read set are byte-identical to the pin's declared `local_edits` digests and to their bytes at the new
HEAD — the patch set that was uncommitted when the pin was written has since been committed unchanged.
Verified by comparing each file's sha256 in three places (worktree, HEAD, `50b371c6`) before touching the
pin, which is what "verify by content, not by branch name" means in practice.

### Certifying bytes moved out of purgeable scratch

The elaboration the installed emulator is actually built from (`089d053b`, `GemminiGsimSerialClkConfig`)
existed **only** under `/scratch/agustin/tmp/`, which the layout convention marks purgeable — and it
matched neither registered artifact (`gemmini_gsim_model` is rooted at ChipTop, `..._testharness` declares
the hand-edited bytes). Both entries verified, so nothing reported a problem; the cert engine had simply
moved to bytes no entry described. The FIRRTL, its model manifest and the 38-member equivalence
certificate now sit beside the emulator in its engine home and are registered as
`gemmini_gsim_model_serialclk`.
