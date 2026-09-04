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

## Known capability gap

The program-oracle GSIM harness returns `{halted, cycles, outputs, reads, writes}` and no
`timing_observations` — the per-unit busy / DMA-beat / overlap decomposition the Verilator harness reads
off the design's own activity ports. It is reported as absent rather than fabricated, which is correct,
but it means adopting GSIM changes more than cost for anything reading occupancy telemetry. That is a
harness gap in the GSIM-side harness, not a fidelity gap, and it is the next thing to close.

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
