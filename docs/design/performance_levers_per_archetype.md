---
title: "Design note: the performance lever is a property of the archetype, not of the compiler"
kind: design
status: current
owner: core
last_verified: 2026-08-30
related: [performance_budget_unit, expert_gap_attribution]
code_refs: [merlin/python/merlin/perf/occupancy.py, merlin/python/merlin/perf/differential.py, merlin/python/merlin/targetgen/rtl/fsm.py, merlin/python/merlin/perf/depgraph.py, merlin/python/merlin/perf/deps/measured.py]
---

# The lever is a property of the archetype

The performance layer was built against one accelerator and produced a real result there: generated
kernels beat the shipped ones 4.64x / 2.90x / 3.87x, bit-exact, and the lever held at 2.51-2.55x
across a twelvefold span of problem size. Porting that work to a second accelerator established
something more useful than a second number: **the lever itself does not port, and the reason is
structural.**

This note records what transfers, what does not, and the three instrument failures found on the way,
because each was a case of a fact about *our view* of a machine being reported as a fact about the
machine.

## Two archetypes, two opposite levers

|  | non-interlocked, self-hosted program | hardware-interlocked, command-driven |
|---|---|---|
| how hazards are resolved | the compiler emits an explicit separation | a reservation station tracks dependencies |
| the lever | **remove** over-conservative separations | **reorder** the stream so units overlap |
| is the lever safe? | no — every step can change the answer | yes — order cannot change the answer |
| what the falsifier catches | correctness (it fired on 3 of 4 levers) | almost nothing, and that is the problem |

On the first machine, separation padding was **63.9% of a tile kernel** (measured per instruction),
so cutting it to the measured floor was most of the win. That lever cannot exist on the second
machine: its instruction set has no stall, and its hazards are resolved in hardware.

On the second machine the lever is scheduling — overlapping movement with compute, keeping an operand
resident across tiles, hoisting an issue ahead of a wait. Its decoupled load / execute / store
controllers exist precisely to be kept busy at once.

### The consequence that is easy to miss

**A falsifier that cannot fire establishes nothing**, and the two archetypes fail differently. On the
non-interlocked machine, bit-exactness is a sharp falsifier: an over-aggressive schedule returns the
wrong answer, loudly. On the interlocked machine the hardware protects you, so *every* reordering
passes the correctness check. A scheduling capsule whose falsifier is bit-exactness therefore learns
nothing there, and its falsifier must instead be **"the reordering did not increase overlap"**,
measured on a joint occupancy vector. Otherwise every candidate "passes" — which is the inert-lever
trap wearing a new costume.

### Fine-grained beats coarse, and it is not close

An interlocked accelerator may expose hardware-loop macros that expand a whole tiled operation on a
fixed, hardware-chosen schedule. They are convenient and they are what a vendor library uses, so
emitting them concedes the scheduling decision and at best ties the baseline. The compiler's whole
advantage is the fine-grained stream, because that is the only level at which any of the above levers
exist. Those macros are therefore left unused **deliberately**, not as unclosed coverage.

Measured, on the same workload: inserting one drain before a configuration change dropped
load-controller busy from 41 to 30 cycles. Serializing the stream destroys the concurrency the
hardware was built to provide.

## What ports, and what does not

Of the performance modules, roughly ten port unchanged (composition, envelope, decomposition,
differential comparison, occupancy, contract and profile derivation, comparand, observation
validation, oracle cost) and six port through a parameter they already take. **Four are structurally
shaped by the first archetype** and should be trait-gated off rather than generalised:

* the kernel emitter — it assembles a self-hosted program with branches and a scalar register file;
* the dependence graph — its edges *are* compiler-inserted separations, and it takes the stall
  mnemonic as a required argument;
* the measured-separation confrontation — every entry point consumes a per-cycle program counter, and
  a machine with three decoupled controllers has no single one;
* the issue-model probe — it sweeps a stall immediate that does not exist.

The missing piece that lets a consumer decide is a **trait for hardware dependency tracking**. Without
it, those modules take an archetype-shaped parameter instead of consulting a fact, which is the
coupling the repo's cardinal rule exists to prevent.

## Three instrument failures, and the rules they produced

Each of these produced a plausible, interesting, wrong number. All three are the same error.

**A unit with no top-level busy port reads as permanently idle.** One machine's vector unit is not
exposed as a port; every port-based instrument counted it as zero, which inflated the corpus idle
figure and made that unit's overlap unobservable by construction. Including it moved one kernel's
idle fraction from 89.9% to 39.2%, and the figure that motivated the whole workstream — 76.7% of
cycles with nothing busy — is 46.2%. *Rule:* an unmeasured unit is UNKNOWN, never idle, and the FSM
inventory is derivable (`targetgen/rtl/fsm.py`) rather than something to notice by hand.

**Zero overlap from a vector that could not have shown overlap is not a measurement.** A joint vector
with fewer than two live columns reports zero arithmetically, and that zero is indistinguishable from
a machine that genuinely serialises. *Rule:* `joint_counts` reports `overlap_observable`, and a zero
without it is not evidence.

**A limit found in our own harness is evidence about the harness.** A 1 MiB memory window was recorded
as a hardware constraint that made whole-model measurement impossible. It is a constant in our own
test harness — the measured footprint is about 7 KB against a base address the responder indexed from
zero, so every access counted as a "wrap". *Rule:* before promoting a limit to a fact about the
machine, check whether the number is ours.

A fourth, from the derivation side: **synthesis answers a different question than observability.** A
tool that exports state machines exports only those whose *re-encoding* would pay off, dropping the
rest as "recoding might result in larger circuit". Measured: fifteen detected, three exported, and
the two controllers whose concurrency was the entire point were among the twelve dropped. Take the
detection, not the export.

## What this implies for a third target

Ask, in order: does the ISA carry an explicit separation (if not, the first machine's headline lever
is absent); are the engines decoupled (if so, expect non-zero overlap and treat a zero as an
instrument fault); and can a falsifier fire on correctness (if not, the falsifier must move to
occupancy). The answers come from the target's own declarations and RTL, and they decide which capsule
families apply before any measurement is bought.
