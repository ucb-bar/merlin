---
title: "Design note: the reorder emitter a command-driven target needs"
kind: design
status: draft
owner: core
last_verified: 2026-08-31
related: [performance_levers_per_archetype, memory_mapping_obligations]
code_refs: [merlin/python/merlin/perf/workload_gen.py, merlin/python/merlin/targetgen/contract/interface_emit.py, merlin/python/merlin/perf/falsifier.py, merlin/contract/capsules/profiles/_perf.yaml]
---

# The reorder emitter a command-driven target needs

The inter-layer scheduling family — issue a transfer before the wait it does not depend on — is the
**only** performance lever a hardware-interlocked, command-driven accelerator has. Its trait gate now
passes: the target's elaborated FIRRTL exposes a decoupled per-engine completion channel, so the
measurement is admissible. It is blocked on its emitter, and the emitter it is declared against cannot
serve it.

This note says what the emitter must be, so that building it is a specification rather than a guess.

## Why the declared entry is wrong

`_perf.yaml` gives the family `emitter.entry = merlin.perf.workload_gen.plan_matmul` with
`status: new:instruction_reorder`. That generator:

* allocates from a **scalar register file** and refuses when the target declares too few registers;
* encodes **backward branches** from the machine's control-flow contract;
* emits **assembly text** (`kernel_s`).

The target this family is now admissible on declares `self_hosted_program: False` — 26 legal opcodes,
widest `0x7e`, endpoint `inline_asm_insn`. It has no self-hosted program, no branches of its own and no
scalar register file to allocate from. So the entry names the *other* archetype's generator, and adding
a knob to it cannot produce a command stream. That mis-declaration is the actual blocker, and it is
worth fixing in the profile even before the emitter exists, so the family's status stops implying that
a knob is all that is missing.

## What the emitter must produce

A **pair** of command streams over the frozen `merlin_iface` grammar, differing only in order.

The grammar already carries an ordered command list — a real capsule parses to
`RES_PACK, MATMUL_RESIDENT, COMMIT, MATMUL_RESIDENT, COMMIT, EVICT` — so the reorder is a permutation
of an existing list rather than a new dialect. The two members are then, by construction, the same work.

```
not_hoisted:  … stage-1 transfer, WAIT, stage-1 compute, stage-2 transfer, WAIT, stage-2 compute …
hoisted:      … stage-1 transfer, stage-2 transfer, WAIT, stage-1 compute, WAIT, stage-2 compute …
```

## The four obligations on it, and why each is load-bearing

**1. Identical work, provable rather than asserted.** The pair must carry the same operand bytes, the
same golden, and the same command MULTISET — only the order may differ. η is a ratio, so a candidate
that quietly does less work raises it without scheduling anything better; the falsifier already refuses
to judge when a work fingerprint differs, and the emitter must supply that fingerprint rather than
leave it absent.

**2. The permutation must be legal, and legality is derived.** A transfer may be hoisted above a wait
only when it does not depend on what that wait is waiting for. The dependence is over the command list's
own operand names (a `COMMIT` writing `Y0` before a `MATMUL_RESIDENT` reading `Y0` may not be reordered),
which is readable from the parsed stream. This is not a hazard question — the reservation station handles
hazards — it is a *semantic* question about whether the two orders compute the same thing.

**3. Reordering is safe here, which is exactly why bit-exactness proves nothing.** On this archetype the
hardware enforces dependencies, so both members return the same answer whatever the order. A capsule
gated on correctness therefore passes every candidate and learns nothing. The pass condition must be η
rising, measured on the joint occupancy vector, with `overlap_observable` carried alongside — a zero η
from a vector that could not show overlap is not evidence.

**4. It must be able to produce a NEGATIVE control.** A pair whose hoist is *impossible* — the transfer
genuinely depends on the wait — must exist and must NOT show a rise. Without it, a rise cannot be
attributed to hoisting rather than to noise or to an unrelated difference between two runs.

## Where it belongs

Not in `workload_gen`, which is the self-hosted-program emitter and should stay that. A sibling —
`merlin/python/merlin/perf/command_stream_gen.py` — consuming the parsed command list from
`targetgen.contract.interface_emit` and returning a permuted pair plus the dependence facts that justify
the permutation. `_perf.yaml`'s entry then points at the sibling, and the family's `emitter.status`
becomes `existing` only when the negative control demonstrably does not fire.

## What still cannot be measured after it is built

η needs a per-cycle trace of a real program, and that needs a built co-simulation model. The emitter
makes the family *generatable*; it does not make it *measurable* on a host with no such model. Those
are two gaps and closing the first does not close the second — the calibration record already states
this by reporting itself as a plan rather than a calibration.
