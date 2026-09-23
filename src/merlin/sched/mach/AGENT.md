# AGENT.md — merlin/python/merlin/sched/mach

## Purpose

The machine a schedule is written against. `merlin.sched.ir` models one instruction stream, one agent
and no time, which is a faithful model of exactly one machine shape. This package adds what the other
shapes need and nothing more: units that issue and execute, the queue each consumes, latency per
*(instruction, unit)* pair, banked memories with ports and an arbiter, a thread hierarchy, and whether
hazards are resolved by hardware or by the compiler.

It is not a scheduling language. The primitives are `merlin.sched.primitives`; the transformation
scripting layer is the `transform` dialect. This is the machine those two are written against.

## Files

- `model.py` — `Machine` and its elements: `Unit`, `Latency`, `Memory`, `Level`/`Hierarchy`, `Unknown`;
  the closed vocabularies `UNIT_KINDS`, `HAZARD_RESOLUTIONS`, `COMPLETION_KINDS`; the overlap queries
  `can_overlap` / `shares_issue` / `resource_of`; and the canonical text form and its digest.

## Invariants

- **A machine is derived, never declared.** A target contributes data — its compute-unit contract, its
  RTL facts, its address space, its declared issue-scheduling rules. It never contributes a `Machine`.
  Nothing here names a target, and `build_tools/scripts/check_no_target_name.py` covers this package.
- **The three-state rule is load-bearing.** An underived quantity is `None` **and** carries an
  `Unknown` saying why — never `0`, never a plausible default. `Machine.__post_init__` refuses a
  `hazard_resolution` of `None` with no matching `Unknown`, because assuming `interlocked` makes every
  illegal schedule look legal and assuming `explicit` makes a correctness gate that cannot fire look
  like one that can.
- **Latency belongs to the pair, not the instruction.** The same logical operation costs 95 cycles on
  one matrix unit and 34 on the other, so a single `latency` attribute cannot express it. `issue` and
  `result` are separate fields: a declared issue *distance* in instruction slots is not a completion
  latency, and conflating them prices a schedule wrongly.
- **The unit vocabulary is imported, not restated.** `UNIT_KINDS` is `compute_units.KINDS | MOVER_KINDS`,
  so the compute-unit taxonomy and this one cannot drift.
- **Provenance is inside the digest.** Two machines derived from different RTL are different machines
  even when every number matches, and a measurement keyed on one must not be reused for the other.
