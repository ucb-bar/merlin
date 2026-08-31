---
title: "Design note: memory mapping is an obligation set, not a heuristic"
kind: design
status: current
owner: core
last_verified: 2026-08-30
related: [performance_levers_per_archetype, performance_budget_unit]
code_refs: [merlin/python/merlin/targetgen/address_space.py, merlin/python/merlin/targetgen/boundary.py, merlin/python/merlin/compile_cli.py, merlin/python/merlin/targetgen/conformance.py]
---

# Memory mapping is an obligation set

The corpus proves arithmetic well. It says almost nothing about *where the operands were put*, and on an
accelerator with a small explicitly-managed on-chip store that is where both the correctness failures and
the performance failures actually live.

Two measured facts frame this note.

**The correctness half already bit us.** A graded backend tiled the iteration space correctly and then
addressed every weight tile as simultaneously resident. At 512x512 that asked for 16384 scratchpad rows
against 16384 available, and the simulator aborted three layers away in a range check — a failure
indistinguishable from an unreachable oracle. The obligation the interface named (`capacity_fit`) existed
and nobody evaluated it, so the backend did not fail its contract; it just crashed.

**The performance half is the compiler's largest single weakness.** The schedule loads operands far
earlier than it needs them, so movement and compute serialise instead of overlapping. On an interlocked
target that costs nothing in correctness — the hardware protects you — which is exactly why no existing
capsule can detect it: every schedule passes.

## Why the address space must be derived

Every quantity below is a fact about the target, read from its own RTL discovery, and none of it may be a
literal in shared code:

| quantity | derived from |
|---|---|
| operand store bytes / declared depth | the facts artifact's `memories` list |
| row width in bytes | the compute array's column count x the datapath element width |
| total rows, bank count | bytes / row width, then rows / depth |
| a *separate* accumulator address space | two stores whose row widths differ, because the accumulate type is wider |

A worked instance: a store of 262144 bytes feeding a 16-column int8 array has 16-byte rows, so 16384 rows;
against a declared depth of 4096 that is 4 banks. Its accumulator, 65536 bytes at a 64-byte int32 row, is
1024 rows over a declared depth of 512 — 2 banks. Both numbers fall out of discovery; neither is typed
anywhere, and a target with a different geometry gets different numbers with no edit.

Where a quantity cannot be derived it is UNKNOWN and reported. It is never zero and never a default: this
repo has repeatedly turned "we could not measure it" into "we measured nothing there", and an
address-space model is the worst possible place to do that, because a capacity obligation assumed
satisfied is a crash with no explanation.

## The regimes, and why they are a coverage axis

A capsule's memory-mapping regime is decided by its working set against the derived capacity — so, like
the semantic cells and the composition shapes, it is *derived per capsule* rather than declared on it.

* **`fits_double`** — the working set fits **twice**. This is the only regime in which movement for the
  next tile can overlap compute on the current one. A corpus made entirely of capsules in this regime can
  never show that the compiler failed to double-buffer.
* **`fits_single`** — fits once. Staging is impossible; the schedule must serialise, and that is correct
  rather than a defect. Distinguishing this from `fits_double` is what stops us charging the compiler for
  an overlap the hardware could not have provided.
* **`fits_on_reuse`** — the sum of *live ranges* fits but the sum of all tensors does not. Only an
  allocator that reuses rows freed by a dead tensor works here. A bump allocator fails, and fails
  *silently* on a target whose store wraps.
* **`spills`** — exceeds capacity however it is allocated, so the compiler must tile and re-load. The
  interesting question stops being "does it fit" and becomes "how much re-load traffic did the loop order
  cost", which is a scheduling question with a measurable answer.
* **`bank_crossing`** — a tile whose rows straddle a bank boundary.
* **`dual_space`** — the program addresses both the operand store and a separate accumulator space.
  Writing one where the other belongs is a wrong-data class, not a crash class.

## The obligations

Functional, provable on the small basis at the cycle-accurate tier:

1. **Capacity is evaluated, not assumed.** An over-capacity program is refused with the numbers, never
   crashed into. Unknown capacity fails closed.
2. **Live ranges are disjoint.** Two tensors live at once never share a row.
3. **A dead tensor's rows are reusable.** Proven by a capsule that fits *only* on reuse — if it fits
   anyway, the obligation is untested.
4. **The two address spaces are not interchangeable.** An accumulator address where an operand address
   belongs is rejected.
5. **Bank and alignment edges are exercised**, at the tail as well as at whole tiles.

Performance, measured as an A/B on identical work:

6. **Staging depth**: with a `fits_double` working set, movement for tile n+1 overlaps compute on tile n.
7. **Residency**: an operand reused across tiles is loaded once, not once per tile.
8. **Loop order**: the order that minimises re-load traffic in the `spills` regime is chosen.

## The falsifier problem, restated for memory

On a hardware-interlocked target every one of 6, 7 and 8 is *correct whatever the compiler chooses*, so a
capsule gated on bit-exactness passes them all and learns nothing — the inert-lever trap. Their falsifier
must be the measured one: **operand rows loaded** (residency), **realised overlap between the movement
and compute engines** (staging), and **re-load traffic** (loop order). Each is a number the joint
occupancy vector and the command trace already carry.

The functional obligations 1-5 are different: those *can* fail on correctness, and 1 already has. They
belong to the functional phase, where bit-exactness is a sharp instrument, and they must be proven before
any schedule is tuned — a faster wrong address is not an optimisation.
