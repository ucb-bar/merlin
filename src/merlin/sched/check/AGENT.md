# AGENT.md — merlin/python/merlin/sched/check

## Purpose

Correctness gates that need no simulator. They run on every schedule or contract change, before a
candidate spends RTL time.

## Modules

- `placement.py` — whether a kernel's on-chip placement is sound: a result written over an operand
  still being read, and a bank two contending engines share under a declared arbiter. Unlike `sync.py`
  this asks ONE question of every machine, because neither failure depends on the archetype.
- `sync.py` — whether a kernel's synchronisation is sound ON A GIVEN MACHINE. The same kernel is
  correct on one archetype and wrong on the other with no instruction changed, so this takes a
  `merlin.sched.mach.Machine` and asks the question that machine poses.
- `epilogue_enum.py` — exhaustive epilogue enumeration (gate G1e). It evaluates two elementwise
  readouts on every accumulator value in a range and reports each disagreement.
  `reachable_accumulator_range` bounds the range from the reduction length and the operand ranges.

## Invariants

- **Enumerate, do not sample.** The accumulator range of a real reduction is small enough to
  enumerate outright; a K=4608 int8 reduction reaches about 1.5e8 values. A sampled stimulus is how a
  dropped sign or a wrong tie rounding has passed before.
- **Neither side is trusted.** A `FlipReport` states where two readouts differ. Whether that is a
  defect or a declared bound is the caller's decision. `exact` is true only if every value in the
  range was evaluated and none differed.
- **The archetype decides the question, not just the answer.** Where hardware tracks dependencies an
  absent wait is not a defect and no correctness check can refute a reordering; where the compiler
  separates hazards the identical omission returns wrong data. A checker that asked one question would
  be wrong on half the corpus, and wrong in the dangerous direction on the half it passed.
- **Undecidable is a state, and its empty problem list is not a clean bill.** A machine that declares no
  hazard model gets neither answer: assuming hardware tracking accepts every illegal schedule, assuming
  compiler responsibility reports defects that may not exist. `SyncReport.decidable` is the field that
  says whether the question was asked at all.
